# -*- coding: utf-8 -*-
"""P3-T3 响度归一化单测：``app/pipeline/loudness.py``。

分两层：
- **合成用例**（默认跑，不依赖真实文件）：滑窗平均/O(n) cumsum、4× 真峰值包络、
  真峰值限幅（正弦/噪声/脉冲、ceiling 收敛、静音直通、非法参数）、ASL 解析值、
  语音掩码、ffmpeg 降采样、V2 控制点（clamp ±3/±6、速率 ≤1.5 dB/s、语音不足跳过、
  退化 static）、块级包络插值/平滑、静态增益逐样本数学（含 ±24 dB clamp）、
  端到端 render（ASL 校正迭代 + 真峰值 + 无削波）、gain+limit 的 RTF；
- **真实文件用例**（``@pytest.mark.slow``，用 ``pytest -m slow`` 显式跑）：
  f24 analyze 缓存 → ``trim.plan_trim/render_clean`` 生成 clean44 → ``loudness.render``
  全量（3806.5 s），断言 asl_out 与 −20 偏差 ≤0.5 dB、wav 真峰值 ≤−1 dBTP、
  削波 0、增益变化率 ≤1.5 dB/s。

约定：
- 测 ASL 时正弦幅度 A 的解析值是 ``20·log10(A) − 3.0103``（25 ms 窗功率 = A²/2）；
- 真峰值统一用 ``resample_poly(y, 4, 1)`` 复核（与限幅器内部同一路子），
  验收线是 ``output TP ≤ ceiling + 0.15 dB``（任务卡 A 项）；
- 真实文件用例与已有 16k/analyze/44k 缓存共存，只新增自己的临时文件并在结束时删除。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from app.pipeline import loudness as loud_mod
from app.pipeline.loudness import (
    SR16,
    SR44,
    LoudnessError,
    block_gain_envelope,
    downsample_wav,
    limit_wav,
    measure_asl,
    measure_true_peak_wav,
    render,
    speech_mask_from_intervals,
    true_peak_limit,
    v2_control_points,
)

from conftest import ROOT, SAMPLES, requires_data, requires_ffmpeg

CACHE_DIR = ROOT / "work" / "cache"
ANALYZE_SUFFIX = ".analyze.json"
SIX = {
    "f00": "621cb91c",
    "f04": "9f3feb49",
    "f05": "0a7fa0ed",
    "f06": "b00e197d",
    "f20": "4db18037",
    "f24": "7927721f",
}

#: 正弦幅度 A 的帧功率解析值（A²/2 → dBFS）
SINE_POWER_OFFSET_DB = 20.0 * math.log10(1.0 / math.sqrt(2.0))


# ---------------------------------------------------------------- 合成数据工具
def _sine(duration_s: float, amp: float, freq: float = 1000.0, sr: int = SR44) -> np.ndarray:
    t = np.arange(int(round(duration_s * sr))) / float(sr)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _write_wav(path: Path, x: np.ndarray, sr: int = SR44) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, np.asarray(x, dtype=np.float32))
    return path


def _read_wav(path: Path) -> np.ndarray:
    return np.asarray(wavfile.read(str(path))[1], dtype=np.float32)


def _true_peak_db(y: np.ndarray, oversample: int = 4) -> float:
    """整段 4× 过采样真峰值（dBTP）。"""
    data = np.asarray(y, dtype=np.float64)
    if data.size == 0:
        return float("-inf")
    peak = float(np.abs(resample_poly(data, oversample, 1)).max())
    return 20.0 * math.log10(max(peak, 1e-12))


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


# ---------------------------------------------------------------- 小工具 / 真峰值包络
def test_moving_avg_matches_direct_average():
    """O(n) cumsum 版滑动平均与 np.convolve 等值（含偶数核、edge padding）。"""
    rng = np.random.default_rng(11)
    x = rng.normal(size=1000)
    for k in (2, 3, 10, 101):
        fast = loud_mod._moving_avg(x, k)
        padded = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
        slow = np.convolve(padded, np.ones(k) / k, mode="valid")
        assert fast.shape == slow.shape
        np.testing.assert_allclose(fast, slow, atol=1e-12)
    # k<=1 原样返回
    np.testing.assert_array_equal(loud_mod._moving_avg(x, 1), x)


def test_tp_envelope_matches_oversampled_peak():
    """逐样本真峰值包络：覆盖采样峰，且 max == 整段 4× 过采样峰值（±0.02 dB）。"""
    for x in (
        _sine(2.0, 0.8),
        np.random.default_rng(4).normal(size=SR44).astype(np.float32),
    ):
        env = loud_mod._tp_envelope_4x(x, 4)
        assert env.shape == x.shape
        assert np.all(env >= np.abs(x) - 1e-5)  # 每个采样点的峰都在自己的包络里
        assert 20 * math.log10(float(env.max())) == pytest.approx(_true_peak_db(x), abs=0.02)


# ---------------------------------------------------------------- 真峰值限幅
def test_true_peak_limit_sine_bounded():
    """正弦满刻度：输出 4× TP ≤ ceiling+0.15、不削波、max|y|<0.999。"""
    sr = SR44
    ceiling = -2.75
    for amp in (0.9, 1.0):
        x = _sine(3.0, amp)
        y, stats = true_peak_limit(x, sr, ceiling)
        assert _true_peak_db(y) <= ceiling + 0.15
        assert float(np.abs(y).max()) < 0.999
        assert int(np.count_nonzero(np.abs(y) >= 0.999)) == 0
        # 正弦的峰值是稳定需求，限幅器几乎恒压（触发比例 100%）
        assert stats["trigger_ratio"] > 0.9
        assert stats["min_gain_db"] < 0.0
        assert stats["ceiling_dbtp"] == pytest.approx(ceiling, abs=1e-6)
        assert stats["ceiling_effective_dbtp"] <= ceiling + 1e-9


def test_true_peak_limit_noise_converges():
    """满刻度白噪：单趟限幅的 4× TP 会过冲，ceiling 收敛后必须回到 ≤ ceiling+0.15。"""
    rng = np.random.default_rng(3)
    x = (1.0 * rng.normal(size=SR44 * 3)).astype(np.float32)
    ceiling = -2.75
    y, stats = true_peak_limit(x, SR44, ceiling)
    assert _true_peak_db(y) <= ceiling + 0.15
    assert float(np.abs(y).max()) < 0.999
    assert int(np.count_nonzero(np.abs(y) >= 0.999)) == 0
    # 白噪这种高密度满刻度信号必须靠压低 ceiling 才达线
    assert stats["tp_iterations"] >= 2
    assert stats["ceiling_effective_dbtp"] < ceiling
    assert stats["tp_measured_dbtp"] <= ceiling + 0.15


def test_true_peak_limit_impulses():
    """脉冲（单点满刻度 / 密集脉冲串）：真峰值受限、不削波。"""
    ceiling = -2.0
    n = SR44 * 3
    single = np.zeros(n, dtype=np.float32)
    single[n // 2] = 1.0
    train = np.zeros(n, dtype=np.float32)
    train[:: int(0.05 * SR44)] = 0.99
    for x in (single, train):
        y, _stats = true_peak_limit(x, SR44, ceiling)
        assert _true_peak_db(y) <= ceiling + 0.15
        assert float(np.abs(y).max()) < 0.999
        assert int(np.count_nonzero(np.abs(y) >= 0.999)) == 0


def test_true_peak_limit_quiet_passthrough():
    """限幅不触发时逐样本 bit 级不变（g 恒为 1）。"""
    x = _sine(2.0, 0.01)
    y, stats = true_peak_limit(x, SR44, -2.0)
    np.testing.assert_array_equal(y, x)
    assert stats["trigger_ratio"] == 0.0
    assert stats["trigger_events"] == 0
    assert stats["min_gain_db"] == pytest.approx(0.0, abs=1e-6)


def test_true_peak_limit_rejects_bad_args():
    x = _sine(0.5, 0.5)
    with pytest.raises(LoudnessError):
        true_peak_limit(x, 0, -2.0)
    with pytest.raises(LoudnessError):
        true_peak_limit(x, SR44, -2.0, oversample=0)
    with pytest.raises(LoudnessError):
        true_peak_limit(x, SR44, -2.0, attack_ms=-1.0)
    with pytest.raises(LoudnessError):
        true_peak_limit(x, SR44, float("nan"))


# ---------------------------------------------------------------- ASL 测量
def test_measure_asl_sine_analytic():
    """正弦 ASL 解析值：20·log10(A) − 3.0103（±0.05 dB）。"""
    for amp in (0.5, 0.1, 0.01):
        x = _sine(3.0, amp, sr=SR16)
        asl = measure_asl(x, [[0.0, 3.0]], None)
        expected = 20 * math.log10(amp) + SINE_POWER_OFFSET_DB
        assert asl == pytest.approx(expected, abs=0.05), (amp, asl, expected)


def test_measure_asl_only_counts_interval_frames():
    """区间只覆盖安静段时，ASL 取安静段电平（不把区间外的大声算进来）。

    区间两端各留 25 ms（窗长）余量，避免帧窗跨过静音/大声的接缝把大声算进来。
    """
    quiet = _sine(1.0, 0.01, sr=SR16)
    loud = _sine(1.0, 0.5, sr=SR16)
    x = np.concatenate([quiet, loud]).astype(np.float32)
    only_quiet = measure_asl(x, [[0.0, 0.95]], None)
    only_loud = measure_asl(x, [[1.05, 2.0]], None)
    assert only_quiet == pytest.approx(20 * math.log10(0.01) + SINE_POWER_OFFSET_DB, abs=0.05)
    assert only_loud == pytest.approx(20 * math.log10(0.5) + SINE_POWER_OFFSET_DB, abs=0.05)
    assert only_loud - only_quiet == pytest.approx(20 * math.log10(50.0), abs=0.1)


def test_measure_asl_empty_returns_nan():
    assert math.isnan(measure_asl(np.zeros(0, dtype=np.float32), [[0.0, 1.0]], None))
    assert math.isnan(measure_asl(_sine(1.0, 0.1, sr=SR16), [], None))


def test_measure_asl_rejects_invalid_intervals():
    x = _sine(1.0, 0.1, sr=SR16)
    with pytest.raises(LoudnessError):
        measure_asl(x, [[float("nan"), 1.0]], None)
    with pytest.raises(LoudnessError):
        measure_asl(x, [["a", "b"]], None)


# ---------------------------------------------------------------- 语音掩码
def test_speech_mask_from_intervals():
    centers = np.arange(0.0, 10.0, 0.01)  # 1000 帧
    mask = speech_mask_from_intervals([[1.0, 2.0], [5.0, 5.5]], len(centers), centers)
    assert mask.sum() == pytest.approx((1.0 + 0.5) / 0.01, abs=1)
    assert mask[150]  # 1.5 s
    assert not mask[300]  # 3 s
    # 反向/空区间被忽略，不影响其它区间
    mask2 = speech_mask_from_intervals([[2.0, 1.0], [5.0, 5.5]], len(centers), centers)
    assert mask2.sum() == pytest.approx(0.5 / 0.01, abs=1)
    # 空码流
    assert speech_mask_from_intervals([[0.0, 1.0]], 0, np.zeros(0)).sum() == 0


# ---------------------------------------------------------------- 降采样
@requires_ffmpeg
def test_downsample_wav(tmp_path: Path):
    """ffmpeg 降采样：采样率/长度正确、电平不变（±0.1 dB）。"""
    x = _sine(2.0, 0.5, sr=SR44)
    src = _write_wav(tmp_path / "in44.wav", x, SR44)
    dst = downsample_wav(src, tmp_path / "out16.wav", SR16)
    sr, y = wavfile.read(str(dst))
    assert sr == SR16
    assert len(y) == pytest.approx(2.0 * SR16, abs=2)
    rms_in = math.sqrt(float(np.mean(x.astype(np.float64) ** 2)))
    rms_out = math.sqrt(float(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
    assert 20 * math.log10(rms_out / rms_in) == pytest.approx(0.0, abs=0.1)
    with pytest.raises(LoudnessError):
        downsample_wav(tmp_path / "missing.wav", tmp_path / "x.wav", SR16)


# ---------------------------------------------------------------- 分块限幅
def test_limit_wav_chunked_matches_whole(tmp_path: Path):
    """分块限幅与整段一致（1 s chunk + 1 s context，逐样本相等），TP 达标。"""
    rng = np.random.default_rng(5)
    n = SR44 * 3
    t = np.arange(n) / SR44
    x = (0.8 * np.sin(2 * np.pi * 300 * t) + 0.3 * rng.normal(size=n)).astype(np.float32)
    src = _write_wav(tmp_path / "in.wav", x, SR44)
    whole = tmp_path / "whole.wav"
    chunked = tmp_path / "chunked.wav"
    st_whole = limit_wav(src, whole, -2.75, None, chunk_s=3.0)
    st_chunk = limit_wav(src, chunked, -2.75, None, chunk_s=1.0)
    y_whole = _read_wav(whole)
    y_chunk = _read_wav(chunked)
    assert y_whole.shape == y_chunk.shape
    np.testing.assert_array_equal(y_chunk, y_whole)
    assert st_chunk["tp_4x_dbtp"] <= -2.75 + 0.15
    assert st_chunk["ceiling_effective_dbtp"] == pytest.approx(
        st_whole["ceiling_effective_dbtp"], abs=1e-6
    )
    assert st_chunk["n_chunks"] > st_whole["n_chunks"]
    assert st_chunk["ok"] is True
    assert st_chunk["trigger_ratio"] > 0.0
    assert st_chunk["min_gain_db"] < 0.0
    # 与整段 true_peak_limit 一致（同一 ceiling 下）
    y_direct, _ = true_peak_limit(x, SR44, st_chunk["ceiling_effective_dbtp"])
    assert _true_peak_db(y_direct) <= -2.75 + 0.15
    # wav 级独立测量与分块统计一致（±0.05 dB）
    assert measure_true_peak_wav(chunked) == pytest.approx(st_chunk["tp_4x_dbtp"], abs=0.05)


def test_sample_env_matches_blocks_and_is_continuous():
    """块级 dB → 样本级线性包络：分块重算与整段一致、无台阶。"""
    times = np.array([0.0, 10.0])
    gains = np.array([0.0, 6.0])
    block_db = block_gain_envelope(times, gains, 20.0, 0.0, 6.0, None)
    n = int(round(20.0 * SR44))
    env = loud_mod._sample_env(block_db, 0, n, SR44)
    assert env.shape == (n,)
    db = 20.0 * np.log10(env.astype(np.float64))
    assert db.min() >= -6.01 and db.max() <= 6.01
    # 余弦交叉淡化 → 相邻样本变化很小，没有块间台阶
    assert float(np.max(np.abs(np.diff(db)))) < 0.02
    # 分块渲染时对子区间重算必须与整段一致（否则块边界会咝啦）
    part = loud_mod._sample_env(block_db, 12345, 12345 + SR44, SR44)
    np.testing.assert_allclose(part, env[12345: 12345 + SR44], atol=1e-6)


# ---------------------------------------------------------------- V2 控制点
def _v2_inputs(duration_s: float = 120.0, sr: int = SR16):
    """构造 10 ms 帧功率/中心/掩码（电平按时间段给定）。"""
    n_frames = int(duration_s / 0.01)
    centers = (np.arange(n_frames) * 0.01 + 0.0125).astype(np.float64)
    mask = np.ones(n_frames, dtype=bool)
    return n_frames, centers, mask


def test_v2_control_points_clamp_and_window_use():
    """clamp ±3：控制点增益不超 base±3；20 s 窗/5 s 跳的窗口数正确。"""
    n_frames, centers, mask = _v2_inputs(120.0)
    power = np.full(n_frames, 10.0 ** (-30.0 / 10.0))  # −30 dBFS
    power[int(60 / 0.01): int(80 / 0.01)] = 10.0 ** (-20.0 / 10.0)  # 60–80 s 更响
    power[int(90 / 0.01):] = 10.0 ** (-40.0 / 10.0)  # 90 s 后更轻
    ref = loud_mod._asl_from_power(power, mask, 15.0)
    base = -20.0 - ref
    times, gains, info = v2_control_points(power, centers, mask, 120.0, base, 3.0, None)

    assert info["degraded"] is False
    assert info["clamp_db"] == pytest.approx(3.0)
    assert info["win_s"] == pytest.approx(20.0) and info["hop_s"] == pytest.approx(5.0)
    # 120 s / 5 s hop → 24 个窗口，每个窗内语音都够
    assert info["n_windows_used"] == 24
    assert times.shape == gains.shape == (24,)
    assert np.all(np.diff(times) > 0)
    assert gains.min() >= base - 3.0 - 1e-9
    assert gains.max() <= base + 3.0 + 1e-9
    # 响的段要压（gain 更小），轻的段要提（gain 更大），且都应被 clamp 顶住
    loud_index = int(np.argmin(np.abs(times - 70.0)))
    quiet_index = int(np.argmin(np.abs(times - 110.0)))
    assert gains[loud_index] == pytest.approx(base - 3.0, abs=1e-6)
    assert gains[quiet_index] == pytest.approx(base + 3.0, abs=1e-6)
    assert info["rate_control_db_s"] <= 1.5 + 1e-6


def test_v2_control_points_clamp6_rate_limited(cfg):
    """clamp ±6：电平剧烈跳变时逐点限速 ≤1.5 dB/s，且不越 base±6。

    用 ``win_s = hop_s = 5 s`` 把窗口缩到不重叠，制造单跳 >7.5 dB 的跳变来逼出限速
    （20 s 窗会把跳变摊到 ~4 个 hop，速率天然 <1.5 dB/s，测不到限速逻辑）。
    """
    cfg5 = cfg.replace({"loudness.v2.win_s": 5.0, "loudness.v2.hop_s": 5.0})
    n_frames, centers, mask = _v2_inputs(120.0)
    power = np.full(n_frames, 10.0 ** (-35.0 / 10.0))
    power[: int(60 / 0.01)] = 10.0 ** (-20.0 / 10.0)  # 前 60 s 更响
    ref = loud_mod._asl_from_power(power, mask, 15.0)
    base = -20.0 - ref
    times, gains, info = v2_control_points(power, centers, mask, 120.0, base, 6.0, cfg5)

    assert info["degraded"] is False
    assert gains.min() >= base - 6.0 - 1e-9
    assert gains.max() <= base + 6.0 + 1e-9
    # 控制点级：相邻变化 ≤ max_rate × hop = 7.5 dB（限速生效，不跳满 12 dB）
    assert np.max(np.abs(np.diff(gains))) <= 7.5 + 1e-6
    assert info["rate_control_db_s"] <= 1.5 + 1e-6
    assert gains.max() - gains.min() > 7.5  # 确实有大跳变需求，否则测不到限速
    # 块级包络的速率同样 ≤1.5 dB/s
    envelope = block_gain_envelope(times, gains, 120.0, base, 6.0, cfg5)
    rate = float(np.max(np.abs(np.diff(envelope)))) / 0.02
    assert rate <= 1.5 + 1e-6


def test_v2_control_points_skips_low_speech_and_degrades():
    """语音不足的窗口跳过；可用控制点 <2 时退化 static。"""
    n_frames, centers, mask = _v2_inputs(60.0)
    power = np.full(n_frames, 10.0 ** (-30.0 / 10.0))
    # 只有前 25 s 有语音 → 后面的窗口（语音 <3 s）被跳过
    mask[int(25 / 0.01):] = False
    base = 5.0
    times, gains, info = v2_control_points(power, centers, mask, 60.0, base, 3.0, None)
    assert info["n_windows_used"] < info["n_windows_total"]
    assert info["n_windows_used"] >= 2
    assert info["degraded"] is False

    # 语音只有最后 2 s（< min_speech_s=3 s）→ 所有窗口都不够语音 → 退化 static
    mask2 = np.zeros(n_frames, dtype=bool)
    mask2[int(58 / 0.01):] = True
    times2, gains2, info2 = v2_control_points(power, centers, mask2, 60.0, base, 3.0, None)
    assert info2["degraded"] is True
    assert info2["n_windows_used"] == 0
    np.testing.assert_allclose(times2, [0.0, 60.0])
    np.testing.assert_allclose(gains2, [base, base])
    assert info2["gain_min_db"] == pytest.approx(base)


def test_v2_control_points_rejects_bad_duration():
    n_frames, centers, mask = _v2_inputs(10.0)
    power = np.full(n_frames, 0.001)
    with pytest.raises(LoudnessError):
        v2_control_points(power, centers, mask, 0.0, 0.0, 3.0, None)


def test_block_gain_envelope_interpolates_and_limits_rate():
    """块级包络：块数 = ceil(dur/0.02)、控制点间线性插值、1 s 平滑、clamp、速率受限。"""
    times = np.array([0.0, 10.0])
    gains = np.array([-3.0, 3.0])  # 10 s 内 +6 dB → 0.6 dB/s < 1.5 dB/s
    env = block_gain_envelope(times, gains, 20.0, 0.0, 3.0, None)
    assert env.size == int(math.ceil(20.0 / 0.02))
    assert env.min() >= -3.0 - 1e-9 and env.max() <= 3.0 + 1e-9
    # 单调递增（1 s 滑动平均只在两端压平）
    assert np.all(np.diff(env) >= -1e-9)
    assert env[-1] == pytest.approx(3.0, abs=1e-6)
    rate = float(np.max(np.abs(np.diff(env)))) / 0.02
    assert rate <= 1.5 + 1e-6
    # 单控制点 → 常数
    env_single = block_gain_envelope(np.array([5.0]), np.array([1.5]), 3.0, 0.0, 3.0, None)
    assert env_single.size == int(math.ceil(3.0 / 0.02))
    np.testing.assert_allclose(env_single, 1.5)
    # 空控制点 → base
    env_empty = block_gain_envelope(np.array([]), np.array([]), 1.0, -2.0, 3.0, None)
    np.testing.assert_allclose(env_empty, -2.0)


# ---------------------------------------------------------------- 端到端 render（合成）
@requires_ffmpeg
def test_render_static_gain_math(cfg, tmp_path: Path):
    """V1 静态增益：输出 == 输入 × 10^(gain/20)（逐样本 <1e-5），ASL 归一到 −20。"""
    x = _sine(3.0, 0.01)  # ASL ≈ −43 dBFS
    src = _write_wav(tmp_path / "clean.wav", x)
    out = tmp_path / "out.wav"
    result = render(src, [[0.0, 3.0]], cfg, out44_wav=out)

    assert result.ok is True
    assert result.asl_in_dbfs == pytest.approx(-43.01, abs=0.1)
    assert result.gain_db == pytest.approx(23.0, abs=0.2)
    assert result.asl_dev_db == pytest.approx(0.0, abs=0.03)
    assert result.gain.type == "static"  # 3 s 太短 → V2 无控制点，退化 static
    assert result.limiter["trigger_ratio"] == 0.0
    assert result.clipped_samples == 0
    assert result.wav_out == str(out)

    y = _read_wav(out)
    expected = (x.astype(np.float64) * _db_to_linear(result.gain_db)).astype(np.float32)
    assert float(np.max(np.abs(y.astype(np.float64) - expected))) < 1e-5

    # 契约字段
    levels = result.report_levels_out()
    assert set(levels) == {"asl_out_dbfs", "lufs_out", "window5m_spread_out_db"}
    assert levels["asl_out_dbfs"] == pytest.approx(-20.0, abs=0.05)
    info = result.gain.as_dict()
    assert set(info) == {
        "type", "gain_db", "min_db", "max_db", "max_rate_db_s", "limiter_triggers",
    }
    assert {"ceiling_dbtp", "oversample", "iterations", "trigger_ratio",
            "min_gain_db"} <= set(result.limiter)


@requires_ffmpeg
def test_render_clamps_gain_at_24db(cfg, tmp_path: Path):
    """ASL 太低（−53 dBFS）→ 静态增益被 clamp 到 +24 dB，输出仍按 24 dB 缩放。"""
    x = _sine(3.0, 0.00316)
    src = _write_wav(tmp_path / "quiet.wav", x)
    out = tmp_path / "quiet_out.wav"
    result = render(src, [[0.0, 3.0]], cfg, out44_wav=out)

    assert result.gain_db == pytest.approx(24.0, abs=1e-6)
    assert result.gain.min_db == pytest.approx(24.0, abs=1e-6)
    assert result.gain.max_db == pytest.approx(24.0, abs=1e-6)
    # 目标 −20 达不到（clamp 住了）→ ok=False + 明确 error，不是静默通过
    assert result.ok is False
    assert "ASL 未收敛" in result.error
    y = _read_wav(out)
    expected = (x.astype(np.float64) * _db_to_linear(24.0)).astype(np.float32)
    assert float(np.max(np.abs(y.astype(np.float64) - expected))) < 1e-5


@requires_ffmpeg
def test_render_v2_envelope_limits_and_is_smooth(cfg, tmp_path: Path):
    """V2：三段电平漂移（−35/−32/−29 dBFS，共 6 dB）→ clamp ±3 内被拉平。

    漂移量刻意取 ±3 dB（正好等于 clamp）：V2 应能把三段都拉回 −20 附近。
    """
    dur = 60.0
    x = np.zeros(int(dur * SR44), dtype=np.float32)
    t = np.arange(x.size) / SR44
    for lo, hi, amp in ((0, 20, _db_to_linear(-35)), (20, 40, _db_to_linear(-32)),
                        (40, 60, _db_to_linear(-29))):
        sel = (t >= lo) & (t < hi)
        x[sel] = (amp * np.sin(2 * np.pi * 440 * t[sel])).astype(np.float32)
    src = _write_wav(tmp_path / "drift.wav", x)
    out = tmp_path / "drift_out.wav"
    result = render(src, [[0.0, dur]], cfg, out44_wav=out)

    assert result.v2["enabled"] is True
    assert result.v2["degraded"] is False
    assert result.v2["n_windows_used"] > 0
    assert result.v2["clamp_db_used"] == pytest.approx(3.0)
    assert result.v2["high_drift"] is False
    assert result.gain.type == "v2"
    # V2 后增益范围被 clamp 在 base±3，变化率 ≤1.5 dB/s
    base = result.gain_db
    assert result.gain.min_db >= base - 3.0 - 0.05
    assert result.gain.max_db <= base + 3.0 + 0.05
    assert result.gain.min_db < result.gain.max_db
    assert result.gain.max_rate_db_s <= 1.5 + 1e-6
    assert abs(result.asl_dev_db) <= 0.5
    assert result.clipped_samples == 0
    # 输出三段的电平差被 V2 拉平到 ~1.5 dB 内（P2 同款帧功率粗看）
    y = _read_wav(out)
    from app.pipeline.analyze import frame_power

    power, centers = frame_power(y, SR44, win_ms=25.0, hop_ms=100.0)
    early = 10 * math.log10(float(np.mean(power[(centers > 2) & (centers < 18)])))
    late = 10 * math.log10(float(np.mean(power[(centers > 42) & (centers < 58)])))
    assert abs(late - early) <= 1.5
    assert early == pytest.approx(-20.0, abs=1.5)
    assert late == pytest.approx(-20.0, abs=1.5)


@requires_ffmpeg
def test_render_limiter_iteration_and_true_peak(cfg, tmp_path: Path):
    """限幅会吃掉 ASL：f00-like（轻声 + 稀疏满刻度尖峰）需一轮校正、wav TP ≤ −1 dBTP。"""
    dur = 60.0
    n = int(dur * SR44)
    t = np.arange(n) / SR44
    x = (_db_to_linear(-30) * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    spikes = np.zeros(n, dtype=np.float32)
    spikes[:: int(0.5 * SR44)] = 0.95
    x = (x + spikes).astype(np.float32)
    src = _write_wav(tmp_path / "f00like.wav", x)
    out = tmp_path / "f00like_out.wav"
    result = render(src, [[0.0, dur]], cfg, out44_wav=out)

    assert result.asl_in_dbfs < -30.0
    assert result.gain_db > 10.0  # 归一化要提很多
    assert abs(result.asl_dev_db) <= 0.03  # 校正后回到目标
    assert result.iterations >= 1  # 限幅确实压掉了电平
    assert result.limiter["trigger_ratio"] > 0.0
    assert result.limiter["min_gain_db"] < 0.0
    assert result.tp_wav_dbtp <= -1.0  # 任务卡硬指标（wav 层）
    assert result.limiter["tp_4x_dbtp"] <= -1.0
    assert result.clipped_samples == 0
    assert float(np.abs(_read_wav(out)).max()) < 0.999


@requires_ffmpeg
def test_render_result_contract_with_analysis_asl(cfg, tmp_path: Path):
    """asl_analysis_dbfs 透传、v2/limiter/timings 关键字段齐全。"""
    src = _write_wav(tmp_path / "c.wav", _sine(2.0, 0.05))
    result = render(
        src, [[0.0, 2.0]], cfg, out44_wav=tmp_path / "c_out.wav",
        analysis_asl_in=-10.1, gain_offset_db=0.0,
    )
    assert result.asl_analysis_dbfs == pytest.approx(-10.1)
    assert set(result.timings) >= {
        "downsample_in_s", "asl_in_s", "render_1_s", "measure_out_1_s",
        "ebur128_s", "total_s",
    }
    assert "n_windows_used" in result.v2 and "degraded" in result.v2
    assert result.timings["total_s"] >= 0.0
    assert result.limiter["ceiling_effective_dbtp"] <= result.limiter["ceiling_dbtp"] + 1e-9


@requires_ffmpeg
def test_render_rejects_bad_inputs(cfg, tmp_path: Path):
    good = _write_wav(tmp_path / "g.wav", _sine(1.0, 0.1))
    with pytest.raises(LoudnessError):
        render(tmp_path / "missing.wav", [[0.0, 1.0]], cfg, out44_wav=tmp_path / "o.wav")
    with pytest.raises(LoudnessError):
        render(good, [], cfg, out44_wav=tmp_path / "o.wav")
    with pytest.raises(LoudnessError):
        render(good, [[0.0, 1.0]], None, out44_wav=tmp_path / "o.wav")
    # 渲染轴必须是 44.1 k
    bad_rate = _write_wav(tmp_path / "bad.wav", _sine(1.0, 0.1, sr=SR16), SR16)
    with pytest.raises(LoudnessError):
        render(bad_rate, [[0.0, 1.0]], cfg, out44_wav=tmp_path / "o.wav")


# ---------------------------------------------------------------- 性能（RTF）
def test_gain_plus_limit_rtf_under_002():
    """60 s 44.1 k 合成信号的 gain+limit 增量 RTF < 0.02（不含 ffmpeg/磁盘 I/O）。"""
    dur = 60.0
    n = int(dur * SR44)
    t = np.arange(n) / SR44
    sig = (0.15 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    sig[:: int(0.3 * SR44)] = 1.0  # 定期满刻度脉冲，逼限幅器工作

    started = time.perf_counter()
    # 增益增量：V2 块级包络 → 样本级包络（这里用常数 +15 dB，走同一条代码路径）
    block_db = np.full(int(math.ceil(dur / 0.02)), 15.0)
    envelope = loud_mod._sample_env(block_db, 0, n, SR44)
    pre = sig * envelope
    y, stats = true_peak_limit(pre, SR44, -2.75)
    elapsed = time.perf_counter() - started

    assert y.shape == pre.shape
    assert stats["tp_measured_dbtp"] <= -2.75 + 0.15
    assert elapsed / dur < 0.02, f"RTF={elapsed / dur:.4f} 超 0.02（{elapsed:.2f}s / {dur:.0f}s）"


# ---------------------------------------------------------------- 真实文件（slow）
def _load_analyze(prefix: str):
    matches = sorted(CACHE_DIR.glob(f"{prefix}*{ANALYZE_SUFFIX}"))
    if not matches:
        return None, ""
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("ok") or "result" not in data:
        return None, ""
    from app.models import AnalyzeResult

    return AnalyzeResult(**data["result"]), matches[0].name[: -len(ANALYZE_SUFFIX)]


@requires_data
@requires_ffmpeg
@pytest.mark.slow
def test_data_f24_full_render():
    """真实 f24：clean44（3806.5 s）→ 全量响度归一化，验收 B 项全部达标。

    走已提交的 ``trim.plan_trim/render_clean/chain_clean_segments`` 生成 clean44，
    再 ``loudness.render``；临时文件放在 work/cache（gitignore）并在结束时删除，
    不动既有 16k/analyze/44k 缓存。默认 ``-m "not slow"`` 不跑。
    """
    if "f24" not in SAMPLES:
        pytest.skip("data/ 下没有 f24 样例")
    analyze, job_id = _load_analyze(SIX["f24"])
    if analyze is None:
        pytest.skip("缺少 f24 analyze 缓存（work/cache/7927721f*.analyze.json）")
    from app.config import load_config
    from app.pipeline.decode import ensure_44k
    from app.pipeline.trim import chain_clean_segments, plan_trim, render_clean

    cfg = load_config()
    src44 = ensure_44k(SAMPLES["f24"], cfg=cfg, job_id=job_id)
    assert src44 is not None and Path(src44).is_file(), "44.1k 渲染轴解码失败"
    plan = plan_trim(analyze, cfg)
    assert plan.clean_duration_s == pytest.approx(3806.5, abs=0.5)
    clean44 = Path(cfg.path("cache_dir")) / f"{job_id}.loudness_test.clean44.wav"
    out44 = Path(cfg.path("cache_dir")) / f"{job_id}.loudness_test.out44.wav"
    try:
        render_clean(src44, plan, clean44)
        intervals = chain_clean_segments(analyze, cfg)
        assert intervals, "clean 轴语音区间为空"
        result = render(
            clean44, intervals, cfg, out44_wav=out44,
            analysis_asl_in=analyze.levels.get("asl_p56_dbfs"),
        )
        # 验收 B
        assert result.ok is True, result.error
        assert abs(result.asl_out_dbfs - (-20.0)) <= 0.5
        assert result.tp_wav_dbtp is not None and result.tp_wav_dbtp <= -1.0
        assert result.clipped_samples == 0
        assert result.gain.max_rate_db_s <= 1.5 + 1e-6
        assert result.limiter["tp_4x_dbtp"] <= -1.0
        # 报告数字的合理范围（对照 PoC：V2 后 5 min 半展宽 0.26–1.44 dB）
        assert result.v2["clamp_db_used"] in (3.0, 6.0)
        assert result.gain.min_db <= result.gain_db <= result.gain.max_db
        assert result.window5m_spread_out_db is not None
        assert result.window5m_spread_out_db <= 2.8  # 半展宽 ≤1.4 dB（PoC 上界）
        span = [w["asl_dbfs"] for w in result.v2["window5m_out"]["windows"]]
        print(
            "F24_ACCEPTANCE " + json.dumps({
                "clean_s": round(plan.clean_duration_s, 2),
                "speech_intervals": len(intervals),
                "asl_in_dbfs": result.asl_in_dbfs,
                "asl_out_dbfs": result.asl_out_dbfs,
                "asl_dev_db": result.asl_dev_db,
                "gain_db": result.gain_db,
                "gain_min_db": result.gain.min_db,
                "gain_max_db": result.gain.max_db,
                "gain_type": result.gain.type,
                "max_rate_db_s": result.gain.max_rate_db_s,
                "clamp_db_used": result.v2["clamp_db_used"],
                "high_drift": result.v2["high_drift"],
                "drift_in_db": result.v2.get("drift_in_db"),
                "n_windows_used": result.v2["n_windows_used"],
                "window5m_spread_out_db": result.window5m_spread_out_db,
                "window5m_half_spread_out_db": round(result.window5m_spread_out_db / 2.0, 2),
                "window5m_levels": span,
                "tp_wav_dbtp": result.tp_wav_dbtp,
                "tp_4x_dbtp": result.limiter["tp_4x_dbtp"],
                "clipped_samples": result.clipped_samples,
                "limiter_trigger_ratio": result.limiter["trigger_ratio"],
                "limiter_min_gain_db": result.limiter["min_gain_db"],
                "limiter_mean_gain_db": result.limiter["mean_gain_db"],
                "limiter_events": result.limiter["trigger_events"],
                "asl_iterations": result.iterations,
                "timings": result.timings,
                "ok": result.ok,
            }, ensure_ascii=False)
        )
    finally:
        for path in (clean44, out44):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass
