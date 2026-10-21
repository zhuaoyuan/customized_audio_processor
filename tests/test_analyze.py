# -*- coding: utf-8 -*-
"""P2 分析单测：VAD 段处理 / ASL 与噪声底 / 5min 窗口 / 内部长静音 / 头尾边界。

分两层：
- **纯函数用例**（默认跑）：帧能量、P.56、掩码、长静音、窗口统计、边界判定（注入假转写器）；
- **真实样例用例**（``@pytest.mark.data`` / ``@pytest.mark.slow``）：6 文件 VAD/电平对报告表、
  f24 内部长静音、头尾边界（含 ASR，较慢）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import pytest

from app.models import AnalyzeResult
from app.pipeline.analyze import (
    AnalyzeError,
    WindowRate,
    analyze,
    analyze_wav,
    classify_segments,
    detect_boundaries,
    frame_power,
    internal_silence,
    make_windows,
    mean_level_db,
    noise_floor_db,
    p56_level,
    percentile_stats,
    power_to_db,
    segment_total,
    speech_mask,
    window_levels,
)
from app.pipeline import asr as asr_mod

from conftest import DATA_DIR, requires_data, requires_ffmpeg


#: 报告 §2.2 的语音 ASL / 噪声底 / SNR / 语音占比（同文件验收基准）
REPORT_2_2 = {
    "f00": {"asl": -39.8, "noise": -49.8, "snr": 10.0, "ratio": 0.66},
    "f04": {"asl": -11.9, "noise": -50.5, "snr": 38.5, "ratio": 0.80},
    "f05": {"asl": -12.7, "noise": -36.5, "snr": 23.8, "ratio": 0.85},
    "f06": {"asl": -21.9, "noise": -29.7, "snr": 7.8, "ratio": 0.70},
    "f20": {"asl": -23.7, "noise": -49.7, "snr": 26.0, "ratio": 0.78},
    "f24": {"asl": -10.1, "noise": -28.2, "snr": 18.1, "ratio": 0.68},
}

#: 样例 key → work/scan/vad/<short>.json 的文件名（P2-T1 段数对照）
SCAN_SHORT = {
    "f00": "f00_xifang20140227",
    "f04": "f04_lixiangguo4",
    "f05": "f05_wulixue",
    "f06": "f06_lunyu",
    "f20": "f20_lixiangguo20141219",
    "f24": "f24_xingershangxue",
}

SCAN_VAD_DIR = DATA_DIR.parent / "work" / "scan" / "vad"


# ---------------------------------------------------------------- 帧能量
def test_frame_power_constant_signal():
    sr = 16000
    x = np.full(sr, 0.5, dtype=np.float32)  # 1 s 直流 0.5
    power, centers = frame_power(x, sr)
    assert power.shape == centers.shape
    assert len(power) == 1 + (sr - 400) // 160  # 25 ms 窗 / 10 ms 跳
    assert np.allclose(power, 0.25, atol=1e-6)
    assert np.allclose(power_to_db(power), -6.0206, atol=1e-3)
    assert centers[0] == pytest.approx(0.0125, abs=1e-6)


def test_frame_power_short_and_empty():
    # 1000 样本、窗 400 / 跳 160 → 1 + (1000-400)//160 = 4 帧（短文件不崩）
    power, centers = frame_power(np.zeros(1000, dtype=np.float32))
    assert len(power) == 4
    power, centers = frame_power(np.zeros(0, dtype=np.float32))
    assert len(power) == 0 and len(centers) == 0


def test_frame_power_matches_bruteforce():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, size=16000 * 3).astype(np.float32)
    power, centers = frame_power(x, 16000)
    w, h = 400, 160
    for k in (0, 10, len(power) - 1):
        seg = x[k * h: k * h + w].astype(np.float64)
        assert power[k] == pytest.approx(float(np.mean(seg ** 2)), rel=1e-9)
        assert centers[k] == pytest.approx((k * h + w / 2) / 16000, abs=1e-9)


# ---------------------------------------------------------------- 电平
def test_p56_level_excludes_quiet_frames():
    loud = np.full(100, 1e-2)   # -20 dBFS
    quiet = np.full(100, 1e-6)  # -60 dBFS
    power = np.concatenate([loud, quiet])
    # 均值 = (1 + 1e-4) / 200 = 5.0005e-3 → -23.01 dB
    assert mean_level_db(power) == pytest.approx(10 * math.log10(5.0005e-3), abs=0.01)
    # P.56 迭代把 -60 dB 的帧剔除，回到 -20 dB
    assert p56_level(power, rel_db=15.0) == pytest.approx(-20.0, abs=0.05)


def test_noise_floor_db_uses_low_percentile():
    frames = np.concatenate([np.full(90, 1e-2), np.full(10, 1e-8)])
    assert noise_floor_db(frames, 10.0) == pytest.approx(-80.0, abs=0.1)
    assert noise_floor_db(np.zeros(0)) != noise_floor_db(np.zeros(0))  # nan


def test_percentile_stats():
    frames = np.concatenate([np.full(50, 1e-2), np.full(50, 1e-4)])
    stats = percentile_stats(frames, [10, 50, 90])
    assert set(stats) == {"p10", "p50", "p90"}
    assert stats["p10"] == pytest.approx(-40.0, abs=0.5)
    # 线性插值：一半 -20 dB、一半 -40 dB → p50 落在中间 -30 dB
    assert stats["p50"] == pytest.approx(-30.0, abs=0.5)
    assert stats["p90"] == pytest.approx(-20.0, abs=0.5)


# ---------------------------------------------------------------- 段工具
def test_speech_mask_by_frame_center():
    centers = np.array([0.005, 0.015, 0.025, 0.035])
    mask = speech_mask([(0.0, 0.01), (0.03, 4.0)], len(centers), centers)
    assert mask.tolist() == [True, False, False, True]


def test_classify_segments_and_total():
    cont, frag = classify_segments([(0, 5), (6, 6.5), (10, 13.2)], 3.0)
    assert cont == [(0.0, 5.0), (10.0, 13.2)]
    assert frag == [(6.0, 6.5)]
    assert segment_total(cont) == pytest.approx(8.2)


# ---------------------------------------------------------------- 内部长静音
def test_internal_silence_counts_and_guard():
    segs = [(0, 5), (10, 15), (30, 32)]
    result = internal_silence(segs, duration=40.0, min_s=3.0, guard=0.25)
    assert result["n_internal"] == 2            # [5,10] 与 [15,30]
    assert result["total_s"] == pytest.approx(20.0)
    assert result["longest_s"] == pytest.approx(15.0)
    internal = [c for c in result["candidates"] if c["edge"] == "internal"]
    assert internal[0]["cut_start"] == pytest.approx(5.25)
    assert internal[0]["cut_end"] == pytest.approx(9.75)
    assert result["n_gaps"] == 3                # 片尾 [32,40]


def test_internal_silence_marks_head_tail_regions():
    segs = [(0, 5), (10, 15), (30, 32)]
    result = internal_silence(segs, duration=40.0, head_s=12.0, tail_s=35.0)
    # [5,10] 落在头裁剪区，[15,30] 才是真正要内部切除的
    assert result["total_s"] == pytest.approx(20.0)
    assert result["in_head_tail_s"] == pytest.approx(5.0)
    assert result["cut_s"] == pytest.approx(15.0)


# ---------------------------------------------------------------- 窗口统计
def test_window_levels_basic():
    sr = 16000
    power, centers = frame_power(np.full(sr * 620, 0.1, dtype=np.float32), sr)
    mask = np.ones(len(power), dtype=bool)
    result = window_levels(power, mask, centers, duration=620.0, win_s=300.0,
                           min_speech_s=30.0)
    assert result["n_windows"] == 2
    assert result["half_spread_db"] == pytest.approx(0.0, abs=0.01)
    assert result["p50"] == pytest.approx(10 * math.log10(0.01), abs=0.1)


def test_window_levels_skips_low_speech_windows():
    sr = 16000
    power, centers = frame_power(np.full(sr * 600, 0.1, dtype=np.float32), sr)
    mask = np.zeros(len(power), dtype=bool)
    mask[: int(10 / 0.01)] = True  # 只有 10 s 语音
    result = window_levels(power, mask, centers, duration=600.0, win_s=300.0,
                           min_speech_s=30.0)
    assert result["n_windows"] == 0


def test_make_windows_absolute_alignment():
    windows = make_windows(450.0, 900.0, 120.0)
    assert windows[0] == (360.0, 480.0)   # 对齐到 120 的整数倍
    assert windows[-1] == (840.0, 900.0)  # 边界截断
    assert all(t1 - t0 <= 120.0 + 1e-9 for t0, t1 in windows)


# ---------------------------------------------------------------- 边界判定（注入假转写器）
SR = 16000


class FakeTranscriber:
    """按音量决定字符率的假转写器：安静段返回空文本，响段按 3 字/秒返回。"""

    def __init__(self, cps: float = 3.0, threshold: float = 0.01) -> None:
        self.cps = cps
        self.threshold = threshold

    def transcribe(self, clip: np.ndarray, sr: int = SR) -> str:
        if clip.size == 0:
            return ""
        rms = float(np.sqrt(np.mean(np.asarray(clip, dtype=np.float64) ** 2)))
        if rms <= self.threshold:
            return ""
        n = int(round(clip.size / sr * self.cps))
        return "讲" * max(0, n)


def _make_signal(duration_s: float, loud_from: float, loud_until: float,
                 loud_amp: float = 0.3, quiet_amp: float = 0.0005) -> np.ndarray:
    x = np.full(int(duration_s * SR), quiet_amp, dtype=np.float32)
    a = int(loud_from * SR)
    b = int(min(duration_s, loud_until) * SR)
    if b > a:
        x[a:b] = loud_amp
    return x


def test_boundary_head_detected(cfg):
    # 0–480 s 安静（音乐/闲聊），480 s 后开始讲课
    x = _make_signal(1200.0, 480.0, 1200.0)
    segs = [(0.0, 1200.0)]  # 段表只影响 speech_ratio 展示，不影响字符率
    result = detect_boundaries(x, segs, 1200.0, cfg, FakeTranscriber(), SR)
    assert result["method"] == "window_asr"
    assert result["head"]["boundary_s"] == pytest.approx(479.5, abs=0.01)
    assert result["head"]["flags"] == []
    assert result["head"]["windows"][4]["teaching"] is True   # 480–600
    assert result["head"]["windows"][0]["teaching"] is False


def test_boundary_head_keeps_soft_intro(cfg):
    """f00 形态：0–120 s 是课前寒暄（cps 1.5，软窗口），不应被切掉。"""
    duration = 600.0
    x = np.zeros(int(duration * SR), dtype=np.float32)
    # 0–30 s 与 60–90 s 有声、其余静音 → 0–120 s 窗口 cps≈1.5（软窗口，未达标）
    x[0: int(30 * SR)] = 0.3
    x[int(60 * SR): int(90 * SR)] = 0.3
    x[int(120 * SR):] = 0.3
    segs = [(0.0, 600.0)]
    result = detect_boundaries(x, segs, duration, cfg, FakeTranscriber(), SR)
    assert result["head"]["windows"][0]["cps"] == pytest.approx(1.5, abs=0.1)
    assert result["head"]["windows"][0]["teaching"] is False
    # 达标窗口从 120 s 开始，但向前吸收了软窗口 → 头边界回到 0
    assert result["head"]["boundary_s"] == 0.0
    assert result["head"]["window_start_s"] == 0.0


def test_boundary_head_refined_to_30s(cfg):
    """f20 形态：0–90 s 静音，**90–120 s 已是讲课**（cps 3.4）。

    120 s 窗口只能给到 119.5 s → 会白切 30 s 讲课；30 s 块级精修应把边界挪到 89.5 s。
    """
    duration = 600.0
    x = np.zeros(int(duration * SR), dtype=np.float32)
    x[int(90 * SR):] = 0.3   # 90 s 起持续讲课
    segs = [(90.0, 600.0)]
    result = detect_boundaries(x, segs, duration, cfg, FakeTranscriber(), SR)
    assert "head_refined_30s" in result["head"]["flags"]
    assert result["head"]["boundary_s"] == pytest.approx(89.5, abs=0.01)
    assert result["head"]["refined_from_s"] == pytest.approx(90.0, abs=0.01)
    # 窗口表本身仍是 120 s 网格（0–120 未达标），说明是「窗级定位 + 块级精修」两段式
    assert result["head"]["windows"][0]["teaching"] is False


def test_refine_helpers():
    from app.pipeline.analyze import _refine_head_start, _refine_tail_end

    class StubChars:
        def __init__(self, fn):
            self.fn = fn

        def cps(self, t0, t1):
            return self.fn(t0, t1)

    # 头：0–120 窗口里从 90 s 起达标 → 精修到 90
    head = _refine_head_start(
        StubChars(lambda a, b: 3.0 if a >= 90 else 0.0), 120.0, 120.0, 2.0, 30.0
    )
    assert head == pytest.approx(90.0)
    # 全静音 → 不精修
    assert _refine_head_start(StubChars(lambda a, b: 0.0), 120.0, 120.0, 2.0, 30.0) is None
    # 尾：360 s 后静音 → 块按起点判定，延伸到 420（块 390–420 的起点 <400 仍算有声）
    tail = _refine_tail_end(
        StubChars(lambda a, b: 3.0 if a < 400 else 0.0), 360.0, 600.0, 1.0, 120.0, 30.0
    )
    assert tail == pytest.approx(420.0)
    # 延长上限 max_extra_s 生效
    capped = _refine_tail_end(
        StubChars(lambda a, b: 5.0), 360.0, 600.0, 1.0, 60.0, 30.0
    )
    assert capped == pytest.approx(420.0)


def test_boundary_tail_detected(cfg):
    # 1200 s 文件：0–900 s 讲课，之后是片尾静音。
    # 840–960 s 窗口只有 60 s 讲课 → cps=1.5（软窗口），严格阈值不算达标，
    # 但片尾允许向后延伸 1 个软窗口（收尾常是轻声，宁少裁）→ 边界 = 960.5。
    x = _make_signal(1200.0, 0.0, 900.0)
    segs = [(0.0, 1200.0)]
    result = detect_boundaries(x, segs, 1200.0, cfg, FakeTranscriber(), SR)
    assert result["tail"]["boundary_s"] == pytest.approx(960.5, abs=0.01)
    assert result["tail"]["flags"] == []


def test_boundary_head_allows_one_soft_window(cfg):
    """验证 f06 形态：达标–软窗口–达标 也能判出起点，而不是跳到后面。"""
    # 0–120 s 音乐；120–240 s 讲课；240–360 s 只有一半时间讲课（cps≈1.5，软）；
    # 360 s 之后正常讲课 → 头边界应落在 119.5，而不是 359.5。
    duration = 720.0
    x = np.zeros(int(duration * SR), dtype=np.float32)
    x[int(120 * SR): int(240 * SR)] = 0.3
    x[int(240 * SR): int(300 * SR)] = 0.3   # 240–300 响，300–360 静 → 该窗 1.5 cps
    x[int(360 * SR):] = 0.3
    segs = [(120.0, 720.0)]
    result = detect_boundaries(x, segs, duration, cfg, FakeTranscriber(), SR)
    assert result["head"]["boundary_s"] == pytest.approx(119.5, abs=0.01)
    assert result["head"]["windows"][1]["cps"] == pytest.approx(2.97, abs=0.1)
    assert result["head"]["windows"][2]["teaching"] is False
    assert result["head"]["windows"][2]["cps"] == pytest.approx(1.48, abs=0.1)


def test_boundary_all_silent_is_uncertain(cfg):
    x = _make_signal(1200.0, 0.0, 0.0)
    result = detect_boundaries(x, [(0.0, 1200.0)], 1200.0, cfg, FakeTranscriber(), SR)
    assert result["head"]["boundary_s"] == 0.0
    assert result["tail"]["boundary_s"] == pytest.approx(1200.0)
    assert "head_uncertain" in result["head"]["flags"]
    assert "tail_uncertain" in result["tail"]["flags"]


def test_boundary_extended_when_beyond_scan(cfg):
    # 讲课要到 1080 s 才开始（超过 15 min 扫描窗）
    x = _make_signal(1500.0, 1080.0, 1500.0)
    result = detect_boundaries(x, [(0.0, 1500.0)], 1500.0, cfg, FakeTranscriber(), SR)
    assert "head_extended" in result["head"]["flags"]
    assert result["head"]["boundary_s"] == pytest.approx(1079.5, abs=0.01)


def test_boundary_disabled_without_transcriber(cfg):
    x = _make_signal(600.0, 0.0, 600.0)
    result = detect_boundaries(x, [(0.0, 600.0)], 600.0, cfg, None, SR)
    assert result["method"] == "disabled"
    assert result["head"]["boundary_s"] == 0.0
    assert result["tail"]["boundary_s"] == pytest.approx(600.0)
    assert "asr_disabled" in result["head"]["flags"]


def test_window_char_rates_partial_and_teaching(cfg):
    from app.pipeline.analyze import window_char_rates

    x = _make_signal(240.0, 0.0, 240.0)  # 全程响 → 假转写器按 3 字/秒
    segs = [(0.0, 240.0)]
    # 不足半个窗口（40 s < 60 s）→ partial_window，不做 ASR、不算达标
    partial = window_char_rates(x, segs, [(120.0, 160.0)], cfg, FakeTranscriber(), SR)[0]
    assert partial.note == "partial_window"
    assert partial.teaching is False and partial.cps == 0.0
    # 完整窗口 → cps≈3.0，判为讲课
    full = window_char_rates(x, segs, [(0.0, 120.0)], cfg, FakeTranscriber(), SR)[0]
    assert full.cps == pytest.approx(3.0, abs=0.2)
    assert full.teaching is True
    # 没给转写器 → no_asr，不判讲课
    no_asr = window_char_rates(x, segs, [(0.0, 120.0)], cfg, None, SR)[0]
    assert no_asr.note == "no_asr" and no_asr.teaching is False


# ---------------------------------------------------------------- ASR 文本工具
def test_asr_clean_text_and_count():
    raw = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>今天讲亚里士多德《形而上学》。😀 好"
    cleaned = asr_mod.clean_text(raw)
    assert cleaned == "今天讲亚里士多德《形而上学》。好"
    assert asr_mod.count_speech_chars(cleaned) == 13  # 12 个汉字 + 好
    assert asr_mod.count_speech_chars("abc 123，中文") == 8
    assert asr_mod.count_speech_chars("") == 0


def test_asr_split_long_segment_and_slice():
    parts = asr_mod.split_long_segment(10.0, 75.0, 30.0)
    assert parts == [(10.0, 31.667), (31.667, 53.333), (53.333, 75.0)]
    assert asr_mod.split_long_segment(0.0, 20.0, 30.0) == [(0.0, 20.0)]
    x = np.arange(16000 * 3, dtype=np.float32)
    clip = asr_mod.slice_audio(x, 1.0, 2.5)
    assert clip.shape == (24000,)
    assert asr_mod.slice_audio(x, 5.0, 6.0).size == 0


def test_asr_unavailable_when_model_missing(cfg):
    bad = cfg.replace({"paths.models.sensevoice_dir": str(cfg.root / "no_such_model")})
    with pytest.raises(asr_mod.AsrUnavailable, match="模型文件不存在"):
        asr_mod.create_transcriber(bad)


# ---------------------------------------------------------------- analyze_wav 主流程
def _write_wav(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    import scipy.io.wavfile as wavfile
    wavfile.write(str(path), sr, np.ascontiguousarray(x, dtype=np.float32))
    return path


def test_analyze_wav_without_asr(cfg, tmp_path: Path):
    x = _make_signal(640.0, 100.0, 500.0, loud_amp=0.2)
    wav = _write_wav(tmp_path / "t.16k.wav", x)
    segs = [(100.0, 500.0)]
    result = analyze_wav(wav, cfg, job_id="t", run_boundary=False, segments=segs,
                         use_cache=True)
    assert isinstance(result, AnalyzeResult)
    assert result.ok
    assert result.duration_s == pytest.approx(640.0, abs=0.01)
    assert result.vad["n_segments"] == 1
    assert result.vad["speech_ratio"] == pytest.approx(400 / 640, abs=1e-3)
    assert "head_uncertain" in result.trim["flags"]
    assert result.levels["asl_in_dbfs"] == pytest.approx(-13.98, abs=0.3)  # 0.2 幅值 → -14 dB
    assert result.trim["head_s"] == 0.0
    assert result.trim["tail_s"] == pytest.approx(640.0)
    # 缓存命中
    cached = analyze_wav(wav, cfg, job_id="t", run_boundary=False, segments=segs)
    assert cached.from_cache is True
    # force 重算
    forced = analyze_wav(wav, cfg, job_id="t", run_boundary=False, segments=segs, force=True)
    assert forced.from_cache is False
    assert (cfg.path("cache_dir") / "t.analyze.json").is_file()
    # segments 参数可复现
    assert result.segments[0] == [100.0, 500.0]


def test_analyze_wav_missing_file(cfg, tmp_path: Path):
    with pytest.raises(AnalyzeError, match="先跑 decode"):
        analyze_wav(tmp_path / "nope.wav", cfg, job_id="x")


# ---------------------------------------------------------------- 真实样例：VAD + 电平
@pytest.fixture(scope="session")
def decoded_samples(samples, session_cfg):
    """一次性把 6 个样例解码到会话级临时缓存，供分析用例复用。"""
    from app.pipeline.decode import decode

    base = session_cfg.path("cache_dir")
    out = {}
    for key, path in samples.items():
        result = decode(path, cache_dir=base, job_id=key, cfg=session_cfg)
        assert result.ok, f"{key}: {result.error}"
        out[key] = Path(result.wav16k)
    return out


@pytest.fixture(scope="session")
def analyzed_samples(decoded_samples, session_cfg):
    """6 文件各分析一次（关 ebur128 提速），三个数据用例共享，避免重复 VAD。"""
    cfg_off = session_cfg.replace({"analyze.ebur128": False, "analyze.cache_enabled": False})
    return {
        key: analyze_wav(wav, cfg_off, job_id=f"an_{key}", run_boundary=False)
        for key, wav in decoded_samples.items()
    }


@requires_data
@requires_ffmpeg
def test_data_vad_and_levels_match_report(analyzed_samples):
    """P2-T1/T2 验收：6 文件 speech_ratio 0.60–0.90；ASL/噪声/SNR 与报告 §2.2 偏差 <1 dB。"""
    for key, result in analyzed_samples.items():
        vad, levels, ref = result.vad, result.levels, REPORT_2_2[key]

        assert 0.60 <= vad["speech_ratio"] <= 0.90, f"{key} speech_ratio={vad['speech_ratio']}"
        assert abs(levels["asl_in_dbfs"] - ref["asl"]) <= 1.0, (
            f"{key} ASL 实测{levels['asl_in_dbfs']} vs 报告{ref['asl']}"
        )
        # 报告 §2.2 的噪声底/SNR 用「全部 VAD 段」口径，这里同时校验
        assert abs(levels["noise_floor_dbfs"] - ref["noise"]) <= 1.0, key
        assert abs(levels["snr_db"] - ref["snr"]) <= 1.5, key
        assert abs(vad["speech_ratio"] - ref["ratio"]) <= 0.05, key

        # 段数与 work/scan/vad/*.json 差异 <10%
        scan_path = SCAN_VAD_DIR / f"{SCAN_SHORT[key]}.json"
        if scan_path.is_file():
            expected = json.loads(scan_path.read_text(encoding="utf-8"))["vad"]["n_segments"]
            diff = abs(vad["n_segments"] - expected) / expected
            assert diff < 0.10, f"{key} 段数 {vad['n_segments']} vs scan {expected}（{diff:.1%}）"


@requires_data
@requires_ffmpeg
def test_data_window_spread_reasonable(analyzed_samples):
    """5 min 窗口必须能算出来（6 文件都有多次窗口），且波动量级在报告范围内（1–8 dB）。"""
    for key, result in analyzed_samples.items():
        win = result.levels["window5m"]
        assert win["n_windows"] >= 3, f"{key} 窗口数 {win['n_windows']}"
        spread = win["spread_db"]
        assert spread is not None and 0.0 <= spread <= 12.0, f"{key} spread={spread}"


@requires_data
@requires_ffmpeg
def test_data_f24_internal_silence(analyzed_samples):
    """P2-T4 验收：f24 内部 ≥3 s 非语音总时长 ≈613 s（±10%），最长 ≈170 s。"""
    silence = analyzed_samples["f24"].internal_silence
    assert silence["total_s"] == pytest.approx(613.0, rel=0.10), silence["total_s"]
    assert silence["longest_s"] == pytest.approx(170.0, rel=0.15), silence["longest_s"]
    assert silence["n_internal"] >= 30


@requires_data
@requires_ffmpeg
@pytest.mark.slow
def test_data_boundaries_six_files(decoded_samples, session_cfg):
    """P2-T3 验收：f24 head 480±30 s；f06/f20 ≈121 s；f05/f00/f04 ≈0–10 s；f24 tail ≈4560 s。

    需要 SenseVoice，单文件约 20–50 s，故标 slow（``pytest -m slow`` 显式运行）。
    """
    cfg_asr = session_cfg.replace({"analyze.ebur128": False, "analyze.cache_enabled": False})
    heads = {}
    tails = {}
    for key, wav in decoded_samples.items():
        result = analyze(wav, cfg_asr, job_id=f"bnd_{key}", run_boundary=True)
        heads[key] = result.trim["head_s"]
        tails[key] = result.trim["tail_s"]
        assert result.boundary["method"] == "window_asr"
        assert not [f for f in result.trim["flags"] if f.endswith("_uncertain")], result.trim["flags"]

    assert heads["f24"] == pytest.approx(480.0, abs=30.0), heads
    assert tails["f24"] == pytest.approx(4560.0, abs=30.0), tails
    # f06：起点 120 s（90–120 s 只有 19 字课前闲聊）；f20：90 s 就开讲 → 块级精修到 89.5
    assert 105.0 <= heads["f06"] <= 140.0, f"f06 head={heads['f06']}"
    assert 85.0 <= heads["f20"] <= 100.0, f"f20 head={heads['f20']}"
    assert heads["f05"] <= 10.0, heads["f05"]
    assert heads["f00"] <= 10.0, heads["f00"]
    assert heads["f04"] <= 10.0, heads["f04"]
    # 片尾：f00/f20 其实一直讲到结尾（30 s 块级精修补回），f24 尾部是噪声
    assert tails["f00"] >= 3040.0, tails["f00"]
    assert tails["f20"] >= 6146.0, tails["f20"]
