# -*- coding: utf-8 -*-
"""P1-T2 解码与缓存单测。

验收要求：
- 6 个样例全部 ``ok=True``，其中 f00 ``repaired=True``；
- 单文件解码 ≤10 s（百分钟级）；
- 16 kHz 必须 float32（源存在 >0 dBFS 过冲，先削成 int16 会丢失判断依据）。
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from app.models import make_job_id
from app.pipeline.decode import (
    ANALYSIS_SR,
    RENDER_SR,
    DecodeError,
    cache_size_bytes,
    cleanup_cache,
    decode,
    decode_audio,
    ensure_44k,
    probe_media,
    read_wav_float32,
    wav_info,
)
from app.pipeline import decode as decode_mod

from conftest import requires_data, requires_ffmpeg


# ---------------------------------------------------------------- 造测试音频
def _sine_wav(path: Path, seconds: float = 2.5, sr: int = 44100,
              freq: float = 440.0, amp: float = 0.5) -> Path:
    import scipy.io.wavfile as wavfile

    t = np.arange(int(sr * seconds)) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    wavfile.write(str(path), sr, x)
    return path


def _encode_m4a(wav: Path, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav),
         "-c:a", "aac", "-b:a", "96k", str(out)],
        check=True,
    )
    return out


@pytest.fixture()
def tiny_m4a(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("PATH 中缺 ffmpeg")
    wav = _sine_wav(tmp_path / "sine.wav")
    return _encode_m4a(wav, tmp_path / "sine.m4a")


# ---------------------------------------------------------------- WAV 头解析
def test_wav_info_float32(tmp_path: Path):
    wav = _sine_wav(tmp_path / "a.wav", seconds=1.0)
    info = wav_info(wav)
    assert info.sample_rate == 44100
    assert info.channels == 1
    assert info.sampwidth == 4
    assert info.n_frames == 44100
    assert info.duration_s == pytest.approx(1.0, abs=1e-6)


def test_wav_info_rejects_non_wav(tmp_path: Path):
    bad = tmp_path / "a.m4a"
    bad.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64)
    with pytest.raises(DecodeError, match="RIFF"):
        wav_info(bad)
    with pytest.raises(DecodeError, match="不存在"):
        wav_info(tmp_path / "missing.wav")


def test_read_wav_float32_keeps_overshoot(tmp_path: Path):
    """float32 wav 里 >1.0 的样本不能被削掉（源过冲信息必须保留）。"""
    import scipy.io.wavfile as wavfile

    x = np.array([0.0, 0.5, 1.0, 1.5, -1.8, 0.2], dtype=np.float32)
    path = tmp_path / "hot.wav"
    wavfile.write(str(path), 16000, x)
    sr, y = read_wav_float32(path)
    assert sr == 16000
    assert y.dtype == np.float32
    assert y.max() == pytest.approx(1.5)
    assert y.min() == pytest.approx(-1.8)


# ---------------------------------------------------------------- 探测
@requires_ffmpeg
def test_probe_media(tiny_m4a: Path):
    info = probe_media(tiny_m4a)
    assert info.duration_s == pytest.approx(2.5, abs=0.15)
    assert info.sample_rate == 44100
    assert info.channels == 1
    assert info.codec == "aac"
    assert info.bit_rate > 0
    assert info.to_dict()["duration_s"] == pytest.approx(2.5, abs=0.15)


def test_probe_media_missing_file(tmp_path: Path):
    with pytest.raises(DecodeError, match="不存在"):
        probe_media(tmp_path / "nope.m4a")


# ---------------------------------------------------------------- 解码主流程
@requires_ffmpeg
def test_decode_basic_and_cache(tiny_m4a: Path, cache_dir: Path, cfg):
    job_id = make_job_id(tiny_m4a)
    result = decode(tiny_m4a, cache_dir=cache_dir, job_id=job_id, cfg=cfg)
    assert result.ok is True, result.error
    assert result.repaired is False
    assert result.sample_rate == ANALYSIS_SR
    assert result.orig_sample_rate == 44100
    assert result.channels == 1
    assert result.from_cache is False
    assert result.duration_s == pytest.approx(2.5, abs=0.15)
    assert Path(result.wav16k).is_file()
    assert result.wav44k is None
    assert result.timings["decode16k_s"] > 0

    info = wav_info(result.wav16k)
    assert info.sample_rate == ANALYSIS_SR
    assert info.sampwidth == 4  # float32
    assert info.n_frames > 0

    # 第二次：命中缓存，不再调用 ffmpeg
    started = time.perf_counter()
    again = decode(tiny_m4a, cache_dir=cache_dir, job_id=job_id, cfg=cfg)
    assert again.ok and again.from_cache is True
    assert again.wav16k == result.wav16k
    assert time.perf_counter() - started < 1.0
    # 缓存元数据落盘
    meta = cache_dir / f"{job_id}.decode.json"
    assert meta.is_file()
    assert "decode16k_s" in result.timings


@requires_ffmpeg
def test_decode_need_44k(tiny_m4a: Path, cache_dir: Path, cfg):
    result = decode(tiny_m4a, cache_dir=cache_dir, job_id="j44", need_44k=True, cfg=cfg)
    assert result.ok is True, result.error
    assert result.wav44k is not None
    info = wav_info(result.wav44k)
    assert info.sample_rate == RENDER_SR
    assert info.sampwidth == 4


@requires_ffmpeg
def test_decode_force_bypasses_cache(tiny_m4a: Path, cache_dir: Path, cfg):
    first = decode(tiny_m4a, cache_dir=cache_dir, job_id="jf", cfg=cfg)
    assert first.ok
    forced = decode(tiny_m4a, cache_dir=cache_dir, job_id="jf", cfg=cfg, force=True)
    assert forced.ok
    assert forced.from_cache is False
    # 源文件变化 → 缓存自动失效
    import os

    os.utime(tiny_m4a, ns=(time.time_ns(), time.time_ns() + 10**9))
    changed = decode(tiny_m4a, cache_dir=cache_dir, job_id="jf", cfg=cfg)
    assert changed.ok and changed.from_cache is False


def test_decode_missing_file(tmp_path: Path, cache_dir: Path, cfg):
    result = decode(tmp_path / "nope.m4a", cache_dir=cache_dir, job_id="jx", cfg=cfg)
    assert result.ok is False
    assert "不存在" in result.error


@requires_ffmpeg
def test_decode_garbage_file(tmp_path: Path, cache_dir: Path, cfg):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not a media file at all" * 10)
    result = decode(bad, cache_dir=cache_dir, job_id="jbad", cfg=cfg)
    assert result.ok is False
    assert result.error  # 必须有可读的失败原因
    assert result.wav16k is None


def test_validate_decoded_rejects_duration_mismatch(tmp_path: Path, cfg):
    wav = _sine_wav(tmp_path / "short.wav", seconds=1.0)
    ok, reason = decode_mod._validate_decoded(wav, expected_duration=100.0, cfg=cfg)
    assert ok is False
    assert "时长异常" in reason
    ok, reason = decode_mod._validate_decoded(tmp_path / "missing.wav", 10.0, cfg)
    assert ok is False and reason


@requires_ffmpeg
def test_cleanup_cache(tiny_m4a: Path, cache_dir: Path, cfg):
    result = decode(tiny_m4a, cache_dir=cache_dir, job_id="jc", need_44k=True, cfg=cfg)
    assert result.ok and cache_size_bytes(cache_dir) > 0
    removed = cleanup_cache(cache_dir, "jc", keep_16k=True)
    names = {Path(p).name for p in removed}
    assert "jc.44k.wav" in names and "jc.decode.json" in names
    assert cache_dir.joinpath("jc.16k.wav").is_file()
    # 全部清掉
    cleanup_cache(cache_dir, None, keep_16k=False)
    assert cache_size_bytes(cache_dir) == 0


# ---------------------------------------------------------------- 真实样例（P1-T2 验收）
@requires_data
@requires_ffmpeg
def test_decode_real_f00_needs_repair(samples, cache_dir: Path, cfg):
    src = samples["f00"]
    started = time.perf_counter()
    result = decode(src, cache_dir=cache_dir, job_id="f00", cfg=cfg)
    elapsed = time.perf_counter() - started
    assert result.ok is True, result.error
    assert result.repaired is True, "f00 必须走 stco 修复路径"
    assert result.duration_s == pytest.approx(3048.6, abs=5.0)   # 50.8 min
    assert result.orig_sample_rate == 32000
    assert result.report_decode() == {"repaired": True, "ok": True}
    assert elapsed <= 10.0, f"单文件解码应 ≤10 s，实测 {elapsed:.2f}s"
    # 修复副本用完即删：cache 里不残留 .fixed.m4a
    assert not list(cache_dir.glob("*.fixed.*"))
    # 解码结果不是静音
    _sr, x = read_wav_float32(result.wav16k)
    assert float(np.max(np.abs(x))) > 0.05


@requires_data
@requires_ffmpeg
def test_decode_real_f04_keeps_overshoot(samples, cache_dir: Path, cfg):
    """f04 源头真峰值 >0 dBFS：16k float32 解码后必须仍能看到 >1.0 的样本。"""
    result = decode(samples["f04"], cache_dir=cache_dir, job_id="f04", cfg=cfg)
    assert result.ok is True, result.error
    _sr, x = read_wav_float32(result.wav16k)
    assert float(np.max(np.abs(x))) > 1.0, "float32 解码丢了过冲信息（可能被削成 int16）"


@requires_data
@requires_ffmpeg
def test_decode_real_f00_44k_after_repair(samples, cache_dir: Path, cfg):
    """44.1k 渲染轴对修复文件同样要能解出来。"""
    result = decode(samples["f00"], cache_dir=cache_dir, job_id="f00x", need_44k=True, cfg=cfg)
    assert result.ok is True, result.error
    assert result.repaired is True
    assert result.wav44k is not None
    assert wav_info(result.wav44k).sample_rate == RENDER_SR
    assert not list(cache_dir.glob("*.fixed.*"))
    # ensure_44k 走缓存，返回同一路径
    assert ensure_44k(samples["f00"], cache_dir=cache_dir, job_id="f00x", cfg=cfg) is not None


@requires_data
@requires_ffmpeg
def test_decode_all_six_samples(samples, tmp_path: Path, cfg):
    """6 个样例全部解码成功；只有 f00 需要修复；单文件 ≤10 s。"""
    assert len(samples) == 6, f"样例数应为 6，实测 {list(samples)}"
    cache = tmp_path / "cache"
    repaired_flags = {}
    try:
        for key, path in samples.items():
            started = time.perf_counter()
            result = decode(path, cache_dir=cache, job_id=key, cfg=cfg)
            elapsed = time.perf_counter() - started
            assert result.ok is True, f"{key}: {result.error}"
            assert result.sample_rate == ANALYSIS_SR
            assert elapsed <= 10.0, f"{key} 解码耗时 {elapsed:.2f}s > 10s"
            repaired_flags[key] = result.repaired
        assert repaired_flags["f00"] is True
        assert all(not repaired_flags[k] for k in repaired_flags if k != "f00")
    finally:
        cleanup_cache(cache, None, keep_16k=False)
