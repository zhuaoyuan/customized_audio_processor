# -*- coding: utf-8 -*-
"""P3-T4 输出模块单测：m4a 编码/回读契约、削波检查、report.json 拼装与 JSON 合法性。

真实 ffmpeg 用例标 ``requires_ffmpeg``；纯函数用例不依赖外部数据。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_ffmpeg

from app.models import AnalyzeResult, DecodeResult, GainInfo, KeptSpan, RemovedSpan
from app.pipeline.export import (
    _clean_json,
    _parse_bitrate,
    build_report,
    encode_m4a,
    measure_m4a,
    speech_preserved_ratio,
    wav_peak_check,
    write_json,
)
from app.pipeline.trim import TrimPlan, edit_map, write_edit_json


# ---------------------------------------------------------------- 纯函数
def test_parse_bitrate():
    assert _parse_bitrate("96k") == 96000
    assert _parse_bitrate("96K") == 96000
    assert _parse_bitrate("1.5m") == 1_500_000
    assert _parse_bitrate("128000") == 128000
    assert _parse_bitrate("bogus") == 0


def test_clean_json_sanitizes_nan_inf():
    payload = {"a": float("nan"), "b": float("inf"), "c": -float("inf"),
               "d": [float("nan"), 1.5], "e": {"f": float("nan")}, "g": None}
    cleaned = _clean_json(payload)
    assert cleaned["a"] is None and cleaned["b"] is None and cleaned["c"] is None
    assert cleaned["d"] == [None, 1.5]
    assert cleaned["e"] == {"f": None}
    # 必须是合法 JSON（不允许 NaN/Infinity 字面量）
    text = json.dumps(cleaned, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_write_json_is_valid_json(tmp_path: Path):
    path = tmp_path / "out" / "report.json"
    write_json(path, {"x": float("nan"), "y": 1.0, "z": [float("inf")]})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"x": None, "y": 1.0, "z": [None]}


def _synthetic_analyze() -> AnalyzeResult:
    return AnalyzeResult(
        ok=True, job_id="t", duration_s=10.0, sample_rate=16000,
        vad={"engine": "silero-v6", "speech_ratio": 0.3, "n_segments": 2},
        levels={"asl_in_dbfs": -25.0, "asl_p56_dbfs": -24.0, "lufs_in": -26.0,
                "true_peak_in_dbtp": -3.0, "noise_floor_dbfs": -50.0,
                "snr_db": 25.0, "window5m_spread_in_db": 2.0},
        trim={"head_s": 0.0, "tail_s": 10.0, "head_method": "window_asr", "flags": []},
        segments=[[1.0, 3.0], [5.0, 6.0]],
    )


def _synthetic_plan() -> TrimPlan:
    return TrimPlan(
        kept=[KeptSpan(0.0, 4.0, 0.0, 4.0), KeptSpan(5.0, 10.0, 4.0, 9.0)],
        removed=[RemovedSpan(4.0, 5.0, "internal_silence")],
        clean_duration_s=9.0, orig_duration_s=10.0, head_s=0.0, tail_s=10.0,
        internal_cut_s=1.0,
    )


def _synthetic_decoded() -> DecodeResult:
    return DecodeResult(ok=True, wav16k="a.wav", wav44k="b.wav", duration_s=10.0,
                        sample_rate=16000, bit_rate=59053, repaired=False)


def _fake_loudness():
    gain = GainInfo(type="static+slow_adaptive", gain_db=-5.0, min_db=-8.0,
                    max_db=-2.0, max_rate_db_s=0.8, limiter_triggers=3)
    return SimpleNamespace(
        asl_in_dbfs=-15.0, asl_out_dbfs=-20.0, asl_dev_db=0.0, gain_db=-5.0,
        gain=gain, v2={"clamp_db": 3.0, "n_windows_used": 10},
        limiter={"ceiling_dbtp": -3.5, "iterations": 2, "trigger_ratio": 0.01},
        iterations=2, tp_wav_dbtp=-3.4, lufs_out=-20.5,
        window5m_spread_out_db=0.9, timings={"total_s": 12.0},
    )


def test_speech_preserved_ratio():
    analyze = _synthetic_analyze()
    plan = _synthetic_plan()
    # VAD 段 [1,3]、[5,6] 全在 kept 内 → 100%
    assert speech_preserved_ratio(analyze, plan) == pytest.approx(1.0)
    # 人为切掉一段语音 → 比例下降
    bad = TrimPlan(
        kept=[KeptSpan(0.0, 2.5, 0.0, 2.5), KeptSpan(5.0, 10.0, 2.5, 7.5)],
        removed=[RemovedSpan(2.5, 5.0, "internal_silence")],
        clean_duration_s=7.5, orig_duration_s=10.0, head_s=0.0, tail_s=10.0,
        internal_cut_s=2.5,
    )
    ratio = speech_preserved_ratio(analyze, bad)
    assert 0.8 < ratio < 1.0  # [1,3] 被切掉 0.5 s，[5,6] 完整


def test_build_report_contract_and_json_safe():
    report = build_report(
        job_id="t", source="data/x.m4a", decoded=_synthetic_decoded(),
        analyze=_synthetic_analyze(), plan=_synthetic_plan(),
        loudness_result=_fake_loudness(),
        m4a_info={"codec_name": "aac", "sample_rate": 44100, "channels": 1,
                  "format_bit_rate": 99000, "size_bytes": 123, "duration_s": 9.0,
                  "target_bit_rate": "96k"},
        m4a_meas={"lufs": -20.5, "lra": 8.0, "tp_dbtp": -2.0, "asl_out_dbfs": -20.0,
                  "max_abs": 0.79, "n_over_threshold": 0,
                  "window5m_spread_out_db": 0.9,
                  "decoded_duration_s": 9.0,
                  "timings": {"ebur128_s": 1.2, "peak_s": 0.5, "asl_s": 0.4}},
        denoise_info={"mode": "afftdn", "filter": "highpass=f=80,afftdn=nr=12:nf=-30:tn=1",
                      "nr": 12.0, "snr_db": 25.0, "elapsed_s": 3.0},
        timings={"decode_s": 1.0, "denoise_s": 3.0, "trim_s": 1.0, "render_s": 12.0,
                 "encode_s": 2.0, "total_s": 20.0},
        asl_target_dbfs=-20.0,
    )
    # 契约 §4.1 的顶层段
    for key in ("id", "source", "duration_s", "sample_rate", "bit_rate",
                "decode", "vad", "levels", "trim", "timing"):
        assert key in report, key
    assert report["decode"] == {"repaired": False, "ok": True}
    assert report["levels"]["asl_out_dbfs"] == -20.0
    assert report["levels"]["asl_dev_db"] == 0.0
    assert report["levels"]["clipped_samples_out"] == 0
    assert report["trim"]["clean_dur_s"] == 9.0
    assert report["trim"]["speech_preserved_ratio"] == pytest.approx(1.0)
    assert report["output"]["codec"] == "aac"
    assert report["loudness"]["gain"]["max_rate_db_s"] == 0.8
    # 回读耗时子项（P3-T4 补）：不再丢掉 measure_m4a 的 ebur128/peak/asl 分项
    assert report["measure"] == {
        "decoded_duration_s": 9.0, "ebur128_s": 1.2, "peak_s": 0.5, "asl_s": 0.4,
    }
    # report 必须能无损 JSON 往返（无 NaN/Inf）
    text = json.dumps(report, ensure_ascii=False, allow_nan=False)
    assert json.loads(text)["id"] == "t"


def test_edit_json_from_plan(tmp_path: Path):
    plan = _synthetic_plan()
    path = tmp_path / "edit.json"
    write_edit_json(plan, path, gain=GainInfo(gain_db=-5.0, min_db=-8.0, max_db=-2.0,
                                              max_rate_db_s=0.8, limiter_triggers=2))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["kept"] == [[0.0, 4.0, 0.0, 4.0], [5.0, 10.0, 4.0, 9.0]]
    assert data["removed"] == [{"orig_start": 4.0, "orig_end": 5.0,
                                "reason": "internal_silence"}]
    assert data["gain"]["type"] == "static"
    assert list(data["reason_enum"]) == [
        "head_nonspeech", "tail_nonspeech", "internal_silence",
        "other_speaker_low_chars", "vad_speech_asr_empty",
    ]
    assert edit_map(plan).to_dict()["kept"] == data["kept"]


# ---------------------------------------------------------------- 真实 ffmpeg
def _write_sine(path: Path, sr: int = 44100, seconds: float = 3.0,
                amp: float = 0.5, freq: float = 440.0) -> None:
    """正弦测试音：带 50 ms 淡入，避免 AAC 编码起点突变造成的码流过冲（不是限幅问题）。"""
    import scipy.io.wavfile as wavfile

    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    fade = min(int(sr * 0.05), len(x))
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    x[:fade] *= ramp
    x[fade:] *= 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, x.astype(np.float32))


@requires_ffmpeg
def test_wav_peak_check_counts_overs(tmp_path: Path):
    import scipy.io.wavfile as wavfile

    path = tmp_path / "x.wav"
    x = np.array([0.1, -0.5, 1.0, 0.9999, -1.2], dtype=np.float32)
    wavfile.write(str(path), 44100, x)
    info = wav_peak_check(path, threshold=0.999)
    assert info["samples"] == 5
    assert info["max_abs"] == pytest.approx(1.2, abs=1e-6)
    assert info["n_over"] == 3  # 1.0 / 0.9999 / -1.2（>0.999）
    assert info["sample_rate"] == 44100


@requires_ffmpeg
def test_encode_m4a_contract_and_measure(tmp_path: Path, cfg):
    wav = tmp_path / "clean44.wav"
    _write_sine(wav, amp=0.5)
    m4a = tmp_path / "clean.m4a"
    info = encode_m4a(wav, m4a, cfg)
    assert info["codec_name"] == "aac"
    assert info["sample_rate"] == 44100
    assert info["channels"] == 1
    assert info["size_bytes"] > 0
    # 96k 目标：容器码率在容差内
    assert info["format_bit_rate"] > 0
    assert m4a.is_file()

    meas = measure_m4a(m4a, [(0.0, 2.9)], cfg, tmp_dir=tmp_path)
    assert meas["tp_dbtp"] is not None and meas["tp_dbtp"] <= -1.0
    assert meas["lufs"] is not None and meas["lufs"] < 0
    assert meas["asl_out_dbfs"] is not None
    assert meas["n_over_threshold"] == 0
    assert meas["max_abs"] < 0.6


@requires_ffmpeg
def test_render_job_missing_source_returns_failure(tmp_path: Path, cfg):
    from app.pipeline.export import render_job

    result = render_job("deadbeef_test", tmp_path / "not_exists.m4a", cfg, out_dir=tmp_path / "out")
    assert result["ok"] is False
    assert result["stage"] in ("decode", "analyze", "denoise", "trim", "loudness", "export")
    assert result["error"]


def test_render_job_measures_readback_once_in_cache_dir(tmp_path: Path, cfg, monkeypatch):
    """回归：最终 clean.m4a 不再二次回读；回读临时 wav 落 cache_dir；分项计时进 report。

    旧实现 export.py 在 ``tmp_m4a.replace()`` 后又对 clean.m4a 跑一次 ``measure_m4a``，
    且这次不计时（本机 D: 外置盘上多花 ~68 s）。这里用假阶段锁住「只回读一次」。
    """
    import app.pipeline.export as export_mod

    job_id = "deadbeef_once"
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    source = tmp_path / "src.m4a"
    source.write_bytes(b"fake")

    decoded = DecodeResult(ok=True, wav16k=str(tmp_path / "a16.wav"),
                           wav44k=str(tmp_path / "a44.wav"), duration_s=9.0,
                           sample_rate=16000, bit_rate=59053)
    analyze = AnalyzeResult(
        ok=True, duration_s=9.0, sample_rate=16000,
        vad={"engine": "silero-v6", "speech_ratio": 0.5, "n_segments": 1},
        levels={"asl_in_dbfs": -15.0, "snr_db": 20.0},
        trim={"head_s": 0.0, "tail_s": 9.0, "flags": []},
        segments=[[0.0, 9.0]],
    )
    plan = TrimPlan(kept=[KeptSpan(0.0, 9.0, 0.0, 9.0)], removed=[],
                    clean_duration_s=9.0, orig_duration_s=9.0,
                    head_s=0.0, tail_s=9.0, internal_cut_s=0.0)
    gain = GainInfo(type="static+slow_adaptive", gain_db=-5.0, min_db=-8.0, max_db=-2.0,
                    max_rate_db_s=0.8, limiter_triggers=1)
    loud = SimpleNamespace(
        asl_in_dbfs=-15.0, asl_out_dbfs=-20.0, asl_dev_db=0.0, gain_db=-5.0,
        gain=gain, v2={"clamp_db": 3.0},
        limiter={"ceiling_dbtp": -2.75, "iterations": 1},
        iterations=1, tp_wav_dbtp=-2.7, lufs_out=-20.5,
        timings={"total_s": 5.0},
    )

    calls = []

    def fake_measure(m4a, intervals, cfg_, tmp_dir=None):
        calls.append((Path(m4a), Path(tmp_dir) if tmp_dir is not None else None))
        return {"lufs": -20.5, "lra": 8.0, "tp_dbtp": -2.0, "asl_out_dbfs": -20.0,
                "max_abs": 0.79, "n_over_threshold": 0, "window5m_spread_out_db": 0.9,
                "decoded_duration_s": 9.0,
                "timings": {"ebur128_s": 0.1, "peak_s": 0.2, "asl_s": 0.3}}

    monkeypatch.setattr(export_mod, "run_decode", lambda *a, **k: decoded)
    monkeypatch.setattr(export_mod, "_load_analyze", lambda *a, **k: analyze)
    monkeypatch.setattr(export_mod, "denoise_44k", lambda *a, **k: tmp_path / "den.wav")
    monkeypatch.setattr(export_mod, "build_filter_chain", lambda *a, **k: ("highpass=f=80", None))
    monkeypatch.setattr(export_mod, "resolve_nr", lambda *a, **k: 12.0)
    monkeypatch.setattr(export_mod, "plan_trim", lambda *a, **k: plan)
    monkeypatch.setattr(export_mod, "render_clean", lambda *a, **k: None)
    monkeypatch.setattr(export_mod, "chain_clean_segments", lambda *a, **k: [(0.0, 9.0)])
    monkeypatch.setattr(export_mod.loud_mod, "render", lambda *a, **k: loud)
    monkeypatch.setattr(export_mod, "measure_m4a", fake_measure)
    monkeypatch.setattr(export_mod, "probe_audio", lambda *a, **k: {
        "codec_name": "aac", "sample_rate": 44100, "channels": 1,
        "format_bit_rate": 99000, "size_bytes": 123, "duration_s": 9.0})

    def fake_encode(wav, m4a, cfg_):
        Path(m4a).write_bytes(b"fake-m4a")
        return {"elapsed_s": 0.1}

    monkeypatch.setattr(export_mod, "encode_m4a", fake_encode)

    result = export_mod.render_job(job_id, source, cfg, out_dir=out_dir, cache_dir=cache_dir)

    assert result["ok"] is True, result.get("error")
    assert len(calls) == 1  # 旧实现是 2 次（循环内 + 替换后的最终回读）
    assert calls[0][1] == cache_dir / f"{job_id}.measure"  # 临时 wav 不落输出盘
    assert not calls[0][1].exists()  # per-job 临时目录收尾清理
    assert (out_dir / job_id / "clean.m4a").is_file()
    report = result["report"]
    assert report["measure"] == {"decoded_duration_s": 9.0, "ebur128_s": 0.1,
                                 "peak_s": 0.2, "asl_s": 0.3}
    assert "write_s" in report["timing"]
    assert "write_s" in result["timings"]
