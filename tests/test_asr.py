# -*- coding: utf-8 -*-
"""P4-T1/T2 单测：SenseVoice 封装 + 分段转写 + clean/orig 双时间轴。

分三层：
- **纯函数**（默认跑，不依赖模型/ffmpeg）：文本清洗、字符统计、长段切分、切片、段表构建；
- **编排**（默认跑，用假转写器 + ffmpeg 解码一个 2 s 合成 clean 音轨）：``decode_clean16k``
  与 ``transcribe_job`` 的产物/报告字段/幂等/失败回退；
- **P4-T1 基准**（默认跑，跳过条件：模型或基准文件缺失）：60 s 样本转写与
  ``work/bench/asr_sv_t4.json`` 的字符相似度 >0.98，二次 ``get_transcriber`` <0.1 s。
"""

from __future__ import annotations

import difflib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from app.models import AnalyzeResult, KeptSpan
from app.pipeline import asr as asr_mod
from app.pipeline.asr import (
    AsrError,
    AsrUnavailable,
    Transcriber,
    build_segment_table,
    clean_text,
    count_speech_chars,
    count_speech_chars_cleaned,
    decode_clean16k,
    get_transcriber,
    is_available,
    reset_transcriber_cache,
    slice_audio,
    split_long_segment,
    strip_punct,
    transcribe_job,
)

from conftest import ROOT, requires_ffmpeg

#: P4-T1 基准（P2 阶段提前落地时实测）
SAMPLE_WAV = ROOT / "work" / "samples" / "mono_600_60.wav"
BENCH_JSON = ROOT / "work" / "bench" / "asr_sv_t4.json"


# ================================================================ 文本清洗 / 统计
def test_clean_text_strips_tags_emoji_and_spaces():
    assert clean_text("<|zh|><|NEUTRAL|>你好，世界<|Speech|>") == "你好，世界"
    assert clean_text("你 好\u3000世 界\n") == "你好世界"
    assert clean_text("你好😀世界👍") == "你好世界"
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_count_speech_chars_excludes_punctuation():
    assert count_speech_chars("你好，世界！abc123") == 10  # 4 汉字 + 3 字母 + 3 数字
    assert count_speech_chars("，。！？ …—") == 0
    assert count_speech_chars("") == 0
    # 清洗后再统计：标签/表情不计
    assert count_speech_chars_cleaned("<|zh|>你好😀，世界") == 4


def test_strip_punct():
    assert strip_punct("你好，世界！a b") == "你好世界ab"
    assert strip_punct("《形而上学》") == "形而上学"


# ================================================================ 分段工具
def test_split_long_segment_short_unchanged():
    assert split_long_segment(1.0, 12.0, 30.0) == [(1.0, 12.0)]
    assert split_long_segment(0.0, 30.0, 30.0) == [(0.0, 30.0)]
    # max_s<=0 视为不切
    assert split_long_segment(0.0, 100.0, 0.0) == [(0.0, 100.0)]


def test_split_long_segment_even_parts():
    parts = split_long_segment(0.0, 65.0, 30.0)
    assert len(parts) == 3
    assert parts[0][0] == 0.0 and parts[-1][1] == 65.0
    assert all(0.0 < b - a <= 30.0 + 1e-9 for a, b in parts)
    # 首尾相接、总长守恒
    for (a1, b1), (a2, _b2) in zip(parts, parts[1:]):
        assert a2 == pytest.approx(b1, abs=1e-6)
    assert sum(b - a for a, b in parts) == pytest.approx(65.0, abs=1e-3)


def test_slice_audio_sample_boundaries():
    x = (np.arange(32000, dtype=np.float32) / 32000.0)
    clip = slice_audio(x, 0.5, 1.0)
    assert clip.dtype == np.float32
    assert clip.flags["C_CONTIGUOUS"]
    assert len(clip) == 8000
    assert clip[0] == pytest.approx(0.25)
    assert clip[-1] == pytest.approx(15999 / 32000)

    # 左越界 clamp 到 0
    left = slice_audio(x, -0.5, 0.1)
    assert len(left) == 1600 and left[0] == pytest.approx(0.0)
    # 完全越界 → 空数组
    assert slice_audio(x, 2.5, 3.0).size == 0
    assert slice_audio(x, 1.5, 1.5).size == 0


# ================================================================ 模型缺失异常
def test_transcriber_missing_model_raises(tmp_path: Path):
    with pytest.raises(AsrUnavailable, match="模型文件不存在"):
        Transcriber(tmp_path / "model.int8.onnx", tmp_path / "tokens.txt")


def test_transcriber_missing_tokens_raises(tmp_path: Path):
    model = tmp_path / "model.int8.onnx"
    model.write_bytes(b"not-a-real-model")
    with pytest.raises(AsrUnavailable, match="tokens"):
        Transcriber(model, tmp_path / "tokens.txt")


def test_is_available_false_when_model_dir_missing(cfg):
    cfg_bad = cfg.replace({"paths.models.sensevoice_dir": str(Path(cfg.root) / "no_such_model_dir")})
    assert is_available(cfg_bad) is False
    with pytest.raises(AsrUnavailable):
        asr_mod.create_transcriber(cfg_bad)


# ================================================================ 段表构建（build_segment_table）
def _ar(segments, duration: float = 40.0) -> AnalyzeResult:
    return AnalyzeResult(job_id="t", duration_s=duration, segments=[[float(a), float(b)] for a, b in segments])


def test_build_segment_table_identity_kept():
    ar = _ar([[0.0, 2.0], [2.5, 4.0], [5.0, 40.0]])
    kept = [[0.0, 40.0, 0.0, 40.0]]
    assert build_segment_table(ar, kept, None) == [
        (0.0, 2.0), (2.5, 4.0), (5.0, 22.5), (22.5, 40.0),
    ]


def test_build_segment_table_filters_short_and_splits_long():
    ar = _ar([[0.0, 0.4], [1.0, 2.0], [3.0, 38.0]], duration=38.0)
    kept = [[0.0, 38.0, 0.0, 38.0]]
    table = build_segment_table(ar, kept, None)
    assert (0.0, 0.4) not in table  # 碎片被过滤
    assert (1.0, 2.0) in table
    long_parts = [row for row in table if row[0] >= 3.0]
    assert len(long_parts) == 2  # 35 s → 2 × 17.5 s
    assert all(b - a <= 30.0 + 1e-9 for a, b in long_parts)


def test_build_segment_table_maps_orig_across_cut():
    """跨内部切口的 orig 段在 clean 轴拆成两段（不重叠、首尾相接）。"""
    ar = _ar([[1.0, 3.0]], duration=5.0)
    kept = [[0.0, 2.0, 0.0, 2.0], [2.5, 5.0, 2.0, 4.5]]
    assert build_segment_table(ar, kept) == [(1.0, 2.0), (2.0, 2.5)]


def test_build_segment_table_drops_fully_cut_segment():
    ar = _ar([[2.0, 2.2], [3.0, 4.0]], duration=5.0)
    # orig 2.0–2.5 被切（内部静音），2.0–2.2 整段落在删除区 → 无 clean 段
    kept = [[0.0, 2.0, 0.0, 2.0], [2.5, 5.0, 2.0, 4.5]]
    assert build_segment_table(ar, kept) == [(2.5, 3.5)]


def test_build_segment_table_sorted_and_non_overlapping():
    ar = _ar([[5.0, 6.0], [0.0, 1.0], [2.0, 3.0]])
    kept = [[0.0, 6.0, 0.0, 6.0]]
    table = build_segment_table(ar, kept)
    assert table == [(0.0, 1.0), (2.0, 3.0), (5.0, 6.0)]
    for (_a1, b1), (a2, _b2) in zip(table, table[1:]):
        assert a2 >= b1


def test_build_segment_table_merge_gap_optional():
    ar = _ar([[0.0, 1.0], [1.1, 2.0], [3.0, 4.0]], duration=4.0)
    kept = [[0.0, 4.0, 0.0, 4.0]]
    # 默认不合并
    assert build_segment_table(ar, kept) == [(0.0, 1.0), (1.1, 2.0), (3.0, 4.0)]
    # gap<=0.2 → 前两段合并
    assert build_segment_table(ar, kept, merge_gap_s=0.2) == [(0.0, 2.0), (3.0, 4.0)]
    # gap<=1.5 → 全合并成一段（总长 4 s <= 30 s）
    assert build_segment_table(ar, kept, merge_gap_s=1.5) == [(0.0, 4.0)]
    # 合并后超过 max_s → 不合并
    ar2 = _ar([[0.0, 15.0], [15.2, 30.1]], duration=30.1)
    table = build_segment_table(ar2, [[0.0, 30.1, 0.0, 30.1]], merge_gap_s=0.5)
    assert len(table) == 2


def test_build_segment_table_accepts_dict_and_keptspan():
    payload = {"ok": True, "wav": "x", "result": {"duration_s": 2.0, "segments": [[0.0, 1.0]]}}
    assert build_segment_table(payload, [KeptSpan(0.0, 2.0, 0.0, 2.0)]) == [(0.0, 1.0)]
    edit = {"kept": [[0.0, 2.0, 0.0, 2.0]], "removed": []}
    assert build_segment_table(payload, edit) == [(0.0, 1.0)]
    # 空段表合法
    assert build_segment_table({"result": {"segments": []}}, [[0.0, 1.0, 0.0, 1.0]]) == []


def test_build_segment_table_invalid_inputs():
    ar = _ar([[0.0, 1.0]])
    with pytest.raises(AsrError, match="kept 为空"):
        build_segment_table(ar, [])
    with pytest.raises(AsrError, match="四元组格式非法"):
        build_segment_table(ar, [[0.0, 1.0, 0.0]])
    with pytest.raises(AsrError, match="重叠"):
        build_segment_table(ar, [[0.0, 2.0, 0.0, 2.0], [1.0, 3.0, 2.0, 4.0]])
    with pytest.raises(AsrError, match="不连续"):
        build_segment_table(ar, [[0.0, 2.0, 0.0, 2.0], [3.0, 4.0, 3.0, 4.0]])
    with pytest.raises(AsrError, match="segments"):
        build_segment_table({"ok": True}, [[0.0, 1.0, 0.0, 1.0]])


def test_build_segment_table_respects_cfg_limits(cfg):
    cfg2 = cfg.replace({"asr.segment_min_s": 0.2, "asr.segment_max_s": 10.0})
    ar = _ar([[0.0, 0.3], [1.0, 26.0]], duration=26.0)
    table = build_segment_table(ar, [[0.0, 26.0, 0.0, 26.0]], cfg2)
    assert (0.0, 0.3) in table  # 0.3 s >= 0.2 s
    long_parts = [row for row in table if row[0] >= 1.0]
    assert len(long_parts) == 3  # 25 s / 10 s → ceil=3
    assert all(b - a <= 10.0 + 1e-9 for a, b in long_parts)


# ================================================================ 测试用夹具：合成 job
class FakeTranscriber:
    """假 SenseVoice：按调用顺序返回预置文本，并记录每段（时长, 秒数）。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def transcribe_raw(self, clip, sr=16000):
        self.calls.append((int(len(clip)), float(len(clip)) / float(sr)))
        idx = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[idx]


def _write_fake_clean_m4a(path: Path, seconds: float = 2.0, sr: int = 16000) -> None:
    """写一个 16k 正弦 WAV 内容但命名为 clean.m4a（ffmpeg 按内容探测，省去 AAC 编码耗时）。"""
    import scipy.io.wavfile as wavfile

    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    x = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    wavfile.write(str(tmp), sr, x)
    tmp.replace(path)


def _prepare_job(
    cfg,
    job_id: str,
    segments,
    *,
    clean_s: float = 2.0,
    kept=None,
    report_clean_s: float | None = None,
):
    """在 cfg 的 out/cache 目录里造一个已 render 完的 job（clean.m4a/edit/report/analyze 缓存）。"""
    out_root = cfg.path("out_dir")
    cache_root = cfg.path("cache_dir")
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    job_out = out_root / job_id
    job_out.mkdir(parents=True, exist_ok=True)
    _write_fake_clean_m4a(job_out / "clean.m4a", clean_s)
    if kept is None:
        kept = [[0.0, clean_s, 0.0, clean_s]]
    (job_out / "edit.json").write_text(
        json.dumps({"kept": kept, "removed": [], "reason_enum": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    report = {
        "id": job_id,
        "source": f"data/{job_id}.m4a",
        "duration_s": clean_s + 10.0,
        "decode": {"ok": True, "repaired": False},
        "vad": {"n_segments": len(segments)},
        "levels": {"asl_in_dbfs": -20.0},
        "trim": {"head_s": 0.0, "clean_dur_s": clean_s if report_clean_s is None else report_clean_s},
        "timing": {"render_s": 1.5},
        "ok": True,
    }
    (job_out / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    ar = AnalyzeResult(job_id=job_id, duration_s=clean_s, segments=[[float(a), float(b)] for a, b in segments])
    cache = {
        "ok": True, "wav": str(cache_root / f"{job_id}.16k.wav"), "wav_bytes": 0,
        "params": {}, "result": ar.to_dict(),
    }
    (cache_root / f"{job_id}.analyze.json").write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8"
    )
    return job_out


# ================================================================ decode_clean16k
@requires_ffmpeg
def test_decode_clean16k_creates_independent_cache(cfg):
    job_out = _prepare_job(cfg, "dc_job", [[0.0, 1.0]], clean_s=2.0)
    path = decode_clean16k("dc_job", cfg)
    assert path == cfg.path("cache_dir") / "dc_job.clean16k.wav"
    assert path.is_file()
    from app.pipeline.decode import wav_info

    info = wav_info(path)
    assert info.sample_rate == 16000 and info.channels == 1
    assert info.n_frames / info.sample_rate == pytest.approx(2.0, abs=0.05)
    # 只新增 clean16k，不动 16k/analyze/decode 缓存
    names = {p.name for p in cfg.path("cache_dir").iterdir()}
    assert "dc_job.clean16k.wav" in names
    assert (job_out / "clean.m4a").is_file()


@requires_ffmpeg
def test_decode_clean16k_duration_mismatch_raises(cfg):
    _prepare_job(cfg, "dc_bad", [[0.0, 1.0]], clean_s=2.0, report_clean_s=2.5)
    with pytest.raises(AsrError, match="偏差"):
        decode_clean16k("dc_bad", cfg)


@requires_ffmpeg
def test_decode_clean16k_missing_clean_m4a(cfg):
    job_out = cfg.path("out_dir") / "dc_missing"
    job_out.mkdir(parents=True, exist_ok=True)
    with pytest.raises(AsrError, match="clean.m4a"):
        decode_clean16k("dc_missing", cfg)


# ================================================================ transcribe_job 编排
@requires_ffmpeg
def test_transcribe_job_end_to_end(cfg, monkeypatch):
    job_id = "tj_job"
    _prepare_job(cfg, job_id, [[0.1, 0.4], [0.5, 1.2], [1.3, 1.9]], clean_s=2.0)
    fake = FakeTranscriber([
        ("你好，世界", "<|zh|>你好，世界<|NEUTRAL|>"),
        ("", "<|zh|>"),
    ])
    monkeypatch.setattr(asr_mod, "get_transcriber", lambda _cfg: fake)

    stages = []
    result = transcribe_job(job_id, cfg, on_stage=lambda s, st, m: stages.append((s, st, dict(m))))
    assert result["ok"] is True and result["skipped"] is False
    assert result["stage"] == "asr" and result["error"] == ""
    assert result["source"] == f"data/{job_id}.m4a"

    # 段表：0.1–0.4 太短被过滤；只有 2 段进 ASR，其中第 2 段空文本不写入
    assert len(fake.calls) == 2
    assert fake.calls[0][0] == int(round(0.7 * 16000))
    transcript_path = Path(result["transcript"])
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert isinstance(transcript, list) and len(transcript) == 1
    seg = transcript[0]
    assert seg["start"] == 0.5 and seg["end"] == 1.2
    assert seg["orig_start"] == 0.5 and seg["orig_end"] == 1.2
    assert seg["speaker"] == "主讲" and seg["flags"] == []
    assert seg["text"] == "你好，世界"
    assert seg["text_raw"] == "<|zh|>你好，世界<|NEUTRAL|>"
    # 按 start 排序
    starts = [s["start"] for s in transcript]
    assert starts == sorted(starts)

    # report：新增 asr/speakers/timing.asr_s，既有字段全部保留
    report = result["report"]
    assert report["asr"] == result["asr"]
    assert report["asr"]["n_segments"] == 2  # 段表总数（含 n_empty）
    assert report["asr"]["chars"] == 4 and report["asr"]["n_empty"] == 1
    assert report["asr"]["engine"] == "sensevoice" and report["asr"]["model"] == "int8"
    assert report["asr"]["rtf"] >= 0.0 and report["asr"]["elapsed_s"] > 0.0
    assert report["speakers"] == {"n": 1, "method": "none"}
    assert report["timing"]["render_s"] == 1.5
    assert report["timing"]["asr_s"] > 0.0
    assert report["decode"] == {"ok": True, "repaired": False}
    assert report["vad"]["n_segments"] == 3
    assert report["id"] == job_id and report["ok"] is True
    # 落盘的 report 与返回值一致
    on_disk = json.loads((cfg.path("out_dir") / job_id / "report.json").read_text(encoding="utf-8"))
    assert on_disk["asr"] == report["asr"]
    assert ("asr", "done") in [(s, st) for s, st, _m in stages]

    # 幂等：已有 transcript + report.asr → 默认跳过，不再调 ASR
    n_calls = len(fake.calls)
    result2 = transcribe_job(job_id, cfg)
    assert result2["ok"] is True and result2["skipped"] is True
    assert result2["asr"] == report["asr"]
    assert len(fake.calls) == n_calls

    # force 重做
    fake.outputs = [("重做文本", "<|zh|>重做文本")]
    result3 = transcribe_job(job_id, cfg, force=True)
    assert result3["ok"] is True and result3["skipped"] is False
    assert len(fake.calls) > n_calls
    transcript3 = json.loads(Path(result3["transcript"]).read_text(encoding="utf-8"))
    assert transcript3[0]["text"] == "重做文本"


@requires_ffmpeg
def test_transcribe_job_merge_gap_and_orig_mapping(cfg, monkeypatch):
    """kept 有内部切：clean/orig 时间轴各自正确，merge_gap_s 生效。"""
    job_id = "tj_cut"
    # orig 0–2 与 2.5–4.5 保留，orig 2–2.5 被切；clean 轴 0–4
    kept = [[0.0, 2.0, 0.0, 2.0], [2.5, 4.5, 2.0, 4.0]]
    _prepare_job(
        cfg, job_id,
        [[0.2, 0.8], [0.9, 1.8], [2.1, 2.4]],
        clean_s=4.0, kept=kept,
    )
    fake = FakeTranscriber([("甲", "甲"), ("乙", "乙")])
    monkeypatch.setattr(asr_mod, "get_transcriber", lambda _cfg: fake)
    result = transcribe_job(job_id, cfg, merge_gap_s=0.2)
    assert result["ok"] is True
    transcript = json.loads(Path(result["transcript"]).read_text(encoding="utf-8"))
    # orig 0.2–0.8 / 0.9–1.8 在 clean 轴原样；gap 0.1 <=0.2 → 合并成一段
    assert len(transcript) == 1
    seg = transcript[0]
    assert seg["start"] == 0.2 and seg["end"] == 1.8
    assert seg["orig_start"] == 0.2 and seg["orig_end"] == 1.8
    # orig 2.1–2.4 落在切口内 → 整段删除，不会出现在 transcript
    assert all(s["start"] < 2.0 for s in transcript)


@requires_ffmpeg
def test_transcribe_job_path_overrides(cfg, monkeypatch, tmp_path: Path):
    out2, cache2 = tmp_path / "custom_out", tmp_path / "custom_cache"
    cfg2 = cfg.replace({"paths.out_dir": str(out2), "paths.cache_dir": str(cache2)})
    _prepare_job(cfg2, "tj_ov", [[0.0, 1.0]], clean_s=1.5)
    fake = FakeTranscriber([("覆盖路径", "覆盖路径")])
    monkeypatch.setattr(asr_mod, "get_transcriber", lambda _cfg: fake)
    result = transcribe_job("tj_ov", cfg, out_dir=out2, cache_dir=cache2)
    assert result["ok"] is True, result["error"]
    assert (out2 / "tj_ov" / "transcript.json").is_file()
    assert (cache2 / "tj_ov.clean16k.wav").is_file()


@requires_ffmpeg
def test_transcribe_job_failure_contract(cfg, monkeypatch):
    """失败不抛异常：返回 ok=False/stage=asr/error，并回调 failed。"""
    stages = []
    result = transcribe_job("no_such_job", cfg, on_stage=lambda s, st, m: stages.append((s, st)))
    assert result["ok"] is False and result["stage"] == "asr" and result["error"]
    assert ("asr", "failed") in stages

    # clean16k 时长与 report 不符 → 失败（不是半成品 transcript）
    _prepare_job(cfg, "tj_bad", [[0.0, 1.0]], clean_s=2.0, report_clean_s=2.6)
    fake = FakeTranscriber([("x", "x")])
    monkeypatch.setattr(asr_mod, "get_transcriber", lambda _cfg: fake)
    result2 = transcribe_job("tj_bad", cfg)
    assert result2["ok"] is False and result2["stage"] == "asr"
    assert not (cfg.path("out_dir") / "tj_bad" / "transcript.json").exists()


@requires_ffmpeg
def test_transcribe_job_missing_analyze_cache(cfg, monkeypatch):
    job_out = cfg.path("out_dir") / "tj_noan"
    job_out.mkdir(parents=True, exist_ok=True)
    _write_fake_clean_m4a(job_out / "clean.m4a", 1.0)
    (job_out / "edit.json").write_text(
        json.dumps({"kept": [[0.0, 1.0, 0.0, 1.0]]}, ensure_ascii=False), encoding="utf-8"
    )
    (job_out / "report.json").write_text(
        json.dumps({"id": "tj_noan", "trim": {"clean_dur_s": 1.0}, "ok": True}), encoding="utf-8"
    )
    fake = FakeTranscriber([("x", "x")])
    monkeypatch.setattr(asr_mod, "get_transcriber", lambda _cfg: fake)
    result = transcribe_job("tj_noan", cfg)
    assert result["ok"] is False and "analyze" in result["error"]


# ================================================================ P4-T1 基准验收
def _bench_cfg():
    from app.config import load_config

    return load_config()


def test_p4_t1_text_similarity_and_singleton():
    """60 s 样本转写与 ``asr_sv_t4.json`` 字符相似度 >0.98；二次取单例 <0.1 s。"""
    if not SAMPLE_WAV.is_file() or not BENCH_JSON.is_file():
        pytest.skip("缺少 P4-T1 基准（work/samples/mono_600_60.wav 或 work/bench/asr_sv_t4.json）")
    cfg = _bench_cfg()
    if not is_available(cfg):
        pytest.skip("SenseVoice 模型文件不在 paths.models.sensevoice_dir")

    from app.pipeline.decode import read_wav_float32

    sr, audio = read_wav_float32(SAMPLE_WAV)
    assert sr == 16000 and len(audio) == 60 * 16000
    bench = json.loads(BENCH_JSON.read_text(encoding="utf-8"))

    reset_transcriber_cache()
    transcriber = get_transcriber(cfg)
    text = transcriber.transcribe(audio)
    similarity = difflib.SequenceMatcher(None, text, bench["text"]).ratio()
    assert similarity > 0.98, f"与基准相似度 {similarity:.4f}（要求 >0.98）\n实测：{text[:120]}"

    t0 = time.perf_counter()
    again = get_transcriber(cfg)
    dt = time.perf_counter() - t0
    assert again is transcriber
    assert dt < 0.1, f"二次 get_transcriber 耗时 {dt:.4f}s（要求 <0.1s）"
