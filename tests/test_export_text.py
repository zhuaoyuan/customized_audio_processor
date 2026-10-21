# -*- coding: utf-8 -*-
"""P5-T4 文本导出单测：md/srt/txt 精确格式化、契约字段、report 读-改-写与边界。

全部用 fixture transcript.json，不依赖真实 ASR / 真实音频；只有「srt 可被 ffprobe 识别」
一项需要 ffmpeg（缺工具时 skip）。约定：
- md 每行 ``[mm:ss (orig mm:ss)] 发言人：text``，逐段一行，**不合并**；
- srt 用 clean 轴 ``HH:MM:SS,mmm``，正文带发言人前缀，``end > start``（start==end 时 +0.3 s）；
- txt 逐段一行 ``[mm:ss] 发言人：text``；
- 空文本 / 纯标点段三个产物都跳过，srt 序号在过滤后连续；
- export_text 只读 transcript.json，report.json 只读-改-写。
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from conftest import requires_ffmpeg

from app.models import TranscriptSegment
from app.pipeline.export import (
    export_text,
    format_mmss,
    format_srt_time,
    transcript_to_md,
    transcript_to_srt,
    transcript_to_txt,
)


# ---------------------------------------------------------------- fixture / 工具
def _segments():
    """3 段 fixture（clean / orig 双时间轴；时间与 PoC 前几段同量级）。"""
    return [
        {"start": 0.0, "end": 1.5, "orig_start": 7.8, "orig_end": 9.3,
         "speaker": "主讲", "text_raw": "我去做饭。", "text": "我去做饭。", "flags": []},
        {"start": 23.4, "end": 27.0, "orig_start": 30.1, "orig_end": 33.7,
         "speaker": "主讲", "text_raw": "俩就个。", "text": "俩就个。", "flags": []},
        {"start": 63.0, "end": 65.0, "orig_start": 555.0, "orig_end": 557.0,
         "speaker": "主讲", "text_raw": "嗯。", "text": "嗯。", "flags": []},
    ]


_EXPECTED_MD = (
    "# 24_形而上学 · 句级时间戳转写\n"
    "\n"
    "> 时间轴：clean 音轨；括号内为原始音轨时间。\n"
    "\n"
    "[00:00 (orig 00:07)] 主讲：我去做饭。\n"
    "[00:23 (orig 00:30)] 主讲：俩就个。\n"
    "[01:03 (orig 09:15)] 主讲：嗯。\n"
)

_EXPECTED_SRT = (
    "1\n"
    "00:00:00,000 --> 00:00:01,500\n"
    "主讲：我去做饭。\n"
    "\n"
    "2\n"
    "00:00:23,400 --> 00:00:27,000\n"
    "主讲：俩就个。\n"
    "\n"
    "3\n"
    "00:01:03,000 --> 00:01:05,000\n"
    "主讲：嗯。\n"
)

_EXPECTED_TXT = (
    "[00:00] 主讲：我去做饭。\n"
    "[00:23] 主讲：俩就个。\n"
    "[01:03] 主讲：嗯。\n"
)

_MD_TIME_RE = re.compile(r"^\[([0-9:]+) \(orig ([0-9:]+)\)\] (.+)$")


def _parse_mmss(text: str) -> int:
    """``mm:ss`` 或 ``h:mm:ss`` → 总秒数。"""
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise AssertionError(f"非法 mm:ss 时间：{text!r}")


def _parse_srt_seconds(text: str) -> float:
    hms, millis = text.split(",")
    hours, minutes, secs = (int(p) for p in hms.split(":"))
    return hours * 3600 + minutes * 60 + secs + int(millis) / 1000.0


def _parse_srt(text: str):
    """SRT → [{index, start, end, body}]。"""
    blocks = [b for b in text.strip().split("\n\n") if b.strip()]
    out = []
    for block in blocks:
        lines = block.split("\n")
        assert len(lines) == 3, f"SRT 块应为 3 行，实际 {lines!r}"
        start_text, end_text = lines[1].split(" --> ")
        out.append({
            "index": int(lines[0]),
            "start": _parse_srt_seconds(start_text),
            "end": _parse_srt_seconds(end_text),
            "body": lines[2],
        })
    return out


def _parse_md_body(md: str):
    """md 正文行 → [{clean_s, orig_s, body}]（跳过文件头）。"""
    out = []
    for line in md.splitlines():
        match = _MD_TIME_RE.match(line)
        if match:
            out.append({
                "clean_s": _parse_mmss(match.group(1)),
                "orig_s": _parse_mmss(match.group(2)),
                "body": match.group(3),
            })
    return out


def _job_dir(cfg, job_id: str) -> Path:
    return cfg.path("out_dir") / job_id


def _write_transcript(cfg, job_id: str, segments, report=None) -> Path:
    job = _job_dir(cfg, job_id)
    job.mkdir(parents=True, exist_ok=True)
    (job / "transcript.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report is not None:
        (job / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return job


# ---------------------------------------------------------------- 纯函数：时间格式
def test_format_mmss():
    assert format_mmss(0) == "00:00"
    assert format_mmss(0.99) == "00:00"       # 向下取整
    assert format_mmss(7.8) == "00:07"
    assert format_mmss(59.9) == "00:59"
    assert format_mmss(60) == "01:00"
    assert format_mmss(3599.9) == "59:59"
    assert format_mmss(3600) == "1:00:00"     # ≥1 h 用 h:mm:ss（不继续累加分钟）
    assert format_mmss(3725.4) == "1:02:05"
    assert format_mmss(7812.4) == "2:10:12"
    assert format_mmss(-5) == "00:00"         # 脏值兜底
    assert format_mmss(float("nan")) == "00:00"
    assert format_mmss(None) == "00:00"


def test_format_srt_time():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(1.5) == "00:00:01,500"
    assert format_srt_time(61.789) == "00:01:01,789"
    assert format_srt_time(3599.9996) == "01:00:00,000"   # 四舍五入进位到小时
    assert format_srt_time(3725.4) == "01:02:05,400"
    assert format_srt_time(-1) == "00:00:00,000"
    assert format_srt_time(float("inf")) == "00:00:00,000"


# ---------------------------------------------------------------- 纯函数：三种格式
def test_md_exact_format():
    assert transcript_to_md(_segments(), title="24_形而上学") == _EXPECTED_MD
    # 标题缺省时用 source 的文件名 stem
    assert transcript_to_md(_segments(), source="data/24_形而上学.m4a").startswith(
        "# 24_形而上学 · 句级时间戳转写"
    )
    # 逐段一行、不合并（正文行数 == 段数）
    assert len(_parse_md_body(_EXPECTED_MD)) == 3


def test_srt_exact_format():
    assert transcript_to_srt(_segments()) == _EXPECTED_SRT
    blocks = _parse_srt(_EXPECTED_SRT)
    assert [b["index"] for b in blocks] == [1, 2, 3]
    assert [b["body"] for b in blocks] == ["主讲：我去做饭。", "主讲：俩就个。", "主讲：嗯。"]


def test_txt_exact_format():
    assert transcript_to_txt(_segments()) == _EXPECTED_TXT
    assert transcript_to_txt([]) == ""
    assert transcript_to_srt([]) == ""


def test_md_time_matches_srt_per_segment():
    """md 每行 clean 时间与 srt 每条 start 逐条一致（都取 start、向下取整到秒）。"""
    md_rows = _parse_md_body(transcript_to_md(_segments(), title="t"))
    srt_rows = _parse_srt(transcript_to_srt(_segments()))
    assert len(md_rows) == len(srt_rows)
    for md_row, srt_row in zip(md_rows, srt_rows):
        assert md_row["clean_s"] == int(srt_row["start"])   # 同一 clean 轴起点
        assert md_row["body"] == srt_row["body"]
    # md 括号内是 orig 轴（来自 orig_start），与 clean 轴不同
    assert md_rows[2]["orig_s"] == 555
    assert md_rows[2]["clean_s"] == 63


def test_srt_invariants():
    blocks = _parse_srt(transcript_to_srt(_segments()))
    starts = [b["start"] for b in blocks]
    assert starts == sorted(starts)                 # 时间单调
    assert all(b["end"] > b["start"] for b in blocks)
    assert [b["index"] for b in blocks] == list(range(1, len(blocks) + 1))


def test_srt_start_equals_end_gets_min_duration():
    seg = [{"start": 5.0, "end": 5.0, "orig_start": 5.0, "orig_end": 5.0,
            "speaker": "主讲", "text": "好。", "flags": []}]
    srt = transcript_to_srt(seg)
    blocks = _parse_srt(srt)
    assert len(blocks) == 1
    assert blocks[0]["start"] == pytest.approx(5.0)
    assert blocks[0]["end"] == pytest.approx(5.3)   # max(end, start+0.3)
    assert "00:00:05,000 --> 00:00:05,300" in srt


def test_over_one_hour_uses_h_mm_ss():
    seg = [{"start": 3725.4, "end": 3728.9, "orig_start": 3730.0, "orig_end": 3733.5,
            "speaker": "主讲", "text": "好。", "flags": []}]
    md = transcript_to_md(seg, title="长讲座")
    assert "[1:02:05 (orig 1:02:10)] 主讲：好。" in md
    srt = transcript_to_srt(seg)
    assert "01:02:05,400 --> 01:02:08,900" in srt
    # 纯格式函数也允许 mm:ss 累加语义的输入（这里确认统一为 h:mm:ss）
    assert format_mmss(3600 + 125.4) == "1:02:05"


def test_empty_and_punct_segments_skipped_in_formatters():
    segs = [
        {"start": 0.0, "end": 1.0, "orig_start": 0.0, "orig_end": 1.0,
         "speaker": "主讲", "text": "。", "flags": []},
        {"start": 1.0, "end": 2.0, "orig_start": 1.0, "orig_end": 2.0,
         "speaker": "主讲", "text": "   ", "flags": []},
        {"start": 2.0, "end": 3.0, "orig_start": 2.0, "orig_end": 3.0,
         "speaker": "主讲", "text": "", "flags": []},
        {"start": 3.0, "end": 4.0, "orig_start": 3.0, "orig_end": 4.0,
         "speaker": "主讲", "text": "好。", "flags": []},
    ]
    assert len(_parse_srt(transcript_to_srt(segs))) == 1
    assert transcript_to_srt(segs).startswith("1\n00:00:03,000")
    assert len(_parse_md_body(transcript_to_md(segs))) == 1
    assert transcript_to_txt(segs) == "[00:03] 主讲：好。\n"


def test_speaker_non_main_and_flags_ignored():
    seg = [{"start": 1.0, "end": 2.0, "orig_start": 1.5, "orig_end": 2.5,
            "speaker": "其他1", "text_raw": "原始", "text": "嗯。",
            "flags": ["rejected_chunk", "uncertain_speaker"]}]
    md = transcript_to_md(seg, title="t")
    assert "其他1：嗯。" in md
    assert "rejected_chunk" not in md and "uncertain_speaker" not in md
    assert "其他1：嗯。" in transcript_to_srt(seg)
    assert "其他1：嗯。" in transcript_to_txt(seg)
    # speaker 缺失 / 空 → 兜底「主讲」
    bare = [{"start": 1.0, "end": 2.0, "orig_start": 1.5, "orig_end": 2.5,
             "text": "嗯。", "flags": []}]
    assert "主讲：嗯。" in transcript_to_md(bare, title="t")


def test_transcript_segment_dataclass_supported():
    seg = TranscriptSegment(start=1.25, end=2.75, orig_start=10.5, orig_end=12.0,
                            text="嗯。", speaker="主讲")
    assert transcript_to_txt([seg]) == "[00:01] 主讲：嗯。\n"
    blocks = _parse_srt(transcript_to_srt([seg]))
    assert blocks[0]["body"] == "主讲：嗯。"
    assert transcript_to_md([seg], title="t").endswith("[00:01 (orig 00:10)] 主讲：嗯。\n")


# ---------------------------------------------------------------- export_text 集成（tmp out）
def test_export_text_writes_files_and_report(cfg):
    report = {
        "id": "t24",
        "source": "data/24_形而上学.m4a",
        "decode": {"repaired": False, "ok": True},
        "trim": {"head_s": 479.5, "clean_dur_s": 3806.46},
        "output": {"file": "clean.m4a", "codec": "aac", "size_bytes": 123},
        "timing": {"decode_s": 1.2, "render_s": 8.0},
    }
    _write_transcript(cfg, "t24", _segments(), report)
    events = []

    result = export_text("t24", cfg, on_stage=lambda *args: events.append(args))

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["stage"] == "export"
    assert result["formats"] == ["md", "srt", "txt"]
    assert result["n_segments"] == 3 and result["n_written"] == 3
    assert result["n_skipped_empty"] == 0
    assert result["timings"]["export_text_s"] >= 0

    job = _job_dir(cfg, "t24")
    for name in ("md", "srt", "txt"):
        assert Path(result["files"][name]).is_file()
        assert Path(result["files"][name]).parent == job
    # md 标题来自 report.source 的 stem；三种格式内容与纯函数一致
    assert (job / "transcript.md").read_text(encoding="utf-8") == transcript_to_md(
        _segments(), source="data/24_形而上学.m4a"
    )
    assert (job / "transcript.srt").read_text(encoding="utf-8") == transcript_to_srt(_segments())
    assert (job / "transcript.txt").read_text(encoding="utf-8") == transcript_to_txt(_segments())

    # report 读-改-写：既有字段全部保留，新增 output.formats + timing.export_text_s
    merged = json.loads((job / "report.json").read_text(encoding="utf-8"))
    assert merged["id"] == "t24"
    assert merged["decode"] == {"repaired": False, "ok": True}
    assert merged["trim"]["clean_dur_s"] == pytest.approx(3806.46)
    assert merged["output"]["file"] == "clean.m4a"
    assert merged["output"]["codec"] == "aac"          # 不被覆盖
    assert merged["output"]["size_bytes"] == 123
    assert merged["output"]["formats"] == ["md", "srt", "txt"]
    assert merged["timing"]["decode_s"] == pytest.approx(1.2)
    assert merged["timing"]["render_s"] == pytest.approx(8.0)
    assert isinstance(merged["timing"]["export_text_s"], (int, float))
    # 回调：export/done + 指标
    assert events and events[-1][0] == "export" and events[-1][1] == "done"
    assert events[-1][2]["n_written"] == 3


def test_export_text_empty_transcript(cfg):
    job = _write_transcript(cfg, "empty", [], {"id": "empty"})
    result = export_text("empty", cfg)
    assert result["ok"] is True
    assert result["n_segments"] == 0 and result["n_written"] == 0
    md = (job / "transcript.md").read_text(encoding="utf-8")
    assert md == (
        "# 转写 · 句级时间戳转写\n"
        "\n"
        "> 时间轴：clean 音轨；括号内为原始音轨时间。\n"
    )
    assert (job / "transcript.srt").read_text(encoding="utf-8") == ""
    assert (job / "transcript.txt").read_text(encoding="utf-8") == ""
    assert json.loads((job / "report.json").read_text(encoding="utf-8"))["output"]["formats"] == [
        "md", "srt", "txt"
    ]


def test_export_text_skips_empty_and_punct_segments(cfg):
    segments = [
        {"start": 0.0, "end": 1.0, "orig_start": 0.0, "orig_end": 1.0,
         "speaker": "主讲", "text": "。", "flags": []},
        {"start": 1.0, "end": 2.0, "orig_start": 1.0, "orig_end": 2.0,
         "speaker": "主讲", "text": "，。！？", "flags": []},
        {"start": 2.0, "end": 3.0, "orig_start": 2.0, "orig_end": 3.0,
         "speaker": "主讲", "text": "", "flags": []},
        {"start": 3.0, "end": 4.0, "orig_start": 3.0, "orig_end": 4.0,
         "speaker": "主讲", "text": "嗯。", "flags": ["rejected_chunk"]},
    ]
    job = _write_transcript(cfg, "punct", segments)
    result = export_text("punct", cfg)
    assert result["ok"] is True
    assert result["n_segments"] == 4
    assert result["n_written"] == 1
    assert result["n_skipped_empty"] == 3
    srt = (job / "transcript.srt").read_text(encoding="utf-8")
    assert srt.startswith("1\n00:00:03,000")
    assert "。" not in srt.replace("嗯。", "")      # 只留下有内容的那段
    assert len(_parse_srt(srt)) == 1


def test_export_text_subset_formats_and_missing_optional(cfg):
    job = _write_transcript(cfg, "onlysrt", _segments())
    result = export_text("onlysrt", cfg, formats="srt")
    assert result["ok"] is True
    assert result["formats"] == ["srt"]
    assert (job / "transcript.srt").is_file()
    assert not (job / "transcript.md").exists()
    assert not (job / "transcript.txt").exists()
    report = json.loads((job / "report.json").read_text(encoding="utf-8"))
    assert report["output"]["formats"] == ["srt"]


def test_export_text_skip_then_force(cfg):
    _write_transcript(cfg, "skipme", _segments()[:1])
    first = export_text("skipme", cfg)
    assert first["ok"] is True and first["skipped"] is False
    md_before = (_job_dir(cfg, "skipme") / "transcript.md").read_text(encoding="utf-8")
    assert len(_parse_md_body(md_before)) == 1

    # 换 transcript 后不 force → 跳过、不覆盖
    _write_transcript(cfg, "skipme", _segments())
    second = export_text("skipme", cfg)
    assert second["ok"] is True and second["skipped"] is True
    assert (_job_dir(cfg, "skipme") / "transcript.md").read_text(encoding="utf-8") == md_before

    # force → 重写
    third = export_text("skipme", cfg, force=True)
    assert third["ok"] is True and third["skipped"] is False
    md_after = (_job_dir(cfg, "skipme") / "transcript.md").read_text(encoding="utf-8")
    assert len(_parse_md_body(md_after)) == 3


def test_export_text_json_only_is_noop(cfg):
    """CLI 允许 ``--formats json``：transcript.json 已由 P4 落盘，不写文本也不报错。"""
    job = _write_transcript(cfg, "jsononly", _segments())
    result = export_text("jsononly", cfg, formats=["json"])
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["formats"] == ["json"]
    assert result["files"] == {}
    assert not (job / "transcript.md").exists()
    assert not (job / "transcript.srt").exists()


def test_export_text_unknown_format_fails(cfg):
    _write_transcript(cfg, "badfmt", _segments())
    result = export_text("badfmt", cfg, formats=["html"])
    assert result["ok"] is False
    assert "formats" in result["error"]


def test_export_text_missing_transcript_fails(cfg):
    (_job_dir(cfg, "missing")).mkdir(parents=True, exist_ok=True)
    result = export_text("missing", cfg)
    assert result["ok"] is False
    assert result["stage"] == "export"
    assert "transcript.json" in result["error"]


def test_export_text_rejects_non_list_top_level(cfg):
    job = _job_dir(cfg, "dicttop")
    job.mkdir(parents=True, exist_ok=True)
    (job / "transcript.json").write_text(
        json.dumps({"segments": _segments()}, ensure_ascii=False), encoding="utf-8"
    )
    result = export_text("dicttop", cfg)
    assert result["ok"] is False
    assert "list" in result["error"]


def test_export_text_rejects_unsorted_transcript(cfg):
    bad = [
        {"start": 10.0, "end": 11.0, "orig_start": 10.0, "orig_end": 11.0,
         "speaker": "主讲", "text": "后。", "flags": []},
        {"start": 5.0, "end": 6.0, "orig_start": 5.0, "orig_end": 6.0,
         "speaker": "主讲", "text": "前。", "flags": []},
    ]
    job = _write_transcript(cfg, "unsorted", bad)
    result = export_text("unsorted", cfg)
    assert result["ok"] is False
    assert "单调" in result["error"]
    assert not (job / "transcript.srt").exists()     # 校验失败不落产物


def test_export_text_utf8_no_bom(cfg):
    job = _write_transcript(cfg, "utf8", _segments())
    assert export_text("utf8", cfg)["ok"] is True
    md_bytes = (job / "transcript.md").read_bytes()
    assert md_bytes.startswith("# ".encode("utf-8"))
    assert not md_bytes.startswith(b"\xef\xbb\xbf")
    assert "句级时间戳转写" in md_bytes.decode("utf-8")
    srt_bytes = (job / "transcript.srt").read_bytes()
    assert "主讲：嗯。".encode("utf-8") in srt_bytes
    assert b"\r\n" not in srt_bytes                  # 统一 LF


# ---------------------------------------------------------------- 真实 ffprobe（可识别性代理）
@requires_ffmpeg
def test_srt_recognized_by_ffprobe(tmp_path: Path):
    srt = tmp_path / "transcript.srt"
    srt.write_text(transcript_to_srt(_segments()), encoding="utf-8")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name",
         "-of", "json", str(srt)],
        capture_output=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    data = json.loads(proc.stdout.decode("utf-8", "replace"))
    assert "srt" in (data.get("format") or {}).get("format_name", "")
