# -*- coding: utf-8 -*-
"""P10 全文成稿单测：分块、校验、去重、缓存、整篇落盘与 CLI 注册。

原则：**绝不访问真实 API**；所有用例 monkeypatch ``compose.chat_completion``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.cli import build_parser, cmd_compose
from app.config import ConfigError, load_config
from app.pipeline import compose as compose_mod
from app.pipeline.compose import (
    Chunk,
    Segment,
    annotate_orig_times,
    compose_job,
    fmt_time,
    load_segments,
    merge_paragraphs,
    parse_card_json,
    parse_paragraphs,
    plan_chunks,
    sanitize_card,
    validate_chunk,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """compose_job 会在调用模型前检查 key；单测统一注入假 key。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-real")


def _segments(n: int = 30, chars: int = 90) -> List[Segment]:
    base = "亚里士多德在物理学第三卷讨论运动与无限，这里有一个签能的概念需要纠正。"
    text = (base * 20)[:chars]
    return [Segment(index=i, start=i * 4.0, end=i * 4.0 + 3.0, text=text) for i in range(n)]


def _response(text: str, *, prompt: int = 1000, completion: int = 500) -> Dict[str, Any]:
    return {"text": text, "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                    "cache_hit_tokens": 0}, "model": "mock-flash"}


def _card(segments: List[Segment]) -> Dict[str, Any]:
    return {
        "title": "亚里士多德《物理学》第三卷",
        "topic": "运动与潜能实现",
        "speaker": "丁耘",
        "outline": [{"time": fmt_time(segments[0].start), "title": "导入"},
                    {"time": fmt_time(segments[-1].start), "title": "总结"}],
        "glossary": [{"correct": "潜能", "variants": ["签能", "潜证"], "note": "同音误写"},
                     {"correct": "无限", "variants": ["无线"], "note": "同音误写"}],
        "people": ["亚里士多德"],
        "books": ["《物理学》"],
        "notes": "去掉语气词",
    }


def _chunk_reply(user: str) -> str:
    """从待整理转写里取核心行，机械去语气词，返回一个带时间戳的自然段。"""
    marker = "【待整理转写】"
    body = user.split(marker, 1)[-1]
    lines = [ln for ln in body.splitlines() if ln.startswith("[")]
    if not lines:
        return "[00:00] 无内容。"
    text = "".join(ln.split("] ", 1)[-1] for ln in lines)
    text = text.replace("呃", "").replace("啊", "").replace("签能", "潜能")
    return f"{lines[0].split(']')[0]}] {text}"


def _install_fake(monkeypatch, card: Dict[str, Any], *, bad_chunks: int = 0,
                  calls: List[Dict[str, Any]] | None = None):
    calls = calls if calls is not None else []
    state = {"bad": bad_chunks}

    def fake(cfg, messages, **kwargs):
        user = messages[-1]["content"]
        calls.append({"user": user, **kwargs})
        if "整讲逐句转写" in user:
            return _response(json.dumps(card, ensure_ascii=False))
        if state["bad"] > 0:
            state["bad"] -= 1
            return _response("没有时间戳的坏输出。")
        return _response(_chunk_reply(user))

    monkeypatch.setattr(compose_mod, "chat_completion", fake)
    return calls


def _make_transcript(cfg, job_id: str = "job1", segments: List[Segment] | None = None) -> Path:
    job_out = Path(cfg.path("out_dir")) / job_id
    job_out.mkdir(parents=True, exist_ok=True)
    segs = segments or _segments()
    payload = [{"start": s.start, "end": s.end, "text": s.text, "text_raw": s.text}
               for s in segs]
    (job_out / "transcript.json").write_text(json.dumps(payload, ensure_ascii=False),
                                             encoding="utf-8")
    return job_out


# ================================================================ 基础工具
def test_fmt_time_keeps_minutes_not_hours():
    assert fmt_time(0) == "00:00"
    assert fmt_time(59.6) == "01:00"
    assert fmt_time(6233) == "103:53"


def test_load_segments_accepts_dict_and_skips_empty(tmp_path: Path):
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps({"segments": [
        {"start": 0, "end": 1, "text": "甲"},
        {"start": 1, "end": 2, "text": ""},
        {"start": 2, "end": 3, "text_raw": "乙"},
    ]}, ensure_ascii=False), encoding="utf-8")
    segs = load_segments(path)
    assert [s.text for s in segs] == ["甲", "乙"]
    assert segs[1].index == 2


# ================================================================ 分块 / 解析 / 校验
def test_plan_chunks_keeps_segments_whole_and_overlaps():
    segs = [Segment(index=i, start=i * 2.0, end=i * 2.0 + 1.9, text="字" * 100)
            for i in range(10)]
    chunks = plan_chunks(segs, cfg=None, chunk_chars=300, overlap_segments=2)
    assert [c.index for c in chunks] == [1, 2, 3, 4]
    assert chunks[0].overlap_count == 0
    assert chunks[1].overlap_count == 2
    assert chunks[1].segs[:2] == chunks[0].segs[-2:]
    # 核心长度不超过目标 + 单段长度（重叠段另计）
    for c in chunks:
        core = c.segs[c.overlap_count:]
        assert len("".join(s.text for s in core)) <= 300 + 100


def test_parse_paragraphs_strips_fences_and_headers():
    text = "```markdown\n说明文字\n[00:10] 第一段。\n\n[00:20] 第二段。\n```"
    paras = parse_paragraphs(text)
    assert [p["time"] for p in paras] == ["00:10", "00:20"]
    assert paras[0]["text"] == "第一段。"


def test_validate_chunk_rules(cfg):
    orig = "[00:10] 呃" + "字" * 100 + " 300 元。"
    good = "[00:10] " + "字" * 90 + " 300 元。"
    assert validate_chunk(orig, good, ["00:10"], cfg)["ok"]
    short = validate_chunk(orig, "[00:10] 太短。", ["00:10"], cfg)
    assert not short["ok"] and any(r.startswith("too_short") for r in short["reasons"])
    added = validate_chunk(orig, good + " 新增 2024 年。", ["00:10"], cfg)
    assert added["ok"] and any(w.startswith("added_numbers") for w in added["warnings"])
    bad_time = validate_chunk(orig, good.replace("[00:10]", "[09:99]"), ["00:10"], cfg)
    assert not bad_time["ok"] and any(r.startswith("bad_times") for r in bad_time["reasons"])
    reversed_ = validate_chunk(orig, "[00:20] 甲。\n\n[00:10] 乙。", ["00:10", "00:20"], cfg)
    assert not reversed_["ok"] and "non_monotonic" in reversed_["reasons"]


def test_validate_chunk_allows_dropping_known_noise_number(cfg):
    orig = "[00:10] " + "字" * 100 + " 8分尼。"
    fixed = "[00:10] " + "字" * 100 + " 巴门尼德。"
    v = validate_chunk(orig, fixed, ["00:10"], cfg, allowed_drop_digits=["8"])
    assert v["ok"] and not v["warnings"]  # 白名单里的转写噪声数字允许消失
    v2 = validate_chunk(orig, fixed, ["00:10"], cfg, allowed_drop_digits=[])
    assert v2["ok"] and any(w.startswith("dropped_numbers") for w in v2["warnings"])


# ================================================================ 课程档案
def test_parse_card_json_tolerates_fence_and_trailing_comma():
    raw = '```json\n{"title": "x", "glossary": [],}\n```'
    assert parse_card_json(raw)["title"] == "x"


def test_sanitize_card_filters_invented_variants(cfg):
    segs = [Segment(index=0, start=0, end=1, text="这里讲签能和潜能，也提到无线和《物理学》。")]
    card = {
        "title": "t", "outline": [{"time": "00:00", "title": "导入"}],
        "glossary": [
            {"correct": "潜能", "variants": ["签能", "不存在的词"]},
            {"correct": "不存在", "variants": ["也不存在"]},
            {"correct": "无限", "variants": ["无线"]},
        ],
        "people": ["亚里士多德", "不存在的人"],
        "books": ["《物理学》", "《不存在的书》"],
        "notes": "n",
    }
    clean = sanitize_card(card, segs, cfg)
    assert [g["correct"] for g in clean["glossary"]] == ["潜能", "无限"]
    assert clean["glossary"][0]["variants"] == ["签能"]
    assert clean["people"] == []
    assert clean["books"] == ["《物理学》"]


# ================================================================ 合并 / 去重
def test_merge_paragraphs_dedupes_chunk_overlap():
    r1 = {"chunk": 1, "paragraphs": [
        {"time": "00:10", "start_s": 10, "text": "甲讲完了。乙讲完了。"}]}
    r2 = {"chunk": 2, "paragraphs": [
        {"time": "00:20", "start_s": 20, "text": "乙讲完了。丙开始。"},
        {"time": "00:20", "start_s": 20, "text": "丙开始。"}]}
    merged = merge_paragraphs([r1, r2])
    assert merged[0]["text"] == "甲讲完了。乙讲完了。"
    assert "乙讲完了。丙开始。" == merged[1]["text"] or merged[1]["text"] == "丙开始。"
    assert all(b["start_s"] >= a["start_s"] for a, b in zip(merged, merged[1:]))


# ================================================================ 时间轴
def test_annotate_orig_times_maps_clean_to_original():
    seg = Segment(index=0, start=10.2, end=12.0, text="甲",
                  orig_start=35.2, orig_end=37.0)
    chunk = Chunk(index=1, segs=[seg])
    paras = [{"time": "00:10", "start_s": 10, "text": "甲", "chunk": 1}]
    out = annotate_orig_times(paras, [chunk])
    assert out[0]["orig_start_s"] == 35.2
    assert out[0]["orig_time"] == "00:35"


def test_load_segments_keeps_both_axes(tmp_path: Path):
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps([
        {"start": 10.2, "end": 12.0, "orig_start": 35.2, "orig_end": 37.0, "text": "甲"},
    ], ensure_ascii=False), encoding="utf-8")
    seg = load_segments(path)[0]
    assert seg.start == 10.2 and seg.orig_start_s == 35.2
    assert seg.orig_end_s == 37.0


# ================================================================ 整篇（mock API）
def test_compose_job_writes_manuscript_and_skips_on_second_run(cfg, monkeypatch):
    job_id = "job_c"
    job_out = _make_transcript(cfg, job_id)
    segs = _segments()
    calls = _install_fake(monkeypatch, _card(segs))
    result = compose_job(job_id, cfg, chunk_chars=600, concurrency=1)
    assert result["ok"] and not result["skipped"]
    md = job_out / "manuscript.md"
    mj = job_out / "manuscript.json"
    report = job_out / "compose_report.json"
    assert md.is_file() and mj.is_file() and report.is_file()
    payload = json.loads(mj.read_text(encoding="utf-8"))
    assert payload["transcript_sha1"]
    assert payload["stats"]["n_paragraphs"] >= 1
    assert payload["stats"]["timestamps"]["monotonic"]
    assert payload["paragraphs"][0]["orig_time"]
    text = md.read_text(encoding="utf-8")
    assert "[00:00]" in text or "[00:04]" in text
    assert "签能" not in text and "潜能" in text
    # 第二次：hash 未变 → 跳过，不再调用模型
    n_calls = len(calls)
    again = compose_job(job_id, cfg, chunk_chars=600, concurrency=1)
    assert again["skipped"] and len(calls) == n_calls
    # force 会重做成稿，但课程档案与分块缓存仍复用 → 不再调用模型
    forced = compose_job(job_id, cfg, chunk_chars=600, concurrency=1, force=True)
    assert not forced["skipped"]
    assert len(calls) == n_calls


def test_compose_job_uses_chunk_cache(cfg, monkeypatch):
    job_id = "job_cache"
    _make_transcript(cfg, job_id)
    segs = _segments()
    calls = _install_fake(monkeypatch, _card(segs))
    compose_job(job_id, cfg, chunk_chars=600, concurrency=1)
    n_first = len(calls)
    # 删掉成稿产物、保留缓存 → 重跑时不该再请求分块
    for name in ("manuscript.md", "manuscript.json", "compose_report.json"):
        (Path(cfg.path("out_dir")) / job_id / name).unlink()
    compose_job(job_id, cfg, chunk_chars=600, concurrency=1)
    # 课程档案也在缓存中，因此不应有任何新请求
    assert len(calls) == n_first


def test_compose_job_falls_back_when_chunk_always_invalid(cfg, monkeypatch):
    job_id = "job_bad"
    job_out = _make_transcript(cfg, job_id)
    segs = _segments()
    _install_fake(monkeypatch, _card(segs), bad_chunks=99)
    result = compose_job(job_id, cfg, chunk_chars=600, concurrency=1)
    assert result["ok"]
    payload = json.loads((job_out / "manuscript.json").read_text(encoding="utf-8"))
    assert payload["stats"]["n_fallback"] >= 1
    assert "〔" not in (job_out / "manuscript.md").read_text(encoding="utf-8")


def test_compose_job_dry_run_needs_no_api(cfg):
    job_id = "job_dry"
    _make_transcript(cfg, job_id)
    result = compose_job(job_id, cfg, dry_run=True, chunk_chars=600)
    assert result["dry_run"] and result["n_chunks"] >= 1
    assert not (Path(cfg.path("out_dir")) / job_id / "manuscript.md").exists()


def test_compose_job_missing_transcript_raises(cfg):
    from app.pipeline.compose import ComposeError

    with pytest.raises(ComposeError):
        compose_job("no_such_job", cfg, dry_run=True)


# ================================================================ 配置 / CLI
def test_config_rejects_bad_compose_values():
    with pytest.raises(ConfigError):
        load_config(overrides={"compose.chunk_chars": 10})
    with pytest.raises(ConfigError):
        load_config(overrides={"compose.min_ratio": 0.9, "compose.max_ratio": 0.5})
    with pytest.raises(ConfigError):
        load_config(overrides={"compose.concurrency": 99})


def test_cli_registers_compose_subcommand():
    parser = build_parser()
    args = parser.parse_args(["compose", "job1", "--dry-run"])
    assert args.func is cmd_compose
    assert args.dry_run is True
    assert args.target == "job1"
