# -*- coding: utf-8 -*-
"""P3-T2 裁剪与映射单测：``app/pipeline/trim.py``。

分两层：
- **合成用例**（默认跑）：裁剪计划（无内部切/全切/跨 head 与 tail 的候选/空候选/
  tail<dur/重叠合并/保护带不重复加）、kept+removed 覆盖性、TimeMap 往返与区间拆分、
  map_segments、chain_clean_segments、edit.json 契约、render_clean（float32/int16/
  分块/空计划/时长校验）。
- **真实缓存用例**（读 ``work/cache/*.analyze.json``，缓存缺失则 skip）：
  f24 clean=3806.5±0.5 s、internal_cut≈274.5 s、六文件合计 32082.2±2 s、VAD 语音保留率 100%；
  ``@pytest.mark.slow`` 的 44.1k 真渲染（f24）默认不跑，用 ``pytest -m slow`` 显式执行。

时间映射的边界说明（有意为之，不是 bug）：clean 时间轴是「拼接」出来的，切口两侧的
orig 时间会被映射到**同一个** clean 时刻（前一段的结尾 = 后一段的开头），所以
orig→clean→orig 在切口端点上是二对一的、不可能同时还原；验收的往返用
「clean 轴边界点」和「kept 区间内/端点附近的 orig 点」来测（转写用的正是 clean 轴）。
"""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from app.models import (
    REMOVE_HEAD,
    REMOVE_INTERNAL_SILENCE,
    REMOVE_REASONS,
    REMOVE_TAIL,
    AnalyzeResult,
    GainInfo,
)
from app.pipeline import trim as trim_mod
from app.pipeline.trim import (
    TrimError,
    TrimPlan,
    build_timemap,
    chain_clean_segments,
    edit_map,
    map_segments,
    plan_trim,
    render_clean,
    write_edit_json,
)

from conftest import ROOT, SAMPLES, requires_data, requires_ffmpeg

CACHE_DIR = ROOT / "work" / "cache"
ANALYZE_SUFFIX = ".analyze.json"

#: 6 个样例的 analyze 缓存前缀（job id 前 8 位）
SIX = {
    "f00": "621cb91c",
    "f04": "9f3feb49",
    "f05": "0a7fa0ed",
    "f06": "b00e197d",
    "f20": "4db18037",
    "f24": "7927721f",
}
#: P2 最终边界 + P3 重订的 clean 时长（docs/工程日志.md「P3 开工」表；f24 旧值作废）
EXPECTED = {
    "f00": (0.0, 3000.5, 2745.3),
    "f04": (0.0, 8295.0, 8131.8),
    "f05": (0.0, 6325.7, 6201.1),
    "f06": (119.5, 6070.6, 5327.8),
    "f20": (89.5, 6150.5, 5869.6),
    "f24": (479.5, 4560.5, 3806.5),
}
#: 六文件 clean 合计（s）
EXPECTED_TOTAL_CLEAN = 32082.2


# ---------------------------------------------------------------- 合成数据工具
def _make_analyze(
    head: float,
    tail: float,
    duration: float,
    candidates: list | None = None,
    segments: list | None = None,
    job_id: str = "syn",
) -> AnalyzeResult:
    """合成 AnalyzeResult（trim + internal_silence.candidates + VAD 段）。"""
    return AnalyzeResult(
        ok=True,
        job_id=job_id,
        duration_s=float(duration),
        trim={"head_s": float(head), "tail_s": float(tail)},
        internal_silence={"candidates": list(candidates or [])},
        segments=[list(seg) for seg in (segments or [])],
    )


def _cand(
    start: float,
    end: float,
    cut_start: float | None = None,
    cut_end: float | None = None,
    edge: str = "internal",
) -> dict:
    """合成一条 internal_silence 候选（cut_start/cut_end 缺省时模拟老缓存）。"""
    item = {
        "start": float(start),
        "end": float(end),
        "duration_s": round(float(end) - float(start), 3),
        "edge": edge,
    }
    if cut_start is not None:
        item["cut_start"] = float(cut_start)
    if cut_end is not None:
        item["cut_end"] = float(cut_end)
    return item


def _assert_plan_consistent(plan: TrimPlan, duration: float) -> None:
    """断言 kept + removed 恰好覆盖 [0, duration] 且无重叠；clean 轴连续。"""
    assert plan.orig_duration_s == pytest.approx(duration, abs=1e-9)
    assert plan.clean_duration_s == pytest.approx(
        sum(k.clean_end - k.clean_start for k in plan.kept), abs=1e-6
    )
    assert plan.internal_cut_s == pytest.approx(
        (plan.tail_s - plan.head_s) - plan.clean_duration_s, abs=1e-6
    )
    assert plan.internal_cut_s == pytest.approx(
        sum(r.duration_s for r in plan.removed if r.reason == REMOVE_INTERNAL_SILENCE),
        abs=1e-6,
    )

    intervals = sorted(
        [(k.orig_start, k.orig_end) for k in plan.kept]
        + [(r.orig_start, r.orig_end) for r in plan.removed]
    )
    assert intervals, "kept/removed 不能同时为空"
    cursor = 0.0
    for start, end in intervals:
        assert start == pytest.approx(cursor, abs=1e-6), f"覆盖缺口/重叠：{start} vs {cursor}"
        assert end > start
        cursor = end
    assert cursor == pytest.approx(duration, abs=1e-6)

    assert [r.orig_start for r in plan.removed] == sorted(r.orig_start for r in plan.removed)
    for removed in plan.removed:
        assert removed.reason in REMOVE_REASONS

    # clean 轴从 0 连续累加，长度等于 orig 长度
    position = 0.0
    for kept in plan.kept:
        assert kept.clean_start == pytest.approx(position, abs=1e-9)
        assert kept.clean_end - kept.clean_start == pytest.approx(
            kept.orig_end - kept.orig_start, abs=1e-9
        )
        position = kept.clean_end
    assert position == pytest.approx(plan.clean_duration_s, abs=1e-9)

    # 头尾 removed 的边界与 head/tail 一致
    head_spans = [r for r in plan.removed if r.reason == REMOVE_HEAD]
    tail_spans = [r for r in plan.removed if r.reason == REMOVE_TAIL]
    if plan.head_s > 0:
        assert len(head_spans) == 1 and head_spans[0].orig_start == pytest.approx(0.0)
        assert head_spans[0].orig_end == pytest.approx(plan.head_s)
    else:
        assert head_spans == []
    if plan.orig_duration_s - plan.tail_s > 1e-9:
        assert len(tail_spans) == 1
        assert tail_spans[0].orig_end == pytest.approx(plan.orig_duration_s)
    else:
        assert tail_spans == []


def _speech_kept_stats(analyze: AnalyzeResult, plan: TrimPlan) -> tuple[float, float]:
    """返回 (裁内 VAD 语音秒数, 映射到 clean 后的语音秒数)——两者必须相等（保留率 100%）。"""
    timemap = build_timemap(plan)
    inside = 0.0
    mapped = 0.0
    for seg in analyze.segments or []:
        start, end = float(seg[0]), float(seg[1])
        clipped_start = max(start, plan.head_s)
        clipped_end = min(end, plan.tail_s)
        if clipped_end > clipped_start:
            inside += clipped_end - clipped_start
        for clean_start, clean_end in timemap.orig_span_to_clean(start, end):
            mapped += clean_end - clean_start
    return inside, mapped


# ---------------------------------------------------------------- 裁剪计划：合成
def test_plan_no_internal_cuts(cfg):
    analyze = _make_analyze(10.0, 90.0, 100.0)
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, 100.0)
    assert plan.clean_duration_s == pytest.approx(80.0)
    assert plan.internal_cut_s == pytest.approx(0.0)
    assert len(plan.kept) == 1
    kept = plan.kept[0]
    assert (kept.orig_start, kept.orig_end) == pytest.approx((10.0, 90.0))
    assert (kept.clean_start, kept.clean_end) == pytest.approx((0.0, 80.0))
    assert [(r.orig_start, r.orig_end) for r in plan.removed] == [(0.0, 10.0), (90.0, 100.0)]
    assert [r.reason for r in plan.removed] == [REMOVE_HEAD, REMOVE_TAIL]


def test_plan_head_zero_and_tail_at_duration(cfg):
    plan = plan_trim(_make_analyze(0.0, 100.0, 100.0), cfg)
    _assert_plan_consistent(plan, 100.0)
    assert plan.removed == []
    assert plan.clean_duration_s == pytest.approx(100.0)


def test_plan_candidate_crossing_head_not_double_counted(cfg):
    """f24 的回归形态：候选整段 0–300 s，head=100 → 只切 [100, 299.75]。"""
    analyze = _make_analyze(
        100.0, 900.0, 1000.0,
        [_cand(0.0, 300.0, cut_start=0.25, cut_end=299.75)],
    )
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, 1000.0)
    assert plan.internal_cut_s == pytest.approx(199.75)
    assert plan.clean_duration_s == pytest.approx(800.0 - 199.75)
    # 不能按候选原始时长（300 s）或未裁剪长度（299.5 s）算
    assert plan.internal_cut_s != pytest.approx(300.0)
    assert plan.internal_cut_s != pytest.approx(299.5)
    internal = [r for r in plan.removed if r.reason == REMOVE_INTERNAL_SILENCE]
    assert len(internal) == 1
    assert (internal[0].orig_start, internal[0].orig_end) == pytest.approx((100.0, 299.75))
    assert plan.kept[0].orig_start == pytest.approx(299.75)
    assert plan.kept[0].clean_start == pytest.approx(0.0)


def test_plan_candidate_crossing_tail_only_cuts_inside(cfg):
    analyze = _make_analyze(
        0.0, 700.0, 1000.0,
        [_cand(600.0, 900.0, cut_start=600.25, cut_end=899.75)],
    )
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, 1000.0)
    assert plan.internal_cut_s == pytest.approx(99.75)  # 700 - 600.25
    assert plan.clean_duration_s == pytest.approx(600.25)
    internal = [r for r in plan.removed if r.reason == REMOVE_INTERNAL_SILENCE]
    assert (internal[0].orig_start, internal[0].orig_end) == pytest.approx((600.25, 700.0))


def test_plan_candidates_outside_trim_ignored(cfg):
    analyze = _make_analyze(
        100.0, 800.0, 1000.0,
        [
            _cand(10.0, 90.0, cut_start=10.25, cut_end=89.75),   # 全在 head 之前
            _cand(850.0, 950.0, cut_start=850.25, cut_end=949.75),  # 全在 tail 之后
            _cand(0.0, 99.0, cut_start=0.0, cut_end=99.0, edge="head"),
            _cand(801.0, 999.0, cut_start=801.25, cut_end=999.0, edge="tail"),
        ],
    )
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, 1000.0)
    assert plan.internal_cut_s == pytest.approx(0.0)
    assert plan.clean_duration_s == pytest.approx(700.0)
    assert [r.reason for r in plan.removed] == [REMOVE_HEAD, REMOVE_TAIL]


def test_plan_empty_candidates(cfg):
    plan = plan_trim(_make_analyze(5.0, 95.0, 100.0, []), cfg)
    _assert_plan_consistent(plan, 100.0)
    assert plan.clean_duration_s == pytest.approx(90.0)


def test_plan_full_cut_keeps_nothing(cfg):
    analyze = _make_analyze(
        10.0, 90.0, 100.0,
        [_cand(10.0, 90.0, cut_start=10.0, cut_end=90.0)],
    )
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, 100.0)
    assert plan.kept == []
    assert plan.clean_duration_s == pytest.approx(0.0)
    assert plan.internal_cut_s == pytest.approx(80.0)
    timemap = build_timemap(plan)
    assert len(timemap) == 0
    assert timemap.orig_to_clean(50.0) is None
    assert timemap.clean_to_orig(0.0) is None
    assert timemap.orig_span_to_clean(0.0, 100.0) == []


def test_plan_cut_starting_at_zero_without_head(cfg):
    analyze = _make_analyze(
        0.0, 100.0, 100.0,
        [_cand(0.0, 100.0, cut_start=0.0, cut_end=100.0)],
    )
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, 100.0)
    assert plan.kept == []
    assert [r.reason for r in plan.removed] == [REMOVE_INTERNAL_SILENCE]
    assert plan.internal_cut_s == pytest.approx(100.0)


def test_plan_short_tail_span_kept(cfg):
    """tail 比 duration 小一点点（缓存 3 位小数四舍五入的常见情况）也要记 tail_nonspeech。"""
    plan = plan_trim(_make_analyze(0.0, 99.5, 100.0), cfg)
    _assert_plan_consistent(plan, 100.0)
    tail = [r for r in plan.removed if r.reason == REMOVE_TAIL]
    assert len(tail) == 1
    assert (tail[0].orig_start, tail[0].orig_end) == pytest.approx((99.5, 100.0))

    # tail 只比 duration 大几毫秒（cache duration 舍入）→ 贴边、不产生 tail 段
    plan2 = plan_trim(_make_analyze(0.0, 100.005, 100.0), cfg)
    _assert_plan_consistent(plan2, 100.0)
    assert [r for r in plan2.removed if r.reason == REMOVE_TAIL] == []
    assert plan2.tail_s == pytest.approx(100.0)


def test_plan_overlapping_candidates_merged(cfg):
    analyze = _make_analyze(
        0.0, 100.0, 100.0,
        [
            _cand(10.0, 40.0, cut_start=10.0, cut_end=40.0),
            _cand(30.0, 60.0, cut_start=30.0, cut_end=60.0),
            _cand(30.0, 60.0, cut_start=30.0, cut_end=60.0),  # 重复条目
        ],
    )
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, 100.0)
    assert plan.internal_cut_s == pytest.approx(50.0)
    assert len([r for r in plan.removed if r.reason == REMOVE_INTERNAL_SILENCE]) == 1
    assert [plan.kept[0].orig_end, plan.kept[1].orig_start] == pytest.approx([10.0, 60.0])


def test_plan_guard_not_added_twice(cfg):
    """cut_start/cut_end 已含保护带：改 cfg 的 guard 不应影响 [cut_start, cut_end]。"""
    analyze = _make_analyze(
        0.0, 100.0, 100.0,
        [_cand(10.0, 20.0, cut_start=10.25, cut_end=19.75)],
    )
    plan_default = plan_trim(analyze, cfg)
    cfg_wide = cfg.replace({"trim.internal_guard_s": 5.0})
    plan_wide = plan_trim(analyze, cfg_wide)
    assert plan_default.internal_cut_s == pytest.approx(9.5)
    assert plan_wide.internal_cut_s == pytest.approx(9.5)


def test_plan_fallback_when_cut_fields_missing(cfg):
    """老缓存没有 cut_start/cut_end 时按 guard 兜底（仅在兜底路径加保护带）。"""
    analyze = _make_analyze(0.0, 100.0, 100.0, [_cand(10.0, 20.0)])
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, 100.0)
    assert plan.internal_cut_s == pytest.approx(9.5)  # 10+0.25 → 20-0.25
    cfg_wide = cfg.replace({"trim.internal_guard_s": 1.0})
    plan_wide = plan_trim(analyze, cfg_wide)
    assert plan_wide.internal_cut_s == pytest.approx(8.0)  # 11 → 19


def test_plan_duration_override_and_source(cfg):
    analyze = _make_analyze(10.0, 90.0, 100.0, job_id="abc_测试")
    plan = plan_trim(analyze, cfg, duration_s=120.0)
    assert plan.orig_duration_s == pytest.approx(120.0)
    assert plan.tail_s == pytest.approx(90.0)
    assert plan.source == "abc_测试"
    _assert_plan_consistent(plan, 120.0)


@pytest.mark.parametrize(
    "head,tail,duration,kwargs,match",
    [
        (-1.0, 90.0, 100.0, None, "不能为负"),
        (90.0, 90.0, 100.0, None, "必须大于"),
        (0.0, 150.0, 100.0, None, "超出原始时长"),
        (0.0, 100.0, 100.0, {"duration_s": 0.0}, "原始时长非法"),
        (0.0, 100.0, 100.0, {"duration_s": -5.0}, "原始时长非法"),
    ],
)
def test_plan_rejects_invalid_boundaries(cfg, head, tail, duration, kwargs, match):
    analyze = _make_analyze(head, tail, duration)
    if kwargs is None:
        with pytest.raises(TrimError, match=match):
            plan_trim(analyze, cfg)
    else:
        with pytest.raises(TrimError, match=match):
            plan_trim(analyze, cfg, **kwargs)


def test_plan_rejects_failed_or_malformed_analyze(cfg):
    failed = AnalyzeResult(ok=False, error="vad 挂了", duration_s=10.0)
    with pytest.raises(TrimError, match="不可用"):
        plan_trim(failed, cfg)
    with pytest.raises(TrimError, match="不是 dict"):
        plan_trim(_make_analyze(0.0, 10.0, 10.0, ["oops"]), cfg)
    with pytest.raises(TrimError, match="字段非法"):
        plan_trim(_make_analyze(0.0, 10.0, 10.0, [{"edge": "internal", "start": 1.0}]), cfg)
    with pytest.raises(TrimError, match="必须是 TrimPlan|缺少 AnalyzeResult"):
        plan_trim(None, cfg)  # type: ignore[arg-type]


# ---------------------------------------------------------------- TimeMap
def _cut_plan(cfg) -> TrimPlan:
    """10–90 s 保留、30–40 s 切掉：kept=(10,30)→(0,20)、(40,90)→(20,70)。"""
    analyze = _make_analyze(
        10.0, 90.0, 100.0,
        [_cand(30.0, 40.0, cut_start=30.0, cut_end=40.0)],
    )
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, 100.0)
    return plan


def test_timemap_point_mapping(cfg):
    timemap = build_timemap(_cut_plan(cfg))
    assert len(timemap) == 2
    # kept 内线性映射
    assert timemap.orig_to_clean(10.0) == pytest.approx(0.0)
    assert timemap.orig_to_clean(20.0) == pytest.approx(10.0)
    assert timemap.orig_to_clean(50.0) == pytest.approx(20.0 + 10.0)
    assert timemap.orig_to_clean(90.0) == pytest.approx(70.0)
    # 删除区 / 越界 / 非法值
    for bad in (0.0, 9.999, 30.0001, 35.0, 39.999, 90.0001, 200.0, -1.0, math.nan):
        assert timemap.orig_to_clean(bad) is None, bad
    # clean 轴
    assert timemap.clean_to_orig(0.0) == pytest.approx(10.0)
    assert timemap.clean_to_orig(20.0) == pytest.approx(40.0)  # 缝上取后一段（clean 轴连续）
    assert timemap.clean_to_orig(70.0) == pytest.approx(90.0)
    assert timemap.clean_to_orig(19.0) == pytest.approx(29.0)
    for bad in (-1.0, 70.1, 200.0, math.nan):
        assert timemap.clean_to_orig(bad) is None, bad


def test_timemap_roundtrip_random_1000(cfg):
    """随机 1000 个 kept 内 orig 点：orig→clean→orig 误差 <1 ms，平均单次 <1 ms。"""
    plan = _cut_plan(cfg)
    timemap = build_timemap(plan)
    kept = plan.kept_intervals
    rng = np.random.default_rng(20261021)
    errors = []
    started = time.perf_counter()
    for _ in range(1000):
        index = int(rng.integers(0, len(kept)))
        start, end = kept[index]
        t = float(rng.uniform(start, end))
        clean = timemap.orig_to_clean(t)
        assert clean is not None
        back = timemap.clean_to_orig(clean)
        assert back is not None
        errors.append(abs(back - t))
    elapsed = time.perf_counter() - started
    assert max(errors) < 1e-3, f"最大往返误差 {max(errors) * 1000:.6f} ms"
    assert elapsed / 1000.0 < 1e-3, f"平均单次往返 {elapsed / 1000.0 * 1e3:.6f} ms"


def test_timemap_roundtrip_boundary_1000(cfg):
    """1000 个边界点：clean 轴边界（含缝）往返 <1 ms；orig 端点附近同样 <1 ms。"""
    timemap = build_timemap(_cut_plan(cfg))
    clean_boundaries = [0.0, timemap.clean_duration_s]
    for kept in _cut_plan(cfg).kept:
        clean_boundaries.extend([kept.clean_start, kept.clean_end])
    points = []
    for base in clean_boundaries:
        for delta in (-1e-3, -1e-6, 0.0, 1e-6, 1e-3):
            points.append(min(max(base + delta, 0.0), timemap.clean_duration_s))
    while len(points) < 1000:
        points.extend(points[: 1000 - len(points)])
    max_error = 0.0
    started = time.perf_counter()
    for clean in points[:1000]:
        orig = timemap.clean_to_orig(clean)
        assert orig is not None
        back = timemap.orig_to_clean(orig)
        assert back is not None
        max_error = max(max_error, abs(back - clean))
    elapsed = time.perf_counter() - started
    assert max_error < 1e-3, f"clean 轴边界最大往返误差 {max_error * 1000:.6f} ms"
    assert elapsed / 1000.0 < 1e-3

    # orig 端点附近（端点内侧 1e-6）：可逆；整段起点/终点单独核对
    errors = []
    for kept in _cut_plan(cfg).kept:
        for t in (kept.orig_start + 1e-6, kept.orig_end - 1e-6):
            clean = timemap.orig_to_clean(t)
            back = timemap.clean_to_orig(clean)
            errors.append(abs(back - t))
    assert max(errors) < 1e-3
    assert timemap.clean_to_orig(timemap.orig_to_clean(10.0)) == pytest.approx(10.0)
    assert timemap.clean_to_orig(timemap.orig_to_clean(90.0)) == pytest.approx(90.0)


def test_timemap_span_to_clean_splits(cfg):
    timemap = build_timemap(_cut_plan(cfg))
    assert timemap.orig_span_to_clean(20.0, 50.0) == pytest.approx([(10.0, 20.0), (20.0, 30.0)])
    assert timemap.orig_span_to_clean(5.0, 15.0) == pytest.approx([(0.0, 5.0)])
    assert timemap.orig_span_to_clean(25.0, 45.0) == pytest.approx([(15.0, 20.0), (20.0, 25.0)])
    assert timemap.orig_span_to_clean(0.0, 100.0) == pytest.approx([(0.0, 20.0), (20.0, 70.0)])
    # 整段落在删除区 / 越界 / 反向区间 → []
    assert timemap.orig_span_to_clean(30.0, 40.0) == []
    assert timemap.orig_span_to_clean(35.0, 38.0) == []
    assert timemap.orig_span_to_clean(95.0, 120.0) == []
    assert timemap.orig_span_to_clean(50.0, 20.0) == []
    assert timemap.orig_span_to_clean(math.nan, 20.0) == []


def test_map_segments(cfg):
    plan = _cut_plan(cfg)
    mapped = map_segments(plan, [(0.0, 5.0), (12.0, 28.0), (35.0, 45.0), (88.0, 95.0)])
    assert mapped == pytest.approx([(2.0, 18.0), (20.0, 25.0), (68.0, 70.0)])
    with pytest.raises(TrimError, match="段格式非法"):
        map_segments(plan, [(1.0,)])


def test_chain_clean_segments_filters_and_maps(cfg):
    segments = [(0.0, 10.0), (20.0, 25.0), (30.0, 60.0), (70.0, 71.0), (80.0, 90.0)]
    analyze = _make_analyze(10.0, 90.0, 100.0, segments=segments)
    # 默认 min_continuous=3.0：丢掉 1 s 的 (70,71)，(0,10) 与 head 不相交也丢掉
    assert chain_clean_segments(analyze, cfg) == pytest.approx(
        [(10.0, 15.0), (20.0, 50.0), (70.0, 80.0)]
    )
    cfg_low = cfg.replace({"levels.min_continuous_speech_s": 0.5})
    assert chain_clean_segments(analyze, cfg_low) == pytest.approx(
        [(10.0, 15.0), (20.0, 50.0), (60.0, 61.0), (70.0, 80.0)]
    )

    # 有内部切时按 clean 轴拆分
    analyze_cut = _make_analyze(
        10.0, 90.0, 100.0,
        [_cand(30.0, 40.0, cut_start=30.0, cut_end=40.0)],
        segments=[(12.0, 60.0)],
    )
    assert chain_clean_segments(analyze_cut, cfg) == pytest.approx([(2.0, 20.0), (20.0, 40.0)])


# ---------------------------------------------------------------- edit.json
def test_edit_map_and_write_edit_json(cfg, tmp_path: Path):
    plan = _cut_plan(cfg)
    gain = GainInfo(type="static", gain_db=-9.9, min_db=-9.9, max_db=-9.9)
    edit = edit_map(plan, gain=gain)
    payload = edit.to_dict()
    assert set(payload) == {"kept", "removed", "reason_enum", "gain"}
    assert payload["reason_enum"] == list(REMOVE_REASONS)
    assert payload["kept"][0] == [10.0, 30.0, 0.0, 20.0]
    assert payload["removed"] == [
        {"orig_start": 0.0, "orig_end": 10.0, "reason": REMOVE_HEAD},
        {"orig_start": 30.0, "orig_end": 40.0, "reason": REMOVE_INTERNAL_SILENCE},
        {"orig_start": 90.0, "orig_end": 100.0, "reason": REMOVE_TAIL},
    ]
    assert payload["gain"]["gain_db"] == pytest.approx(-9.9)

    out = write_edit_json(plan, tmp_path / "out" / "edit.json", gain=gain)
    assert Path(out).is_file()
    text = Path(out).read_text(encoding="utf-8")
    assert "\n  \"kept\":" in text  # indent=2
    assert "\\u" not in text  # ensure_ascii=False
    loaded = json.loads(text)
    assert loaded == edit.to_dict()
    assert not list(Path(out).parent.glob("*.tmp"))


# ---------------------------------------------------------------- render_clean
def _write_wav(path: Path, x: np.ndarray, sr: int, kind: str = "f32") -> Path:
    import scipy.io.wavfile as wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "f32":
        wavfile.write(str(path), sr, np.ascontiguousarray(x, dtype=np.float32))
    elif kind == "i16":
        wavfile.write(str(path), sr, np.ascontiguousarray(x, dtype=np.int16))
    else:  # pragma: no cover - 测试内部用
        raise ValueError(kind)
    return path


def _read_float_at(path: Path, layout, frame_index: int) -> float:
    with open(path, "rb") as fh:
        fh.seek(layout.data_offset + frame_index * 4)
        return float(struct.unpack("<f", fh.read(4))[0])


def test_render_clean_float32_exact(cfg, tmp_path: Path):
    """切片拼接逐样本一致：kept=(0.2,0.6)+(0.9,1.6) @ 8 kHz float32。"""
    sr = 8000
    x = np.linspace(-0.5, 0.5, sr * 2, dtype=np.float32)
    src = _write_wav(tmp_path / "src.wav", x, sr)
    analyze = _make_analyze(
        0.2, 1.6, 2.0,
        [_cand(0.6, 0.9, cut_start=0.6, cut_end=0.9)],
    )
    plan = plan_trim(analyze, cfg)
    assert plan.clean_duration_s == pytest.approx(1.1)
    dst = tmp_path / "clean.wav"
    stats = render_clean(src, plan, dst)

    assert stats["ok"] is True
    assert stats["sr"] == sr
    assert stats["samples"] == 8800
    assert stats["duration_s"] == pytest.approx(1.1, abs=1e-9)
    assert stats["elapsed_s"] >= 0.0

    import scipy.io.wavfile as wavfile

    out_sr, out = wavfile.read(str(dst))
    assert out_sr == sr
    assert out.dtype == np.float32
    expected = np.concatenate([x[1600:4800], x[7200:12800]])
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)
    # 头信息与文件大小对得上
    layout = trim_mod._read_wav_layout(dst)
    assert layout.n_frames == 8800
    assert layout.sample_rate == sr
    assert Path(dst).stat().st_size == 44 + 8800 * 4


def test_render_clean_int16_source(tmp_path: Path):
    sr = 8000
    x = (np.sin(np.arange(sr) / 20.0) * 20000).astype(np.int16)
    src = _write_wav(tmp_path / "i16.wav", x, sr, kind="i16")
    plan = plan_trim(_make_analyze(0.0, 1.0, 1.0))
    dst = tmp_path / "i16_clean.wav"
    stats = render_clean(src, plan, dst)
    assert stats["samples"] == sr
    import scipy.io.wavfile as wavfile

    _, out = wavfile.read(str(dst))
    assert np.allclose(out, x.astype(np.float32) / 32768.0, atol=1e-7)


def test_render_clean_chunk_boundaries(cfg, tmp_path: Path, monkeypatch):
    """把分块大小调小，验证跨块拷贝拼接仍逐样本一致。"""
    monkeypatch.setattr(trim_mod, "RENDER_CHUNK_SAMPLES", 1000)
    sr = 8000
    x = np.linspace(-1.0, 1.0, sr, dtype=np.float32)
    src = _write_wav(tmp_path / "chunk.wav", x, sr)
    plan = plan_trim(_make_analyze(0.1, 0.9, 1.0))
    dst = tmp_path / "chunk_clean.wav"
    stats = render_clean(src, plan, dst)
    assert stats["samples"] == int(round(0.8 * sr))
    import scipy.io.wavfile as wavfile

    _, out = wavfile.read(str(dst))
    assert np.array_equal(out, x[800:7200])


def test_render_clean_empty_plan(cfg, tmp_path: Path):
    sr = 8000
    src = _write_wav(tmp_path / "empty.wav", np.zeros(sr, dtype=np.float32), sr)
    plan = plan_trim(_make_analyze(
        0.2, 0.4, 0.5,
        [_cand(0.2, 0.4, cut_start=0.2, cut_end=0.4)],
    ))
    assert plan.kept == [] and plan.clean_duration_s == 0.0
    dst = tmp_path / "empty_clean.wav"
    stats = render_clean(src, plan, dst)
    assert stats["samples"] == 0
    assert stats["duration_s"] == pytest.approx(0.0)
    assert Path(dst).stat().st_size == 44  # 只有 44 字节 wav 头（空 data 段）


def test_render_clean_rejects_bad_input(cfg, tmp_path: Path):
    sr = 8000
    src = _write_wav(tmp_path / "s.wav", np.zeros(sr, dtype=np.float32), sr)
    plan = plan_trim(_make_analyze(0.0, 1.0, 1.0))
    with pytest.raises(TrimError, match="不存在"):
        render_clean(tmp_path / "nope.wav", plan, tmp_path / "o.wav")
    with pytest.raises(TrimError, match="TrimPlan"):
        render_clean(src, {"kept": []}, tmp_path / "o.wav")  # type: ignore[arg-type]

    # 计划 clean 时长与 kept 实际长度不符 → 时长校验失败
    broken = plan_trim(_make_analyze(0.0, 1.0, 1.0))
    broken.clean_duration_s = broken.clean_duration_s + 0.5
    with pytest.raises(TrimError, match="渲染时长与计划不一致"):
        render_clean(src, broken, tmp_path / "bad.wav")


# ---------------------------------------------------------------- 真实缓存用例
def _cache_path(prefix: str) -> Path | None:
    matches = sorted(CACHE_DIR.glob(f"{prefix}*{ANALYZE_SUFFIX}"))
    return matches[0] if matches else None


def _load_analyze(prefix: str) -> tuple[AnalyzeResult | None, str]:
    path = _cache_path(prefix)
    if path is None:
        return None, ""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("ok") or "result" not in data:
        return None, ""
    return AnalyzeResult(**data["result"]), path.name[: -len(ANALYZE_SUFFIX)]


def test_data_f24_plan_baseline(cfg):
    """f24：clean=3806.5±0.5 s、internal_cut≈274.5 s（旧的 3633.8/3412 作废）。"""
    analyze, _ = _load_analyze(SIX["f24"])
    if analyze is None:
        pytest.skip("缺少 f24 analyze 缓存（work/cache/7927721f*.analyze.json）")
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, analyze.duration_s)

    assert plan.head_s == pytest.approx(479.5, abs=0.1)
    assert plan.tail_s == pytest.approx(4560.5, abs=0.1)
    assert plan.clean_duration_s == pytest.approx(3806.5, abs=0.5)
    assert plan.internal_cut_s == pytest.approx(274.5, abs=0.5)
    # 旧值：3633.8（double count + 保护带）、3412（更早 VAD-only 口径）都不能再出现
    assert plan.clean_duration_s > 3700.0
    # analyze 的旧语义字段保留原值（供对照，不是 P3 切除量）
    assert float(analyze.trim["internal_cut_s"]) == pytest.approx(447.2, abs=0.1)

    # 52 条内部切：第一条从 head 起切到 502.72（head 落在那条 170.4 s 长静音里）
    assert len(plan.kept) == 52
    assert len(plan.removed) == 54  # head + 52 internal + tail
    assert plan.kept[0].orig_start == pytest.approx(502.72, abs=0.05)
    assert plan.kept[0].clean_start == pytest.approx(0.0)
    assert plan.removed[0].reason == REMOVE_HEAD
    assert plan.removed[-1].reason == REMOVE_TAIL
    assert plan.removed[-1].orig_end == pytest.approx(analyze.duration_s)

    # 语音保留率 100%：裁内 VAD 语音 → clean 后一秒不少
    inside, mapped = _speech_kept_stats(analyze, plan)
    assert inside > 0.0
    assert mapped == pytest.approx(inside, abs=1e-6)
    assert mapped / inside == pytest.approx(1.0, abs=1e-9)


def test_data_six_files_clean_total(cfg):
    """六文件 clean 合计 32082.2±2 s（各文件同时对照 P2 最终边界表）。"""
    plans: dict[str, TrimPlan] = {}
    for key, prefix in SIX.items():
        analyze, _ = _load_analyze(prefix)
        if analyze is None:
            pytest.skip(f"缺少 {key} 的 analyze 缓存（work/cache/{prefix}*.analyze.json）")
        plan = plan_trim(analyze, cfg)
        _assert_plan_consistent(plan, analyze.duration_s)
        head, tail, clean = EXPECTED[key]
        assert plan.head_s == pytest.approx(head, abs=0.1), key
        assert plan.tail_s == pytest.approx(tail, abs=0.1), key
        assert plan.clean_duration_s == pytest.approx(clean, abs=0.5), key
        inside, mapped = _speech_kept_stats(analyze, plan)
        assert mapped == pytest.approx(inside, abs=1e-6), key  # 语音保留率 100%
        plans[key] = plan

    assert len(plans) == 6
    total = sum(plan.clean_duration_s for plan in plans.values())
    assert total == pytest.approx(EXPECTED_TOTAL_CLEAN, abs=2.0)
    # f24 在基准表里单独强调过
    assert plans["f24"].clean_duration_s == pytest.approx(3806.5, abs=0.5)


# ---------------------------------------------------------------- 真实渲染（slow）
@requires_data
@requires_ffmpeg
@pytest.mark.slow
def test_data_render_clean_f24():
    """真实 f24：44.1k 源 → render_clean，时长一致 / 内容逐样本 / 语音保留 100% / 往返 <1ms。

    44.1k wav（≈824 MB）与临时 clean 输出都走 work/cache（gitignore），跑完删掉 clean 输出，
    不影响既有 16k / analyze 缓存。默认 ``-m "not slow"`` 不跑，验收时用 ``pytest -m slow``。
    """
    analyze, job_id = _load_analyze(SIX["f24"])
    if analyze is None:
        pytest.skip("缺少 f24 analyze 缓存（work/cache/7927721f*.analyze.json）")
    if "f24" not in SAMPLES:
        pytest.skip("data/ 下没有 f24 样例")

    from app.config import load_config
    from app.pipeline.decode import ensure_44k, wav_info

    cfg = load_config()
    src44 = ensure_44k(SAMPLES["f24"], cfg=cfg, job_id=job_id)
    assert src44 is not None and Path(src44).is_file(), "44.1k 渲染轴解码失败"
    plan = plan_trim(analyze, cfg)
    _assert_plan_consistent(plan, analyze.duration_s)

    dst = Path(cfg.path("cache_dir")) / f"{job_id}.trim_test.wav"
    try:
        stats = render_clean(src44, plan, dst)
        info = wav_info(dst)
        assert stats["sr"] == 44100
        assert info.sample_rate == 44100
        assert info.n_frames == stats["samples"]
        assert stats["duration_s"] == pytest.approx(plan.clean_duration_s, abs=1e-6)
        assert abs(info.duration_s - plan.clean_duration_s) < 0.010

        # 内容抽查：clean 开头/若干随机点 = 源对应样本（逐 float32 相等）
        src_layout = trim_mod._read_wav_layout(src44)
        dst_layout = trim_mod._read_wav_layout(dst)
        sr = 44100
        sample_spans = []
        for kept in plan.kept:
            start = max(0, min(int(round(kept.orig_start * sr)), src_layout.n_frames))
            end = max(start, min(int(round(kept.orig_end * sr)), src_layout.n_frames))
            sample_spans.append((start, end))
        rng = np.random.default_rng(7)
        checked = 0
        for _ in range(10):
            index = int(rng.integers(0, len(sample_spans)))
            start, end = sample_spans[index]
            if end <= start:
                continue
            offset = int(rng.integers(0, end - start))
            out_index = sum(e - s for s, e in sample_spans[:index]) + offset
            assert _read_float_at(src44, src_layout, start + offset) == pytest.approx(
                _read_float_at(dst, dst_layout, out_index), abs=1e-7
            )
            checked += 1
        assert checked >= 8

        # 语音保留率 100%
        inside, mapped = _speech_kept_stats(analyze, plan)
        ratio = mapped / inside if inside else 0.0

        # 随机 20 个 orig 点往返
        timemap = build_timemap(plan)
        kept = plan.kept_intervals
        errors = []
        for _ in range(20):
            index = int(rng.integers(0, len(kept)))
            start, end = kept[index]
            t = float(rng.uniform(start, end))
            back = timemap.clean_to_orig(timemap.orig_to_clean(t))
            assert back is not None
            errors.append(abs(back - t))
        max_error_ms = max(errors) * 1000.0

        assert ratio == pytest.approx(1.0, abs=1e-9)
        assert max_error_ms < 1.0
        print(
            f"[P3-T2] f24 render_clean: samples={stats['samples']} "
            f"duration={stats['duration_s']:.3f}s sr={stats['sr']} "
            f"elapsed={stats['elapsed_s']:.1f}s; plan_clean={plan.clean_duration_s:.3f}s "
            f"delta={abs(info.duration_s - plan.clean_duration_s) * 1000:.3f}ms; "
            f"speech_preserved={ratio * 100:.6f}% ({mapped:.3f}s/{inside:.3f}s); "
            f"roundtrip_max={max_error_ms:.6f}ms"
        )
    finally:
        Path(dst).unlink(missing_ok=True)
