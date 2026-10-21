# -*- coding: utf-8 -*-
"""裁剪与映射（P3-T2）：head/tail + 内部长静音 → kept/removed、orig↔clean 映射、clean wav 渲染。

口径锁定（详见 ``docs/工程日志.md``「P3 开工第一件事」一节）
------------------------------------------------------------
- **head/tail 直接用 analyze 的最终边界**：``trim.head_s`` / ``trim.tail_s``（已含人工修正）。
- **内部切除**只取 ``internal_silence.candidates`` 里 ``edge == "internal"`` 的候选：
  先按 ``[head, tail]`` 裁剪（``cs = max(cut_start, head)``、``ce = min(cut_end, tail)``），
  ``ce > cs`` 才切。``cut_start/cut_end`` **已含 0.25 s 保护带**，这里绝不能再加一次；
  跨 head/tail 的候选**只切裁内部分，不得整段计算**——这是旧 f24 clean 3633.8 s 的根因：
  f24 最长内部静音 332.55–502.97 s（170.4 s）被 head=479.5 s 落在里面，旧算法整段扣掉，
  其中 147.2 s 在 head 之前本来就被 head 裁掉 → 重复计算 147.2 s；又因为按候选原始时长
  而不是 ``cut_start/cut_end`` 扣，52 条候选双侧漏扣 0.25 s 保护带，合计 26.0 s；
  3633.8 + 173.2 ≈ 3806.9 s，与逐区间复算一致。因此 **f24 clean = 3806.5 s**（验收 ±0.5 s），
  计划里的 3412 s（更早 VAD-only 口径）与旧日志 3633.8 s 均已作废；六文件合计 clean = 32082.2 s。
- ``kept`` = ``[head, tail]`` 去掉内部切后的补集；clean 时间轴从 0 连续累加。
- ``removed`` = ``[0, head)``（head_nonspeech）+ 各内部切（internal_silence）+ ``[tail, duration)``
  （tail_nonspeech），按 orig_start 升序。
- ``internal_cut_s = (tail − head) − clean_duration_s``（P3 实测 kept 差；**不是**
  ``AnalyzeResult.trim.internal_cut_s`` 的旧语义——后者是「头尾区之外候选的原始时长和」，
  既含保护带重复、也含跨 head/tail 的重复，只能对照，不能拿来算 clean）。

渲染轴（44.1k）很大：f24 的 44.1k wav ≈ 824 MB、f06 ≈ 1.07 GB，clean 输出也有数百 MB，
所以 ``render_clean`` 用 numpy memmap + 分块拷贝（峰值内存只与块大小有关），
输出 float32 单声道 wav，采样率与源一致。
"""

from __future__ import annotations

import bisect
import json
import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_setup import get_logger
from ..models import (
    REMOVE_HEAD,
    REMOVE_INTERNAL_SILENCE,
    REMOVE_TAIL,
    AnalyzeResult,
    EditMap,
    GainInfo,
    KeptSpan,
    RemovedSpan,
)

log = get_logger(__name__)

#: 时间比较容差（秒）；analyze 缓存里的时间都是 0.01/0.001 s 对齐的，这个量级足够
EPS = 1e-9
#: 合并「重叠/相接」内部切时的容差（秒）
MERGE_EPS = 1e-6
#: 渲染输出时长与 ``TrimPlan.clean_duration_s`` 的最大允许偏差（秒）
DURATION_TOL_S = 0.010
#: 渲染时分块拷的样本数（1M 样本 ≈ 4 MB float32，峰值内存可控）
RENDER_CHUNK_SAMPLES = 1 << 20

_WAV_FMT_PCM = 1
_WAV_FMT_FLOAT = 3
_WAV_FMT_EXTENSIBLE = 0xFFFE


class TrimError(RuntimeError):
    """裁剪/映射/渲染失败（analyze 结果非法、wav 不可用、时长对不上等）。"""


# ---------------------------------------------------------------- 裁剪计划
@dataclass
class TrimPlan:
    """一次裁剪的完整计划（P3-T2 产出，export/cli 消费）。

    - ``kept``：保留区间四元组（orig_start, orig_end, clean_start, clean_end），clean 轴从 0 连续；
    - ``removed``：删除区间（orig 轴，含原因），与 kept 一起恰好覆盖 ``[0, orig_duration_s]``；
    - ``internal_cut_s``：P3 实测的内部切除量 = ``(tail − head) − clean_duration_s``。
    """

    kept: List[KeptSpan] = field(default_factory=list)
    removed: List[RemovedSpan] = field(default_factory=list)
    clean_duration_s: float = 0.0
    orig_duration_s: float = 0.0
    head_s: float = 0.0
    tail_s: float = 0.0
    internal_cut_s: float = 0.0
    source: str = ""

    @property
    def kept_intervals(self) -> List[Tuple[float, float]]:
        """kept 的 orig 区间 ``[(orig_start, orig_end), ...]``。"""
        return [(k.orig_start, k.orig_end) for k in self.kept]

    @property
    def cut_ratio(self) -> float:
        """删除比例（0–1）。"""
        if self.orig_duration_s <= 0:
            return 0.0
        return max(0.0, 1.0 - self.clean_duration_s / self.orig_duration_s)

    def to_dict(self) -> Dict[str, Any]:
        """摘要（日志/测试用；``edit.json`` 走 :func:`edit_map` 的契约结构）。"""
        return {
            "source": self.source,
            "orig_duration_s": round(self.orig_duration_s, 3),
            "clean_duration_s": round(self.clean_duration_s, 3),
            "head_s": round(self.head_s, 3),
            "tail_s": round(self.tail_s, 3),
            "internal_cut_s": round(self.internal_cut_s, 3),
            "cut_ratio": round(self.cut_ratio, 4),
            "n_kept": len(self.kept),
            "n_removed": len(self.removed),
        }


def _merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """把候选区间排序并合并重叠/相接项（保证补集与覆盖断言无歧义）。"""
    ordered = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    merged: List[Tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + MERGE_EPS:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _clamp_cuts(
    candidates: Sequence[Any],
    head: float,
    tail: float,
    min_s: float,
    guard: float,
) -> List[Tuple[float, float]]:
    """内部静音候选 → 裁内切除区间 ``[(cut_start, cut_end), ...]``。

    只处理 ``edge == "internal"``；先夹到 ``[head, tail]``（跨边界只切裁内部分）。
    ``cut_start/cut_end`` 已含保护带，**原样使用**；只有老缓存缺这两个字段时才按
    ``start + guard`` / ``end − guard`` 兜底重算（并用 ``min_s`` 过滤）。
    """
    cuts: List[Tuple[float, float]] = []
    for index, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            raise TrimError(f"internal_silence.candidates[{index}] 不是 dict：{cand!r}")
        if str(cand.get("edge", "")).strip().lower() != "internal":
            continue
        try:
            if "cut_start" in cand and "cut_end" in cand:
                cut_start = float(cand["cut_start"])
                cut_end = float(cand["cut_end"])
            else:  # 老缓存兜底：按 guard 现场算，不重复加保护带
                start = float(cand["start"])
                end = float(cand["end"])
                if end - start < min_s - EPS:
                    continue
                cut_start = start + guard
                cut_end = end - guard
        except (KeyError, TypeError, ValueError) as exc:
            raise TrimError(f"internal_silence.candidates[{index}] 字段非法：{cand!r}") from exc
        if not (math.isfinite(cut_start) and math.isfinite(cut_end)):
            raise TrimError(f"internal_silence.candidates[{index}] 含非有限值：{cand!r}")
        clamped_start = max(cut_start, head)
        clamped_end = min(cut_end, tail)
        if clamped_end - clamped_start > MERGE_EPS:
            cuts.append((clamped_start, clamped_end))
    return _merge_intervals(cuts)


def _complement(
    head: float, tail: float, cuts: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """``[head, tail]`` 去掉 cuts 后的补集（有序、不重叠、长度为 0 的段不返回）。"""
    spans: List[Tuple[float, float]] = []
    cursor = head
    for start, end in cuts:
        start = max(start, cursor)
        if start - cursor > EPS:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if tail - cursor > EPS:
        spans.append((cursor, tail))
    return spans


def plan_trim(
    analyze: AnalyzeResult,
    cfg: Any = None,
    *,
    duration_s: Optional[float] = None,
) -> TrimPlan:
    """由 ``AnalyzeResult`` 生成裁剪计划（纯函数，不碰文件）。

    ``duration_s`` 可覆盖 ``analyze.duration_s``（真实缓存里它被四舍五入到 3 位小数，
    渲染校验时可用源 wav 的真实时长）。``cfg`` 只读 ``trim.internal_silence_min_s`` /
    ``trim.internal_guard_s``，且**仅**用于候选缺 ``cut_start/cut_end`` 的兜底，正常不会再加保护带。
    """
    if analyze is None:
        raise TrimError("缺少 AnalyzeResult")
    if not getattr(analyze, "ok", True):
        raise TrimError(f"分析结果不可用（ok=False）：{getattr(analyze, 'error', '')}")

    dur = float(analyze.duration_s if duration_s is None else duration_s)
    if not math.isfinite(dur) or dur <= 0.0:
        raise TrimError(f"原始时长非法：{dur!r}")

    trim = analyze.trim or {}
    try:
        head = float(trim.get("head_s", 0.0))
        tail = float(trim.get("tail_s", dur))
    except (TypeError, ValueError) as exc:
        raise TrimError(f"trim.head_s/tail_s 不是数值：{trim!r}") from exc
    if not (math.isfinite(head) and math.isfinite(tail)):
        raise TrimError(f"trim.head_s/tail_s 含非有限值：head={head!r} tail={tail!r}")
    if head < 0.0:
        raise TrimError(f"head_s 不能为负：{head}")
    if tail <= head:
        raise TrimError(f"tail_s({tail}) 必须大于 head_s({head})")
    if tail > dur + EPS:
        # analyze 缓存把 duration_s 四舍五入到 3 位，tail 可能比它大几毫秒；在 10 ms 内视作贴边
        if tail - dur <= DURATION_TOL_S:
            tail = dur
        else:
            raise TrimError(f"tail_s({tail}) 超出原始时长 {dur}")
    if head >= dur:
        raise TrimError(f"head_s({head}) 不小于原始时长 {dur}")

    min_s = float(cfg.get("trim.internal_silence_min_s", 3.0)) if cfg is not None else 3.0
    guard = float(cfg.get("trim.internal_guard_s", 0.25)) if cfg is not None else 0.25
    candidates = (analyze.internal_silence or {}).get("candidates") or []
    cuts = _clamp_cuts(candidates, head, tail, min_s, guard)

    kept: List[KeptSpan] = []
    clean_duration = 0.0
    for start, end in _complement(head, tail, cuts):
        length = end - start
        kept.append(KeptSpan(
            orig_start=start, orig_end=end,
            clean_start=clean_duration, clean_end=clean_duration + length,
        ))
        clean_duration += length

    removed: List[RemovedSpan] = []
    if head > EPS:
        removed.append(RemovedSpan(0.0, head, REMOVE_HEAD))
    for start, end in cuts:
        removed.append(RemovedSpan(start, end, REMOVE_INTERNAL_SILENCE))
    if dur - tail > EPS:
        removed.append(RemovedSpan(tail, dur, REMOVE_TAIL))
    removed.sort(key=lambda item: item.orig_start)

    plan = TrimPlan(
        kept=kept,
        removed=removed,
        clean_duration_s=clean_duration,
        orig_duration_s=dur,
        head_s=head,
        tail_s=tail,
        internal_cut_s=(tail - head) - clean_duration,
        source=str(getattr(analyze, "job_id", "") or ""),
    )
    log.info(
        "裁剪计划：%s dur=%.3fs head=%.2fs tail=%.2fs kept=%d removed=%d "
        "clean=%.3fs internal_cut=%.3fs",
        plan.source or "?", plan.orig_duration_s, plan.head_s, plan.tail_s,
        len(plan.kept), len(plan.removed), plan.clean_duration_s, plan.internal_cut_s,
    )
    return plan


# ---------------------------------------------------------------- 时间映射
class TimeMap:
    """orig ↔ clean 时间映射（二分查找，容差 ``1e-6`` s）。

    语义边界：
    - ``orig_to_clean(t)``：t 落在某个 kept 区间（含端点，容差 1e-6）→ clean 秒；
      落在删除区/越界 → ``None``；
    - ``clean_to_orig(t)``：clean 轴 [0, clean_duration_s] 内（含端点，容差 1e-6）→ orig 秒，否则 ``None``；
    - ``orig_span_to_clean(a, b)``：orig 区间 → clean 区间列表；与被切的区间相交时按 kept 拆成多段
      （若切口正好落在区间内，两段在 clean 轴上相邻，仍返回两段）。
    """

    #: 端点比较容差（秒）：契约要求 1e-6
    TOL = 1e-6

    __slots__ = ("_os", "_oe", "_cs", "_ce", "clean_duration_s", "orig_duration_s")

    def __init__(self, kept: Sequence[KeptSpan], orig_duration_s: Optional[float] = None) -> None:
        spans = sorted(kept, key=lambda k: k.orig_start)
        self._os = [float(k.orig_start) for k in spans]
        self._oe = [float(k.orig_end) for k in spans]
        self._cs = [float(k.clean_start) for k in spans]
        self._ce = [float(k.clean_end) for k in spans]
        self.clean_duration_s = self._ce[-1] if spans else 0.0
        self.orig_duration_s = (
            float(orig_duration_s) if orig_duration_s is not None else self._oe[-1] if spans else 0.0
        )

    def __len__(self) -> int:
        return len(self._os)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"TimeMap(spans={len(self._os)}, clean={self.clean_duration_s:.3f}s)"

    def orig_to_clean(self, t: float) -> Optional[float]:
        """orig 时间 → clean 时间；被切除或越界返回 ``None``。"""
        t = float(t)
        if not math.isfinite(t):
            return None
        index = bisect.bisect_right(self._os, t) - 1
        if index < 0:
            return None
        if t <= self._oe[index] + self.TOL:
            return self._cs[index] + (t - self._os[index])
        return None

    def clean_to_orig(self, t: float) -> Optional[float]:
        """clean 时间 → orig 时间；越界返回 ``None``。"""
        t = float(t)
        if not math.isfinite(t) or t < -self.TOL or t > self.clean_duration_s + self.TOL:
            return None
        index = bisect.bisect_right(self._cs, t) - 1
        if index < 0:
            return None
        if t <= self._ce[index] + self.TOL:
            return self._os[index] + (t - self._cs[index])
        return None

    def orig_span_to_clean(self, a: float, b: float) -> List[Tuple[float, float]]:
        """orig 区间 ``[a, b)`` → clean 区间列表（按 kept 拆分，空的返回 ``[]``）。"""
        a, b = float(a), float(b)
        if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
            return []
        out: List[Tuple[float, float]] = []
        index = bisect.bisect_right(self._oe, a)
        total = len(self._os)
        while index < total:
            if self._os[index] >= b:
                break
            start = max(a, self._os[index])
            end = min(b, self._oe[index])
            if end - start > EPS:
                offset = self._cs[index] - self._os[index]
                out.append((start + offset, end + offset))
            index += 1
        return out


def build_timemap(plan: TrimPlan) -> TimeMap:
    """由裁剪计划构建 :class:`TimeMap`。"""
    return TimeMap(plan.kept, orig_duration_s=plan.orig_duration_s)


def map_segments(
    plan: TrimPlan, segments: Sequence[Sequence[float]]
) -> List[Tuple[float, float]]:
    """orig 轴 VAD/转写段 → clean 轴段列表（被切除部分裁掉，跨切口则拆成多段）。"""
    timemap = build_timemap(plan)
    out: List[Tuple[float, float]] = []
    for item in segments:
        if len(item) < 2:
            raise TrimError(f"段格式非法（需要 [start, end]）：{item!r}")
        out.extend(timemap.orig_span_to_clean(item[0], item[1]))
    return out


def chain_clean_segments(analyze: AnalyzeResult, cfg: Any = None) -> List[Tuple[float, float]]:
    """≥ ``levels.min_continuous_speech_s`` 的连续语音段 → clean 轴区间（loudness 的 ASL 掩码用）。"""
    min_continuous = (
        float(cfg.get("levels.min_continuous_speech_s", 3.0)) if cfg is not None else 3.0
    )
    segments = [
        (float(seg[0]), float(seg[1]))
        for seg in (analyze.segments or [])
        if float(seg[1]) - float(seg[0]) >= min_continuous - EPS
    ]
    return map_segments(plan_trim(analyze, cfg), segments)


# ---------------------------------------------------------------- WAV 读写（渲染轴）
@dataclass
class _WavLayout:
    """WAV 头信息 + data 段偏移（用于 memmap，不把整个文件读进内存）。"""

    sample_rate: int
    channels: int
    sampwidth: int
    format_tag: int
    n_frames: int
    data_offset: int
    dtype: str


def _read_wav_layout(path: str | Path) -> _WavLayout:
    """解析 RIFF/WAVE 头，返回采样率/声道/格式/帧数/data 偏移。"""
    wav = Path(path)
    try:
        file_size = wav.stat().st_size
    except OSError as exc:
        raise TrimError(f"wav 不可用：{wav}：{exc}") from exc
    with open(wav, "rb") as fh:
        header = fh.read(12)
        if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise TrimError(f"不是合法的 WAV（缺 RIFF/WAVE 头）：{wav}")
        pos = 12
        sample_rate = channels = sampwidth = format_tag = 0
        fmt_seen = False
        data_offset = -1
        data_bytes = 0
        while pos + 8 <= file_size:
            fh.seek(pos)
            chunk_head = fh.read(8)
            if len(chunk_head) < 8:
                break
            chunk_id = chunk_head[0:4]
            chunk_size = struct.unpack_from("<I", chunk_head, 4)[0]
            body = pos + 8
            if chunk_id == b"fmt ":
                fh.seek(body)
                raw = fh.read(min(chunk_size, 64))
                if len(raw) < 16:
                    raise TrimError(f"WAV fmt 段过短：{wav}")
                format_tag = struct.unpack_from("<H", raw, 0)[0]
                channels = struct.unpack_from("<H", raw, 2)[0]
                sample_rate = struct.unpack_from("<I", raw, 4)[0]
                sampwidth = max(1, struct.unpack_from("<H", raw, 14)[0] // 8)
                if format_tag == _WAV_FMT_EXTENSIBLE and len(raw) >= 26:
                    # WAVE_FORMAT_EXTENSIBLE：真实格式在 SubFormat 前两字节
                    format_tag = struct.unpack_from("<H", raw, 24)[0]
                fmt_seen = True
            elif chunk_id == b"data":
                if not fmt_seen:
                    raise TrimError(f"WAV data 段出现在 fmt 段之前：{wav}")
                data_offset = body
                data_bytes = chunk_size
                break
            pos = body + chunk_size + (chunk_size & 1)

    if data_offset < 0:
        raise TrimError(f"WAV 缺少 data 段：{wav}")
    if sample_rate <= 0 or channels <= 0:
        raise TrimError(f"WAV 采样率/声道非法（sr={sample_rate}, ch={channels}）：{wav}")
    if format_tag == _WAV_FMT_FLOAT and sampwidth == 4:
        dtype = "<f4"
    elif format_tag == _WAV_FMT_PCM and sampwidth == 2:
        dtype = "<i2"
    elif format_tag == _WAV_FMT_PCM and sampwidth == 4:
        dtype = "<i4"
    elif format_tag == _WAV_FMT_PCM and sampwidth == 1:
        dtype = "u1"
    else:
        raise TrimError(
            f"不支持的 WAV 编码（format_tag={format_tag}, sampwidth={sampwidth}）：{wav}"
        )
    block = channels * sampwidth
    data_bytes = min(int(data_bytes), max(0, file_size - data_offset))
    n_frames = data_bytes // block
    if n_frames <= 0:
        raise TrimError(f"WAV 没有音频帧：{wav}")
    return _WavLayout(
        sample_rate=int(sample_rate),
        channels=int(channels),
        sampwidth=int(sampwidth),
        format_tag=int(format_tag),
        n_frames=int(n_frames),
        data_offset=int(data_offset),
        dtype=dtype,
    )


def _open_memmap(path: Path, layout: _WavLayout) -> np.memmap:
    shape: Tuple[int, ...] = (
        (layout.n_frames,) if layout.channels == 1 else (layout.n_frames, layout.channels)
    )
    return np.memmap(
        str(path), dtype=layout.dtype, mode="r",
        offset=layout.data_offset, shape=shape,
    )


def _read_frame_chunk(mm: np.ndarray, layout: _WavLayout, start: int, stop: int) -> np.ndarray:
    """取 ``[start, stop)`` 帧并转 float32（多声道取第 0 声道）。始终返回拷贝，避免持有 memmap。"""
    if layout.channels == 1:
        raw = np.array(mm[start:stop], copy=True)
    else:
        raw = np.array(mm[start:stop, 0], copy=True)
    if layout.dtype == "<i2":
        return raw.astype(np.float32) / 32768.0
    if layout.dtype == "<i4":
        return raw.astype(np.float32) / 2147483648.0
    if layout.dtype == "u1":
        return (raw.astype(np.float32) - 128.0) / 128.0
    return raw.astype(np.float32, copy=False)


def _write_wav_f32_header(fh: Any, sample_rate: int, n_frames: int) -> None:
    """写 44 字节 float32 单声道 WAV 头。"""
    data_bytes = int(n_frames) * 4
    block_align = 4
    byte_rate = sample_rate * block_align
    fh.write(b"RIFF")
    fh.write(struct.pack("<I", 36 + data_bytes))
    fh.write(b"WAVE")
    fh.write(b"fmt ")
    fh.write(struct.pack("<IHHIIHH", 16, _WAV_FMT_FLOAT, 1, sample_rate,
                         byte_rate, block_align, 32))
    fh.write(b"data")
    fh.write(struct.pack("<I", data_bytes))


def render_clean(
    src_wav: str | Path,
    plan: TrimPlan,
    dst_wav: str | Path,
) -> Dict[str, Any]:
    """按 kept 区间把渲染轴源 wav 切片拼接成 clean wav（float32 单声道，采样率不变）。

    用 memmap + ``RENDER_CHUNK_SAMPLES`` 分块拷贝，f24（824 MB）/f06（1.07 GB）级源也不会
    把内存打爆。写完后校验输出时长与 ``plan.clean_duration_s`` 相差 < 10 ms。

    返回 ``{ok, samples, duration_s, sr, elapsed_s}``。
    """
    started = time.perf_counter()
    src = Path(src_wav)
    dst = Path(dst_wav)
    if not isinstance(plan, TrimPlan):
        raise TrimError(f"plan 必须是 TrimPlan，收到：{type(plan).__name__}")
    if not src.is_file():
        raise TrimError(f"渲染源 wav 不存在：{src}")
    layout = _read_wav_layout(src)
    sr = layout.sample_rate
    n_src = layout.n_frames

    # 1) kept 区间 → 源样本下标（clamp 到源长度；时长校验会兜住越界）
    sample_spans: List[Tuple[int, int]] = []
    out_frames = 0
    for kept in plan.kept:
        start = int(round(float(kept.orig_start) * sr))
        end = int(round(float(kept.orig_end) * sr))
        start = max(0, min(start, n_src))
        end = max(start, min(end, n_src))
        if end > start:
            sample_spans.append((start, end))
            out_frames += end - start

    # 2) 先写头并预分配文件，再用 r+ memmap 分块填充（out_frames=0 时只留 44 字节头）
    dst.parent.mkdir(parents=True, exist_ok=True)
    data_bytes = out_frames * 4
    try:
        with open(dst, "wb") as fh:
            _write_wav_f32_header(fh, sr, out_frames)
            fh.truncate(44 + data_bytes)
        src_mm = _open_memmap(src, layout)
        try:
            dst_mm = (
                np.memmap(str(dst), dtype="<f4", mode="r+", offset=44, shape=(out_frames,))
                if out_frames else None
            )
            try:
                position = 0
                for span_start, span_end in sample_spans:
                    cursor = span_start
                    while cursor < span_end:
                        chunk_end = min(span_end, cursor + int(RENDER_CHUNK_SAMPLES))
                        chunk = _read_frame_chunk(src_mm, layout, cursor, chunk_end)
                        if dst_mm is not None:
                            dst_mm[position: position + chunk.size] = chunk
                        position += int(chunk.size)
                        cursor = chunk_end
                if dst_mm is not None:
                    dst_mm.flush()
            finally:
                if dst_mm is not None:
                    del dst_mm
        finally:
            del src_mm
    except OSError as exc:
        raise TrimError(f"渲染 clean wav 失败：{dst}：{exc}") from exc

    duration = out_frames / float(sr)
    expected = float(plan.clean_duration_s)
    delta = abs(duration - expected)
    elapsed = time.perf_counter() - started
    if delta > DURATION_TOL_S:
        raise TrimError(
            f"渲染时长与计划不一致：实际 {duration:.4f}s，计划 {expected:.4f}s"
            f"（差 {delta * 1000.0:.1f}ms > {DURATION_TOL_S * 1000.0:.0f}ms）：{dst}"
        )
    log.info(
        "clean 渲染完成：%s samples=%d dur=%.3fs sr=%d（%.1fs）",
        dst.name, out_frames, duration, sr, elapsed,
    )
    return {
        "ok": True,
        "samples": int(out_frames),
        "duration_s": duration,
        "sr": int(sr),
        "elapsed_s": round(elapsed, 3),
    }


# ---------------------------------------------------------------- edit.json
def edit_map(plan: TrimPlan, gain: Optional[GainInfo] = None) -> EditMap:
    """裁剪计划 → ``edit.json`` 契约对象（kept 四元组 / removed 原因 / gain）。"""
    return EditMap(
        kept=list(plan.kept),
        removed=list(plan.removed),
        gain=gain,
    )


def write_edit_json(
    plan: TrimPlan,
    path: str | Path,
    gain: Optional[GainInfo] = None,
) -> Path:
    """写 ``edit.json``（UTF-8、``ensure_ascii=False``、indent=2，原子替换）。"""
    dst = Path(path)
    payload = edit_map(plan, gain=gain).to_dict()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp.replace(dst)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - 清理失败不掩盖原因
            pass
        raise TrimError(f"写 edit.json 失败：{dst}：{exc}") from exc
    return dst


__all__ = [
    "DURATION_TOL_S",
    "RENDER_CHUNK_SAMPLES",
    "TimeMap",
    "TrimError",
    "TrimPlan",
    "build_timemap",
    "chain_clean_segments",
    "edit_map",
    "map_segments",
    "plan_trim",
    "render_clean",
    "write_edit_json",
]
