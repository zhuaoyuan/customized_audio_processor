# -*- coding: utf-8 -*-
"""分析阶段（P2）：Silero VAD → 语音段；语音电平 ASL/噪声底/SNR；5 min 窗口波动；
内部长静音候选；头尾边界（**120 s 窗口聚合 ASR 字符率**）。

为什么头尾边界不能只看 VAD
--------------------------
Silero VAD 会把片头音乐/掌声判成语音（f24 的 0–7 s、f06/f20 的 0–121 s）；
段级 ASR 也不够（f24 在 59 s 出现「觉的旁边他有」这类音乐短幻觉词，段级字符率 >2）。
所以按计划 P2-T3：**按 120 s 窗口**聚合 `chars/s ≥ 2.0` 连续 2 窗口才开始算讲课。
窗口内的字符来自「≤30 s 分块转写」（不是只送 VAD 语音段，理由见
``window_char_rates`` 的 docstring：实测 VAD 漏检会让 cps 掉到 1.72 而误判）。

「连续 2 窗口」的工程细化（有实测依据）
--------------------------------------
纯严格连续会被**轻声窗口**打断：f06 的 120–240 s 是正常讲课（cps 2.97），但
240–360 s 只有 1.82（停顿多/轻声），严格规则会把头边界从 121 s 跳到 359.5 s，
白切 240 s 讲课。因此允许中间夹 **1 个软窗口**（``1.0 ≤ cps < 2.0``）：音乐/掌声窗口
cps 通常 <0.5，不会被误当软窗口。片尾同理，并把终点向后延伸最多 1 个软窗口
（收尾常是轻声，保守：宁少裁）。

口径（与调研报告对齐）
----------------------
- 帧：25 ms 窗 / 10 ms 跳；
- 语音帧：帧**中心**落在某个 VAD 语音段内；
- ASL：只统计「≥3 s 连续语音段」内的语音帧，``10·log10(帧功率均值)``（报告 §1.2 锁定定义）；
  另给 ``asl_all_vad_dbfs``（全部 VAD 段，即报告 §2.2 的旧口径）与 ``asl_p56_dbfs``
  （P.56 相对门限 rel_db=15，P3 响度用）作对照；
- 噪声底：全帧 dB 的低 ``noise_floor_percentile``（默认 10%）分位所对应帧的功率均值；
  SNR = ASL − 噪声底；
- 5 min 窗口：窗口内语音不足 ``window_min_speech_s`` 不统计（避免片头音乐窗口）；
- 内部长静音：``internal_silence.total_s`` = 语音段之间的空隙总和（报告 §2.3 口径），
  ``cut_s`` = 头尾裁剪区之外、P3 实际要切掉的量（避免与头尾重复计数）。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_setup import get_logger
from ..models import AnalyzeResult
from . import asr as asr_mod
from .decode import ANALYSIS_SR, read_wav_float32, wav_info

log = get_logger(__name__)

SR = ANALYSIS_SR
WIN_MS = 25.0
HOP_MS = 10.0
#: 单次计算的分块帧数（200k 帧 = 2000 s），避免百分钟文件内存爆掉
CHUNK_FRAMES = 200_000

CACHE_SUFFIX = ".analyze.json"
#: 分析结果缓存版本：算法/口径变化时 +1，自动让旧缓存失效
#: v7：新增「课前闲聊/片头片尾空白」自动切除（head_chatter_trim / head_leading_nonspeech /
#: tail_trailing_nonspeech），见 :func:`_sustained_teaching_start` 与 :func:`auto_edge_trim`。
ANALYZE_VERSION = 7


class AnalyzeError(RuntimeError):
    """分析失败（wav 缺失、VAD 不可用等）。"""


# ---------------------------------------------------------------- 帧能量
def frame_power(
    x: np.ndarray,
    sr: int = SR,
    win_ms: float = WIN_MS,
    hop_ms: float = HOP_MS,
    chunk_frames: int = CHUNK_FRAMES,
) -> Tuple[np.ndarray, np.ndarray]:
    """25 ms 窗 / 10 ms 跳的帧功率（分块 cumsum，向量化）。

    返回 ``(power, centers_s)``；``power`` 为 float64 均方功率，``centers_s`` 为帧中心秒数。
    """
    x = np.asarray(x, dtype=np.float32)
    w = int(round(sr * win_ms / 1000.0))
    h = int(round(sr * hop_ms / 1000.0))
    if w <= 0 or h <= 0:
        raise AnalyzeError(f"非法窗参数：win_ms={win_ms}, hop_ms={hop_ms}")
    if len(x) < w:
        if len(x) == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        return (
            np.array([float(np.mean(x.astype(np.float64) ** 2))]),
            np.array([len(x) / 2.0 / sr]),
        )

    n_frames = 1 + (len(x) - w) // h
    power = np.empty(n_frames, dtype=np.float64)
    for s0 in range(0, n_frames, chunk_frames):
        s1 = min(n_frames, s0 + chunk_frames)
        a = s0 * h
        b = (s1 - 1) * h + w
        seg = np.asarray(x[a:b], dtype=np.float64)
        cum = np.concatenate([[0.0], np.cumsum(seg * seg)])
        idx = np.arange(s0, s1) * h - a
        power[s0:s1] = (cum[idx + w] - cum[idx]) / w
    centers = (np.arange(n_frames) * h + w / 2.0) / sr
    return power, centers


def power_to_db(power: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), eps))


def mean_level_db(power: np.ndarray, eps: float = 1e-20) -> float:
    """功率均值 → dBFS（ASL 的基本口径）。"""
    p = np.asarray(power, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    return float(10.0 * np.log10(float(np.mean(p)) + eps))


def p56_level(power: np.ndarray, rel_db: float = 15.0, max_iter: int = 20) -> float:
    """ITU-T P.56 思路的活跃语音电平：迭代剔除低于 (L − rel_db) 的帧。"""
    p = np.asarray(power, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    level = mean_level_db(p)
    for _ in range(max_iter):
        active = p[power_to_db(p) > level - rel_db]
        if active.size == 0:
            break
        new = mean_level_db(active)
        if abs(new - level) < 0.01:
            return new
        level = new
    return level


def noise_floor_db(power: np.ndarray, percentile: float = 10.0) -> float:
    """噪声底：全帧 dB 的低 percentile 分位所对应帧的功率均值（dBFS）。"""
    p = np.asarray(power, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    db = power_to_db(p)
    threshold = float(np.percentile(db, percentile))
    quiet = p[db <= threshold]
    if quiet.size == 0:
        quiet = p
    return mean_level_db(quiet)


def percentile_stats(
    power: np.ndarray, percentiles: Sequence[float] = (10, 50, 90)
) -> Dict[str, float]:
    if np.asarray(power).size == 0:
        return {}
    db = power_to_db(np.asarray(power, dtype=np.float64))
    return {f"p{int(q)}": round(float(np.percentile(db, q)), 2) for q in percentiles}


# ---------------------------------------------------------------- 语音段工具
def speech_mask(
    segments: Sequence[Sequence[float]], n_frames: int, centers_s: np.ndarray
) -> np.ndarray:
    """帧级布尔掩码：帧中心落在任一语音段 [start, end) 内。"""
    mask = np.zeros(int(n_frames), dtype=bool)
    if n_frames == 0:
        return mask
    centers = np.asarray(centers_s, dtype=np.float64)[: int(n_frames)]
    for seg in segments:
        start, end = float(seg[0]), float(seg[1])
        if end <= start:
            continue
        mask |= (centers >= start) & (centers < end)
    return mask


def classify_segments(
    segments: Sequence[Sequence[float]], min_continuous_s: float = 3.0
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """区分「≥min_continuous_s 连续说话段」与「碎片」。"""
    continuous: List[Tuple[float, float]] = []
    fragments: List[Tuple[float, float]] = []
    for seg in segments:
        start, end = float(seg[0]), float(seg[1])
        if end - start >= min_continuous_s - 1e-9:
            continuous.append((start, end))
        else:
            fragments.append((start, end))
    return continuous, fragments


def segment_total(segments: Sequence[Sequence[float]]) -> float:
    return float(sum(max(0.0, float(s[1]) - float(s[0])) for s in segments))


# ---------------------------------------------------------------- VAD
def detect_speech(x: np.ndarray, sr: int = SR, cfg: Any = None) -> List[Tuple[float, float]]:
    """Silero VAD（faster-whisper 内置 onnx 模型）→ 10 ms 对齐的语音段列表。

    参数来自 ``cfg.vad``：threshold / min_speech_s / min_silence_s / pad_s。
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ModuleNotFoundError as exc:  # pragma: no cover - 环境缺依赖
        raise AnalyzeError("缺少 faster-whisper（Silero VAD 模型来自其内置 assets）") from exc

    threshold = float(cfg.get("vad.threshold", 0.5) if cfg else 0.5)
    min_speech = float(cfg.get("vad.min_speech_s", 0.25) if cfg else 0.25)
    min_silence = float(cfg.get("vad.min_silence_s", 0.30) if cfg else 0.30)
    pad = float(cfg.get("vad.pad_s", 0.10) if cfg else 0.10)

    options = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=int(round(min_speech * 1000)),
        min_silence_duration_ms=int(round(min_silence * 1000)),
        speech_pad_ms=int(round(pad * 1000)),
    )
    x = np.asarray(x, dtype=np.float32)
    stamps = get_speech_timestamps(x, options, sampling_rate=int(sr))
    segments: List[Tuple[float, float]] = []
    for item in stamps:
        start = round(item["start"] / float(sr) / 0.01) * 0.01
        end = round(item["end"] / float(sr) / 0.01) * 0.01
        if end > start:
            segments.append((round(start, 2), round(end, 2)))
    return segments


# ---------------------------------------------------------------- 长静音
def internal_silence(
    segments: Sequence[Sequence[float]],
    duration: float,
    min_s: float = 3.0,
    guard: float = 0.25,
    head_s: float = 0.0,
    tail_s: Optional[float] = None,
) -> Dict[str, Any]:
    """VAD 非语音段 ≥ ``min_s`` 的候选切除区间（两侧各留 ``guard`` 保护带）。

    返回里 ``total_s`` 是**语音段之间**的空隙总和（与报告 §2.3 的「内部 ≥3 s 非语音」
    对齐）；``in_head_tail_s`` 是其中落在头/尾裁剪区的部分；``cut_s`` 是 P3 真正要
    「内部切除」的量（头尾已覆盖的不重复切）。
    """
    tail = duration if tail_s is None else float(tail_s)
    gaps: List[Dict[str, Any]] = []
    previous_end = 0.0
    previous_is_edge = True  # 第一个空隙前面没有语音段 → 属于片头
    for seg in segments:
        start, end = float(seg[0]), float(seg[1])
        gap = start - previous_end
        if gap >= min_s - 1e-9:
            g0, g1 = previous_end, start
            cut_start = g0 + (0.0 if previous_is_edge else guard)
            cut_end = g1 - guard
            if cut_end > cut_start:
                gaps.append({
                    "start": round(g0, 3),
                    "end": round(g1, 3),
                    "duration_s": round(gap, 3),
                    "cut_start": round(max(cut_start, g0), 3),
                    "cut_end": round(min(cut_end, g1), 3),
                    "edge": "head" if previous_is_edge else "internal",
                })
        previous_end = max(previous_end, end)
        previous_is_edge = False
    # 片尾
    if duration - previous_end >= min_s - 1e-9:
        g0, g1 = previous_end, duration
        gaps.append({
            "start": round(g0, 3),
            "end": round(g1, 3),
            "duration_s": round(g1 - g0, 3),
            "cut_start": round(g0 + guard, 3),
            "cut_end": round(g1, 3),
            "edge": "tail",
        })

    internal_gaps = [g for g in gaps if g["edge"] == "internal"]
    in_trim = [
        g for g in internal_gaps
        if g["end"] <= head_s + 1e-9 or g["start"] >= tail - 1e-9
    ]
    out_trim = [g for g in internal_gaps if g not in in_trim]
    return {
        "min_s": min_s,
        "guard_s": guard,
        "n_gaps": len(gaps),
        "n_internal": len(internal_gaps),
        # 报告 §2.3 口径：语音段之间的空隙总和（f24 实测 613 s 量级）
        "total_s": round(sum(g["duration_s"] for g in internal_gaps), 1),
        # 其中落在头/尾裁剪区内、由头尾裁剪覆盖的部分
        "in_head_tail_s": round(sum(g["duration_s"] for g in in_trim), 1),
        # P3 真正需要「内部切除」的量（头尾已覆盖的不重复切）
        "cut_s": round(sum(g["duration_s"] for g in out_trim), 1),
        "longest_s": round(max((g["duration_s"] for g in internal_gaps), default=0.0), 1),
        "candidates": gaps,
    }


# ---------------------------------------------------------------- 5 min 窗口
def window_levels(
    power: np.ndarray,
    mask: np.ndarray,
    centers_s: np.ndarray,
    duration: float,
    win_s: float = 300.0,
    hop_s: Optional[float] = None,
    min_speech_s: float = 30.0,
    rel_db: float = 15.0,
) -> Dict[str, Any]:
    """按 ``win_s`` 窗口统计 ASL（P.56 口径），返回分布与半展宽。"""
    hop = win_s if hop_s is None else float(hop_s)
    power = np.asarray(power, dtype=np.float64)
    n = min(len(power), len(mask))
    t = 0.0
    windows: List[Dict[str, Any]] = []
    while t < duration - 1e-9:
        end = min(duration, t + win_s)
        a, b = int(round(t * 100.0)), int(round(end * 100.0))
        a, b = min(a, n), min(b, n)
        if b > a:
            sel = mask[a:b]
            count = int(sel.sum())
            if count * 0.01 >= min_speech_s:
                windows.append({
                    "t0": round(t, 1),
                    "t1": round(end, 1),
                    "speech_s": round(count * 0.01, 1),
                    "asl_dbfs": round(p56_level(power[a:b][sel], rel_db=rel_db), 2),
                })
        t += hop
    levels = [w["asl_dbfs"] for w in windows]
    out: Dict[str, Any] = {"n_windows": len(windows), "windows": windows,
                           "win_s": win_s, "hop_s": hop}
    if len(levels) >= 2:
        arr = np.asarray(levels, dtype=np.float64)
        p5, p95 = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
        out.update({
            "p5": round(p5, 2), "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(p95, 2),
            "spread_db": round(p95 - p5, 2),
            "half_spread_db": round((p95 - p5) / 2.0, 2),
            "min": round(float(arr.min()), 2), "max": round(float(arr.max()), 2),
        })
    else:
        out.update({"p5": None, "p50": None, "p95": None, "spread_db": None,
                    "half_spread_db": None, "min": None, "max": None})
    return out


# ---------------------------------------------------------------- EBU R128（对照）
_R128_RE = re.compile(
    r"Summary:.*?I:\s+(-?\d+\.?\d*)\s+LUFS.*?LRA:\s+(-?\d+\.?\d*)\s+LU.*?"
    r"Peak:\s+(-?\d+\.?\d*)\s+dBFS",
    re.S,
)


def measure_ebur128(path: str | Path, timeout_s: float = 900.0) -> Dict[str, Any]:
    """用 ffmpeg ``ebur128=peak=true`` 测 integrated LUFS / LRA / 真峰值 dBTP。"""
    p = Path(path)
    if shutil.which("ffmpeg") is None:
        return {"lufs": None, "lra": None, "tp_dbtp": None, "error": "缺 ffmpeg"}
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(p),
           "-af", "ebur128=peak=true", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"lufs": None, "lra": None, "tp_dbtp": None, "error": "ebur128 超时"}
    match = _R128_RE.search(proc.stderr.decode("utf-8", "replace"))
    if not match:
        return {"lufs": None, "lra": None, "tp_dbtp": None, "error": "ebur128 输出无法解析"}
    return {
        "lufs": round(float(match.group(1)), 2),
        "lra": round(float(match.group(2)), 2),
        "tp_dbtp": round(float(match.group(3)), 2),
        "error": "",
    }


# ---------------------------------------------------------------- 头尾边界
@dataclass
class WindowRate:
    """一个 120 s 窗口的聚合指标。"""

    t0: float
    t1: float
    duration_s: float
    chars: int = 0
    cps: float = 0.0
    speech_s: float = 0.0
    speech_ratio: float = 0.0
    teaching: bool = False
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "t0": round(self.t0, 1), "t1": round(self.t1, 1),
            "duration_s": round(self.duration_s, 1), "chars": self.chars,
            "cps": round(self.cps, 2), "speech_s": round(self.speech_s, 1),
            "speech_ratio": round(self.speech_ratio, 3),
            "teaching": self.teaching, "note": self.note,
        }


def make_windows(start: float, end: float, win_s: float) -> List[Tuple[float, float]]:
    """生成绝对时间对齐的窗口（起点为 win_s 的整数倍），便于跨文件/跨 run 对比。"""
    if end <= start or win_s <= 0:
        return []
    first = int(start // win_s) * win_s
    windows: List[Tuple[float, float]] = []
    t = float(first)
    while t < end - 1e-9:
        windows.append((t, min(t + win_s, end)))
        t += win_s
    return windows


class _ChunkTranscriber:
    """按 ``asr.segment_max_s``（默认 30 s）分块转写并缓存块级字符数。

    窗口聚合（120 s）与边界精修（30 s）共用同一份块缓存，避免重复 ASR。
    窗口边界都是 120 s 的整数倍、30 s 块从窗口起点切，二者天然对齐。
    """

    def __init__(self, x: np.ndarray, cfg: Any,
                 transcriber: "asr_mod.Transcriber", sr: int = SR) -> None:
        self.x = x
        self.cfg = cfg
        self.transcriber = transcriber
        self.sr = sr
        self.max_seg = float(cfg.get("asr.segment_max_s", 30.0))
        self.min_clip = max(0.3, float(cfg.get("asr.segment_min_s", 0.5)))
        self._cache: Dict[Tuple[float, float], int] = {}

    def _transcribe(self, t0: float, t1: float) -> int:
        if t1 - t0 < self.min_clip:
            return 0
        clip = asr_mod.slice_audio(self.x, t0, t1, self.sr)
        if clip.size == 0:
            return 0
        return asr_mod.count_speech_chars(self.transcriber.transcribe(clip, self.sr))

    def chars(self, t0: float, t1: float) -> int:
        """区间内字符数（按 30 s 块累加，块级缓存）。"""
        total = 0
        t = t0
        while t < t1 - 1e-9:
            end = min(t1, t + self.max_seg)
            key = (round(t, 3), round(end, 3))
            if key not in self._cache:
                self._cache[key] = self._transcribe(t, end)
            total += self._cache[key]
            t = end
        return total

    def cps(self, t0: float, t1: float) -> float:
        duration = t1 - t0
        return self.chars(t0, t1) / duration if duration > 0 else 0.0


def window_char_rates(
    x: np.ndarray,
    segments: Sequence[Sequence[float]],
    windows: Sequence[Tuple[float, float]],
    cfg: Any,
    transcriber: Optional["asr_mod.Transcriber"],
    sr: int = SR,
    chars: Optional["_ChunkTranscriber"] = None,
) -> List[WindowRate]:
    """逐窗口聚合：VAD 语音占比 + ASR 有效字符数/秒（**判定讲课的唯一依据**）。

    **边界判定用「窗口内 ≤30 s 分块转写」，不是「只转写 VAD 语音段」**——实测证据：
    f24 的 480–600 s 窗口若只送 VAD 语音段（覆盖 38 s）只得 206 字（cps 1.72，低于
    阈值 2.0，头边界会错到 599.5 s）；整窗 30 s 分块转写得 270 字（cps 2.25，与报告
    §2.3 的 2.26 一致，头边界 480 s 正确）。原因是 VAD 会漏掉与音乐/回声重叠的讲课声，
    而分块转写能把它捞回来——音乐段本身字符率仍很低（0–120 s 仅 0.03），不会误判。
    """
    win_s = float(cfg.get("trim.boundary_window_s", 120.0))
    min_cps = float(cfg.get("trim.boundary_min_chars_per_sec", 2.0))
    min_rate_window = win_s / 2.0

    if chars is None and transcriber is not None:
        chars = _ChunkTranscriber(x, cfg, transcriber, sr)

    rates: List[WindowRate] = []
    for t0, t1 in windows:
        duration = t1 - t0
        speech = sum(
            max(0.0, min(t1, float(s[1])) - max(t0, float(s[0])))
            for s in segments
        )
        char_count = 0
        if chars is not None and duration >= min_rate_window:
            char_count = chars.chars(t0, t1)
        rate = WindowRate(
            t0=t0, t1=t1, duration_s=duration, chars=char_count,
            speech_s=speech, speech_ratio=speech / duration if duration else 0.0,
        )
        if transcriber is None and chars is None:
            rate.note = "no_asr"
        elif duration < min_rate_window:
            rate.note = "partial_window"
        else:
            rate.cps = char_count / duration if duration else 0.0
            rate.teaching = rate.cps >= min_cps
            if not rate.teaching and rate.speech_ratio > 0.5 and rate.cps < 0.5:
                rate.note = "vad_speech_but_low_chars"  # 音乐/掌声被 VAD 误判的典型形态
        rates.append(rate)
    return rates


def _first_teaching_run(
    windows: Sequence[WindowRate],
    limit_s: float,
    need: int,
    min_cps: float,
    soft_cps: float = 1.0,
) -> Optional[int]:
    """找「讲课起点窗口」：need 个达标窗口，允许中间夹 ≤(need−1) 个「软窗口」。

    为什么允许软窗口：f06 的 120–240 s 是正常讲课（cps 2.97），但 240–360 s 只有 1.82
    （轻声/停顿多）；若要求**严格连续**达标，头边界会从 121 s 跳到 359.5 s，白切 240 s
    讲课——正是计划里要避免的「多切讲课内容」。软窗口要求 cps ≥ ``soft_cps``（默认 1.0），
    音乐/掌声窗口 cps 通常 <0.5，不会被误当成软窗口。
    """
    for i, window in enumerate(windows):
        if not (window.teaching and window.cps >= min_cps):
            continue
        if window.t0 >= limit_s - 1e-9:
            return None
        found = [i]
        k = i + 1
        # 最多向后看 (need-1) 个「软窗口」，窗口跨度上限 = 2*(need-1)
        while k < len(windows) and (k - i) <= 2 * (need - 1):
            candidate = windows[k]
            if candidate.teaching and candidate.cps >= min_cps:
                found.append(k)
                break
            if candidate.note == "partial_window" or candidate.cps < soft_cps:
                break
            k += 1
        if len(found) >= need:
            return i
    return None


def _last_teaching_run(
    windows: Sequence[WindowRate],
    limit_start: float,
    need: int,
    min_cps: float,
    soft_cps: float = 1.0,
) -> Optional[int]:
    """找「讲课终点窗口」：从尾反向找 need 个达标窗口（同样允许夹软窗口），

    并把终点向后延伸最多 (need−1) 个软窗口——片尾讲课收尾常是轻声，直接按严格窗口
    切会切掉最后一段内容（保守：宁可少裁）。
    """
    for j in range(len(windows) - 1, -1, -1):
        window = windows[j]
        if not (window.teaching and window.cps >= min_cps):
            continue
        if window.t1 <= limit_start + 1e-9:
            return None
        found = [j]
        k = j - 1
        while k >= 0 and (j - k) <= 2 * (need - 1):
            candidate = windows[k]
            if candidate.teaching and candidate.cps >= min_cps:
                found.append(k)
                break
            if candidate.note == "partial_window" or candidate.cps < soft_cps:
                break
            k -= 1
        if len(found) >= need:
            end = j
            extra = 0
            while (
                end + 1 < len(windows)
                and extra < need - 1
                and windows[end + 1].note != "partial_window"
                and windows[end + 1].cps >= soft_cps
            ):
                end += 1
                extra += 1
            return end
    return None


def _refine_head_start(
    chars: "_ChunkTranscriber", window_start: float, win_s: float,
    min_cps: float, chunk_s: float,
) -> Optional[float]:
    """把「首个讲课窗口起点」向前精修到 30 s 块粒度。

    120 s 窗口只能定位到 120 s 网格；实测 f20 的讲课从 **90 s** 就开始（90–120 s
    有 102 字、cps 3.40），而窗口级判定只能给到 119.5 s，会白切 30 s 讲课。
    这里在 [窗口起点−120 s, 窗口起点) 内按 30 s 块找**第一个 cps≥min_cps 的块**，
    找到了就把边界挪到该块起点（f24 的 360–480 s 全是 0 字块，不受影响）。
    """
    start = max(0.0, window_start - win_s)
    t = start
    while t < window_start - 1e-9:
        end = min(t + chunk_s, window_start)
        if end - t >= 10.0 and chars.cps(t, end) >= min_cps:
            return t
        t = end
    return None


def _refine_tail_end(
    chars: "_ChunkTranscriber", window_end: float, duration: float,
    min_cps: float, max_extra_s: float, chunk_s: float,
) -> float:
    """把「最后一个讲课窗口终点」向后精修到 30 s 块粒度（最长延长 ``max_extra_s``）。

    片尾常被 120 s 网格切在窗口中间：实测 f00 的 3000–3047 s 仍有 2.2–2.7 cps 的讲课
    尾声（窗口只有 47 s 被判 partial 而没转写）、f20 的 6120–6151 s 有 4.8 cps、
    f04 的 8280–8295 s 有 5.1 cps——窗口法会整段裁掉。这里按 30 s 块向后延伸，
    要求块 cps ≥ ``min_cps``（与讲课阈值 2.0 一致，只延长**明确是讲课**的尾巴；
    噪声/掌声块 cps 通常 <1，不会被误延长，f24 的 4560–4620 就属此类）。
    """
    end = window_end
    limit = min(duration, window_end + max_extra_s)
    t = end
    while t < limit - 1e-9:
        nxt = min(t + chunk_s, duration)
        if nxt - t >= 10.0 and chars.cps(t, nxt) >= min_cps:
            end = nxt
            t = nxt
            continue
        break
    return end


def _sustained_teaching_start(
    chars: "_ChunkTranscriber",
    *,
    chunk_s: float,
    max_start_s: float,
    need_blocks: int,
    min_cps: float,
    quiet_cps: float,
    min_quiet_blocks: int,
) -> Optional[Dict[str, Any]]:
    """在 ``[0, max_start_s]`` 内找「持续讲课」的起点（30 s 块级）。

    为什么需要它（2026-10-25 用户反馈：片头 0–90 s 是课前闲聊，但窗口法判成了「软窗口」
    并向前吸收，head 落到 0）：
    - 120 s 窗口的字符率会被「闲聊 + 静音」平均掉：本文件 0–120 s cps=1.81（<2.0，非讲课），
      但 120–240 s 已达 3.87 —— 窗口法于是把 0–120 s 当软窗口留着；
    - 30 s 块级能看到真实结构：``3.13, 0.87, 0.23, 3.00, 3.60, 4.00``，即「闲聊 → 空 → 持续讲课」；
    - 但**首块 3.13 cps 也不低**，所以不能简单取「第一个达标块」（那会把边界拉回 0），
      必须要求「连续 need_blocks 块都达标」（默认 3 块 = 90 s 持续讲课）才算进入讲课。

    判据（三条同时满足才返回边界，宁可不裁也不误切）：
    1. 从第 b 块起连续 ``need_blocks`` 个块的 cps ≥ ``min_cps``；
    2. ``b*chunk_s <= max_start_s``（默认 120 s；更晚的起点交给人工，不自动裁）；
    3. b 之前至少有 ``min_quiet_blocks`` 个块的 cps < ``quiet_cps``（默认 2 块）——
       证明开头确实「先空/先闲聊」，而不是一段连续但轻声的讲课（轻声讲课不会出现近空白块）。

    返回 ``{block_index, boundary_s, block_s, block_cps, quiet_blocks}``；不满足返回 None。
    """
    if chars is None or chunk_s <= 0:
        return None
    max_b = int(max(0.0, max_start_s) // chunk_s + 1e-9)
    if max_b < 0:
        return None
    block_cps: List[float] = []
    for b in range(0, max_b + need_blocks):
        t0 = b * chunk_s
        t1 = t0 + chunk_s
        block_cps.append(float(chars.cps(t0, t1)))
    for b in range(0, max_b + 1):
        window = block_cps[b:b + need_blocks]
        if len(window) < need_blocks or any(c < min_cps for c in window):
            continue
        quiet = sum(1 for c in block_cps[:b] if c < quiet_cps)
        if quiet < min_quiet_blocks:
            continue
        return {
            "block_index": b,
            "boundary_s": round(max(0.0, b * chunk_s - 0.5), 3),
            "block_s": float(chunk_s),
            "block_cps": [round(c, 2) for c in block_cps[:b + need_blocks]],
            "quiet_blocks": quiet,
        }
    return None


def auto_edge_trim(
    segments: Sequence[Sequence[float]],
    duration: float,
    head_s: float,
    tail_s: float,
    *,
    lead_min_s: float = 1.0,
    trail_min_s: float = 3.0,
    pad_s: float = 0.5,
) -> Tuple[float, float, List[str]]:
    """头尾「边缘非语音」自动切除（纯函数，P2-T5）。

    用户反馈（2026-10-25）：片头 0–7.7 s 没有语音，但 0.5–1.0 s 有一记削顶的咔哒/轰隆
    （源峰值 +3.8 dBFS），交付音频把这段噪声原样留在开头；片尾同理有收设备/房间噪声。

    规则（只动 VAD 明确判为非语音的区段）：
    - ``head_s == 0`` 且首段语音前空白 ≥ ``lead_min_s`` → ``head = 首段起点 − pad``；
      head 已经 >0 时不改（那说明窗口法/闲聊规则已经处理过开头，不能再往下推）。
    - 末段语音后空白 ≥ ``trail_min_s`` → ``tail = 末段终点 + pad``。

    返回 ``(head_s, tail_s, flags)``；没有任何改动时 flags 为空。
    """
    flags: List[str] = []
    if not segments:
        return head_s, tail_s, flags
    first_start = max(0.0, float(segments[0][0]))
    last_end = min(float(duration), float(segments[-1][1]))
    if abs(float(head_s)) <= 1e-9 and first_start - pad_s >= lead_min_s:
        head_s = round(max(0.0, min(float(duration), first_start - pad_s)), 3)
        flags.append("head_leading_nonspeech")
    if float(tail_s) >= float(duration) - 1e-6 and float(duration) - last_end >= trail_min_s:
        tail_s = round(max(head_s, min(float(duration), last_end + pad_s)), 3)
        flags.append("tail_trailing_nonspeech")
    return head_s, tail_s, flags


def detect_boundaries(
    x: np.ndarray,
    segments: Sequence[Sequence[float]],
    duration: float,
    cfg: Any,
    transcriber: Optional["asr_mod.Transcriber"] = None,
    sr: int = SR,
) -> Dict[str, Any]:
    """判定头/尾讲课边界（120 s 窗口 + ASR 字符率，允许 1 个软窗口，**带早停**）。

    返回 ``{"head": {...}, "tail": {...}, "params": {...}}``；
    没找到/无法判定时 head=0、tail=duration 并写 flag（保守：不裁）。
    """
    win_s = float(cfg.get("trim.boundary_window_s", 120.0))
    min_cps = float(cfg.get("trim.boundary_min_chars_per_sec", 2.0))
    soft_cps = float(cfg.get("trim.boundary_soft_chars_per_sec", 1.0))
    need = int(cfg.get("trim.boundary_consecutive_windows", 2))
    pad = float(cfg.get("trim.boundary_pad_s", 0.5))
    scan_s = float(cfg.get("trim.head_tail_scan_s", 900.0))
    extended_s = float(cfg.get("trim.head_tail_scan_extended_s", 1800.0))
    use_asr = bool(cfg.get("analyze.boundary_use_asr", True)) if cfg else True

    result: Dict[str, Any] = {
        "method": "window_asr" if (use_asr and transcriber is not None) else "disabled",
        "window_s": win_s,
        "min_chars_per_sec": min_cps,
        "soft_chars_per_sec": soft_cps,
        "consecutive_windows": need,
        "pad_s": pad,
        "scan_s": scan_s,
        "extended_s": extended_s,
        "asr_chunk_s": float(cfg.get("asr.segment_max_s", 30.0)),
    }
    head: Dict[str, Any] = {"boundary_s": 0.0, "flags": [], "windows": []}
    tail: Dict[str, Any] = {"boundary_s": round(duration, 3), "flags": [], "windows": []}

    if not use_asr or transcriber is None:
        head["flags"].append("asr_disabled")
        head["flags"].append("head_uncertain")
        tail["flags"].append("tail_uncertain")
        result.update({"head": head, "tail": tail})
        return result

    # ---------------- 头：从 0 往后扫，找到即停
    chunk_s = float(cfg.get("asr.segment_max_s", 30.0))
    chars = _ChunkTranscriber(x, cfg, transcriber, sr)
    head_limit = min(extended_s, duration)
    head_scan_limit = min(scan_s, duration)
    head_rates: List[WindowRate] = []
    idx: Optional[int] = None
    for t0, t1 in make_windows(0.0, head_limit, win_s):
        if t0 >= head_limit - 1e-9:
            break
        head_rates.append(
            window_char_rates(x, segments, [(t0, t1)], cfg, transcriber, sr, chars)[0]
        )
        idx = _first_teaching_run(head_rates, head_scan_limit, need, min_cps, soft_cps)
        if idx is not None:
            break
    if idx is None:
        # 15 min 内没找到 → 扩到 30 min 范围里再找一次（已算过的窗口不重复转写）
        idx = _first_teaching_run(head_rates, head_limit, need, min_cps, soft_cps)
        if idx is not None:
            head["flags"].append("head_extended")
    if idx is None:
        head["flags"].append("head_uncertain")
        head["boundary_s"] = 0.0
    else:
        # 起点向前吸收最多 (need−1) 个软窗口：课前寒暄/轻声开场 cps 常在 1.0–2.0 之间，
        # 基线（f00）要求「不裁课前内容」，宁可留着也不要切掉第一个字。
        start_idx = idx
        extra = 0
        while (
            start_idx - 1 >= 0
            and extra < need - 1
            and head_rates[start_idx - 1].note != "partial_window"
            and head_rates[start_idx - 1].cps >= soft_cps
        ):
            start_idx -= 1
            extra += 1
        window_start = head_rates[start_idx].t0
        # 30 s 块级精修（fix：120 s 网格会切掉窗口内已开始的讲课）
        refined = _refine_head_start(chars, window_start, win_s, min_cps, chunk_s)
        if refined is not None and refined < window_start - 1e-9:
            head["flags"].append("head_refined_30s")
            head["window_start_s"] = round(window_start, 3)
            head["refined_from_s"] = round(refined, 3)
            window_start = refined
        head["boundary_s"] = round(max(0.0, window_start - pad), 3)
        head["window_start_s"] = round(window_start, 3)
        head["cps"] = round(head_rates[idx].cps, 2)
        if start_idx != idx:
            head["extended_back_s"] = round(head_rates[start_idx].t0, 3)
    # 兜底：窗口法/软窗口吸收后仍是「不裁头」，再检查开头是否「先空/先闲聊、后持续讲课」
    # （用户 2026-10-25 反馈：片头 90 s 课前闲聊被留在交付音频里）。只在 head==0 时触发，
    # 不影响已有的 f20（89.5）/f06（119.5）/f24（479.5）等边界。
    if idx is not None and idx >= 1 and head["boundary_s"] <= 1e-9:
        auto = _sustained_teaching_start(
            chars,
            chunk_s=chunk_s,
            max_start_s=float(cfg.get("trim.head_auto_max_s", 120.0)),
            need_blocks=int(cfg.get("trim.head_auto_run_blocks", 3)),
            min_cps=min_cps,
            quiet_cps=soft_cps,
            min_quiet_blocks=int(cfg.get("trim.head_auto_min_quiet_blocks", 2)),
        )
        if auto is not None and auto["boundary_s"] > 1e-9:
            head["boundary_s"] = auto["boundary_s"]
            head["flags"].append("head_chatter_trim")
            head["chatter_trim"] = auto
    head["windows"] = [w.as_dict() for w in head_rates]

    # ---------------- 尾：从末尾往前扫，找到即停
    tail_span = min(extended_s, duration)
    tail_scan_start = max(0.0, duration - min(scan_s, duration))
    tail_rates: List[WindowRate] = []
    last: Optional[int] = None
    for t0, t1 in reversed(make_windows(max(0.0, duration - tail_span), duration, win_s)):
        tail_rates.insert(
            0, window_char_rates(x, segments, [(t0, t1)], cfg, transcriber, sr, chars)[0]
        )
        last = _last_teaching_run(tail_rates, tail_scan_start, need, min_cps, soft_cps)
        if last is not None:
            break
    if last is None:
        last = _last_teaching_run(
            tail_rates, max(0.0, duration - tail_span), need, min_cps, soft_cps
        )
        if last is not None:
            tail["flags"].append("tail_extended")
    if last is None:
        tail["flags"].append("tail_uncertain")
        tail["boundary_s"] = round(duration, 3)
    else:
        window_end = tail_rates[last].t1
        refined_end = _refine_tail_end(chars, window_end, duration, min_cps, win_s, chunk_s)
        if refined_end > window_end + 1e-9:
            tail["flags"].append("tail_refined_30s")
            tail["refined_from_s"] = round(window_end, 3)
            window_end = refined_end
        tail["boundary_s"] = round(min(duration, window_end + pad), 3)
        tail["window_end_s"] = round(window_end, 3)
        tail["cps"] = round(tail_rates[last].cps, 2)
    tail["windows"] = [w.as_dict() for w in tail_rates]

    result.update({"head": head, "tail": tail})
    return result


# ---------------------------------------------------------------- 人工抽听修正
def manual_review_path(cfg: Any) -> Optional[Path]:
    """``analyze.manual_review_file`` 的绝对路径；未配置返回 None。"""
    if cfg is None:
        return None
    value = cfg.get("analyze.manual_review_file", "")
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = Path(getattr(cfg, "root", Path.cwd()))
    return root / path


def load_manual_reviews(cfg: Any) -> Dict[str, Any]:
    """读取人工抽听修正表（``configs/manual_review.yaml``）。

    支持两种写法：``reviews: {<job_id 或 8 位前缀>: {...}}`` 或
    ``reviews: [{job_id: ..., head_s: ..., tail_s: ..., note: ...}]``。
    文件不存在/格式不对时返回空表（不阻塞流水线）。
    """
    path = manual_review_path(cfg)
    if path is None or not path.is_file():
        return {}
    try:
        from ..config import read_yaml

        data = read_yaml(path)
    except Exception as exc:  # pragma: no cover - 配置坏掉不该阻塞分析
        log.warning("人工修正表读取失败（忽略）：%s：%s", path, exc)
        return {}
    reviews = data.get("reviews") if isinstance(data, dict) else None
    if isinstance(reviews, list):
        out: Dict[str, Any] = {}
        for item in reviews:
            if isinstance(item, dict) and item.get("job_id"):
                out[str(item["job_id"])] = item
        return out
    return reviews if isinstance(reviews, dict) else {}


def review_fingerprint(cfg: Any) -> str:
    """人工修正表内容指纹（进了分析缓存 key，改文件自动失效）。"""
    path = manual_review_path(cfg)
    if path is None or not path.is_file():
        return ""
    try:
        import hashlib

        return hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    except OSError:  # pragma: no cover
        return ""


def _match_review(reviews: Dict[str, Any], job_id: str) -> Optional[Dict[str, Any]]:
    """按 job id 全名 / 8 位 hash 前缀 / 名字前缀匹配人工修正条目。"""
    if not reviews or not job_id:
        return None
    if job_id in reviews:
        item = reviews[job_id]
        return item if isinstance(item, dict) else None
    head = job_id[:8]
    if head in reviews:
        item = reviews[head]
        return item if isinstance(item, dict) else None
    for key, value in reviews.items():
        if isinstance(value, dict) and job_id.startswith(str(key)):
            return value
    return None


def apply_manual_review(
    trim: Dict[str, Any],
    duration: float,
    reviews: Dict[str, Any],
    job_id: str,
) -> Optional[Dict[str, Any]]:
    """把人工修正覆盖到 ``trim``（head_s/tail_s）；返回已应用的条目。

    先校验再改 ``trim``：人工表写错（越界/非数值）时**不改动任何字段**，只打 warning。
    """
    review = _match_review(reviews, job_id)
    if review is None:
        return None
    applied: Dict[str, Any] = {"note": str(review.get("note", ""))}
    new_head = float(trim["head_s"])
    new_tail = float(trim["tail_s"])
    changed = False
    for key in ("head_s", "tail_s"):
        if review.get(key) is None:
            continue
        try:
            value = float(review[key])
        except (TypeError, ValueError):
            log.warning("人工修正 %s.%s 不是数值，忽略：%r", job_id, key, review[key])
            continue
        if key == "head_s":
            new_head = value
        else:
            new_tail = value
        applied[key] = round(value, 2)
        changed = True
    if not changed:
        return None
    if not (0.0 <= new_head < duration) or not (new_head < new_tail <= duration + 1e-6):
        log.warning(
            "人工修正后边界不合法（head=%.1f tail=%.1f dur=%.1f），已忽略：%s",
            new_head, new_tail, duration, job_id,
        )
        return None
    trim["head_s"] = round(new_head, 2)
    trim["tail_s"] = round(new_tail, 2)
    trim["manual"] = applied
    trim["flags"] = sorted(set(trim.get("flags") or []) | {"manual_review"})
    log.info(
        "应用人工抽听修正：%s head=%.1fs tail=%.1fs（%s）",
        job_id, new_head, new_tail, applied["note"][:40],
    )
    return applied


# ---------------------------------------------------------------- 缓存
def _cache_path(cache_dir: str | Path, job_id: str) -> Path:
    return Path(cache_dir) / f"{job_id}{CACHE_SUFFIX}"


def _cache_params(cfg: Any, run_boundary: bool) -> Dict[str, Any]:
    return {
        "analyze_version": ANALYZE_VERSION,
        "vad.threshold": float(cfg.get("vad.threshold", 0.5)),
        "trim.boundary_window_s": float(cfg.get("trim.boundary_window_s", 120.0)),
        "trim.boundary_min_chars_per_sec": float(cfg.get("trim.boundary_min_chars_per_sec", 2.0)),
        "trim.boundary_soft_chars_per_sec": float(cfg.get("trim.boundary_soft_chars_per_sec", 1.0)),
        "levels.min_continuous_speech_s": float(cfg.get("levels.min_continuous_speech_s", 3.0)),
        "boundary_ran": bool(run_boundary),
        "manual_review": review_fingerprint(cfg),
    }


def _load_cache(path: Path, wav: Path, cfg: Any, run_boundary: bool) -> Optional[AnalyzeResult]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        return None
    try:
        stat = wav.stat()
    except OSError:
        return None
    if data.get("wav") != str(wav) or data.get("wav_bytes") != stat.st_size:
        return None
    params = data.get("params") or {}
    for key, value in _cache_params(cfg, run_boundary).items():
        if params.get(key) != value:
            return None
    try:
        return AnalyzeResult(**data["result"])
    except TypeError:  # 缓存结构变化（旧版本）→ 视为未命中
        return None


def _save_cache(path: Path, wav: Path, cfg: Any, result: AnalyzeResult,
                run_boundary: bool) -> None:
    payload = {
        "ok": True,
        "wav": str(wav),
        "wav_bytes": wav.stat().st_size,
        "params": _cache_params(cfg, run_boundary),
        "result": result.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- 主流程
def analyze_wav(
    wav_path: str | Path,
    cfg: Any,
    job_id: str = "",
    run_boundary: bool = True,
    transcriber: Optional["asr_mod.Transcriber"] = None,
    use_cache: bool = True,
    force: bool = False,
    segments: Optional[Sequence[Sequence[float]]] = None,
) -> AnalyzeResult:
    """对一个 16 kHz wav 做完整分析（VAD/电平/窗口/长静音/边界）。

    ``segments`` 可传入已算好的 VAD 段（测试用）；否则调用 Silero VAD。
    """
    wav = Path(wav_path)
    if not wav.is_file():
        raise AnalyzeError(f"16k wav 不存在：{wav}（先跑 decode）")

    cache_dir = cfg.path("cache_dir") if cfg is not None else wav.parent
    job_id = job_id or wav.stem
    cache_path = _cache_path(cache_dir, job_id)
    cache_enabled = bool(cfg.get("analyze.cache_enabled", True)) if cfg else False
    if cache_enabled and use_cache and not force:
        cached = _load_cache(cache_path, wav, cfg, run_boundary)
        if cached is not None:
            cached.from_cache = True
            log.info("分析缓存命中：%s", job_id)
            return cached

    timings: Dict[str, float] = {}
    started = time.perf_counter()

    # 1) 读 wav
    t0 = time.perf_counter()
    info = wav_info(wav)
    sr, x = read_wav_float32(wav)
    duration = len(x) / float(sr)
    timings["load_s"] = time.perf_counter() - t0

    # 2) VAD
    t0 = time.perf_counter()
    if segments is None:
        segments = detect_speech(x, sr, cfg)
    seg_list = [(float(s[0]), float(s[1])) for s in segments]
    timings["vad_s"] = time.perf_counter() - t0
    speech_s = segment_total(seg_list)
    min_cont = float(cfg.get("levels.min_continuous_speech_s", 3.0)) if cfg else 3.0
    continuous, fragments = classify_segments(seg_list, min_cont)

    # 3) 帧功率 / 掩码
    t0 = time.perf_counter()
    power, centers = frame_power(
        x, sr,
        cfg.get("levels.frame_win_ms", WIN_MS) if cfg else WIN_MS,
        cfg.get("levels.frame_hop_ms", HOP_MS) if cfg else HOP_MS,
    )
    mask_all = speech_mask(seg_list, len(power), centers)
    mask_cont = speech_mask(continuous, len(power), centers)
    timings["frames_s"] = time.perf_counter() - t0

    # 4) 电平
    t0 = time.perf_counter()
    rel_db = float(cfg.get("levels.rel_db", 15.0)) if cfg else 15.0
    asl_cont = mean_level_db(power[mask_cont])
    asl_all = mean_level_db(power[mask_all])
    asl_p56 = p56_level(power[mask_cont], rel_db=rel_db)
    noise = noise_floor_db(
        power, float(cfg.get("levels.noise_floor_percentile", 10.0)) if cfg else 10.0
    )
    percentiles = cfg.get("levels.percentiles", [10, 50, 90]) if cfg else [10, 50, 90]
    levels: Dict[str, Any] = {
        "asl_in_dbfs": round(asl_cont, 2),
        "asl_all_vad_dbfs": round(asl_all, 2),
        "asl_p56_dbfs": round(asl_p56, 2),
        "noise_floor_dbfs": round(noise, 2),
        "snr_db": round(asl_cont - noise, 2),
        "all_frames_rms_dbfs": round(mean_level_db(power), 2),
        "speech_frame_percentiles_dbfs": percentile_stats(
            power[mask_cont], [float(q) for q in percentiles]
        ),
        "asl_method": "mean_power_over_continuous_speech_frames (>=3s)",
    }
    win = window_levels(
        power, mask_cont, centers, duration,
        win_s=float(cfg.get("levels.window_s", 300.0)) if cfg else 300.0,
        min_speech_s=float(cfg.get("levels.window_min_speech_s", 30.0)) if cfg else 30.0,
        rel_db=rel_db,
    )
    levels["window5m"] = win
    levels["window5m_spread_in_db"] = win.get("spread_db")
    levels["window5m_half_spread_in_db"] = win.get("half_spread_db")
    timings["levels_s"] = time.perf_counter() - t0

    # 5) EBU R128（对照，慢；失败不阻塞）
    t0 = time.perf_counter()
    if cfg.get("analyze.ebur128", True) if cfg else False:
        r128 = measure_ebur128(wav)
        levels["lufs_in"] = r128.get("lufs")
        levels["lra_in"] = r128.get("lra")
        levels["true_peak_in_dbtp"] = r128.get("tp_dbtp")
        if r128.get("error"):
            levels["ebur128_error"] = r128["error"]
    timings["ebur128_s"] = time.perf_counter() - t0

    # 6) 头尾边界（需要 ASR；纯函数部分已可单测）
    t0 = time.perf_counter()
    boundary = detect_boundaries(
        x, seg_list, duration, cfg, transcriber if run_boundary else None, sr
    )
    timings["boundary_s"] = time.perf_counter() - t0
    head_s = float(boundary["head"]["boundary_s"])
    tail_s = float(boundary["tail"]["boundary_s"])
    head_method = "window_asr" if boundary["method"] == "window_asr" else boundary["method"]
    flags = sorted(set(boundary["head"]["flags"] + boundary["tail"]["flags"]))

    # 6.4) 边缘非语音自动切除：片头空白/咔哒（head==0 时）与片尾空白（P2-T5）
    trimmed_head, trimmed_tail, edge_flags = auto_edge_trim(
        seg_list, duration, head_s, tail_s,
        lead_min_s=float(cfg.get("trim.edge_nonspeech_min_s", 1.0)) if cfg else 1.0,
        trail_min_s=float(cfg.get("trim.tail_nonspeech_min_s", 3.0)) if cfg else 3.0,
        pad_s=float(cfg.get("trim.boundary_pad_s", 0.5)) if cfg else 0.5,
    )
    if edge_flags:
        head_s, tail_s = trimmed_head, trimmed_tail
        flags = sorted(set(flags) | set(edge_flags))
        log.info(
            "边缘非语音自动切除：%s head %.2f→%.2f tail %.2f→%.2f（%s）",
            job_id, float(boundary["head"]["boundary_s"]), head_s,
            float(boundary["tail"]["boundary_s"]), tail_s, ",".join(edge_flags),
        )

    # 6.5) 人工抽听修正（configs/manual_review.yaml；只覆盖边界，不改算法）
    trim: Dict[str, Any] = {
        "head_s": round(head_s, 2),
        "tail_s": round(tail_s, 2),
        "head_method": head_method,
        "flags": flags,
    }
    manual_reviews = load_manual_reviews(cfg)
    if manual_reviews:
        apply_manual_review(trim, duration, manual_reviews, job_id)
        head_s = float(trim["head_s"])
        tail_s = float(trim["tail_s"])

    # 7) 内部长静音：用**最终**头尾边界分类（头尾区内的空隙归头尾裁剪）
    silence = internal_silence(
        seg_list, duration,
        min_s=float(cfg.get("trim.internal_silence_min_s", 3.0)) if cfg else 3.0,
        guard=float(cfg.get("trim.internal_guard_s", 0.25)) if cfg else 0.25,
        head_s=head_s, tail_s=tail_s,
    )

    trim.update({
        "internal_cut_s": silence["cut_s"],
        "internal_gap_total_s": silence["total_s"],
    })

    vad = {
        "engine": "silero-v6",
        "params": {
            "threshold": float(cfg.get("vad.threshold", 0.5)) if cfg else 0.5,
            "min_speech_s": float(cfg.get("vad.min_speech_s", 0.25)) if cfg else 0.25,
            "min_silence_s": float(cfg.get("vad.min_silence_s", 0.30)) if cfg else 0.30,
            "pad_s": float(cfg.get("vad.pad_s", 0.10)) if cfg else 0.10,
        },
        "n_segments": len(seg_list),
        "speech_s": round(speech_s, 1),
        "speech_ratio": round(speech_s / duration, 4) if duration else 0.0,
        "continuous": {
            "n": len(continuous), "total_s": round(segment_total(continuous), 1),
            "mean_s": round(segment_total(continuous) / len(continuous), 2) if continuous else 0.0,
            "longest_s": round(max((e - s for s, e in continuous), default=0.0), 2),
        },
        "fragments": {
            "n": len(fragments), "total_s": round(segment_total(fragments), 1),
        },
    }

    timings["total_s"] = time.perf_counter() - started
    result = AnalyzeResult(
        ok=True,
        job_id=job_id,
        duration_s=duration,
        sample_rate=sr,
        vad=vad,
        levels=levels,
        trim=trim,
        internal_silence=silence,
        boundary=boundary,
        segments=[[round(s, 3), round(e, 3)] for s, e in seg_list],
        timings=timings,
    )
    if cache_enabled:
        try:
            _save_cache(cache_path, wav, cfg, result, run_boundary)
        except OSError as exc:  # pragma: no cover - 缓存写失败不影响主流程
            log.warning("分析缓存写入失败：%s", exc)

    log.info(
        "分析完成：%s dur=%.1fs speech=%.1f%% ASL=%.2fdBFS noise=%.1fdB SNR=%.1fdB "
        "head=%.1fs tail=%.1fs flags=%s（%.1fs）",
        job_id, duration, vad["speech_ratio"] * 100, levels["asl_in_dbfs"],
        levels["noise_floor_dbfs"], levels["snr_db"], head_s, tail_s,
        trim["flags"], timings["total_s"],
    )
    return result


def analyze(
    wav_path: str | Path,
    cfg: Any,
    job_id: str = "",
    run_boundary: bool = True,
    use_cache: bool = True,
    force: bool = False,
) -> AnalyzeResult:
    """对外入口：必要时加载 SenseVoice 做边界判定（ASR 不可用时降级 + flag）。"""
    transcriber = None
    need_asr = run_boundary and bool(cfg.get("analyze.boundary_use_asr", True))
    if need_asr:
        try:
            transcriber = asr_mod.get_transcriber(cfg)
        except asr_mod.AsrUnavailable as exc:
            log.warning("ASR 不可用，头尾边界退化为「不裁」：%s", exc)
            transcriber = None
    return analyze_wav(
        wav_path, cfg, job_id=job_id, run_boundary=run_boundary,
        transcriber=transcriber, use_cache=use_cache, force=force,
    )
