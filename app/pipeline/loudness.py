# -*- coding: utf-8 -*-
"""P3-T3 响度归一化：ASL 测量 + V1 静态增益 + V2 慢变增益 + 真峰值限幅。

口径（与 P2 ``analyze`` / ``docs/工程日志.md`` 锁定，不要重新发明）
--------------------------------------------------------------------
- **ASL**：16 k 轴、25 ms 窗 / 10 ms 跳的帧功率（``analyze.frame_power``），
  P.56 活跃语音电平（``analyze.p56_level``，``levels.rel_db``=15 dB），
  只统计「帧中心落在 clean 轴 ≥3 s 连续语音段（``trim.chain_clean_segments``）内」的帧。
  与 P2 ``levels.asl_p56_dbfs`` 同一口径（P2 实测 f24 = −9.55 dBFS）。
- **测量轴与渲染轴分离**：ASL 一律在 clean 44.1 k 经 ffmpeg 降采样到 16 k 后测
  （``loudness.measure.clean_sr``）；增益/限幅在 44.1 k 渲染轴上做。这样与 PoC
  （``work/audio-edit``）和 P2 的 16 k 分析轴一致，跨文件 ASL 才可比。
- **V1**：``base = clip(T − asl_in, gain_min_db, gain_max_db)``（T 默认 −20 dBFS）。
- **V2**：20 s 窗 / 5 s 跳（窗内语音帧 ≥ ``min_speech_s``=3 s 才更新），
  每控制点 ``g = clip(T − level, base ± clamp)``；3 点滑动平均 → 逐点限速
  （``max_rate_db_s × hop_s``）→ 再 clamp；到 20 ms 块中心线性插值 → 1 s 滑动平均
  → 再 clamp；块间 20 ms 余弦交叉淡化展开到样本（与 PoC ``env_from_block_gains`` 同款，
  这里向量化 + 分块，不整段展开）。
  ``drift``（源内 5 min 窗口 ASL 的 p5–p95 展宽）> ``drift_threshold_db``(6 dB)
  时 clamp 放宽到 ``high_drift_clamp_db``(6 dB)（f20 会命中）。
- **真峰值限幅**：4× 过采样测真峰（``resample_poly(x, 4, 1)``），每原样本取
  过采样 |y| 的窗口 max；``g_req = min(1, ceil_lin / tp)``；奇数窗
  ``minimum_filter1d(origin=-(w//2))`` 做 attack 前瞻（w = sr × attack_ms）；
  与 release 长度的对称滑动平均取 min（**cumsum 的 O(n) 版**，``np.convolve``
  长核会慢 7 倍，见 ``work/_proto2.py`` 注释）；输出 ``x·g``。
  ceiling 用 ``loudness.limiter.ceiling_dbfs`` = −2.75 dBTP（不是 −1 dBTP：
  实测 4x 限到 −2.75 → 渲染 wav 4x TP ≈ −2.65、ebur128 TP ≈ −2.70，
  给 AAC 编码过冲留余量；样本峰限幅实测会到 +0.8 dBTP，禁止使用）。
- **ASL 校正迭代**：限幅会压低语音电平（f00 这类满刻度瞬态源第一轮掉 ~1.5 dB），
  渲染后重新测 asl_out，偏差 > ``measure.asl_tolerance_db``(0.03) 就把
  ``gain_offset_db += T − asl_out`` 重渲染，最多 ``measure.max_asl_iterations``(3) 次。
  迭代时 V2 包络整体随 base 平移（见 ``v2_control_points`` 的说明），否则 V2 会把
  偏移吃掉、asl_out 不收敛。

实测依据（``work/_proto2.py``，Lead 在 f00 10 min 切片上验证）：ceiling −2.75、
4× 过采样，600 s 切片限幅 3.6 s；渲染 wav 4× TP −2.65 / ebur128 −2.70 / ASL 可控。
分块（``loudness.render.chunk_s``=120 s + ``context_s``=1 s 前后各留 context 再丢弃）
是硬需求：f04 clean 有 3.58 亿样本，整段 4× float64 要 11 GB。

所有中间文件（clean16 / out16）放 ``paths.cache_dir`` 下的临时目录，函数结束删除；
不动 ``work/cache`` 里已有的 16k / analyze / 44k 缓存。
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d
from scipy.signal import resample_poly

from ..logging_setup import get_logger
from ..models import GainInfo
from .analyze import frame_power, measure_ebur128, p56_level, speech_mask, window_levels

log = get_logger(__name__)

#: 分析/测量轴采样率（与 P2 analyze 一致，不是渲染轴）
SR16 = 16000
#: 渲染轴采样率（锁定：44.1 kHz mono float32）
SR44 = 44100

#: V2 块级包络粒度 = 20 ms（= ``loudness.v2.crossfade_ms`` 默认）
BLOCK_MS = 20.0
#: ``analyze.frame_power`` 的默认帧跳（10 ms）——按帧下标换算时间只能用这个值
FRAME_HOP_S = 0.01
#: 限幅器判「触发」的增益阈值（与 PoC 一致：g < 0.999 视为限幅介入）
_TRIGGER_G = 0.999
#: 削波判定阈值（与 PoC / trim 测试一致）
_CLIP_LEVEL = 0.999


class LoudnessError(RuntimeError):
    """响度阶段失败（wav 缺失/非法、语音区间为空、ffmpeg 不可用、ASL 非有限等）。"""


# ---------------------------------------------------------------- 配置读取
def _get(cfg: Any, dotted: str, default: Any) -> Any:
    """``cfg.get`` 的容错版：``cfg`` 为 ``None`` 或抛异常时给默认值。"""
    if cfg is None:
        return default
    try:
        return cfg.get(dotted, default)
    except Exception:  # pragma: no cover - 配置对象异常时不该阻塞渲染
        return default


def _as_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise LoudnessError(f"{name} 不是数值：{value!r}") from exc
    if not math.isfinite(out):
        raise LoudnessError(f"{name} 不是有限值：{value!r}")
    return out


def _as_int(value: Any, name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise LoudnessError(f"{name} 不是整数：{value!r}") from exc
    return out


@dataclass
class _Params:
    """一次渲染用的全部锁定参数（从 cfg 读一次，循环里不再碰 cfg）。"""

    target_db: float = -20.0
    gain_min_db: float = -24.0
    gain_max_db: float = 24.0
    clean_sr: int = SR16
    rel_db: float = 15.0
    window_s: float = 300.0
    window_min_speech_s: float = 30.0
    v2_enabled: bool = True
    win_s: float = 20.0
    hop_s: float = 5.0
    clamp_db: float = 3.0
    high_drift_clamp_db: float = 6.0
    drift_threshold_db: float = 6.0
    min_speech_s: float = 3.0
    max_rate_db_s: float = 1.5
    block_ms: float = BLOCK_MS
    smooth_s: float = 1.0
    limiter_target_tp_dbtp: float = -1.0
    limiter_ceiling_dbtp: float = -2.75
    limiter_oversample: int = 4
    limiter_attack_ms: float = 5.0
    limiter_release_ms: float = 150.0
    limiter_max_iterations: int = 3
    asl_tolerance_db: float = 0.03
    max_asl_iterations: int = 3
    cross_file_tolerance_db: float = 0.5
    chunk_s: float = 120.0
    context_s: float = 1.0


def _params_from_cfg(cfg: Any) -> _Params:
    """从配置读锁定参数（缺键用 default.yaml 同值兜底，不静默改行为）。"""
    default = _Params()
    params = _Params(
        target_db=_as_float(_get(cfg, "loudness.target_asl_dbfs", default.target_db),
                            "loudness.target_asl_dbfs"),
        gain_min_db=_as_float(_get(cfg, "loudness.gain_min_db", default.gain_min_db),
                              "loudness.gain_min_db"),
        gain_max_db=_as_float(_get(cfg, "loudness.gain_max_db", default.gain_max_db),
                              "loudness.gain_max_db"),
        clean_sr=_as_int(_get(cfg, "loudness.measure.clean_sr", default.clean_sr),
                         "loudness.measure.clean_sr"),
        rel_db=_as_float(_get(cfg, "levels.rel_db", default.rel_db), "levels.rel_db"),
        window_s=_as_float(_get(cfg, "levels.window_s", default.window_s), "levels.window_s"),
        window_min_speech_s=_as_float(
            _get(cfg, "levels.window_min_speech_s", default.window_min_speech_s),
            "levels.window_min_speech_s",
        ),
        v2_enabled=bool(_get(cfg, "loudness.v2.enabled", default.v2_enabled)),
        win_s=_as_float(_get(cfg, "loudness.v2.win_s", default.win_s), "loudness.v2.win_s"),
        hop_s=_as_float(_get(cfg, "loudness.v2.hop_s", default.hop_s), "loudness.v2.hop_s"),
        clamp_db=_as_float(_get(cfg, "loudness.v2.clamp_db", default.clamp_db),
                           "loudness.v2.clamp_db"),
        high_drift_clamp_db=_as_float(
            _get(cfg, "loudness.v2.high_drift_clamp_db", default.high_drift_clamp_db),
            "loudness.v2.high_drift_clamp_db",
        ),
        drift_threshold_db=_as_float(
            _get(cfg, "loudness.v2.drift_threshold_db", default.drift_threshold_db),
            "loudness.v2.drift_threshold_db",
        ),
        min_speech_s=_as_float(_get(cfg, "loudness.v2.min_speech_s", default.min_speech_s),
                               "loudness.v2.min_speech_s"),
        max_rate_db_s=_as_float(_get(cfg, "loudness.v2.max_rate_db_s", default.max_rate_db_s),
                                "loudness.v2.max_rate_db_s"),
        block_ms=_as_float(_get(cfg, "loudness.v2.crossfade_ms", default.block_ms),
                           "loudness.v2.crossfade_ms"),
        smooth_s=_as_float(_get(cfg, "loudness.v2.smooth_s", default.smooth_s),
                           "loudness.v2.smooth_s"),
        limiter_target_tp_dbtp=_as_float(
            _get(cfg, "loudness.limiter.target_tp_dbtp", default.limiter_target_tp_dbtp),
            "loudness.limiter.target_tp_dbtp",
        ),
        limiter_ceiling_dbtp=_as_float(
            _get(cfg, "loudness.limiter.ceiling_dbfs", default.limiter_ceiling_dbtp),
            "loudness.limiter.ceiling_dbfs",
        ),
        limiter_oversample=_as_int(
            _get(cfg, "loudness.limiter.oversample", default.limiter_oversample),
            "loudness.limiter.oversample",
        ),
        limiter_attack_ms=_as_float(
            _get(cfg, "loudness.limiter.attack_ms", default.limiter_attack_ms),
            "loudness.limiter.attack_ms",
        ),
        limiter_release_ms=_as_float(
            _get(cfg, "loudness.limiter.release_ms", default.limiter_release_ms),
            "loudness.limiter.release_ms",
        ),
        limiter_max_iterations=_as_int(
            _get(cfg, "loudness.limiter.max_iterations", default.limiter_max_iterations),
            "loudness.limiter.max_iterations",
        ),
        asl_tolerance_db=_as_float(
            _get(cfg, "loudness.measure.asl_tolerance_db", default.asl_tolerance_db),
            "loudness.measure.asl_tolerance_db",
        ),
        max_asl_iterations=_as_int(
            _get(cfg, "loudness.measure.max_asl_iterations", default.max_asl_iterations),
            "loudness.measure.max_asl_iterations",
        ),
        cross_file_tolerance_db=_as_float(
            _get(cfg, "levels.cross_file_tolerance_db", default.cross_file_tolerance_db),
            "levels.cross_file_tolerance_db",
        ),
        chunk_s=_as_float(_get(cfg, "loudness.render.chunk_s", default.chunk_s),
                          "loudness.render.chunk_s"),
        context_s=_as_float(_get(cfg, "loudness.render.context_s", default.context_s),
                            "loudness.render.context_s"),
    )
    # 基本自洽校验（配置坏掉时尽早失败，别产出错音频）
    if params.gain_min_db >= params.gain_max_db:
        raise LoudnessError("loudness.gain_min_db 必须小于 gain_max_db")
    if params.clean_sr <= 0:
        raise LoudnessError("loudness.measure.clean_sr 必须为正")
    if params.hop_s <= 0 or params.win_s <= 0:
        raise LoudnessError("loudness.v2.hop_s / win_s 必须为正")
    if params.clamp_db < 0 or params.high_drift_clamp_db < params.clamp_db:
        raise LoudnessError("loudness.v2 clamp 配置非法（high_drift_clamp_db < clamp_db）")
    if params.max_rate_db_s <= 0:
        raise LoudnessError("loudness.v2.max_rate_db_s 必须为正")
    if params.limiter_oversample < 1:
        raise LoudnessError("loudness.limiter.oversample 必须 ≥1")
    if params.chunk_s <= 0 or params.context_s < 0:
        raise LoudnessError("loudness.render.chunk_s / context_s 非法")
    if params.limiter_max_iterations < 0:
        raise LoudnessError("loudness.limiter.max_iterations 不能为负")
    if params.block_ms <= 0 or params.smooth_s < 0:
        raise LoudnessError("loudness.v2.crossfade_ms / smooth_s 非法")
    return params


# ---------------------------------------------------------------- 小工具
def _db(linear: float, eps: float = 1e-12) -> float:
    """线性幅度 → dB（0 附近夹到 −240 dB，避免 log(0)）。"""
    return 20.0 * math.log10(max(float(linear), eps))


def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    """O(n) 对称滑动平均（edge padding）。

    与 ``np.convolve(pad, ones(k)/k, "valid")`` 数值等价，但长核（release 150 ms
    ≈ 6615 样本）快 7 倍：``work/_proto2.py`` 实测结论，这里沿用同一实现。
    """
    arr = np.asarray(x, dtype=np.float64)
    if k <= 1 or arr.size == 0:
        return arr
    lo, hi = k // 2, k - 1 - k // 2
    padded = np.pad(arr, (lo, hi), mode="edge")
    cumsum = np.concatenate(([0.0], np.cumsum(padded)))
    return (cumsum[k:] - cumsum[:-k]) / float(k)


def _count_events(mask: np.ndarray) -> int:
    """布尔序列里连续 True 的段数（限幅触发「事件」计数）。"""
    if mask.size == 0:
        return 0
    edges = np.count_nonzero(np.diff(mask.astype(np.int8)) == 1)
    return int(edges + (1 if bool(mask[0]) else 0))


def _norm_intervals(intervals: Sequence[Sequence[float]] | None) -> List[Tuple[float, float]]:
    """校验并规范化语音区间 ``[[start, end], ...]``（丢掉空区间，非法值报错）。"""
    out: List[Tuple[float, float]] = []
    for index, item in enumerate(intervals or []):
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise LoudnessError(f"语音区间[{index}] 非法：{item!r}") from exc
        if not (math.isfinite(start) and math.isfinite(end)):
            raise LoudnessError(f"语音区间[{index}] 含非有限值：{item!r}")
        if end > start:
            out.append((start, end))
    return out


# ---------------------------------------------------------------- wav 读写
def _wav_open(path: str | Path) -> Tuple[int, np.ndarray]:
    """打开 wav：float32 优先走 memmap（百分钟文件不整段入内存）；其它格式转 float32。

    mono 单声道是流水线前提；万一多声道取第 0 声道并 warning（不静默改口径）。
    """
    p = Path(path)
    try:
        sr, data = wavfile.read(str(p), mmap=True)
    except (ValueError, OSError):
        try:
            sr, data = wavfile.read(str(p))
        except (ValueError, OSError) as exc:
            raise LoudnessError(f"读 wav 失败：{p}：{exc}") from exc
    if data.ndim > 1:
        log.warning("wav 是多声道，取第 0 声道：%s shape=%s", p.name, data.shape)
        data = data[:, 0]
    if data.dtype == np.float32:
        return int(sr), data
    if data.dtype == np.float64:
        return int(sr), data.astype(np.float32)
    if data.dtype == np.int16:
        return int(sr), data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return int(sr), data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return int(sr), (data.astype(np.float32) - 128.0) / 128.0
    return int(sr), data.astype(np.float32)


def _wav_close(data: Any) -> None:
    """关闭 scipy memmap（Windows 上不关会占住文件，临时目录删不掉）。"""
    mm = getattr(data, "_mmap", None)
    if mm is not None:
        try:
            mm.close()
        except Exception:  # pragma: no cover - 关闭失败不影响结果
            pass


def _write_wav_f32_header(fh: Any, sample_rate: int, n_frames: int) -> None:
    """写 44 字节 float32 单声道 WAV 头（与 ``trim._write_wav_f32_header`` 同格式）。"""
    data_bytes = int(n_frames) * 4
    block_align = 4
    byte_rate = int(sample_rate) * block_align
    fh.write(b"RIFF")
    fh.write(struct.pack("<I", 36 + data_bytes))
    fh.write(b"WAVE")
    fh.write(b"fmt ")
    fh.write(struct.pack("<IHHIIHH", 16, 3, 1, int(sample_rate), byte_rate, block_align, 32))
    fh.write(b"data")
    fh.write(struct.pack("<I", data_bytes))


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise LoudnessError("PATH 中缺 ffmpeg")
    return exe


def downsample_wav(src: str | Path, dst: str | Path, sr: int = SR16) -> Path:
    """用 ffmpeg 把 wav 降采样成单声道 float32（``-ac 1 -ar sr -c:a pcm_f32le``）。

    与 P2/解码链同一路子（ffmpeg swresample），保证 ASL 测量轴一致。
    """
    src_path, dst_path = Path(src), Path(dst)
    if not src_path.is_file():
        raise LoudnessError(f"降采样源不存在：{src_path}")
    _require_ffmpeg()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-i", str(src_path), "-ac", "1", "-ar", str(int(sr)),
        "-c:a", "pcm_f32le", str(dst_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except OSError as exc:  # pragma: no cover - ffmpeg 存在但起不来
        raise LoudnessError(f"调用 ffmpeg 失败：{exc}") from exc
    if proc.returncode != 0:
        raise LoudnessError(
            f"ffmpeg 降采样失败（{proc.returncode}）：{proc.stderr.decode('utf-8', 'replace')[-500:]}"
        )
    if not dst_path.is_file() or dst_path.stat().st_size <= 44:
        raise LoudnessError(f"ffmpeg 没有产出有效 wav：{dst_path}")
    return dst_path


# ---------------------------------------------------------------- ASL 测量
def speech_mask_from_intervals(
    intervals: Sequence[Sequence[float]], n_frames: int, centers_s: np.ndarray
) -> np.ndarray:
    """帧级布尔掩码：帧中心落在任一语音区间 ``[start, end)`` 内。"""
    clean = _norm_intervals(intervals)
    return speech_mask(clean, int(n_frames), np.asarray(centers_s, dtype=np.float64))


def _asl_from_power(power: np.ndarray, mask: np.ndarray, rel_db: float) -> float:
    """P.56 ASL：对掩码内帧功率做 ``p56_level``；无帧返回 NaN。"""
    power = np.asarray(power, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = min(power.size, mask.size)
    selected = power[:n][mask[:n]]
    if selected.size == 0:
        return float("nan")
    return p56_level(selected, rel_db=float(rel_db))


def measure_asl(
    x16: np.ndarray, speech_intervals: Sequence[Sequence[float]], cfg: Any = None
) -> float:
    """16 k 音频 + clean 轴语音区间 → P.56 ASL（dBFS）。

    与 P2 ``levels.asl_p56_dbfs`` 同口径：25 ms 窗 / 10 ms 跳帧功率，只统计
    中心落在语音区间内的帧，``rel_db=levels.rel_db``(15 dB)。
    """
    params = _params_from_cfg(cfg) if cfg is not None else _Params()
    x = np.asarray(x16, dtype=np.float32)
    if x.size == 0:
        return float("nan")
    power, centers = frame_power(x, params.clean_sr)
    mask = speech_mask_from_intervals(speech_intervals, len(power), centers)
    return _asl_from_power(power, mask, params.rel_db)


def _window5m(
    power: np.ndarray,
    mask: np.ndarray,
    centers: np.ndarray,
    duration_s: float,
    params: _Params,
) -> Dict[str, Any]:
    """5 min 窗口 ASL 分布（复用 ``analyze.window_levels``，P2 同口径）。"""
    return window_levels(
        power,
        mask,
        centers,
        duration_s,
        win_s=params.window_s,
        hop_s=params.window_s,
        min_speech_s=params.window_min_speech_s,
        rel_db=params.rel_db,
    )


# ---------------------------------------------------------------- 真峰值限幅
def _tp_envelope_4x(x: np.ndarray, oversample: int = 4) -> np.ndarray:
    """每个原样本的真峰值包络：过采样 |y| 在「本样本对应的 oversample+2 点窗」内取 max。

    移植 ``work/_proto2.py:tp_envelope_4x``（Lead 已在 f00 实测过）：窗口
    ``oversample+2``、头部补 1 点后按 oversample 步进，正好覆盖每个原样本的
    过采样点以及左右各半个点，避免真峰落在采样点之间被漏掉。
    """
    x64 = np.asarray(x, dtype=np.float64)
    n = int(x64.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    y = np.abs(resample_poly(x64, int(oversample), 1))
    padded = np.concatenate(([y[0]], y, [y[-1]]))
    window = np.lib.stride_tricks.sliding_window_view(padded, int(oversample) + 2)
    env = window[0:: int(oversample)][:n].max(axis=1)
    return env.astype(np.float32)


def _limit_core(
    x: np.ndarray,
    sr: int,
    ceiling_dbtp: float,
    *,
    oversample: int,
    attack_ms: float,
    release_ms: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """单块真峰值限幅核心（返回限幅后样本、逐样本线性增益、统计）。"""
    x = np.asarray(x, dtype=np.float32)
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32), {
            "min_gain_db": 0.0, "mean_gain_db": 0.0,
            "trigger_ratio": 0.0, "trigger_samples": 0, "trigger_events": 0,
        }
    ceiling_linear = 10.0 ** (float(ceiling_dbtp) / 20.0)
    tp = _tp_envelope_4x(x, oversample)
    # g_req=1 表示这一样本的真峰已经低于 ceiling，不需要压
    g_req = np.minimum(1.0, ceiling_linear / np.maximum(tp, 1e-9))

    # attack 前瞻：窗口奇数，origin=-(w//2) 让滤波器看向未来（限幅必须提前压）
    w = max(1, int(round(sr * float(attack_ms) / 1000.0)))
    if w % 2 == 0:
        w += 1
    g_future = minimum_filter1d(g_req, size=w, mode="nearest", origin=-(w // 2))

    # release：对称滑动平均（平滑还原），与前瞻取 min 保证两边的约束
    r = max(1, int(round(sr * float(release_ms) / 1000.0)))
    g_slow = _moving_avg(g_req, r)

    g = np.minimum(g_future, g_slow).astype(np.float32)
    y = (x * g).astype(np.float32)

    trigger = g < _TRIGGER_G
    stats = {
        "min_gain_db": round(_db(float(g.min())), 3),
        "mean_gain_db": round(_db(float(g.mean())), 3),
        "trigger_ratio": round(float(np.mean(trigger)), 6) if n else 0.0,
        "trigger_samples": int(np.count_nonzero(trigger)),
        "trigger_events": _count_events(trigger),
    }
    return y, g, stats


def _tp_max_region(y: np.ndarray, start: int, stop: int, oversample: int = 4) -> float:
    """``y[start:stop)`` 区间的 4× 过采样真峰值（线性幅度）。

    先把整段 ``y`` 过采样、再只取区间对应的过采样点：这样区间内每个点的滤波
    邻域都是真实样本，不受 resample 边缘补零影响（分块渲染时校验块中心区域用）。
    """
    y64 = np.asarray(y, dtype=np.float64)
    n = int(y64.size)
    if n == 0:
        return 0.0
    a = max(0, int(start))
    b = min(n, int(stop))
    if b <= a:
        return 0.0
    y4 = np.abs(resample_poly(y64, int(oversample), 1))
    return float(y4[a * oversample: b * oversample].max())


def true_peak_limit(
    x: np.ndarray,
    sr: int,
    ceiling_dbtp: float,
    *,
    oversample: int = 4,
    attack_ms: float = 5.0,
    release_ms: float = 150.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """真峰值限幅（4× 过采样测峰 → 前瞻 + 释放包络 → 乘增益），带 ceiling 收敛。

    **禁止用采样峰限幅**：PoC 实测样本峰限幅只到 +0.8 dBTP，不可接受。
    限幅是对时变增益的乘法，输出在采样点之间的真峰会比 ceiling 略微过冲
    （实测正弦 0.0 dB、语音类 ~0.1 dB、满刻度白噪 ~1.1 dB）。这里在函数内部
    回读输出的 4× TP，若超过 ``ceiling + 0.10 dB`` 就把 ceiling 压低
    ``过冲 + 0.05`` 重做（最多 3 次），保证返回值满足 ``tp ≤ ceiling + 0.15``
    （验收断言），``stats`` 里给出 ``ceiling_effective_dbtp`` / ``tp_measured_dbtp``。

    返回 ``(y_float32, stats)``。
    """
    sr = _as_int(sr, "sr")
    if sr <= 0:
        raise LoudnessError(f"非法采样率：{sr}")
    oversample = _as_int(oversample, "oversample")
    if oversample < 1:
        raise LoudnessError(f"非法过采样倍数：{oversample}")
    attack_ms = _as_float(attack_ms, "attack_ms")
    release_ms = _as_float(release_ms, "release_ms")
    if attack_ms < 0 or release_ms < 0:
        raise LoudnessError("attack_ms / release_ms 不能为负")
    ceiling_dbtp = _as_float(ceiling_dbtp, "ceiling_dbtp")

    source = np.asarray(x, dtype=np.float32)
    n = int(source.size)
    # 回读容差：验收要求 tp ≤ ceiling+0.15，这里按 +0.10 收（留 0.05 余量）
    tp_tolerance_db = 0.10
    ceiling_now = float(ceiling_dbtp)
    y = source
    stats: Dict[str, Any] = {}
    passes = 0
    tp_measured = 0.0
    for passes in range(1, 4):
        y, _g, stats = _limit_core(
            source, sr, ceiling_now,
            oversample=oversample, attack_ms=attack_ms, release_ms=release_ms,
        )
        tp_measured = _db(_tp_max_region(y, 0, n, oversample))
        if tp_measured <= ceiling_dbtp + tp_tolerance_db or passes == 3:
            break
        ceiling_now -= (tp_measured - ceiling_dbtp) + 0.05

    out = dict(stats)
    out.update({
        "ceiling_dbtp": round(float(ceiling_dbtp), 3),
        "ceiling_effective_dbtp": round(ceiling_now, 3),
        "tp_measured_dbtp": round(tp_measured, 3),
        "tp_iterations": passes,
        "oversample": oversample,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
    })
    return y, out


def measure_true_peak_wav(
    path: str | Path,
    *,
    oversample: int = 4,
    chunk_s: float = 60.0,
    context_s: float = 0.2,
) -> float:
    """分块测 wav 的 4× 过采样真峰值（dBTP）；整段 4× float64 会爆内存。

    每块前后各留 ``context_s`` 只取块中心对应的过采样点（避免 resample 边缘效应），
    全文件取 max。
    """
    sr, data = _wav_open(path)
    try:
        n = int(data.size)
        if n == 0:
            return float("nan")
        chunk = max(1, int(round(chunk_s * sr)))
        ctx = max(0, int(round(context_s * sr)))
        peak = 0.0
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            lo = max(0, start - ctx)
            hi = min(n, end + ctx)
            y = np.abs(resample_poly(np.asarray(data[lo:hi], dtype=np.float64), oversample, 1))
            seg = y[(start - lo) * oversample: (end - lo) * oversample]
            if seg.size:
                peak = max(peak, float(seg.max()))
        return _db(peak)
    finally:
        _wav_close(data)


# ---------------------------------------------------------------- 分块渲染
def _sample_env(
    block_db: np.ndarray, start: int, stop: int, sr: int, block_ms: float = BLOCK_MS
) -> np.ndarray:
    """把 20 ms 块级增益（dB）展开成样本级线性包络 ``[start, stop)``。

    块内从前一块增益余弦交叉淡化到本块（与 PoC ``env_from_block_gains`` 同款，
    这里向量化，且支持任意区间——分块渲染时只需要当前块及其 context）。
    """
    block = max(1, int(round(sr * float(block_ms) / 1000.0)))
    n_block = int(np.asarray(block_db).size)
    if stop <= start or n_block == 0:
        return np.ones(0, dtype=np.float32)
    i0 = min(int(start) // block, n_block - 1)
    i1 = min(n_block - 1, (int(stop) - 1) // block)
    seg = np.asarray(block_db[i0: i1 + 1], dtype=np.float64)
    prev = np.empty_like(seg)
    prev[0] = block_db[i0 - 1] if i0 > 0 else block_db[i0]
    prev[1:] = seg[:-1]
    ramp = np.tile(
        0.5 - 0.5 * np.cos(np.pi * (np.arange(block) + 0.5) / block), seg.size
    )
    env_db = np.repeat(prev, block) + np.repeat(seg - prev, block) * ramp
    offset = int(start) - i0 * block
    env_db = env_db[offset: offset + (int(stop) - int(start))]
    if env_db.size < int(stop) - int(start):  # pragma: no cover - block_db 覆盖整段时不会发生
        env_db = np.pad(env_db, (0, int(stop) - int(start) - env_db.size), mode="edge")
    return np.power(10.0, env_db / 20.0).astype(np.float32)


def _render_pass(
    src: np.ndarray,
    sr: int,
    dst: Path,
    block_db: np.ndarray,
    params: _Params,
    ceiling_dbtp: float,
) -> Dict[str, Any]:
    """一整遍渲染：分块施加增益包络 → 真峰值限幅 → 写 float32 wav。

    分块（``params.chunk_s``）前后各留 ``params.context_s`` 样本做 context，
    限幅后只写回块中心区域：前瞻/释放/滤波都在真实邻域内计算，块边界不留痕。
    每块顺带回读块中心区域的 4× TP（``tp_4x_dbtp``），供上层判断真峰值是否达标。
    """
    n = int(src.size)
    if n <= 0:
        raise LoudnessError("渲染源为空")
    block = max(1, int(round(sr * params.block_ms / 1000.0)))
    if int(np.asarray(block_db).size) < int(math.ceil(n / block)):
        raise LoudnessError("块级增益包络长度不足（内部错误）")
    chunk = max(1, int(round(params.chunk_s * sr)))
    context = max(0, int(round(params.context_s * sr)))
    started = time.perf_counter()
    dst.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = 0
    core_samples = 0
    trigger_samples = 0
    trigger_events = 0
    clipped = 0
    min_gain = 1.0
    gain_sum = 0.0
    tp_peak = 0.0
    with open(dst, "wb") as fh:
        _write_wav_f32_header(fh, sr, n)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            lo = max(0, start - context)
            hi = min(n, end + context)
            pre = np.asarray(src[lo:hi], dtype=np.float32) * _sample_env(
                block_db, lo, hi, sr, params.block_ms
            )
            y_ext, g_ext, _stats = _limit_core(
                pre, sr, ceiling_dbtp,
                oversample=params.limiter_oversample,
                attack_ms=params.limiter_attack_ms,
                release_ms=params.limiter_release_ms,
            )
            core_offset = start - lo
            core_n = end - start
            core = y_ext[core_offset: core_offset + core_n]
            g_core = g_ext[core_offset: core_offset + core_n]
            fh.write(np.ascontiguousarray(core, dtype="<f4").tobytes())

            core_samples += core_n
            min_gain = min(min_gain, float(g_core.min()))
            gain_sum += float(np.sum(g_core, dtype=np.float64))
            trigger = g_core < _TRIGGER_G
            trigger_samples += int(np.count_nonzero(trigger))
            trigger_events += _count_events(trigger)
            clipped += int(np.count_nonzero(np.abs(core) >= _CLIP_LEVEL))
            # 块中心区域的 4× TP（context 只用来算增益，不参与统计，避免块边界误判）
            tp_peak = max(
                tp_peak,
                _tp_max_region(y_ext, core_offset, core_offset + core_n,
                               params.limiter_oversample),
            )
            n_chunks += 1

    if core_samples != n:  # pragma: no cover - 分块逻辑自检
        raise LoudnessError(f"渲染写出样本数不一致：{core_samples} != {n}")
    elapsed = time.perf_counter() - started
    return {
        "samples": n,
        "sr": sr,
        "n_chunks": n_chunks,
        "chunk_s": params.chunk_s,
        "context_s": params.context_s,
        "ceiling_dbtp": round(float(ceiling_dbtp), 3),
        "trigger_ratio": round(trigger_samples / max(1, core_samples), 6),
        "trigger_samples": trigger_samples,
        "trigger_events": trigger_events,
        "min_gain_db": round(_db(min_gain), 3),
        "mean_gain_db": round(_db(gain_sum / max(1, core_samples)), 3),
        "clipped_samples": clipped,
        "tp_4x_dbtp": round(_db(tp_peak), 3),
        "elapsed_s": round(elapsed, 3),
    }


def limit_wav(
    src_wav: str | Path,
    dst_wav: str | Path,
    ceiling_dbtp: float,
    cfg: Any = None,
    *,
    chunk_s: Optional[float] = None,
) -> Dict[str, Any]:
    """对整段 44.1 k wav 做真峰值限幅（分块，不整段入内存），返回统计。

    独立工具函数：只限幅、不加增益。``chunk_s`` 覆盖 ``loudness.render.chunk_s``
    （测试里用小 chunk 验证分块等价性）。与 :func:`true_peak_limit` 一样回读 4× TP，
    超过 ``ceiling + 0.10 dB`` 就整段下调 ceiling 重做（最多 3 次）——**整段同一个
    ceiling**，避免分块之间出现增益台阶。
    """
    src, dst = Path(src_wav), Path(dst_wav)
    if not src.is_file():
        raise LoudnessError(f"限幅源不存在：{src}")
    params = _params_from_cfg(cfg)
    if chunk_s is not None:
        params = replace(params, chunk_s=_as_float(chunk_s, "chunk_s"))
    nominal = _as_float(ceiling_dbtp, "ceiling_dbtp")
    sr, data = _wav_open(src)
    try:
        n = int(data.size)
        if n == 0:
            raise LoudnessError(f"限幅源为空：{src}")
        block = max(1, int(round(sr * params.block_ms / 1000.0)))
        n_block = int(math.ceil(n / block))
        env_db = np.zeros(n_block, dtype=np.float64)  # 全 0 dB → 样本包络恒为 1.0
        ceiling_now = nominal
        stats: Dict[str, Any] = {}
        attempts = 1
        for attempt in range(1, 4):
            attempts = attempt
            stats = _render_pass(data, sr, dst, env_db, params, ceiling_now)
            tp_now = float(stats.get("tp_4x_dbtp", float("-inf")))
            if tp_now <= nominal + 0.10 or attempt == 3:
                break
            ceiling_now = round(ceiling_now - (tp_now - nominal) - 0.05, 3)
    finally:
        _wav_close(data)
    stats["ok"] = True
    stats["ceiling_dbtp"] = round(nominal, 3)
    stats["ceiling_effective_dbtp"] = round(ceiling_now, 3)
    stats["tp_iterations"] = attempts
    return stats


# ---------------------------------------------------------------- V2 慢变增益
def _ref_level_from_power(power: np.ndarray, mask: np.ndarray, rel_db: float) -> float:
    """全段语音帧的 P.56 电平（= asl_in，同一口径；V2 控制点增益的相对基准）。"""
    return _asl_from_power(power, mask, rel_db)


def v2_control_points(
    power16: np.ndarray,
    centers16: np.ndarray,
    mask16: np.ndarray,
    duration_s: float,
    base_gain_db: float,
    clamp_db: float,
    cfg: Any = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """V2 慢变增益的控制点 ``(times_s, gains_db, info)``。

    20 s 窗 / 5 s 跳；窗内语音帧 ≥ ``min_speech_s`` 才更新（否则跳过，避免音乐段
    把增益拉飞）；控制点增益 ``clip(base + (ref − level), base ± clamp)``，其中
    ``ref`` 是全段语音帧的 P.56 电平（= asl_in）。首次渲染时 ``base = T − asl_in``，
    所以与锁定式 ``clip(T − level, base ± clamp)`` 严格等价；写成相对式是为了
    ASL 校正迭代时 base 平移能整体带动包络（否则 V2 会把偏移吃掉、asl_out 不收敛）。

    之后：3 点滑动平均 → 逐点限速 ``max_rate_db_s × hop_s`` → 再 clamp。
    可更新窗口 <2 个时退化 static（返回 ``[0, duration]`` 两点同值，``info.degraded``）。
    """
    params = _params_from_cfg(cfg) if cfg is not None else _Params()
    power = np.asarray(power16, dtype=np.float64)
    mask = np.asarray(mask16, dtype=bool)
    centers = np.asarray(centers16, dtype=np.float64)
    n = min(power.size, mask.size)
    duration = _as_float(duration_s, "duration_s")
    if duration <= 0:
        raise LoudnessError(f"渲染时长非法：{duration}")
    base = _as_float(base_gain_db, "base_gain_db")
    clamp = abs(_as_float(clamp_db, "clamp_db"))

    ref_level = _ref_level_from_power(power[:n], mask[:n], params.rel_db)
    times: List[float] = []
    gains: List[float] = []
    n_total = 0
    t = 0.0
    while t < duration - 1e-9:
        n_total += 1
        end = min(duration, t + params.win_s)
        a = min(int(round(t / FRAME_HOP_S)), n)
        b = min(int(round(end / FRAME_HOP_S)), n)
        if b > a:
            selected = mask[a:b]
            if int(np.count_nonzero(selected)) * FRAME_HOP_S >= params.min_speech_s - 1e-9:
                level = p56_level(power[a:b][selected], rel_db=params.rel_db)
                if math.isfinite(level) and math.isfinite(ref_level):
                    times.append((t + end) / 2.0)
                    gains.append(float(np.clip(base + (ref_level - level),
                                               base - clamp, base + clamp)))
        t += params.hop_s

    info: Dict[str, Any] = {
        "win_s": params.win_s,
        "hop_s": params.hop_s,
        "clamp_db": round(clamp, 2),
        "min_speech_s": params.min_speech_s,
        "max_rate_db_s": params.max_rate_db_s,
        "n_windows_total": n_total,
        "n_windows_used": len(times),
        "ref_level_db": round(ref_level, 3) if math.isfinite(ref_level) else None,
        "degraded": False,
    }
    if len(times) < 2:
        info.update({
            "degraded": True,
            "gain_min_db": round(base, 2),
            "gain_max_db": round(base, 2),
            "rate_control_db_s": 0.0,
        })
        log.info("V2 可用窗口不足（%d 个），退化为静态增益 %.2f dB", len(times), base)
        return (
            np.array([0.0, duration], dtype=np.float64),
            np.array([base, base], dtype=np.float64),
            info,
        )

    ctrl_times = np.asarray(times, dtype=np.float64)
    ctrl_gains = np.asarray(gains, dtype=np.float64)

    # 1) 3 点滑动平均（edge padding；控制点 <3 时不动）
    if ctrl_gains.size >= 3:
        ctrl_gains = _moving_avg(ctrl_gains, 3)

    # 2) 逐点限速（控制点级：max_rate_db_s × hop_s）
    max_delta = params.max_rate_db_s * params.hop_s
    limited = ctrl_gains.copy()
    for i in range(1, limited.size):
        limited[i] = float(np.clip(
            limited[i], limited[i - 1] - max_delta, limited[i - 1] + max_delta
        ))

    # 3) 再压回 ±clamp
    limited = np.clip(limited, base - clamp, base + clamp)
    rate = (
        float(np.max(np.abs(np.diff(limited)))) / params.hop_s if limited.size > 1 else 0.0
    )
    info.update({
        "gain_min_db": round(float(limited.min()), 2),
        "gain_max_db": round(float(limited.max()), 2),
        "rate_control_db_s": round(rate, 3),
        "control_times_s": [round(float(v), 2) for v in ctrl_times[:8]],
    })
    return ctrl_times, limited, info


def block_gain_envelope(
    ctrl_times: np.ndarray,
    ctrl_gains_db: np.ndarray,
    duration_s: float,
    base_gain_db: float,
    clamp_db: float,
    cfg: Any = None,
) -> np.ndarray:
    """控制点 → 20 ms 块级增益（dB）：块中心线性插值 → 1 s 滑动平均 → clamp。

    返回长度 ``ceil(duration / 0.02)`` 的块级 dB 数组；样本级展开由
    :func:`_sample_env` 做（分块、余弦交叉淡化）。
    """
    params = _params_from_cfg(cfg) if cfg is not None else _Params()
    duration = _as_float(duration_s, "duration_s")
    if duration <= 0:
        raise LoudnessError(f"渲染时长非法：{duration}")
    base = _as_float(base_gain_db, "base_gain_db")
    clamp = abs(_as_float(clamp_db, "clamp_db"))
    block_s = params.block_ms / 1000.0
    n_block = int(math.ceil(duration / block_s - 1e-9))
    if n_block <= 0:
        raise LoudnessError(f"块级包络长度非法：{n_block}")
    centers = (np.arange(n_block, dtype=np.float64) + 0.5) * block_s
    times = np.asarray(ctrl_times, dtype=np.float64)
    gains = np.asarray(ctrl_gains_db, dtype=np.float64)
    if times.size == 0 or gains.size == 0:
        envelope = np.full(n_block, base, dtype=np.float64)
    elif times.size == 1:
        envelope = np.full(n_block, float(gains[0]), dtype=np.float64)
    else:
        n = min(times.size, gains.size)
        envelope = np.interp(centers, times[:n], gains[:n], left=gains[0], right=gains[n - 1])
    if params.smooth_s > 0:
        k = max(1, int(round(params.smooth_s / block_s)))
        envelope = _moving_avg(envelope, k)
    return np.clip(envelope, base - clamp, base + clamp)


# ---------------------------------------------------------------- 结果对象
@dataclass
class LoudnessResult:
    """``render()`` 的返回值（契约见任务卡；P5 export 直接取字段写 report/edit）。"""

    ok: bool = True
    #: 传入的 P2 原始 ASL（对照用，可为 None）
    asl_analysis_dbfs: Optional[float] = None
    #: clean16k（降噪+裁剪后）P56 ASL
    asl_in_dbfs: float = float("nan")
    #: 渲染 wav 降采样 16k 后 P56 ASL
    asl_out_dbfs: float = float("nan")
    asl_dev_db: float = float("nan")
    #: V1 静态增益（含 gain_offset，已 clamp）
    gain_db: float = 0.0
    #: app/models.py 的 GainInfo → edit.json.gain
    gain: GainInfo = field(default_factory=GainInfo)
    v2: Dict[str, Any] = field(default_factory=dict)
    limiter: Dict[str, Any] = field(default_factory=dict)
    #: 渲染 wav 的 ebur128 真峰值（dBTP）
    tp_wav_dbtp: float = float("nan")
    lufs_out: float = float("nan")
    window5m_spread_out_db: Optional[float] = None
    clipped_samples: int = 0
    #: ASL 校正重渲染次数
    iterations: int = 0
    timings: Dict[str, float] = field(default_factory=dict)
    wav_out: str = ""
    error: str = ""

    def report_levels_out(self) -> Dict[str, Any]:
        """``report.json.levels`` 的 out 段（契约 §4.1）。"""

        def _r(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            return round(out, 2) if math.isfinite(out) else None

        return {
            "asl_out_dbfs": _r(self.asl_out_dbfs),
            "lufs_out": _r(self.lufs_out),
            "window5m_spread_out_db": _r(self.window5m_spread_out_db),
        }


# ---------------------------------------------------------------- 主入口
def _cache_dir(cfg: Any) -> Path:
    try:
        path = cfg.path("cache_dir")
    except Exception:  # pragma: no cover - 配置异常时兜底到 work/cache
        path = Path(__file__).resolve().parents[2] / "work" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _move_into_place(src: Path, dst: Path) -> None:
    """把临时输出搬到最终路径（同盘 rename，跨盘复制，覆盖旧文件）。"""
    try:
        os.replace(str(src), str(dst))
    except OSError:
        shutil.move(str(src), str(dst))


def _render_loop(
    x44: np.ndarray,
    sr44: int,
    tmp_out: Path,
    tmp_out16: Path,
    block_db: np.ndarray,
    params: _Params,
    ceiling_dbtp: float,
    intervals: List[Tuple[float, float]],
    result: LoudnessResult,
    t_target: float,
    asl_in: float,
    base_static: float,
    offset: float,
) -> Dict[str, Any]:
    """渲染 + ASL 校正迭代 + 真峰值回读迭代；返回最后一遍的统计与测量结果。

    两条闭环合成一个循环（都是「重渲染一次」的代价）：
    - ASL 偏差 > ``measure.asl_tolerance_db`` → 把差值补到增益上重渲染；
    - 4× TP 回读 > ``limiter.target_tp_dbtp + 0.15`` → 下调 ceiling 重渲染
      （整段同一个 ceiling，避免分块之间出现增益台阶）。
    """
    passes = 0
    render_stats: Dict[str, Any] = {}
    asl_out = float("nan")
    dev = float("nan")
    power_out = np.zeros(0, dtype=np.float64)
    mask_out = np.zeros(0, dtype=bool)
    centers_out = np.zeros(0, dtype=np.float64)
    base_now = float(np.clip(t_target - asl_in + offset, params.gain_min_db, params.gain_max_db))
    duration_out = 0.0
    ceiling_now = float(ceiling_dbtp)
    max_passes = 1 + max(params.max_asl_iterations, params.limiter_max_iterations)
    tp_ok = True

    while True:
        passes += 1
        base_now = float(np.clip(
            t_target - asl_in + offset, params.gain_min_db, params.gain_max_db
        ))
        block_show = block_db + (base_now - base_static)
        started = time.perf_counter()
        render_stats = _render_pass(x44, sr44, tmp_out, block_show, params, ceiling_now)
        result.timings[f"render_{passes}_s"] = round(time.perf_counter() - started, 2)

        started = time.perf_counter()
        downsample_wav(tmp_out, tmp_out16, params.clean_sr)
        sr_out, data_out = _wav_open(tmp_out16)
        try:
            duration_out = data_out.size / float(sr_out)
            power_out, centers_out = frame_power(data_out, params.clean_sr)
        finally:
            _wav_close(data_out)
        mask_out = speech_mask_from_intervals(intervals, len(power_out), centers_out)
        asl_out = _asl_from_power(power_out, mask_out, params.rel_db)
        result.timings[f"measure_out_{passes}_s"] = round(time.perf_counter() - started, 2)
        if not math.isfinite(asl_out):
            raise LoudnessError("asl_out 非有限值（输出语音帧为空？）")
        dev = asl_out - t_target
        tp_now = render_stats.get("tp_4x_dbtp")
        tp_ok = (
            tp_now is None
            or not math.isfinite(float(tp_now))
            or float(tp_now) <= params.limiter_target_tp_dbtp + 0.15
        )
        log.info(
            "响度渲染 pass %d（上限 %d）：base=%+.2f dB ceiling=%.2f asl_out=%.3f dBFS "
            "dev=%+.3f dB tp4x=%s（渲染 %.1fs / 测量 %.1fs）",
            passes, max_passes, base_now, ceiling_now, asl_out, dev,
            "%.2f" % tp_now if tp_now is not None else "n/a",
            render_stats["elapsed_s"], result.timings[f"measure_out_{passes}_s"],
        )
        asl_ok = abs(dev) <= params.asl_tolerance_db
        if (asl_ok and tp_ok) or passes >= max_passes:
            break
        if not tp_ok:
            # 限幅后的真峰仍超硬指标：整段压低 ceiling 重渲染（不逐块压低，避免台阶）
            backoff = (float(tp_now) - params.limiter_target_tp_dbtp) + 0.10
            ceiling_now = round(ceiling_now - backoff, 3)
            log.warning(
                "真峰值 %.2f dBTP 超目标 %.2f dBTP，ceiling 下调 %.2f → %.2f dBTP 重渲染",
                float(tp_now), params.limiter_target_tp_dbtp, backoff, ceiling_now,
            )
        else:
            # 限幅压掉了语音电平：把差值补到增益上重渲染（下一轮 V2 包络整体平移）
            next_offset = offset + t_target - asl_out
            next_base = float(np.clip(
                t_target - asl_in + next_offset, params.gain_min_db, params.gain_max_db
            ))
            if abs(next_base - base_now) < 1e-6:
                # 已经顶到 ±24 dB 的 gain clamp，再迭代也不会变，别空转
                log.warning(
                    "ASL 偏差 %+.2f dB 但静态增益已顶到 clamp（%.2f dB），停止迭代",
                    dev, base_now,
                )
                break
            offset = next_offset

    return {
        "render_stats": render_stats,
        "asl_out": asl_out,
        "dev": dev,
        "base_now": base_now,
        "power_out": power_out,
        "mask_out": mask_out,
        "centers_out": centers_out,
        "duration_out": duration_out,
        "passes": passes,
        "ceiling_used": ceiling_now,
        "tp_ok": tp_ok,
    }


def render(
    clean44_wav: str | Path,
    speech_intervals_clean: Sequence[Sequence[float]],
    cfg: Any,
    *,
    out44_wav: str | Path,
    ceiling_dbtp: Optional[float] = None,
    gain_offset_db: float = 0.0,
    drift_db: Optional[float] = None,
    analysis_asl_in: Optional[float] = None,
) -> LoudnessResult:
    """P3-T3 响度归一化：clean 44.1 k wav → 归一化 + 限幅后的 44.1 k wav。

    参数
    ----
    clean44_wav
        降噪 + 裁剪后的 44.1 k float32 单声道 wav（P3-T1/T2 产出）。
    speech_intervals_clean
        '''clean 轴'''的语音区间（``trim.chain_clean_segments`` 的输出，≥3 s 连续段）。
        ASL 只统计帧中心落在这些区间内的帧；同时也作为 V2 的语音掩码。
    cfg
        已加载配置（读 ``loudness.*`` / ``levels.*``）。
    out44_wav
        渲染输出路径（float32 44.1 k mono），先写临时文件、成功后才落盘。
    ceiling_dbtp
        真峰值 ceiling，None 时用 ``loudness.limiter.ceiling_dbfs``（−2.75）。
    gain_offset_db
        初始静态增益偏移（外部修正用）；ASL 迭代会在此基础上继续累加。
    drift_db
        源内 5 min 窗口 ASL 的 p5–p95 展宽；None 时用 clean16 现算。
        > ``loudness.v2.drift_threshold_db`` 时 clamp 放宽到 ``high_drift_clamp_db``。
    analysis_asl_in
        P2 原始 ASL（对照，写入 ``asl_analysis_dbfs``）。

    返回 :class:`LoudnessResult`。失败抛 :class:`LoudnessError`（输入/环境问题），
    不静默产出坏音频。
    """
    started = time.perf_counter()
    if cfg is None:
        raise LoudnessError("缺少配置对象 cfg")
    clean44 = Path(clean44_wav)
    out44 = Path(out44_wav)
    if not clean44.is_file():
        raise LoudnessError(f"clean 44.1k wav 不存在：{clean44}")
    params = _params_from_cfg(cfg)
    intervals = _norm_intervals(speech_intervals_clean)
    if not intervals:
        raise LoudnessError("语音区间为空，无法测量/归一化 ASL")

    t_target = params.target_db
    ceiling = params.limiter_ceiling_dbtp if ceiling_dbtp is None else _as_float(
        ceiling_dbtp, "ceiling_dbtp"
    )
    if ceiling > params.limiter_target_tp_dbtp:
        log.warning(
            "ceiling_dbtp=%.2f 高于目标真峰值 %.2f dBTP，输出可能不达标",
            ceiling, params.limiter_target_tp_dbtp,
        )
    offset = _as_float(gain_offset_db, "gain_offset_db")

    result = LoudnessResult(
        asl_analysis_dbfs=None if analysis_asl_in is None else round(float(analysis_asl_in), 3),
        wav_out=str(out44),
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="loudness_", dir=str(_cache_dir(cfg))))
    out44.parent.mkdir(parents=True, exist_ok=True)
    try:
        # ---------------------------------------------------------- 1) 16k 测量轴
        clean16 = tmp_dir / "clean.16k.wav"
        started_step = time.perf_counter()
        downsample_wav(clean44, clean16, params.clean_sr)
        result.timings["downsample_in_s"] = round(time.perf_counter() - started_step, 2)

        started_step = time.perf_counter()
        sr16, data16 = _wav_open(clean16)
        try:
            if sr16 != params.clean_sr:
                raise LoudnessError(f"clean16 采样率异常：{sr16} != {params.clean_sr}")
            duration = data16.size / float(sr16)
            if duration <= 0:
                raise LoudnessError("clean16 为空")
            power16, centers16 = frame_power(data16, sr16)
        finally:
            _wav_close(data16)
        mask16 = speech_mask_from_intervals(intervals, len(power16), centers16)
        if not bool(mask16.any()):
            raise LoudnessError(
                "语音区间内没有任何帧（intervals 是否在 clean 轴上？区间超出 clean 时长？）"
            )
        asl_in = _asl_from_power(power16, mask16, params.rel_db)
        if not math.isfinite(asl_in):
            raise LoudnessError("asl_in 非有限值")
        result.asl_in_dbfs = round(asl_in, 3)
        result.timings["asl_in_s"] = round(time.perf_counter() - started_step, 2)

        # ---------------------------------------------------------- 2) drift / clamp
        started_step = time.perf_counter()
        window5m_in: Dict[str, Any] = {}
        if drift_db is None:
            window5m_in = _window5m(power16, mask16, centers16, duration, params)
            drift_value = window5m_in.get("spread_db")
            drift_source = "clean16"
        else:
            drift_value = _as_float(drift_db, "drift_db")
            drift_source = "input"
        drift = float(drift_value) if drift_value is not None else 0.0
        high_drift = drift > params.drift_threshold_db
        clamp_db = params.high_drift_clamp_db if high_drift else params.clamp_db
        result.timings["drift_s"] = round(time.perf_counter() - started_step, 2)

        # ---------------------------------------------------------- 3) V1 + V2 包络
        started_step = time.perf_counter()
        base_static = float(np.clip(t_target - asl_in, params.gain_min_db, params.gain_max_db))
        if abs((t_target - asl_in) - base_static) > 1e-6:
            log.warning(
                "静态增益 %.2f dB 被 clamp 到 [%.1f, %.1f]：%.2f dB",
                t_target - asl_in, params.gain_min_db, params.gain_max_db, base_static,
            )
        if params.v2_enabled:
            ctrl_times, ctrl_gains, v2_info = v2_control_points(
                power16, centers16, mask16, duration, base_static, clamp_db, cfg
            )
            block_db = block_gain_envelope(
                ctrl_times, ctrl_gains, duration, base_static, clamp_db, cfg
            )
        else:
            n_block = int(math.ceil(duration / (params.block_ms / 1000.0) - 1e-9))
            block_db = np.full(n_block, base_static, dtype=np.float64)
            v2_info = {
                "enabled": False,
                "degraded": True,
                "win_s": params.win_s,
                "hop_s": params.hop_s,
                "clamp_db": round(clamp_db, 2),
                "n_windows_total": 0,
                "n_windows_used": 0,
                "gain_min_db": round(base_static, 2),
                "gain_max_db": round(base_static, 2),
                "rate_control_db_s": 0.0,
            }
        v2_info.update({
            "enabled": bool(params.v2_enabled),
            "drift_in_db": round(drift, 2),
            "drift_source": drift_source,
            "drift_threshold_db": params.drift_threshold_db,
            "high_drift": bool(high_drift),
            "clamp_db_used": round(clamp_db, 2),
            "n_block": int(block_db.size),
            "block_ms": params.block_ms,
            "window5m_in": window5m_in,
        })
        result.v2 = v2_info
        result.timings["v2_s"] = round(time.perf_counter() - started_step, 3)

        # ---------------------------------------------------------- 4) 渲染 + ASL 迭代
        sr44, data44 = _wav_open(clean44)
        try:
            if sr44 != SR44:
                raise LoudnessError(f"渲染轴必须 {SR44} Hz mono，实际 {sr44} Hz：{clean44}")
            tmp_out = tmp_dir / "out.44k.wav"
            tmp_out16 = tmp_dir / "out.16k.wav"
            loop = _render_loop(
                data44, sr44, tmp_out, tmp_out16, block_db, params, ceiling,
                intervals, result, t_target, asl_in, base_static, offset,
            )
        finally:
            _wav_close(data44)

        render_stats = loop["render_stats"]
        asl_out = loop["asl_out"]
        dev = loop["dev"]
        base_now = loop["base_now"]
        power_out = loop["power_out"]
        mask_out = loop["mask_out"]
        centers_out = loop["centers_out"]
        duration_out = loop["duration_out"]
        passes = loop["passes"]
        ceiling_used = loop["ceiling_used"]
        tp_ok = loop["tp_ok"]
        tp_4x = render_stats.get("tp_4x_dbtp")

        result.iterations = max(0, int(passes) - 1)
        result.asl_out_dbfs = round(asl_out, 3)
        result.asl_dev_db = round(dev, 3)
        result.clipped_samples = int(render_stats.get("clipped_samples", 0))

        # ---------------------------------------------------------- 5) 输出侧统计
        started_step = time.perf_counter()
        window5m_out = _window5m(power_out, mask_out, centers_out, duration_out, params)
        result.window5m_spread_out_db = window5m_out.get("spread_db")
        v2_info["window5m_out"] = window5m_out
        result.timings["window5m_out_s"] = round(time.perf_counter() - started_step, 2)

        started_step = time.perf_counter()
        r128 = measure_ebur128(tmp_out)
        result.tp_wav_dbtp = (
            float(r128["tp_dbtp"]) if r128.get("tp_dbtp") is not None else float("nan")
        )
        result.lufs_out = (
            float(r128["lufs"]) if r128.get("lufs") is not None else float("nan")
        )
        result.timings["ebur128_s"] = round(time.perf_counter() - started_step, 2)
        if r128.get("error"):
            log.warning("ebur128 测量异常：%s", r128["error"])

        # ---------------------------------------------------------- 6) 汇总
        gain_min_block = float(np.min(block_db)) + (base_now - base_static)
        gain_max_block = float(np.max(block_db)) + (base_now - base_static)
        max_rate = (
            float(np.max(np.abs(np.diff(block_db)))) / (params.block_ms / 1000.0)
            if block_db.size > 1 else 0.0
        )
        v2_info["gain_min_db"] = round(
            max(gain_min_block, params.gain_min_db), 2
        )
        v2_info["gain_max_db"] = round(
            min(gain_max_block, params.gain_max_db), 2
        )
        v2_info["max_rate_db_s"] = round(max_rate, 3)
        v2_info["gain_mean_db"] = round(
            _db(float(np.mean(np.power(10.0, (block_db + (base_now - base_static)) / 20.0)))), 2
        )
        if v2_info["max_rate_db_s"] > params.max_rate_db_s + 1e-6:
            log.warning(
                "V2 实际增益变化率 %.2f dB/s 超过上限 %.2f dB/s",
                v2_info["max_rate_db_s"], params.max_rate_db_s,
            )

        limiter_type = "v2" if (params.v2_enabled and not v2_info.get("degraded")) else "static"
        result.gain_db = round(base_now, 3)
        result.gain = GainInfo(
            type=limiter_type,
            gain_db=base_now,
            min_db=float(v2_info["gain_min_db"]),
            max_db=float(v2_info["gain_max_db"]),
            max_rate_db_s=float(v2_info["max_rate_db_s"]),
            limiter_triggers=int(render_stats.get("trigger_events", 0)),
        )
        result.limiter = {
            "ceiling_dbtp": round(float(ceiling), 3),
            "ceiling_effective_dbtp": round(float(ceiling_used), 3),
            "oversample": params.limiter_oversample,
            "attack_ms": params.limiter_attack_ms,
            "release_ms": params.limiter_release_ms,
            "iterations": passes,
            "trigger_ratio": float(render_stats.get("trigger_ratio", 0.0)),
            "trigger_events": int(render_stats.get("trigger_events", 0)),
            "min_gain_db": float(render_stats.get("min_gain_db", 0.0)),
            "mean_gain_db": float(render_stats.get("mean_gain_db", 0.0)),
            "clipped_samples": result.clipped_samples,
            "tp_4x_dbtp": round(float(tp_4x), 2) if tp_4x is not None else None,
            "n_chunks": int(render_stats.get("n_chunks", 0)),
            "chunk_s": params.chunk_s,
            "context_s": params.context_s,
            "elapsed_s": float(render_stats.get("elapsed_s", 0.0)),
            "target_tp_dbtp": params.limiter_target_tp_dbtp,
            "tp_ok": bool(tp_ok),
        }

        _move_into_place(tmp_out, out44)
        result.wav_out = str(out44)

        if abs(dev) > params.cross_file_tolerance_db:
            result.ok = False
            result.error = (
                f"ASL 未收敛：asl_out={asl_out:.2f} dBFS 目标 {t_target:.2f}，"
                f"偏差 {dev:+.2f} dB（限幅压得太狠？ceiling={ceiling_used:.2f} dBTP）"
            )
            log.warning("%s", result.error)
        elif not tp_ok:
            result.ok = False
            result.error = (
                f"真峰值未达标：tp_4x={tp_4x} dBTP 目标 {params.limiter_target_tp_dbtp}"
                f"（ceiling_effective={ceiling_used:.2f} dBTP）"
            )
            log.warning("%s", result.error)
        elif result.clipped_samples > 0:
            result.ok = False
            result.error = f"输出有 {result.clipped_samples} 个削波样本（≥{_CLIP_LEVEL}）"
            log.warning("%s", result.error)
        result.timings["total_s"] = round(time.perf_counter() - started, 2)
        log.info(
            "响度归一化完成：%s asl_in=%.2f → asl_out=%.2f（dev %+.2f dB）"
            " gain=%.2f dB [%.2f, %.2f] rate=%.2f dB/s TP4x=%.2f dBTP ebur128 TP=%.2f"
            " clipped=%d 迭代=%d 耗时 %.1fs",
            out44.name, result.asl_in_dbfs, result.asl_out_dbfs, result.asl_dev_db,
            result.gain_db, result.gain.min_db, result.gain.max_db,
            result.gain.max_rate_db_s,
            result.limiter["tp_4x_dbtp"] if result.limiter["tp_4x_dbtp"] is not None
            else float("nan"),
            result.tp_wav_dbtp, result.clipped_samples, result.iterations,
            result.timings["total_s"],
        )
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


__all__ = [
    "BLOCK_MS",
    "LoudnessError",
    "LoudnessResult",
    "SR16",
    "SR44",
    "block_gain_envelope",
    "downsample_wav",
    "limit_wav",
    "measure_asl",
    "measure_true_peak_wav",
    "render",
    "speech_mask_from_intervals",
    "true_peak_limit",
    "v2_control_points",
]
