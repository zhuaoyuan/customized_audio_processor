# -*- coding: utf-8 -*-
"""说话人标注（P4-T3）：默认单标签「主讲」，抽样确认多人后才全量 diar。

设计口径（与 docs/实施计划.md P4-T3、docs/调研报告_需求对齐版.md §6/§7.4 一致）
----------------------------------------------------------------------------------
1. **默认 ``diar.mode=none``**：只把 ``transcript.json`` 的每个段标成 ``主讲``，
   ``report.json`` 读-改-写 ``"speakers": {"n": 1, "method": "none"}``；**不改** text/flags。
   本批 6 文件都是单人讲课，none 就是生产默认。
2. **``sampled``**：从 clean 轴正文（避开头尾各 ``body_margin_s``）按总预算
   ``diar.sample_minutes`` 均分取 ``diar.sample_segments`` 个窗口，跑
   sherpa-onnx pyannote segmentation + cam++ embedding + FastClustering(threshold)。
   同一窗口内的原始簇 id 由 sherpa 给出；**不同窗口的簇 id 不可比**，因此再按段级
   cam++ embedding 的时长加权质心做 average-linkage 余弦归并（``merge_threshold``，
   默认 0.7）——即 PoC ``step_merge.py`` 的二阶段归并。
   只有「稳定说话人」（每人 ``>= min_speaker_s`` 且 ``>= min_speaker_ratio`` 语音）
   达到 **>= 2 人** 才升 full；否则保持单标签（抽样证据写进 report，便于复核）。
3. **``full``**：对整段 clean16k 跑同一个 diar 流水（同阈值/同归并），最长簇 = ``主讲``，
   其余满足稳定条件的簇编 ``其他1/其他2…``；映射到 transcript 段时取「重叠时长最大」的簇，
   重叠 < ``min_overlap_ratio``（默认 0.30）或命中碎片簇 → 保守标 ``主讲`` 并加
   ``uncertain_speaker`` flag（``app/models.py``）。
4. **绝不臆造发言人**：模型缺失/失败 → 不改 transcript，全部保持 ``主讲``，返回
   ``{"ok": False, "stage": "diarize", "error": ...}``；sampled 判定 n=1 时也绝不生成
   「其他N」标签。

采样预算的读法
--------------
``diar.sample_minutes`` 解释为**采样总分钟数**（各窗口合计），``diar.sample_segments``
是窗口个数；默认 10 min / 3 段 → 每窗 200 s（与前期 PoC 的 600 s 单窗总预算同量级）。
P4-T3 任务卡里「3 个 5 min 片段」用 ``sample_minutes=15, sample_segments=3`` 复现
（work/diar_p4_check.json 记录实际窗口）。

模型（``paths.models.diar_dir``，默认 ``work/diar/models/``）
------------------------------------------------------------
- ``sherpa-onnx-pyannote-segmentation-3-0/model.onnx``
- ``3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx``
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..logging_setup import get_logger
from .finalize import resolve_product

log = get_logger(__name__)

# ---------------------------------------------------------------- 常量
#: 分析与 diar 统一的采样率
SR = 16000

#: 主讲人标签（与 app/models.py SPEAKER_MAIN 一致；此处不 import models 避免耦合）
SPEAKER_MAIN = "主讲"
OTHER_PREFIX = "其他"

#: report.json 里标记本模块口径的版本号（skip 判定用，改口径时递增）
DIAR_VERSION = "p4-t3-v1"

#: clean16k 缓存文件名（transcribe-dev 的 asr.decode_clean16k 也用这个名字；
#: 本模块读取时兼容 float32/int16 两种 wav，写入时用 float32 与 P4-T1 契约一致）
CLEAN16K_SUFFIX = ".clean16k.wav"

#: 二阶段归并时，每条 diar 段取 embedding 的最短/最长时长（秒）
EMBED_MIN_S = 0.3
EMBED_MAX_S = 12.0

#: 单条 diar 段太短（无 embedding、也无法安全归并）→ 并回主讲
FRAGMENT_FALLBACK_S = 1.0


class DiarError(RuntimeError):
    """说话人分离流程错误（输入缺失、配置非法等）。"""


class DiarUnavailable(DiarError):
    """sherpa-onnx 或 diar 模型文件不可用。"""


# ---------------------------------------------------------------- 配置小工具
def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """``cfg.get`` 容错读取（None/缺配置时返回 default）。"""
    if cfg is None:
        return default
    try:
        value = cfg.get(key, default)
    except Exception:  # pragma: no cover - 防御式
        return default
    return default if value is None else value


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    value = _cfg_get(cfg, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(cfg: Any, key: str, default: int) -> int:
    value = _cfg_get(cfg, key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def num_threads_for_diar(cfg: Any) -> int:
    """diar 使用的 CPU 线程数：``diar.num_threads``（缺省 4）。

    与 ASR 的 8 线程错开，避免批量时两阶段同时抢满 CPU（P3 实测进程池/线程争用会变慢）。
    """
    threads = _cfg_int(cfg, "diar.num_threads", 4)
    return max(1, min(16, threads))


# ---------------------------------------------------------------- 纯函数：路径/模型
def diar_model_paths(cfg: Any) -> Tuple[Path, Path]:
    """返回 ``(segmentation_model, embedding_model)`` 绝对路径。"""
    try:
        diar_dir = Path(cfg.model_path("diar_dir"))
    except Exception as exc:  # pragma: no cover - 配置缺 diar_dir
        raise DiarUnavailable(f"配置缺少 paths.models.diar_dir：{exc}") from exc
    seg = diar_dir / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    emb = diar_dir / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
    return seg, emb


def diar_available(cfg: Any) -> bool:
    """模型文件是否就绪（不加载模型）。"""
    try:
        seg, emb = diar_model_paths(cfg)
    except DiarUnavailable:
        return False
    return seg.is_file() and emb.is_file()


# ---------------------------------------------------------------- 纯函数：采样窗口
def plan_sample_windows(
    clean_duration_s: float,
    *,
    sample_minutes: float = 10.0,
    sample_segments: int = 3,
    body_margin_s: float = 120.0,
) -> List[Tuple[float, float]]:
    """把「正文」按 ``sample_segments`` 等分，在每份中心取一个采样窗口。

    参数
    ----
    clean_duration_s : clean 音轨时长（秒）
    sample_minutes   : **总**采样分钟数（所有窗口合计）
    sample_segments  : 窗口个数（>=1）
    body_margin_s    : 正文判定：跳过 clean 轴开头/结尾各这么多秒（本批文件 head 已裁掉
                       课前闲聊，2 min 余量足够避开片头残留音乐/片尾收场）

    返回 ``[(start_s, end_s), ...]``，按时间升序、互不重叠、落在正文内；
    音频太短（正文 < 60 s）时退化为整段采样；非法输入返回 ``[]``。
    """
    duration = float(clean_duration_s)
    if not math.isfinite(duration) or duration <= 0:
        return []
    n = max(1, int(sample_segments))
    total_s = max(0.0, float(sample_minutes)) * 60.0
    margin = max(0.0, float(body_margin_s))

    body_start, body_end = margin, duration - margin
    if body_end - body_start < 60.0:  # 太短：不做正文裁剪
        body_start, body_end = 0.0, duration
    if body_end - body_start < 1.0:
        return []

    slice_len = (body_end - body_start) / n
    per = min(total_s / n, slice_len) if total_s > 0 else slice_len
    if per < 1.0:
        return []

    windows: List[Tuple[float, float]] = []
    for i in range(n):
        center = body_start + (i + 0.5) * slice_len
        start = center - per / 2.0
        end = center + per / 2.0
        if start < body_start:
            start, end = body_start, body_start + per
        if end > body_end:
            start, end = body_end - per, body_end
        windows.append((round(max(0.0, start), 3), round(min(duration, end), 3)))
    return windows


# ---------------------------------------------------------------- 纯函数：归并 / 判定
def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化（零向量保持为零）。"""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


def merge_cluster_centroids(
    centroids: np.ndarray, threshold: float = 0.7
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """簇质心 average-linkage 余弦归并（``step_merge.py`` 的纯 numpy 版）。

    距离 = 1 − 余弦相似度；两簇相似度取**所有跨簇质心对的平均**：
    ``sim(i,j) = (Σ_{a∈i} c_a)·(Σ_{b∈j} c_b) / (n_i·n_j)``
    （质心先做 L2 归一化）。反复合并「平均余弦最大且 >= threshold」的一对，直到没有
    可合并对；相似度并列时取簇编号最小的一对，保证**同输入同结果**。

    参数
    ----
    centroids : shape ``(K, D)``，第 k 行是原始簇 k 的质心（内部会再归一化）
    threshold : 余弦阈值（越大越不容易合并）

    返回 ``(labels, history)``：``labels`` 长度 K，取值 0..M-1（按首次出现顺序）；
    ``history`` 是每次合并的 ``{"i","j","sim","labels"}``（调试/报告用）。
    """
    C = _normalize_rows(np.asarray(centroids, dtype=np.float64))
    k = C.shape[0]
    if k == 0:
        return np.zeros(0, dtype=int), []
    if k == 1:
        return np.zeros(1, dtype=int), []

    thr = float(threshold)
    # 每簇维护「质心之和」与「成员数」，平均相似度用它们点积算，避免每次重算 O(n^2) 对
    sums = [C[i].copy() for i in range(k)]
    counts = [1] * k
    members = [[i] for i in range(k)]
    history: List[Dict[str, Any]] = []

    while len(members) > 1:
        best: Optional[Tuple[float, int, int]] = None
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sim = float(sums[i] @ sums[j]) / float(counts[i] * counts[j])
                if best is None or sim > best[0] + 1e-12:
                    best = (sim, i, j)
        if best is None or best[0] < thr:
            break
        sim, i, j = best
        history.append({"i": int(members[i][0]), "j": int(members[j][0]),
                        "sim": round(sim, 6), "n_merged": counts[i] + counts[j]})
        sums[i] = sums[i] + sums[j]
        counts[i] += counts[j]
        members[i] = members[i] + members[j]
        del sums[j], counts[j], members[j]

    labels = np.zeros(k, dtype=int)
    for new_id, group in enumerate(members):
        for raw in group:
            labels[raw] = new_id
    return labels, history


def cluster_durations(
    clusters: Iterable[int], durations: Iterable[float]
) -> Dict[int, float]:
    """按簇累计时长，返回 ``{cluster_id: 秒}``（纯函数，空输入返回 ``{}``）。"""
    out: Dict[int, float] = {}
    for c, d in zip(clusters, durations):
        try:
            key = int(c)
            val = float(d)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val) or val <= 0:
            continue
        out[key] = out.get(key, 0.0) + val
    return out


def judge_stable_speakers(
    per_speaker_s: Mapping[int, float],
    *,
    min_speaker_s: float = 5.0,
    min_speaker_ratio: float = 0.03,
) -> Dict[str, Any]:
    """判定「稳定说话人」：每人时长 >= ``min_speaker_s`` 且占语音 >= ``min_speaker_ratio``。

    返回::

        {"speech_s", "n", "n_stable", "stable": [{cluster,dur_s,ratio}, ...],
         "main_ratio", "is_multi"}

    只有 ``n_stable >= 2``（``is_multi=True``）才允许升级 full；否则绝不造第二个身份。
    """
    dur = {int(k): float(v) for k, v in (per_speaker_s or {}).items() if float(v) > 0}
    total = float(sum(dur.values()))
    stable: List[Dict[str, Any]] = []
    for cluster, seconds in sorted(dur.items(), key=lambda kv: -kv[1]):
        ratio = seconds / total if total > 0 else 0.0
        if seconds >= float(min_speaker_s) - 1e-9 and ratio >= float(min_speaker_ratio) - 1e-12:
            stable.append({"cluster": int(cluster), "dur_s": round(seconds, 3),
                           "ratio": round(ratio, 4)})
    main_ratio = (max(dur.values()) / total) if total > 0 and dur else 0.0
    return {
        "speech_s": round(total, 3),
        "n": len(dur),
        "n_stable": len(stable),
        "stable": stable,
        "main_ratio": round(main_ratio, 4),
        "is_multi": len(stable) >= 2,
    }


# ---------------------------------------------------------------- 纯函数：段映射
def assign_labels(
    segments: Sequence[Mapping[str, Any]],
    diar_segments: Sequence[Mapping[str, Any]],
    *,
    speaker_by_cluster: Mapping[int, str],
    main_cluster: int = 0,
    min_overlap_ratio: float = 0.30,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """把 diar 簇标签映射到 transcript 段（保守规则，``step_build.assign_speakers`` 口径）。

    规则：
    1. 取与段重叠时长最大的 diar 簇；无重叠 → 主讲；
    2. 重叠占比 < ``min_overlap_ratio`` → 主讲（并计 ``uncertain``，可加 flag）；
    3. 簇不在 ``speaker_by_cluster``（碎片/未验收簇）→ 主讲（并计 ``fragment``）。

    返回 ``(assignments, stats)``；``assignments[i] = {speaker, cluster, conf,
    conservative, uncertain}``。
    """
    main_cluster = int(main_cluster)
    allowed = {int(k): str(v) for k, v in speaker_by_cluster.items()}
    diar = [{"start": float(d["start"]), "end": float(d["end"]),
             "speaker": int(d["speaker"])} for d in diar_segments]
    stats = {"n": 0, "no_overlap": 0, "low_overlap": 0, "fragment": 0, "uncertain": 0}
    out: List[Dict[str, Any]] = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        stats["n"] += 1
        overlaps: Dict[int, float] = {}
        for d in diar:
            o = min(end, d["end"]) - max(start, d["start"])
            if o > 0:
                overlaps[d["speaker"]] = overlaps.get(d["speaker"], 0.0) + o
        if not overlaps:
            stats["no_overlap"] += 1
            stats["uncertain"] += 1
            out.append({"speaker": SPEAKER_MAIN, "cluster": main_cluster, "conf": 0.0,
                        "conservative": True, "uncertain": True})
            continue
        cluster = max(overlaps, key=lambda c: (overlaps[c], -c))
        conf = overlaps[cluster] / max(end - start, 1e-6)
        conservative = False
        uncertain = False
        if conf < float(min_overlap_ratio):
            cluster = main_cluster
            conservative = True
            uncertain = True
            stats["low_overlap"] += 1
        elif cluster not in allowed:
            cluster = main_cluster
            conservative = True
            stats["fragment"] += 1
        if uncertain:
            stats["uncertain"] += 1
        out.append({"speaker": allowed.get(cluster, SPEAKER_MAIN), "cluster": int(cluster),
                    "conf": round(conf, 4), "conservative": conservative,
                    "uncertain": uncertain})
    return out, stats


def other_speaker_name(index: int) -> str:
    """非主讲说话人标签：``其他1``、``其他2``…（从 1 开始）。"""
    return f"{OTHER_PREFIX}{int(index)}"


# ---------------------------------------------------------------- sherpa 引擎
_engine_cache: Dict[Tuple[str, str, int, float], Any] = {}
_embedder_cache: Dict[Tuple[str, int], Any] = {}


def _import_sherpa() -> Any:
    try:
        import sherpa_onnx
    except ModuleNotFoundError as exc:  # pragma: no cover - 环境缺依赖
        raise DiarUnavailable("缺少 sherpa-onnx：pip install -r requirements.txt") from exc
    return sherpa_onnx


def create_diar_engine(cfg: Any, *, threshold: Optional[float] = None,
                       num_threads: Optional[int] = None) -> Any:
    """新建 ``OfflineSpeakerDiarization``（pyannote-seg + cam++ + FastClustering）。"""
    seg_model, emb_model = diar_model_paths(cfg)
    if not seg_model.is_file():
        raise DiarUnavailable(f"pyannote 分割模型不存在：{seg_model}")
    if not emb_model.is_file():
        raise DiarUnavailable(f"cam++ embedding 模型不存在：{emb_model}")
    sherpa = _import_sherpa()
    threshold = _cfg_float(cfg, "diar.threshold", 0.7) if threshold is None else float(threshold)
    threads = num_threads_for_diar(cfg) if num_threads is None else max(1, int(num_threads))
    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model)),
            num_threads=threads, debug=False, provider="cpu",
        ),
        embedding=sherpa.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model), num_threads=threads, debug=False, provider="cpu",
        ),
        clustering=sherpa.FastClusteringConfig(num_clusters=-1, threshold=float(threshold)),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():  # pragma: no cover - 模型/参数损坏
        raise DiarUnavailable(f"diar 配置校验失败（seg={seg_model.name}, emb={emb_model.name}）")
    return sherpa.OfflineSpeakerDiarization(config)


def get_diar_engine(cfg: Any, *, threshold: Optional[float] = None,
                    num_threads: Optional[int] = None) -> Any:
    """进程内缓存 diar 引擎（同模型/线程/阈值复用，省 1–2 s 加载）。"""
    seg_model, emb_model = diar_model_paths(cfg)
    threshold = _cfg_float(cfg, "diar.threshold", 0.7) if threshold is None else float(threshold)
    threads = num_threads_for_diar(cfg) if num_threads is None else max(1, int(num_threads))
    key = (str(seg_model), str(emb_model), int(threads), float(threshold))
    engine = _engine_cache.get(key)
    if engine is None:
        engine = create_diar_engine(cfg, threshold=threshold, num_threads=threads)
        _engine_cache[key] = engine
    return engine


def get_embedder(cfg: Any, *, num_threads: Optional[int] = None) -> Any:
    """进程内缓存 cam++ ``SpeakerEmbeddingExtractor``。"""
    _, emb_model = diar_model_paths(cfg)
    if not emb_model.is_file():
        raise DiarUnavailable(f"cam++ embedding 模型不存在：{emb_model}")
    threads = num_threads_for_diar(cfg) if num_threads is None else max(1, int(num_threads))
    key = (str(emb_model), int(threads))
    embedder = _embedder_cache.get(key)
    if embedder is None:
        sherpa = _import_sherpa()
        embedder = sherpa.SpeakerEmbeddingExtractor(
            sherpa.SpeakerEmbeddingExtractorConfig(
                model=str(emb_model), num_threads=int(threads), debug=False, provider="cpu",
            )
        )
        _embedder_cache[key] = embedder
    return embedder


def reset_diar_cache() -> None:
    """清空引擎/embedder 缓存（测试或换模型时用）。"""
    _engine_cache.clear()
    _embedder_cache.clear()


def _run_engine(engine: Any, samples: np.ndarray) -> List[Dict[str, Any]]:
    """跑一次 diar，返回按 start 排序的段 ``[{start,end,speaker,dur}]``（局部时间轴）。"""
    x = np.ascontiguousarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    if x.size == 0:
        return []
    result = engine.process(x)
    rows: List[Dict[str, Any]] = []
    for seg in result.sort_by_start_time():
        start, end = float(seg.start), float(seg.end)
        if end - start <= 0:
            continue
        rows.append({"start": round(start, 3), "end": round(end, 3),
                     "speaker": int(seg.speaker), "dur": round(end - start, 3)})
    return rows


def _embed_clip(embedder: Any, clip: np.ndarray) -> Optional[np.ndarray]:
    """单段 cam++ embedding（L2 归一化）；空/太短返回 None。"""
    x = np.ascontiguousarray(clip, dtype=np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    if x.size < int(EMBED_MIN_S * SR):
        return None
    stream = embedder.create_stream()
    stream.accept_waveform(SR, x)
    try:
        stream.input_finished()
    except Exception:  # pragma: no cover - 老版本 API 无此方法
        pass
    emb = np.asarray(embedder.compute(stream), dtype=np.float64).reshape(-1)
    if emb.size == 0:
        return None
    norm = float(np.linalg.norm(emb))
    if not math.isfinite(norm) or norm <= 0:
        return None
    return emb / norm


def _slice(x: np.ndarray, start: float, end: float) -> np.ndarray:
    a = max(0, int(round(float(start) * SR)))
    b = min(len(x), int(round(float(end) * SR)))
    if b <= a:
        return np.zeros(0, dtype=np.float32)
    return x[a:b]


def diarize_windows(
    wav16k: Any,
    cfg: Any,
    *,
    windows: Sequence[Tuple[float, float]],
    threshold: Optional[float] = None,
    merge_threshold: Optional[float] = None,
    num_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """在若干 clean 轴窗口上跑 diar，再做跨窗口质心归并。

    ``wav16k`` 可以是 16k float32 ndarray 或 wav 路径。返回::

        {"ok", "sr", "duration_s", "windows", "n_windows", "threshold",
         "merge_threshold", "raw_speakers", "raw_n_segments", "merged_speakers",
         "per_speaker_s", "main_ratio", "segments": [{start,end,dur,raw_speaker,
         merged_speaker}], "per_raw_speaker_s", "merge_history", "timings"}
    """
    t0 = time.perf_counter()
    x = load_wav16k(wav16k) if not isinstance(wav16k, np.ndarray) else np.asarray(wav16k, dtype=np.float32)
    duration = len(x) / float(SR)
    threshold = _cfg_float(cfg, "diar.threshold", 0.7) if threshold is None else float(threshold)
    merge_threshold = (
        _cfg_float(cfg, "diar.merge_threshold", 0.7) if merge_threshold is None else float(merge_threshold)
    )
    planned = [(round(float(a), 3), round(float(b), 3)) for a, b in windows
               if float(b) - float(a) >= 1.0]
    if not planned:
        return {
            "ok": False, "sr": SR, "duration_s": round(duration, 3), "windows": [],
            "n_windows": 0, "threshold": threshold, "merge_threshold": merge_threshold,
            "raw_speakers": 0, "raw_n_segments": 0, "merged_speakers": 0,
            "per_speaker_s": {}, "main_ratio": 0.0, "segments": [],
            "per_raw_speaker_s": {}, "merge_history": [],
            "timings": {"diar_s": 0.0, "embed_s": 0.0, "merge_s": 0.0},
        }

    engine = get_diar_engine(cfg, threshold=threshold, num_threads=num_threads)
    embedder = get_embedder(cfg, num_threads=num_threads)

    t_diar0 = time.perf_counter()
    raw_rows: List[Dict[str, Any]] = []
    for wi, (win_start, win_end) in enumerate(planned):
        clip = _slice(x, win_start, win_end)
        if clip.size < int(0.5 * SR):
            continue
        for row in _run_engine(engine, clip):
            raw_rows.append({
                "start": round(win_start + row["start"], 3),
                "end": round(win_start + row["end"], 3),
                "dur": row["dur"],
                "raw_speaker": (wi, int(row["speaker"])),
                "window": wi,
            })
    t_diar = time.perf_counter() - t_diar0

    # 段级 embedding（用于跨窗口簇归并）
    t_emb0 = time.perf_counter()
    raw_sum: Dict[Tuple[int, int], np.ndarray] = {}
    raw_w: Dict[Tuple[int, int], float] = {}
    raw_dur: Dict[Tuple[int, int], float] = {}
    for row in raw_rows:
        key = row["raw_speaker"]
        raw_dur[key] = raw_dur.get(key, 0.0) + float(row["dur"])
        dur = float(row["dur"])
        if dur < EMBED_MIN_S:
            continue
        clip = _slice(x, row["start"], min(row["end"], row["start"] + EMBED_MAX_S))
        if clip.size < int(EMBED_MIN_S * SR):
            continue
        emb = _embed_clip(embedder, clip)
        if emb is None:
            continue
        bucket = raw_sum.get(key)
        raw_sum[key] = emb * dur if bucket is None else bucket + emb * dur
        raw_w[key] = raw_w.get(key, 0.0) + dur
    t_emb = time.perf_counter() - t_emb0

    # 有 embedding 的原始簇：质心归并；无 embedding 的碎片簇：并回主讲（保守）
    t_merge0 = time.perf_counter()
    keys = sorted(raw_sum, key=lambda k: (raw_dur.get(k, 0.0), k), reverse=True)
    centroids = np.stack([raw_sum[k] / max(raw_w.get(k, 0.0), 1e-9) for k in keys]) if keys else np.zeros((0, 1))
    merged_of_raw: Dict[Tuple[int, int], int] = {}
    merge_history: List[Dict[str, Any]] = []
    if len(keys) > 0:
        labels, merge_history = merge_cluster_centroids(centroids, threshold=merge_threshold)
        # 簇时长 → 按时长降序重编号（0=主讲）
        cluster_dur: Dict[int, float] = {}
        for k, lab in zip(keys, labels):
            cluster_dur[int(lab)] = cluster_dur.get(int(lab), 0.0) + raw_dur.get(k, 0.0)
        order = sorted(cluster_dur, key=lambda c: (-cluster_dur[c], c))
        remap = {c: i for i, c in enumerate(order)}
        for k, lab in zip(keys, labels):
            merged_of_raw[k] = remap[int(lab)]
    t_merge = time.perf_counter() - t_merge0

    rows: List[Dict[str, Any]] = []
    per_speaker: Dict[int, float] = {}
    for row in raw_rows:
        key = row["raw_speaker"]
        cluster = merged_of_raw.get(key, 0)  # 碎片簇 → 主讲
        rows.append({"start": row["start"], "end": row["end"], "dur": row["dur"],
                     "raw_speaker": list(key), "merged_speaker": int(cluster)})
        per_speaker[cluster] = per_speaker.get(cluster, 0.0) + float(row["dur"])
    rows.sort(key=lambda r: r["start"])

    speech_s = float(sum(per_speaker.values()))
    main_ratio = (max(per_speaker.values()) / speech_s) if speech_s > 0 else 0.0
    return {
        "ok": bool(rows),
        "sr": SR,
        "duration_s": round(duration, 3),
        "windows": [[a, b] for a, b in planned],
        "n_windows": len(planned),
        "threshold": round(float(threshold), 4),
        "merge_threshold": round(float(merge_threshold), 4),
        "raw_speakers": len(raw_dur),
        "raw_n_segments": len(raw_rows),
        "merged_speakers": len(per_speaker),
        "per_speaker_s": {str(k): round(v, 3) for k, v in sorted(per_speaker.items(), key=lambda kv: -kv[1])},
        "per_raw_speaker_s": {f"{w}:{s}": round(v, 3) for (w, s), v in
                              sorted(raw_dur.items(), key=lambda kv: -kv[1])},
        "main_ratio": round(main_ratio, 4),
        "segments": rows,
        "merge_history": merge_history,
        "timings": {"diar_s": round(t_diar, 2), "embed_s": round(t_emb, 2),
                    "merge_s": round(t_merge, 3), "total_s": round(time.perf_counter() - t0, 2)},
    }


# ---------------------------------------------------------------- 抽样 / 全量
def sample_speakers(
    wav16k: Any,
    cfg: Any,
    *,
    windows: Optional[Sequence[Tuple[float, float]]] = None,
    body_margin_s: Optional[float] = None,
    threshold: Optional[float] = None,
    merge_threshold: Optional[float] = None,
    min_speaker_s: Optional[float] = None,
    min_speaker_ratio: Optional[float] = None,
    num_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """抽样 diar + 稳定说话人判定（P4-T3 的核心，可独立调用/测试）。

    返回在 :func:`diarize_windows` 结果上追加::

        {"is_multi", "n_stable", "stable": [...], "n_speakers", "sampled": True,
         "min_speaker_s", "min_speaker_ratio", "verdict": "multi"/"single"/"empty"}
    """
    x = load_wav16k(wav16k) if not isinstance(wav16k, np.ndarray) else np.asarray(wav16k, dtype=np.float32)
    duration = len(x) / float(SR)
    if windows is None:
        windows = plan_sample_windows(
            duration,
            sample_minutes=_cfg_float(cfg, "diar.sample_minutes", 10.0),
            sample_segments=_cfg_int(cfg, "diar.sample_segments", 3),
            body_margin_s=(_cfg_float(cfg, "diar.sample_body_margin_s", 120.0)
                           if body_margin_s is None else float(body_margin_s)),
        )
    result = diarize_windows(
        x, cfg, windows=windows, threshold=threshold,
        merge_threshold=merge_threshold, num_threads=num_threads,
    )
    if min_speaker_s is None:
        min_speaker_s = _cfg_float(cfg, "diar.min_speaker_s", 5.0)
    if min_speaker_ratio is None:
        min_speaker_ratio = _cfg_float(cfg, "diar.min_speaker_ratio", 0.03)
    judge = judge_stable_speakers(
        {int(k): float(v) for k, v in result.get("per_speaker_s", {}).items()},
        min_speaker_s=float(min_speaker_s),
        min_speaker_ratio=float(min_speaker_ratio),
    )
    result.update({
        "sampled": True,
        "min_speaker_s": round(float(min_speaker_s), 3),
        "min_speaker_ratio": round(float(min_speaker_ratio), 4),
        "n_speakers": judge["n"],
        "n_stable": judge["n_stable"],
        "stable": judge["stable"],
        "judge": judge,
        "is_multi": bool(judge["is_multi"]),
        "verdict": "multi" if judge["is_multi"] else ("single" if judge["n"] else "empty"),
    })
    return result


def full_diarize(
    wav16k: Any,
    cfg: Any,
    *,
    threshold: Optional[float] = None,
    merge_threshold: Optional[float] = None,
    min_speaker_s: Optional[float] = None,
    min_speaker_ratio: Optional[float] = None,
    fragment_s: Optional[float] = None,
    num_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """全量 diar：最长簇=主讲，其余满足稳定条件的簇编 ``其他N``（绝不造碎片身份）。"""
    x = load_wav16k(wav16k) if not isinstance(wav16k, np.ndarray) else np.asarray(wav16k, dtype=np.float32)
    duration = len(x) / float(SR)
    base = diarize_windows(
        x, cfg, windows=[(0.0, duration)], threshold=threshold,
        merge_threshold=merge_threshold, num_threads=num_threads,
    )
    if fragment_s is None:
        fragment_s = _cfg_float(cfg, "diar.fragment_s", 3.0)
    if min_speaker_s is None:
        min_speaker_s = _cfg_float(cfg, "diar.min_speaker_s", 5.0)
    if min_speaker_ratio is None:
        min_speaker_ratio = _cfg_float(cfg, "diar.min_speaker_ratio", 0.03)

    per = {int(k): float(v) for k, v in base.get("per_speaker_s", {}).items()}
    if not per:
        return {**base, "full": True, "n_speakers": 0, "speaker_by_cluster": {},
                "main_cluster": 0, "clusters": []}
    total = float(sum(per.values()))
    order = sorted(per, key=lambda c: (-per[c], c))
    main_cluster = order[0]
    # 身份门槛：时长 >= max(fragment_s, min_speaker_s) 且占比 >= min_speaker_ratio
    identity_min_s = max(float(fragment_s), float(min_speaker_s))
    speaker_by_cluster: Dict[int, str] = {main_cluster: SPEAKER_MAIN}
    others: List[int] = []
    for cluster in order[1:]:
        seconds = per[cluster]
        ratio = seconds / total if total > 0 else 0.0
        if seconds >= identity_min_s - 1e-9 and ratio >= float(min_speaker_ratio) - 1e-12:
            others.append(cluster)
    for i, cluster in enumerate(others):
        speaker_by_cluster[cluster] = other_speaker_name(i + 1)

    rows: List[Dict[str, Any]] = []
    for row in base.get("segments", []):
        cluster = int(row["merged_speaker"])
        rows.append({**row, "speaker": speaker_by_cluster.get(cluster, SPEAKER_MAIN),
                     "named": cluster in speaker_by_cluster})
    clusters = [{"cluster": c, "dur_s": round(per[c], 3),
                 "ratio": round(per[c] / total, 4) if total > 0 else 0.0,
                 "speaker": speaker_by_cluster.get(c, SPEAKER_MAIN),
                 "named": c in speaker_by_cluster}
                for c in order]
    return {
        **base,
        "full": True,
        "n_speakers": len(speaker_by_cluster),
        "speaker_by_cluster": {int(k): v for k, v in speaker_by_cluster.items()},
        "main_cluster": int(main_cluster),
        "clusters": clusters,
        "segments": rows,
        "identity_min_s": round(identity_min_s, 3),
    }


# ---------------------------------------------------------------- 音频 / JSON IO
def load_wav16k(path: Any) -> np.ndarray:
    """读 16k wav 为 float32 单声道 ndarray（兼容 int16/float32 两种写入格式）。"""
    from scipy.io import wavfile

    sr, data = wavfile.read(str(path))
    if int(sr) != SR:
        raise DiarError(f"clean16k 采样率应为 {SR}，实际 {sr}：{path}")
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        x = data.astype(np.float32, copy=False)
    elif data.dtype == np.float64:
        x = data.astype(np.float32)
    else:  # pragma: no cover - 少见格式
        raise DiarError(f"不支持的 wav dtype={data.dtype}：{path}")
    if x.ndim > 1:
        x = x[:, 0]
    return np.ascontiguousarray(x, dtype=np.float32)


def clean16k_path(job_id: str, cfg: Any, *, cache_dir: Optional[str | Path] = None) -> Path:
    """``work/cache/<id>.clean16k.wav`` 的规范路径。"""
    root = Path(cache_dir) if cache_dir else cfg.path("cache_dir")
    return root / f"{job_id}{CLEAN16K_SUFFIX}"


def ensure_clean16k(
    job_id: str,
    cfg: Any,
    *,
    out_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """确保 clean16k wav 存在（缺时从 ``out/<id>/clean.m4a`` 用 ffmpeg 解）。

    - 已存在且可读 → 直接复用（transcribe-dev 的 ``asr.decode_clean16k`` 产物也能读）；
    - 写入用「临时文件 + ``os.replace``」原子替换，避免与 ASR 阶段并行解码时互相踩；
    - 时长与 ``report.json.trim.clean_dur_s`` 偏差 >0.1 s 时记 warning（不失败）。
    """
    root = Path(cache_dir) if cache_dir else cfg.path("cache_dir")
    out_root = Path(out_dir) if out_dir else cfg.path("out_dir")
    dst = clean16k_path(job_id, cfg, cache_dir=root)
    if dst.is_file() and not force:
        try:
            from scipy.io import wavfile

            sr, data = wavfile.read(str(dst))
            if int(sr) == SR and len(data) > 0:
                _warn_duration_mismatch(job_id, cfg, out_root, len(data) / float(SR))
                return dst
        except Exception as exc:  # 缓存损坏 → 重解
            log.warning("clean16k 缓存读取失败，重解：%s（%s）", dst, exc)
    src = resolve_product(out_root / job_id, out_root, "clean.m4a")
    if src is None:
        raise DiarError(f"clean.m4a 不存在，无法解 clean16k：{out_root / job_id / 'clean.m4a'}")
    if shutil.which("ffmpeg") is None:  # pragma: no cover - 环境缺 ffmpeg
        raise DiarError("PATH 中缺 ffmpeg，无法解 clean16k")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.stem}.tmp{os.getpid()}{dst.suffix}")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
           "-i", str(src), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR),
           "-c:a", "pcm_f32le", "-f", "wav", str(tmp)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=1800.0)
    except subprocess.TimeoutExpired as exc:
        raise DiarError(f"ffmpeg 解 clean16k 超时：{src}") from exc
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise DiarError(f"ffmpeg 解 clean16k 失败（{proc.returncode}）："
                        f"{proc.stderr.decode('utf-8', 'ignore')[:400]}")
    if not tmp.is_file() or tmp.stat().st_size <= 44:
        tmp.unlink(missing_ok=True)
        raise DiarError(f"ffmpeg 未产出有效 wav：{tmp}")
    os.replace(tmp, dst)
    _warn_duration_mismatch(job_id, cfg, out_root, _wav_duration(dst))
    log.info("已解 clean16k：%s（%.2fs）", dst, time.perf_counter() - t0)
    return dst


def _wav_duration(path: Path) -> float:
    from scipy.io import wavfile

    sr, data = wavfile.read(str(path))
    return len(data) / float(sr)


def _warn_duration_mismatch(job_id: str, cfg: Any, out_root: Path, duration_s: float) -> None:
    report_path = out_root / job_id / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    clean_dur = ((report.get("trim") or {}).get("clean_dur_s") if isinstance(report, dict) else None)
    if isinstance(clean_dur, (int, float)) and abs(float(clean_dur) - float(duration_s)) > 0.1:
        log.warning("clean16k 时长 %.3fs 与 report.trim.clean_dur_s %.3fs 偏差 >0.1s（%s）",
                    duration_s, float(clean_dur), job_id)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> Path:
    """UTF-8、无 NaN、原子替换（与 export.write_json 同口径）。"""
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def _read_transcript(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise DiarError(f"transcript.json 不存在（先跑转写）：{path}")
    data = _read_json(path)
    if not isinstance(data, list):
        raise DiarError(f"transcript.json 顶层应为 list（契约 §4.3）：{path}")
    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(data):
        if not isinstance(seg, dict):
            raise DiarError(f"transcript.json 第 {i} 条不是对象：{path}")
        out.append(dict(seg))
    return out


def _read_report(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = _read_json(path)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------- 应用标签
def apply_full_labels(
    segments: List[Dict[str, Any]],
    full: Mapping[str, Any],
    *,
    min_overlap_ratio: float = 0.30,
) -> Tuple[int, Dict[str, int]]:
    """把 full diar 结果写回 transcript 段（返回 ``(change_count, stats)``）。"""
    speaker_by_cluster = {int(k): str(v) for k, v in (full.get("speaker_by_cluster") or {}).items()}
    assignments, stats = assign_labels(
        segments, full.get("segments", []),
        speaker_by_cluster=speaker_by_cluster,
        main_cluster=int(full.get("main_cluster", 0)),
        min_overlap_ratio=min_overlap_ratio,
    )
    changed = 0
    for seg, info in zip(segments, assignments):
        if seg.get("speaker") != info["speaker"]:
            changed += 1
        seg["speaker"] = info["speaker"]
        if info["uncertain"]:
            flags = seg.get("flags")
            if not isinstance(flags, list):
                flags = []
            if "uncertain_speaker" not in flags:
                flags.append("uncertain_speaker")
            seg["flags"] = flags
    return changed, stats


# ---------------------------------------------------------------- 主入口
def _apply_full_result(
    segments: List[Dict[str, Any]],
    full_result: Mapping[str, Any],
    *,
    mode: str,
    sampled: Optional[Mapping[str, Any]] = None,
    min_overlap_ratio: float = 0.30,
) -> Tuple[int, Dict[str, Any]]:
    """全量 diar 结果落到 transcript：确认 >=2 人才打 ``其他N``，否则保守单标签。

    返回 ``(changed, report_speakers)``。
    """
    confirmed = int(full_result.get("n_speakers", 0) or 0)
    if full_result.get("speaker_by_cluster") and confirmed >= 2:
        changed, assign_stats = apply_full_labels(
            segments, full_result, min_overlap_ratio=min_overlap_ratio)
        speakers: Dict[str, Any] = {
            "n": confirmed, "method": mode, "diar_version": DIAR_VERSION,
            "raw_clusters": full_result.get("raw_speakers"),
            "merged_clusters": full_result.get("merged_speakers"),
            "main_ratio": full_result.get("main_ratio"),
            "full_clusters": full_result.get("clusters"),
            "assign": assign_stats,
        }
        if sampled is not None:
            speakers.update({
                "sampled_judged_multi": True,
                "trigger": "sampled",
                "sample_windows": sampled.get("windows"),
                "sample_raw_clusters": sampled.get("raw_speakers"),
                "sample_merged_clusters": sampled.get("merged_speakers"),
                "sample_main_ratio": sampled.get("main_ratio"),
            })
        log.info("全量 diar 确认 %d 位说话人（mode=%s），已重标 %d 段",
                 confirmed, mode, changed)
        return changed, speakers
    # 全量没有确认第二人 → 保持单标签（绝不臆造）
    changed = sum(1 for s in segments if s.get("speaker") != SPEAKER_MAIN)
    for seg in segments:
        seg["speaker"] = SPEAKER_MAIN
    speakers = {
        "n": 1, "method": mode, "diar_version": DIAR_VERSION,
        "full_confirmed_n": confirmed,
        "raw_clusters": full_result.get("raw_speakers"),
        "merged_clusters": full_result.get("merged_speakers"),
        "main_ratio": full_result.get("main_ratio"),
        "full_clusters": full_result.get("clusters"),
    }
    if sampled is not None:
        speakers.update({"sampled_judged_multi": True,
                         "sample_windows": sampled.get("windows")})
    log.info("全量 diar 未确认第二人（n=%s），保守保持全部「%s」：%d 段",
             confirmed, SPEAKER_MAIN, len(segments))
    return changed, speakers


def diarize_job(
    job_id: str,
    cfg: Any,
    *,
    out_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    force: bool = False,
    on_stage: Optional[Any] = None,
) -> Dict[str, Any]:
    """单个 job 的说话人标注阶段（P4-T3）。

    参数
    ----
    job_id : 任务 id（``out/<id>/``）
    cfg    : ``app.config.Config``
    force  : 忽略已有结果重算
    on_stage : ``on_stage("diarize", "done"/"failed", metrics)`` 回调（可空）

    返回 ``{"ok", "job_id", "stage", "mode", "skipped", "n_segments", "changed",
    "speakers", "elapsed_s", "diar", "report"}``；失败不抛异常（``ok=False`` + ``error``）。
    """
    started = time.perf_counter()
    mode = str(_cfg_get(cfg, "diar.mode", "none") or "none").strip().lower()
    out_root = Path(out_dir) if out_dir else cfg.path("out_dir")
    job_out = out_root / job_id
    transcript_path = job_out / "transcript.json"
    report_path = job_out / "report.json"
    result: Dict[str, Any] = {
        "ok": False, "job_id": job_id, "stage": "diarize", "mode": mode,
        "skipped": False, "n_segments": 0, "changed": 0, "speakers": None,
        "elapsed_s": 0.0, "diar": None, "report": None,
    }

    def notify(status: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        if on_stage is None:
            return
        try:
            on_stage("diarize", status, metrics or {})
        except Exception:  # pragma: no cover - 回调失败不影响主流程
            log.debug("on_stage 回调失败", exc_info=True)

    try:
        if mode not in ("none", "sampled", "full"):
            raise DiarError(f"diar.mode={mode!r} 非法（none|sampled|full）")
        segments = _read_transcript(transcript_path)
        report = _read_report(report_path)
        prev = report.get("speakers") if isinstance(report.get("speakers"), dict) else {}
        result["n_segments"] = len(segments)

        # 已有结果 → 跳过（--force 重做）
        target_method = "none" if mode == "none" else mode
        settled = (prev.get("method") == target_method
                   or (target_method == "sampled" and prev.get("method") == "full"
                       and prev.get("trigger") == "sampled"))
        if (not force and prev.get("diar_version") == DIAR_VERSION and settled
                and (mode != "none" or all(s.get("speaker") == SPEAKER_MAIN for s in segments))):
            result.update({
                "ok": True, "skipped": True, "speakers": dict(prev),
                "report": report, "elapsed_s": round(time.perf_counter() - started, 2),
                "stage": "diarize",
            })
            notify("done", {"skipped": True, "mode": mode, "speakers": dict(prev)})
            log.info("说话人结果已存在，跳过：%s（mode=%s）", job_id, mode)
            return result

        # ---- none：全部主讲，绝不改 text/flags
        if mode == "none":
            changed = sum(1 for s in segments if s.get("speaker") != SPEAKER_MAIN)
            for seg in segments:
                seg["speaker"] = SPEAKER_MAIN
            speakers = {"n": 1, "method": "none", "diar_version": DIAR_VERSION}
            if changed:
                _write_json_atomic(transcript_path, segments)
            _update_report(report_path, report, {"speakers": speakers,
                                                 "timing": {"diarize_s": round(time.perf_counter() - started, 2)}})
            result.update({
                "ok": True, "changed": changed, "speakers": speakers,
                "report": _read_report(report_path),
                "elapsed_s": round(time.perf_counter() - started, 2),
            })
            notify("done", {"mode": "none", "n": 1, "changed": changed})
            log.info("说话人标注（none）：%s %d 段全部标「%s」", job_id, len(segments), SPEAKER_MAIN)
            return result

        # ---- sampled / full 需要模型
        if not diar_available(cfg):
            seg_model, emb_model = diar_model_paths(cfg)
            raise DiarUnavailable(f"diar 模型不完整：seg={seg_model} emb={emb_model}")

        wav_path = ensure_clean16k(job_id, cfg, out_dir=out_root, cache_dir=cache_dir)
        x = load_wav16k(wav_path)
        diar_evidence: Dict[str, Any] = {}
        changed = 0
        speakers: Dict[str, Any]

        if mode == "full":
            # 显式 full：直接全量 diar（不经抽样门槛）
            full_result = full_diarize(x, cfg)
            diar_evidence["full"] = _compact_full(full_result)
            changed, speakers = _apply_full_result(
                segments, full_result, mode="full",
                min_overlap_ratio=_cfg_float(cfg, "diar.min_overlap_ratio", 0.30),
            )
        else:
            sampled = sample_speakers(
                x, cfg,
                body_margin_s=_cfg_float(cfg, "diar.sample_body_margin_s", 120.0),
            )
            diar_evidence["sampled"] = _compact_sampled(sampled)
            if not sampled.get("is_multi"):
                # 抽样结论 n=1（本批文件的正确结果）：保持/恢复全部主讲
                changed = sum(1 for s in segments if s.get("speaker") != SPEAKER_MAIN)
                for seg in segments:
                    seg["speaker"] = SPEAKER_MAIN
                speakers = {
                    "n": 1, "method": "sampled", "diar_version": DIAR_VERSION,
                    "sampled_judged_multi": False,
                    "raw_clusters": sampled.get("raw_speakers"),
                    "merged_clusters": sampled.get("merged_speakers"),
                    "main_ratio": sampled.get("main_ratio"),
                    "stable_speakers": sampled.get("n_stable"),
                    "sample_windows": sampled.get("windows"),
                    "min_speaker_s": sampled.get("min_speaker_s"),
                    "min_speaker_ratio": sampled.get("min_speaker_ratio"),
                }
            else:
                # 抽样发现 >=2 个稳定说话人 → 全量 diar
                log.info("抽样判定 %d 个稳定说话人，升级全量 diar：%s",
                         sampled.get("n_stable"), job_id)
                full_result = full_diarize(x, cfg)
                diar_evidence["full"] = _compact_full(full_result)
                changed, speakers = _apply_full_result(
                    segments, full_result, mode="full", sampled=sampled,
                    min_overlap_ratio=_cfg_float(cfg, "diar.min_overlap_ratio", 0.30),
                )

        _write_json_atomic(transcript_path, segments)
        elapsed = round(time.perf_counter() - started, 2)
        _update_report(report_path, report, {
            "speakers": speakers,
            "timing": {"diarize_s": elapsed},
        })
        result.update({
            "ok": True, "changed": int(changed), "speakers": speakers,
            "diar": diar_evidence, "report": _read_report(report_path),
            "elapsed_s": elapsed,
        })
        notify("done", {"mode": mode, "n": speakers.get("n"), "method": speakers.get("method"),
                        "changed": int(changed), "elapsed_s": elapsed})
        log.info("说话人标注完成：%s mode=%s n=%s changed=%d（%.1fs）",
                 job_id, speakers.get("method"), speakers.get("n"), int(changed), elapsed)
        return result

    except Exception as exc:
        msg = str(exc) or exc.__class__.__name__
        log.error("说话人标注失败（%s）：%s", job_id, msg)
        notify("failed", {"error": msg})
        result.update({"ok": False, "stage": "diarize", "error": msg,
                       "elapsed_s": round(time.perf_counter() - started, 2)})
        return result


def _compact_sampled(sampled: Mapping[str, Any]) -> Dict[str, Any]:
    """报告用的抽样证据（控制体积：不写逐段列表）。"""
    return {
        "windows": sampled.get("windows"),
        "threshold": sampled.get("threshold"),
        "merge_threshold": sampled.get("merge_threshold"),
        "raw_clusters": sampled.get("raw_speakers"),
        "raw_n_segments": sampled.get("raw_n_segments"),
        "merged_clusters": sampled.get("merged_speakers"),
        "per_speaker_s": dict(list((sampled.get("per_speaker_s") or {}).items())[:5]),
        "main_ratio": sampled.get("main_ratio"),
        "n_stable": sampled.get("n_stable"),
        "stable": sampled.get("stable"),
        "min_speaker_s": sampled.get("min_speaker_s"),
        "min_speaker_ratio": sampled.get("min_speaker_ratio"),
        "verdict": sampled.get("verdict"),
        "timings": sampled.get("timings"),
    }


def _compact_full(full: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "n_speakers": full.get("n_speakers"),
        "main_cluster": full.get("main_cluster"),
        "clusters": full.get("clusters"),
        "identity_min_s": full.get("identity_min_s"),
        "timings": full.get("timings"),
        "threshold": full.get("threshold"),
        "merge_threshold": full.get("merge_threshold"),
        "raw_clusters": full.get("raw_speakers"),
        "merged_clusters": full.get("merged_speakers"),
    }


def _update_report(report_path: Path, report: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """report.json 读-改-写：只覆盖 ``updates`` 里的（可嵌套）键，保留其余字段。"""
    merged: Dict[str, Any] = dict(report) if isinstance(report, Mapping) else {}
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            section = dict(merged[key])
            section.update(value)
            merged[key] = section
        else:
            merged[key] = value
    if "ok" in report:
        merged["ok"] = report.get("ok")
    _write_json_atomic(report_path, merged)
    return merged


def summarize(sampled_or_full: Mapping[str, Any]) -> Dict[str, Any]:
    """把 diar 结果压成一行可读摘要（CLI/日志用）。"""
    if sampled_or_full.get("full"):
        return {"method": "full", "n": sampled_or_full.get("n_speakers"),
                "main_ratio": sampled_or_full.get("main_ratio")}
    return {"method": "sampled", "n": sampled_or_full.get("n_stable"),
            "raw_clusters": sampled_or_full.get("raw_speakers"),
            "merged_clusters": sampled_or_full.get("merged_speakers"),
            "main_ratio": sampled_or_full.get("main_ratio")}


# ---------------------------------------------------------------- CLI（调试用）
def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - 人工调试
    """``python -m app.pipeline.diarize --job <id> [--mode sampled] [--json]``。"""
    import argparse

    from ..config import load_config

    ap = argparse.ArgumentParser(description="说话人抽样/全量检查（P4-T3）")
    ap.add_argument("--job", required=True, help="out/<id> 的 id")
    ap.add_argument("--mode", default="", help="none|sampled|full（默认读配置）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.mode:
        cfg = cfg.replace({"diar.mode": args.mode})
    result = diarize_job(args.job, cfg, force=True)
    payload = {"ok": result.get("ok"), "mode": result.get("mode"),
               "speakers": result.get("speakers"), "diar": result.get("diar"),
               "error": result.get("error", "")}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"[diarize] ok={payload['ok']} mode={payload['mode']} "
              f"speakers={payload['speakers']} error={payload['error']}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
