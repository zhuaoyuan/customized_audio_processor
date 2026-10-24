# -*- coding: utf-8 -*-
"""SenseVoice 转写封装（P4-T1，P2-T3 头尾边界判定先依赖它）。

为什么提前到这里
----------------
实施计划 P2-T3 明确「按 VAD 段逐段送 SenseVoice（**复用 P4-T1 封装**）」，所以本模块
在 P2 阶段先行落地最小可用版：**单例 recognizer + 单段/批量转写 + 文本清洗**。P4 阶段
只需在其上补「分段与时间戳组装」，不需要改这里的接口。

P4-T2 在本模块追加「分段转写 + clean/orig 双时间轴」：
``decode_clean16k`` → ``build_segment_table`` → ``transcribe_job``（详细口径见各函数 docstring）。

关键约束
--------
- recognizer 加载 1–3 s，**必须复用**（`get_transcriber` 做进程内单例），禁止每段重建；
- SenseVoice 会输出 ``<|zh|><|NEUTRAL|>`` 等标签与表情符号，必须清洗后再统计字符；
- 模型文件缺失要抛明确异常（`AsrUnavailable`），由调用方降级并记 flag，而不是静默出空文本。
- 段表以 **P2 analyze 缓存 `result.segments`（orig 轴）为唯一权威**，用 P3 `edit.json.kept`
  的 `TimeMap` 映射到 clean 轴；**禁止用字符比例估时间**（计划 §2 硬约束）。
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_setup import get_logger
from ..models import SPEAKER_MAIN, AnalyzeResult, KeptSpan, TranscriptSegment
from .decode import decode_audio, read_wav_float32, wav_info
from .finalize import resolve_product
from .trim import TimeMap

log = get_logger(__name__)

#: 采样率：SenseVoice / VAD 统一 16 kHz
SR = 16000

#: SenseVoice 输出的语言/情感/事件标签
_TAG_RE = re.compile(r"<\|[^|]*\|>")
#: 表情符号（含变体选择符）
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]+")
#: 空白（含全角空格）
_SPACE_RE = re.compile(r"\s+")
#: 统计「有效字符」用：汉字 + 字母 + 数字（标点不计）
_SPEECH_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
#: 标点/符号（用于句级切分，P4 用）
_PUNCT = "，。！？；：、,.!?;:…—～~（）()「」『』《》“”\"'"


class AsrUnavailable(RuntimeError):
    """模型文件缺失或 sherpa-onnx 不可用。"""


# ---------------------------------------------------------------- 文本清洗
def clean_text(text: Optional[str]) -> str:
    """去掉 SenseVoice 标签、表情、空白，返回可直接展示/统计的文本。"""
    t = _TAG_RE.sub("", text or "")
    t = _EMOJI_RE.sub("", t)
    t = _SPACE_RE.sub("", t)
    return t.strip()


def count_speech_chars(text: str) -> int:
    """统计有效字符数（汉字/字母/数字，不含标点）——与 PoC 的 ``cn_len`` 口径一致。"""
    return len(_SPEECH_CHAR_RE.findall(text or ""))


def count_speech_chars_cleaned(text: Optional[str]) -> int:
    return count_speech_chars(clean_text(text))


def strip_punct(text: str) -> str:
    """仅用于统计/对齐：去掉标点与空白。"""
    return "".join(ch for ch in (text or "") if ch not in _PUNCT and not ch.isspace())


# ---------------------------------------------------------------- recognizer
class Transcriber:
    """SenseVoice（sherpa-onnx OfflineRecognizer）薄封装。

    用法::

        tr = Transcriber(model_path, tokens_path, num_threads=8)
        text = tr.transcribe(clip_float32_16k)
        texts = tr.transcribe_many([clip1, clip2])
    """

    def __init__(
        self,
        model_path: str | Path,
        tokens_path: str | Path,
        num_threads: int = 8,
        use_itn: bool = True,
        language: str = "zh",
        debug: bool = False,
    ) -> None:
        model = Path(model_path)
        tokens = Path(tokens_path)
        if not model.is_file():
            raise AsrUnavailable(f"SenseVoice 模型文件不存在：{model}")
        if not tokens.is_file():
            raise AsrUnavailable(f"SenseVoice tokens 文件不存在：{tokens}")
        try:
            import sherpa_onnx
        except ModuleNotFoundError as exc:  # pragma: no cover - 环境缺依赖
            raise AsrUnavailable(
                "缺少 sherpa-onnx：pip install -r requirements.txt"
            ) from exc

        self.model_path = model
        self.tokens_path = tokens
        self.num_threads = int(num_threads)
        self.use_itn = bool(use_itn)
        self.language = language
        self._lock = threading.Lock()
        self.load_s = 0.0
        t0 = time.perf_counter()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(tokens),
            num_threads=self.num_threads,
            use_itn=self.use_itn,
            language=language,
            debug=bool(debug),
        )
        self.load_s = time.perf_counter() - t0
        log.info(
            "SenseVoice 已加载：%s threads=%d itn=%s lang=%s（%.2fs）",
            model.name, self.num_threads, self.use_itn, language, self.load_s,
        )

    # -- 单段
    def transcribe(self, clip: np.ndarray, sr: int = SR) -> str:
        """转写一段 float32 音频，返回清洗后的文本。"""
        text, _ = self.transcribe_raw(clip, sr)
        return text

    def transcribe_raw(self, clip: np.ndarray, sr: int = SR) -> Tuple[str, str]:
        """返回 ``(清洗后文本, 原始文本)``。"""
        x = np.ascontiguousarray(clip, dtype=np.float32)
        if x.ndim > 1:
            x = x[:, 0]
        if x.size == 0:
            return "", ""
        with self._lock:  # sherpa-onnx recognizer 非线程安全，串行解码
            stream = self._recognizer.create_stream()
            stream.accept_waveform(int(sr), x)
            self._recognizer.decode_stream(stream)
            raw = stream.result.text or ""
        return clean_text(raw), raw

    # -- 批量
    def transcribe_many(
        self, clips: Iterable[np.ndarray], sr: int = SR
    ) -> List[str]:
        return [self.transcribe(clip, sr) for clip in clips]

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"Transcriber(model={self.model_path.name}, threads={self.num_threads}, "
            f"itn={self.use_itn}, lang={self.language})"
        )


# ---------------------------------------------------------------- 工厂 / 单例
_cache: dict[tuple, Transcriber] = {}
_cache_lock = threading.Lock()


def model_paths(cfg: Any) -> Tuple[Path, Path]:
    """从配置取 SenseVoice 的 onnx 与 tokens 路径。"""
    model_dir = cfg.model_path("sensevoice_dir")
    model_name = cfg.get("asr.model_file", "model.int8.onnx")
    tokens_name = cfg.get("asr.tokens_file", "tokens.txt")
    return model_dir / str(model_name), model_dir / str(tokens_name)


def create_transcriber(cfg: Any) -> Transcriber:
    """按配置新建一个 Transcriber（不缓存）。"""
    model, tokens = model_paths(cfg)
    return Transcriber(
        model,
        tokens,
        num_threads=int(cfg.get("asr.num_threads", 8)),
        use_itn=bool(cfg.get("asr.use_itn", True)),
        language=str(cfg.get("asr.language", "zh")),
    )


def get_transcriber(cfg: Any) -> Transcriber:
    """进程内单例：同 (model, tokens, threads, itn, lang) 复用同一个 recognizer。

    加载一次 1–3 s，分段转写/边界扫描会反复调用，**不要**每段重建。
    """
    model, tokens = model_paths(cfg)
    key = (
        str(model), str(tokens),
        int(cfg.get("asr.num_threads", 8)),
        bool(cfg.get("asr.use_itn", True)),
        str(cfg.get("asr.language", "zh")),
    )
    with _cache_lock:
        existing = _cache.get(key)
        if existing is not None:
            return existing
        transcriber = create_transcriber(cfg)
        _cache[key] = transcriber
        return transcriber


def reset_transcriber_cache() -> None:
    """释放缓存的 recognizer（测试/换模型时用）。"""
    with _cache_lock:
        _cache.clear()


def is_available(cfg: Any) -> bool:
    """模型文件是否就绪（不加载模型）。"""
    try:
        model, tokens = model_paths(cfg)
    except Exception:  # pragma: no cover - 配置缺路径
        return False
    return model.is_file() and tokens.is_file()


# ---------------------------------------------------------------- 分段工具（P4 用）
def split_long_segment(start: float, end: float, max_s: float) -> List[Tuple[float, float]]:
    """把超过 ``max_s`` 的语音段均分（P4-T2 规则：过长按 30 s 切）。"""
    duration = float(end) - float(start)
    if duration <= max_s or max_s <= 0:
        return [(float(start), float(end))]
    parts = int(np.ceil(duration / max_s))
    step = duration / parts
    out: List[Tuple[float, float]] = []
    for i in range(parts):
        a = float(start) + i * step
        b = float(end) if i == parts - 1 else float(start) + (i + 1) * step
        out.append((round(a, 3), round(b, 3)))
    return out


def slice_audio(x: np.ndarray, start: float, end: float, sr: int = SR) -> np.ndarray:
    """按秒取片段（自动按样本边界裁剪，返回连续 float32 数组）。"""
    a = max(0, int(round(start * sr)))
    b = min(len(x), int(round(end * sr)))
    if b <= a:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(x[a:b], dtype=np.float32)


# ================================================================ P4-T2 分段转写
#: clean 音轨 16k 缓存后缀（独立文件，**不覆盖** ``.16k.wav``/``.analyze.json``/``.decode.json``）
CLEAN16K_SUFFIX = ".clean16k.wav"
#: 产物/输入文件名（与契约 §4.1/§4.2/§4.3 一致）
TRANSCRIPT_NAME = "transcript.json"
REPORT_NAME = "report.json"
EDIT_NAME = "edit.json"
#: 时间比较容差（秒）：段表/映射一律 1e-6 级
_EPS = 1e-6


class AsrError(RuntimeError):
    """分段转写的输入/产物不一致（缺缓存、时长对不上、段表非法等）。"""


# ---------------------------------------------------------------- 读写小工具
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AsrError(f"读取失败：{path}：{exc}") from exc
    except ValueError as exc:
        raise AsrError(f"JSON 解析失败：{path}：{exc}") from exc


def _write_json(path: Path, payload: Any) -> Path:
    """原子写 UTF-8 JSON（复用 P3 的 ``export.write_json``：无 NaN/Inf、tmp 后 replace）。

    延迟导入：``export`` → ``analyze`` → ``asr`` 存在模块级循环，放到函数内只解析一次。
    """
    from .export import write_json

    return write_json(path, payload)


def _notify(on_stage: Any, status: str, metrics: Optional[Dict[str, Any]] = None) -> None:
    """调用 ``on_stage("asr", status, metrics)``；回调异常不影响主流程。"""
    if on_stage is None:
        return
    try:
        on_stage("asr", status, metrics or {})
    except Exception:  # pragma: no cover - 回调失败只打 debug
        log.debug("on_stage 回调失败", exc_info=True)


def _cfg_for_dirs(cfg: Any, out_dir: Optional[str | Path], cache_dir: Optional[str | Path]) -> Any:
    """把 ``out_dir``/``cache_dir`` 参数叠加到配置视图（让 decode/写盘走同一路径）。"""
    if out_dir is None and cache_dir is None:
        return cfg
    overrides: Dict[str, Any] = {}
    if out_dir is not None:
        overrides["paths.out_dir"] = str(Path(out_dir))
    if cache_dir is not None:
        overrides["paths.cache_dir"] = str(Path(cache_dir))
    replace = getattr(cfg, "replace", None)
    if callable(replace):
        try:
            return replace(overrides)
        except Exception:  # pragma: no cover - 非标准配置对象时降级
            log.debug("配置路径覆盖失败，退回原始配置", exc_info=True)
    return cfg


def _out_root(cfg: Any, out_dir: Optional[str | Path]) -> Path:
    return Path(out_dir) if out_dir is not None else cfg.path("out_dir")


def _cache_root(cfg: Any, cache_dir: Optional[str | Path]) -> Path:
    return Path(cache_dir) if cache_dir is not None else cfg.path("cache_dir")


# ---------------------------------------------------------------- 输入解析（analyze 缓存 / edit.json）
def _extract_segments(analyze_result: Any) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    """从 ``AnalyzeResult`` 或 analyze 缓存 dict 里取 orig 轴 VAD 段表与源时长。"""
    if isinstance(analyze_result, AnalyzeResult):
        raw: Any = analyze_result.segments
        duration: Optional[float] = float(analyze_result.duration_s or 0.0) or None
    elif isinstance(analyze_result, dict):
        result = analyze_result.get("result")
        data = result if isinstance(result, dict) else analyze_result
        raw = data.get("segments")
        try:
            duration = float(data.get("duration_s") or 0.0) or None
        except (TypeError, ValueError):
            duration = None
    else:  # pragma: no cover - 调用方传错类型
        raise AsrError(f"analyze_result 类型非法：{type(analyze_result).__name__}")
    if raw is None:
        raise AsrError("analyze 结果里没有 segments（缓存不完整或不是 P2 analyze 产物）")
    segments: List[Tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise AsrError(f"analyze 段格式非法（需要 [start, end]）：{item!r}")
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError) as exc:
            raise AsrError(f"analyze 段时间非数值：{item!r}") from exc
        if not (math.isfinite(start) and math.isfinite(end)):
            raise AsrError(f"analyze 段时间非有限值：{item!r}")
        if end <= start:
            continue
        segments.append((start, end))
    return segments, duration


def _extract_kept(kept: Any) -> List[KeptSpan]:
    """把 ``edit.json.kept``（四元组列表 / KeptSpan 列表 / edit dict）转成 KeptSpan 并校验。"""
    if isinstance(kept, dict):
        kept = kept.get("kept")
    if not isinstance(kept, (list, tuple)) or not kept:
        raise AsrError("edit.json.kept 为空（先跑 render 生成裁剪映射）")
    spans: List[KeptSpan] = []
    for item in kept:
        if isinstance(item, KeptSpan):
            spans.append(item)
            continue
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            raise AsrError(f"kept 四元组格式非法（需要 [orig_s,orig_e,clean_s,clean_e]）：{item!r}")
        try:
            values = [float(item[i]) for i in range(4)]
        except (TypeError, ValueError) as exc:
            raise AsrError(f"kept 四元组非数值：{item!r}") from exc
        orig_s, orig_e, clean_s, clean_e = values
        if not all(math.isfinite(v) for v in values):
            raise AsrError(f"kept 四元组含非有限值：{item!r}")
        if orig_e <= orig_s or clean_e <= clean_s:
            raise AsrError(f"kept 四元组区间非法（end<=start）：{item!r}")
        spans.append(KeptSpan(orig_s, orig_e, clean_s, clean_e))

    spans.sort(key=lambda k: k.orig_start)
    prev: Optional[KeptSpan] = None
    for span in spans:
        if prev is not None:
            if span.orig_start < prev.orig_end - _EPS:
                raise AsrError(
                    f"kept 原始轴重叠：{prev.as_list()} 与 {span.as_list()}"
                )
            if abs(span.clean_start - prev.clean_end) > 1e-3:
                raise AsrError(
                    f"kept clean 轴不连续：{prev.as_list()} 与 {span.as_list()}"
                )
        prev = span
    return spans


def _build_table(
    analyze_result: Any,
    kept: Any,
    cfg: Any = None,
    merge_gap_s: Optional[float] = None,
) -> Tuple[List[Tuple[float, float]], TimeMap]:
    """段表构建核心：orig 段 → clean 段（跨切口拆分）→ 过滤/切长/可选合并。"""
    segments, duration = _extract_segments(analyze_result)
    spans = _extract_kept(kept)
    timemap = TimeMap(spans, orig_duration_s=duration)

    min_s = float(cfg.get("asr.segment_min_s", 0.5)) if cfg is not None else 0.5
    max_s = float(cfg.get("asr.segment_max_s", 30.0)) if cfg is not None else 30.0
    if not math.isfinite(min_s) or min_s < 0:
        min_s = 0.0
    if not math.isfinite(max_s) or max_s <= 0:
        max_s = 30.0

    rows: List[Tuple[float, float]] = []
    for orig_s, orig_e in segments:
        for clean_s, clean_e in timemap.orig_span_to_clean(orig_s, orig_e):
            for start, end in split_long_segment(clean_s, clean_e, max_s):
                if end - start < min_s - _EPS:
                    continue
                rows.append((float(start), float(end)))
    rows.sort(key=lambda r: r[0])

    # 可选：合并 gap<=merge_gap_s 的相邻段（默认 None = 不合并，不为凑段数丢/并段）
    if merge_gap_s is not None:
        gap_limit = float(merge_gap_s)
        if not math.isfinite(gap_limit) or gap_limit < 0:
            raise AsrError(f"merge_gap_s 非法（须为 >=0 的有限数）：{merge_gap_s!r}")
        merged: List[Tuple[float, float]] = []
        for start, end in rows:
            if (
                merged
                and (start - merged[-1][1]) <= gap_limit + _EPS
                and (end - merged[-1][0]) <= max_s + _EPS
            ):
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        rows = merged

    # 兜底：段不得重叠、按 start 排序、丢掉零长段
    table: List[Tuple[float, float]] = []
    for start, end in rows:
        if end <= start + _EPS:
            continue
        if table and start < table[-1][1] - _EPS:
            log.debug("段表重叠已裁剪：%.3f→%.3f（前段结束 %.3f）", start, table[-1][1], table[-1][1])
            start = table[-1][1]
        if end - start <= _EPS:
            continue
        table.append((start, end))
    return table, timemap


def build_segment_table(
    analyze_result: Any,
    kept: Any,
    cfg: Any = None,
    merge_gap_s: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """P2 analyze 段（orig 轴）→ clean 轴转写段表（P4-T2）。

    Parameters
    ----------
    analyze_result : ``AnalyzeResult`` 或 ``work/cache/<id>.analyze.json`` 的内容
        （``{"ok":..,"wav":..,"result":{"segments":[[s,e],...]}}``，两种都接受）。
    kept : ``edit.json``（dict）或其 ``kept`` 四元组列表 / ``KeptSpan`` 列表。
    cfg : 配置视图；取 ``asr.segment_min_s``（默认 0.5）与 ``asr.segment_max_s``（默认 30）。
    merge_gap_s : 默认 ``None`` = 不合并；显式传入时合并 gap<=该值且合并后 <=max_s 的相邻段。

    Returns
    -------
    list[(start, end)]：clean 轴、按 start 升序、互不重叠；短于 ``segment_min_s`` 的碎片
    已丢弃，长于 ``segment_max_s`` 的段已用 :func:`split_long_segment` 均分。
    """
    table, _timemap = _build_table(analyze_result, kept, cfg, merge_gap_s)
    return table


# ---------------------------------------------------------------- clean 音轨 16k 解码
def decode_clean16k(job_id: str, cfg: Any, out_dir: Optional[str | Path] = None) -> Path:
    """把 ``out/<id>/clean.m4a`` 解码成 ``work/cache/<id>.clean16k.wav``（mono 16k float32）。

    - 用 ffmpeg 走 P1 的 ``decode_audio``（``-ac 1 -ar 16000 -c:a pcm_f32le``）；
    - 校验解码时长与 ``report.trim.clean_dur_s`` 的偏差 < 0.1 s（AAC 编码器 padding 会带来
      零点几毫秒到几十毫秒的偏差，**这是需要记录的现象，不是错误**）；
    - 只写独立文件名 ``.clean16k.wav``，**不调用** ``decode.cleanup_cache``，不碰
      ``.16k.wav`` / ``.analyze.json`` / ``.decode.json``。
    """
    out_root = _out_root(cfg, out_dir)
    job_out = out_root / job_id
    clean_m4a = resolve_product(job_out, out_root, "clean.m4a")
    if clean_m4a is None:
        raise AsrError(f"clean.m4a 不存在（先跑 render）：{job_out / 'clean.m4a'}")

    # 预期时长：report.trim.clean_dur_s（P3 实测 kept 差）
    expected: Optional[float] = None
    report_path = job_out / REPORT_NAME
    if report_path.is_file():
        try:
            report = _read_json(report_path)
            if isinstance(report, dict):
                value = (report.get("trim") or {}).get("clean_dur_s")
                if value is not None and math.isfinite(float(value)):
                    expected = float(value)
        except AsrError:
            log.warning("report.json 不可读，跳过 clean16k 时长校验：%s", report_path)
    else:
        log.warning("report.json 不存在，跳过 clean16k 时长校验：%s", report_path)

    dst = _cache_root(cfg, None) / f"{job_id}{CLEAN16K_SUFFIX}"
    started = time.perf_counter()
    decode_audio(clean_m4a, dst, sample_rate=SR, channels=1, sample_fmt="pcm_f32le")
    elapsed = time.perf_counter() - started

    info = wav_info(dst)
    if info.sample_rate != SR or info.channels != 1:
        raise AsrError(
            f"clean16k 格式不符：sr={info.sample_rate} channels={info.channels}（期望 16000/mono）"
        )
    actual = info.n_frames / float(info.sample_rate)
    if expected is not None:
        deviation = actual - expected
        log.info(
            "clean16k 解码：%s 实际 %.3fs vs report.trim.clean_dur_s %.3fs，"
            "AAC padding 偏差 %+.3fs（阈值 ±0.1s，解码 %.1fs）",
            dst.name, actual, expected, deviation, elapsed,
        )
        if abs(deviation) >= 0.1:
            raise AsrError(
                f"clean16k 时长与 report.trim.clean_dur_s 偏差 {deviation:+.3f}s（>=0.1s）："
                f"{dst}；请检查 clean.m4a/report 是否同一次 render 产物"
            )
    else:
        log.info("clean16k 解码完成：%s（%.3fs，解码 %.1fs；无时长基准可校验）",
                 dst.name, actual, elapsed)
    return dst


# ---------------------------------------------------------------- report 读-改-写
def _update_report(report_path: Path, metrics: Dict[str, Any], elapsed_s: float) -> Dict[str, Any]:
    """在既有 ``report.json`` 上追加 ``asr`` / ``speakers`` / ``timing.asr_s``，保留全部既有字段。"""
    if not report_path.is_file():
        raise AsrError(f"report.json 不存在（先跑 render）：{report_path}")
    report = _read_json(report_path)
    if not isinstance(report, dict):
        raise AsrError(f"report.json 顶层不是对象：{report_path}")
    report["asr"] = dict(metrics)
    # P4-T3 默认单说话人；diarize 阶段（sampled/full）会用更可信的结论覆盖
    report["speakers"] = {"n": 1, "method": "none"}
    timing = report.get("timing")
    if not isinstance(timing, dict):
        timing = {}
    timing["asr_s"] = round(float(elapsed_s), 2)
    report["timing"] = timing
    _write_json(report_path, report)
    return report


# ---------------------------------------------------------------- 分段转写（P4-T2）
def transcribe_job(
    job_id: str,
    cfg: Any,
    *,
    out_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    force: bool = False,
    merge_gap_s: Optional[float] = None,
    on_stage: Optional[Any] = None,
) -> Dict[str, Any]:
    """对单个 job 做「clean 轴分段 SenseVoice 转写 + clean/orig 双时间轴」。

    流程：读 ``edit.json.kept``（权威）+ ``<id>.analyze.json``（orig 轴 VAD 段）→
    ``build_segment_table`` → ``decode_clean16k`` → 逐段 ``transcribe_raw``（单例 recognizer）
    → 写 ``transcript.json``（§4.3，顶层 list）→ ``report.json`` 读-改-写追加
    ``asr``/``speakers``/``timing.asr_s``。

    口径：
    - ``start/end`` = clean 轴段边界（round 3）；``orig_start/orig_end`` 由 ``TimeMap.clean_to_orig``
      反查（往返误差 <1 ms，超限只 warn 不失败）；
    - 纯标点/空文本**不写入** ``transcript.json``，计入 ``n_empty``；``n_segments`` 是段表总数
      （= 写出的条数 + ``n_empty``），与 P4 基准 1175 同口径；
    - ``chars`` 用 :func:`count_speech_chars` 口径；``rtf`` = 逐段解码耗时 / clean 时长
      （不含模型加载与 clean 解码，与 PoC/bench 的 RTF 口径一致）。

    幂等/失败：已有 ``transcript.json`` 且 report 有 ``asr`` 时默认跳过（``force`` 重做）；
    失败**不抛异常**，返回 ``{"ok": False, "stage": "asr", "error": ...}``。
    ``on_stage("asr", "done"/"failed", metrics)`` 供编排器/CLI 落库。
    """
    started = time.perf_counter()
    work_cfg = _cfg_for_dirs(cfg, out_dir, cache_dir)
    job_out = _out_root(work_cfg, None) / job_id
    cache_root = _cache_root(work_cfg, None)
    transcript_path = job_out / TRANSCRIPT_NAME
    report_path = job_out / REPORT_NAME

    result: Dict[str, Any] = {
        "ok": False, "job_id": job_id, "source": None, "out_dir": str(job_out),
        "skipped": False, "stage": "asr", "error": "", "report": None,
        "transcript": None, "asr": None, "timings": {},
    }

    # 1) 幂等：已有 transcript.json 且 report 有 asr → 跳过
    if not force and transcript_path.is_file() and report_path.is_file():
        old_report: Any = None
        try:
            old_report = _read_json(report_path)
            _read_json(transcript_path)  # 产物可解析才认；坏了就重做
        except AsrError:
            old_report = None
        if isinstance(old_report, dict) and isinstance(old_report.get("asr"), dict):
            metrics = dict(old_report["asr"])
            result.update({
                "ok": True, "skipped": True, "stage": "asr", "error": "",
                "report": old_report, "transcript": str(transcript_path),
                "asr": metrics, "source": old_report.get("source"),
                "timings": {"asr_s": float(metrics.get("elapsed_s") or 0.0)},
            })
            log.info("转写产物已存在，跳过：%s（segs=%s）", job_id, metrics.get("n_segments"))
            _notify(on_stage, "done", {**metrics, "skipped": True})
            return result

    try:
        # 2) 输入产物
        if not job_out.is_dir():
            raise AsrError(f"job 输出目录不存在（先跑 render）：{job_out}")
        if not report_path.is_file():
            raise AsrError(f"report.json 不存在（先跑 render）：{report_path}")
        edit_path = job_out / EDIT_NAME
        if not edit_path.is_file():
            raise AsrError(f"edit.json 不存在（先跑 render 生成裁剪映射）：{edit_path}")
        edit = _read_json(edit_path)
        kept = edit.get("kept") if isinstance(edit, dict) else edit
        analyze_path = cache_root / f"{job_id}.analyze.json"
        if not analyze_path.is_file():
            raise AsrError(f"analyze 缓存缺失（先跑 analyze/render）：{analyze_path}")
        analyze_data = _read_json(analyze_path)
        if isinstance(analyze_data, dict) and analyze_data.get("ok") is False:
            raise AsrError(f"analyze 缓存标记失败：{analyze_path}")

        # 3) 段表（orig 轴 VAD 段 → clean 轴；过滤/切长/可选合并）
        table, timemap = _build_table(analyze_data, kept, cfg, merge_gap_s)
        if not table:
            raise AsrError(f"段表为空：analyze 段与 kept 无交集（{job_id}）")
        log.info("段表：%s %d 段（merge_gap_s=%s）", job_id, len(table), merge_gap_s)

        # 4) clean 音轨 16k（独立缓存文件）
        clean16k = decode_clean16k(job_id, work_cfg)
        sr, audio = read_wav_float32(clean16k)
        if sr != SR:
            raise AsrError(f"clean16k 采样率 {sr} != {SR}：{clean16k}")
        clean_dur = len(audio) / float(sr)
        log.info("clean16k 已载入：%s %.3fs（float32 %.0f MB）",
                 clean16k.name, clean_dur, audio.nbytes / 1024 / 1024)

        # 5) 逐段转写（单例 recognizer；内存只保留当前段）
        transcriber = get_transcriber(work_cfg)
        segments: List[TranscriptSegment] = []
        n_empty = 0
        chars = 0
        decode_s = 0.0
        max_roundtrip_err = 0.0
        t_loop = time.perf_counter()
        for idx, (seg_start, seg_end) in enumerate(table, 1):
            clip = slice_audio(audio, seg_start, seg_end, sr)
            if clip.size == 0:
                n_empty += 1
                continue
            t0 = time.perf_counter()
            text, text_raw = transcriber.transcribe_raw(clip, sr)
            decode_s += time.perf_counter() - t0
            del clip
            if count_speech_chars(text) == 0:  # 纯标点/空文本：不写入 transcript.json
                n_empty += 1
                continue
            start_r = round(float(seg_start), 3)
            end_r = round(float(seg_end), 3)
            orig_start = timemap.clean_to_orig(start_r)
            orig_end = timemap.clean_to_orig(end_r)
            if orig_start is None or orig_end is None:
                raise AsrError(
                    f"clean→orig 映射失败：seg[{idx}] {start_r}-{end_r}s"
                    f"（clean 时长 {timemap.clean_duration_s:.3f}s）"
                )
            segment = TranscriptSegment(
                start=start_r, end=end_r,
                orig_start=round(float(orig_start), 3), orig_end=round(float(orig_end), 3),
                speaker=SPEAKER_MAIN, text_raw=text_raw, text=text, flags=[],
            )
            back = timemap.orig_to_clean(segment.orig_start)
            if back is not None:
                max_roundtrip_err = max(max_roundtrip_err, abs(back - start_r))
            segments.append(segment)
            chars += count_speech_chars(text)
            if idx % 100 == 0:
                log.info("  转写进度 %d/%d（%.1fs）", idx, len(table),
                         time.perf_counter() - t_loop)
        del audio
        loop_s = time.perf_counter() - t_loop
        segments.sort(key=lambda s: s.start)
        if max_roundtrip_err > 1e-3:  # 契约：往返误差 <1ms
            log.warning("clean/orig 往返误差最大 %.6fs（>1ms）：%s", max_roundtrip_err, job_id)

        # 6) transcript.json（§4.3：顶层 list、按 start 排序、UTF-8、无 NaN、原子替换）
        _write_json(transcript_path, [seg.to_dict() for seg in segments])

        # 7) report.json 读-改-写（保留既有字段与 ok）
        elapsed = time.perf_counter() - started
        rtf = (decode_s / clean_dur) if clean_dur > 0 else 0.0
        metrics: Dict[str, Any] = {
            "engine": str(cfg.get("asr.engine", "sensevoice")),
            "model": str(cfg.get("asr.model", "int8")),
            "n_segments": int(len(table)),
            "chars": int(chars),
            "n_empty": int(n_empty),
            "rtf": round(float(rtf), 4),
            "elapsed_s": round(float(elapsed), 2),
        }
        report = _update_report(report_path, metrics, elapsed)
        result.update({
            "ok": True, "skipped": False, "stage": "asr", "error": "",
            "report": report, "transcript": str(transcript_path),
            "asr": metrics, "source": report.get("source"),
            "timings": {"asr_s": round(float(elapsed), 2)},
        })
        log.info(
            "转写完成：%s 段=%d（空 %d）字符=%d rtf=%.4f 逐段解码=%.1fs 总耗时=%.1fs → %s",
            job_id, len(table), n_empty, chars, rtf, loop_s, elapsed, transcript_path.name,
        )
        _notify(on_stage, "done", dict(metrics))
        return result

    except Exception as exc:
        msg = str(exc) or exc.__class__.__name__
        log.error("转写失败（asr）：%s：%s", job_id, msg)
        _notify(on_stage, "failed", {"error": msg})
        result.update({"ok": False, "skipped": False, "stage": "asr", "error": msg})
        return result
