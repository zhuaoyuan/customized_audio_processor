# -*- coding: utf-8 -*-
"""SenseVoice 转写封装（P4-T1，P2-T3 头尾边界判定先依赖它）。

为什么提前到这里
----------------
实施计划 P2-T3 明确「按 VAD 段逐段送 SenseVoice（**复用 P4-T1 封装**）」，所以本模块
在 P2 阶段先行落地最小可用版：**单例 recognizer + 单段/批量转写 + 文本清洗**。P4 阶段
只需在其上补「分段与时间戳组装」，不需要改这里的接口。

关键约束
--------
- recognizer 加载 1–3 s，**必须复用**（`get_transcriber` 做进程内单例），禁止每段重建；
- SenseVoice 会输出 ``<|zh|><|NEUTRAL|>`` 等标签与表情符号，必须清洗后再统计字符；
- 模型文件缺失要抛明确异常（`AsrUnavailable`），由调用方降级并记 flag，而不是静默出空文本。
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_setup import get_logger

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
