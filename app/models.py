# -*- coding: utf-8 -*-
"""数据模型与常量（字段名以 ``docs/实施计划.md`` §4「数据契约」为准）。

本模块只放**纯数据**与**id 生成**，不碰数据库、不碰文件系统，
便于单测与后续阶段（analyze/trim/asr/export）复用。
"""

from __future__ import annotations

import hashlib
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- 状态常量（契约 §4.5）
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

JOB_STATUSES = (STATUS_PENDING, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED)
STAGE_STATUSES = JOB_STATUSES

#: 流水线阶段顺序（P6-T1）
STAGE_DECODE = "decode"
STAGE_ANALYZE = "analyze"
STAGE_DENOISE = "denoise"
STAGE_TRIM = "trim"
STAGE_LOUDNESS = "loudness"
STAGE_ASR = "asr"
STAGE_DIARIZE = "diarize"
STAGE_CORRECT = "correct"
STAGE_EXPORT = "export"
STAGES = (
    STAGE_DECODE, STAGE_ANALYZE, STAGE_DENOISE, STAGE_TRIM, STAGE_LOUDNESS,
    STAGE_ASR, STAGE_DIARIZE, STAGE_CORRECT, STAGE_EXPORT,
)

#: 断点续跑时需要重新拿起来的 job 状态（running = 上次被中断）
RESUMABLE_JOB_STATUSES = (STATUS_PENDING, STATUS_RUNNING, STATUS_FAILED)

#: 裁剪删除原因（契约 §4.2 reason_enum）
REMOVE_HEAD = "head_nonspeech"
REMOVE_TAIL = "tail_nonspeech"
REMOVE_INTERNAL_SILENCE = "internal_silence"
REMOVE_OTHER_SPEAKER = "other_speaker_low_chars"
REMOVE_VAD_EMPTY_ASR = "vad_speech_asr_empty"
REMOVE_REASONS = (
    REMOVE_HEAD, REMOVE_TAIL, REMOVE_INTERNAL_SILENCE,
    REMOVE_OTHER_SPEAKER, REMOVE_VAD_EMPTY_ASR,
)

#: 默认发音人标签
SPEAKER_MAIN = "主讲"

#: 转写条目 flags
FLAG_REJECTED_CHUNK = "rejected_chunk"
FLAG_UNCERTAIN_SPEAKER = "uncertain_speaker"

#: 媒体扩展名（与 configs/default.yaml scan.extensions 保持一致的兜底值）
DEFAULT_EXTENSIONS = (
    ".m4a", ".mp3", ".wav", ".mp4", ".flac", ".aac", ".wma", ".ogg", ".opus", ".webm",
)

#: slug 里要丢弃的文件名噪声词（下载站常见的后缀/占位词）
_SLUG_NOISE = {
    "normal", "final", "new", "copy", "audio", "video", "online", "download",
    "mp3", "m4a", "wav", "flac", "mp4", "aac",
}

_ID_SLUG_MAX = 20
_ID_SLUG_MIN = 2


# ---------------------------------------------------------------- 通用小工具
def now_ts() -> int:
    """当前 Unix 时间戳（秒，UTC）。"""
    return int(time.time())


def _ascii_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _keep_char(ch: str) -> bool:
    """保留中/日/韩文字与 ASCII 字母数字；其余（空格、括号、点、下划线）统一当分隔符。"""
    if _ascii_alnum(ch):
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") and not ch.isascii()


def slugify(name: str, max_len: int = _ID_SLUG_MAX) -> str:
    """把文件名主体转成可读短名：保留中日韩文字与 ASCII 字母数字，按 token 边界截断。"""
    text = unicodedata.normalize("NFKC", name or "")
    tokens: List[str] = []
    buf: List[str] = []
    for ch in text:
        if _keep_char(ch):
            buf.append(ch.lower() if ch.isascii() else ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))

    # 去掉纯噪声词（但保证至少留一个 token）
    cleaned = [t for t in tokens if t not in _SLUG_NOISE]
    if not cleaned:
        cleaned = tokens or ["audio"]

    out = ""
    for token in cleaned:
        candidate = token if not out else f"{out}_{token}"
        if len(candidate) > max_len:
            if len(out) >= _ID_SLUG_MIN:
                break
            # 第一个 token 太长：硬截断
            out = token[:max_len]
            break
        out = candidate
    out = out.strip("_")
    if len(out) < _ID_SLUG_MIN:
        out = "audio"
    return out


def file_hash8(path: str | Path) -> str:
    """文件名（basename）的 sha1 前 8 位——移动目录不变，便于稳定 id。"""
    name = Path(path).name
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]


def make_job_id(path: str | Path, taken: Optional[Dict[str, str]] = None) -> str:
    """由文件名生成稳定 job id：``<hash8>_<可读短名>``。

    ``taken`` 传入 {id: source}；若 id 已被**另一个**源文件占用，则追加 4 位全路径 hash 区分，
    保证同名文件放在不同目录时也能各自成 job。
    """
    p = Path(path)
    stem = p.stem
    # ``xxx.m4a`` 之外，去掉结尾的 ``.`` 造成的空扩展（样例里有 ``_normal..m4a``）
    base_id = f"{file_hash8(p)}_{slugify(stem)}"
    if taken:
        owner = taken.get(base_id)
        if owner is not None and Path(owner) != p:
            extra = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:4]
            base_id = f"{base_id}_{extra}"
    return base_id


# ---------------------------------------------------------------- job / stage
@dataclass
class Job:
    """一个输入文件对应的任务（契约 §4.5 jobs 表）。"""

    id: str
    source: str
    status: str = STATUS_PENDING
    stage: str = ""
    error: str = ""
    attempts: int = 0
    created_at: int = 0
    updated_at: int = 0
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = now_ts()
        if not self.updated_at:
            self.updated_at = self.created_at

    @classmethod
    def new(cls, job_id: str, source: str, config_hash: str = "") -> "Job":
        return cls(id=job_id, source=str(source), config_hash=config_hash)

    @classmethod
    def from_row(cls, row: Any) -> "Job":
        """从 sqlite3.Row / mapping 构造。"""
        keys = row.keys() if hasattr(row, "keys") else row
        get = row.__getitem__
        return cls(
            id=get("id"),
            source=get("source"),
            status=get("status") or STATUS_PENDING,
            stage=(get("stage") if "stage" in keys else "") or "",
            error=(get("error") if "error" in keys else "") or "",
            attempts=int(get("attempts") if "attempts" in keys else 0) or 0,
            created_at=int(get("created_at") if "created_at" in keys else 0) or 0,
            updated_at=int(get("updated_at") if "updated_at" in keys else 0) or 0,
            config_hash=(get("config_hash") if "config_hash" in keys else "") or "",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_hash": self.config_hash,
        }


@dataclass
class StageResult:
    """单个阶段的执行结果（契约 §4.5 stages 表；stage 级 error 落在 metrics.error）。"""

    job_id: str
    stage: str
    status: str = STATUS_PENDING
    started_at: Optional[int] = None
    ended_at: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return float(self.ended_at - self.started_at)

    @classmethod
    def from_row(cls, row: Any) -> "StageResult":
        keys = row.keys() if hasattr(row, "keys") else row
        import json

        metrics: Dict[str, Any] = {}
        raw = row["metrics_json"] if "metrics_json" in keys else "{}"
        if raw:
            try:
                loaded = json.loads(raw)
                metrics = loaded if isinstance(loaded, dict) else {"value": loaded}
            except (TypeError, ValueError):
                metrics = {"raw": str(raw)}
        error = ""
        if isinstance(metrics.get("error"), str):
            error = metrics["error"]
        return cls(
            job_id=row["job_id"],
            stage=row["stage"],
            status=row["status"],
            started_at=row["started_at"] if "started_at" in keys else None,
            ended_at=row["ended_at"] if "ended_at" in keys else None,
            metrics=metrics,
            error=error,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------- 解码结果
@dataclass
class DecodeResult:
    """``app.pipeline.decode.decode()`` 的返回值（P1-T2 契约字段 + 诊断附加字段）。"""

    ok: bool
    wav16k: Optional[str] = None
    wav44k: Optional[str] = None
    duration_s: float = 0.0
    sample_rate: int = 0
    orig_sample_rate: int = 0
    channels: int = 0
    bit_rate: int = 0
    repaired: bool = False
    decode_source: str = ""
    codec: str = ""
    format_name: str = ""
    from_cache: bool = False
    error: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    repair: Dict[str, Any] = field(default_factory=dict)
    job_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "ok": self.ok,
            "wav16k": self.wav16k,
            "wav44k": self.wav44k,
            "duration_s": round(float(self.duration_s), 3),
            "sample_rate": self.sample_rate,
            "orig_sample_rate": self.orig_sample_rate,
            "channels": self.channels,
            "bit_rate": self.bit_rate,
            "codec": self.codec,
            "format_name": self.format_name,
            "repaired": self.repaired,
            "decode_source": self.decode_source,
            "from_cache": self.from_cache,
            "error": self.error,
            "timings": {k: round(float(v), 3) for k, v in self.timings.items()},
            "repair": self.repair,
        }

    def report_decode(self) -> Dict[str, Any]:
        """report.json 的 ``decode`` 段（契约 §4.1）。"""
        return {"repaired": self.repaired, "ok": self.ok}


# ---------------------------------------------------------------- 编辑映射（契约 §4.2）
@dataclass
class KeptSpan:
    """``edit.json.kept`` 四元组：[orig_start, orig_end, clean_start, clean_end]。"""

    orig_start: float
    orig_end: float
    clean_start: float
    clean_end: float

    def as_list(self) -> List[float]:
        return [
            round(self.orig_start, 3), round(self.orig_end, 3),
            round(self.clean_start, 3), round(self.clean_end, 3),
        ]


@dataclass
class RemovedSpan:
    orig_start: float
    orig_end: float
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        if self.reason not in REMOVE_REASONS:
            raise ValueError(f"非法删除原因：{self.reason}，可选：{REMOVE_REASONS}")
        return {
            "orig_start": round(self.orig_start, 3),
            "orig_end": round(self.orig_end, 3),
            "reason": self.reason,
        }

    @property
    def duration_s(self) -> float:
        return max(0.0, self.orig_end - self.orig_start)


@dataclass
class GainInfo:
    """``edit.json.gain``（契约 §4.2）。"""

    type: str = "static"
    gain_db: float = 0.0
    min_db: float = 0.0
    max_db: float = 0.0
    max_rate_db_s: float = 0.0
    limiter_triggers: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "gain_db": round(float(self.gain_db), 2),
            "min_db": round(float(self.min_db), 2),
            "max_db": round(float(self.max_db), 2),
            "max_rate_db_s": round(float(self.max_rate_db_s), 2),
            "limiter_triggers": int(self.limiter_triggers),
        }


@dataclass
class EditMap:
    """``edit.json`` 整体（P3-T2 产出）。"""

    kept: List[KeptSpan] = field(default_factory=list)
    removed: List[RemovedSpan] = field(default_factory=list)
    gain: Optional[GainInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kept": [k.as_list() for k in self.kept],
            "removed": [r.as_dict() for r in self.removed],
            "reason_enum": list(REMOVE_REASONS),
        }
        if self.gain is not None:
            out["gain"] = self.gain.as_dict()
        return out


@dataclass
class TranscriptSegment:
    """``transcript.json`` 单条（契约 §4.3）。"""

    start: float
    end: float
    orig_start: float
    orig_end: float
    speaker: str = SPEAKER_MAIN
    text_raw: str = ""
    text: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "orig_start": round(float(self.orig_start), 3),
            "orig_end": round(float(self.orig_end), 3),
            "speaker": self.speaker,
            "text_raw": self.text_raw,
            "text": self.text,
            "flags": list(self.flags),
        }
