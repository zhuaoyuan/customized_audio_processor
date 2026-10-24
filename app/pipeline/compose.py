# -*- coding: utf-8 -*-
"""P10 全文成稿：以整讲为单位建立全局语境，再把 ASR 口语转写整理为书面稿。

与 ``correct`` 的区别
--------------------
- ``correct`` 只改同音/近音错字，逐块 2,000–2,500 字，禁止润色与合并句子；
- ``compose`` 做「成稿」：先通读**整讲**生成课程档案（结构大纲 + 术语表），
  再按约 1,800 字分块整理为完整、通顺的书面讲稿，允许合并碎片句、删除语气词、
  按术语表纠错；实质内容、数字、时间轴必须保留。

流程
----
1. ``analyze_lecture``：全讲逐句转写 → 课程档案（1 次调用，JSON）；
2. ``plan_chunks``：按段边界切块（块间带 2 段重叠，用于衔接）；
3. ``compose_chunk``：每块带课程档案 + 前文原文结尾 → 书面自然段（可并发）；
4. ``_dedupe_*``：去掉块间重叠造成的重复；``compose_job`` 汇总校验并落盘。

产物（``out/<job>/``）
--------------------
- ``manuscript.md``：成稿（段首 ``[mm:ss]`` 为 **clean 音轨** 时间；原始轴见 JSON ``orig_time``）；
- ``manuscript.json``：段落/时间/来源块 + 统计；
- ``compose_card.json``：课程档案（缓存，transcript 变化即失效）；
- ``compose_report.json``：分块校验、回退、用量与成本。

约定
----
- 本模块**不在** ``run`` 全链路里（不改变现有 9 阶段），由 ``cli compose`` 显式调用；
- API 复用 ``correct`` 的 key/base_url/model（可用 ``compose.*`` 覆盖）；
- 任何一块失败都不阻塞整篇：回退该块为「机械去语气词」并标 ``fallback``；
- 绝不访问真实 API 的单测通过 monkeypatch ``chat_completion`` 完成。
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..logging_setup import get_logger
from .correct import (
    CorrectError,
    chat_completion,
    compute_cost,
    load_hotwords,
    resolve_api_key,
)

log = get_logger(__name__)

#: 成稿算法版本（变更 prompt/分块/校验时递增，旧缓存自动失效）
COMPOSE_VERSION = 1

#: 无 prompt 文件时的兜底 system prompt（正常情况读 configs/prompts/compose_system.txt）
DEFAULT_SYSTEM_PROMPT = (
    "你是中文学术讲座录音转写稿的整理者。把逐句 ASR 转写整理成完整、通顺的书面讲稿；"
    "只整理原文已有内容，不添加原文没有的观点、例证或引文；删除语气词与无意义重复，"
    "合并碎片句，按术语表纠正同音错字；保留实质内容、例子、数字、书名与引文；"
    "每个自然段以输入行中的 [mm:ss] 时间戳开头。"
)

DEFAULT_ANALYZE_PROMPT = (
    "你是中文学术讲座转写稿的编辑。通读整讲转写，只输出 JSON："
    '{"title": string, "topic": string, "speaker": string, '
    '"outline": [{"time": "mm:ss", "title": string}], '
    '"glossary": [{"correct": string, "variants": [string], "note": string}], '
    '"people": [string], "books": [string], "notes": string}。'
    "outline 8–15 条；glossary 20–50 条，variants 必须是原文真实出现的错写。"
)

_FILLER_RE = re.compile(r"(?:呃+|啊+|嗯+|这个这个|就是就是|那个那个)+")
_SPACE_RE = re.compile(r"\s+")
_TIME_RE = re.compile(r"^\s*\[(\d{1,3}:\d{2})\]\s*")
_NUM_RE = re.compile(r"\d+")
_MD_PREFIX_RE = re.compile(r"^(?:[-*#>]+\s*|\*\*|\d+[.、]\s*)")
_SENT_RE = re.compile(r"[^。！？!?]+[。！？!?]?")


class ComposeError(RuntimeError):
    """成稿阶段错误（配置/输入/API/产物）。"""


# ================================================================ 配置小工具
def _get(cfg: Any, dotted: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        value = cfg.get(dotted, default)
    except AttributeError:  # 兼容测试里的假 cfg
        return default
    return default if value is None else value


def _root(cfg: Any) -> Path:
    root = getattr(cfg, "root", None)
    return Path(root) if root else Path(__file__).resolve().parents[2]


def _resolve_path(cfg: Any, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _root(cfg) / path


def _out_root(cfg: Any, out_dir: Optional[str | Path] = None) -> Path:
    if out_dir:
        return Path(out_dir)
    try:
        return Path(cfg.path("out_dir"))
    except Exception:  # pragma: no cover - 假 cfg
        return Path("out")


def _work_root(cfg: Any) -> Path:
    try:
        return Path(cfg.path("work_dir"))
    except Exception:  # pragma: no cover
        return _root(cfg) / "work"


# ================================================================ 时间与文本
def fmt_time(seconds: float) -> str:
    """秒 → ``mm:ss``（分钟不折成小时，与 transcript.md 口径一致）。"""
    total = max(0, int(round(float(seconds))))
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_time(stamp: str) -> int:
    mm, ss = str(stamp).strip().split(":")
    return int(mm) * 60 + int(ss)


def normalize_text(text: Optional[str]) -> str:
    """去掉时间戳前缀与全部空白，用于长度/数字口径比较。"""
    text = _TIME_RE.sub("", text or "", count=0)
    text = re.sub(r"\[?\d{1,3}:\d{2}\]?", "", text)
    return _SPACE_RE.sub("", text)


# ================================================================ 段落 / 输入
@dataclass
class Segment:
    """一段 ASR 转写（``index`` 保留原 transcript 数组下标）。

    ``start/end`` 是 **clean 音轨**（处理/裁剪后）时间，``orig_start/orig_end``
    是原始音轨时间；没有原始轴时回退到 clean 轴。
    """

    index: int
    start: float
    end: float
    text: str
    orig_start: Optional[float] = None
    orig_end: Optional[float] = None

    @property
    def orig_start_s(self) -> float:
        return float(self.orig_start if self.orig_start is not None else self.start)

    @property
    def orig_end_s(self) -> float:
        return float(self.orig_end if self.orig_end is not None else self.end)


def _seg_text(item: Dict[str, Any]) -> str:
    return str(item.get("text") or item.get("text_raw") or "").strip()


def load_segments(transcript_path: str | Path) -> List[Segment]:
    """读 ``transcript.json``（兼容 list / {"segments": [...]}）；跳过空段。"""
    path = Path(transcript_path)
    if not path.is_file():
        raise ComposeError(f"找不到 transcript.json：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ComposeError(f"读取 transcript.json 失败：{exc}") from exc
    items = raw
    if isinstance(raw, dict):
        items = raw.get("segments") or raw.get("items") or []
    if not isinstance(items, list):
        raise ComposeError("transcript.json 格式不对：segments 不是数组")
    out: List[Segment] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        text = _seg_text(item)
        if not text:
            continue
        start = item.get("start")
        if start is None:
            start = item.get("orig_start", 0.0)
        end = item.get("end")
        if end is None:
            end = item.get("orig_end", start)
        try:
            start_f = float(start)
            end_f = float(end)
            orig_start = item.get("orig_start")
            orig_end = item.get("orig_end")
            orig_start_f = float(orig_start) if orig_start is not None else start_f
            orig_end_f = float(orig_end) if orig_end is not None else max(start_f, end_f)
        except (TypeError, ValueError):
            continue
        out.append(Segment(index=i, start=start_f, end=max(start_f, end_f), text=text,
                           orig_start=orig_start_f, orig_end=max(orig_start_f, orig_end_f)))
    out.sort(key=lambda s: (s.start, s.index))
    return out


def source_lines(segments: Sequence[Segment]) -> List[str]:
    return [f"[{fmt_time(s.start)}] {s.text}" for s in segments]


def strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:markdown|md|json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _load_prompt(cfg: Any, key: str, filename: str, default: str) -> str:
    configured = str(_get(cfg, key, "") or "").strip()
    path = _resolve_path(cfg, configured or filename)
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        log.warning("成稿 prompt 文件读取失败，使用内置兜底：%s", path)
    return default


def _chat_with_deadline(
    cfg: Any,
    messages: Sequence[Dict[str, str]],
    *,
    timeout_s: float,
    **kwargs: Any,
) -> Dict[str, Any]:
    """带**硬截止**的 chat 调用。

    urllib 的 timeout 是单次 socket 操作超时；网关偶尔会持续发 keep-alive 字节，
    导致 300 s 的请求挂十几分钟。这里用 daemon 线程 + queue 做总时限，超时即失败，
    避免整篇成稿被一个卡死请求拖住（被放弃的线程会自行随 socket 超时退出）。
    """
    deadline = max(30.0, float(timeout_s) + 30.0)
    box: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            box.put(("ok", chat_completion(cfg, messages, timeout_s=timeout_s, **kwargs)))
        except BaseException as exc:  # noqa: BLE001 - 原样转交给主线程
            box.put(("err", exc))

    worker = threading.Thread(target=run, name="compose-chat", daemon=True)
    worker.start()
    try:
        status, payload = box.get(timeout=deadline)
    except queue.Empty as exc:
        raise CorrectError(f"请求超过硬截止 {deadline:.0f}s：{kwargs.get('model') or 'model'}") from exc
    if status == "err":
        raise payload
    return payload


# ================================================================ 课程档案
def build_analyze_messages(
    segments: Sequence[Segment],
    cfg: Any,
    *,
    job_id: str = "",
    source: str = "",
    hotwords: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    system = _load_prompt(
        cfg, "compose.analyze_prompt_file",
        "configs/prompts/compose_analyze_system.txt", DEFAULT_ANALYZE_PROMPT,
    )
    head = ["【整讲逐句转写】"]
    if source:
        head.append(f"文件：{Path(str(source).replace(chr(92), '/')).name}")
    if hotwords:
        head.append("已知热词：" + "、".join(str(w) for w in hotwords))
    head.append("")
    body = "\n".join(source_lines(segments))
    user = "\n".join(head) + body + "\n\n【输出 JSON】"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_card_json(text: str) -> Dict[str, Any]:
    """从模型输出里取第一个 JSON 对象；容忍 ```json 包裹与尾逗号。"""
    cleaned = strip_fences(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ComposeError(f"课程档案不是 JSON：{cleaned[:200]!r}")
    snippet = cleaned[start : end + 1]
    for candidate in (snippet, re.sub(r",(\s*[}\]])", r"\1", snippet)):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    raise ComposeError(f"课程档案 JSON 解析失败：{snippet[:200]!r}")


def _as_str(value: Any, limit: int = 300) -> str:
    return str(value).strip()[:limit] if value is not None else ""


def sanitize_card(card: Dict[str, Any], segments: Sequence[Segment], cfg: Any) -> Dict[str, Any]:
    """清洗模型输出：只保留真实存在的术语变体，时间戳吸附到真实段起点。"""
    corpus = "".join(s.text for s in segments)
    valid_starts = [s.start for s in segments]
    max_glossary = int(_get(cfg, "compose.max_glossary", 60))

    def snap_time(stamp: Any) -> str:
        raw = _as_str(stamp, 16)
        try:
            target = parse_time(raw)
        except (ValueError, TypeError):
            return ""
        if not valid_starts:
            return ""
        nearest = min(valid_starts, key=lambda s: abs(int(round(s)) - target))
        return fmt_time(nearest)

    outline: List[Dict[str, str]] = []
    for item in card.get("outline") or []:
        if not isinstance(item, dict):
            continue
        stamp = snap_time(item.get("time"))
        title = _as_str(item.get("title"), 40)
        if stamp and title:
            outline.append({"time": stamp, "title": title})
    if len(outline) > 20:
        outline = outline[:20]

    glossary: List[Dict[str, Any]] = []
    seen = set()
    for item in card.get("glossary") or []:
        if not isinstance(item, dict):
            continue
        correct = _as_str(item.get("correct"), 40)
        if not correct or correct in seen:
            continue
        variants = []
        for v in item.get("variants") or []:
            word = _as_str(v, 40)
            if word and word != correct and word in corpus and word not in variants:
                variants.append(word)
        if not variants:
            # 没有真实出现的错写，就不必进术语表（避免模型凭空“纠错”）
            continue
        seen.add(correct)
        glossary.append({"correct": correct, "variants": variants,
                         "note": _as_str(item.get("note"), 120)})
        if len(glossary) >= max_glossary:
            break

    people = []
    for p in card.get("people") or []:
        word = _as_str(p, 30)
        if word and word not in people and word in corpus:
            people.append(word)
    books = []
    for b in card.get("books") or []:
        word = _as_str(b, 40)
        if word and word not in books and word.strip("《》") in corpus:
            books.append(word)
    return {
        "title": _as_str(card.get("title"), 80) or "讲座整理稿",
        "topic": _as_str(card.get("topic"), 400),
        "speaker": _as_str(card.get("speaker"), 40),
        "outline": outline,
        "glossary": glossary,
        "people": people[:30],
        "books": books[:30],
        "notes": _as_str(card.get("notes"), 300),
    }


def _card_hash(card: Dict[str, Any]) -> str:
    payload = json.dumps(card, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def analyze_lecture(
    segments: Sequence[Segment],
    cfg: Any,
    *,
    job_id: str = "",
    source: str = "",
    usage_sink: Optional[List[Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """全讲 → 课程档案（可能调用真实 API；调用方负责缓存）。"""
    total_chars = sum(len(s.text) for s in segments)
    max_chars = int(_get(cfg, "compose.max_input_chars", 120000))
    if total_chars > max_chars:
        raise ComposeError(
            f"整讲 {total_chars} 字超过 compose.max_input_chars={max_chars}；"
            "请调大上限或先拆分课程"
        )
    hotwords: Sequence[str] = []
    try:
        hotwords = load_hotwords(cfg, job_id=job_id, source=source)
    except Exception:  # 热词缺失不影响成稿
        log.debug("读取热词失败（忽略）", exc_info=True)
    messages = build_analyze_messages(segments, cfg, job_id=job_id, source=source,
                                      hotwords=hotwords)
    started = time.time()
    res = _chat_with_deadline(
        cfg, messages,
        timeout_s=float(_get(cfg, "compose.analyze_timeout_s", 600)),
        base_url=str(_get(cfg, "compose.base_url", "") or "") or None,
        model=str(_get(cfg, "compose.model", "") or "") or None,
        reasoning_effort=str(_get(cfg, "compose.analyze_reasoning_effort",
                                  _get(cfg, "compose.reasoning_effort", "high")) or ""),
    )
    card = sanitize_card(parse_card_json(res["text"]), segments, cfg)
    usage = res["usage"]
    if usage_sink is not None:
        usage_sink.append(usage)
    card["_meta"] = {
        "version": COMPOSE_VERSION,
        "transcript_sha1": transcript_hash(segments),
        "model": res.get("model", ""),
        "usage": usage,
        "cost_cny": round(compute_cost(usage, cfg), 6),
        "elapsed_s": round(time.time() - started, 1),
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return card


def transcript_hash(segments: Sequence[Segment]) -> str:
    h = hashlib.sha1()
    for s in segments:
        h.update(f"{s.index}\t{s.start:.3f}\t{s.text}\n".encode("utf-8"))
    return h.hexdigest()


# ================================================================ 分块
@dataclass
class Chunk:
    index: int  # 1-based
    segs: List[Segment]
    overlap_count: int = 0
    text: str = ""

    @property
    def start_s(self) -> float:
        return float(self.segs[0].start)

    @property
    def end_s(self) -> float:
        return float(self.segs[-1].end)

    @property
    def valid_times(self) -> List[str]:
        return [fmt_time(s.start) for s in self.segs]

    @property
    def hash(self) -> str:
        h = hashlib.sha1()
        for s in self.segs:
            h.update(f"{s.index}\t{s.text}\n".encode("utf-8"))
        return h.hexdigest()


def plan_chunks(
    segments: Sequence[Segment],
    cfg: Any,
    *,
    chunk_chars: Optional[int] = None,
    overlap_segments: Optional[int] = None,
) -> List[Chunk]:
    """按段边界切块（不切段）；第 2 块起带上一块末尾 1–3 段作衔接重叠。"""
    size = int(chunk_chars or _get(cfg, "compose.chunk_chars", 1800))
    size = max(300, size)
    overlap_n = int(overlap_segments if overlap_segments is not None
                    else _get(cfg, "compose.overlap_segments", 2))
    overlap_n = max(0, min(overlap_n, 5))
    segs = [s for s in segments if s.text.strip()]
    if not segs:
        return []
    groups: List[List[Segment]] = []
    cur: List[Segment] = []
    cur_len = 0
    for seg in segs:
        if cur and cur_len + len(seg.text) > size:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(seg)
        cur_len += len(seg.text)
    if cur:
        groups.append(cur)

    chunks: List[Chunk] = []
    for i, group in enumerate(groups, start=1):
        overlap: List[Segment] = []
        if i > 1 and overlap_n:
            prev = groups[i - 2]
            overlap = prev[-overlap_n:]
            # 重叠段若已在本块开头（极端情况），去掉
            ids = {id(s) for s in group}
            overlap = [s for s in overlap if id(s) not in ids]
        all_segs = overlap + group
        chunks.append(Chunk(index=i, segs=all_segs, overlap_count=len(overlap),
                            text="\n".join(s.text for s in all_segs)))
    return chunks


# ================================================================ 请求构造
def build_card_prompt(card: Dict[str, Any]) -> str:
    lines = ["【课程档案】", f"标题：{card.get('title', '')}", f"主题：{card.get('topic', '')}"]
    if card.get("speaker"):
        lines.append(f"讲授人：{card['speaker']}")
    outline = card.get("outline") or []
    if outline:
        lines.append("结构：" + "；".join(f"[{o['time']}] {o['title']}" for o in outline))
    glossary = card.get("glossary") or []
    if glossary:
        parts = []
        for item in glossary:
            variants = "/".join(item.get("variants") or [])
            parts.append(f"{item['correct']}（转写作 {variants}）" if variants else item["correct"])
        lines.append("术语表：" + "；".join(parts))
    if card.get("people"):
        lines.append("人物：" + "、".join(card["people"]))
    if card.get("books"):
        lines.append("书目：" + "、".join(card["books"]))
    if card.get("notes"):
        lines.append("注意：" + str(card["notes"]))
    return "\n".join(lines)


def build_chunk_messages(
    chunk: Chunk,
    card: Dict[str, Any],
    cfg: Any,
    *,
    prev_tail: str = "",
    strict: bool = False,
) -> List[Dict[str, str]]:
    system = _load_prompt(cfg, "compose.system_prompt_file",
                          "configs/prompts/compose_system.txt", DEFAULT_SYSTEM_PROMPT)
    parts = [build_card_prompt(card), ""]
    if prev_tail:
        parts += ["【前文结尾（仅供衔接，不要重复输出）】", prev_tail, ""]
    overlap = chunk.segs[: chunk.overlap_count]
    core = chunk.segs[chunk.overlap_count:]
    if overlap:
        parts += ["【衔接上下文（用于接上前文，不要重复已整理内容）】"]
        parts += [f"[{fmt_time(s.start)}] {s.text}" for s in overlap]
        parts.append("")
    parts += ["【待整理转写】"]
    parts += [f"[{fmt_time(s.start)}] {s.text}" for s in core]
    parts.append("")
    task = ("【输出】直接输出整理后的自然段，每段以 [mm:ss] 开头，段落之间空一行。"
            "时间戳只能取自上面出现的 [mm:ss]。")
    if strict:
        task += ("注意：不要添加原文没有的内容，不要删掉例子、论证与数字；"
                 "无法辨认的乱码用〔原音不清〕标注，不要编造。")
    parts.append(task)
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


# ================================================================ 输出解析 / 校验
def parse_paragraphs(text: str) -> List[Dict[str, Any]]:
    """把模型输出解析成 ``[{time, start_s, text}]``；容忍代码块/多余表头。"""
    cleaned = strip_fences(text)
    paragraphs: List[Dict[str, Any]] = []
    cur_time = ""
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf, cur_time
        body = "".join(buf).strip()
        body = _MD_PREFIX_RE.sub("", body).strip()
        body = re.sub(r"\*\*", "", body).strip()
        if cur_time and body:
            paragraphs.append({"time": cur_time, "start_s": parse_time(cur_time), "text": body})
        buf = []

    for line in cleaned.splitlines():
        m = _TIME_RE.match(line)
        if m:
            flush()
            cur_time = m.group(1)
            buf = [line[m.end():].strip()]
        elif buf and line.strip():
            buf.append(line.strip())
        elif not buf and line.strip():
            # 时间戳之前的标题/说明一律丢弃
            continue
    flush()
    return paragraphs


def _allowed_drop_digits(card: Dict[str, Any]) -> set:
    allowed = set()
    for item in card.get("glossary") or []:
        for variant in item.get("variants") or []:
            allowed.update(_NUM_RE.findall(variant or ""))
    return allowed


def _book_count(text: str) -> int:
    return text.count("《")


def validate_chunk(
    orig: str,
    fixed: str,
    valid_times: Sequence[str],
    cfg: Any,
    *,
    allowed_drop_digits: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """校验一块成稿：长度/时间轴为硬条件；增删数字、丢书名号为软提示（进报告）。"""
    o = normalize_text(orig)
    f = normalize_text(fixed)
    hard: List[str] = []
    soft: List[str] = []
    if not f:
        return {"ok": False, "reasons": ["empty"], "warnings": [], "ratio": 0.0}
    min_ratio = float(_get(cfg, "compose.min_ratio", 0.42))
    short_chars = int(_get(cfg, "compose.short_chars", 400))
    if len(o) < short_chars:
        min_ratio = min(min_ratio, float(_get(cfg, "compose.min_ratio_short", 0.30)))
    max_ratio = float(_get(cfg, "compose.max_ratio", 1.25))
    ratio = len(f) / max(1, len(o))
    if ratio < min_ratio:
        hard.append(f"too_short:{ratio:.2f}")
    if ratio > max_ratio:
        hard.append(f"too_long:{ratio:.2f}")

    on, fn = _NUM_RE.findall(o), _NUM_RE.findall(f)
    added = [x for x in fn if x not in on]
    if added:
        # 不作为硬失败：Bekker 页码等常被模型规范化（2011A→201A1），
        # 真正要防的是丢内容/编内容，数字差异留给人看报告。
        soft.append(f"added_numbers:{added[:6]}")
    allowed = set(str(x) for x in (allowed_drop_digits or []))
    dropped = [x for x in on if x not in fn and x not in allowed]
    if dropped:
        soft.append(f"dropped_numbers:{dropped[:6]}")
    if _book_count(f) < _book_count(o):
        soft.append(f"books_lost:{_book_count(o)}->{_book_count(f)}")

    stamps = re.findall(r"^\s*\[(\d{1,3}:\d{2})\]", fixed, flags=re.M)
    if not stamps:
        hard.append("no_timestamps")
    else:
        valid = set(valid_times)
        bad = [s for s in stamps if s not in valid]
        if bad:
            hard.append(f"bad_times:{bad[:3]}")
        secs = [parse_time(s) for s in stamps]
        if any(b < a for a, b in zip(secs, secs[1:])):
            hard.append("non_monotonic")
    return {"ok": not hard, "reasons": hard, "warnings": soft, "ratio": round(ratio, 3)}


def mechanical_cleanup(text: str) -> str:
    """回退路径：只做最保守的去语气词/去重复标点，不改内容。"""
    cleaned = _FILLER_RE.sub("", text or "")
    cleaned = re.sub(r"[，、；：]{2,}", "，", cleaned)
    cleaned = re.sub(r"[。！？]{2,}", "。", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.replace(" ，", "，").replace(" 。", "。")


# ================================================================ 单块执行
def _chunk_cache_path(
    cache_root: Path, transcript_sha: str, chunk: Chunk, card_hash: str,
    prev_tail: str, strict: bool,
) -> Path:
    key = hashlib.sha1()
    key.update(f"{COMPOSE_VERSION}\t{card_hash}\t{chunk.hash}\t{strict}\t{prev_tail}".encode("utf-8"))
    return cache_root / transcript_sha / f"chunk_{chunk.index:03d}_{key.hexdigest()[:16]}.json"


def compose_chunk(
    chunk: Chunk,
    card: Dict[str, Any],
    cfg: Any,
    *,
    transcript_sha: str = "",
    prev_tail: str = "",
    cache_root: Optional[Path] = None,
    cache_enabled: bool = True,
    no_cache: bool = False,
    system_cache: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """整理一块；失败重试一次后回退，保证整篇可继续。"""
    started = time.time()
    card_hash = _card_hash(card)
    cache_path = None
    if cache_enabled and cache_root is not None and transcript_sha:
        cache_path = _chunk_cache_path(cache_root, transcript_sha, chunk, card_hash,
                                       prev_tail, False)
        if not no_cache and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("text") and cached.get("validation", {}).get("ok"):
                    cached["cached"] = True
                    # 保留首次生成时的 elapsed_s/cost_cny，供整篇统计「生成本稿的成本」
                    return cached
            except (OSError, ValueError):
                log.debug("成稿缓存损坏，忽略：%s", cache_path)

    max_retries = max(0, int(_get(cfg, "compose.max_retries", 1)))
    timeout = float(_get(cfg, "compose.timeout_s", 300))
    base_url = str(_get(cfg, "compose.base_url", "") or "") or None
    model = str(_get(cfg, "compose.model", "") or "") or None
    allowed_drop = _allowed_drop_digits(card)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0,
                   "cache_hit_tokens": 0, "reasoning_tokens": 0}
    text_out = ""
    validation: Dict[str, Any] = {"ok": False, "reasons": ["no_attempt"], "warnings": []}
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        messages = build_chunk_messages(chunk, card, cfg, prev_tail=prev_tail,
                                        strict=attempt > 0)
        try:
            res = _chat_with_deadline(
                cfg, messages,
                timeout_s=timeout, base_url=base_url, model=model,
                reasoning_effort=str(_get(cfg, "compose.reasoning_effort", "high") or ""),
            )
        except CorrectError as exc:
            validation = {"ok": False, "reasons": [f"api_error:{exc}"], "warnings": []}
            continue
        for k in usage_total:
            usage_total[k] += int(res["usage"].get(k, 0) or 0)
        fixed = strip_fences(res["text"])
        validation = validate_chunk(chunk.text, fixed, chunk.valid_times, cfg,
                                    allowed_drop_digits=allowed_drop)
        text_out = fixed
        if validation["ok"]:
            break

    if validation.get("ok") and text_out:
        paragraphs = parse_paragraphs(text_out)
        if not paragraphs:  # 理论上有时间戳就会解析出来；兜底
            validation = {"ok": False, "reasons": ["no_paragraphs"], "warnings": []}
    if not validation.get("ok"):
        stamp = fmt_time(chunk.start_s)
        fallback_text = mechanical_cleanup(chunk.text)
        text_out = f"[{stamp}] {fallback_text}"
        paragraphs = parse_paragraphs(text_out)
        validation = {**validation, "ok": False, "fallback": True}

    result = {
        "chunk": chunk.index,
        "start": fmt_time(chunk.start_s),
        "end": fmt_time(chunk.end_s),
        "n_seg": len(chunk.segs),
        "overlap": chunk.overlap_count,
        "orig_chars": len(normalize_text(chunk.text)),
        "out_chars": len(normalize_text(text_out)),
        "attempts": attempts,
        "cached": False,
        "text": text_out,
        "paragraphs": paragraphs,
        "validation": validation,
        "ok": bool(validation.get("ok")),
        "fallback": bool(validation.get("fallback")),
        "usage": usage_total,
        "cost_cny": round(compute_cost(usage_total, cfg), 6),
        "elapsed_s": round(time.time() - started, 1),
    }
    if (result["ok"] and cache_path is not None and not no_cache):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        except OSError:
            log.debug("写成稿缓存失败（忽略）：%s", cache_path)
    return result


# ================================================================ 合并 / 去重
def _dedupe_prefix(prev_text: str, cur_text: str) -> str:
    """去掉 cur 开头与 prev 结尾重复的部分（字符级 + 句级，容忍轻微改写）。"""
    import difflib

    if not prev_text or not cur_text:
        return cur_text
    prev_tail = prev_text[-500:]
    head = cur_text[:100]
    sm = difflib.SequenceMatcher(None, prev_tail, head, autojunk=False)
    cut = 0
    for _a, b, size in sm.get_matching_blocks():
        if size >= 8 and b <= 10:
            cut = max(cut, b + size)
    if cut:
        cur_text = re.sub(r"^[，。；：、\s]+", "", cur_text[cut:])
        if not cur_text:
            return ""

    prev_sents = [s for s in _SENT_RE.findall(prev_text) if s.strip()]
    cur_sents = [s for s in _SENT_RE.findall(cur_text) if s.strip()]
    while cur_sents:
        a = re.sub(r"[^\u4e00-\u9fff]", "", cur_sents[0])
        hit = False
        for cand in prev_sents[-3:]:
            b = re.sub(r"[^\u4e00-\u9fff]", "", cand)
            if not a or not b:
                continue
            if difflib.SequenceMatcher(None, a, b).ratio() >= 0.8 or (
                (a in b or b in a) and min(len(a), len(b)) >= 8
            ):
                cur_sents.pop(0)
                hit = True
                break
        if not hit:
            break
    return re.sub(r"^[，。；：、\s]+", "", "".join(cur_sents).strip())


def merge_paragraphs(chunk_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并各块段落，处理块间重复与时间倒序。"""
    merged: List[Dict[str, Any]] = []
    prev_text = ""
    for result in chunk_results:
        for para in result.get("paragraphs") or []:
            body = str(para.get("text") or "").strip()
            if not body:
                continue
            if prev_text:
                body = _dedupe_prefix(prev_text, body)
                if not body:
                    continue
            stamp = str(para.get("time") or "")
            if not stamp:
                continue
            start_s = int(para.get("start_s") or parse_time(stamp))
            if merged and start_s < merged[-1]["start_s"]:
                # 重叠段整理后仍早于上一段：保留内容，时间钉到上一段
                start_s = merged[-1]["start_s"]
                stamp = fmt_time(start_s)
            merged.append({
                "time": stamp,
                "start_s": start_s,
                "text": body,
                "chunk": int(result.get("chunk") or 0),
            })
            prev_text = body
    return merged


def annotate_orig_times(paragraphs: Sequence[Dict[str, Any]],
                        chunks: Sequence[Chunk]) -> List[Dict[str, Any]]:
    """给每段补原始音轨时间（``orig_time`` / ``orig_start_s``）。

    成稿的 ``[mm:ss]`` 取自 transcript 的 clean 轴（处理后音频）；原轴只用于对照原视频，
    由「clean 时间 + 所属块」回查源段得到。
    """
    by_time: Dict[str, List[Tuple[int, Segment]]] = {}
    all_segs: List[Segment] = []
    for chunk in chunks:
        for seg in chunk.segs:
            by_time.setdefault(fmt_time(seg.start), []).append((chunk.index, seg))
            all_segs.append(seg)
    out: List[Dict[str, Any]] = []
    for para in paragraphs:
        item = dict(para)
        stamp = str(item.get("time") or "")
        candidates = by_time.get(stamp) or []
        same_chunk = [seg for idx, seg in candidates if idx == item.get("chunk")]
        pool = same_chunk or [seg for _idx, seg in candidates]
        if pool:
            seg = min(pool, key=lambda s: abs(s.start - float(item["start_s"])))
        elif all_segs:
            seg = min(all_segs, key=lambda s: abs(s.start - float(item["start_s"])))
        else:
            out.append(item)
            continue
        item["orig_start_s"] = round(seg.orig_start_s, 3)
        item["orig_time"] = fmt_time(seg.orig_start_s)
        out.append(item)
    return out


# ================================================================ 落盘
def _paragraph_time_bounds(paragraphs: Sequence[Dict[str, Any]], segments: Sequence[Segment],
                           end_s: float) -> List[Dict[str, Any]]:
    out = []
    for i, para in enumerate(paragraphs):
        start = float(para["start_s"])
        if i + 1 < len(paragraphs):
            end = float(paragraphs[i + 1]["start_s"])
        else:
            end = max(start, end_s)
        out.append({**para, "start_s": start, "end_s": end})
    return out


def _stats(
    segments: Sequence[Segment],
    paragraphs: Sequence[Dict[str, Any]],
    chunk_results: Sequence[Dict[str, Any]],
    card: Dict[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    source_text = "".join(s.text for s in segments)
    manuscript_text = "".join(str(p["text"]) for p in paragraphs)
    normalized = normalize_text(manuscript_text)
    glossary = card.get("glossary") or []
    residual = []
    for item in glossary:
        for variant in item.get("variants") or []:
            if variant and variant in manuscript_text and variant not in item["correct"]:
                residual.append({"correct": item["correct"], "variant": variant})
    missing_terms = [item["correct"] for item in glossary
                    if item["correct"] and item["correct"] not in manuscript_text]
    fillers = {f: {"source": source_text.count(f), "manuscript": manuscript_text.count(f)}
               for f in ("呃", "啊", "嗯", "这个这个", "就是就是")}
    fallback = [r["chunk"] for r in chunk_results if r.get("fallback")]
    stamps = [int(p["start_s"]) for p in paragraphs]
    return {
        "source_chars": len(source_text),
        "source_segments": len(segments),
        "manuscript_chars": len(normalized),
        "char_ratio": round(len(normalized) / max(1, len(source_text)), 4),
        "n_paragraphs": len(paragraphs),
        "n_chunks": len(chunk_results),
        "fallback_chunks": fallback,
        "n_fallback": len(fallback),
        "timestamps": {"n": len(stamps), "first": fmt_time(stamps[0]) if stamps else "",
                       "last": fmt_time(stamps[-1]) if stamps else "",
                       "monotonic": all(b >= a for a, b in zip(stamps, stamps[1:]))},
        "glossary_terms": len(glossary),
        "glossary_missing": missing_terms,
        "residual_variants": residual,
        "fillers": fillers,
        "chunk_ok": sum(1 for r in chunk_results if r.get("ok")),
        "chunk_cached": sum(1 for r in chunk_results if r.get("cached")),
    }


def _write_manuscript_md(
    path: Path,
    job_id: str,
    title: str,
    segments: Sequence[Segment],
    paragraphs: Sequence[Dict[str, Any]],
    stats: Dict[str, Any],
    cost_cny: float,
    elapsed_s: float,
    model: str,
    chunk_chars: int,
) -> None:
    duration = fmt_time(segments[-1].end if segments else 0)
    header = f"""# {title}

> 来源：`out/{job_id}/transcript.json`（clean 音轨 {duration}，{stats['source_segments']:,} 段，{stats['source_chars']:,} 字）
> 方法：先通读全篇生成课程档案（结构 + 术语表），再按约 {chunk_chars:,} 字分块整理；`[mm:ss]` 为 **clean 音轨（处理后的 clean.m4a）** 时间，原始音轨时间见 `manuscript.json` 的 `orig_time`。
> 规模：{stats['n_paragraphs']} 段，约 {stats['manuscript_chars']:,} 字（原文的 {stats['char_ratio']:.0%}）。
> 说明：已清理语气词与明显同音误写；成稿未经逐字校对，引文、页码与专名请对照原音复核。
> 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} · {model or 'deepseek'} · 成本约 ¥{cost_cny:.2f} · 耗时 {elapsed_s / 60:.1f} min

"""
    body = "\n\n".join(f"[{p['time']}] {p['text']}" for p in paragraphs)
    path.write_text(header + body + "\n", encoding="utf-8")


# ================================================================ 主流程
def compose_job(
    job_id: str,
    cfg: Any,
    *,
    out_dir: Optional[str | Path] = None,
    force: bool = False,
    no_cache: bool = False,
    dry_run: bool = False,
    chunk_chars: Optional[int] = None,
    concurrency: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """对单个 job 执行成稿；返回 ``{ok, skipped, stats, ...}``，失败抛 ComposeError。"""
    started = time.time()
    job_out = _out_root(cfg, out_dir) / job_id
    transcript_path = job_out / "transcript.json"
    segments = load_segments(transcript_path)
    if not segments:
        raise ComposeError(f"transcript.json 没有可用文本：{transcript_path}")
    tsha = transcript_hash(segments)
    size = int(chunk_chars or _get(cfg, "compose.chunk_chars", 1800))
    chunks = plan_chunks(segments, cfg, chunk_chars=size)
    if dry_run:
        return {
            "ok": True, "job_id": job_id, "dry_run": True, "skipped": False,
            "n_segments": len(segments), "source_chars": sum(len(s.text) for s in segments),
            "n_chunks": len(chunks), "chunk_chars": size,
            "chunks": [{"index": c.index, "start": fmt_time(c.start_s),
                        "end": fmt_time(c.end_s), "n_seg": len(c.segs)} for c in chunks],
        }

    manuscript_md = job_out / "manuscript.md"
    manuscript_json = job_out / "manuscript.json"
    card_path = job_out / "compose_card.json"
    report_path = job_out / "compose_report.json"
    if manuscript_json.is_file() and manuscript_md.is_file() and not force:
        try:
            old = json.loads(manuscript_json.read_text(encoding="utf-8"))
            if old.get("transcript_sha1") == tsha and old.get("version") == COMPOSE_VERSION:
                return {"ok": True, "job_id": job_id, "skipped": True,
                        "manuscript": str(manuscript_md), "stats": old.get("stats", {}),
                        "cost_cny": old.get("cost_cny", 0.0),
                        "elapsed_s": round(time.time() - started, 1)}
        except (OSError, ValueError):
            pass

    api_key = resolve_api_key(cfg)
    if not api_key:
        raise ComposeError("缺少 API key（correct.api_key 或 DEEPSEEK_API_KEY）")

    card: Optional[Dict[str, Any]] = None
    if card_path.is_file():
        try:
            cached_card = json.loads(card_path.read_text(encoding="utf-8"))
            meta = cached_card.get("_meta") or {}
            if meta.get("transcript_sha1") == tsha and meta.get("version") == COMPOSE_VERSION:
                card = cached_card
        except (OSError, ValueError):
            card = None
    usage_all: List[Dict[str, int]] = []
    if card is None:
        card = analyze_lecture(segments, cfg, job_id=job_id, source=str(transcript_path))
        job_out.mkdir(parents=True, exist_ok=True)
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    card_meta = card.get("_meta") or {}
    cost_analysis = float(card_meta.get("cost_cny") or 0.0)

    model = str((card.get("_meta") or {}).get("model") or _get(cfg, "correct.model", "") or "")
    cache_enabled = bool(_get(cfg, "compose.cache_enabled", True))
    cache_root = _work_root(cfg) / str(_get(cfg, "compose.cache_subdir", "compose_cache"))
    workers = int(concurrency or _get(cfg, "compose.concurrency", 1))
    workers = max(1, min(workers, 16))
    results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
    total = len(chunks)

    def run_one(idx: int) -> Tuple[int, Dict[str, Any]]:
        chunk = chunks[idx]
        prev_tail = ""
        if idx > 0:
            prev_text = chunks[idx - 1].text
            prev_tail = prev_text[-200:]
        result = compose_chunk(
            chunk, card, cfg,
            transcript_sha=tsha, prev_tail=prev_tail,
            cache_root=cache_root, cache_enabled=cache_enabled, no_cache=no_cache,
        )
        return idx, result

    try:
        if workers == 1 or total <= 1:
            for idx in range(total):
                i, result = run_one(idx)
                results[i] = result
                if on_progress:
                    on_progress(i + 1, total, result)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_one, idx): idx for idx in range(total)}
                done = 0
                for fut in concurrent.futures.as_completed(futures):
                    i, result = fut.result()
                    results[i] = result
                    done += 1
                    if on_progress:
                        on_progress(done, total, result)
    except KeyboardInterrupt:  # pragma: no cover - 中断保留已有缓存
        raise

    chunk_results = [r for r in results if r is not None]
    if len(chunk_results) != total:
        raise ComposeError("成稿分块结果不完整（内部错误）")
    paragraphs = merge_paragraphs(chunk_results)
    if not paragraphs:
        raise ComposeError("所有分块都未产出可用段落")
    paragraphs = _paragraph_time_bounds(paragraphs, segments, segments[-1].end)
    paragraphs = annotate_orig_times(paragraphs, chunks)
    stats = _stats(segments, paragraphs, chunk_results, card, cfg)
    cost_chunks = sum(float(r.get("cost_cny") or 0.0) for r in chunk_results)
    cost_total = round(cost_analysis + cost_chunks, 6)
    generation_elapsed = round(
        float(card_meta.get("elapsed_s") or 0.0)
        + sum(float(r.get("elapsed_s") or 0.0) for r in chunk_results), 1
    )
    elapsed = round(time.time() - started, 1)
    title = str(card.get("title") or job_id)
    stats["cost_cny"] = cost_total
    stats["generation_elapsed_s"] = generation_elapsed
    stats["elapsed_s"] = elapsed  # 本次运行墙钟（可能有缓存命中）
    stats["concurrency"] = workers
    stats["model"] = model

    job_out.mkdir(parents=True, exist_ok=True)
    _write_manuscript_md(manuscript_md, job_id, title, segments, paragraphs, stats,
                         cost_total, generation_elapsed, model, size)
    payload = {
        "version": COMPOSE_VERSION,
        "job_id": job_id,
        "title": title,
        "transcript_sha1": tsha,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(transcript_path),
        "paragraphs": paragraphs,
        "stats": stats,
        "cost_cny": cost_total,
        "generation_elapsed_s": generation_elapsed,
        "elapsed_s": elapsed,
        "card": card,
    }
    manuscript_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "version": COMPOSE_VERSION,
        "job_id": job_id,
        "ok": True,
        "stats": stats,
        "cost_cny": cost_total,
        "generation_elapsed_s": generation_elapsed,
        "elapsed_s": elapsed,
        "card_meta": card.get("_meta"),
        "chunks": [
            {k: v for k, v in r.items() if k not in ("text", "paragraphs", "usage")}
            | {"usage": r.get("usage")}
            for r in chunk_results
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True, "job_id": job_id, "skipped": False,
        "manuscript": str(manuscript_md),
        "manuscript_json": str(manuscript_json),
        "report": str(report_path),
        "stats": stats, "cost_cny": cost_total, "elapsed_s": elapsed,
        "generation_elapsed_s": generation_elapsed,
    }
