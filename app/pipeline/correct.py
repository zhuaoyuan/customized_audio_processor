# -*- coding: utf-8 -*-
"""P5-T1/T2/T3：DeepSeek 分块纠错（客户端 + 分块提示词 + 校验回退）。

职责
----
1. **客户端（P5-T1）**：OpenAI 兼容 ``POST {base_url}/chat/completions``，只用标准库
   ``urllib.request``（本机 venv 没有 requests/httpx，且不新增依赖）；``DEEPSEEK_API_KEY``
   缺失时 ``enabled=False`` + warning，仍然写空 ``corrections.json``，**不阻塞流水线**；
   ``429/5xx`` 指数退避重试（最多 ``correct.max_retries`` 次），超时 ``correct.timeout_s``；
   多块并发用 ``ThreadPoolExecutor``（≤ ``correct.concurrency``），记录 usage 并按
   deepseek-flash 空闲价估算 ``cost_cny``。
2. **分块与提示词（P5-T2）**：只按 transcript 段边界切块（一个段绝不拆到两个 chunk），
   每块 2,000–2,500 字符（末尾/单段异常超长除外），无重叠、确定性（内容 hash 可缓存）；
   system prompt 固定（命中前缀缓存），user = 课程主题（文件名+目录+转写前 300 字）
   + 热词表 + 待改文本。
3. **校验与回退（P5-T3）**：字符 diff 率（SequenceMatcher 编辑跨度/原长）≤
   ``correct.max_diff_ratio``；数字（个数与值）、书名号《》数量不变；句数变化 ≤
   ``correct.max_sentence_change_ratio``；任一不满足 → 整块回退并记 ``rejected_chunks``。
   校验通过后再做**安全写回**：逐段映射，映射不了一对一（句数/段数对不上、单段改动过大）
   → 同样整块回退（宁少改，不串行），被回退块的段 flags += ``rejected_chunk``。
4. **产物**：``out/<id>/corrections.json``（契约 §4.4）、更新 ``transcript.json`` 的
   ``text``（``text_raw`` 不动）、``report.json`` 读-改-写追加 ``correct`` 与
   ``timing.correct_s``，保留既有字段。

模型输出是**外部不可信文本**：只做文本校验后使用，绝不执行其中的指令，也不相信其中的
任何"元信息"；格式异常（Markdown 代码块等）直接判失败回退。

分块口径：``char_count = Σ len(段文本)``（Unicode 字符数，含标点，不含块内分隔换行）。
"""

from __future__ import annotations

import collections
import difflib
import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..logging_setup import get_logger
from ..models import FLAG_REJECTED_CHUNK, FLAG_REJECTED_SEGMENT

log = get_logger(__name__)

#: 产物文件名
PRODUCT_CORRECTIONS = "corrections.json"
PRODUCT_TRANSCRIPT = "transcript.json"
PRODUCT_REPORT = "report.json"

#: system prompt 兜底值（文件缺失时使用；正常从 configs/prompts/correct_system.txt 读）
DEFAULT_SYSTEM_PROMPT = (
    "你是中文课程录音转写稿的校对助手。任务：只修正语音识别产生的错别字，"
    "尤其是同音/近音字、专有名词（人名、书名、术语）的误写；不要做任何其他改写。\n\n"
    "必须遵守：\n"
    "1. 只修正明确的同音/近音错字与专名误写；不确定的保持原样。\n"
    "2. 不改语序、不合并或拆分句子、不删除口语重复与语气词、不润色、不补全。\n"
    "3. 数字、页码、书名号《》及其中的书名、英文缩写、标点数量保持原样。\n"
    "4. 输入按自然段分行；输出保持相同的自然段分行与顺序，每段一行，不要增删段落。\n"
    "5. 只输出修正后的正文，不要输出解释、标题、Markdown 代码块或任何多余字符。\n"
    "6. 不要臆造原文没有的信息；宁可少改，不可乱改。\n"
)

#: 句末标点（连续出现按 1 句算，如「……」「？！」）
_SENT_TERMINATORS = set("。！？；!?;…")
#: 句末标点后可能紧跟的收尾符号
_SENT_CLOSERS = set("”’」』）)》】〕〉\"'")
#: 数字串
_NUM_RE = re.compile(r"\d+")
#: 全角数字 → 半角（数字比较口径，避免模型把「２０２０」写成「2020」就被误判）
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
#: 数字千位逗号（1,234 → 1234）
_NUM_SEP_RE = re.compile(r"(?<=\d),(?=\d)")
#: 纯标点/格式符号（用于给 item 生成 reason 启发式）
_PUNCT_ONLY_RE = re.compile(r"^[\s，。！？；：、,.!?;:…—～~（）()「」『』《》“”\"'\-]+$")
_WHITESPACE_RE = re.compile(r"\s+")


class CorrectError(RuntimeError):
    """纠错阶段的错误（配置/HTTP/响应格式/产物缺失等）。"""


# ================================================================ 配置小工具
def _get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """点号取配置；``None`` 视为缺省（配置文件里空值走默认）。"""
    if cfg is None:
        return default
    try:
        value = cfg.get(dotted, default)
    except AttributeError:  # 兼容 SimpleNamespace 之外的假 cfg
        return default
    return default if value is None else value


def _root(cfg: Any) -> Path:
    root = getattr(cfg, "root", None)
    return Path(root) if root else Path(__file__).resolve().parents[2]


def _resolve_path(cfg: Any, value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = _root(cfg) / p
    return p


def _out_root(cfg: Any, out_dir: Optional[str | Path]) -> Path:
    if out_dir:
        return Path(out_dir)
    try:
        return Path(cfg.path("out_dir"))
    except Exception:  # pragma: no cover - 假 cfg 兜底
        return Path("out")


def resolve_api_key(cfg: Any, env: Optional[Dict[str, str]] = None) -> str:
    """API key：``correct.api_key`` 优先，其次配置指定的环境变量（默认 DEEPSEEK_API_KEY）。"""
    key = str(_get(cfg, "correct.api_key", "") or "").strip()
    if key:
        return key
    env_name = str(_get(cfg, "correct.api_key_env", "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY")
    environ = os.environ if env is None else env
    return str(environ.get(env_name, "") or "").strip()


# ================================================================ 文本工具
def _flat(text: Optional[str]) -> str:
    """去掉所有空白（含换行/全角空格），用于 diff/数字/书名号/句数口径。"""
    return _WHITESPACE_RE.sub("", text or "")


def split_sentences(text: Optional[str]) -> List[str]:
    """按句末标点切句（。！？；!?;…），不丢标点、不切分引号/括号收尾。

    文本没有句末标点时整段算 1 句；连续句末标点（如「……」「？！」）算 1 句。
    """
    t = (text or "").strip()
    if not t:
        return []
    out: List[str] = []
    buf: List[str] = []
    i, n = 0, len(t)
    while i < n:
        buf.append(t[i])
        if t[i] in _SENT_TERMINATORS:
            i += 1
            while i < n and (t[i] in _SENT_TERMINATORS or t[i] in _SENT_CLOSERS):
                buf.append(t[i])
                i += 1
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
            continue
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def diff_ratio(orig: str, fixed: str) -> float:
    """字符差异率 = Σ 非 equal opcode 的编辑跨度 / 原文长度（空白不计）。"""
    o, f = _flat(orig), _flat(fixed)
    if not o:
        return 0.0 if not f else 1.0
    sm = difflib.SequenceMatcher(None, o, f, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed / len(o)


def _normalize_digits(text: str) -> str:
    """数字比较口径（P9）：全角数字转半角、去掉数字间千位逗号。

    只影响数字校验，不改写回文本本身；中文数字（「二」「十四」）暂不折算，仍按原样严格比较
    （避免把「一样」「第一」这类普通用字误当数字）。
    """
    return _NUM_SEP_RE.sub("", (text or "").translate(_FULLWIDTH_DIGITS))


def _num_runs(text: str) -> collections.Counter:
    """数字串计数（比较口径见 :func:`_normalize_digits`）。"""
    return collections.Counter(_NUM_RE.findall(_normalize_digits(_flat(text))))


def _book_counts(text: str) -> Tuple[int, int]:
    f = _flat(text)
    return f.count("《"), f.count("》")


# ================================================================ 配置读取 / 提示词 / 热词
def load_system_prompt(cfg: Any) -> str:
    """读固定 system prompt（文件缺失时退回内置文本并告警；内容必须稳定以命中前缀缓存）。"""
    value = str(_get(cfg, "correct.prompt_file", "") or "").strip()
    if value:
        path = _resolve_path(cfg, value)
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
            except OSError as exc:  # pragma: no cover - 读失败退回内置
                log.warning("读取纠错 prompt 失败（%s）：%s", path, exc)
        else:
            log.warning("纠错 prompt 文件不存在，使用内置提示词：%s", path)
    return DEFAULT_SYSTEM_PROMPT


def _hotwords_path(cfg: Any, job_id: str, source: str) -> Optional[Path]:
    """热词文件查找：``<job_id>.txt`` → 课程名（文件名/目录名出现在 txt 名里）→ ``default.txt``。"""
    raw = str(_get(cfg, "correct.hotwords_dir", "") or "").strip()
    if not raw:
        return None
    directory = _resolve_path(cfg, raw)
    if not directory.is_dir():
        return None

    def non_empty(p: Path) -> Optional[Path]:
        try:
            if p.is_file() and p.read_text(encoding="utf-8").strip():
                return p
        except OSError:  # pragma: no cover
            pass
        return None

    if job_id:
        hit = non_empty(directory / f"{job_id}.txt")
        if hit:
            return hit

    source_l = (source or "").replace("\\", "/").lower()
    job_l = (job_id or "").lower()
    # 课程名匹配：取 txt 文件名主干，看它是否出现在源文件名 / job id 中（取最长匹配）
    candidates: List[Tuple[int, Path]] = []
    for path in directory.glob("*.txt"):
        stem = path.stem.strip()
        if not stem or stem.lower() == "default":
            continue
        stem_l = stem.lower()
        if (stem_l and stem_l in source_l) or (stem_l and stem_l in job_l):
            candidates.append((len(stem_l), path))
    for _score, path in sorted(candidates, key=lambda item: (-item[0], item[1].name)):
        hit = non_empty(path)
        if hit:
            return hit

    return non_empty(directory / "default.txt")


def load_hotwords(cfg: Any, job_id: str = "", source: str = "") -> List[str]:
    """读热词表（缺失/空文件不报错，返回 []）。

    一行一个词；允许用「、/，/,/；/;」或空白分隔多个词；``#`` 之后为注释。
    """
    path = _hotwords_path(cfg, job_id, source)
    if path is None:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover
        log.warning("读取热词表失败：%s：%s", path, exc)
        return []
    words: List[str] = []
    seen = set()
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for word in re.split(r"[、,，;；\s]+", line):
            word = word.strip()
            if word and word not in seen:
                seen.add(word)
                words.append(word)
    return words


def topic_text(segments: Sequence[Any], limit: int = 300) -> str:
    """转写前 ``limit`` 字（按段拼接，用于 user prompt 的课程主题）。"""
    buf: List[str] = []
    total = 0
    for seg in segments:
        text = _seg_text(seg)
        if not text:
            continue
        buf.append(text)
        total += len(text)
        if total >= limit:
            break
    return "".join(buf)[:limit]


def build_messages(
    chunk: "Chunk",
    cfg: Any,
    *,
    job_id: str = "",
    source: str = "",
    hotwords: Optional[Sequence[str]] = None,
    topic: str = "",
) -> List[Dict[str, str]]:
    """构造 messages：system 固定；user = 课程主题 + 热词表 + 待改文本。"""
    system = load_system_prompt(cfg)
    lines: List[str] = ["【课程主题】"]
    if source:
        src = Path(str(source).replace("\\", "/"))
        lines.append(f"文件：{src.name}")
        if src.parent.name:
            lines.append(f"目录：{src.parent.name}")
    if job_id:
        lines.append(f"任务：{job_id}")
    if topic:
        lines.append(f"转写开头（前 300 字）：{topic}")
    lines.append("")
    lines.append("【热词表】")
    if hotwords:
        lines.append("、".join(str(w) for w in hotwords))
    else:
        lines.append("（无）")
    lines.append("")
    lines.append("【待改文本】")
    lines.append(chunk.text)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


# ================================================================ 分块
@dataclass
class Chunk:
    """一个待纠错块：**连续若干完整 transcript 段** 的文本（段内不切）。"""

    index: int  # 1-based，对应 corrections.json 的 chunk 字段
    text: str
    segment_indices: List[int] = field(default_factory=list)
    segment_texts: List[str] = field(default_factory=list)
    sentence_counts: List[int] = field(default_factory=list)
    char_count: int = 0
    content_hash: str = ""

    @property
    def n_segments(self) -> int:
        return len(self.segment_indices)


def _seg_text(seg: Any) -> str:
    """取段文本（兼容 dict 与 TranscriptSegment；优先 ``text``，空则 ``text_raw``）。"""
    if isinstance(seg, dict):
        text = seg.get("text") or ""
        if not text:
            text = seg.get("text_raw") or ""
        return str(text)
    text = getattr(seg, "text", "") or getattr(seg, "text_raw", "") or ""
    return str(text)


def plan_chunks(segments: Sequence[Any], cfg: Any) -> List[Chunk]:
    """按 **段边界** 切块：每块 = 连续若干整段，长度 2,000–2,500 字符，无重叠、确定性。

    - 段内不切（段 = VAD 切出的完整话语，段边界才是安全句边界）；块内多句一起送模型；
    - 单段异常长（> chunk_max）时允许该块超上限，但**绝不切段**；
    - 末尾碎片块会从上一块尾部搬段回填到 ≥ chunk_min（两块仍 ≤ chunk_max 时）；
      实在无法满足（总文本太短/单段超长）时按原样保留，不为了凑长度拆段。
    """
    min_c = int(_get(cfg, "correct.chunk_min_chars", 2000))
    max_c = int(_get(cfg, "correct.chunk_max_chars", 2500))
    if min_c <= 0:
        min_c = 1
    if max_c < min_c:
        max_c = min_c

    entries: List[Tuple[int, str, int]] = []  # (段索引, 段文本, 句数)
    for i, seg in enumerate(segments):
        text = _seg_text(seg).strip()
        if not text:
            continue
        entries.append((i, text, len(split_sentences(text))))
    if not entries:
        return []

    lengths = {id(e): len(e[1]) for e in entries}
    groups: List[List[Tuple[int, str, int]]] = []
    cur: List[Tuple[int, str, int]] = []
    cur_len = 0
    for entry in entries:
        seg_len = lengths[id(entry)]
        if cur and cur_len + seg_len > max_c:
            groups.append(cur)
            cur = []
            cur_len = 0
        cur.append(entry)
        cur_len += seg_len
    if cur:
        groups.append(cur)

    def group_len(group: Sequence[Tuple[int, str, int]]) -> int:
        return sum(lengths[id(e)] for e in group)

    # 末尾碎片块回填：从上一块尾部搬整段过来（两块都尽量落在 [min_c, max_c]）
    if len(groups) >= 2 and group_len(groups[-1]) < min_c:
        last, prev = groups[-1], groups[-2]
        while group_len(last) < min_c and len(prev) > 1:
            candidate = prev[-1]
            cand_len = lengths[id(candidate)]
            if group_len(last) + cand_len > max_c:
                break
            if group_len(prev) - cand_len < min_c:
                break
            prev.pop()
            last.insert(0, candidate)
        if group_len(last) < min_c:
            merged_len = group_len(prev) + group_len(last)
            if merged_len <= max_c and len(groups) >= 2:
                groups[-2] = prev + last
                groups.pop()
            # 否则接受尾部小于 chunk_min（宁可短，也不切段/不硬凑）

    chunks: List[Chunk] = []
    for idx, group in enumerate(groups, start=1):
        text = "\n".join(e[1] for e in group)
        chunks.append(
            Chunk(
                index=idx,
                text=text,
                segment_indices=[e[0] for e in group],
                segment_texts=[e[1] for e in group],
                sentence_counts=[e[2] for e in group],
                char_count=sum(lengths[id(e)] for e in group),
                content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest(),
            )
        )
    return chunks


# ================================================================ 校验 / diff items
def validate_correction(orig: str, fixed: str, cfg: Any) -> Dict[str, Any]:
    """校验模型输出：diff 率 / 数字 / 书名号 / 句数，返回校验明细（ok=False 时带 reason）。"""
    max_diff = float(_get(cfg, "correct.max_diff_ratio", 0.12))
    max_sent = float(_get(cfg, "correct.max_sentence_change_ratio", 0.05))
    o, f = _flat(orig), _flat(fixed)
    n_orig = len(split_sentences(o))
    n_fixed = len(split_sentences(f))
    ratio = diff_ratio(o, f)
    detail: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "diff_ratio": round(ratio, 6),
        "chars_orig": len(o),
        "chars_fixed": len(f),
        "n_sentences_orig": n_orig,
        "n_sentences_fixed": n_fixed,
        "sentence_change_ratio": (abs(n_fixed - n_orig) / n_orig) if n_orig else 0.0,
        "numbers_orig": [k for k, v in sorted(_num_runs(o).items()) for _ in range(v)],
        "book_marks_orig": list(_book_counts(o)),
    }
    if not f:
        detail["reason"] = "empty_output"
        return detail
    if "```" in fixed:
        # 模型输出夹带 Markdown 代码块 = 格式违规（不可信输出，直接回退）
        detail["reason"] = "markdown_wrapper"
        return detail
    if ratio > max_diff:
        detail["reason"] = "diff_too_large"
        return detail
    if _num_runs(o) != _num_runs(f):
        detail["reason"] = "number_changed"
        return detail
    if _book_counts(o) != _book_counts(f):
        detail["reason"] = "book_title_changed"
        return detail
    if n_orig and abs(n_fixed - n_orig) / n_orig > max_sent:
        detail["reason"] = "sentence_count_changed"
        return detail
    detail["ok"] = True
    return detail


def validate_block(orig: str, fixed: str, cfg: Any) -> Dict[str, Any]:
    """**块级**安全门（P9）：空输出 / Markdown 包裹 / 整块 diff / 整块句数。

    与 :func:`validate_correction` 的区别：数字与书名号**不再在块级一票否决**，而是下放到
    :func:`_seg_guard` 逐段检查——真实模型常在一段里顺手改掉数字/书名号，块级回退会把整块
    200 段的有用修正全部丢掉。块级只保留「整体可信度」判据（diff 与句数），防止模型大改。
    """
    max_diff = float(_get(cfg, "correct.max_diff_ratio", 0.12))
    max_sent = float(_get(cfg, "correct.max_sentence_change_ratio", 0.05))
    o, f = _flat(orig), _flat(fixed)
    n_orig = len(split_sentences(o))
    n_fixed = len(split_sentences(f))
    ratio = diff_ratio(o, f)
    detail: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "diff_ratio": round(ratio, 6),
        "n_sentences_orig": n_orig,
        "n_sentences_fixed": n_fixed,
        "sentence_change_ratio": (abs(n_fixed - n_orig) / n_orig) if n_orig else 0.0,
    }
    if not f:
        detail["reason"] = "empty_output"
        return detail
    if "```" in fixed:
        detail["reason"] = "markdown_wrapper"
        return detail
    if ratio > max_diff:
        detail["reason"] = "diff_too_large"
        return detail
    if n_orig and abs(n_fixed - n_orig) / n_orig > max_sent:
        detail["reason"] = "sentence_count_changed"
        return detail
    detail["ok"] = True
    return detail


@dataclass
class Edit:
    """原文→修正文本的一处最小差异（字符级，带双侧位置，供写回覆盖检查）。"""

    orig: str
    fixed: str
    i1: int
    i2: int
    j1: int
    j2: int
    tag: str = "replace"


def diff_edits(orig: str, fixed: str) -> List[Edit]:
    """字符级差异（空白不计）：每个非 equal opcode 一条 Edit。"""
    o, f = _flat(orig), _flat(fixed)
    sm = difflib.SequenceMatcher(None, o, f, autojunk=False)
    edits: List[Edit] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        edits.append(Edit(orig=o[i1:i2], fixed=f[j1:j2], i1=i1, i2=i2, j1=j1, j2=j2, tag=tag))
    return edits


def _edit_reason(edit: Edit, hotwords: Sequence[str]) -> str:
    o, f = edit.orig, edit.fixed
    if not f and o:
        return "删除多余字（候选）"
    if not o and f:
        return "补字（候选）"
    if _PUNCT_ONLY_RE.match((o or "") + (f or "")):
        return "标点修正（候选）"
    if _NUM_RE.search(o or "") or _NUM_RE.search(f or ""):
        return "数字/页码修正（候选）"
    for word in hotwords or []:
        word = str(word)
        if word and (word in f or word in o or (len(f) >= 2 and f in word)):
            return f"专名/术语（热词：{word}）"
    return "同音/近音错字或专名（候选）"


def extract_items(
    chunk: "Chunk", mapped_texts: Sequence[str], hotwords: Sequence[str]
) -> List[Dict[str, Any]]:
    """由逐段「原文→修正文本」生成 corrections.json 的 items（chunk 从 1 起）。"""
    items: List[Dict[str, Any]] = []
    for orig, fixed in zip(chunk.segment_texts, mapped_texts):
        for edit in diff_edits(orig, fixed):
            items.append(
                {
                    "orig": edit.orig,
                    "fixed": edit.fixed,
                    "reason": _edit_reason(edit, hotwords),
                    "chunk": chunk.index,
                }
            )
    return items


# ================================================================ usage / cost
def _normalize_usage(usage: Any) -> Dict[str, int]:
    """归一 usage：prompt/completion/cache_hit/reasoning（兼容 DeepSeek 多种字段名）。"""
    out = {"prompt_tokens": 0, "completion_tokens": 0,
           "cache_hit_tokens": 0, "reasoning_tokens": 0}
    if not isinstance(usage, dict):
        return out

    def _int(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    out["prompt_tokens"] = _int(usage.get("prompt_tokens"))
    out["completion_tokens"] = _int(usage.get("completion_tokens"))
    hit = 0
    for key in ("cache_hit_tokens", "prompt_cache_hit_tokens"):
        if usage.get(key) is not None:
            hit = _int(usage.get(key))
            break
    else:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            hit = _int(details.get("cached_tokens"))
    out["cache_hit_tokens"] = hit
    # 幂等：_correct_chunk 返回的 usage 已经归一过一次，这里要保留 reasoning_tokens，
    # 否则 correct_job 二次归一后会把 reasoning_tokens 抹成 0。
    reasoning = usage.get("reasoning_tokens")
    if reasoning is not None:
        out["reasoning_tokens"] = _int(reasoning)
    else:
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
            out["reasoning_tokens"] = _int(details.get("reasoning_tokens"))
    return out


def reasoning_payload(effort: Optional[str]) -> Dict[str, Any]:
    """把 ``reasoning_effort`` 配置映射成 API 请求字段（DeepSeek V4.1）。

    - 空/``default``：不传，走服务端默认（deepseek-flash 默认 thinking=on、effort=high）；
    - ``none``：显式关闭思考模式；
    - ``low`` / ``high`` / ``max``：开启思考模式并指定强度。
    """
    value = str(effort or "").strip().lower()
    if not value or value in ("default", "auto"):
        return {}
    if value == "none":
        return {"thinking": {"type": "disabled"}}
    return {"thinking": {"type": "enabled"}, "reasoning_effort": value}


def compute_cost(usage: Dict[str, int], cfg: Any) -> float:
    """费用估算（CNY）：缺省 deepseek-flash 空闲价；可用 correct.price_* 覆盖。"""
    price_in = float(_get(cfg, "correct.price_input_cny_per_m", 1.0))
    price_out = float(_get(cfg, "correct.price_output_cny_per_m", 4.0))
    price_hit = float(_get(cfg, "correct.price_cache_hit_cny_per_m", 0.02))
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    hit = min(int(usage.get("cache_hit_tokens", 0) or 0), prompt)
    miss = max(0, prompt - hit)
    return (miss * price_in + hit * price_hit + completion * price_out) / 1_000_000.0


# ================================================================ HTTP 客户端
def _build_opener(url: str) -> urllib.request.OpenerDirector:
    """localhost 直连（避免测试被系统代理劫持）；公网走默认 opener。"""
    host = urllib.parse.urlparse(url).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _chat_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _post_once(
    url: str,
    payload: Dict[str, Any],
    api_key: str,
    timeout_s: float,
) -> Dict[str, Any]:
    """发一次 POST，返回解析后的 JSON；HTTP/网络错误原样抛给调用方重试/包装。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "audio_processer/0.1 (correct)",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = _build_opener(url)
    with opener.open(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", "replace")
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise CorrectError(f"模型响应不是合法 JSON：{body[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise CorrectError("模型响应顶层不是 JSON 对象")
    return parsed


def _extract_content(payload: Dict[str, Any]) -> Tuple[str, Dict[str, int], str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise CorrectError("模型响应缺少 choices[0]")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise CorrectError("模型响应缺少 choices[0].message.content")
    usage = _normalize_usage(payload.get("usage"))
    return content, usage, str(payload.get("model") or "")


def chat_completion(
    cfg: Any,
    messages: Sequence[Dict[str, str]],
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
    max_retries: Optional[int] = None,
    backoff_s: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """调用 OpenAI 兼容 chat/completions（标准库 urllib）。

    返回 ``{"text", "usage", "model"}``；失败抛 :class:`CorrectError`（带明确中文原因）。
    仅对 429 / 5xx 做指数退避重试，其余 4xx 直接失败；超时/网络错误立即失败。
    """
    base = str(base_url or _get(cfg, "correct.base_url", "https://api.deepseek.com/v1")).strip()
    url = _chat_url(base)
    key = str(api_key or resolve_api_key(cfg) or "").strip()
    if not key:
        raise CorrectError("缺少 API key（correct.api_key 或 DEEPSEEK_API_KEY）")
    model_name = str(model or _get(cfg, "correct.model", "deepseek-flash")).strip()
    timeout = float(timeout_s if timeout_s is not None else _get(cfg, "correct.timeout_s", 120))
    retries = int(max_retries if max_retries is not None else _get(cfg, "correct.max_retries", 3))
    retries = max(0, retries)
    backoff = float(backoff_s if backoff_s is not None else _get(cfg, "correct.retry_backoff_s", 1.0))
    backoff = max(0.0, backoff)
    backoff_max = float(_get(cfg, "correct.retry_backoff_max_s", 30.0))

    payload = {
        "model": model_name,
        "messages": list(messages),
        "temperature": 0.0,
        "stream": False,
    }
    payload.update(reasoning_payload(reasoning_effort))
    last_error = ""
    response: Optional[Dict[str, Any]] = None
    for attempt in range(retries + 1):
        try:
            response = _post_once(url, payload, key, timeout)
            break
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # pragma: no cover - 读错误体失败不影响判断
                pass
            last_error = f"HTTP {exc.code}：{detail or exc.reason}"
            retryable = exc.code == 429 or 500 <= int(exc.code) < 600
            if retryable and attempt < retries:
                delay = min(backoff_max, backoff * (2 ** attempt))
                log.warning("纠错请求 %s，%.2fs 后重试（%d/%d）", last_error, delay,
                            attempt + 1, retries)
                if delay > 0:
                    time.sleep(delay)
                continue
            raise CorrectError(last_error) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise CorrectError(f"请求超时（{timeout:.1f}s）：{url}") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise CorrectError(f"请求超时（{timeout:.1f}s）：{url}") from exc
            raise CorrectError(f"网络错误：{reason}") from exc
        except OSError as exc:  # pragma: no cover - 连接被重置等
            raise CorrectError(f"网络错误：{exc}") from exc
    if response is None:  # pragma: no cover - 理论不可达
        raise CorrectError(last_error or "请求失败")
    content, usage, resp_model = _extract_content(response)
    return {"text": content.strip(), "usage": usage, "model": resp_model or model_name}


# ================================================================ 安全写回
def _mapped_ok(chunk: Chunk, candidates: Sequence[str], cfg: Any) -> bool:
    """逐段写回前的安全检查：每段非空、单段 diff 率不超限（防模型把段间内容搬串）。"""
    if len(candidates) != len(chunk.segment_texts):
        return False
    max_diff = float(_get(cfg, "correct.max_diff_ratio", 0.12))
    for orig, new in zip(chunk.segment_texts, candidates):
        new = (new or "").strip()
        if not new:
            return False
        if diff_ratio(orig, new) > max_diff:
            return False
    return True


def _segment_spans(chunk: Chunk) -> Optional[List[Tuple[int, int]]]:
    """``chunk.text`` 中每段的 ``[start, end)`` 字符区间（``plan_chunks`` 用 ``\\n`` 拼接）。"""
    text = chunk.text or ""
    spans: List[Tuple[int, int]] = []
    pos = 0
    for seg_text in chunk.segment_texts:
        start = text.find(seg_text, pos)
        if start < 0:
            return None
        spans.append((start, start + len(seg_text)))
        pos = start + len(seg_text)
    return spans


def _aligned_candidates(chunk: Chunk, fixed_text: str) -> Optional[List[str]]:
    """字符级对齐兜底（P9）：用 ``difflib`` opcodes 把修正文本映射回各段。

    适用于模型改了换行/标点、导致「行数、句数都对不上」但**内容顺序未变**的情况：
    只要段内差异不大（由 :func:`_seg_guard` 判断），仍可安全写回。
    插入文本归给「前一个」段（模型在段尾补标点的常见行为）。
    """
    spans = _segment_spans(chunk)
    if not spans:
        return None
    orig = chunk.text or ""
    fixed = fixed_text or ""
    if not orig or not fixed:
        return None
    sm = difflib.SequenceMatcher(None, orig, fixed, autojunk=False)
    boundary = [0] * (len(orig) + 1)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1 + 1):
                boundary[i1 + k] = j1 + k
        elif tag == "replace":
            span = i2 - i1
            for k in range(span + 1):
                # 向上取整：替换区间末尾的字符（常是段尾标点）归给本段，避免把「，」丢掉
                boundary[i1 + k] = j1 + (((j2 - j1) * k + span - 1) // span if span else 0)
        elif tag == "delete":
            for k in range(i2 - i1 + 1):
                boundary[i1 + k] = j1
        else:  # insert
            boundary[i1] = j2
    candidates = [fixed[boundary[start]:boundary[end]].strip() for start, end in spans]
    return candidates


def _candidate_strategies(chunk: Chunk, fixed_text: str):
    """返回按可靠度排序的 ``[(strategy, candidates)]``：行 → 句 → 字符对齐。"""
    strategies = []
    lines = [ln.strip() for ln in (fixed_text or "").splitlines() if ln.strip()]
    if len(lines) == len(chunk.segment_texts):
        strategies.append(("lines", lines))

    flat = _flat(fixed_text)
    sentences = split_sentences(flat)
    expected = sum(chunk.sentence_counts)
    if sentences and expected and len(sentences) == expected:
        candidates: List[str] = []
        pos = 0
        for count in chunk.sentence_counts:
            candidates.append("".join(sentences[pos:pos + count]))
            pos += count
        strategies.append(("sentences", candidates))

    aligned = _aligned_candidates(chunk, fixed_text)
    if aligned:
        strategies.append(("alignment", aligned))
    return strategies


def map_fixed_to_segments(chunk: Chunk, fixed_text: str, cfg: Any) -> Optional[List[str]]:
    """把整块修正文本安全映射回块内各段文本；映射不了一对一回填时返回 ``None``。

    顺序尝试：
    1. 按行映射（prompt 要求保持自然段分行；行数正好等于段数）；
    2. 按句映射（行数对不上时，句数总和正好等于各段句数之和，按段句数顺序分配）；
    3. 字符级对齐（P9 新增：行/句都对不上、但内容顺序未变时按 opcode 求段区间）。
    任一策略要求**每段**都过 ``_mapped_ok``，否则返回 ``None``（宁少改，不串行）。
    只放过部分段的情况走 :func:`_salvage_segments`（逐段回退，不整块丢弃）。
    """
    for _strategy, candidates in _candidate_strategies(chunk, fixed_text):
        if _mapped_ok(chunk, candidates, cfg):
            return candidates
    return None


def _seg_guard(orig: str, new: str, cfg: Any) -> str:
    """单段写回护栏（P9）：空段 / diff / 数字 / 书名号 / 句数，返回失败原因（空串=通过）。

    数字比较用 ``_num_runs``（全角→半角、去千位逗号）；句数变化允许
    ``correct.max_sentence_change_ratio``（短段通常就是 1 句，任何改动都会超限 → 回退该段）。
    """
    new = (new or "").strip()
    if not new:
        return "empty_segment"
    max_diff = float(_get(cfg, "correct.max_diff_ratio", 0.12))
    if diff_ratio(orig, new) > max_diff:
        return "diff_too_large"
    if _num_runs(orig) != _num_runs(new):
        return "number_changed"
    if _book_counts(orig) != _book_counts(new):
        return "book_title_changed"
    max_sent = float(_get(cfg, "correct.max_sentence_change_ratio", 0.05))
    o, f = _flat(orig), _flat(new)
    n_orig = len(split_sentences(o))
    n_fixed = len(split_sentences(f))
    if n_orig and abs(n_fixed - n_orig) / n_orig > max_sent:
        return "sentence_count_changed"
    return ""


def _salvage_segments(chunk: Chunk, fixed_text: str, cfg: Any):
    """逐段抢救（P9）：选「通过段数最多」的映射策略，失败的段保留原文。

    返回 ``(texts, rejections, strategy)``：
    - ``texts`` 长度为段数（失败段 = 原文），``None`` 表示没有任何策略可用（整块回退）；
    - ``rejections`` = ``[{"segment": 段内下标, "reason": ...}]``，只含回退的段；
    - ``strategy`` 为采用的映射策略名（lines/sentences/alignment）。
    真实模型常只在个别段改坏数字/书名号，这种「段级回退」能保住同块其它段的修正。
    """
    best = None
    for strategy, candidates in _candidate_strategies(chunk, fixed_text):
        texts: List[str] = []
        rejections: List[Dict[str, Any]] = []
        for i, (orig, new) in enumerate(zip(chunk.segment_texts, candidates)):
            reason = _seg_guard(orig, new, cfg)
            if reason:
                rejections.append({"segment": i, "reason": reason})
                texts.append(orig)
            else:
                texts.append(new.strip())
        accepted = len(texts) - len(rejections)
        if best is None or accepted > best[0]:
            best = (accepted, texts, rejections, strategy)
    if best is None:
        return None, [], ""
    _accepted, texts, rejections, strategy = best
    return texts, rejections, strategy


def _common_reason(rejections: Sequence[Dict[str, Any]]) -> str:
    """整块全被拒时的原因：各段原因一致就沿用，否则归为 ``unsafe_writeback``。"""
    reasons = {str(r.get("reason") or "") for r in rejections}
    if len(reasons) == 1:
        return next(iter(reasons)) or "unsafe_writeback"
    return "unsafe_writeback"


def _add_flag(seg: Dict[str, Any], flag: str) -> None:
    flags = seg.get("flags")
    if not isinstance(flags, list):
        flags = []
    if flag not in flags:
        flags.append(flag)
    seg["flags"] = flags


# ================================================================ 单块纠错（可直接测试/金标验收）
@dataclass
class _ChunkContext:
    job_id: str
    source: str
    hotwords: List[str]
    topic: str


def _correct_chunk(chunk: Chunk, cfg: Any, ctx: _ChunkContext) -> Dict[str, Any]:
    messages = build_messages(
        chunk, cfg, job_id=ctx.job_id, source=ctx.source,
        hotwords=ctx.hotwords, topic=ctx.topic,
    )
    try:
        response = chat_completion(
            cfg, messages,
            reasoning_effort=str(_get(cfg, "correct.reasoning_effort", "low") or ""),
        )
    except CorrectError as exc:
        return {"ok": False, "error": str(exc), "chunk": chunk.index}
    return {
        "ok": True,
        "text": response["text"],
        "usage": response["usage"],
        "model": response.get("model", ""),
        "chunk": chunk.index,
    }


def correct_text(
    text: str,
    cfg: Any,
    *,
    job_id: str = "",
    source: str = "",
    hotwords: Optional[Sequence[str]] = None,
    topic: str = "",
    chunk_index: int = 1,
) -> Dict[str, Any]:
    """对单块文本跑一次「请求→校验」（供金标验收/调试；不做写回）。

    返回 ``{"ok", "enabled", "orig", "fixed", "diff_ratio", "reason", "items"...}``；
    ``ok=False`` 表示请求失败或校验/映射不安全；``orig`` 为原文（未改动）。
    """
    started = time.perf_counter()
    chunk = Chunk(
        index=chunk_index,
        text=text,
        segment_indices=[0],
        segment_texts=[text],
        sentence_counts=[len(split_sentences(text))],
        char_count=len(text),
        content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest(),
    )
    ctx = _ChunkContext(
        job_id=job_id,
        source=source,
        hotwords=list(hotwords) if hotwords is not None else load_hotwords(cfg, job_id, source),
        topic=topic or text[:300],
    )
    out: Dict[str, Any] = {
        "ok": False, "enabled": True, "orig": text, "fixed": text,
        "diff_ratio": 0.0, "reason": "", "items": [], "usage": _normalize_usage(None),
        "cost_cny": 0.0, "elapsed_s": 0.0, "error": "",
    }
    if not resolve_api_key(cfg):
        out.update({"enabled": False, "ok": True, "reason": "disabled_no_api_key",
                    "elapsed_s": round(time.perf_counter() - started, 3)})
        return out
    response = _correct_chunk(chunk, cfg, ctx)
    out["elapsed_s"] = round(time.perf_counter() - started, 3)
    if not response.get("ok"):
        out.update({"ok": False, "reason": "request_failed",
                    "error": response.get("error", "请求失败")})
        return out
    fixed = response["text"]
    usage = _normalize_usage(response.get("usage"))
    out["usage"] = usage
    out["cost_cny"] = round(compute_cost(usage, cfg), 6)
    detail = validate_correction(text, fixed, cfg)
    block = validate_block(text, fixed, cfg)
    out.update({
        "diff_ratio": detail["diff_ratio"],
        "validation": detail,
        "block": block,
        "fixed_raw": fixed,
    })
    if not block["ok"]:
        out.update({"reason": block["reason"], "fixed": text})
        return out
    texts, rejections, strategy = _salvage_segments(chunk, fixed, cfg)
    if texts is None:
        out.update({"reason": "unsafe_writeback", "fixed": text})
        return out
    accepted = len(texts) - len(rejections)
    if accepted <= 0:
        out.update({"reason": _common_reason(rejections), "fixed": text,
                    "segment_rejections": rejections})
        return out
    out.update({
        "ok": True,
        "fixed": texts[0] if len(texts) == 1 else "\n".join(texts),
        "items": extract_items(chunk, texts, ctx.hotwords),
        "segment_rejections": rejections,
        "mapping": strategy,
    })
    return out


# ================================================================ JSON 产物读写
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.debug("读取 JSON 失败（%s）：%s", path, exc)
        return None


def _write_json(path: Path, payload: Any) -> Path:
    """原子写 JSON（UTF-8、不转义中文、无 NaN/Inf）。"""
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return value if value == value and value not in (float("inf"), float("-inf")) else None
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _empty_corrections() -> Dict[str, Any]:
    return {"total": 0, "items": [], "rejected_chunks": [], "segment_rejections": []}


def _update_report(
    job_out: Path,
    job_id: str,
    *,
    enabled: bool,
    chunks: int,
    edits: int,
    rejected: int,
    cost_cny: float,
    elapsed_s: float,
    rejected_segments: int = 0,
    usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """report.json 读-改-写：只增/换 ``correct`` 与 ``timing.correct_s``，保留既有字段。

    ``usage`` 是本次纠错的 token 用量（prompt/completion/cache_hit），与 ``cost_cny``
    同源落盘，便于事后按 token 对账（此前只写 cost，无法核对）。无 key/空文本时为 0。
    """
    path = job_out / PRODUCT_REPORT
    report = _read_json(path)
    if not isinstance(report, dict):
        report = {}
    report.setdefault("id", job_id)
    usage = usage or {}
    report["correct"] = {
        "enabled": bool(enabled),
        "chunks": int(chunks),
        "edits": int(edits),
        "rejected": int(rejected),
        "rejected_segments": int(rejected_segments),
        "cost_cny": round(float(cost_cny), 6),
        "elapsed_s": round(float(elapsed_s), 3),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "cache_hit_tokens": int(usage.get("cache_hit_tokens", 0) or 0),
            "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
        },
    }
    timing = report.get("timing")
    if not isinstance(timing, dict):
        timing = {}
    timing["correct_s"] = round(float(elapsed_s), 3)
    report["timing"] = timing
    _write_json(path, report)
    return report


def _load_transcript(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise CorrectError(f"缺少或损坏 transcript.json：{path}")
    return [dict(item) for item in data if isinstance(item, dict)]


# ================================================================ 主流程
def correct_job(
    job_id: str,
    cfg: Any,
    *,
    out_dir: Optional[str | Path] = None,
    force: bool = False,
    on_stage: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """对一个 job 跑分块纠错，返回结果字典（失败不抛，返回 ok=False/stage/error）。

    流程：读 transcript → 已有 corrections 且未 force 跳过 → 无 key 跳过但写空结构 →
    plan_chunks → 并发请求 → 逐块校验/安全写回 → 写 transcript/corrections.json/report.json。
    """
    started = time.perf_counter()
    job_out = _out_root(cfg, out_dir) / job_id
    transcript_path = job_out / PRODUCT_TRANSCRIPT
    corrections_path = job_out / PRODUCT_CORRECTIONS
    result: Dict[str, Any] = {
        "ok": False,
        "job_id": job_id,
        "stage": "correct",
        "skipped": False,
        "enabled": False,
        "chunks": 0,
        "edits": 0,
        "rejected": 0,
        "rejected_segments": 0,
        "cost_cny": 0.0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                   "cache_hit_tokens": 0, "reasoning_tokens": 0},
        "elapsed_s": 0.0,
        "error": "",
        "corrections": None,
        "report": None,
        "timings": {},
    }

    def notify(status: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        if on_stage is None:
            return
        try:
            on_stage("correct", status, metrics or {})
        except Exception:  # pragma: no cover - 回调失败不影响主流程
            log.debug("on_stage 回调失败", exc_info=True)

    def finish_ok(metrics: Dict[str, Any]) -> Dict[str, Any]:
        metrics["elapsed_s"] = round(time.perf_counter() - started, 3)
        result.update(metrics)
        result["ok"] = True
        result["timings"] = {"correct_s": result["elapsed_s"]}
        notify("done", {
            "enabled": bool(result.get("enabled")),
            "skipped": bool(result.get("skipped")),
            "chunks": result.get("chunks"),
            "edits": result.get("edits"),
            "rejected": result.get("rejected"),
            "rejected_segments": result.get("rejected_segments"),
            "cost_cny": result.get("cost_cny"),
            "usage": dict(result.get("usage") or {}),
            "elapsed_s": result["elapsed_s"],
        })
        return result

    def finish_fail(message: str) -> Dict[str, Any]:
        elapsed = round(time.perf_counter() - started, 3)
        log.error("纠错失败（%s）：%s", job_id, message)
        result.update({"ok": False, "stage": "correct", "error": message,
                       "elapsed_s": elapsed, "timings": {"correct_s": elapsed}})
        notify("failed", {"error": message})
        return result

    try:
        # 0) 已有 corrections.json → 跳过（--force/--no-cache 重做）
        if corrections_path.is_file() and not force:
            existing = _read_json(corrections_path)
            report = _read_json(job_out / PRODUCT_REPORT)
            if not isinstance(existing, dict):
                existing = _empty_corrections()
            enabled = bool(((report or {}).get("correct") or {}).get("enabled", False)) \
                if isinstance(report, dict) else False
            existing_items = [item for item in (existing.get("items") or [])
                              if isinstance(item, dict)]
            existing_chunks = max(
                [int(item.get("chunk") or 0) for item in existing_items] or [0]
            )
            log.info("corrections.json 已存在，跳过：%s", job_out)
            return finish_ok({
                "skipped": True,
                "enabled": enabled,
                "chunks": existing_chunks,
                "edits": int(existing.get("total") or 0),
                "rejected": len(existing.get("rejected_chunks") or []),
                "rejected_segments": len(existing.get("segment_rejections") or []),
                "corrections": existing,
                "report": report,
            })

        # 1) 读 transcript（纠错只改 text，text_raw/时间/说话人都不动）
        segments = _load_transcript(transcript_path)

        # 2) enabled + API key；缺 key 不阻塞：写空结构 + report，流水线继续
        api_key = resolve_api_key(cfg)
        enabled = bool(_get(cfg, "correct.enabled", True)) and bool(api_key)
        if not enabled:
            reason = "无 API key（correct.api_key / DEEPSEEK_API_KEY）" if not api_key \
                else "correct.enabled=false"
            log.warning("纠错跳过（%s）：%s", reason, job_id)
            corrections = _empty_corrections()
            _write_json(corrections_path, corrections)
            elapsed = round(time.perf_counter() - started, 3)
            report = _update_report(
                job_out, job_id, enabled=False, chunks=0, edits=0,
                rejected=0, cost_cny=0.0, elapsed_s=elapsed,
            )
            return finish_ok({
                "skipped": True, "enabled": False, "chunks": 0, "edits": 0,
                "rejected": 0, "cost_cny": 0.0, "corrections": corrections,
                "report": report,
            })

        # 3) 分块（只按段边界；无文本段跳过）
        chunks = plan_chunks(segments, cfg)
        if not chunks:
            corrections = _empty_corrections()
            _write_json(corrections_path, corrections)
            elapsed = round(time.perf_counter() - started, 3)
            report = _update_report(
                job_out, job_id, enabled=True, chunks=0, edits=0,
                rejected=0, cost_cny=0.0, elapsed_s=elapsed,
            )
            log.info("纠错：没有可纠错文本，写空结构：%s", job_out)
            return finish_ok({
                "enabled": True, "chunks": 0, "edits": 0, "rejected": 0,
                "cost_cny": 0.0, "corrections": corrections, "report": report,
            })

        # 4) 提示词素材：热词 + 课程主题
        source = ""
        old_report = _read_json(job_out / PRODUCT_REPORT)
        if isinstance(old_report, dict) and old_report.get("source"):
            source = str(old_report.get("source") or "")
        hotwords = load_hotwords(cfg, job_id, source)
        ctx = _ChunkContext(
            job_id=job_id, source=source, hotwords=hotwords,
            topic=topic_text(segments),
        )
        log.info("纠错开始：%s chunks=%d chars=%d 热词=%d 并发=%d",
                 job_id, len(chunks), sum(c.char_count for c in chunks),
                 len(hotwords), max(1, int(_get(cfg, "correct.concurrency", 8))))

        # 5) 并发请求（每块一次往返；块内顺序无关，写回按块顺序逐块进行）
        workers = max(1, min(int(_get(cfg, "correct.concurrency", 8)), len(chunks)))
        chunk_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_correct_chunk, chunk, cfg, ctx) for chunk in chunks]
            for future in futures:
                chunk_results.append(future.result())

        # 6) 任一请求失败 → 整个阶段失败（不写半成品）
        failures = [r for r in chunk_results if not r.get("ok")]
        if failures:
            first = failures[0]
            raise CorrectError(
                f"第 {first.get('chunk')} 块请求失败：{first.get('error') or '未知错误'}"
            )

        # 7) 逐块：块级安全门 + 逐段护栏（P9：个别段被拒时保留同块其它段的修正）
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0,
                       "cache_hit_tokens": 0, "reasoning_tokens": 0}
        items: List[Dict[str, Any]] = []
        rejected_chunks: List[Dict[str, Any]] = []
        segment_rejections: List[Dict[str, Any]] = []
        n_changed = 0
        dirty = False
        for chunk, response in zip(chunks, chunk_results):
            usage = _normalize_usage(response.get("usage"))
            for key in usage_total:
                usage_total[key] += usage[key]
            fixed_text = str(response.get("text") or "")
            block = validate_block(chunk.text, fixed_text, cfg)
            if not block["ok"]:
                rejected_chunks.append({
                    "chunk": chunk.index,
                    "diff_ratio": block["diff_ratio"],
                    "reason": block["reason"],
                })
                for idx in chunk.segment_indices:
                    _add_flag(segments[idx], FLAG_REJECTED_CHUNK)
                dirty = True
                log.warning("纠错块 %d 块级校验不过（%s，diff=%.4f），整块回退",
                            chunk.index, block["reason"], block["diff_ratio"])
                continue
            texts, rejections, strategy = _salvage_segments(chunk, fixed_text, cfg)
            if texts is None:
                rejected_chunks.append({
                    "chunk": chunk.index,
                    "diff_ratio": block["diff_ratio"],
                    "reason": "unsafe_writeback",
                })
                for idx in chunk.segment_indices:
                    _add_flag(segments[idx], FLAG_REJECTED_CHUNK)
                dirty = True
                log.warning("纠错块 %d 句/段映射不安全，整块回退", chunk.index)
                continue
            accepted = len(texts) - len(rejections)
            if accepted <= 0:
                reason = _common_reason(rejections)
                rejected_chunks.append({
                    "chunk": chunk.index,
                    "diff_ratio": block["diff_ratio"],
                    "reason": reason,
                })
                for idx in chunk.segment_indices:
                    _add_flag(segments[idx], FLAG_REJECTED_CHUNK)
                dirty = True
                log.warning("纠错块 %d 所有段被逐段护栏拒绝（%s），整块回退",
                            chunk.index, reason)
                continue
            for rejection in rejections:
                local_i = int(rejection.get("segment") or 0)
                seg_idx = chunk.segment_indices[local_i]
                _add_flag(segments[seg_idx], FLAG_REJECTED_SEGMENT)
                segment_rejections.append({
                    "chunk": chunk.index,
                    "segment": seg_idx,
                    "segment_in_chunk": local_i,
                    "reason": rejection.get("reason") or "",
                })
                dirty = True
            for idx, new_text in zip(chunk.segment_indices, texts):
                new_text = new_text.strip()
                if new_text and new_text != _seg_text(segments[idx]):
                    segments[idx]["text"] = new_text
                    n_changed += 1
            items.extend(extract_items(chunk, texts, hotwords))
            if rejections:
                log.info("纠错块 %d：策略=%s 段=%d 逐段回退=%d",
                         chunk.index, strategy, len(texts), len(rejections))

        cost_cny = compute_cost(usage_total, cfg)
        elapsed = round(time.perf_counter() - started, 3)

        # 8) 产物：transcript（有改动/flags 变化才写）+ corrections.json + report.json
        if dirty or n_changed:
            _write_json(transcript_path, segments)
        corrections = {
            "total": len(items),
            "items": items,
            "rejected_chunks": rejected_chunks,
            "segment_rejections": segment_rejections,
        }
        _write_json(corrections_path, corrections)
        report = _update_report(
            job_out, job_id, enabled=True, chunks=len(chunks), edits=len(items),
            rejected=len(rejected_chunks), cost_cny=cost_cny, elapsed_s=elapsed,
            rejected_segments=len(segment_rejections), usage=usage_total,
        )
        log.info(
            "纠错完成：%s chunks=%d edits=%d rejected=%d rejected_segments=%d 改段=%d "
            "cost=¥%.5f %.1fs",
            job_id, len(chunks), len(items), len(rejected_chunks),
            len(segment_rejections), n_changed, cost_cny, elapsed,
        )
        return finish_ok({
            "enabled": True,
            "chunks": len(chunks),
            "edits": len(items),
            "rejected": len(rejected_chunks),
            "rejected_segments": len(segment_rejections),
            "cost_cny": round(cost_cny, 6),
            "usage": usage_total,
            "corrections": corrections,
            "report": report,
            "correct": report.get("correct"),
        })
    except CorrectError as exc:
        return finish_fail(str(exc))
    except Exception as exc:  # pragma: no cover - 兜底：绝不把异常抛给批处理
        log.error("纠错未预期异常：%s", job_id, exc_info=True)
        return finish_fail(f"{exc.__class__.__name__}：{exc}")
