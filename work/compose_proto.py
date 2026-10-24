"""Prototype: 全文语境 + 分块整理成稿（P10 原型）.

用法:
    python work/compose_proto.py analyze
    python work/compose_proto.py chunks 3      # 只跑前 3 块
    python work/compose_proto.py full
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 读本地 key（不打印）
raw_key = (ROOT / "work" / "set_key.ps1").read_text(encoding="utf-8", errors="replace")
m = re.search(r"sk-[A-Za-z0-9]+", raw_key)
if m and not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = m.group(0)

from app.config import load_config
from app.pipeline.correct import chat_completion

OUT = ROOT / "work" / "compose_proto"
OUT.mkdir(parents=True, exist_ok=True)

JOB = "0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05"
TRANSCRIPT = ROOT / "out" / JOB / "transcript.json"

SYSTEM = """你是中文学术讲座录音转写稿的整理者。输入是语音识别（ASR）的逐句转写，含大量口语碎片、重复、语气词和同音误写。你的任务：把它整理成一篇完整、通顺、可阅读的书面讲稿。

必须严格遵守：
1. 忠实：只整理原文已有内容，绝不添加原文没有的观点、论据、例证、引文、人名或评价；不改变讲授者的立场、论证顺序和例子。没有把握的地方保留原意，不要臆测补全。
2. 纠错：依据【课程档案】中的术语表统一专名和术语；明显的同音错字（如「无线→无限」「签能→潜能」）直接改正；没有把握的保持原样，不要乱改。
3. 去口语：删除「呃、啊、嗯、这个这个、就是就是」等语气词与无意义重复；合并被停顿切碎的句子；依据上下文补全省略的主语、宾语和连接词（只能补结构，不得增加实质信息）；课堂管理、寒暄等与讲授内容无关的闲话可大幅删减。
4. 保留：实质讲授内容（概念解释、论证步骤、例子、书目、页码、数字、引文）必须保留；书名统一加《》，但只能给原文确实提到的书名加；数字只保留原文有意义的数字（卷次、年份、页码等），不新增数字，不换算、不补零、不改全角半角；转写噪声里的数字按术语表改正。
5. 文体：第一人称讲授口吻的书面稿；自然段，每段 80–400 字；不要 Markdown 标题、列表、加粗、代码块；不要任何「原文/整理后」式说明。
6. 时间：输入每行以 [mm:ss] 开头。输出的每个自然段以 [mm:ss] 开头，写该段第一句对应的输入行时间（原样复制，不要自己换算）。段落之间空一行，不要输出其它时间标记。"""

ANALYZE_SYSTEM = """你是中文学术讲座转写稿的编辑。给你一整节课的逐句转写（每行前是时间戳）。请通读全文，输出一份供后续分块整理使用的「课程档案」，只输出 JSON，不要 Markdown 代码块或解释。

JSON 结构：
{
  "title": "课程名 + 本讲主题（简短）",
  "topic": "本讲核心问题与内容概述（100–200 字）",
  "speaker": "讲授人（文中可辨认时）",
  "outline": [{"time": "mm:ss", "title": "该段内容的小标题（8–16 字）"}],
  "glossary": [{"correct": "正确写法", "variants": ["转写中实际出现的错写/异写"], "note": "为何这样改（简短）"}],
  "people": ["本讲涉及的人物"],
  "books": ["本讲涉及的书名，含《》"],
  "notes": "整理时需特别注意的转写系统性问题（50–150 字）"
}

要求：
- outline 覆盖全讲，按时间顺序 8–15 条，time 必须取自输入中真实出现的时间戳。
- glossary 20–50 条，只收本讲实际出现、且转写确有错误或需要统一的专名/术语；variants 必须是输入中真实出现的字面串，不要虚构。
- people/books 只收本讲实际提到的，不要用先验知识扩充。
- 全部使用中文；JSON 字符串内不要出现未转义的换行。"""


def fmt_time(sec: float) -> str:
    total = int(round(sec))
    return f"{total // 60:02d}:{total % 60:02d}"


def load_segments() -> List[Dict[str, Any]]:
    segs = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    return [s for s in segs if (s.get("text") or "").strip()]


def source_lines(segs: Sequence[Dict[str, Any]]) -> List[str]:
    return [f"[{fmt_time(s['start'])}] {s['text'].strip()}" for s in segs]


@dataclass
class Chunk:
    index: int
    segs: List[Dict[str, Any]]
    overlap: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""

    @property
    def start_s(self) -> float:
        return float(self.segs[0]["start"])

    @property
    def end_s(self) -> float:
        return float(self.segs[-1]["end"])


def plan_chunks(segs: Sequence[Dict[str, Any]], target: int = 1800, overlap: int = 2) -> List[Chunk]:
    chunks: List[Chunk] = []
    cur: List[Dict[str, Any]] = []
    cur_len = 0
    start_idx = 0
    while start_idx < len(segs):
        cur = []
        cur_len = 0
        j = start_idx
        while j < len(segs):
            t = (segs[j].get("text") or "").strip()
            if cur and cur_len + len(t) > target:
                break
            cur.append(segs[j])
            cur_len += len(t)
            j += 1
        if not cur:
            break
        chunks.append(Chunk(index=len(chunks) + 1, segs=list(cur)))
        if j >= len(segs):
            break
        start_idx = max(start_idx + 1, j - overlap)
    return chunks


def strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:markdown|md|json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def analyze(cfg: Any, segs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lines = source_lines(segs)
    body = "\n".join(lines)
    t0 = time.time()
    res = chat_completion(
        cfg,
        [
            {"role": "system", "content": ANALYZE_SYSTEM},
            {"role": "user", "content": f"【全课转写】\n{body}\n\n【输出 JSON】"},
        ],
        timeout_s=600,
    )
    txt = strip_fences(res["text"])
    start = txt.find("{")
    end = txt.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError(f"no json in analysis output: {txt[:300]}")
    card = json.loads(txt[start : end + 1])
    # 校验 variants 真实出现
    full = "".join((s.get("text") or "") for s in segs)
    dropped = 0
    for g in card.get("glossary", []):
        vs = [v for v in g.get("variants", []) if v and v in full]
        dropped += len(g.get("variants", [])) - len(vs)
        g["variants"] = vs
    card["_meta"] = {"elapsed_s": round(time.time() - t0, 1), "usage": res["usage"], "dropped_variants": dropped}
    (OUT / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return card


def card_prompt(card: Dict[str, Any]) -> str:
    lines = ["【课程档案】", f"标题：{card.get('title', '')}", f"主题：{card.get('topic', '')}"]
    outline = card.get("outline") or []
    if outline:
        lines.append("结构：" + "；".join(f"[{o.get('time')}] {o.get('title')}" for o in outline))
    gl = card.get("glossary") or []
    if gl:
        parts = []
        for g in gl:
            vs = "/".join(g.get("variants") or [])
            parts.append(f"{g.get('correct')}（转写作 {vs}）" if vs else str(g.get("correct")))
        lines.append("术语表：" + "；".join(parts))
    if card.get("people"):
        lines.append("人物：" + "、".join(card["people"]))
    if card.get("books"):
        lines.append("书目：" + "、".join(card["books"]))
    if card.get("notes"):
        lines.append("注意：" + str(card["notes"]))
    return "\n".join(lines)


def build_chunk_messages(chunk: Chunk, card: Dict[str, Any], prev_tail: str, strict: bool = False) -> List[Dict[str, str]]:
    header = card_prompt(card)
    src = "\n".join(f"[{fmt_time(s['start'])}] {s['text'].strip()}" for s in chunk.segs)
    parts = [header, ""]
    if prev_tail:
        parts += ["【前文结尾（仅供衔接，不要重复输出）】", prev_tail, ""]
    parts += ["【待整理转写】", src, ""]
    task = "【输出】直接输出整理后的自然段，每段以 [mm:ss] 开头，段落之间空一行。"
    if strict:
        task += "注意：不要添加原文没有的内容，不要删掉例子、论证与数字，时间戳只能取自上面输入行。"
    parts.append(task)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


NUM_RE = re.compile(r"\d+")


def norm_text(text: str) -> str:
    text = re.sub(r"^\[?\d{1,3}:\d{2}\]?\s*", "", (text or "").strip(), flags=re.M)
    text = re.sub(r"\s+", "", text)
    return text


def validate(orig: str, fixed: str, valid_times: Sequence[str], allowed_drop_digits: Sequence[str] = ()) -> Dict[str, Any]:
    """校验整理稿：硬性（空/长度/时间戳/新增数字）+ 软性（丢失数字、书名号增加）。"""
    o = norm_text(orig)
    f = norm_text(fixed)
    hard: List[str] = []
    soft: List[str] = []
    if not f:
        return {"ok": False, "reasons": ["empty"], "warnings": []}
    ratio = len(f) / max(1, len(o))
    if ratio < 0.42:
        hard.append(f"too_short:{ratio:.2f}")
    if ratio > 1.25:
        hard.append(f"too_long:{ratio:.2f}")
    on = NUM_RE.findall(o)
    fn = NUM_RE.findall(f)
    added = [x for x in fn if x not in on]
    if added:
        hard.append(f"added_numbers:{added[:6]}")
    allowed = set(allowed_drop_digits)
    dropped = [x for x in on if x not in fn and x not in allowed]
    if dropped:
        soft.append(f"dropped_numbers:{dropped[:6]}")
    ob = o.count("《")
    fb = f.count("《")
    if fb < ob:
        soft.append(f"books_lost:{ob}->{fb}")
    elif fb > ob:
        soft.append(f"books_added:{ob}->{fb}")
    # timestamps
    stamps = re.findall(r"^\s*\[(\d{1,3}:\d{2})\]", fixed, flags=re.M)
    if len(stamps) < 1:
        hard.append("no_timestamps")
    else:
        vset = set(valid_times)
        bad = [s for s in stamps if s not in vset]
        if bad:
            hard.append(f"bad_times:{bad[:3]}")
        secs = [int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in stamps]
        if any(b < a for a, b in zip(secs, secs[1:])):
            hard.append("non_monotonic")
    return {"ok": not hard, "reasons": hard, "warnings": soft, "ratio": round(ratio, 3)}


def run_chunks(cfg: Any, segs: Sequence[Dict[str, Any]], card: Dict[str, Any], limit: Optional[int] = None,
               target: int = 1800) -> Dict[str, Any]:
    chunks = plan_chunks(segs, target=target)
    if limit:
        chunks = chunks[:limit]
    cdir = OUT / "chunks"
    cdir.mkdir(exist_ok=True)
    allowed_drop = set()
    for g in card.get("glossary", []):
        for v in g.get("variants", []):
            allowed_drop.update(NUM_RE.findall(v or ""))
    out_lines: List[str] = []
    prev_tail = ""
    report = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0}
    t_start = time.time()
    for ch in chunks:
        valid_times = [fmt_time(s["start"]) for s in ch.segs]
        orig = "\n".join(s["text"].strip() for s in ch.segs)
        msgs = build_chunk_messages(ch, card, prev_tail)
        attempt = 0
        result = None
        t_chunk = time.time()
        while attempt < 2:
            res = chat_completion(cfg, msgs, timeout_s=300)
            for k in usage_total:
                usage_total[k] += int(res["usage"].get(k, 0) or 0)
            text = strip_fences(res["text"])
            (cdir / f"chunk_{ch.index:02d}_attempt{attempt + 1}.txt").write_text(text, encoding="utf-8")
            v = validate(orig, text, valid_times, allowed_drop)
            result = {"text": text, "validation": v, "usage": res["usage"], "attempt": attempt + 1}
            if v["ok"]:
                break
            attempt += 1
            msgs = build_chunk_messages(ch, card, prev_tail, strict=True)
        text = result["text"] if result else ""
        if not (result and result["validation"]["ok"]):
            # 兜底：机械去语气词，保留原顺序
            text = f"[{fmt_time(ch.start_s)}] " + re.sub(r"[呃啊嗯]+|[，。]{2,}", "", orig)
            (cdir / f"chunk_{ch.index:02d}_fallback.txt").write_text(text, encoding="utf-8")
        out_lines.append(text.strip())
        tail = norm_text(text)[-200:]
        prev_tail = tail
        entry = {
            "chunk": ch.index,
            "start": fmt_time(ch.start_s),
            "end": fmt_time(ch.end_s),
            "n_seg": len(ch.segs),
            "orig_chars": len(orig),
            "out_chars": len(norm_text(text)),
            "elapsed_s": round(time.time() - t_chunk, 1),
        }
        entry.update(result["validation"] if result else {"ok": False, "reasons": ["no_result"]})
        report.append(entry)
        print(f"chunk {ch.index}/{len(chunks)} {fmt_time(ch.start_s)}-{fmt_time(ch.end_s)} "
              f"ok={report[-1].get('ok')} ratio={report[-1].get('ratio')} "
              f"reasons={report[-1].get('reasons')} warn={report[-1].get('warnings')}",
              flush=True)
    total_out = "\n\n".join(out_lines)
    (OUT / f"manuscript_{len(chunks)}chunks.md").write_text(total_out + "\n", encoding="utf-8")
    rep = {
        "job": JOB,
        "chunks": len(chunks),
        "usage": usage_total,
        "elapsed_s": round(time.time() - t_start, 1),
        "chunk_report": report,
    }
    (OUT / f"report_{len(chunks)}chunks.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    segs = load_segments()
    mode = sys.argv[1] if len(sys.argv) > 1 else "chunks"
    if mode == "analyze":
        card = analyze(cfg, segs)
        print(json.dumps(card, ensure_ascii=False, indent=2)[:5000])
    else:
        card_path = OUT / "card.json"
        if card_path.exists():
            card = json.loads(card_path.read_text(encoding="utf-8"))
        else:
            card = analyze(cfg, segs)
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        rep = run_chunks(cfg, segs, card, limit=limit)
        print(json.dumps(rep["usage"], ensure_ascii=False))
        print("elapsed", rep["elapsed_s"], "chunks", rep["chunks"])


if __name__ == "__main__":
    main()
