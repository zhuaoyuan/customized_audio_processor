# -*- coding: utf-8 -*-
"""推理强度（thinking / reasoning_effort）小样本实验。

只读产物 + 真实 API 调用；结果写 work/reasoning_lab/（gitignore）。
用法：
    python work/reasoning_lab.py smoke
    python work/reasoning_lab.py correct
    python work/reasoning_lab.py compose
    python work/reasoning_lab.py analyze
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app.pipeline.correct as C  # noqa: E402
import app.pipeline.compose as M  # noqa: E402
from app.config import load_config  # noqa: E402

JOB_ID = "1c34b7df_复旦哲学系丁耘_亚里士多德_物理学_01"
JOB_OUT = Path(r"D:\course\test\output") / JOB_ID
LAB = ROOT / "work" / "reasoning_lab"
TEXT_DIR = LAB / "texts"
CALLS_PATH = LAB / "calls.jsonl"

EFFORTS = {
    "off": {"thinking": {"type": "disabled"}},
    "low": {"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
    "high": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    "max": {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
}

_TLS = threading.local()
_GLOBAL = {"extra": {}}
_LOCK = threading.Lock()


def _current_extra():
    val = getattr(_TLS, "extra", None)
    if val is not None:
        return val
    return _GLOBAL["extra"]


def _patched_chat(cfg, messages, *, base_url=None, api_key=None, model=None,
                  timeout_s=None, max_retries=None, backoff_s=None):
    """复刻 correct.chat_completion，但注入 thinking / reasoning_effort。"""
    import urllib.error

    base = str(base_url or C._get(cfg, "correct.base_url", "https://api.deepseek.com/v1")).strip()
    url = C._chat_url(base)
    key = str(api_key or C.resolve_api_key(cfg) or "").strip()
    model_name = str(model or C._get(cfg, "correct.model", "deepseek-flash")).strip()
    timeout = 600.0
    extra = dict(_current_extra())
    payload = {
        "model": model_name,
        "messages": list(messages),
        "temperature": 0.0,
        "stream": False,
    }
    payload.update(extra)
    last = None
    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            parsed = C._post_once(url, payload, key, timeout)
            elapsed = round(time.perf_counter() - t0, 2)
            text, usage, resp_model = C._extract_content(parsed)
            raw = parsed.get("usage") or {}
            reasoning = int(((raw.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
            usage["reasoning_tokens"] = reasoning
            call = {"elapsed_s": elapsed, "usage": usage,
                    "model": resp_model or model_name, "effort": extra}
            if getattr(_TLS, "owner", None) == "worker":
                _TLS.calls = list(getattr(_TLS, "calls", [])) + [call]
            else:
                with _LOCK:
                    _GLOBAL.setdefault("calls", []).append(call)
            return {"text": text.strip(), "usage": usage,
                    "model": resp_model or model_name}
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}"
            if exc.code == 429 or 500 <= int(exc.code) < 600:
                if attempt == 0:
                    time.sleep(2.0)
                    continue
            raise C.CorrectError(last) from exc
        except Exception as exc:  # noqa: BLE001
            last = f"{exc.__class__.__name__}: {exc}"
            if attempt == 0 and "timeout" not in str(exc).lower():
                time.sleep(2.0)
                continue
            raise C.CorrectError(last) from exc
    raise C.CorrectError(last or "request failed")


C.chat_completion = _patched_chat
M.chat_completion = _patched_chat


def _load_samples():
    transcript = json.loads((JOB_OUT / "transcript.json").read_text(encoding="utf-8"))
    # 纠错实验必须用 text_raw（ASR 原始输出）；transcript.json 的 text 已被上一轮 correct 改过
    raw_transcript = [dict(seg, text=(seg.get("text_raw") or seg.get("text") or ""))
                      for seg in transcript]
    cfg = load_config(overrides={"compose.timeout_s": 900, "compose.analyze_timeout_s": 900})
    hotwords = C.load_hotwords(cfg, JOB_ID, str(JOB_OUT))
    topic = C.topic_text(raw_transcript)
    chunks = C.plan_chunks(raw_transcript, cfg)
    gold = (ROOT / "work" / "cost" / "correct_test" / "input_chunk1.txt").read_text(encoding="utf-8")
    correct_samples = [("gold4200", gold)]
    for idx in (1, 7):
        if idx < len(chunks):
            correct_samples.append((f"real_c{idx + 1}", chunks[idx].text))

    segments = M.load_segments(JOB_OUT / "transcript.json")
    mchunks = M.plan_chunks(segments, cfg)
    card = json.loads((JOB_OUT / "compose_card.json").read_text(encoding="utf-8"))
    compose_samples = [("m1", mchunks[0]), ("m9", mchunks[8])]
    return cfg, correct_samples, hotwords, topic, compose_samples, card


def _record(row):
    with _LOCK:
        with CALLS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_correct(name, text, effort, cfg, hotwords, topic):
    _TLS.owner = "worker"
    _TLS.extra = EFFORTS[effort]
    _TLS.calls = []
    t0 = time.perf_counter()
    try:
        res = C.correct_text(text, cfg, job_id=JOB_ID, source=str(JOB_OUT),
                             hotwords=hotwords, topic=topic)
    except Exception as exc:  # noqa: BLE001
        res = {"ok": False, "error": f"{exc.__class__.__name__}: {exc}", "fixed": text,
               "orig": text, "usage": {}, "cost_cny": 0.0, "items": [],
               "block": {}, "validation": {}, "segment_rejections": []}
    elapsed = round(time.perf_counter() - t0, 2)
    row = {"task": "correct", "sample": name, "effort": effort,
           "elapsed_s": elapsed, "calls": list(getattr(_TLS, "calls", [])),
           "ok": bool(res.get("ok")), "reason": res.get("reason", ""),
           "block": res.get("block"), "diff_ratio": res.get("diff_ratio"),
           "n_items": len(res.get("items") or []),
           "n_seg_rej": len(res.get("segment_rejections") or []),
           "usage": res.get("usage"), "cost_cny": res.get("cost_cny")}
    fixed = res.get("fixed") or text
    row["orig_chars"] = len(text)
    row["fixed_chars"] = len(fixed)
    row["num_ok"] = C._num_runs(text) == C._num_runs(fixed)
    row["book_ok"] = text.count("《") == fixed.count("《")
    row["fixed"] = fixed
    (TEXT_DIR / f"correct_{name}_{effort}.txt").write_text(fixed, encoding="utf-8")
    _record(row)
    return row


def _run_compose(name, chunk, effort, cfg, card):
    _GLOBAL["extra"] = EFFORTS[effort]
    _GLOBAL["calls"] = []
    _TLS.extra = None
    _TLS.calls = []
    t0 = time.perf_counter()
    try:
        res = M.compose_chunk(chunk, card, cfg, cache_enabled=False)
    except Exception as exc:  # noqa: BLE001
        res = {"ok": False, "fallback": True, "error": f"{exc.__class__.__name__}: {exc}",
               "validation": {"reasons": ["exception"], "warnings": []}, "usage": {},
               "cost_cny": 0.0, "text": "", "paragraphs": [], "out_chars": 0,
               "orig_chars": len(chunk.text), "attempts": 0, "elapsed_s": 0}
    elapsed = round(time.perf_counter() - t0, 2)
    usage = res.get("usage") or {}
    row = {"task": "compose", "sample": name, "effort": effort,
           "elapsed_s": elapsed, "calls": list(_GLOBAL.get("calls") or []),
           "ok": bool(res.get("ok")), "fallback": bool(res.get("fallback")),
           "attempts": res.get("attempts"), "validation": res.get("validation"),
           "out_chars": res.get("out_chars"), "orig_chars": res.get("orig_chars"),
           "n_paras": len(res.get("paragraphs") or []),
           "usage": usage, "cost_cny": res.get("cost_cny"),
           "text": res.get("text") or ""}
    (TEXT_DIR / f"compose_{name}_{effort}.txt").write_text(row["text"], encoding="utf-8")
    _record(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "correct", "compose", "card", "analyze"])
    args = ap.parse_args()
    LAB.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    if not (os.environ.get("DEEPSEEK_API_KEY") or "").strip():
        sys.exit("需要先设置 DEEPSEEK_API_KEY")

    cfg, correct_samples, hotwords, topic, compose_samples, card = _load_samples()

    if args.phase == "smoke":
        rows = [_run_correct("gold4200", correct_samples[0][1], e, cfg, hotwords, topic)
                for e in ("off", "low")]
        for r in rows:
            print(json.dumps({k: r[k] for k in ("sample", "effort", "ok", "elapsed_s",
                                              "cost_cny", "usage", "num_ok")},
                             ensure_ascii=False))
        return

    if args.phase == "correct":
        # 先清掉旧记录，避免重复累计
        if CALLS_PATH.exists():
            CALLS_PATH.unlink()
        jobs = [(name, text, effort) for name, text in correct_samples for effort in EFFORTS]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(_run_correct, n, t, e, cfg, hotwords, topic) for n, t, e in jobs]
            for fut in as_completed(futs):
                r = fut.result()
                print(f"[correct] {r['sample']:<10} {r['effort']:<4} ok={r['ok']} "
                      f"elapsed={r['elapsed_s']}s cost={r['cost_cny']} items={r['n_items']}")
        return

    if args.phase == "compose":
        # 顺序跑：_chat_with_deadline 在子线程调用，全局注入更稳
        for name, chunk in compose_samples:
            for effort in EFFORTS:
                r = _run_compose(name, chunk, effort, cfg, card)
                print(f"[compose] {name:<4} {effort:<4} ok={r['ok']} fallback={r['fallback']} "
                      f"elapsed={r['elapsed_s']}s cost={r['cost_cny']} paras={r['n_paras']}")
        return

    if args.phase == "card":
        segments = M.load_segments(JOB_OUT / "transcript.json")
        for effort in ("low", "high"):
            _GLOBAL["extra"] = EFFORTS[effort]
            _GLOBAL["calls"] = []
            t0 = time.perf_counter()
            row = {"task": "card", "sample": "card", "effort": effort}
            try:
                got = M.analyze_lecture(segments, cfg, job_id=JOB_ID, source=str(JOB_OUT))
                row.update({
                    "ok": True,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                    "usage": (got.get("_meta") or {}).get("usage"),
                    "cost_cny": (got.get("_meta") or {}).get("cost_cny"),
                    "n_outline": len(got.get("outline") or []),
                    "n_glossary": len(got.get("glossary") or []),
                    "n_people": len(got.get("people") or []),
                    "n_books": len(got.get("books") or []),
                    "title": got.get("title"),
                    "calls": list(_GLOBAL.get("calls") or []),
                })
                (TEXT_DIR / f"card_{effort}.json").write_text(
                    json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                row.update({"ok": False, "elapsed_s": round(time.perf_counter() - t0, 2),
                            "error": f"{exc.__class__.__name__}: {exc}"})
            _record(row)
            print(f"[card] {effort:<4} ok={row['ok']} elapsed={row['elapsed_s']}s "
                  f"cost={row.get('cost_cny')} outline={row.get('n_outline')} "
                  f"glossary={row.get('n_glossary')} err={row.get('error','')}")
        return

    analyze()


# ================================================================ analyze
GOLD = json.loads((ROOT / "work" / "cost" / "correct_test" / "corrections_chunk1.json")
                  .read_text(encoding="utf-8"))
GOLD_REF = (ROOT / "work" / "cost" / "correct_test" / "corrected_chunk1.txt").read_text(encoding="utf-8")


def _variants(raw):
    parts = [p.strip() for p in str(raw).split("/") if p.strip()]
    return parts


def _gold_recall(orig, fixed):
    total = hit = 0
    remaining = 0
    for item in GOLD["items"]:
        for v in _variants(item.get("orig", "")):
            n = orig.count(v)
            if not n:
                continue
            total += n
            left = fixed.count(v)
            hit += max(0, n - left)
            remaining += min(left, n)
    return {"wrong_total": total, "fixed_away": hit,
            "recall": round(hit / total, 4) if total else None,
            "wrong_remaining": remaining}


def _glossary_recall(orig, fixed, card):
    total = hit = remaining = 0
    for item in card.get("glossary") or []:
        for v in (item.get("variants") or []):
            v = str(v).strip()
            if not v:
                continue
            n = orig.count(v)
            if not n:
                continue
            total += n
            left = min(n, fixed.count(v))
            hit += n - left
            remaining += left
    return {"total": total, "recall": round(hit / total, 4) if total else None,
            "remaining": remaining}


def _sim(a, b):
    return round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 4)


def analyze():
    rows = [json.loads(line) for line in CALLS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    # 小样本原文（correct 的 gold/real 块、compose 的 m1/m9）
    cfg, correct_samples, hotwords, topic, compose_samples, card = _load_samples()
    orig_by_name = {n: txt for n, txt in correct_samples}
    mchunks = M.plan_chunks(M.load_segments(JOB_OUT / "transcript.json"), cfg)
    orig_by_name.update({n: ch.text for n, ch in compose_samples})
    card_glossary = [str(g.get("correct") or "") for g in (card.get("glossary") or [])]
    fillers = ("呃", "啊", "嗯", "这个这个", "就是就是")
    highs = {(r["task"], r["sample"]): (r.get("fixed") or r.get("text") or "")
             for r in rows if r["effort"] == "high"}

    slim = []
    for r in rows:
        usage = r.get("usage") or {}
        reasoning = sum(int((c.get("usage") or {}).get("reasoning_tokens") or 0)
                        for c in r.get("calls") or [])
        orig = orig_by_name.get(r["sample"], "")
        fixed = r.get("fixed") or r.get("text") or ""
        row = {
            "task": r["task"], "sample": r["sample"], "effort": r["effort"],
            "ok": r["ok"], "elapsed_s": r["elapsed_s"],
            "cost_cny": round(float(r.get("cost_cny") or 0), 6),
            "prompt": int(usage.get("prompt_tokens") or 0),
            "completion": int(usage.get("completion_tokens") or 0),
            "cache_hit": int(usage.get("cache_hit_tokens") or 0),
            "reasoning": reasoning,
        }
        if r["task"] == "card":
            row.update({
                "ok": bool(r.get("ok")),
                "n_outline": r.get("n_outline"),
                "n_glossary": r.get("n_glossary"),
                "n_people": r.get("n_people"),
                "n_books": r.get("n_books"),
            })
        elif r["task"] == "correct":
            row.update({
                "ok": bool(r.get("ok")),
                "diff_ratio": r.get("diff_ratio"),
                "n_items": r.get("n_items"),
                "n_seg_rej": r.get("n_seg_rej"),
                "reason": r.get("reason"),
                "sim_to_orig": _sim(fixed, orig),
                "num_equal": r.get("num_ok"),
                "book_equal": r.get("book_ok"),
                "diff_vs_high": _sim(fixed, highs.get(("correct", r["sample"]), fixed)),
            })
            if r["sample"] == "gold4200":
                g = _gold_recall(orig, fixed)
                row.update({
                    "gold_wrong_total": g["wrong_total"],
                    "gold_recall": g["recall"],
                    "gold_wrong_remaining": g["wrong_remaining"],
                    "sim_to_gold_ref": _sim(fixed, GOLD_REF),
                })
            else:
                g = _glossary_recall(orig, fixed, card)
                row.update({
                    "glossary_total": g["total"],
                    "glossary_recall": g["recall"],
                    "glossary_remaining": g["remaining"],
                })
        else:
            val = r.get("validation") or {}
            row.update({
                "ok": bool(r.get("ok")),
                "fallback": bool(r.get("fallback")),
                "attempts": r.get("attempts"),
                "ratio": round((r.get("out_chars") or 0) / max(1, r.get("orig_chars") or 1), 3),
                "n_paras": r.get("n_paras"),
                "warnings": val.get("warnings") or [],
                "validation_ok": bool(val.get("ok")),
                "glossary_hit": sum(1 for term in card_glossary if term and term in fixed),
                "fillers": {f: fixed.count(f) for f in fillers},
                "sim_to_high": _sim(fixed, highs.get(("compose", r["sample"]), fixed)),
            })
        slim.append(row)

    (LAB / "summary.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = []

    def table(rows_, cols, title):
        lines.append(f"\n### {title}\n")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for rr in rows_:
            lines.append("| " + " | ".join(str(rr.get(c, "")) for c in cols) + " |")

    crows = [r for r in slim if r["task"] == "correct"]
    table(crows, ["sample", "effort", "ok", "elapsed_s", "cost_cny", "prompt", "completion",
                  "reasoning", "diff_ratio", "n_items", "gold_recall", "gold_wrong_remaining",
                  "glossary_recall", "glossary_remaining", "sim_to_orig", "num_equal", "book_equal"],
          "correct 小样本（gold4200 有金标；real_c2/c8 为 ASR 原文，评价术语表变体召回）")
    mrows = [r for r in slim if r["task"] == "compose"]
    table(mrows, ["sample", "effort", "ok", "fallback", "attempts", "elapsed_s", "cost_cny",
                  "prompt", "completion", "reasoning", "ratio", "n_paras", "warnings",
                  "glossary_hit", "sim_to_high"],
          "compose 小样本（2 块）")
    krows = [r for r in slim if r["task"] == "card"]
    table(krows, ["sample", "effort", "ok", "elapsed_s", "cost_cny", "prompt", "completion",
                  "reasoning", "n_outline", "n_glossary", "n_people", "n_books"],
          "compose 课程档案（低 vs 高；生产默认 high）")
    text = "\n".join(lines)
    (LAB / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
