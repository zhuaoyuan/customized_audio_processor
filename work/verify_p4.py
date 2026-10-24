# -*- coding: utf-8 -*-
"""P4 六文件验收：transcript 契约、双轴往返、report 字段、stages、抽段。"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config
from app.models import AnalyzeResult, KeptSpan
from app.pipeline.asr import build_segment_table
from app.pipeline.trim import TimeMap

cfg = load_config()
out_root = cfg.path("out_dir")
cache = cfg.path("cache_dir")
job_dirs = sorted(p for p in out_root.iterdir() if p.is_dir() and (p / "transcript.json").is_file())

rows = []
all_ok = True
for d in job_dirs:
    job_id = d.name
    tr = json.loads((d / "transcript.json").read_text(encoding="utf-8"))
    report = json.loads((d / "report.json").read_text(encoding="utf-8"))
    analyze = json.loads((cache / f"{job_id}.analyze.json").read_text(encoding="utf-8"))
    ar = AnalyzeResult(**analyze["result"])
    edit = json.loads((d / "edit.json").read_text(encoding="utf-8"))
    tm = TimeMap([KeptSpan(*k) for k in edit["kept"]])
    expected = build_segment_table(ar, edit["kept"], cfg)
    asr = report.get("asr") or {}
    speakers = report.get("speakers") or {}
    errors = []
    if not isinstance(tr, list):
        errors.append("transcript 顶层不是 list")
    if len(tr) != asr.get("n_segments", -1) - asr.get("n_empty", -1):
        errors.append(f"条数 {len(tr)} != n_segments-n_empty {asr.get('n_segments')}-{asr.get('n_empty')}")
    if len(expected) != asr.get("n_segments"):
        errors.append(f"段表 {len(expected)} != report {asr.get('n_segments')}")
    if any(tr[i]["start"] < tr[i - 1]["start"] for i in range(1, len(tr))):
        errors.append("start 非升序")
    overlaps = sum(1 for i in range(1, len(tr)) if tr[i]["start"] < tr[i - 1]["end"] - 1e-6)
    if overlaps:
        errors.append(f"重叠 {overlaps} 段")
    if any(x["end"] <= x["start"] for x in tr):
        errors.append("存在 end<=start")
    if any(x["speaker"] != "主讲" for x in tr):
        errors.append("存在非主讲")
    if any(not isinstance(x["flags"], list) for x in tr):
        errors.append("flags 非 list")
    if any(not x["text"] for x in tr):
        errors.append("存在空 text")
    if speakers.get("n") != 1 or speakers.get("method") != "none":
        errors.append(f"speakers 字段异常 {speakers}")
    if not isinstance(asr.get("rtf"), (int, float)):
        errors.append("rtf 缺失")
    if "asr_s" not in (report.get("timing") or {}):
        errors.append("timing.asr_s 缺失")
    # 双轴往返
    max_rt = 0.0
    for x in tr:
        o = tm.clean_to_orig(x["start"])
        if o is None:
            errors.append(f"clean->orig 失败 {x['start']}")
            break
        back = tm.orig_to_clean(o)
        max_rt = max(max_rt, abs(back - x["start"]) if back is not None else 9e9)
    if max_rt > 1e-3:
        errors.append(f"往返 {max_rt * 1000:.4f}ms >1ms")
    clean_dur = float((report.get("trim") or {}).get("clean_dur_s") or 0)
    if tr and (tr[-1]["end"] > clean_dur + 0.01 or tr[0]["start"] < -0.01):
        errors.append("时间越界")
    chars = sum(len([c for c in x["text"] if ("\u4e00" <= c <= "\u9fff" or c.isalnum())]) for x in tr)
    if chars != asr.get("chars"):
        errors.append(f"chars {chars} != report {asr.get('chars')}")
    # 抽段：最长、最短 + 均匀 8 段
    idxs = sorted(range(len(tr)), key=lambda i: tr[i]["end"] - tr[i]["start"])
    pick = idxs[:1] + idxs[-1:] + [round(i * (len(tr) - 1) / 7) for i in range(8)]
    pick = sorted(set(pick))
    samples = [(i, round(tr[i]["end"] - tr[i]["start"], 2), tr[i]["text"][:60]) for i in pick]
    rows.append({
        "id": job_id[:44], "n": len(tr), "n_expected": len(expected),
        "chars": chars, "n_empty": asr.get("n_empty"), "empty_ratio": round(asr.get("n_empty", 0) / max(asr.get("n_segments", 1), 1), 4),
        "rtf": asr.get("rtf"), "max_roundtrip_ms": round(max_rt * 1000, 6),
        "ok": not errors, "errors": errors, "samples": samples,
        "speakers": speakers,
    })
    if errors:
        all_ok = False

print(f"{'id':46s} {'segs':>5s} {'exp':>5s} {'chars':>6s} {'empty':>5s} {'ratio':>6s} {'rtf':>7s} {'rt_ms':>8s} ok")
for r in rows:
    print(f"{r['id']:46s} {r['n']:5d} {r['n_expected']:5d} {r['chars']:6d} {r['n_empty']:5d} "
          f"{r['empty_ratio']:6.3f} {r['rtf']:7.4f} {r['max_roundtrip_ms']:8.5f} {r['ok']}")
    if r["errors"]:
        print("   ERRORS:", r["errors"])
print("\n--- 抽段（10 段：最短/最长 + 均匀）---")
for r in rows[:1] + rows[-1:]:
    print(r["id"])
    for s in r["samples"]:
        print("  ", s)

# stages 表
db = sqlite3.connect(str(cfg.path("db_path")))
cur = db.execute("SELECT job_id, status, metrics_json FROM stages WHERE stage='asr'")
stage_rows = cur.fetchall()
print(f"\nstages(asr) rows={len(stage_rows)}")
for jid, status, metrics in stage_rows:
    m = json.loads(metrics or "{}")
    print("  ", jid[:40], status, "segs=", m.get("n_segments"), "chars=", m.get("chars"))
print("\nALL_OK", all_ok)
