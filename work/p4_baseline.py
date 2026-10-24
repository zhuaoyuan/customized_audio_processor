# -*- coding: utf-8 -*-
"""P4 开工基准重订：用 P3 最终边界（analyze 缓存 + edit.json）预计算段数/语音秒。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config
from app.models import AnalyzeResult, KeptSpan
from app.pipeline.asr import split_long_segment
from app.pipeline.trim import TimeMap, plan_trim

ROOT = Path(__file__).resolve().parent.parent
cfg = load_config()
out = cfg.path("out_dir")
cache = cfg.path("cache_dir")

rows = []
for job_dir in sorted(p for p in out.iterdir() if p.is_dir() and (p / "edit.json").is_file()):
    job_id = job_dir.name
    analyze_path = cache / f"{job_id}.analyze.json"
    if not analyze_path.is_file():
        print(f"MISSING analyze: {job_dir.name}")
        continue
    raw = json.loads(analyze_path.read_text(encoding="utf-8"))
    ar = AnalyzeResult(**raw["result"])
    edit = json.loads((job_dir / "edit.json").read_text(encoding="utf-8"))
    plan = plan_trim(ar, cfg, duration_s=ar.duration_s)
    tm = TimeMap([KeptSpan(*k) for k in edit["kept"]])
    n_raw = n_clean = n_drop = n_split = 0
    speech_s = 0.0
    for s, e in ar.segments:
        n_raw += 1
        for cs, ce in tm.orig_span_to_clean(s, e):
            for a, b in split_long_segment(cs, ce, 30.0):
                if b - a < 0.5:
                    n_drop += 1
                    continue
                if b - a > 30.0 + 1e-9:
                    n_split += 1
                n_clean += 1
                speech_s += b - a
    rows.append({
        "job_id": job_id, "clean_s": plan.clean_duration_s,
        "clean_min": plan.clean_duration_s / 60, "n_orig_segments": n_raw,
        "n_segments": n_clean, "n_dropped": n_drop, "n_split": n_split,
        "speech_s": speech_s,
    })

print(f"{'short':28s} {'clean_s':>9s} {'segments':>8s} {'dropped':>7s} {'split':>5s} {'speech_s':>9s}")
for r in rows:
    short = r["job_id"].split("_")[0] + "_" + r["job_id"].split("_", 1)[1][:12]
    print(f"{r['job_id'][:28]:28s} {r['clean_s']:9.2f} {r['n_segments']:8d} "
          f"{r['n_dropped']:7d} {r['n_split']:5d} {r['speech_s']:9.1f}")
print("total clean_s", round(sum(r["clean_s"] for r in rows), 2))
