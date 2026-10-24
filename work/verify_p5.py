# -*- coding: utf-8 -*-
"""P5 六文件验收：五件套、corrections/report 契约、md/srt 一致性、ffprobe 识别。"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config

cfg = load_config()
out_root = cfg.path("out_dir")
REQUIRED = ["clean.m4a", "transcript.json", "transcript.md", "transcript.srt",
            "transcript.txt", "corrections.json", "report.json"]


def parse_mmss(text):
    parts = [int(x) for x in text.split(":")]
    return parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]


def parse_srt_time(text):
    m = re.match(r"(\d+):(\d+):(\d+),(\d{3})$", text.strip())
    if not m:
        raise ValueError(text)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000.0


all_ok = True
rows = []
for d in sorted(p for p in out_root.iterdir() if p.is_dir() and (p / "transcript.json").is_file()):
    job_id = d.name
    errors = []
    missing = [name for name in REQUIRED if not (d / name).is_file()]
    if missing:
        errors.append(f"缺文件 {missing}")
    report = json.loads((d / "report.json").read_text(encoding="utf-8"))
    tr = json.loads((d / "transcript.json").read_text(encoding="utf-8"))
    corr = json.loads((d / "corrections.json").read_text(encoding="utf-8"))
    md_text = (d / "transcript.md").read_text(encoding="utf-8")
    srt_text = (d / "transcript.srt").read_text(encoding="utf-8")
    txt_text = (d / "transcript.txt").read_text(encoding="utf-8")
    for name, text in (("md", md_text), ("srt", srt_text), ("txt", txt_text)):
        if text.startswith("\ufeff"):
            errors.append(f"{name} 有 BOM")
    # corrections 契约
    if not (isinstance(corr, dict) and corr.get("total") == len(corr.get("items") or [])
            and isinstance(corr.get("rejected_chunks"), list)):
        errors.append("corrections.json 契约不符")
    # report correct / output.formats / timing
    correct = report.get("correct") or {}
    for key in ("enabled", "chunks", "edits", "rejected", "cost_cny", "elapsed_s"):
        if key not in correct:
            errors.append(f"report.correct 缺 {key}")
    output = report.get("output") or {}
    formats = output.get("formats") or []
    if not all(f in formats for f in ("md", "srt", "txt")):
        errors.append(f"report.output.formats 缺项 {formats}")
    if "export_text_s" not in (report.get("timing") or {}):
        errors.append("report.timing.export_text_s 缺失")
    # md 行数 / srt 块数 / transcript 条数
    md_lines = [ln for ln in md_text.splitlines() if ln.startswith("[")]
    srt_blocks = [b for b in srt_text.strip().split("\n\n") if b.strip()]
    txt_lines = [ln for ln in txt_text.splitlines() if ln.startswith("[")]
    if not (len(tr) == len(md_lines) == len(srt_blocks) == len(txt_lines)):
        errors.append(f"条数不一致 tr={len(tr)} md={len(md_lines)} srt={len(srt_blocks)} txt={len(txt_lines)}")
    # md clean 秒 == floor(srt 起点秒)
    md_clean = []
    for line in md_lines:
        m = re.match(r"^\[([0-9:]+) \(orig ([0-9:]+)\)\]", line)
        if not m:
            errors.append(f"md 行格式异常: {line[:60]}")
            break
        md_clean.append(parse_mmss(m.group(1)))
    srt_starts = []
    prev_start = -1.0
    for i, block in enumerate(srt_blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0].strip() != str(i):
            errors.append(f"srt 块 {i} 格式/序号异常")
            break
        a, b = lines[1].split("-->")
        sa, sb = parse_srt_time(a), parse_srt_time(b)
        if sb <= sa:
            errors.append(f"srt 块 {i} end<=start")
        if sa < prev_start:
            errors.append(f"srt 块 {i} 非单调")
        prev_start = sa
        srt_starts.append(sa)
        if not lines[2].startswith("主讲："):
            errors.append(f"srt 块 {i} 缺主讲前缀")
            break
    if md_clean and srt_starts and md_clean != [int(x) for x in srt_starts]:
        errors.append("md 时间与 srt 起点不一致")
    # ffprobe 识别 srt
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=format_name",
             "-of", "default=nw=1", str(d / "transcript.srt")],
            capture_output=True, timeout=30,
        )
        out = proc.stdout.decode("utf-8", "replace")
        if "format_name=srt" not in out:
            errors.append(f"ffprobe 未识别 srt: {out.strip()[:80]}")
    except Exception as exc:  # pragma: no cover
        errors.append(f"ffprobe 失败: {exc}")
    rows.append({
        "id": job_id[:46], "segs": len(tr), "md": len(md_lines), "srt": len(srt_blocks),
        "txt": len(txt_lines), "corrections": corr.get("total"),
        "enabled": correct.get("enabled"), "ok": not errors, "errors": errors,
    })
    if errors:
        all_ok = False

print(f"{'id':48s} {'segs':>5s} {'md':>5s} {'srt':>5s} {'txt':>5s} {'corr':>5s} {'enabled':>8s} ok")
for r in rows:
    print(f"{r['id']:48s} {r['segs']:5d} {r['md']:5d} {r['srt']:5d} {r['txt']:5d} "
          f"{r['corrections']:5d} {str(r['enabled']):>8s} {r['ok']}")
    for e in r["errors"]:
        print("   ERROR:", e)
print("\nALL_OK", all_ok)
