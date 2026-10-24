import json, os, re, sys, time
from pathlib import Path

ROOT = Path(r"C:\projects\local\audio_processer")
sys.path.insert(0, str(ROOT))
raw = (ROOT / "work" / "set_key.ps1").read_text(encoding="utf-8", errors="replace")
m = re.search(r"sk-[A-Za-z0-9]+", raw)
if m and not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = m.group(0)

from app.config import load_config
from app.pipeline.compose import load_segments, analyze_lecture

JOB = "0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05"
job_out = ROOT / "out" / JOB
cfg = load_config()
segs = load_segments(job_out / "transcript.json")
print("segments", len(segs), "chars", sum(len(s.text) for s in segs), flush=True)
t = time.time()
card = analyze_lecture(segs, cfg, job_id=JOB, source=str(job_out / "transcript.json"))
print("analysis elapsed", round(time.time() - t, 1), flush=True)
(job_out / "compose_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
print("card saved", len(card.get("glossary") or []), len(card.get("outline") or []), flush=True)
