import json, re, sys, time
from pathlib import Path

ROOT = Path(r"C:\projects\local\audio_processer")
sys.path.insert(0, str(ROOT))
raw = (ROOT / "work" / "set_key.ps1").read_text(encoding="utf-8", errors="replace")
import os
m = re.search(r"sk-[A-Za-z0-9]+", raw)
if m and not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = m.group(0)

from app.config import load_config
from app.pipeline.compose import compose_job

src = json.loads((ROOT / "out" / "0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05" / "transcript.json").read_text(encoding="utf-8"))
out = ROOT / "work" / "compose_real_smoke"
job = out / "smoke_job"
job.mkdir(parents=True, exist_ok=True)
(job / "transcript.json").write_text(json.dumps(src[:12], ensure_ascii=False), encoding="utf-8")
cfg = load_config(overrides={"paths.out_dir": str(out)})
t = time.time()
res = compose_job("smoke_job", cfg, chunk_chars=800, concurrency=1)
print("elapsed", round(time.time() - t, 1))
print(json.dumps({k: v for k, v in res.items() if k != "stats"}, ensure_ascii=False, indent=2))
print(json.dumps(res.get("stats"), ensure_ascii=False, indent=2))
print((job / "manuscript.md").read_text(encoding="utf-8")[:2000])
