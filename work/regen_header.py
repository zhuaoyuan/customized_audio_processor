import json, os, re, sys, time
from pathlib import Path

ROOT = Path(r"C:\projects\local\audio_processer")
sys.path.insert(0, str(ROOT))
raw = (ROOT / "work" / "set_key.ps1").read_text(encoding="utf-8", errors="replace")
m = re.search(r"sk-[A-Za-z0-9]+", raw)
if m and not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = m.group(0)

from app.config import load_config
from app.pipeline.compose import compose_job

JOB = "0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05"
res = compose_job(JOB, load_config(), concurrency=1, force=True)
print(json.dumps({k: v for k, v in res.items() if k != "stats"}, ensure_ascii=False))
print(json.dumps(res["stats"], ensure_ascii=False, indent=2))
