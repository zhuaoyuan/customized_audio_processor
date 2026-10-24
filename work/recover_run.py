import json, os, re, shutil, sys, time
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
job_out = ROOT / "out" / JOB
payload = json.loads((job_out / "manuscript.json").read_text(encoding="utf-8"))
card = payload["card"]
(job_out / "compose_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
backup = ROOT / "work" / "compose_proto"
backup.mkdir(parents=True, exist_ok=True)
for name in ("manuscript.md", "manuscript.json", "compose_report.json"):
    src = job_out / name
    if src.exists():
        shutil.move(str(src), str(backup / f"run1_{name}"))
print("restored card hash", (card.get("_meta") or {}).get("transcript_sha1"), flush=True)

cfg = load_config()
t = time.time()
def progress(done, total, result):
    print(f"chunk {done}/{total} {result.get('start')}-{result.get('end')} "
          f"ok={result.get('ok')} fallback={result.get('fallback')} "
          f"cached={result.get('cached')} {result.get('elapsed_s')}s", flush=True)
res = compose_job(JOB, cfg, concurrency=1, on_progress=progress)
print("DONE", round(time.time() - t, 1), json.dumps({k: v for k, v in res.items() if k != "stats"}, ensure_ascii=False), flush=True)
print(json.dumps(res.get("stats"), ensure_ascii=False, indent=2), flush=True)
