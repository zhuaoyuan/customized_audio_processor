"""Quick API connectivity check (no secrets printed)."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

raw = (ROOT / "work" / "set_key.ps1").read_text(encoding="utf-8", errors="replace")
import re
m = re.search(r"sk-[A-Za-z0-9]+", raw)
if m and not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = m.group(0)

from app.config import load_config
from app.pipeline.correct import chat_completion

cfg = load_config(ROOT / "configs" / "default.yaml")
res = chat_completion(cfg, [{"role": "user", "content": "只回复两个字：可用"}], timeout_s=60)
print("OK model=", res["model"], "usage=", res["usage"], "text=", res["text"][:20])
