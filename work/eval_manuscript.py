import json, re
from pathlib import Path

OUT = Path(r"work/compose_proto")
chunks = sorted(OUT.glob("chunks/chunk_*_attempt1.txt"))
parts = []
for p in chunks:
    m = re.match(r"chunk_(\d+)_", p.name)
    if not m:
        continue
    parts.append((int(m.group(1)), p.read_text(encoding="utf-8").strip()))
parts.sort()
manuscript = "\n\n".join(t for _, t in parts) + "\n"
(OUT / "manuscript.md").write_text(manuscript, encoding="utf-8")

src = json.loads(Path(r"out/0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05/transcript.json").read_text(encoding="utf-8"))
source_text = "".join((s.get("text") or "") for s in src)
norm = lambda t: re.sub(r"\s+", "", re.sub(r"\[?\d{1,3}:\d{2}\]?", "", t))
mtext = norm(manuscript)
print("source chars", len(source_text), "manuscript chars", len(mtext), "ratio", round(len(mtext) / len(source_text), 3))
print("paragraphs", manuscript.count("\n\n") + 1, "timestamps", len(re.findall(r"^\s*\[\d{1,3}:\d{2}\]", manuscript, flags=re.M)))
for f in ["呃", "啊", "嗯", "这个这个", "就是就是", "签能", "潜证", "无线", "定影", "行而上学", "刑商", "以而上学"]:
    print(f"{f}: source={source_text.count(f)} manuscript={mtext.count(f)}")

card = json.loads((OUT / "card.json").read_text(encoding="utf-8"))
missing = [g["correct"] for g in card.get("glossary", []) if g["correct"] not in mtext]
print("glossary terms", len(card.get("glossary", [])), "missing in manuscript", len(missing), missing[:10])
residual = [(g["correct"], v, mtext.count(v)) for g in card.get("glossary", []) for v in g.get("variants", []) if v and v in mtext]
print("residual wrong variants", len(residual), residual[:15])

# cost
r = json.loads((OUT / "report_19chunks.json").read_text(encoding="utf-8"))
u = r["usage"]
cost = ((u["prompt_tokens"] - u["cache_hit_tokens"]) * 1.0 + u["cache_hit_tokens"] * 0.02 + u["completion_tokens"] * 4.0) / 1e6
print("usage", u, "cost_cny", round(cost, 4), "elapsed_min", round(r["elapsed_s"] / 60, 1))
