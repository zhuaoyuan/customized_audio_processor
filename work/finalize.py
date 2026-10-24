import json, re
from pathlib import Path

OUT = Path(r"work/compose_proto")
JOB = "0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05"
clean = (OUT / "manuscript_clean.md").read_text(encoding="utf-8").strip()
paras = [p.strip() for p in clean.split("\n\n") if p.strip()]
card = json.loads((OUT / "card.json").read_text(encoding="utf-8"))
src = json.loads((Path("out") / JOB / "transcript.json").read_text(encoding="utf-8"))
source_text = "".join((s.get("text") or "") for s in src)
mtext = re.sub(r"\s+", "", re.sub(r"\[?\d{1,3}:\d{2}\]?", "", clean))

header = f"""# 亚里士多德《物理学》第三卷（第 5 讲）· 全文整理稿

> 来源：`out/{JOB}/transcript.md`（原音轨约 1:43:20，1,332 段，31,151 字）
> 方法：先通读全篇生成「课程档案」（结构 + 45 条术语），再按约 1,800 字段分块、带前文衔接整理；`[mm:ss]` 为该段起始的原音轨时间。
> 规模：{len(paras)} 段，约 {len(mtext)/10000:.2f} 万字（原文的 {len(mtext)/len(source_text)*100:.0f}%）。
> 说明：已清理语气词、口吃重复与明显同音误写；未逐字校对，引文、页码与专名请对照原音复核。

"""
final = header + clean + "\n"
Path("out")/JOB/"manuscript.md"
(Path("out") / JOB / "manuscript.md").write_text(final, encoding="utf-8")
print("written", Path("out") / JOB / "manuscript.md", len(final))

# 覆盖率：大纲时间点附近 4 分钟内是否有段落
stamps = [int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in re.findall(r"^\[(\d{1,3}:\d{2})\]", clean, flags=re.M)]
miss = []
for o in card.get("outline", []):
    t = int(o["time"].split(":")[0]) * 60 + int(o["time"].split(":")[1])
    if not any(abs(s - t) <= 240 for s in stamps):
        miss.append(o)
print("outline items", len(card.get("outline", [])), "uncovered", len(miss), miss)

# 术语残留
resid = [(g["correct"], v, mtext.count(v)) for g in card.get("glossary", []) for v in g.get("variants", []) if v and v in mtext]
print("residual variants (incl. substrings)", len(resid), resid[:10])
print("paragraphs", len(paras), "chars", len(mtext), "ratio", round(len(mtext)/len(source_text), 3))

# 生成 before/after 对照样例（前三处有代表性的）
before = []
for s in src[:40]:
    before.append(f"{int(s['start'])//60:02d}:{int(s['start'])%60:02d} {s['text']}")
Path(OUT / "before_sample.txt").write_text("\n".join(before), encoding="utf-8")
