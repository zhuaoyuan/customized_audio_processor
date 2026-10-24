import json, re, collections, io, sys
from pathlib import Path

out = io.StringIO()
def P(*a):
    print(*a, file=out)

p = Path(r"out/0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05/transcript.json")
segs = json.loads(p.read_text(encoding="utf-8"))
text = "".join((s.get("text") or s.get("text_raw") or "") for s in segs)
lens = sorted(len((s.get("text") or "")) for s in segs)
P("n_segments", len(segs), "chars", len(text))
P("seg chars median", lens[len(lens)//2], "p90", lens[int(len(lens)*0.9)], "min", lens[0], "max", lens[-1])
P("segments <15 chars:", sum(1 for x in lens if x < 15), "/", len(lens))
for f in ["呃", "啊", "嗯", "这个这个", "就是就是", "那个那个", "你你", "就就", "在在", "他他"]:
    P(f"{f}: {text.count(f)}")
P("top duplicated words:", collections.Counter(re.findall(r"([\u4e00-\u9fff]{2,4})\1", text)).most_common(12))
variants = {
    "亚里士多德": ["亚里士多德", "亚里斯多德", "亚里士德", "亚里斯德", "亚里士罗", "亚里斯"],
    "潜能": ["潜能", "签能", "潜证", "前能", "潜"],
    "实现": ["实现", "实线", "时现"],
    "形式": ["形式", "情式", "形商"],
    "定义": ["定义", "定影", "定意"],
    "形而上学": ["形而上学", "刑而上学", "行而上学", "提而上学"],
    "运动": ["运动", "运 动"],
    "无限": ["无限", "无线"],
}
for k, vs in variants.items():
    P(k, {v: text.count(v) for v in vs})
# context snippets around suspicious variants
for bad in ["签能", "定影", "无线", "亚里斯德", "亚里士德", "刑而上学", "情式", "世博会"]:
    idx = text.find(bad)
    if idx >= 0:
        P(f"\n[{bad}] ...{text[max(0,idx-40):idx+40]}...")
# random-ish samples: every 5 min
P("\n--- samples by time ---")
for s in segs[::120][:12]:
    P(f"{s['start']:.0f}s: {s.get('text')}")
Path("work/quality_report.txt").write_text(out.getvalue(), encoding="utf-8")
print("written work/quality_report.txt")
