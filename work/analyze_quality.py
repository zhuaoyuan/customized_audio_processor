import json, re, collections
from pathlib import Path

p = Path(r"out/0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05/transcript.json")
data = json.loads(p.read_text(encoding="utf-8"))
print(type(data), list(data)[:10] if isinstance(data, dict) else len(data))
segs = data["segments"] if isinstance(data, dict) and "segments" in data else data
print("n_segments", len(segs), "keys", list(segs[0]))
print(json.dumps(segs[0], ensure_ascii=False)[:500])
text = "".join((s.get("text") or s.get("text_raw") or "") for s in segs)
print("chars", len(text))
lens = [len((s.get("text") or "")) for s in segs]
lens.sort()
print("seg chars: median", lens[len(lens)//2], "p90", lens[int(len(lens)*0.9)], "<15:", sum(1 for x in lens if x < 15), "max", lens[-1])
fillers = ["呃", "啊", "嗯", "这个这个", "就是就是", "那个那个", "你你", "就就", "在在", "他他"]
for f in fillers:
    print(f, text.count(f))
# repeated chars generic
rep = collections.Counter(re.findall(r"(.)\1", text))
print("top repeated pairs", rep.most_common(15))
# adjacent duplicate words
w = collections.Counter(re.findall(r"([\u4e00-\u9fff]{2,4})\1", text))
print("top duplicated words", w.most_common(15))
# flags
flags = collections.Counter()
for s in segs:
    for f in (s.get("flags") or []):
        flags[f] += 1
print("flags", flags)
