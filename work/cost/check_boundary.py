import json

d = json.load(open(r"C:\projects\local\audio_processer\work\timeline\24_形而上学\asr.json", encoding="utf-8"))
segs = d["segments"]
print("total segs:", len(segs), "dur:", d["dur"])

def is_punct(t):
    t = (t or "").strip()
    return all(c in "。，？！、…,.?!;；:：" for c in t) or len(t) == 0

rows = []
for s in segs:
    dur = max(s["end"] - s["start"], 1e-6)
    txt = s.get("text") or ""
    chars = len([c for c in txt if c not in "。，？！、…,.?!;；:： "])
    cps = chars / dur
    rows.append((s["start"], s["end"], dur, chars, cps, txt))

print("\n前 25 段（看头部音乐区的判定）:")
print(f"{'start':>8}{'end':>8}{'dur':>7}{'chars':>7}{'cps':>6}  text")
for r in rows[:25]:
    print(f"{r[0]:>8.1f}{r[1]:>8.1f}{r[2]:>7.1f}{r[3]:>7}{r[4]:>6.2f}  {r[5][:40]}")

# 讲课段判定：cps>=2 且非纯标点，连续 2 段
flag = [ (cps >= 2.0 and not is_punct(txt)) for _,_,_,_,cps,txt in rows ]
first = None
for i in range(len(flag)-1):
    if flag[i] and flag[i+1]:
        first = rows[i][0]; break
last = None
for i in range(len(flag)-1, 0, -1):
    if flag[i] and flag[i-1]:
        last = rows[i][1]; break
print(f"\n[ASR 边界] 首个连续 2 段讲课起点 = {first} s; 末个 = {last} s")
print(f"[VAD 边界] v1 记录的真实讲课起点大约 480 s、终点约 4520 s")

# 尾部 25 段
print("\n末 25 段:")
for r in rows[-25:]:
    print(f"{r[0]:>8.1f}{r[1]:>8.1f}{r[2]:>7.1f}{r[3]:>7}{r[4]:>6.2f}  {r[5][:40]}")
