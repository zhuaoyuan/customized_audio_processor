import json
import numpy as np

VD = r"C:\projects\local\audio_processer\work\timeline\24_形而上学\vad.json"
AD = r"C:\projects\local\audio_processer\work\timeline\24_形而上学\asr.json"
d = json.load(open(AD, encoding="utf-8"))
dur = d["dur"]
segs = d["segments"]

v = json.load(open(VD, encoding="utf-8"))
print("vad keys:", list(v.keys())[:8])
vsegs = v.get("segments") or v.get("speech_segments") or []
print("vad segs:", len(vsegs), vsegs[:2] if vsegs else None)

W = 120.0
n = int(dur // W) + 1
speech = np.zeros(n)
chars = np.zeros(n)
for s in vsegs:
    a, b = s["start"], s["end"]
    for k in range(int(a // W), min(int(b // W) + 1, n)):
        lo, hi = k * W, min((k + 1) * W, dur)
        speech[k] += max(0, min(b, hi) - max(a, lo))
for s in segs:
    a, b = s["start"], s["end"]
    txt = s.get("text") or ""
    c = len([x for x in txt if x not in "。，？！、…,.?!;；:： "])
    k = int(a // W)
    if k < n:
        chars[k] += c

print(f"{'win':>6}{'t0':>8}{'speech%':>9}{'chars':>8}{'chars/s':>9}")
for k in range(min(n, 12)):
    t0 = k * W
    print(f"{k:>6}{t0:>8.0f}{(speech[k]/W*100 if speech[k]<=W else 100):>8.0f}%{chars[k]:>8.0f}{chars[k]/W:>9.2f}")
print("... 尾部 ...")
for k in range(max(0, n - 6), n):
    t0 = k * W
    print(f"{k:>6}{t0:>8.0f}{(speech[k]/W*100 if speech[k]<=W else 100):>8.0f}%{chars[k]:>8.0f}{chars[k]/W:>9.2f}")
