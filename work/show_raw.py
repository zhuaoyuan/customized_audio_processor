import json
segs = json.load(open(r"out/0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05/transcript.json", encoding="utf-8"))
for s in segs:
    if 395 <= s["start"] <= 480:
        t = int(s["start"])
        print(f"[{t//60:02d}:{t%60:02d}] {s['text']}")
