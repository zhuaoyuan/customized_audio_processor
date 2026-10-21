import json
p = r"C:\projects\local\audio_processer\work\diar\asr_compare.json"
d = json.load(open(p, encoding="utf-8"))
if isinstance(d, dict):
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)):
            print(f"{k}: {v}")
        elif isinstance(v, list):
            print(f"{k}: list[{len(v)}]")
            if v and isinstance(v[0], dict):
                print("   first:", json.dumps(v[0], ensure_ascii=False)[:300])
        elif isinstance(v, dict):
            print(f"{k}: dict keys={list(v.keys())[:10]}")
            print("   ", json.dumps(v, ensure_ascii=False)[:500])
