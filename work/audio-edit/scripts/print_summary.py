# -*- coding: utf-8 -*-
import json

try:
    d = json.load(open("work/audio-edit/_summary_tmp.json", encoding="utf-8"))["rows"]
except UnicodeDecodeError:
    d = json.load(open("work/audio-edit/_summary_tmp.json", encoding="utf-16"))["rows"]
hdr = ("file", "dur", "cut%", "head", "tail", "int", "pres%", "srcASL", "preASL",
       "V1sp", "V1h", "V2sp", "V2h", "V1tp", "V2tp", "V2rate", "V2lim%", "wall", "clamp")
print(("{:<6}" + "{:>7}" * (len(hdr) - 1)).format(*hdr))
for r in d:
    v1, v2 = r["v1"], r["v2"]
    vals = [r["s"], "{:.0f}".format(r["dur"]), "{:.1f}".format(r["cut"] * 100),
            "{:.1f}".format(r["head"]), "{:.1f}".format(r["tail"]), "{:.1f}".format(r["int"]),
            "{:.1f}".format(r["pres"] * 100), "{:.1f}".format(r["src_asl"]),
            "{:.1f}".format(r["pre_asl"]), "{:.2f}".format(v1["spread"]), "{:.2f}".format(v1["half"]),
            "{:.2f}".format(v2["spread"]), "{:.2f}".format(v2["half"]),
            "{:.1f}".format(v1["tp"]), "{:.1f}".format(v2["tp"]), "{:.2f}".format(v2["rate"]),
            "{:.2f}".format(v2["lim"]["trigger_ratio"] * 100), "{:.0f}".format(r["wall"]),
            "{:.1f}".format(r["v2clamp"])]
    print(("{:<6}" + "{:>7}" * (len(vals) - 1)).format(*vals))
print()
for r in d:
    print(r["s"], "V1", {k: r["v1"][k] for k in ("asl", "dev", "lufs", "gmin", "gmax", "clip")},
          "V2", {k: r["v2"][k] for k in ("asl", "dev", "lufs", "gmin", "gmax", "clip")},
          "V2clip", r["v2"]["clip"], "V2lra", r["v2"]["lra"], "V1lra", r["v1"]["lra"])
print()
hs = json.load(open("work/scan/head_scan.json", encoding="utf-8"))
for k in ("f24_xingershangxue", "f00_xifang20140227"):
    if k not in hs:
        continue
    print("=== ", k, " ===")
    for c in hs[k]["head_chunks"][:26]:
        print("{:6.0f}-{:6.0f}  cps={:5.2f}  chars={:4d}".format(c["t0"], c["t1"], c["cps"], c["chars"]))
