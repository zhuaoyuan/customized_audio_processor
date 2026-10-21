import json, glob, os
rows = []
for p in sorted(glob.glob(r"C:\projects\local\audio_processer\work\audio-edit\*\stats.json")):
    if "pilot" in p:
        continue
    d = json.load(open(p, encoding="utf-8"))
    pre = d.get("pre", {})
    var = d.get("variants", {})
    row = {"short": d.get("short"), "chosen": d.get("clean_variant"),
           "asl_in": pre.get("speech_asl_dbfs"),
           "w5_in": (pre.get("window5min") or {}).get("spread")}
    for k in ("v1", "v2", "v3"):
        v = var.get(k) or {}
        row[f"{k}_asl"] = v.get("speech_asl_dbfs")
        row[f"{k}_w5"] = (v.get("window5min") or {}).get("spread")
        row[f"{k}_gmin"] = (v.get("gain") or {}).get("gain_min")
        row[f"{k}_gmax"] = (v.get("gain") or {}).get("gain_max")
        row[f"{k}_rate"] = v.get("gain_rate_db_s_max")
        row[f"{k}_lra"] = (v.get("r128") or {}).get("lra")
        row[f"{k}_tp"] = (v.get("r128") or {}).get("tp_dbtp")
    row["cut%"] = (d.get("crop") or {}).get("cut_ratio")
    rows.append(row)

def g(r, k, w=6, dd=2):
    v = r.get(k)
    return f"{v:{w}.{dd}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

print("file  chosen |  w5_in   v1_w5   v2_w5  |  v1_lra  v2_lra | v2_rate | v1_tp  v2_tp")
for r in rows:
    print(f"{r['short']:<6}{str(r['chosen']):>7} | {g(r,'w5_in',7)}{g(r,'v1_w5',8)}{g(r,'v2_w5',8)} | {g(r,'v1_lra',8,1)}{g(r,'v2_lra',8,1)} | {g(r,'v2_rate',7,2)} | {g(r,'v1_tp',6,2)}{g(r,'v2_tp',7,2)}")
print()
for r in rows:
    print(f"{r['short']:<6} asl_in={g(r,'asl_in',7)} v1={g(r,'v1_asl',7)} v2={g(r,'v2_asl',7)} v3={g(r,'v3_asl',7)}  v2_gain=[{g(r,'v2_gmin',6)},{g(r,'v2_gmax',6)}]  cut={g(r,'cut%',6,3)}")
