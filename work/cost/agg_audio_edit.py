import json, glob, os
rows = []
for p in sorted(glob.glob(r"C:\projects\local\audio_processer\work\audio-edit\*\stats.json")):
    d = json.load(open(p, encoding="utf-8"))
    short = d.get("short")
    if short == "xes" and "pilot" in os.path.basename(os.path.dirname(p)):
        continue
    if "pilot" in p:
        continue
    pre = d.get("pre", {})
    var = d.get("variants", {})
    v1 = var.get("v1", {})
    v2 = var.get("v2", {})
    crop = d.get("crop", {})
    rows.append(dict(
        short=short,
        dur_min=round(d.get("pre", {}).get("source_speech_asl_dbfs", 0) and 0 or 0, 1),
        asl_in=pre.get("speech_asl_dbfs"),
        asl_out=v1.get("speech_asl_dbfs"),
        gain=d.get("base_gain_db"),
        w5_in=(pre.get("window5min") or {}).get("spread"),
        w5_out=(v1.get("window5min") or {}).get("spread"),
        lufs_in=(pre.get("source_r128") or {}).get("lufs"),
        lufs_out=(v1.get("r128") or {}).get("lufs"),
        tp_out=(v1.get("r128") or {}).get("tp_dbtp"),
        cut_ratio=crop.get("cut_ratio"),
        head=crop.get("head_trim_s"), tail=crop.get("tail_trim_s"), internal=crop.get("internal_cut_s"),
        speech_pres=crop.get("speech_preserved_ratio"),
        v2_asl=v2.get("speech_asl_dbfs"),
        v2_gain_min=(v2.get("gain") or {}).get("gain_min"),
        v2_gain_max=(v2.get("gain") or {}).get("gain_max"),
        v2_rate=v2.get("gain_rate_db_s_max"),
        t=d.get("timing_total_s"),
        orig=d.get("orig_dur_s") or (d.get("crop") or {}).get("orig_dur_s"),
    ))
hdr = f"{'short':<6}{'asl_in':>8}{'asl_out':>8}{'gain':>7}{'w5in':>7}{'w5out':>7}{'lufs_out':>9}{'tp':>6}{'cut%':>7}{'head':>7}{'tail':>7}{'int':>8}{'sp_pres':>8}"
print(hdr)
for r in rows:
    def f(k, w=6, d=1):
        v = r.get(k)
        return f"{v:{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
    print(f"{r['short']:<6}{f('asl_in',8)}{f('asl_out',8)}{f('gain',7)}{f('w5_in',7)}{f('w5_out',7)}{f('lufs_out',9)}{f('tp',6)}{f('cut_ratio',7,3)}{f('head',7)}{f('tail',7)}{f('internal',8)}{f('speech_pres',8,3)}")
out = [r for r in rows if r["asl_out"] is not None]
if out:
    vals = [r["asl_out"] for r in out]
    print(f"\n跨文件 out ASL: min={min(vals):.2f} max={max(vals):.2f} span={max(vals)-min(vals):.2f} dB")
    tp = [r["tp_out"] for r in out if r["tp_out"] is not None]
    if tp: print(f"out true peak max: {max(tp):.2f} dBTP")
