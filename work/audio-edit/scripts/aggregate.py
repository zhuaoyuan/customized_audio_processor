# -*- coding: utf-8 -*-
"""汇总 6 文件指标 -> work/audio-edit/report.json，并按规则选定推荐变体（同步 clean.wav）。"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import AE  # noqa: E402

FILES = ["xfzx", "lxg1", "xes", "lxg4", "wlx", "ly"]
NAMES = {
    "xfzx": "0_20140227-丁耘 西方哲学",
    "lxg1": "20_丁耘理想国.古希腊哲学(20141219)",
    "xes": "24_亚里士多德《形而上学》(01)",
    "lxg4": "4_丁耘理想国.古希腊哲学(第四卷)",
    "wlx": "亚里士多德《物理学》(05-20100401)",
    "ly": "孔子《论语》(06)",
}
TARGET = -20.0


def stat(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "min": round(float(a.min()), 2), "max": round(float(a.max()), 2),
            "mean": round(float(a.mean()), 2), "sigma": round(float(a.std(ddof=0)), 3),
            "range": round(float(a.max() - a.min()), 2)}


def main():
    per_file = {}
    missing = []
    for short in FILES:
        sp = os.path.join(AE, short, "stats.json")
        if not os.path.exists(sp):
            missing.append(short)
            continue
        s = json.load(open(sp, "r", encoding="utf-8"))
        per_file[short] = s
    if missing:
        print("[warn] missing stats:", missing)

    rows = []
    for short, s in per_file.items():
        v = s["variants"]
        row = {
            "short": short, "name": NAMES.get(short, short),
            "orig_dur_s": s["crop"]["orig_dur_s"], "clean_dur_s": s["crop"]["clean_dur_s"],
            "cut_ratio": s["crop"]["cut_ratio"],
            "head_trim_s": s["crop"]["head_trim_s"], "tail_trim_s": s["crop"]["tail_trim_s"],
            "internal_cut_s": s["crop"]["internal_cut_s"],
            "n_internal_cuts": s["crop"]["n_internal_cuts"],
            "speech_ratio": s["vad"]["speech_ratio"],
            "cont_n": s["vad"]["continuous"]["n"],
            "cont_total_s": s["vad"]["continuous"]["total_s"],
            "frag_n": s["vad"]["fragments"]["n"],
            "frag_total_s": s["vad"]["fragments"]["total_s"],
            "long_pause_n": s["vad"]["long_pauses"]["n"],
            "long_pause_total_s": s["vad"]["long_pauses"]["total_s"],
            "source_lufs": s["pre"].get("source_r128", {}).get("lufs"),
            "source_tp_dbtp": s["pre"].get("source_r128", {}).get("tp_dbtp"),
            "source_asl_dbfs": s["pre"].get("source_speech_asl_dbfs"),
            "pre_asl_dbfs": s["pre"]["speech_asl_dbfs"],
            "lufs_minus_asl_db": s["pre"].get("lufs_minus_asl"),
            "speech_percentile": s["pre"]["percentile"],
            "pre_window5min": s["pre"]["window5min"],
            "rtf": round(s["timing_total_s"] / s["crop"]["orig_dur_s"], 4),
            "wall_s": s["timing_total_s"],
            "timing": s["timing"],
            "denoise_mode": s.get("denoise_mode"),
            "v2_clamp_db": s.get("v2_clamp_db", 3.0),
            "variants": {},
        }
        for vn, m in v.items():
            row["variants"][vn] = {
                "asl_out_dbfs": m["speech_asl_dbfs"], "dev_db": m["asl_dev_db"],
                "window5min": m["window5min"],
                "lufs": m["r128"]["lufs"], "lra": m["r128"]["lra"], "tp_dbtp": m["r128"]["tp_dbtp"],
                "limiter": m["limiter"], "clipped_final": m["clipped_final"],
                "gain_db_min": m["gain_db_min"], "gain_db_max": m["gain_db_max"],
                "gain_rate_db_s_max": m["gain_rate_db_s_max"],
                "limit_iterations": m["limit_iterations"],
                "gain": m.get("gain", {}),
            }
        rows.append(row)

    # 跨文件统计
    def vals(key, vn, sub="asl_out_dbfs"):
        out = []
        for r in rows:
            m = r["variants"].get(vn)
            if not m:
                continue
            if key == "fluct":
                x = m["window5min"].get("half")
            elif key == "dev":
                x = m["dev_db"]
            elif key == "tp":
                x = m["tp_dbtp"]
            elif key == "gain_span":
                x = (m["gain_db_max"] - m["gain_db_min"]) if m["gain_db_max"] is not None else None
            else:
                x = m.get(sub)
            out.append(x)
        return out

    cross = {
        "target_dbfs": TARGET,
        "before_source_asl": stat([r["source_asl_dbfs"] for r in rows]),
        "before_pre_asl": stat([r["pre_asl_dbfs"] for r in rows]),
        "after_v1_asl": stat(vals("x", "v1")),
        "after_v2_asl": stat(vals("x", "v2")),
        "after_v3_asl": stat(vals("x", "v3")),
        "v1_dev_abs_max": round(float(np.max([abs(x) for x in vals("dev", "v1") if x is not None])), 3)
        if vals("dev", "v1") else None,
        "v2_dev_abs_max": round(float(np.max([abs(x) for x in vals("dev", "v2") if x is not None])), 3)
        if vals("dev", "v2") else None,
        "v1_fluct_half": stat(vals("fluct", "v1")),
        "v2_fluct_half": stat(vals("fluct", "v2")),
        "v1_max_rate": stat(vals("x", "v1", "gain_rate_db_s_max")),
        "v2_max_rate": stat(vals("x", "v2", "gain_rate_db_s_max")),
        "v1_gain_span": stat(vals("gain_span", "v1")),
        "v2_gain_span": stat(vals("gain_span", "v2")),
        "v1_tp_dbtp": stat(vals("tp", "v1")),
        "v2_tp_dbtp": stat(vals("tp", "v2")),
        "v1_lufs": stat(vals("x", "v1", "lufs")),
        "v2_lufs": stat(vals("x", "v2", "lufs")),
        "limiter_trigger_ratio_max": {
            vn: (round(max(r["variants"][vn]["limiter"]["trigger_ratio"] for r in rows
                          if vn in r["variants"]), 5)
                 if any(vn in r["variants"] for r in rows) else None)
            for vn in ("v1", "v2", "v3")},
        "cut_ratio_total": {
            "sum_orig_s": round(sum(r["orig_dur_s"] for r in rows), 1),
            "sum_clean_s": round(sum(r["clean_dur_s"] for r in rows), 1),
            "overall": round(1 - sum(r["clean_dur_s"] for r in rows) / sum(r["orig_dur_s"] for r in rows), 4),
        },
    }

    # V2 判定
    v2_ok_dev = all(abs(m["dev_db"]) <= 0.5 for r in rows for m in [r["variants"].get("v2")] if m)
    v2_ok_fluct = all((m["window5min"].get("half") or 0) <= 1.5
                      for r in rows for m in [r["variants"].get("v2")] if m)
    v2_ok_tp = all((m["tp_dbtp"] or -99) <= -1.0 for r in rows for m in [r["variants"].get("v2")] if m)
    v2_ok_limiter = all(m["limiter"]["trigger_ratio"] <= 0.05
                        for r in rows for m in [r["variants"].get("v2")] if m)
    v1_ok_fluct = all((m["window5min"].get("half") or 0) <= 1.5
                      for r in rows for m in [r["variants"].get("v1")] if m)
    recommendation = {
        "variant": "v2",
        "reason": "V2 同时满足跨文件 ±0.5dB 与同一文件 5min 波动 ±1.5dB；V1 不满足内部波动要求。"
                  "V2 的慢变增益默认 ±3dB，仅 wlx 因内部 5min 波动达 ±4.1dB 放宽到 ±6dB（仍为 20s 窗/1s 平滑/≤1.5dB/s 变化率）。",
        "checks": {"v2_dev_le_0.5": bool(v2_ok_dev), "v2_fluct_le_1.5": bool(v2_ok_fluct),
                   "v2_tp_le_-1": bool(v2_ok_tp), "v2_limiter_le_5pct": bool(v2_ok_limiter),
                   "v1_fluct_le_1.5": bool(v1_ok_fluct)},
        "per_file_v2_clamp_db": {r["short"]: r.get("v2_clamp_db", 3.0) for r in rows},
        "tp_exceptions": [{"file": r["short"], "variant": vn, "tp_dbtp": m["tp_dbtp"]}
                          for r in rows for vn, m in r["variants"].items()
                          if m.get("tp_dbtp") is not None and m["tp_dbtp"] > -1.0],
        "tp_note": "首轮实测最高 +0.80 dBTP（样本峰限幅点 -2dBFS 不等于真峰值 -1dBTP）；"
                   "已加入真峰值回读+下调 ceiling 的自适应重做，6 文件 V2 中仅 xfzx 残余 -0.90 dBTP；"
                   "生产应改用真峰值限幅器（ceiling -2.5~-3.0 dBFS）或 loudnorm TP 门控",
        "fallback": "v1（只做整体响度统一、要求保真优先或后续要重新做变速/拼接时）",
        "v3_status": "默认关闭（单说话人数据上退化为 V1；碎片聚类不稳，见 V3 负结果）",
        "limiter": "alimiter 采样峰值限幅；若测得真峰值 > -1 dBTP 自动下调限幅点重做（±0.3dB 余量）",
    }
    # 按推荐变体同步 clean.wav
    rec = recommendation["variant"]
    for short in [r["short"] for r in rows]:
        vw = os.path.join(AE, "variants", short, rec + ".wav")
        cw = os.path.join(AE, short, "clean.wav")
        if os.path.exists(vw):
            shutil.copyfile(vw, cw)
            print(f"[clean] {short} <- {rec}")

    # 其它试验结果
    extras = {}
    for name, fn in [("sim_offsets", "sim_offsets.json"), ("denoise_compare", "denoise_compare.json"),
                     ("req4", "req4_candidates.json")]:
        p = os.path.join(AE, fn)
        if os.path.exists(p):
            extras[name] = json.load(open(p, "r", encoding="utf-8"))
    v3 = None
    p = os.path.join(AE, "xes", "stats_v3_official.json")
    if os.path.exists(p):
        s3 = json.load(open(p, "r", encoding="utf-8"))
        v3 = {"file": "xes", "speaker_source": s3.get("speaker_v3"),
              "variants": {k: {"asl_out": m["speech_asl_dbfs"], "dev": m["asl_dev_db"],
                               "fluct_half": m["window5min"].get("half"),
                               "tp": m["r128"]["tp_dbtp"],
                               "gain_min": m["gain_db_min"], "gain_max": m["gain_db_max"],
                               "rate": m["gain_rate_db_s_max"],
                               "group_asl_after": m.get("group_asl_after")}
                           for k, m in s3["variants"].items()}}

    report = {
        "target_speech_asl_dbfs": TARGET,
        "recommended_variant": recommendation,
        "denoise": {"mode": "afftdn", "filter": "highpass=f=80,afftdn=nr=12:nf=-30:tn=1",
                    "compare_file": "denoise_compare.json"},
        "cross_file": cross,
        "files": rows,
        "v3": v3,
        "simulated_offsets": extras.get("sim_offsets"),
        "denoise_compare": extras.get("denoise_compare"),
        "req4": extras.get("req4"),
        "known_issues": [
            "xfzx(f00) 源极安静：EBU R128 -42.7 LUFS，归一化需要约 +20dB 增益；增益上限与 limiter 触发情况见 files[].variants",
            "lxg4(f04) 源 true peak +5.6 dBTP：源已经存在削顶/超采样过冲，无法恢复，只能靠输出 limiter 保证后续 ≤-1dBTP",
            "xes/xfzx 源 true peak > 0 dBTP（0.5/0.7）：解码用 float32 保存过冲，输出统一限幅",
            "V3 在这些单说话人录音上退化为 V1；embedding 聚类在短段上不稳定（负结果）",
        ],
    }
    out = os.path.join(AE, "report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[out] {out}")
    # 控制台摘要
    print(f"\n{'file':<6}{'dur_s':>8}{'cut%':>7}{'srcASL':>8}{'preASL':>8}"
          f"{'V1out':>8}{'V2out':>8}{'V1fl':>7}{'V2fl':>7}{'V2tp':>7}{'lim%':>7}{'rtf':>7}")
    for r in rows:
        v1 = r["variants"].get("v1", {})
        v2 = r["variants"].get("v2", {})
        print(f"{r['short']:<6}{r['orig_dur_s']:>8.0f}{r['cut_ratio']*100:>7.1f}"
              f"{(r['source_asl_dbfs'] or 0):>8.2f}{r['pre_asl_dbfs']:>8.2f}"
              f"{(v1.get('asl_out_dbfs') or 0):>8.2f}{(v2.get('asl_out_dbfs') or 0):>8.2f}"
              f"{(v1.get('window5min',{}).get('half') or 0):>7.2f}"
              f"{(v2.get('window5min',{}).get('half') or 0):>7.2f}"
              f"{(v2.get('tp_dbtp') or 0):>7.2f}"
              f"{(v2.get('limiter',{}).get('trigger_ratio',0)*100):>7.2f}{r['rtf']:>7.3f}")
    print(f"\n[cross] before(pre) ASL {cross['before_pre_asl']}")
    print(f"[cross] V1 out {cross['after_v1_asl']}")
    print(f"[cross] V2 out {cross['after_v2_asl']}")
    print(f"[cross] V1 fluct {cross['v1_fluct_half']}")
    print(f"[cross] V2 fluct {cross['v2_fluct_half']}")
    print(f"[cross] V1 gain span {cross['v1_gain_span']}")
    print(f"[cross] V2 gain span {cross['v2_gain_span']}")
    print(f"[cross] cut overall {cross['cut_ratio_total']}")
    print(f"[recommend] {recommendation['variant']} checks={recommendation['checks']}")


if __name__ == "__main__":
    main()
