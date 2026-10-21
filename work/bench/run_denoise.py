"""ffmpeg 降噪基线：对 s1_mid.wav 跑 afftdn / anlmdn / arnndn，测耗时 + 处理前后指标。"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audiometrics import metrics, db, read_wav, frame_rms

BENCH = os.path.dirname(os.path.abspath(__file__))
FFMPEG = "ffmpeg"


def run_ffmpeg(inp, out, af, sr=16000, extra=None, cwd=None):
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", inp]
    if af:
        cmd += ["-af", af]
    cmd += ["-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", out]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
    wall = time.perf_counter() - t0
    return cmd, wall, p


def summarize(tag, path, dur, wall=None, cmd=None, err=None):
    d = metrics(path)
    row = {"tag": tag, "file": os.path.basename(path), "dur": round(d["dur"], 2),
           "rms_dbfs": round(d["rms_dbfs"], 2), "peak_dbfs": round(d["peak_dbfs"], 2),
           "p1": round(d["p1"], 2), "p10": round(d["p10"], 2), "p50": round(d["p50"], 2),
           "p90": round(d["p90"], 2), "p99": round(d["p99"], 2),
           "snr_proxy": round(d["snr_proxy"], 2),
           "noise_floor_dbfs": round(d["noise_floor_dbfs"], 2),
           "speech_level_dbfs": round(d["speech_level_dbfs"], 2),
           "snr_est_db": round(d["snr_est_db"], 2),
           "above_p10_10db_pct": round(d["above_p10_10db_pct"], 1),
           "below_p10_3db_pct": round(d["below_p10_3db_pct"], 1)}
    if wall is not None:
        row["wall_s"] = round(wall, 2)
        row["rtf"] = round(wall / dur, 3)
    if err:
        row["stderr_tail"] = err[-600:]
    if cmd:
        row["cmd"] = " ".join(cmd[cmd.index("-af") + 1:cmd.index("-ar")]) if "-af" in cmd else ""
    print(json.dumps(row, ensure_ascii=False))
    return row


def main():
    inp = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else r"work\samples\s1_mid.wav")
    rr_dir = os.path.join(BENCH, "models", "rnnoise")
    rr = os.path.join(rr_dir, "sh.rnnn")
    outdir = os.path.join(BENCH, "denoise")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(inp))[0]
    rows = [summarize("baseline(原始)", inp, 0)]
    dur = rows[0]["dur"]

    # af, cwd（arnndn 的模型路径含 "C:" 冒号会破坏 ffmpeg filter 解析，用相对路径 + cwd 规避）
    variants = [
        ("afftdn_nr12", "afftdn=nr=12:nf=-30:tn=1", None),
        ("afftdn_nr20", "afftdn=nr=20:nf=-25:tn=1", None),
        ("anlmdn_s1e-3", "anlmdn=s=0.001:p=0.002:r=0.006", None),
        ("anlmdn_s1e-2", "anlmdn=s=0.01:p=0.002:r=0.006", None),
        ("anlmdn_s5e-2", "anlmdn=s=0.05:p=0.002:r=0.006", None),
        ("afftdn+anlmdn", "afftdn=nr=12:nf=-30:tn=1,anlmdn=s=0.001:p=0.002:r=0.006", None),
        ("highpass80+afftdn", "highpass=f=80,afftdn=nr=12:nf=-30:tn=1", None),
    ]
    if os.path.exists(rr):
        variants.append(("arnndn_sh", "arnndn=m=sh.rnnn", rr_dir))
        variants.append(("highpass80+arnndn", "highpass=f=80,arnndn=m=sh.rnnn", rr_dir))
        variants.append(("afftdn+arnndn", "afftdn=nr=12:nf=-30:tn=1,arnndn=m=sh.rnnn", rr_dir))
    else:
        print(f"[warn] 未找到 rnnoise 模型 {rr}，跳过 arnndn", file=sys.stderr)

    for tag, af, cwd in variants:
        outp = os.path.join(outdir, f"{base}_{tag}.wav")
        cmd, wall, p = run_ffmpeg(inp, outp, af, cwd=cwd)
        if p.returncode != 0:
            row = {"tag": tag, "file": None, "error": p.returncode,
                   "af": af, "stderr_tail": (p.stderr or "")[-800:]}
            print(json.dumps(row, ensure_ascii=False))
            continue
        rows.append(summarize(tag, outp, dur, wall, cmd))

    with open(os.path.join(BENCH, f"denoise_{base}.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[out] {os.path.join(BENCH, f'denoise_{base}.json')}")


if __name__ == "__main__":
    main()
