# -*- coding: utf-8 -*-
"""模拟“不同来源”试验：3 个文件 × 5 个语音段，人工加 -12/-8/-4/+3/+6 dB 后走同一归一化，
验证输出 speech_asl 是否都回到 T±0.5 dB。
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (AE, SR, asl_from_power, detect_speech, frame_mask_from_intervals,
                    frame_power, limiter_file, read_wav, write_wav)  # noqa: E402

T = -20.0
OFFSETS = [-12.0, -8.0, -4.0, 3.0, 6.0]
FILES = ["xes", "wlx", "ly"]
FRACS = [0.10, 0.30, 0.50, 0.70, 0.90]


def pick_clip(x, sr, center_s):
    """在 center_s 附近 30s 窗口内找最长连续语音段，返回 (a,b) 样本界。"""
    dur = len(x) / sr
    w0 = max(0.0, min(dur - 32.0, center_s - 15.0))
    w1 = min(dur, w0 + 30.0)
    segs = detect_speech(x[int(w0 * sr): int(w1 * sr)], sr)
    cont = [s for s in segs if s[1] - s[0] >= 3.0]
    if not cont:
        cont = segs or [(0.0, w1 - w0)]
    s, e = max(cont, key=lambda z: z[1] - z[0])
    s += w0
    e += w0
    # 左右各留 1s，最短 6s
    a = max(0.0, s - 1.0)
    b = min(dur, max(e + 1.0, a + 6.0))
    return a, b, x[int(a * sr): int(b * sr)]


def norm_v1(clip, sr, tmpdir, tag):
    """V1：静态增益到 T + limiter，返回输出音频与指标。"""
    p, _ = frame_power(clip, sr)
    segs = detect_speech(clip, sr)
    cont = [s for s in segs if s[1] - s[0] >= 3.0] or segs
    mask = frame_mask_from_intervals(cont, len(clip), sr)
    n = min(len(p), len(mask))
    if mask[:n].sum() == 0:
        mask[:n] = True
    asl_in, _, ns = asl_from_power(p[:n][mask[:n]])
    gain = T - asl_in
    y = (clip.astype(np.float64) * 10.0 ** (gain / 20.0)).astype(np.float32)
    pre = os.path.join(tmpdir, tag + "_pre.wav")
    lim = os.path.join(tmpdir, tag + "_lim.wav")
    write_wav(pre, y, sr, subtype="PCM_32")
    limiter_file(pre, lim, limit_db=-2.0)
    _, yl = read_wav(lim)
    if len(yl) > len(y):
        yl = yl[: len(y)]
    p2, _ = frame_power(yl, sr)
    n2 = min(len(p2), len(mask))
    asl_out, _, _ = asl_from_power(p2[:n2][mask[:n2]])
    for f in (pre, lim):
        try:
            os.remove(f)
        except OSError:
            pass
    return {"asl_in_dbfs": round(float(asl_in), 2), "gain_db": round(float(gain), 2),
            "asl_out_dbfs": round(float(asl_out), 3),
            "dev_db": round(float(asl_out - T), 3), "speech_sec": round(ns * 0.01, 1)}


def main():
    tmpdir = os.path.join(AE, "samples", "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for fi, short in enumerate(FILES):
        sr, x = read_wav(os.path.join(AE, "wav", short + ".wav"))
        dur = len(x) / sr
        for si, frac in enumerate(FRACS):
            off = OFFSETS[(fi * 2 + si) % len(OFFSETS)]
            a, b, clip = pick_clip(x, sr, frac * dur)
            clip_off = (clip.astype(np.float64) * 10.0 ** (off / 20.0)).astype(np.float32)
            tag = f"sim_{short}_{si}"
            m = norm_v1(clip_off, sr, tmpdir, tag)
            m.update({"file": short, "start_s": round(a, 2), "end_s": round(b, 2),
                      "clip_dur_s": round((b - a), 2), "offset_db": off})
            rows.append(m)
            print(f"[{short}] seg{si} {a:.0f}-{b:.0f}s off={off:+.0f}dB in={m['asl_in_dbfs']:.2f} "
                  f"gain={m['gain_db']:+.2f} out={m['asl_out_dbfs']:.2f} dev={m['dev_db']:+.2f}")
    devs = np.array([r["dev_db"] for r in rows])
    res = {"target_dbfs": T, "n": len(rows), "offsets": OFFSETS,
           "dev_abs_max": round(float(np.abs(devs).max()), 3),
           "dev_abs_mean": round(float(np.abs(devs).mean()), 3),
           "pass_within_0.5": int((np.abs(devs) <= 0.5).sum()),
           "rows": rows, "wall_s": round(time.perf_counter() - t0, 1)}
    out = os.path.join(AE, "sim_offsets.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[out] {out}  max|dev|={res['dev_abs_max']}  pass={res['pass_within_0.5']}/{res['n']}")


if __name__ == "__main__":
    main()
