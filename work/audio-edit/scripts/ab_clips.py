# -*- coding: utf-8 -*-
"""生成 20s A/B 试听片段：before=原始解码音频，after=clean.wav（处理后紧凑时间轴）。"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (AE, SR, asl_from_power, detect_speech, frame_mask_from_intervals,
                    frame_power, read_wav, write_wav)  # noqa: E402

# (short, clean 时间轴比例)
CLIPS = [("xes", 0.50), ("wlx", 0.30), ("ly", 0.70)]
DUR = 20.0


def clean_to_orig(kept, t):
    c = 0.0
    for ks, ke in kept:
        d = ke - ks
        if c <= t < c + d:
            return ks + (t - c)
        c += d
    return None


def level_of(x, sr):
    p, _ = frame_power(x, sr)
    segs = detect_speech(x, sr)
    from common import classify_segments
    cont, _ = classify_segments(segs, 3.0)
    use = cont or segs
    mask = frame_mask_from_intervals(use, len(x), sr)
    n = min(len(p), len(mask))
    if n == 0 or mask[:n].sum() == 0:
        return None
    a, _, _ = asl_from_power(p[:n][mask[:n]])
    return round(float(a), 2)


def main():
    os.makedirs(os.path.join(AE, "ab"), exist_ok=True)
    rows = []
    for short, frac in CLIPS:
        outdir = os.path.join(AE, short)
        clean_p = os.path.join(outdir, "clean.wav")
        map_p = os.path.join(outdir, "mapping.json")
        if not (os.path.exists(clean_p) and os.path.exists(map_p)):
            print(f"[skip] {short} missing clean/mapping")
            continue
        sr, clean = read_wav(clean_p)
        sr2, src = read_wav(os.path.join(AE, "wav", short + ".wav"))
        mapping = json.load(open(map_p, "r", encoding="utf-8"))
        kept = [(k[0], k[1]) for k in mapping["kept"]]
        t0 = max(0.0, min(len(clean) / sr - DUR, frac * len(clean) / sr - DUR / 2))
        orig0 = clean_to_orig(kept, t0)
        if orig0 is None:
            print(f"[skip] {short} mapping fail")
            continue
        before = src[int(orig0 * sr2): int((orig0 + DUR) * sr2)]
        after = clean[int(t0 * sr): int((t0 + DUR) * sr)]
        name = f"{short}_{int(round(frac*100))}pct"
        bp = os.path.join(AE, "ab", name + "_before.wav")
        ap = os.path.join(AE, "ab", name + "_after.wav")
        write_wav(bp, before, SR, subtype="PCM_16")
        write_wav(ap, after, SR, subtype="PCM_16")
        rows.append({"name": name, "file": short, "clean_start_s": round(t0, 2),
                     "orig_start_s": round(orig0, 2), "dur_s": DUR,
                     "before_wav": bp, "after_wav": ap,
                     "before_asl_dbfs": level_of(before, sr2),
                     "after_asl_dbfs": level_of(after, sr)})
        print(f"[ab] {name}: orig {orig0:.1f}s clean {t0:.1f}s "
              f"before_asl={rows[-1]['before_asl_dbfs']} after_asl={rows[-1]['after_asl_dbfs']}")
    out = os.path.join(AE, "ab", "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"note": "before=原始解码16k；after=处理后 clean.wav；同一内容对齐", "clips": rows},
                  f, ensure_ascii=False, indent=1)
    print("[out]", out)


if __name__ == "__main__":
    main()
