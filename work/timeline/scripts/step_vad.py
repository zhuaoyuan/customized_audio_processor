"""Step 1: Silero VAD 切段（32ms 原生流式概率 + 10ms 能量谷精修边界）。

用法:
  python -X utf8 step_vad.py --wav work/timeline/_cache/24_xx_16k.wav \
      --out work/timeline/24_xx/vad.json --threads 8
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--neg-threshold", type=float, default=0.35)
    ap.add_argument("--min-speech", type=float, default=0.60)
    ap.add_argument("--min-silence", type=float, default=0.45)
    ap.add_argument("--max-speech", type=float, default=30.0)
    ap.add_argument("--max-gap", type=float, default=0.20)
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--refine", type=int, default=1)
    args = ap.parse_args()

    arr, sr = common.read_wav(args.wav)
    assert sr == 16000, sr
    x = arr[:, 0].astype(np.float32)
    dur = len(x) / sr
    print(f"[audio] {common.rel(args.wav)} {dur:.1f}s", flush=True)

    t0 = time.perf_counter()
    vad = common.SileroVAD(threads=args.threads)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    probs = vad.probs(x)
    t_probs = time.perf_counter() - t0
    hop = common.SileroVAD.HOP / sr
    print(f"[vad] load={t_load:.2f}s probs={t_probs:.2f}s (RTF {t_probs/dur:.4f}) "
          f"frames={len(probs)} speech_ratio={np.mean(probs >= args.threshold)*100:.1f}%", flush=True)

    t0 = time.perf_counter()
    raw = common.probs_to_segments(probs, hop=hop, threshold=args.threshold,
                                   neg_threshold=args.neg_threshold,
                                   min_speech=args.min_speech, min_silence=args.min_silence,
                                   max_speech=args.max_speech, pad=0.0, total_dur=dur)
    merged = common.post_merge(raw, max_gap=args.max_gap, min_dur=args.min_speech,
                               max_dur=args.max_speech)
    final = common.refine_boundaries(merged, x) if args.refine else merged
    # 报告边界 = 精修边界再向两侧补 pad（ASR 解码窗），逐段裁剪到与相邻段中点
    segs = []
    for i, (s, e) in enumerate(final):
        a = max(0.0, s - args.pad)
        b = min(dur, e + args.pad)
        if i > 0:
            a = max(a, (final[i - 1][1] + s) / 2)
        if i + 1 < len(final):
            b = min(b, (e + final[i + 1][0]) / 2)
        segs.append({"start": round(a, 3), "end": round(b, 3),
                     "vad_start": round(s, 3), "vad_end": round(e, 3),
                     "dur": round(b - a, 3)})
    t_seg = time.perf_counter() - t0

    speech_s = sum(s["vad_end"] - s["vad_start"] for s in segs)
    res = {
        "wav": common.rel(args.wav), "sr": sr, "dur": round(dur, 3),
        "params": {"vad_model": common.rel(common.VAD_MODEL), "threshold": args.threshold,
                   "neg_threshold": args.neg_threshold, "min_speech": args.min_speech,
                   "min_silence": args.min_silence, "max_speech": args.max_speech,
                   "max_gap": args.max_gap, "pad": args.pad,
                   "refine_10ms": bool(args.refine)},
        "timings": {"load_s": round(t_load, 2), "probs_s": round(t_probs, 2),
                    "segment_s": round(t_seg, 2), "total_s": round(t_load + t_probs + t_seg, 2)},
        "n_frames": int(len(probs)), "frame_hop_ms": 32,
        "n_segments": len(segs), "speech_s": round(speech_s, 1),
        "speech_ratio": round(speech_s / dur, 4),
        "segments": segs,
        "raw_segments": [{"start": round(s, 3), "end": round(e, 3)} for s, e in merged],
    }
    common.write_json(args.out, res)
    print(f"[segs] {len(segs)} segments, speech={speech_s:.0f}s ({res['speech_ratio']*100:.1f}%), "
          f"avg={speech_s/max(len(segs),1):.2f}s, segment+refine={t_seg:.2f}s", flush=True)
    print(f"[out] {common.rel(args.out)}")


if __name__ == "__main__":
    main()
