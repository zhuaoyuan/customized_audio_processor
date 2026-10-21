# -*- coding: utf-8 -*-
"""官方 sherpa-onnx 离线说话人分离（pyannote segmentation + cam++ embedding）。

用法: python diar_official.py --wav work/audio-edit/wav/ly.wav --threshold 0.7 \
          --out work/audio-edit/diar/ly_t0.7.json [--start 0 --end 0 --threads 6]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from common import read_wav  # noqa: E402

SEG = os.path.join(ROOT, "work", "diar", "models", "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
EMB = os.path.join(ROOT, "work", "diar", "models", "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--min-dur-on", type=float, default=0.3)
    ap.add_argument("--min-dur-off", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sherpa_onnx
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG),
            num_threads=args.threads, debug=False, provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=EMB, num_threads=args.threads, debug=False, provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=args.threshold),
        min_duration_on=args.min_dur_on, min_duration_off=args.min_dur_off)
    if not cfg.validate():
        raise SystemExit("config invalid")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    sr, x = read_wav(args.wav)
    if args.start or args.end:
        a = int(args.start * sr)
        b = int(args.end * sr) if args.end > 0 else len(x)
        x = x[a:b]
    t0 = time.perf_counter()
    res = sd.process(x)
    dt = time.perf_counter() - t0
    segs = res.sort_by_start_time()
    dur = len(x) / sr
    rows = [{"start": round(float(s.start), 3), "end": round(float(s.end), 3),
             "dur": round(float(s.end - s.start), 3), "spk": int(s.speaker)} for s in segs]
    per = {}
    for r in rows:
        per[r["spk"]] = per.get(r["spk"], 0.0) + r["dur"]
    out = {"wav": os.path.abspath(args.wav), "sr": sr, "dur": round(dur, 3),
           "start": args.start, "end": args.end, "threshold": args.threshold,
           "n_speakers": int(res.num_speakers), "n_segments": len(rows),
           "process_s": round(dt, 2), "rtf": round(dt / dur, 5),
           "per_speaker_sec": {str(k): round(v, 2) for k, v in sorted(per.items(), key=lambda kv: -kv[1])},
           "segments": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    main_spk = max(per, key=per.get)
    print(f"[diar] {args.wav} dur={dur:.1f}s speakers={res.num_speakers} segs={len(rows)} "
          f"main=S{main_spk} {per[main_spk]:.1f}s ({per[main_spk]/dur*100:.1f}%) "
          f"time={dt:.1f}s rtf={dt/dur:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
