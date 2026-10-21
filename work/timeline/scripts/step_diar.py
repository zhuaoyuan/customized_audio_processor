"""Step 3a: sherpa-onnx 离线说话人分离（pyannote-seg + cam++ embedding + FastClustering）。

用法（整片，threshold=0.7）:
  python -X utf8 step_diar.py --wav <16k wav> --out diar.json --threshold 0.7 --threads 8
用法（10 分钟片段，用于稳定性对照）:
  ... --start 600 --end 1200
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
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0, help="0=到文件末尾")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--num-clusters", type=int, default=-1)
    ap.add_argument("--seg-model", default=common.DIAR_SEG_MODEL)
    ap.add_argument("--emb-model", default=common.DIAR_EMB_MODEL)
    ap.add_argument("--min-dur-on", type=float, default=0.3)
    ap.add_argument("--min-dur-off", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import sherpa_onnx
    arr, sr = common.read_wav(args.wav)
    x = arr[:, 0].astype(np.float32)
    if args.start or args.end:
        a = int(args.start * sr)
        b = int(args.end * sr) if args.end > 0 else len(x)
        x = x[a:b]
    dur = len(x) / sr
    print(f"[audio] {common.rel(args.wav)} [{args.start},{args.end or 'end'}] {dur:.1f}s", flush=True)

    t0 = time.perf_counter()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=args.seg_model),
            num_threads=args.threads, debug=False, provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.emb_model, num_threads=args.threads, debug=False, provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=args.num_clusters, threshold=args.threshold),
        min_duration_on=args.min_dur_on,
        min_duration_off=args.min_dur_off,
    )
    if not config.validate():
        print("[err] diar config invalid", file=sys.stderr)
        sys.exit(2)
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = sd.process(x)
    t_proc = time.perf_counter() - t0
    segs = result.sort_by_start_time()
    rows = []
    per = {}
    for s in segs:
        d = {"start": round(float(s.start + args.start), 3),
             "end": round(float(s.end + args.start), 3),
             "speaker": int(s.speaker)}
        d["dur"] = round(d["end"] - d["start"], 3)
        rows.append(d)
        per[d["speaker"]] = per.get(d["speaker"], 0.0) + d["dur"]
    speech_s = sum(per.values())
    res = {
        "wav": common.rel(args.wav), "start": args.start, "end": args.end,
        "dur": round(dur, 3), "threshold": args.threshold,
        "num_clusters_arg": args.num_clusters,
        "min_duration_on": args.min_dur_on, "min_duration_off": args.min_dur_off,
        "num_speakers": int(result.num_speakers), "n_segments": len(rows),
        "load_s": round(t_load, 2), "process_s": round(t_proc, 2),
        "rtf": round(t_proc / dur, 4), "speech_s": round(speech_s, 1),
        "per_speaker": {str(k): round(v, 1) for k, v in sorted(per.items(), key=lambda z: -z[1])},
        "segments": rows,
    }
    common.write_json(args.out, res)
    top = sorted(per.items(), key=lambda z: -z[1])[:5]
    print(f"[diar] speakers={result.num_speakers} segments={len(rows)} "
          f"process={t_proc:.1f}s RTF={t_proc/dur:.4f} speech={speech_s:.0f}s", flush=True)
    print("[speakers] " + " | ".join(f"S{k}:{v:.0f}s({v/speech_s*100:.1f}%)" for k, v in top), flush=True)
    print(f"[out] {common.rel(args.out)}")


if __name__ == "__main__":
    main()
