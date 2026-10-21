"""sherpa-onnx 离线说话人分离 PoC。

用法:
  python -X utf8 run_diar.py --wav work/full_16k.wav --start 0 --end 600 \
      --threshold 0.5 --out work/diar/diar_10min_t0.5.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from audiometrics import read_wav  # noqa: E402


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PMC()
        c.cb = ctypes.sizeof(c)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                 ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 1024 / 1024


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(here, "..", "full_16k.wav"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0, help="0 = 到文件末尾")
    ap.add_argument("--seg-model", default=os.path.join(here, "models",
                    "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx"))
    ap.add_argument("--emb-model", default=os.path.join(here, "models",
                    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num-clusters", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--min-dur-on", type=float, default=0.3)
    ap.add_argument("--min-dur-off", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import sherpa_onnx

    t0 = time.perf_counter()
    seg_path = os.path.normpath(os.path.join(here, "..", "..", args.seg_model)) if not os.path.isabs(args.seg_model) else args.seg_model
    emb_path = os.path.normpath(os.path.join(here, "..", "..", args.emb_model)) if not os.path.isabs(args.emb_model) else args.emb_model
    wav_path = os.path.normpath(args.wav if os.path.isabs(args.wav) else os.path.join(here, "..", "..", args.wav))

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_path),
            num_threads=args.threads,
            debug=False,
            provider=args.provider,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=emb_path,
            num_threads=args.threads,
            debug=False,
            provider=args.provider,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=args.num_clusters,
            threshold=args.threshold,
        ),
        min_duration_on=args.min_dur_on,
        min_duration_off=args.min_dur_off,
    )
    if not config.validate():
        print("[err] config.validate() failed", file=sys.stderr)
        sys.exit(2)
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    t_load = time.perf_counter() - t0
    print(f"[load] diar init {t_load:.2f}s sr={sd.sample_rate} mem={rss_mb():.0f}MB")

    arr, sr = read_wav(wav_path)
    samples = arr[:, 0].astype(np.float32)
    if args.start or args.end:
        a = int(args.start * sr)
        b = int(args.end * sr) if args.end > 0 else len(samples)
        samples = samples[a:b]
    dur = len(samples) / sr
    print(f"[audio] {wav_path} {sr}Hz seg=[{args.start},{args.end}] dur={dur:.1f}s")

    t0 = time.perf_counter()
    result = sd.process(samples)
    t_proc = time.perf_counter() - t0
    segs = result.sort_by_start_time()
    n_spk = result.num_speakers
    print(f"[diar] segments={len(segs)} speakers={n_spk} process={t_proc:.2f}s "
          f"RTF={t_proc/dur:.4f} mem={rss_mb():.0f}MB")

    rows = []
    per = {}
    for s in segs:
        d = {"start": round(float(s.start), 3), "end": round(float(s.end), 3),
             "dur": round(float(s.end - s.start), 3), "speaker": int(s.speaker)}
        rows.append(d)
        per[d["speaker"]] = per.get(d["speaker"], 0.0) + d["dur"]

    speech_total = sum(per.values())
    print("[speakers] " + " | ".join(
        f"S{k}: {v:.1f}s ({v/dur*100:.1f}% of file, {v/speech_total*100:.1f}% of speech)"
        for k, v in sorted(per.items(), key=lambda x: -x[1])))

    res = {
        "wav": wav_path, "sr": sr, "start": args.start, "end": args.end, "dur": dur,
        "seg_model": seg_path, "emb_model": emb_path,
        "threshold": args.threshold, "num_clusters_arg": args.num_clusters,
        "min_duration_on": args.min_dur_on, "min_duration_off": args.min_dur_off,
        "num_speakers": n_spk, "num_segments": len(segs),
        "load_s": t_load, "process_s": t_proc, "rtf": t_proc / dur,
        "speech_total_s": speech_total,
        "speech_ratio_of_file": speech_total / dur,
        "per_speaker": {str(k): v for k, v in sorted(per.items(), key=lambda x: -x[1])},
        "segments": rows,
    }
    out = args.out or os.path.join(here, "diar_%ds_t%.2f.json" % (int(dur), args.threshold))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[out] {out}")

    # 粗略 RTTM 便于外部查看
    rttm = os.path.splitext(out)[0] + ".rttm"
    with open(rttm, "w", encoding="utf-8") as f:
        for d in rows:
            f.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> S%d <NA> <NA>\n" % (
                os.path.basename(wav_path), d["start"], d["dur"], d["speaker"]))
    print(f"[out] {rttm}")


if __name__ == "__main__":
    main()
