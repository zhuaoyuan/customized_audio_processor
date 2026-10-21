"""在固定片段上扫 threshold / num_clusters / min_duration_on，比较说话人数与主导说话人占比。

模型只加载一次。结果写 JSON，供调参决策。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "bench"))
from audiometrics import read_wav  # noqa: E402

MODELS = os.path.join(here, "models")


def build(threshold, num_clusters, min_on, min_off, threads=8):
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=os.path.join(MODELS, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")),
            num_threads=threads, debug=False, provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=os.path.join(MODELS, "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
            num_threads=threads, debug=False, provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold),
        min_duration_on=min_on, min_duration_off=min_off,
    )
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def stats(segs, dur):
    per = {}
    for s in segs:
        per[int(s.speaker)] = per.get(int(s.speaker), 0.0) + (s.end - s.start)
    tot = sum(per.values())
    order = sorted(per.items(), key=lambda x: -x[1])
    return {
        "n_seg": len(segs), "n_spk": len(per), "speech_s": tot,
        "speech_ratio": tot / dur,
        "top": [{"spk": k, "s": round(v, 1), "ratio_file": round(v / dur, 4),
                 "ratio_speech": round(v / tot, 4) if tot else 0} for k, v in order[:5]],
        "n_spk_ge_5s": sum(1 for k, v in per.items() if v >= 5.0),
        "n_spk_ge_10s": sum(1 for k, v in per.items() if v >= 10.0),
        "n_spk_ge_30s": sum(1 for k, v in per.items() if v >= 30.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(here, "..", "full_16k.wav"))
    ap.add_argument("--start", type=float, default=600)
    ap.add_argument("--end", type=float, default=1200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(here, "tune_600_1200.json"))
    ap.add_argument("--configs", default="0.5,-1;0.6,-1;0.7,-1;0.8,-1;0.9,-1;0.7,2;0.7,3;0.7,4")
    args = ap.parse_args()

    arr, sr = read_wav(os.path.normpath(args.wav if os.path.isabs(args.wav)
                                       else os.path.join(here, "..", "..", args.wav)))
    samples = arr[:, 0].astype(np.float32)
    a, b = int(args.start * sr), int(args.end * sr)
    samples = samples[a:b]
    dur = len(samples) / sr
    print(f"[audio] {args.start}-{args.end}s dur={dur:.1f}s sr={sr}", flush=True)

    rows = []
    for item in args.configs.split(";"):
        th_s, nc_s = item.split(",")
        th, nc = float(th_s), int(nc_s)
        t0 = time.perf_counter()
        sd = build(th, nc, 0.3, 0.5, args.threads)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = sd.process(samples)
        segs = res.sort_by_start_time()
        t_proc = time.perf_counter() - t0
        st = stats(segs, dur)
        st.update({"threshold": th, "num_clusters": nc, "load_s": round(t_load, 2),
                   "proc_s": round(t_proc, 2), "rtf": round(t_proc / dur, 4)})
        rows.append(st)
        print(f"[cfg th={th} nc={nc}] spk={st['n_spk']} (>=5s:{st['n_spk_ge_5s']} "
              f">=10s:{st['n_spk_ge_10s']} >=30s:{st['n_spk_ge_30s']}) "
              f"speech={st['speech_s']:.0f}s({st['speech_ratio']*100:.1f}%) "
              f"top={[(x['spk'], x['s']) for x in st['top'][:3]]} proc={t_proc:.1f}s", flush=True)
        if (th, nc) == (0.7, -1):
            with open(os.path.splitext(args.out)[0] + "_th0.7_segs.json", "w", encoding="utf-8") as f:
                json.dump({"segments": [{"start": float(s.start), "end": float(s.end),
                                         "speaker": int(s.speaker)} for s in segs]},
                          f, ensure_ascii=False)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"start": args.start, "end": args.end, "dur": dur, "runs": rows},
                  f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
