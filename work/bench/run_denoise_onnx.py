"""ONNX 降噪模型基线：sherpa-onnx OfflineSpeechDenoiser + GTCRN（gtcrn_simple.onnx）。"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audiometrics import read_wav, metrics

BENCH = os.path.dirname(os.path.abspath(__file__))


def summarize(tag, path, dur, wall=None, model_mb=None):
    d = metrics(path)
    row = {"tag": tag, "file": os.path.basename(path), "dur": round(d["dur"], 2),
           "rms_dbfs": round(d["rms_dbfs"], 2), "peak_dbfs": round(d["peak_dbfs"], 2),
           "p1": round(d["p1"], 2), "p10": round(d["p10"], 2), "p50": round(d["p50"], 2),
           "p90": round(d["p90"], 2), "p99": round(d["p99"], 2),
           "snr_proxy": round(d["snr_proxy"], 2),
           "noise_floor_dbfs": round(d["noise_floor_dbfs"], 2),
           "speech_level_dbfs": round(d["speech_level_dbfs"], 2),
           "snr_est_db": round(d["snr_est_db"], 2)}
    if wall is not None:
        row["wall_s"] = round(wall, 2)
        row["rtf"] = round(wall / dur, 4)
    if model_mb:
        row["model_mb"] = round(model_mb, 2)
    return row


def main():
    import sherpa_onnx

    model = os.path.join(BENCH, "models", "gtcrn", "gtcrn_simple.onnx")
    if not os.path.exists(model):
        print(f"[FAIL] 模型不存在: {model}")
        sys.exit(2)
    model_mb = os.path.getsize(model) / 1024 / 1024
    outdir = os.path.join(BENCH, "denoise")
    os.makedirs(outdir, exist_ok=True)
    rows = []

    t0 = time.perf_counter()
    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=model),
            num_threads=4,
        )
    )
    denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
    t_load = time.perf_counter() - t0
    print(f"[load] OfflineSpeechDenoiser(gtcrn) init {t_load:.2f}s model={model_mb:.2f} MB")

    for src in sys.argv[1:] or [os.path.join(BENCH, "..", "samples", "s1_mid.wav")]:
        src = os.path.abspath(src)
        base = os.path.splitext(os.path.basename(src))[0]
        arr, sr = read_wav(src)
        audio = arr[:, 0].astype(np.float32)
        if sr != 16000:
            print(f"[skip] {src} sr={sr} != 16000")
            continue
        t0 = time.perf_counter()
        da = denoiser.run(audio, sample_rate=sr)
        wall = time.perf_counter() - t0
        out, out_sr = np.asarray(da.samples, dtype=np.float32), int(da.sample_rate)
        outp = os.path.join(outdir, f"{base}_gtcrn.wav")
        sherpa_onnx.write_wave(outp, out, sample_rate=out_sr)
        dur = len(audio) / sr
        row = summarize("gtcrn_onnx", outp, dur, wall, model_mb)
        row["load_s"] = round(t_load, 2)
        row["rtf"] = round(wall / dur, 4)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        print(f"[time] {base}: {wall:.2f}s / {dur:.1f}s  RTF={wall/dur:.4f}  "
              f"(1s 音频 {wall/dur*1000:.0f} ms; 78min -> {78*60*(wall/dur)/60:.2f} min)")
        # 与原始对比
        orig = summarize("baseline", src, dur)
        print("[基线] " + json.dumps(orig, ensure_ascii=False))

    out_json = os.path.join(BENCH, "denoise_gtcrn.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[out] {out_json}")


if __name__ == "__main__":
    main()
