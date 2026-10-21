"""按固定窗口用 SenseVoice 转写整片 wav，输出每窗文本（供切前/切后对齐比较）。"""
import argparse
import json
import os
import sys
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "bench"))
from audiometrics import read_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model-dir", default=os.path.join(here, "..", "bench", "models", "sensevoice"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--itn", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sherpa_onnx

    model_dir = os.path.normpath(args.model_dir if os.path.isabs(args.model_dir)
                                 else os.path.join(here, "..", "..", args.model_dir))
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=os.path.join(model_dir, "model.int8.onnx"),
        tokens=os.path.join(model_dir, "tokens.txt"),
        num_threads=args.threads, use_itn=bool(args.itn), language="zh", debug=False)
    wav_path = os.path.normpath(args.wav if os.path.isabs(args.wav)
                                else os.path.join(here, "..", "..", args.wav))
    arr, sr = read_wav(wav_path)
    x = arr[:, 0].astype(np.float32)
    dur = len(x) / sr

    step = int(args.window * sr)
    t0 = time.perf_counter()
    wins = []
    for i in range(0, len(x), step):
        seg = x[i:i + step]
        if len(seg) < sr * 0.3:
            continue
        s = recognizer.create_stream()
        s.accept_waveform(sr, seg)
        recognizer.decode_stream(s)
        wins.append({"start": round(i / sr, 2), "end": round(min(len(x), i + step) / sr, 2),
                     "text": s.result.text.strip()})
    wall = time.perf_counter() - t0
    text = "".join(w["text"] for w in wins)
    res = {"wav": wav_path, "sr": sr, "dur": dur, "window": args.window, "threads": args.threads,
           "itn": args.itn, "wall_s": round(wall, 2), "rtf": round(wall / dur, 4),
           "n_windows": len(wins), "chars": len(text), "text": text, "windows": wins}
    out = os.path.normpath(args.out if os.path.isabs(args.out) else os.path.join(here, "..", "..", args.out))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[asr] {wav_path} dur={dur:.1f}s windows={len(wins)} wall={wall:.1f}s RTF={wall/dur:.4f} "
          f"chars={len(text)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
