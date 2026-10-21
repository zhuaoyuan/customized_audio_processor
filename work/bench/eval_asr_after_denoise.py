"""降噪对识别质量的影响：对原始/各种降噪结果跑 SenseVoice，算与基线的字符级相似度。"""
import difflib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audiometrics import read_wav

BENCH = os.path.dirname(os.path.abspath(__file__))


def main():
    import sherpa_onnx

    model_dir = os.path.join(BENCH, "models", "sensevoice")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=os.path.join(model_dir, "model.int8.onnx"),
        tokens=os.path.join(model_dir, "tokens.txt"),
        num_threads=4, use_itn=True, language="zh", debug=False,
    )
    files = sys.argv[1:]
    if not files:
        print("usage: eval_asr_after_denoise.py a.wav b.wav ...（第一个作为基线）")
        sys.exit(2)

    def transcribe(path):
        arr, sr = read_wav(path)
        audio = arr[:, 0].astype(np.float32)
        s = recognizer.create_stream()
        s.accept_waveform(sr, audio)
        recognizer.decode_stream(s)
        return s.result.text, len(audio) / sr

    base_text = None
    rows = []
    for p in files:
        text, dur = transcribe(p)
        if base_text is None:
            base_text = text
            ratio = 1.0
        else:
            ratio = difflib.SequenceMatcher(None, base_text, text).ratio()
        row = {"file": os.path.basename(p), "chars": len(text), "dur": round(dur, 1),
               "similarity_to_baseline": round(ratio, 3), "text": text}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}, ensure_ascii=False))
        print("  " + text)
    out = os.path.join(BENCH, "asr_after_denoise.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
