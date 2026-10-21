"""Step 2: 逐 VAD 段 SenseVoice 转写 → 句级 start/end + 文本。

用法:
  python -X utf8 step_asr.py --wav <16k wav> --vad vad.json --out asr.json --threads 8
"""
import argparse
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TAG_RE = re.compile(r"<\|[^|]*\|>")
SPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]+")


def clean_text(t):
    t = TAG_RE.sub("", t or "")
    t = EMOJI_RE.sub("", t)
    t = SPACE_RE.sub("", t)
    return t.strip()


def build_recognizer(model_dir, threads, itn=1):
    import sherpa_onnx
    model = os.path.join(model_dir, "model.int8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model, tokens=tokens, num_threads=threads,
        use_itn=bool(itn), language="zh", debug=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=common.SENSEVOICE_DIR)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--itn", type=int, default=1)
    args = ap.parse_args()

    arr, sr = common.read_wav(args.wav)
    x = arr[:, 0].astype(np.float32)
    vad = common.read_json(args.vad)
    segs = vad["segments"]
    print(f"[in] {common.rel(args.wav)} {len(x)/sr:.1f}s, {len(segs)} VAD segments", flush=True)

    t0 = time.perf_counter()
    rec = build_recognizer(args.model_dir, args.threads, args.itn)
    t_load = time.perf_counter() - t0
    print(f"[load] SenseVoice {common.rel(args.model_dir)} threads={args.threads} "
          f"itn={args.itn} {t_load:.2f}s", flush=True)

    t0 = time.perf_counter()
    out = []
    decode_sum = 0.0
    for i, s in enumerate(segs):
        a = max(0, int(s["start"] * sr))
        b = min(len(x), int(s["end"] * sr))
        clip = x[a:b]
        st = rec.create_stream()
        st.accept_waveform(sr, clip)
        td = time.perf_counter()
        rec.decode_stream(st)
        decode_sum += time.perf_counter() - td
        text = clean_text(st.result.text)
        out.append({"start": s["start"], "end": s["end"],
                    "vad_start": s.get("vad_start", s["start"]),
                    "vad_end": s.get("vad_end", s["end"]),
                    "text": text, "clip_s": round(b / sr - a / sr, 3)})
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  asr {i+1}/{len(segs)} elapsed={el:.1f}s decode={decode_sum:.1f}s", flush=True)
    wall = time.perf_counter() - t0
    speech_s = sum(o["end"] - o["start"] for o in out)
    chars = sum(len(re.sub(r"[^\u4e00-\u9fff]", "", o["text"])) for o in out)
    empty = sum(1 for o in out if not o["text"])
    rtf = wall / (len(x) / sr)
    res = {
        "wav": common.rel(args.wav), "vad": common.rel(args.vad),
        "model_dir": common.rel(args.model_dir), "threads": args.threads,
        "itn": args.itn, "dur": round(len(x) / sr, 3),
        "load_s": round(t_load, 2), "wall_s": round(wall, 2),
        "decode_s": round(decode_sum, 2), "rtf": round(rtf, 4),
        "rtf_decode": round(decode_sum / (len(x) / sr), 4),
        "n_segments": len(out), "n_empty": empty,
        "chars_cn": chars, "clip_s": round(speech_s, 1),
        "segments": out,
    }
    common.write_json(args.out, res)
    print(f"[asr] wall={wall:.1f}s RTF={rtf:.4f} (decode-only {decode_sum:.1f}s) "
          f"chars={chars} empty={empty}", flush=True)
    print(f"[out] {common.rel(args.out)}")


if __name__ == "__main__":
    main()
