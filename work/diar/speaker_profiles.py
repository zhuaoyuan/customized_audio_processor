"""给每个（归并后）说话人做客观画像：时长、段数、F0 中位数、谱质心、cam++ 到主讲人相似度、ASR 文本。

用于代替"人工试听"判断谁在讲课（无法播放音频时）。
输出 speaker_profiles.json + speaker_profiles.md 片段。
"""
import argparse
import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "bench"))
from audiometrics import read_wav  # noqa: E402


def f0_autocorr(x, sr, fmin=60.0, fmax=400.0, win=0.04, hop=0.01):
    w, h = int(sr * win), int(sr * hop)
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    vals = []
    for i in range(0, max(len(x) - w - lag_max, 1), h):
        fr = x[i:i + w]
        if np.sqrt(np.mean(fr ** 2)) < 5e-3:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, mode="full")[w - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lag_min:lag_max]
        if len(seg) < 2:
            continue
        k = int(np.argmax(seg))
        if seg[k] / ac[0] < 0.25:
            continue
        vals.append(sr / (k + lag_min))
    return (float(np.median(vals)), len(vals)) if vals else (0.0, 0)


def spec_centroid(x, sr):
    n = 1024
    vals = []
    for i in range(0, max(len(x) - n, 1), n // 2):
        fr = x[i:i + n] * np.hanning(n)
        sp = np.abs(np.fft.rfft(fr))
        s = sp.sum()
        if s <= 1e-6:
            continue
        vals.append(float((np.arange(len(sp)) * sr / n * sp).sum() / s))
    return float(np.median(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(here, "..", "full_16k.wav"))
    ap.add_argument("--diar", default=os.path.join(here, "diar_merged_th0.7_diar.json"))
    ap.add_argument("--model-dir", default=os.path.join(here, "..", "bench", "models", "sensevoice"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--asr-cap", type=float, default=60.0, help="每个说话人最多取多少秒做 ASR")
    ap.add_argument("--out", default=os.path.join(here, "speaker_profiles.json"))
    args = ap.parse_args()

    import sherpa_onnx
    from scipy.spatial.distance import cosine

    wav_path = os.path.normpath(args.wav if os.path.isabs(args.wav)
                                else os.path.join(here, "..", "..", args.wav))
    diar_path = os.path.normpath(args.diar if os.path.isabs(args.diar)
                                 else os.path.join(here, "..", "..", args.diar))
    arr, sr = read_wav(wav_path)
    x = arr[:, 0].astype(np.float32)
    segs = json.load(open(diar_path, encoding="utf-8"))["segments"]

    md = os.path.normpath(args.model_dir if os.path.isabs(args.model_dir)
                          else os.path.join(here, "..", "..", args.model_dir))
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=os.path.join(md, "model.int8.onnx"), tokens=os.path.join(md, "tokens.txt"),
        num_threads=args.threads, use_itn=True, language="zh", debug=False)
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=os.path.join(here, "models", "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
        num_threads=args.threads, debug=False, provider="cpu"))

    per = {}
    for s in segs:
        per.setdefault(s["speaker"], []).append(s)
    dur = {k: sum(v["end"] - v["start"] for v in vs) for k, vs in per.items()}
    order = sorted(dur, key=lambda k: -dur[k])
    tot_speech = sum(dur.values())

    def emb_of(ss, cap=30.0):
        vals = []
        acc = 0.0
        for s in sorted(ss, key=lambda z: -(z["end"] - z["start"])):
            if acc >= cap:
                break
            a, b = int(s["start"] * sr), min(len(x), int(s["end"] * sr))
            b = min(b, a + int((cap - acc) * sr))
            if b <= a:
                continue
            st = ext.create_stream()
            st.accept_waveform(sr, x[a:b])
            e = np.asarray(ext.compute(st), dtype=np.float32)
            n = np.linalg.norm(e)
            if n > 0:
                vals.append((e / n) * (s["end"] - s["start"]))
            acc += (b - a) / sr
        v = np.sum(vals, axis=0)
        return v / max(np.linalg.norm(v), 1e-9)

    main_e = emb_of(per[order[0]])
    out = []
    for k in order[:args.top]:
        ss = per[k]
        # 取样音频：最长的若干段拼接到 cap
        chunks, acc = [], 0.0
        for s in sorted(ss, key=lambda z: -(z["end"] - z["start"])):
            if acc >= args.asr_cap:
                break
            a, b = int(s["start"] * sr), min(len(x), int(s["end"] * sr))
            b = min(b, a + int((args.asr_cap - acc) * sr))
            if b > a:
                chunks.append(x[a:b])
                acc += (b - a) / sr
        y = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        st = rec.create_stream()
        st.accept_waveform(sr, y)
        rec.decode_stream(st)
        text = st.result.text.strip()
        f0, nvoiced = f0_autocorr(y, sr)
        e = emb_of(ss)
        out.append({
            "speaker": k, "dur_s": round(dur[k], 1),
            "ratio_speech": round(dur[k] / tot_speech, 4),
            "n_segments": len(ss),
            "mean_seg_s": round(dur[k] / len(ss), 2),
            "max_seg_s": round(max(s["end"] - s["start"] for s in ss), 2),
            "f0_median_hz": round(f0, 1), "voiced_frames": nvoiced,
            "spec_centroid_hz": round(spec_centroid(y, sr), 1),
            "cos_to_main": round(float(1.0 - cosine(e + 1e-9, main_e + 1e-9)), 3),
            "asr_audio_s": round(acc, 1), "asr_text": text,
        })
        print(f"[S{k}] {dur[k]:.0f}s ({dur[k]/tot_speech*100:.1f}%) segs={len(ss)} "
              f"f0={f0:.0f}Hz cos={out[-1]['cos_to_main']:.2f} text={text[:70]}", flush=True)

    res = {"wav": wav_path, "diar": diar_path, "dur": len(x) / sr,
           "total_speech_s": round(tot_speech, 1),
           "speakers": sorted(out, key=lambda r: -r["dur_s"])}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
