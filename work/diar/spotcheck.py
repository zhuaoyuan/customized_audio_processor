"""人工抽检替代方案：10 段 x 10s，输出 ASR 文本 + 谱特征 + 图（时间轴、频谱图）。

无法试听时，用"ASR 是否有连续文本 + 谱特征（F0/谱质心/能量）"判断每段属于
主讲人 / 其他说话人 / 非语音，并注明判据与局限。
"""
import argparse
import json
import os
import sys
import wave

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "bench"))
from audiometrics import read_wav  # noqa: E402


def write_wav(path, x, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767.0).astype(np.int16).tobytes())


def f0_autocorr(x, sr, fmin=60.0, fmax=400.0, win=0.04, hop=0.01):
    w, h = int(sr * win), int(sr * hop)
    lo, hi = int(sr / fmax), int(sr / fmin)
    vals = []
    for i in range(0, max(len(x) - w - hi, 1), h):
        fr = x[i:i + w]
        if np.sqrt(np.mean(fr ** 2)) < 5e-3:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, mode="full")[w - 1:]
        if ac[0] <= 0 or len(ac[lo:hi]) < 2:
            continue
        seg = ac[lo:hi]
        k = int(np.argmax(seg))
        if seg[k] / ac[0] < 0.25:
            continue
        vals.append(sr / (k + lo))
    return float(np.median(vals)) if vals else 0.0


def spec_centroid(x, sr, n=1024):
    vals = []
    for i in range(0, max(len(x) - n, 1), n // 2):
        sp = np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n)))
        s = sp.sum()
        if s > 1e-6:
            vals.append(float((np.arange(len(sp)) * sr / n * sp).sum() / s))
    return float(np.median(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(here, "..", "full_16k.wav"))
    ap.add_argument("--diar", default=os.path.join(here, "final_diar.json"))
    ap.add_argument("--model-dir", default=os.path.join(here, "..", "bench", "models", "sensevoice"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--outdir", default=os.path.join(here, "spotcheck"))
    args = ap.parse_args()

    import sherpa_onnx

    wav_path = os.path.normpath(args.wav if os.path.isabs(args.wav)
                                else os.path.join(here, "..", "..", args.wav))
    diar_path = os.path.normpath(args.diar if os.path.isabs(args.diar)
                                 else os.path.join(here, "..", "..", args.diar))
    arr, sr = read_wav(wav_path)
    x = arr[:, 0].astype(np.float32)
    dur = len(x) / sr
    segs = json.load(open(diar_path, encoding="utf-8"))["segments"]
    os.makedirs(args.outdir, exist_ok=True)

    md = os.path.normpath(args.model_dir if os.path.isabs(args.model_dir)
                          else os.path.join(here, "..", "..", args.model_dir))
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=os.path.join(md, "model.int8.onnx"), tokens=os.path.join(md, "tokens.txt"),
        num_threads=args.threads, use_itn=True, language="zh", debug=False)

    per = {}
    for s in segs:
        per.setdefault(s["speaker"], []).append(s)

    picks = []
    # 主讲人 4 段（分散）
    for t in (500, 1250, 2400, 4000):
        picks.append((f"main_{t}s", t, 0))
    # 非主讲人：按时长取前 4 个簇里最长的段，居中 10s
    others = sorted([k for k in per if k not in (0, 99)], key=lambda k: -sum(z["dur"] for z in per[k]))[:4]
    for k in others:
        s = max(per[k], key=lambda z: z["end"] - z["start"])
        c = (s["start"] + s["end"]) / 2
        picks.append((f"other_S{k}", max(0.0, c - 5), k))
    # 非语音 2 段（片头/片尾）
    picks.append(("nonspeech_head", 5.0, 99))
    picks.append(("nonspeech_tail", dur - 35.0, 99))

    rows = []
    for name, t, lab in picks:
        a, b = int(t * sr), min(len(x), int((t + 10) * sr))
        y = x[a:b]
        st = rec.create_stream()
        st.accept_waveform(sr, y)
        rec.decode_stream(st)
        text = st.result.text.strip()
        write_wav(os.path.join(args.outdir, f"{name}_{int(t)}s.wav"), y, sr)
        rms = 20 * np.log10(max(float(np.sqrt(np.mean(y ** 2))), 1e-9))
        row = {"name": name, "start": round(b / sr - 10, 1), "end": round(b / sr, 1),
               "label": {0: "主讲人", 99: "非语音"}.get(lab, f"其他(S{lab})"),
               "rms_dbfs": round(rms, 1), "f0_hz": round(f0_autocorr(y, sr), 1),
               "spec_centroid_hz": round(spec_centroid(y, sr), 1),
               "asr_text": text, "asr_chars": len("".join(text.split())),
               "chars_per_s": round(len("".join(text.split())) / 10.0, 2)}
        rows.append(row)
        print(f"{name:18s} {row['start']:7.1f}-{row['end']:7.1f} [{row['label']}] "
              f"rms={row['rms_dbfs']:6.1f}dB F0={row['f0_hz']:5.1f}Hz cent={row['spec_centroid_hz']:5.0f} "
              f"chars/s={row['chars_per_s']:.2f} | {text[:60]}", flush=True)

    with open(os.path.join(args.outdir, "spotcheck.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    # --- 图1：时间轴（RMS + 说话人掩码）---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hop = 0.01
        nf = int(dur / hop) + 1
        masks = {"main": np.zeros(nf, bool), "other": np.zeros(nf, bool), "nonspeech": np.zeros(nf, bool)}
        for s in segs:
            a, b = max(0, int(round(s["start"] / hop))), min(nf, int(round(s["end"] / hop)))
            key = "main" if s["speaker"] == 0 else ("nonspeech" if s["speaker"] == 99 else "other")
            masks[key][a:b] = True
        sec = np.arange(nf) * hop
        step = 100  # 1s 分辨率
        m = {k: v[:len(v) // step * step].reshape(-1, step).mean(1) for k, v in masks.items()}
        rms = np.sqrt(np.mean(x[:len(x) // (sr) * sr].reshape(-1, sr) ** 2, axis=1))
        fig, ax = plt.subplots(2, 1, figsize=(14, 5.2), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.2]})
        ax[0].plot(np.arange(len(rms)), 20 * np.log10(np.maximum(rms, 1e-9)), lw=0.5, color="k")
        ax[0].set_ylabel("RMS dBFS"); ax[0].set_title("78-min recording: energy and diarization timeline")
        ax[0].grid(alpha=.3)
        t1 = np.arange(len(m["main"]))
        ax[1].fill_between(t1, 0, m["main"], color="tab:green", label="main speaker", step="mid")
        ax[1].fill_between(t1, 1, 1 + m["other"], color="tab:red", label="other speech", step="mid")
        ax[1].fill_between(t1, 2, 2 + m["nonspeech"], color="tab:gray", label="non-speech (music/noise)", step="mid")
        ax[1].set_yticks([0.5, 1.5, 2.5]); ax[1].set_yticklabels(["main", "other", "non-speech"])
        ax[1].set_xlabel("time (s)"); ax[1].legend(loc="upper right", fontsize=8)
        ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fig_timeline.png"), dpi=110)
        plt.close(fig)

        # --- 图2：3 段频谱图对比 ---
        sel = [("lecture / main (500-510s)", 500, 510),
               ("other speech / student (S21)", rows[4]["start"], rows[4]["end"]),
               ("head non-speech (5-15s)", 5, 15)]
        fig, axs = plt.subplots(3, 1, figsize=(10, 6.5))
        for axx, (title, t0, t1) in zip(axs, sel):
            a, b = int(t0 * sr), min(len(x), int(t1 * sr))
            axx.specgram(x[a:b], NFFT=512, Fs=sr, noverlap=256, cmap="magma")
            axx.set_title(title, fontsize=9); axx.set_ylim(0, 4000); axx.set_ylabel("Hz")
        axs[-1].set_xlabel("time within clip (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fig_spectrograms.png"), dpi=110)
        plt.close(fig)
        print(f"[fig] {os.path.join(args.outdir, 'fig_timeline.png')}")
        print(f"[fig] {os.path.join(args.outdir, 'fig_spectrograms.png')}")
    except Exception as e:
        print(f"[warn] figure failed: {e}")


if __name__ == "__main__":
    main()
