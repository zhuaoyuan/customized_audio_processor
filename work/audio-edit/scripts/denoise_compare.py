# -*- coding: utf-8 -*-
"""降噪对比：afftdn（默认链） vs arnndn vs 仅 highpass。

只在“语音段内/长停顿时段内”统计，避免片头音乐污染：
- speech_rms_db：连续语音段(>=3s)内帧均方功率
- noise_rms_db：距离语音 >=0.3s 的长空隙(>=1.5s)内帧均方功率
- noise_floor_p10_db：上述空隙帧 dB 的 p10
- SNR_proxy = speech_rms - noise_rms
同时记录 300s 片段 RTF；并输出一张频谱对比图。
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (AE, SR, classify_segments, db, detect_speech, frame_power, read_wav)  # noqa: E402

CHUNKS = [("xes", 600.0, 900.0), ("wlx", 1200.0, 1500.0), ("ly", 2400.0, 2700.0)]
MODES = [("highpass", "highpass=f=80"),
         ("afftdn", "highpass=f=80,afftdn=nr=12:nf=-30:tn=1"),
         # rnnoise(sh.rnnn) 只支持 48kHz：必须 aresample 到 48k 再回来，否则输出被误当 16k（前面踩过坑）
         ("arnndn", "aresample=48000,highpass=f=80,arnndn=m=sh.rnnn,aresample=16000")]


def run_ff(src, dst, af, cwd=None):
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-af", af, "-ar", "16000", "-c:a", "pcm_f32le", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1000:])
    return dst


def masks_for(x, sr):
    segs = detect_speech(x, sr)
    cont, frag = classify_segments(segs, 3.0)
    p, _ = frame_power(x, sr)
    n = min(len(p), int(np.ceil(len(x) / (sr * 0.01))))
    centers = np.arange(n) * 0.01 + 0.01  # 近似帧中心(秒)，仅用于区间判定
    sp = np.zeros(n, dtype=bool)
    for s, e in cont:
        sp |= (centers >= s) & (centers < e)
    noise = np.zeros(n, dtype=bool)
    prev = 0.0
    for s, e in segs:
        gap, ge = prev, s
        if ge - gap >= 1.5:
            noise |= (centers >= gap + 0.3) & (centers < ge - 0.3)
        prev = max(prev, e)
    if len(x) / sr - prev >= 1.5:
        noise |= (centers >= prev + 0.3) & (centers < len(x) / sr - 0.3)
    return p, sp, noise, segs, cont


def metrics(x, sr):
    p, sp, noise, segs, cont = masks_for(x, sr)
    n = min(len(p), len(sp))
    ps = p[:n][sp[:n]]
    pn = p[:n][noise[:n]]
    sp_db = db(np.mean(ps)) if len(ps) else float("nan")
    ns_db = db(np.mean(pn)) if len(pn) else float("nan")
    return {"speech_rms_db": round(float(sp_db), 2),
            "noise_rms_db": round(float(ns_db), 2),
            "noise_floor_p10_db": round(float(np.percentile(db(pn), 10)), 2) if len(pn) else None,
            "snr_proxy_db": round(float(sp_db - ns_db), 2) if len(pn) else None,
            "speech_ratio": round(sum(e - s for s, e in cont) / (len(x) / sr), 4)}


def spectrogram(path, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, x = read_wav(path)
    sr = SR
    x = x[int(5 * sr): int(15 * sr)]
    nfft, hop = 1024, 256
    frames = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop] * np.hanning(nfft)
    S = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    Sdb = 10 * np.log10(S + 1e-12)
    plt.figure(figsize=(9, 3.2))
    plt.imshow(Sdb.T, origin="lower", aspect="auto", cmap="viridis",
               extent=[0, 10, 0, sr / 2000], vmin=Sdb.max() - 80, vmax=Sdb.max())
    plt.title(title)
    plt.xlabel("time in 10s clip (s)")
    plt.ylabel("freq (kHz)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()


def main():
    outdir = os.path.join(AE, "denoise")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(AE, "figs"), exist_ok=True)
    rows = []
    t_all = time.perf_counter()
    for short, a, b in CHUNKS:
        sr, x = read_wav(os.path.join(AE, "wav", short + ".wav"))
        clip = x[int(a * sr): int(b * sr)]
        src = os.path.join(outdir, f"{short}_{int(a)}_{int(b)}_orig.wav")
        from common import write_wav
        write_wav(src, clip, sr, subtype="PCM_32")
        base = metrics(clip, sr)
        row = {"file": short, "start": a, "end": b, "dur": round((b - a), 1), "orig": base}
        for name, af in MODES:
            dst = os.path.join(outdir, f"{short}_{int(a)}_{int(b)}_{name}.wav")
            t0 = time.perf_counter()
            run_ff(src, dst, af, cwd=os.path.join(os.path.dirname(__file__), "..", "..",
                                                  "bench", "models", "rnnoise"))
            dt = time.perf_counter() - t0
            _, y = read_wav(dst)
            m = metrics(y, sr)
            m["rtf"] = round(dt / (b - a), 4)
            m["wall_s"] = round(dt, 2)
            m["speech_delta_db"] = round(m["speech_rms_db"] - base["speech_rms_db"], 2)
            m["noise_delta_db"] = (round(m["noise_rms_db"] - base["noise_rms_db"], 2)
                                   if m["noise_rms_db"] is not None and base["noise_rms_db"] is not None else None)
            m["snr_delta_db"] = (round(m["snr_proxy_db"] - base["snr_proxy_db"], 2)
                                 if m.get("snr_proxy_db") is not None and base.get("snr_proxy_db") is not None else None)
            row[name] = m
            print(f"[{short} {a:.0f}-{b:.0f}] {name}: speech {m['speech_delta_db']:+.2f}dB "
                  f"noise {m['noise_delta_db']:+.2f}dB snr {m['snr_delta_db']:+.2f}dB rtf {m['rtf']}")
        rows.append(row)
        if short == "xes":
            for name, _ in [("orig", None)] + MODES:
                p = src if name == "orig" else os.path.join(outdir, f"{short}_{int(a)}_{int(b)}_{name}.wav")
                spectrogram(p, os.path.join(AE, "figs", f"denoise_{name}.png"), f"{short} 600-900s {name}")
    # 汇总
    summary = {}
    for name, _ in [("highpass", None), ("afftdn", None), ("arnndn", None)]:
        sd = np.mean([r[name]["speech_delta_db"] for r in rows if r.get(name)])
        nd = np.mean([r[name]["noise_delta_db"] for r in rows if r.get(name)])
        sn = np.mean([r[name]["snr_delta_db"] for r in rows if r.get(name)])
        rtf = np.mean([r[name]["rtf"] for r in rows if r.get(name)])
        summary[name] = {"speech_delta_db_mean": round(float(sd), 2),
                         "noise_delta_db_mean": round(float(nd), 2),
                         "snr_delta_db_mean": round(float(sn), 2),
                         "rtf_mean": round(float(rtf), 4)}
    res = {"chunks": rows, "summary": summary,
           "wall_s": round(time.perf_counter() - t_all, 1)}
    out = os.path.join(AE, "denoise_compare.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[out]", out)
    print("[summary]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
