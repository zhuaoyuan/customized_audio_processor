# -*- coding: utf-8 -*-
"""降噪对比：afftdn / arnndn 在 5 分钟样本上的 RTF 与语音段/非语音段指标。"""
import os
import subprocess
import sys
import time

import numpy as np
import scipy.io.wavfile as wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AE = os.path.join(ROOT, "work", "audio-edit")
SRC = os.path.join(ROOT, "work", "samples", "mono_600_60.wav")
LONG = os.path.join(AE, "samples", "test_5min.wav")
RNN = os.path.join(ROOT, "work", "bench", "models", "rnnoise")


def run_ff(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        print("FFMPEG FAIL", p.returncode, p.stderr[-1500:])
    return p


def load(path):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x[:, 0]
    return sr, x.astype(np.float32)


def frames(x, sr, win=0.02, hop=0.01):
    w, h = int(sr * win), int(sr * hop)
    n = 1 + max(0, (len(x) - w)) // h
    idx = np.arange(n) * h
    view = np.lib.stride_tricks.sliding_window_view(x, w)[::h]
    return view, idx


def db(p):
    return 10 * np.log10(np.maximum(p, 1e-12))


def main():
    # 5min 样本：取 data/24 的原解码 16k 文件 600-900s
    sr, xfull = load(os.path.join(ROOT, "work", "full_16k.wav"))
    seg = xfull[600 * sr:900 * sr]
    wavfile.write(LONG, sr, (seg * 32767).astype(np.int16))
    print(f"[src] 5min sample {len(seg)/sr:.1f}s sr={sr}")

    outs = {}
    # 默认链
    out1 = os.path.join(AE, "samples", "test_5min_afftdn.wav")
    t0 = time.perf_counter()
    run_ff(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", LONG,
            "-af", "highpass=f=80,afftdn=nr=12:nf=-30:tn=1",
            "-c:a", "pcm_s16le", out1])
    t1 = time.perf_counter() - t0
    print(f"[afftdn] {t1:.2f}s rtf={t1/len(seg)*sr:.4f}")
    outs["afftdn"] = out1

    # arnndn
    out2 = os.path.join(AE, "samples", "test_5min_arnndn.wav")
    t0 = time.perf_counter()
    p = run_ff(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", LONG,
                "-af", "highpass=f=80,arnndn=m=sh.rnnn",
                "-c:a", "pcm_s16le", out2])
    t1 = time.perf_counter() - t0
    if os.path.exists(out2) and os.path.getsize(out2) > 1000:
        print(f"[arnndn] {t1:.2f}s rtf={t1/len(seg)*sr:.4f}")
        outs["arnndn"] = out2
    else:
        print("[arnndn] failed, skip")

    # 只 highpass 对照
    out3 = os.path.join(AE, "samples", "test_5min_hp.wav")
    t0 = time.perf_counter()
    run_ff(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", LONG,
            "-af", "highpass=f=80", "-c:a", "pcm_s16le", out3])
    t1 = time.perf_counter() - t0
    print(f"[highpass] {t1:.2f}s rtf={t1/len(seg)*sr:.4f}")
    outs["highpass"] = out3

    # 用 faster-whisper VAD 在原 5min 上取语音/非语音帧
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(seg, VadOptions(threshold=0.5, min_speech_duration_ms=250,
                                               min_silence_duration_ms=300, speech_pad_ms=0),
                               sampling_rate=sr)
    sp_mask = np.zeros(len(seg), dtype=bool)
    for s in ts:
        sp_mask[s["start"]:s["end"]] = True
    print(f"[vad] segs={len(ts)} speech={sp_mask.mean()*100:.1f}%")

    ids = {"orig": seg}
    for k, pth in outs.items():
        _, y = load(pth)
        if len(y) > len(seg):
            y = y[:len(seg)]
        ids[k] = y

    print(f"\n{'ver':<10}{'speech RMS dB':>14}{'nonspeech RMS dB':>18}{'speech->orig':>14}")
    base_sp = None
    for k, y in ids.items():
        sp = 10 * np.log10(np.mean(y[sp_mask] ** 2) + 1e-12)
        ns = 10 * np.log10(np.mean(y[~sp_mask] ** 2) + 1e-12)
        if k == "orig":
            base_sp = sp
        print(f"{k:<10}{sp:>14.2f}{ns:>18.2f}{sp-base_sp:>14.2f}")


if __name__ == "__main__":
    main()
