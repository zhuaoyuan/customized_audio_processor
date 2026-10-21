# -*- coding: utf-8 -*-
"""测试 Silero VAD 可用性：sherpa-onnx VoiceActivityDetector vs faster-whisper 内置 Silero。"""
import os
import sys
import time

import numpy as np
import scipy.io.wavfile as wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
MODEL = os.path.join(ROOT, "work", "audio-edit", "models", "silero_vad.onnx")
SAMPLE = os.path.join(ROOT, "work", "samples", "mono_600_60.wav")


def load(path):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.float32:
        pass
    else:
        raise ValueError(x.dtype)
    if x.ndim > 1:
        x = x[:, 0]
    return sr, x


def try_sherpa():
    import sherpa_onnx
    print("[sherpa-onnx]", sherpa_onnx.__version__ if hasattr(sherpa_onnx, "__version__") else "?")
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = MODEL
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.3
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.window_size = 512
    cfg.sample_rate = 16000
    cfg.num_threads = 2
    if not os.path.exists(MODEL):
        print("[sherpa-onnx] model missing:", MODEL)
        return None
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
    sr, x = load(SAMPLE)
    t0 = time.perf_counter()
    win = cfg.silero_vad.window_size
    for i in range(0, len(x) - win, win):
        vad.accept_waveform(x[i:i + win])
    while not vad.empty():
        vad.pop()
    t = time.perf_counter() - t0
    print(f"[sherpa-onnx] dur={len(x)/sr:.1f}s time={t:.3f}s rtf={t/(len(x)/sr):.4f}")
    segs = []
    while not vad.empty():
        s = vad.front
        segs.append((s.start / sr, s.start / sr + len(s.samples) / sr))
        vad.pop()
    print(f"[sherpa-onnx] segments={len(segs)} first10={[tuple(round(v,2) for v in s) for s in segs[:10]]}")
    return segs


def try_fw():
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    sr, x = load(SAMPLE)
    opts = VadOptions(threshold=0.5, min_speech_duration_ms=250,
                      min_silence_duration_ms=300, speech_pad_ms=0)
    t0 = time.perf_counter()
    ts = get_speech_timestamps(x, opts, sampling_rate=sr)
    t = time.perf_counter() - t0
    print(f"[faster-whisper] dur={len(x)/sr:.1f}s time={t:.3f}s rtf={t/(len(x)/sr):.4f} segs={len(ts)}")
    print("  first10=", [(round(s['start'] / sr, 2), round(s['end'] / sr, 2)) for s in ts[:10]])
    return ts


if __name__ == "__main__":
    print("model exists:", os.path.exists(MODEL))
    try:
        try_sherpa()
    except Exception as e:
        print("[sherpa-onnx] FAIL:", repr(e))
    try:
        try_fw()
    except Exception as e:
        print("[faster-whisper] FAIL:", repr(e))
