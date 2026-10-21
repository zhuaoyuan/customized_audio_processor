"""SenseVoiceSmall(ONNX, sherpa-onnx) 转写基线：测 RTF / 内存 / 文本。"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PROCESS_MEMORY_COUNTERS()
        c.cb = ctypes.sizeof(c)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                 ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model-dir", default=r"work\bench\models\sensevoice")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=float, default=0.0, help=">0 时按该秒数切块分别解码")
    ap.add_argument("--itn", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import sherpa_onnx

    model = os.path.join(args.model_dir, "model.int8.onnx")
    tokens = os.path.join(args.model_dir, "tokens.txt")
    print(f"[env] sherpa_onnx={sherpa_onnx.__version__ if hasattr(sherpa_onnx,'__version__') else '?'} "
          f"ort={getattr(sherpa_onnx,'ort_version','?')} model={model} "
          f"({os.path.getsize(model)/1024/1024:.1f} MB) threads={args.threads} itn={args.itn}")
    print(f"[mem] baseline RSS = {rss_mb():.0f} MB")

    t0 = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model, tokens=tokens, num_threads=args.threads,
        use_itn=bool(args.itn), language="zh", debug=False,
    )
    t_load = time.perf_counter() - t0
    mem_after_load = rss_mb()
    print(f"[load] recognizer init {t_load:.2f}s, RSS={mem_after_load:.0f} MB")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from audiometrics import read_wav
    (arr, sr) = read_wav(args.wav)
    samples = arr[:, 0].astype(np.float32)
    audio_dur = len(samples) / sr
    print(f"[audio] {args.wav} {sr} Hz {audio_dur:.2f}s")

    peak = {"rss": mem_after_load}
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], rss_mb())
            time.sleep(0.05)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    t0 = time.perf_counter()
    if args.chunk and args.chunk > 0:
        step = int(args.chunk * sr)
        texts = []
        total_decode = 0.0
        for i in range(0, len(samples), step):
            seg = samples[i:i + step]
            if len(seg) < sr * 0.2:
                continue
            s = recognizer.create_stream()
            s.accept_waveform(sr, seg)
            td = time.perf_counter()
            recognizer.decode_stream(s)
            total_decode += time.perf_counter() - td
            texts.append(s.result.text)
        text = "".join(texts)
        print(f"[decode] chunked {args.chunk}s, pure decode {total_decode:.2f}s")
    else:
        s = recognizer.create_stream()
        s.accept_waveform(sr, samples)
        recognizer.decode_stream(s)
        text = s.result.text
        total_decode = time.perf_counter() - t0
    t1 = time.perf_counter()
    stop.set()
    th.join(timeout=1)

    wall = t1 - t0
    rtf = wall / audio_dur
    print(f"[time] wall={wall:.2f}s (incl. stream setup)  pure_decode={total_decode:.2f}s  "
          f"audio={audio_dur:.1f}s")
    print(f"[rtf] RTF = {rtf:.4f}  (1s 音频 {rtf*1000:.0f} ms 处理; 78min -> {78*60*rtf/60:.2f} min)")
    print(f"[mem] peak RSS = {peak['rss']:.0f} MB (delta +{peak['rss']-mem_after_load:.0f} MB)")
    print("[text] " + text)
    res = {"wav": args.wav, "sr": sr, "dur": audio_dur, "load_s": t_load,
           "decode_s": total_decode, "wall_s": wall, "rtf": rtf,
           "peak_rss_mb": peak["rss"], "text": text, "threads": args.threads,
           "chunk": args.chunk, "itn": args.itn}
    out = args.out or (os.path.splitext(args.wav)[0] + f".sensevoice_t{args.threads}"
                       + (f"_chunk{int(args.chunk)}" if args.chunk else "") + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
