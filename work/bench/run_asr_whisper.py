"""faster-whisper (CTranslate2) 转写基线：CPU int8，测模型体积/RTF/内存/文本。"""
import argparse
import json
import os
import re
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audiometrics import read_wav


def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024


def dir_size_mb(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=r"work\samples\mono_600_60.wav")
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=0, help="1=先做一次不计时的预热解码（GPU 稳态）")
    ap.add_argument("--download-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "whisper"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.makedirs(args.download_root, exist_ok=True)
    from faster_whisper import WhisperModel

    print(f"[env] faster_whisper device={args.device} compute_type={args.compute_type} "
          f"threads={args.threads} beam={args.beam} HF_ENDPOINT={os.environ.get('HF_ENDPOINT','')}")
    print(f"[mem] baseline RSS = {rss_mb():.0f} MB")
    mdir_before = set(os.listdir(args.download_root))

    t0 = time.perf_counter()
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type,
                             download_root=args.download_root, cpu_threads=args.threads,
                             num_workers=1)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"[FAIL] model load ({args.device}/{args.compute_type}): {msg}")
        print("[FAIL] " + msg[:1200].replace("\n", " | "))
        sys.exit(3)
    t_load = time.perf_counter() - t0
    mem_load = rss_mb()

    mdir = os.path.join(args.download_root, f"models--Systran--faster-whisper-{args.model}")
    size = dir_size_mb(mdir) if os.path.isdir(mdir) else dir_size_mb(args.download_root)
    nfiles = []
    for root, _, files in os.walk(mdir):
        for f in files:
            nfiles.append((os.path.join(root, f), os.path.getsize(os.path.join(root, f))))
    big = sorted(nfiles, key=lambda t: -t[1])[:3]
    print(f"[model] dir={mdir} size={size:.1f} MB files={len(nfiles)} biggest=" +
          ", ".join(f"{os.path.basename(p)}({s/1024/1024:.1f}MB)" for p, s in big))
    print(f"[load] {t_load:.2f}s  RSS={mem_load:.0f} MB")

    arr, sr = read_wav(args.wav)
    audio = arr[:, 0].astype(np.float32)
    dur = len(audio) / sr
    print(f"[audio] {args.wav} {sr} Hz {dur:.2f}s")

    peak = {"rss": mem_load}
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], rss_mb())
            time.sleep(0.05)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    if args.warmup:
        t_w = time.perf_counter()
        ws, _ = model.transcribe(audio[:sr * 5], language="zh", beam_size=args.beam,
                                 vad_filter=False, condition_on_previous_text=False)
        list(ws)
        print(f"[warmup] 5s 预热解码 {time.perf_counter()-t_w:.2f}s")
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio, language="zh", beam_size=args.beam, vad_filter=False,
        condition_on_previous_text=False, word_timestamps=False,
    )
    segs = []
    for s in segments:
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text})
    wall = time.perf_counter() - t0
    stop.set()
    th.join(timeout=1)
    text = "".join(s["text"] for s in segs).strip()
    rtf = wall / dur
    print(f"[time] wall={wall:.2f}s audio={dur:.1f}s RTF={rtf:.4f} "
          f"(1s 音频 {rtf*1000:.0f} ms; 78min -> {78*60*rtf/60:.2f} min)")
    print(f"[mem] peak RSS = {peak['rss']:.0f} MB (delta +{peak['rss']-mem_load:.0f} MB)")
    cn = len(re.sub(r"[^\u4e00-\u9fff]", "", text))
    print(f"[text] chars={cn}cn  segs={len(segs)}")
    print("[text] " + text[:600])
    print("[segs] " + json.dumps(segs[:4], ensure_ascii=False))
    res = {"model": args.model, "device": args.device, "compute_type": args.compute_type,
           "threads": args.threads, "beam": args.beam, "wav": args.wav, "dur": dur,
           "load_s": t_load, "wall_s": wall, "rtf": rtf, "peak_rss_mb": peak["rss"],
           "model_size_mb": size, "text": text, "segs": segs}
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"asr_whisper_{args.model}_{args.device}_{args.compute_type}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
