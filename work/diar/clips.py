"""在"非主讲人语音最多"的时间窗生成 20s 切前/切后抽检片段，并对片段做 ASR（代替人工试听）。

输出:
  clips/clip_{i}_orig_{t}s.wav   原始混音 20s
  clips/clip_{i}_clean_{t}s.wav  干净音轨同窗口 20s
  clips/clip_{i}_removed_{t}s.wav 被切除部分 20s
  clips/clips.json               含每对的 ASR 文本
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(here, "..", "full_16k.wav"))
    ap.add_argument("--clean", default=os.path.join(here, "clean_main.wav"))
    ap.add_argument("--removed", default=os.path.join(here, "main_only_removed.wav"))
    ap.add_argument("--diar", default=os.path.join(here, "diar_merged_th0.7_diar.json"))
    ap.add_argument("--main", type=int, default=0)
    ap.add_argument("--win", type=float, default=20.0)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--min-spacing", type=float, default=120.0)
    ap.add_argument("--exclude-speakers", default="99",
                    help="打分时排除的标签（非语音簇），避免抽到音乐片段")
    ap.add_argument("--windows", default="", help="逗号分隔的显式窗口起点（秒），给了就忽略自动挑选")
    ap.add_argument("--outdir", default=os.path.join(here, "clips"))
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    wav_path = os.path.normpath(args.wav if os.path.isabs(args.wav)
                                else os.path.join(here, "..", "..", args.wav))
    diar_path = os.path.normpath(args.diar if os.path.isabs(args.diar)
                                 else os.path.join(here, "..", "..", args.diar))
    arr, sr = read_wav(wav_path)
    x = arr[:, 0].astype(np.float32)
    n = len(x)
    dur = n / sr
    segs = json.load(open(diar_path, encoding="utf-8"))["segments"]

    hop = int(sr * 0.01)
    nf = (n + hop - 1) // hop
    excl = {int(v) for v in args.exclude_speakers.split(",") if v.strip() != ""}
    other = np.zeros(nf, dtype=bool)
    for s in segs:
        if s["speaker"] == args.main or s["speaker"] in excl:
            continue
        a, b = max(0, int(round(s["start"] / 0.01))), min(nf, int(round(s["end"] / 0.01)))
        other[a:b] = True
    # 平滑：只统计连续 >=0.3s 的其他说话人语音（滤掉零碎误报）
    from scipy.ndimage import binary_opening
    other = binary_opening(other, structure=np.ones(30, dtype=bool))

    step = int(args.win / 0.01)
    hstep = int(args.hop / 0.01)
    if args.windows:
        picked = [(0.0, float(v)) for v in args.windows.split(",")]
    else:
        cands = []
        for a in range(0, nf - step, hstep):
            score = other[a:a + step].sum() * 0.01
            if score > 0.5:
                cands.append((score, a * 0.01))
        cands.sort(reverse=True)
        picked = []
        for score, t in cands:
            if all(abs(t - p) >= args.min_spacing for _, p in picked):
                picked.append((score, t))
            if len(picked) >= args.n:
                break
        picked.sort(key=lambda z: z[1])

    clean_path = os.path.normpath(args.clean if os.path.isabs(args.clean)
                                  else os.path.join(here, "..", "..", args.clean))
    rem_path = os.path.normpath(args.removed if os.path.isabs(args.removed)
                                else os.path.join(here, "..", "..", args.removed))
    yc, _ = read_wav(clean_path)
    yc = yc[:, 0]
    has_rem = os.path.exists(rem_path)
    yr = read_wav(rem_path)[0][:, 0] if has_rem else None
    os.makedirs(args.outdir, exist_ok=True)

    rec = None
    if not args.no_asr:
        import sherpa_onnx
        md = os.path.join(here, "..", "bench", "models", "sensevoice")
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=os.path.normpath(os.path.join(md, "model.int8.onnx")),
            tokens=os.path.normpath(os.path.join(md, "tokens.txt")),
            num_threads=args.threads, use_itn=True, language="zh", debug=False)

    def asr(y):
        if rec is None:
            return ""
        s = rec.create_stream()
        s.accept_waveform(sr, y)
        rec.decode_stream(s)
        return s.result.text.strip()

    clips = []
    for i, (score, t) in enumerate(picked, 1):
        a = int(t * sr)
        b = min(n, a + int(args.win * sr))
        name = f"clip_{i}_{int(t)}s"
        xo = x[a:b].copy()
        xcl = yc[a:b].copy() if len(yc) >= b else np.zeros(b - a, dtype=np.float32)
        write_wav(os.path.join(args.outdir, f"{name}_orig.wav"), xo, sr)
        write_wav(os.path.join(args.outdir, f"{name}_clean.wav"), xcl, sr)
        row = {"i": i, "start": t, "end": round(b / sr, 1),
               "other_speech_in_win": round(float(score), 2),
               "text_orig": asr(xo), "text_clean": asr(xcl)}
        if yr is not None:
            xr = yr[a:b].copy()
            write_wav(os.path.join(args.outdir, f"{name}_removed.wav"), xr, sr)
            row["text_removed"] = asr(xr)
        clips.append(row)
        print(f"[clip{i}] {t:.0f}-{b/sr:.0f}s other={score:.1f}s", flush=True)
        print(f"   orig   : {row['text_orig'][:120]}", flush=True)
        print(f"   clean  : {row['text_clean'][:120]}", flush=True)
        if "text_removed" in row:
            print(f"   removed: {row['text_removed'][:120]}", flush=True)

    with open(os.path.join(args.outdir, "clips.json"), "w", encoding="utf-8") as f:
        json.dump({"win": args.win, "main": args.main, "clips": clips}, f, ensure_ascii=False, indent=1)
    print(f"[out] {os.path.join(args.outdir, 'clips.json')}")


if __name__ == "__main__":
    main()
