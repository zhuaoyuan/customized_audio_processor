"""Step 5: 时间戳精度验证 —— 以 faster-whisper 词级时间戳为参照，比较 VAD 段边界。

做法：
  * 取同一 16k wav 的 [start, start+span] 窗口（默认 600s）。
  * faster-whisper base（CPU int8, beam5, word_timestamps=True）解出词级 start/end。
  * 对窗口内每个 VAD 段，取"中点落在该段内的词"的首词 start / 末词 end 作参照边界，
    算 我们边界 - 参照边界 的有符号误差 / 绝对误差（median, P90）。
  * 同一窗口再用 whisper small 解一遍，按词序列对齐（difflib）比较两模型词边界差异
    —— 用来估计"参照本身的时间戳噪声"。
"""
import argparse
import difflib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def run_whisper(audio, model_name, threads, beam=5):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from faster_whisper import WhisperModel
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         download_root=common.WHISPER_ROOT, cpu_threads=threads,
                         num_workers=1)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio, language="zh", beam_size=beam, vad_filter=False,
        condition_on_previous_text=False, word_timestamps=True)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            txt = (w.word or "").strip()
            if txt:
                words.append({"start": round(float(w.start), 3),
                              "end": round(float(w.end), 3), "word": txt})
    wall = time.perf_counter() - t0
    return words, {"model": model_name, "load_s": round(t_load, 2), "wall_s": round(wall, 2),
                   "n_words": len(words)}


def stat(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)),
            "median": round(float(np.median(x)), 3),
            "abs_median": round(float(np.median(np.abs(x))), 3),
            "mean": round(float(np.mean(x)), 3),
            "p90_abs": round(float(np.percentile(np.abs(x), 90)), 3),
            "max_abs": round(float(np.max(np.abs(x))), 3)}


def eval_boundaries(segs, words, key_start, key_end, win_start, win_end):
    """对窗口内每个段：段边界 vs 中点落在段内的 whisper 词首/词尾。"""
    ds, de = [], []
    used = 0
    for s in segs:
        if s["start"] < win_start + 1.0 or s["end"] > win_end - 1.0:
            continue
        inside = [w for w in words if s["start"] <= (w["start"] + w["end"]) / 2 <= s["end"]]
        if not inside:
            continue
        ref_s = min(w["start"] for w in inside)
        ref_e = max(w["end"] for w in inside)
        ds.append(s[key_start] - ref_s)
        de.append(s[key_end] - ref_e)
        used += 1
    return {"n": used, "start_signed": stat(ds), "end_signed": stat(de),
            "start_abs": stat(np.abs(ds)), "end_abs": stat(np.abs(de))}


def cross_model(words_a, words_b):
    """两模型词级时间戳对齐后的差异（参照噪声估计）。"""
    ta = [w["word"] for w in words_a]
    tb = [w["word"] for w in words_b]
    sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    ds, de = [], []
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            wa = words_a[block.a + k]
            wb = words_b[block.b + k]
            ds.append(wa["start"] - wb["start"])
            de.append(wa["end"] - wb["end"])
    return {"matched_words": len(ds), "start": stat(ds), "end": stat(de)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=600.0)
    ap.add_argument("--span", type=float, default=600.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--ref-model", default="base")
    ap.add_argument("--noise-model", default="small")
    args = ap.parse_args()

    arr, sr = common.read_wav(args.wav)
    x = arr[:, 0].astype(np.float32)
    a = int(args.start * sr)
    b = min(len(x), int((args.start + args.span) * sr))
    win = x[a:b]
    win_start, win_end = args.start, b / sr
    print(f"[window] {win_start:.0f}-{win_end:.0f}s ({len(win)/sr:.0f}s)", flush=True)

    vad = common.read_json(args.vad)
    segs = vad["segments"]

    t0 = time.perf_counter()
    words_ref, run_ref = run_whisper(win, args.ref_model, args.threads, args.beam)
    print(f"[whisper-{args.ref_model}] load={run_ref['load_s']}s wall={run_ref['wall_s']}s "
          f"words={run_ref['n_words']}", flush=True)
    words_noise, run_noise = run_whisper(win, args.noise_model, args.threads, args.beam)
    print(f"[whisper-{args.noise_model}] load={run_noise['load_s']}s wall={run_noise['wall_s']}s "
          f"words={run_noise['n_words']}", flush=True)

    offset = np.array([win_start] * 1)
    words_ref = [{"start": w["start"] + win_start, "end": w["end"] + win_start,
                  "word": w["word"]} for w in words_ref]
    words_noise = [{"start": w["start"] + win_start, "end": w["end"] + win_start,
                    "word": w["word"]} for w in words_noise]

    res = {
        "wav": common.rel(args.wav), "vad": common.rel(args.vad),
        "window": [win_start, round(win_end, 1)],
        "whisper_ref": run_ref, "whisper_noise": run_noise,
        "n_vad_segments_in_window": sum(1 for s in segs
                                        if s["start"] >= win_start and s["end"] <= win_end),
        "boundaries": {},
        "reference_noise": cross_model(words_ref, words_noise),
        "whisper_words_ref_sample": words_ref[:40],
        "words_ref": words_ref,
        "words_noise": words_noise,
    }
    # 分别评估：精修后边界(vad_start/vad_end)、上报边界(start/end, 含 pad)
    for name, ks, ke in (("refined_10ms", "vad_start", "vad_end"),
                         ("reported_with_pad", "start", "end")):
        r = eval_boundaries(segs, words_ref, ks, ke, win_start, win_end)
        res["boundaries"][name] = r
        print(f"[{name}] n={r['n']} start: signed med={r['start_signed'].get('median')}s "
              f"abs med={r['start_abs'].get('median')}s p90={r['start_abs'].get('p90_abs')}s | "
              f"end: signed med={r['end_signed'].get('median')}s "
              f"abs med={r['end_abs'].get('median')}s p90={r['end_abs'].get('p90_abs')}s",
              flush=True)
    print(f"[ref-noise] {args.ref_model} vs {args.noise_model}: matched="
          f"{res['reference_noise']['matched_words']} start abs-med="
          f"{res['reference_noise']['start'].get('median')}s end abs-med="
          f"{res['reference_noise']['end'].get('median')}s", flush=True)
    common.write_json(args.out, res)
    print(f"[out] {common.rel(args.out)}")


if __name__ == "__main__":
    main()
