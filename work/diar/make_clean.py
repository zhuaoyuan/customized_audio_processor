"""根据 diarization 结果生成"只保留主讲人"的干净音轨 + 统计 + 抽检片段。

输出:
  clean_main.wav      全长同时间轴，非主讲人语音=静音（可 A/B 对齐）
  clean_main_cut.wav  只拼接主讲人段的紧凑版（时长变短）
  main_only_removed.wav 被去掉的部分（非主讲人语音），用于 ASR 看"丢了什么"
  stats.json / kept_regions.json
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
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767.0).astype(np.int16).tobytes())
    print(f"[out] {path}  {len(x)/sr:.1f}s  {os.path.getsize(path)/1024/1024:.1f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(here, "..", "full_16k.wav"))
    ap.add_argument("--diar", default=os.path.join(here, "diar_full.json"))
    ap.add_argument("--hop", type=float, default=0.01)
    ap.add_argument("--pad", type=float, default=0.15, help="主讲人段前后保护带（秒）")
    ap.add_argument("--fade", type=float, default=0.02)
    ap.add_argument("--gap", type=float, default=0.30, help="主讲人段之间小于该间隙则合并")
    ap.add_argument("--main", type=int, default=-1, help="指定主讲人 id；-1=按时长自动")
    ap.add_argument("--nonspeech-speakers", default="99",
                    help="逗号分隔的非语音标签（音乐/噪声），不计入其他说话人语音")
    ap.add_argument("--outdir", default=here)
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

    # --- 帧级占用统计（10ms）---
    hop = int(sr * args.hop)
    nf = (n + hop - 1) // hop
    per = {}
    for s in segs:
        a = int(round(s["start"] / args.hop))
        b = int(round(s["end"] / args.hop))
        a, b = max(0, a), min(nf, b)
        if b <= a:
            continue
        if s["speaker"] not in per:
            per[s["speaker"]] = np.zeros(nf, dtype=bool)
        per[s["speaker"]][a:b] = True
    total = np.zeros(nf, dtype=bool)
    for m in per.values():
        total |= m
    spk_time = {k: float(m.sum() * args.hop) for k, m in per.items()}
    main = args.main
    if main < 0:
        main = max(spk_time, key=spk_time.get)
    main_mask = per.get(main, np.zeros(nf, dtype=bool))
    ns_labels = [int(v) for v in args.nonspeech_speakers.split(",") if v.strip() != ""]
    ns_mask = np.zeros(nf, dtype=bool)
    for k in ns_labels:
        if k in per:
            ns_mask |= per[k]
    other_mask = total & ~main_mask & ~ns_mask
    overlap = sum(m.sum() for m in per.values()) * args.hop - total.sum() * args.hop

    # --- 主讲人段合并成区间（允许小间隙）---
    idx = np.flatnonzero(main_mask)
    regions = []
    if len(idx):
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if (i - prev) * args.hop > args.gap:
                regions.append([start * args.hop, (prev + 1) * args.hop])
                start = i
            prev = i
        regions.append([start * args.hop, (prev + 1) * args.hop])
    # 两侧 pad
    regions = [[max(0.0, a - args.pad), min(dur, b + args.pad)] for a, b in regions]

    keep = np.zeros(n, dtype=bool)
    hop_s = int(round(sr * args.hop))
    for a, b in regions:
        keep[int(a * sr):min(n, int(b * sr))] = True

    # 保留区内的非主讲人语音（pad 造成的溢出）
    spill = keep & np.repeat(other_mask, hop_s)[:n]
    # 用帧掩码判断
    fm = np.repeat(other_mask, hop_s)[:n]
    spill_s = float((keep & fm).sum()) / sr

    removed = ~keep
    # --- 写干净音轨（非主讲人静音）---
    y = x.copy()
    y[removed] = 0.0
    # 每个保留区间两端做线性淡入/淡出，避免爆音
    nfade = int(args.fade * sr)
    for a, b in regions:
        ia, ib = int(a * sr), min(n, int(b * sr))
        nf_ = min(nfade, max(1, (ib - ia) // 2))
        if nf_ > 1:
            y[ia:ia + nf_] *= np.linspace(0.0, 1.0, nf_, dtype=np.float32)
            y[ib - nf_:ib] *= np.linspace(1.0, 0.0, nf_, dtype=np.float32)

    write_wav(os.path.join(args.outdir, "clean_main.wav"), y, sr)

    # 紧凑版（拼接）
    parts = [y[int(a * sr):min(n, int(b * sr))] for a, b in regions]
    ycut = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    write_wav(os.path.join(args.outdir, "clean_main_cut.wav"), ycut, sr)

    # 被去掉的内容（全长，主讲人处静音）
    z = x.copy()
    z[keep] = 0.0
    write_wav(os.path.join(args.outdir, "main_only_removed.wav"), z, sr)

    kept_dur = float(keep.sum()) / sr
    stats = {
        "wav": wav_path, "dur": dur, "sr": sr, "diar": diar_path,
        "n_speakers": len(per), "main_speaker": main,
        "main_speech_s": spk_time.get(main, 0.0),
        "main_speech_ratio_file": spk_time.get(main, 0.0) / dur,
        "other_speech_s": float(other_mask.sum() * args.hop),
        "other_speech_ratio_file": float(other_mask.sum() * args.hop) / dur,
        "nonspeech_s": float(ns_mask.sum() * args.hop),
        "nonspeech_ratio_file": float(ns_mask.sum() * args.hop) / dur,
        "all_speech_s": float(total.sum() * args.hop),
        "all_speech_ratio_file": float(total.sum() * args.hop) / dur,
        "unlabeled_s": float((~total).sum() * args.hop),
        "nonspeech_labels": ns_labels,
        "overlap_s": float(overlap),
        "n_speakers_ge_10s": sum(1 for v in spk_time.values() if v >= 10),
        "per_speaker_s": {str(k): round(v, 2) for k, v in sorted(spk_time.items(), key=lambda x: -x[1])},
        "kept_regions": len(regions),
        "kept_dur_s": kept_dur, "kept_ratio": kept_dur / dur,
        "removed_dur_s": dur - kept_dur, "removed_ratio": 1 - kept_dur / dur,
        "pad_s": args.pad, "gap_s": args.gap,
        "other_speech_inside_kept_s": spill_s,
        "cut_dur_s": len(ycut) / sr,
    }
    with open(os.path.join(args.outdir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.outdir, "kept_regions.json"), "w", encoding="utf-8") as f:
        json.dump({"main_speaker": main, "pad": args.pad,
                   "regions": [[round(a, 3), round(b, 3)] for a, b in regions]},
                  f, ensure_ascii=False)

    print(f"[stats] spk={len(per)} main=S{main} main_speech={stats['main_speech_s']:.0f}s "
          f"({stats['main_speech_ratio_file']*100:.1f}%)")
    print(f"[stats] all_speech={stats['all_speech_s']:.0f}s ({stats['all_speech_ratio_file']*100:.1f}%) "
          f"other_speech={stats['other_speech_s']:.0f}s ({stats['other_speech_ratio_file']*100:.1f}%) "
          f"nonspeech={stats['nonspeech_s']:.0f}s ({stats['nonspeech_ratio_file']*100:.1f}%) "
          f"unlabeled={stats['unlabeled_s']:.0f}s overlap={stats['overlap_s']:.0f}s")
    print(f"[stats] kept_regions={len(regions)} kept={kept_dur:.0f}s ({kept_dur/dur*100:.1f}%) "
          f"removed={dur-kept_dur:.0f}s ({(1-kept_dur/dur)*100:.1f}%) other_speech_inside_kept={spill_s:.1f}s")
    print(json.dumps({k: stats[k] for k in ("n_speakers", "main_speaker", "main_speech_s",
                                            "other_speech_s", "kept_regions", "kept_dur_s")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
