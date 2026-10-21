"""Step 4: 合并 VAD/ASR/说话人 → transcript.json / transcript.md / transcript.srt + 统计。

用法:
  python -X utf8 step_build.py --short 24_形而上学 --vad vad.json --asr asr.json \
      --diar-merged merge.json --outdir work/timeline/24_形而上学 [--mapping mapping.json]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CN_RE = re.compile(r"[\u4e00-\u9fff]")
CONTENT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def has_content(text):
    return bool(CONTENT_RE.search(text or ""))


# ---------------------------------------------------------------------------
# 可选：work/audio-edit/<短名>/mapping.json 的 orig<->clean 时间映射
# 兼容若干常见字段名；识别失败时返回 None 并在报告里注明。
# ---------------------------------------------------------------------------
def load_mapping(path):
    if not path or not os.path.exists(path):
        return None
    try:
        d = common.read_json(path)
    except Exception:
        return None
    cand = None
    if isinstance(d, list):
        cand = d
    elif isinstance(d, dict):
        for key in ("kept", "segments", "chunks", "map", "mapping", "pieces", "items", "regions"):
            if isinstance(d.get(key), list):
                cand = d[key]
                break
        if cand is None and all(k in d for k in ("orig_start", "clean_start")):
            cand = [d]
    if not cand:
        return None
    rows = []
    for it in cand:
        # work/audio-edit 约定：kept 每项 = [orig_start, orig_end, clean_start, clean_end]
        if isinstance(it, (list, tuple)) and len(it) >= 4:
            try:
                rows.append((float(it[2]), float(it[3]), float(it[0]), float(it[1])))
            except (TypeError, ValueError):
                continue
            continue
        if not isinstance(it, dict):
            continue
        g = lambda *names: next((float(it[n]) for n in names if n in it), None)  # noqa: E731
        os_ = g("orig_start", "original_start", "src_start", "start_orig")
        oe = g("orig_end", "original_end", "src_end", "end_orig")
        cs = g("clean_start", "dst_start", "start_clean", "clean_from")
        ce = g("clean_end", "dst_end", "end_clean", "clean_to")
        if None in (os_, oe, cs, ce):
            os_, oe = g("start"), g("end")
            cs, ce = g("clean_start", "new_start"), g("clean_end", "new_end")
        if None in (os_, oe, cs, ce):
            continue
        rows.append((cs, ce, os_, oe))
    if not rows:
        return None
    rows.sort()
    return {"rows": rows, "path": path}


def clean_to_orig(mp, t):
    if not mp:
        return None
    rows = mp["rows"]
    for cs, ce, os_, oe in rows:
        if cs <= t <= ce:
            if ce - cs <= 0:
                return os_
            return os_ + (t - cs) * (oe - os_) / (ce - cs)
    # 落在被删区间：吸附到最近映射端点
    if t < rows[0][0]:
        return rows[0][2]
    if t > rows[-1][1]:
        return rows[-1][3]
    best = min(rows, key=lambda r: min(abs(t - r[0]), abs(t - r[1])))
    return best[2] if abs(t - best[0]) <= abs(t - best[1]) else best[3]


# ---------------------------------------------------------------------------
def assign_speakers(asr_segs, diar_segs, merge_th=3.0, min_overlap_ratio=0.30):
    """给每个 ASR(VAD) 段打说话人：取重叠时长最大的 diar 簇；不确定→主讲(0)。

    返回 (spk_ids, info)：spk_ids 是 merge 后的簇 id（0=主讲，其余按时长），
    info 记录保守规则命中数。
    """
    # 簇时长 -> 保守标签规则（<3s 的碎片簇不造新身份）
    dur = {}
    for s in diar_segs:
        dur[int(s["speaker"])] = dur.get(int(s["speaker"]), 0.0) + s["dur"]
    main = max(dur, key=dur.get) if dur else 0
    big = sorted([k for k, v in dur.items() if k != main and v >= merge_th],
                 key=lambda k: -dur[k])
    label = {main: "主讲"}
    for i, k in enumerate(big):
        label[k] = f"其他{i + 1}"
    result = []
    stats = {"no_overlap": 0, "low_overlap_to_main": 0, "short_fragment": 0}
    for a in asr_segs:
        s, e = a["start"], a["end"]
        ov = {}
        for d in diar_segs:
            o = min(e, d["end"]) - max(s, d["start"])
            if o > 0:
                k = int(d["speaker"])
                ov[k] = ov.get(k, 0.0) + o
        if not ov:
            # 无重叠：按中点找最近的 diar 段
            mid = (s + e) / 2
            if diar_segs:
                nearest = min(diar_segs, key=lambda d: min(abs(mid - d["start"]), abs(mid - d["end"])))
                k = int(nearest["speaker"])
            else:
                k = main
            stats["no_overlap"] += 1
            conf = 0.0
        else:
            k = max(ov, key=ov.get)
            conf = ov[k] / max(e - s, 1e-6)
        spk = k
        if conf < min_overlap_ratio:
            spk = main
            stats["low_overlap_to_main"] += 1
        if label.get(spk) is None:
            # 碎片簇 → 保守标主讲
            spk = main
            stats["short_fragment"] += 1
        result.append({"speaker": label[spk], "cluster": spk, "conf": round(conf, 3)})
    return result, stats, dur, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--diar-merged", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mapping", default="")
    ap.add_argument("--time-base", default="original", choices=["original", "clean"])
    ap.add_argument("--fragment-th-s", type=float, default=3.0)
    ap.add_argument("--para-gap", type=float, default=2.0)
    ap.add_argument("--para-max-s", type=float, default=75.0)
    ap.add_argument("--para-max-chars", type=int, default=220)
    args = ap.parse_args()

    vad = common.read_json(args.vad)
    asr = common.read_json(args.asr)
    segs = asr["segments"]
    mp = load_mapping(args.mapping)
    mp_ok = mp is not None

    if args.diar_merged:
        diar = common.read_json(args.diar_merged)
        diar_segs = diar["segments"]
        spk_info, spk_stats, spk_dur, spk_label = assign_speakers(
            segs, diar_segs, merge_th=args.fragment_th_s)
        raw_speakers = diar.get("raw_speakers")
        merged_speakers = diar.get("n_speakers")
        merge_sweep = diar.get("sweep", [])
        cluster_rep = diar.get("clusters", [])
    else:
        spk_info = [{"speaker": "主讲", "cluster": 0, "conf": 1.0} for _ in segs]
        spk_stats, spk_dur, spk_label = {}, {}, {0: "主讲"}
        raw_speakers = merged_speakers = None
        merge_sweep, cluster_rep = [], []

    out_segs = []
    for i, (a, si) in enumerate(zip(segs, spk_info)):
        # 上报边界优先用 VAD 精修边界（vs 词级参照误差更小，见 docs/research/timeline-poc.md §3）
        s = float(a.get("vad_start", a["start"]))
        e = float(a.get("vad_end", a["end"]))
        rec = {"idx": i + 1, "start": round(s, 3), "end": round(e, 3),
               "speaker": si["speaker"], "text": a["text"],
               "spk_cluster": si["cluster"], "spk_conf": si["conf"]}
        if mp_ok:
            rec["orig_start"] = round(clean_to_orig(mp, s), 3)
            rec["orig_end"] = round(clean_to_orig(mp, e), 3)
        out_segs.append(rec)

    # ---- md：同说话人连续段合并成段（gap <= para_gap） ----
    no_content = [r for r in out_segs if not has_content(r["text"])]
    paras = []
    for r in out_segs:
        if not has_content(r["text"]):
            continue
        if (paras and paras[-1]["speaker"] == r["speaker"]
                and r["start"] - paras[-1]["end"] <= args.para_gap
                and r["end"] - paras[-1]["start"] <= args.para_max_s
                and len(paras[-1]["text"]) + len(r["text"]) <= args.para_max_chars):
            paras[-1]["end"] = r["end"]
            paras[-1]["text"] += r["text"]
            if mp_ok:
                paras[-1]["orig_end"] = r["orig_end"]
        else:
            paras.append({"start": r["start"], "end": r["end"], "speaker": r["speaker"],
                          "text": r["text"], "orig_start": r.get("orig_start")})
    md = [f"# {args.short} · 句级时间戳转写（VAD+ASR）", ""]
    if mp_ok:
        md.append(f"> 时间轴：clean 音轨；括号内为原始音轨时间。映射文件 `{common.rel(args.mapping)}`。")
    else:
        md.append("> 时间轴：原始音频（未使用降噪/裁剪版；work/audio-edit 产物不存在或未指定）。")
    md.append(f"> 说话人：{spk_label and ' / '.join(sorted(set(spk_label.values()), key=lambda x: (x != '主讲', x)))}")
    md.append("")
    for p in paras:
        t = common.fmt_mmss(p["start"])
        if mp_ok and p.get("orig_start") is not None:
            t = f"{t} (orig {common.fmt_mmss(p['orig_start'])})"
        md.append(f"[{t}] {p['speaker']}：{p['text']}")
    md_text = "\n".join(md) + "\n"

    # ---- srt：一句一条 ----
    srt = []
    n = 0
    for r in out_segs:
        if not has_content(r["text"]):
            continue
        n += 1
        srt.append(str(n))
        srt.append(f"{common.fmt_srt(r['start'])} --> {common.fmt_srt(max(r['end'], r['start'] + 0.3))}")
        srt.append(f"{r['speaker']}：{r['text']}")
        srt.append("")
    srt_text = "\n".join(srt)

    outdir = common.ensure_dir(args.outdir)
    common.write_json(os.path.join(outdir, "transcript.json"),
                      {"short": args.short, "source": {
                          "vad": common.rel(args.vad), "asr": common.rel(args.asr),
                          "diar_merged": common.rel(args.diar_merged) if args.diar_merged else None,
                          "audio": vad.get("wav"), "time_base": args.time_base,
                          "mapping": common.rel(args.mapping) if mp_ok else None},
                       "segments": out_segs})
    with open(os.path.join(outdir, "transcript.md"), "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(os.path.join(outdir, "transcript.srt"), "w", encoding="utf-8") as f:
        f.write(srt_text)

    # ---- 统计 ----
    dur = vad["dur"]
    cn = sum(len(CN_RE.findall(r["text"])) for r in out_segs)
    total_chars = sum(len(r["text"]) for r in out_segs)
    by = {}
    for r in out_segs:
        b = by.setdefault(r["speaker"], {"segs": 0, "dur": 0.0, "cn": 0, "texts": []})
        b["segs"] += 1
        b["dur"] += r["end"] - r["start"]
        b["cn"] += len(CN_RE.findall(r["text"]))
        b["texts"].append(r["text"])
    speech_s = sum(b["dur"] for b in by.values())
    stats = {
        "short": args.short,
        "n_segments": len(out_segs),
        "n_with_text": sum(1 for r in out_segs if has_content(r["text"])),
        "n_no_content": len(no_content),
        "n_chars_cn": cn,
        "n_chars_total": total_chars,
        "audio_dur_s": dur,
        "seg_cover_s": round(speech_s, 1),
        "seg_cover_ratio_of_file": round(speech_s / dur, 4),
        "avg_seg_s": round(speech_s / max(len(out_segs), 1), 3),
        "median_seg_s": round(float(np.median([r["end"] - r["start"] for r in out_segs])), 3),
        "n_paras": len(paras),
        "asr": {"load_s": asr.get("load_s"), "wall_s": asr.get("wall_s"),
                "decode_s": asr.get("decode_s"), "rtf": asr.get("rtf"),
                "rtf_decode": asr.get("rtf_decode"), "threads": asr.get("threads")},
        "vad": {"n_segments": vad.get("n_segments"), "speech_ratio": vad.get("speech_ratio"),
                "timings": vad.get("timings"), "params": vad.get("params")},
        "diar": {"raw_speakers": raw_speakers, "merged_speakers": merged_speakers,
                 "merge_sweep": merge_sweep, "clusters": cluster_rep},
        "speaker_stats": {k: {"segs": v["segs"], "dur_s": round(v["dur"], 1),
                              "ratio_of_speech": round(v["dur"] / speech_s, 4),
                              "chars_cn": v["cn"]} for k, v in by.items()},
        "assign_stats": spk_stats,
        "main_ratio_of_speech": round(by.get("主讲", {}).get("dur", 0.0) / speech_s, 4),
        "n_other_labels": len([k for k in by if k.startswith("其他")]),
        "map_applied": mp_ok,
    }
    common.write_json(os.path.join(outdir, "stats.json"), stats)

    print(f"[build] segs={stats['n_segments']} chars_cn={cn} paras={len(paras)} "
          f"avg_seg={stats['avg_seg_s']}s main={stats['main_ratio_of_speech']*100:.1f}%")
    for k, v in sorted(by.items(), key=lambda z: -z[1]["dur"]):
        print(f"  {k}: segs={v['segs']} dur={v['dur']:.0f}s "
              f"({v['dur']/speech_s*100:.1f}%) chars={v['cn']}")
    print(f"[out] {common.rel(os.path.join(outdir, 'transcript.json/.md/.srt'))}")


if __name__ == "__main__":
    main()
