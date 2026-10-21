# -*- coding: utf-8 -*-
"""单文件音频编辑管线：降噪 -> Silero VAD -> 裁剪(紧凑时间轴+mapping) -> V1/V2/V3 音量归一化 -> limiter。

用法（全量）:
  python run_one.py --short xes --mode afftdn
用法（5 分钟试跑）:
  python run_one.py --short xes --start 600 --end 900 --outdir <dir> --pilot
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (AE, ROOT, SR, active_speech_level, apply_keep, asl_from_power,
                    build_keep_intervals, classify_segments, db, denoise_file, detect_speech,
                    dump_json, fluctuation, frame_mask_from_intervals, frame_power,
                    limiter_file, map_orig_to_clean, mapping_from_kept, measure_r128,
                    power_percentiles, read_wav, speech_frame_db_percentiles,
                    window_levels, window_levels_p, write_wav)
import speakers

BLOCK_MS = 20


# ------------------------------------------------------------------ 增益包络
def env_from_block_gains(gb, n_samples, sr=SR, block_ms=BLOCK_MS):
    """把 20ms 块增益变成样本级包络：块内从前一块增益余弦交叉淡化到本块增益。"""
    block = int(round(sr * block_ms / 1000))
    nb = len(gb)
    prev = np.empty(nb, dtype=np.float64)
    prev[0] = gb[0]
    prev[1:] = gb[:-1]
    w = 0.5 - 0.5 * np.cos(np.pi * (np.arange(block) + 0.5) / block)
    base = np.repeat(prev, block)
    delta = np.repeat(np.asarray(gb, dtype=np.float64) - prev, block)
    env = base + delta * np.tile(w, nb)
    return env[:n_samples].astype(np.float32)


def smooth_arr(x, k):
    if k <= 1 or len(x) < k:
        return x
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def measure_region(x, sr, mask, p=None):
    if p is None:
        p, _ = frame_power(x, sr)
    n = min(len(p), len(mask))
    ps = p[:n][mask[:n]]
    asl, na, ns = asl_from_power(ps)
    return {"speech_asl_dbfs": round(float(asl), 3), "n_active_frames": na,
            "n_speech_frames": ns, "speech_sec": round(ns * 0.01, 2),
            "percentile": power_percentiles(ps),
            "window5min": fluctuation(window_levels_p(p, mask, 300.0)[1])}


# ------------------------------------------------------------------ V1 / V2 / V3
def build_v2(x, sr, mask, base_gain, T, p=None, win_s=20.0, hop_s=5.0, clamp_db=3.0,
             max_rate=1.5, block_ms=BLOCK_MS, verbose=False):
    if p is None:
        p, _ = frame_power(x, sr)
    n_all = min(len(p), len(mask))
    dur = len(x) / sr
    centers, gains = [], []
    t = 0.0
    while t < dur - 1e-6:
        e = min(dur, t + win_s)
        a, b = int(round(t * 100)), min(int(round(e * 100)), n_all)
        a = min(a, n_all)
        ps = p[a:b][mask[a:b]]
        if len(ps) >= 300:  # 窗口内 >=3s 语音帧才更新
            lv, _, _ = asl_from_power(ps)
            centers.append((t + e) / 2)
            gains.append(float(np.clip(T - lv, base_gain - clamp_db, base_gain + clamp_db)))
        t += hop_s
    info = {"win_s": win_s, "hop_s": hop_s, "clamp_db": clamp_db,
            "max_rate_db_s": max_rate, "n_windows_used": len(centers)}
    if len(centers) < 2:
        env = np.full(len(x), float(base_gain), dtype=np.float32)
        info.update({"degraded": True, "gain_min": float(base_gain), "gain_max": float(base_gain)})
        return env, info
    centers = np.array(centers)
    gains = np.array(gains)
    if len(gains) >= 3:
        gains = smooth_arr(gains, 3)
    maxd = max_rate * hop_s
    g = gains.copy()
    for i in range(1, len(g)):
        g[i] = np.clip(g[i], g[i - 1] - maxd, g[i - 1] + maxd)
    block = int(round(sr * block_ms / 1000))
    nb = int(np.ceil(len(x) / block))
    bc = (np.arange(nb) * block + block / 2.0) / sr
    gb = np.interp(bc, centers, g, left=g[0], right=g[-1])
    gb = smooth_arr(gb, max(1, int(round(1.0 / (block_ms / 1000.0)))))  # 1s 平滑
    gb = np.clip(gb, base_gain - clamp_db, base_gain + clamp_db)
    env = env_from_block_gains(gb, len(x), sr, block_ms)
    rate = float(np.max(np.abs(np.diff(env.astype(np.float64)))) * sr) if len(env) > 1 else 0.0
    info.update({"degraded": False, "gain_min": round(float(env.min()), 2),
                 "gain_max": round(float(env.max()), 2),
                 "gain_mean": round(float(env.mean()), 2),
                 "max_actual_rate_db_s": round(rate, 2),
                 "control_centers_s": [round(float(c), 1) for c in centers[:5]]})
    return env, info


def build_v3(x, sr, mask, segs_clean, seg_labels, T, p=None, block_ms=BLOCK_MS,
             min_group_s=5.0, verbose=False):
    if p is None:
        p, _ = frame_power(x, sr)
    n_frames = int(np.ceil(len(x) / (sr * 0.01)))
    flab = np.zeros(n_frames, dtype=np.int16)
    for (s, e), lab in zip(segs_clean, seg_labels):
        a = max(0, int(round(s / 0.01)))
        b = min(n_frames, int(round(e / 0.01)))
        if b > a:
            flab[a:b] = int(lab)
    mask = mask[:n_frames]
    groups = sorted(set(int(v) for v in flab[mask]))
    asl_g, gain_g, detail = {}, {}, {}
    for gid in groups:
        m = mask & (flab == gid)
        ps = p[:n_frames][m]
        asl, na, ns = asl_from_power(ps)
        detail[str(gid)] = {"speech_sec": round(ns * 0.01, 2),
                            "asl_dbfs": round(float(asl), 2)}
        if ns * 0.01 >= min_group_s:
            gain_g[gid] = float(T - asl)
            asl_g[gid] = float(asl)
    if 0 not in gain_g:
        # 主讲帧不足（异常），退化 V1
        ps = p[:n_frames][mask]
        base = float(T - asl_from_power(ps)[0])
        env = np.full(len(x), base, dtype=np.float32)
        return env, {"degraded": True, "reason": "main_speaker_too_short",
                     "groups": detail, "gain": base}
    main_gain = gain_g[0]
    for gid in groups:
        if gid not in gain_g:
            gain_g[gid] = main_gain
    block = int(round(sr * block_ms / 1000))
    nb = int(np.ceil(len(x) / block))
    idx = np.clip(((np.arange(nb) * block + block // 2) / 0.01).astype(np.int64), 0, n_frames - 1)
    gb = np.array([gain_g.get(int(v), main_gain) for v in flab[idx]], dtype=np.float64)
    env = env_from_block_gains(gb, len(x), sr, block_ms)
    rate = float(np.max(np.abs(np.diff(env.astype(np.float64)))) * sr) if len(env) > 1 else 0.0
    info = {"degraded": len(gain_g) <= 1, "n_groups": len(gain_g),
            "group_gains_db": {str(k): round(v, 2) for k, v in gain_g.items()},
            "group_detail": detail, "gain_min": round(float(env.min()), 2),
            "gain_max": round(float(env.max()), 2), "gain_mean": round(float(env.mean()), 2),
            "max_actual_rate_db_s": round(rate, 2)}
    return env, info


# ------------------------------------------------------------------ 渲染 + limiter
def render_variant(x, sr, env_db, mask, out_path, T, tmp_prefix, limit_db=-2.0):
    """渲染：增益 -> 限幅(≤limit) -> 用 ASL 校正 -> 校验真峰值 <= -1 dBTP。

    注意采样峰值限到 limit 并不等于真峰值合格：16k 源本身带削顶/超采样过冲时，
    4x 过采样的真峰值可比采样峰值高约 2-3 dB。这里先按 limit_db 跑，测真峰值；
    若 > -1 dBTP 则下调 limit 重做（最多 2 次）。
    """
    pre = (x.astype(np.float64) * (10.0 ** (env_db.astype(np.float64) / 20.0))).astype(np.float32)
    block = int(round(sr * 0.01))
    nf = len(pre) // block
    peak_frames = np.abs(pre[: nf * block]).reshape(nf, block).max(axis=1)
    clip_pre = int(np.sum(np.abs(pre) >= 0.999))
    gain_metrics = {"gain_db_min": round(float(env_db.min()), 2),
                    "gain_db_max": round(float(env_db.max()), 2),
                    "gain_db_mean": round(float(env_db.mean()), 2),
                    "gain_rate_db_s_max": (round(float(np.max(np.abs(np.diff(env_db.astype(np.float64)))) * sr), 2)
                                           if len(env_db) > 1 else 0.0),
                    "clip_samples_pre_limiter": clip_pre}
    limit_cur = float(limit_db)
    correction_db = 0.0
    iters = 0
    tp_retries = 0
    yl = None
    r128 = {"lufs": None, "lra": None, "tp_dbtp": None}
    for _tp_try in range(3):
        y = (pre.astype(np.float64) * (10.0 ** (correction_db / 20.0))).astype(np.float32)
        for it in range(3):
            iters += 1
            wavfile_path = tmp_prefix + "_pre.wav"
            write_wav(wavfile_path, y, sr, subtype="PCM_32")
            lim_path = tmp_prefix + "_lim.wav"
            limiter_file(wavfile_path, lim_path, limit_db=limit_cur)
            _, yl = read_wav(lim_path)
            if len(yl) < len(y):
                yl = np.pad(yl, (0, len(y) - len(yl)))
            elif len(yl) > len(y):
                yl = yl[: len(y)]
            p_yl, _ = frame_power(yl, sr)
            n = min(len(p_yl), len(mask))
            asl, _, _ = asl_from_power(p_yl[:n][mask[:n]])
            delta = T - asl
            if abs(delta) <= 0.03:
                break
            y = (yl * (10.0 ** (delta / 20.0))).astype(np.float32)
            correction_db += delta
        write_wav(out_path, yl, sr, subtype="PCM_16")
        r128 = measure_r128(out_path)
        tp = r128.get("tp_dbtp")
        if tp is None or tp <= -1.0 or _tp_try >= 2:
            break
        # 真峰值超限：按“真峰值相对采样峰值限幅点的过冲”精确下调，再从 pre 重做
        overshoot_db = tp - limit_cur
        limit_cur = round(-1.0 - overshoot_db - 0.3, 2)
        correction_db = 0.0
        iters = 0
        tp_retries += 1
    lim_lin = 10.0 ** (limit_cur / 20.0)
    trig = peak_frames > lim_lin
    over = 20 * np.log10(np.maximum(peak_frames[trig].max(), 1e-12) / lim_lin) if trig.any() else 0.0
    gain_metrics["limiter"] = {"limit_db": round(limit_cur, 2), "trigger_frames": int(trig.sum()),
                               "trigger_ratio": round(float(trig.mean()) if nf else 0.0, 5),
                               "max_over_db": round(float(over), 2),
                               "tp_retries": tp_retries}
    _, yf = read_wav(out_path)
    pf, _ = frame_power(yf, sr)
    final = measure_region(yf, sr, mask, p=pf)
    final["r128"] = r128
    final["asl_dev_db"] = round(float(final["speech_asl_dbfs"] - T), 2)
    final["limit_iterations"] = iters
    final["clipped_final"] = int(np.sum(np.abs(yf) >= 0.999))
    for k in ("tmp", "pre", "lim"):
        p = tmp_prefix + f"_{k}.wav"
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    final.update(gain_metrics)
    return final


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True)
    ap.add_argument("--wav", default="")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--mode", default="afftdn", choices=["afftdn", "arnndn", "highpass"])
    ap.add_argument("--T", type=float, default=-20.0)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--variants", default="v1,v2,v3")
    ap.add_argument("--diar-json", default="")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit-db", type=float, default=-2.0)
    ap.add_argument("--clamp-db", type=float, default=3.0, help="V2 慢变增益相对静态增益的上下限(dB)")
    args = ap.parse_args()

    short = args.short
    src = args.wav or os.path.join(AE, "wav", short + ".wav")
    outdir = args.outdir or os.path.join(AE, short)
    os.makedirs(outdir, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    tmpdir = os.path.join(AE, "samples", "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    t_all = time.perf_counter()
    timing = {}

    # 0) 读原音频（可切片）
    t0 = time.perf_counter()
    sr, xsrc = read_wav(src)
    total_dur = len(xsrc) / sr
    a = int(round(args.start * sr))
    b = int(round(args.end * sr)) if args.end > 0 else len(xsrc)
    xsrc = np.ascontiguousarray(xsrc[a:b])
    dur = len(xsrc) / sr
    timing["load_s"] = round(time.perf_counter() - t0, 2)

    # 1) 降噪（写成 s16 输入再 f32 输出）
    t0 = time.perf_counter()
    slice_s16 = os.path.join(tmpdir, f"{short}_input.wav")
    if args.start or args.end or not os.path.exists(os.path.join(AE, "denoised", f"{short}_{args.mode}.wav")):
        write_wav(slice_s16, xsrc, sr, subtype="PCM_16")
        den_in = slice_s16
    else:
        den_in = src
    full_mode = (not args.pilot) and not (args.start or args.end)
    den_path = (os.path.join(AE, "denoised", f"{short}_{args.mode}.wav") if full_mode
                else os.path.join(outdir, "denoised.wav"))
    if not (full_mode and os.path.exists(den_path)):
        os.makedirs(os.path.dirname(den_path), exist_ok=True)
        denoise_file(den_in, den_path, mode=args.mode)
    timing["denoise_s"] = round(time.perf_counter() - t0, 2)
    sr, xden = read_wav(den_path)
    if abs(len(xden) - len(xsrc)) > int(0.02 * sr):  # 缓存长度不符（例如被试跑切片污染）-> 重建
        if full_mode:
            denoise_file(src, den_path, mode=args.mode)
        else:
            denoise_file(den_in, den_path, mode=args.mode)
        sr, xden = read_wav(den_path)
    if len(xden) > len(xsrc):
        xden = xden[: len(xsrc)]
    elif len(xden) < len(xsrc):
        xden = np.pad(xden, (0, len(xsrc) - len(xden)))
    dur = len(xden) / sr

    # 2) VAD（在降噪后音频上）
    t0 = time.perf_counter()
    segs = detect_speech(xden, sr)
    cont, frag = classify_segments(segs, 3.0)
    timing["vad_s"] = round(time.perf_counter() - t0, 2)

    # 3) 裁剪
    t0 = time.perf_counter()
    kept, removed = build_keep_intervals(dur, segs, cont, head_pad=0.3, tail_pad=0.3,
                                         pause_min=3.0, guard=0.25)
    kmap, clean_dur = mapping_from_kept(kept)
    clean = apply_keep(xden, kept)
    timing["crop_s"] = round(time.perf_counter() - t0, 2)

    # 裁剪统计
    total_cut = dur - clean_dur
    head_trim = next((r["end"] - r["start"] for r in removed if r["reason"] == "head_trim"), 0.0)
    tail_trim = next((r["end"] - r["start"] for r in removed if r["reason"] == "tail_trim"), 0.0)
    internal_cut = sum(r["end"] - r["start"] for r in removed if r["reason"] == "internal_pause")
    # 长停顿统计（orig 时间轴）
    gaps = []
    prev = 0.0
    for s, e in segs:
        if s > prev:
            gaps.append((prev, s))
        prev = max(prev, e)
    if prev < dur:
        gaps.append((prev, dur))
    long_gaps = [(s, e) for s, e in gaps if e - s >= 3.0]
    inner_lo = cont[0][0] if cont else 0.0
    inner_hi = cont[-1][1] if cont else dur
    inner_long = [(s, e) for s, e in long_gaps if s >= inner_lo and e <= inner_hi]
    # 语音保留率
    def coverage(seglist):
        tot = 0.0
        kept_tot = 0.0
        for s, e in seglist:
            tot += e - s
            for cs, ce in map_orig_to_clean(kept, s, e):
                pass
            # 计算与 kept 交集
            ov = 0.0
            for ks, ke in kept:
                lo, hi = max(s, ks), min(e, ke)
                if hi > lo:
                    ov += hi - lo
            kept_tot += ov
        return round(kept_tot / tot, 5) if tot > 0 else None

    vad_stats = {
        "n_segments": len(segs),
        "speech_total_s": round(sum(e - s for s, e in segs), 2),
        "speech_ratio": round(sum(e - s for s, e in segs) / dur, 4) if dur else 0,
        "continuous": {"n": len(cont), "total_s": round(sum(e - s for s, e in cont), 2),
                       "mean_s": round(float(np.mean([e - s for s, e in cont])), 2) if cont else 0,
                       "longest_s": round(max([e - s for s, e in cont], default=0), 2)},
        "fragments": {"n": len(frag), "total_s": round(sum(e - s for s, e in frag), 2),
                      "mean_s": round(float(np.mean([e - s for s, e in frag])), 2) if frag else 0},
        "long_pauses": {"n": len(long_gaps), "total_s": round(sum(e - s for s, e in long_gaps), 2),
                        "longest_s": round(max([e - s for s, e in long_gaps], default=0), 2),
                        "internal_n": len(inner_long),
                        "internal_total_s": round(sum(e - s for s, e in inner_long), 2)},
    }
    crop_stats = {
        "orig_dur_s": round(dur, 2), "clean_dur_s": round(clean_dur, 2),
        "head_trim_s": round(head_trim, 2), "tail_trim_s": round(tail_trim, 2),
        "internal_cut_s": round(internal_cut, 2), "total_cut_s": round(total_cut, 2),
        "cut_ratio": round(total_cut / dur, 4) if dur else 0,
        "n_internal_cuts": len([r for r in removed if r["reason"] == "internal_pause"]),
        "speech_preserved_ratio": coverage(segs),
        "continuous_preserved_ratio": coverage(cont),
        "fragments_preserved_ratio": coverage(frag),
    }

    # 4) 归一化前测量（干净紧凑时间轴，只统计连续段语音帧）
    t0 = time.perf_counter()
    clean_segs = []
    for s, e in cont:
        for cs, ce in map_orig_to_clean(kept, s, e):
            clean_segs.append((cs, ce))
    mask = frame_mask_from_intervals(clean_segs, len(clean), sr)
    p_clean, _ = frame_power(clean, sr)
    pre = measure_region(clean, sr, mask, p=p_clean)
    # 源文件（未降噪）同样用 orig 时间轴掩码测 ASL
    orig_mask = frame_mask_from_intervals(cont, len(xsrc), sr)
    p_src, _ = frame_power(xsrc, sr)
    n_src = min(len(p_src), len(orig_mask))
    src_asl, _, _ = asl_from_power(p_src[:n_src][orig_mask[:n_src]])
    pre["source_speech_asl_dbfs"] = round(float(src_asl), 3)
    if not args.pilot:
        pre["source_r128"] = measure_r128(src)
    pre["lufs_minus_asl"] = (round(pre["source_r128"]["lufs"] - src_asl, 2)
                             if "source_r128" in pre else None)
    timing["measure_pre_s"] = round(time.perf_counter() - t0, 2)

    # 5) 说话人（V3 用）
    t0 = time.perf_counter()
    diar_json = args.diar_json
    if not diar_json:
        cand = os.path.join(AE, "diar", f"{short}_t0.7.json")
        diar_json = cand if os.path.exists(cand) else ""
    v3_info = {"seg_labels": None, "info": {"method": "skipped"}}
    if "v3" in variants:
        if diar_json and os.path.exists(diar_json):
            seg_labels, sinfo = speakers.seg_labels_from_official(diar_json, clean_segs, kept)
            v3_info = {"seg_labels": seg_labels, "info": sinfo}
        else:
            try:
                emb = speakers.Embedder(num_threads=4)
                seg_labels, sinfo = speakers.cluster_segments(clean, sr, clean_segs, emb)
                v3_info = {"seg_labels": seg_labels, "info": sinfo}
            except Exception as e:  # embedding 失败不阻塞主流程
                v3_info = {"seg_labels": None, "info": {"method": "failed", "error": repr(e)}}
    timing["speaker_s"] = round(time.perf_counter() - t0, 2)

    # 6) 三个变体
    base_gain = float(args.T - pre["speech_asl_dbfs"])
    results = {}
    rec = None
    for v in variants:
        t0 = time.perf_counter()
        if v == "v1":
            env = np.full(len(clean), base_gain, dtype=np.float32)
            ginfo = {"type": "static", "gain_db": round(base_gain, 2),
                     "gain_rate_db_s_max": 0.0}
        elif v == "v2":
            env, ginfo = build_v2(clean, sr, mask, base_gain, args.T, p=p_clean,
                                  clamp_db=args.clamp_db)
            ginfo["type"] = "static+slow_adaptive"
        else:
            if v3_info["seg_labels"] is None:
                env = np.full(len(clean), base_gain, dtype=np.float32)
                ginfo = {"type": "speaker_static", "gain_db": round(base_gain, 2),
                         "degraded": True, "reason": "no_speaker_labels"}
            else:
                env, ginfo = build_v3(clean, sr, mask, clean_segs, v3_info["seg_labels"],
                                      args.T, p=p_clean)
                ginfo["seg_labels"] = [int(x) for x in v3_info["seg_labels"]]
                ginfo["speaker_info"] = v3_info["info"]
        vdir = os.path.join(AE, "variants", short)
        os.makedirs(vdir, exist_ok=True)
        out_wav = os.path.join(vdir, v + ".wav")
        tmp_prefix = os.path.join(tmpdir, f"{short}_{v}")
        m = render_variant(clean, sr, env, mask, out_wav, args.T, tmp_prefix,
                           limit_db=args.limit_db)
        # V3 分组后测（主讲/其他分别）
        if v == "v3" and v3_info["seg_labels"] is not None:
            _, yf = read_wav(out_wav)
            pf, _ = frame_power(yf, sr)
            flab = np.zeros(int(np.ceil(len(clean) / 160)), dtype=np.int16)
            for (s, e), lab in zip(clean_segs, v3_info["seg_labels"]):
                flab[int(round(s / 0.01)):int(round(e / 0.01))] = int(lab)
            per_group = {}
            for gid in sorted(set(int(x) for x in flab[mask[:len(flab)]])):
                gm = mask[:len(flab)] & (flab == gid)
                a2, _, ns2 = asl_from_power(pf[:len(gm)][gm])
                per_group[str(gid)] = {"asl_dbfs": round(float(a2), 2),
                                       "speech_sec": round(ns2 * 0.01, 2),
                                       "dev_db": round(float(a2 - args.T), 2)}
            m["group_asl_after"] = per_group
        m["gain"] = ginfo
        results[v] = m
        timing[f"render_{v}_s"] = round(time.perf_counter() - t0, 2)

    # 7) 输出 mapping / clean.wav / stats
    mapping = {"short": short, "src": os.path.abspath(src),
               "sample_rate": sr, "orig_dur_s": round(dur, 2), "clean_dur_s": clean_dur,
               "kept": kmap, "removed": removed,
               "crop": crop_stats,
               "note": "kept 每项=[orig_start, orig_end, clean_start, clean_end]（秒）；"
                       "clean.wav 时间轴 = kept 依次拼接"}
    dump_json(os.path.join(outdir, "mapping.json"), mapping)
    # clean.wav 默认取 V2（推荐变体在 aggregate 阶段最终确定）
    clean_variant = "v2" if "v2" in variants else variants[0]
    shutil.copyfile(os.path.join(AE, "variants", short, clean_variant + ".wav"),
                    os.path.join(outdir, "clean.wav"))
    stats = {
        "short": short, "src": os.path.abspath(src), "denoise_mode": args.mode,
        "target_asl_dbfs": args.T, "limiter_limit_db": args.limit_db,
        "pilot": {"start": args.start, "end": args.end} if (args.start or args.end) else None,
        "timing": timing, "vad": vad_stats, "crop": crop_stats, "pre": pre,
        "base_gain_db": round(base_gain, 2), "variants": results,
        "v2_clamp_db": args.clamp_db,
        "clean_variant": clean_variant,
        "speaker_v3": (v3_info["info"] if v3_info["seg_labels"] is not None else v3_info["info"]),
        "timing_total_s": round(time.perf_counter() - t_all, 2),
    }
    dump_json(os.path.join(outdir, "stats.json"), stats)
    print(f"[done] {short} orig={dur:.0f}s clean={clean_dur:.0f}s cut={crop_stats['cut_ratio']*100:.1f}% "
          f"asl_in={pre['speech_asl_dbfs']:.2f} base_gain={base_gain:+.2f}dB "
          f"speakers={stats['speaker_v3'].get('n_speakers', 1)} "
          f"total={stats['timing_total_s']:.1f}s")
    for v in variants:
        m = results[v]
        print(f"  {v}: asl_out={m['speech_asl_dbfs']:.2f} dev={m['asl_dev_db']:+.2f} "
              f"tp={m['r128']['tp_dbtp']} lufs={m['r128']['lufs']} "
              f"fluct_half={m['window5min']['half']} rate={m['gain_rate_db_s_max']} "
              f"lim_trig={m['limiter']['trigger_ratio']*100:.2f}% clip={m['clipped_final']}")


if __name__ == "__main__":
    main()
