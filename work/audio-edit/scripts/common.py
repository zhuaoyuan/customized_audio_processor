# -*- coding: utf-8 -*-
"""音频编辑管线公共工具：IO、分帧、P.56 式 ASL、R128、VAD、裁剪映射、FFmpeg 封装。

统一采样率 16 kHz、单声道。
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import scipy.io.wavfile as wavfile

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AE = os.path.join(ROOT, "work", "audio-edit")
RNN_DIR = os.path.join(ROOT, "work", "bench", "models", "rnnoise")


# ---------------------------------------------------------------- IO
def read_wav(path):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype != np.float32:
        raise ValueError(f"unsupported dtype {x.dtype}")
    if x.ndim > 1:
        x = x[:, 0]
    return sr, np.ascontiguousarray(x, dtype=np.float32)


def write_wav(path, x, sr=SR, subtype="PCM_16"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if subtype == "PCM_16":
        y = np.clip(x, -1.0, 1.0)
        y = (y * 32767.0).astype(np.int16)
    elif subtype.endswith("32"):
        y = x.astype(np.float32)
        subtype = "FLOAT"
    else:
        y = x.astype(np.float32)
    wavfile.write(path, sr, y)


# ---------------------------------------------------------------- 分帧
def frame_power(x, sr=SR, win_ms=20, hop_ms=10, chunk_s=600.0):
    """返回 (每帧均方功率, 帧中心样本位置)。20ms 窗、10ms 步进；分块 float32 计算省内存。"""
    w = int(round(sr * win_ms / 1000))
    h = int(round(sr * hop_ms / 1000))
    n = len(x)
    if n < w:
        return np.array([float(np.mean(x ** 2))]) if n else np.array([]), np.array([0])
    nframes = 1 + (n - w) // h
    p = np.empty(nframes, dtype=np.float64)
    nper = max(1, int(round(chunk_s * sr / h)))
    k = 0
    while k < nframes:
        k2 = min(nframes, k + nper)
        start = k * h
        end = (k2 - 1) * h + w
        view = np.lib.stride_tricks.sliding_window_view(x[start:end], w)[::h]
        p[k:k2] = np.mean(view.astype(np.float32) ** 2, axis=1)
        k = k2
    centers = np.arange(nframes) * h + w // 2
    return p, centers


def frame_mask_from_intervals(intervals, n_samples, sr=SR, hop_ms=10, win_ms=20):
    """10ms 帧级布尔掩码：帧中心（与 frame_power 一致，20ms 窗中心）落在区间内即为 True。"""
    h = int(round(sr * hop_ms / 1000))
    w = int(round(sr * win_ms / 1000))
    n = int(np.ceil(n_samples / h))
    centers = np.arange(n) * h + w // 2
    m = np.zeros(n, dtype=bool)
    for s, e in intervals:
        a = int(np.searchsorted(centers, s * sr, side="left"))
        b = int(np.searchsorted(centers, e * sr, side="left"))
        m[a:b] = True
    return m


def mask_to_intervals(mask, hop_ms=10):
    """10ms 帧掩码 -> 秒区间。"""
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(float(s * hop_ms / 1000), float(e * hop_ms / 1000)) for s, e in zip(starts, ends)]


def db(v, eps=1e-12):
    return 10.0 * np.log10(np.maximum(v, eps))


# ---------------------------------------------------------------- P.56 式 ASL（预计算帧功率版）
def asl_from_power(ps, rel_db=15.0, max_iter=20):
    """由语音帧功率数组直接算 P.56 式 ASL。返回 (dbfs, n_active, n_frames)。"""
    ps = np.asarray(ps, dtype=np.float64)
    if len(ps) == 0:
        return float("nan"), 0, 0
    level = db(np.mean(ps))
    thr = level - rel_db
    for _ in range(max_iter):
        act = ps[db(ps) > thr]
        if len(act) == 0:
            break
        new = db(np.mean(act))
        if abs(new - level) < 0.01:
            level = new
            break
        level = new
        thr = level - rel_db
    act = ps[db(ps) > thr]
    na = int(len(act)) if len(act) else int(len(ps))
    return float(level), na, int(len(ps))


def power_percentiles(ps):
    if len(ps) == 0:
        return {}
    pd = db(np.asarray(ps, dtype=np.float64))
    return {f"p{q}": round(float(np.percentile(pd, q)), 2) for q in (10, 25, 50, 75, 90)}


def window_levels_p(p, mask, win_s=300.0, hop_s=None):
    """用预计算帧功率按 5min 窗口统计 ASL。p/mask 均为 10ms 帧级。"""
    hop_s = hop_s or win_s
    n = min(len(p), len(mask))
    dur = n * 0.01
    times, levels, secs = [], [], []
    t = 0.0
    while t < dur - 1e-9:
        e = min(dur, t + win_s)
        a, b = int(round(t * 100)), int(round(e * 100))
        ps = p[a:b][mask[a:b]]
        if len(ps):
            lv, _, _ = asl_from_power(ps)
            times.append((t + e) / 2)
            levels.append(lv)
            secs.append(len(ps) * 0.01)
        t += hop_s
    return times, levels, secs


# ---------------------------------------------------------------- P.56 式 ASL
def active_speech_level(x, sr, speech_mask, win_ms=20, hop_ms=10, rel_db=15.0, max_iter=20):
    """P.56 思路的活跃语音电平（dBFS）。

    1) 取 10ms 步进、20ms 窗的帧功率；只保留中心落在 VAD 连续语音段内的帧；
    2) 初值 L = 10*log10(mean(p))；阈值 T = L - rel_db；
    3) 保留 p > T 的“活跃帧”，重算 L；迭代至收敛（<0.01 dB）。
    返回 (asl_dbfs, n_active_frames, n_speech_frames)。
    """
    p, centers = frame_power(x, sr, win_ms, hop_ms)
    if len(p) == 0:
        return float("nan"), 0, 0
    n = min(len(p), len(speech_mask))
    sel = speech_mask[:n] & (centers[:n] < len(x))
    ps = p[:n][sel]
    if len(ps) == 0:
        return float("nan"), 0, 0
    level = db(np.mean(ps))
    thr = level - rel_db
    for _ in range(max_iter):
        act = ps[db(ps) > thr]
        if len(act) == 0:
            break
        new = db(np.mean(act))
        if abs(new - level) < 0.01:
            level = new
            break
        level = new
        thr = level - rel_db
    act = ps[db(ps) > thr]
    na = int(len(act)) if len(act) else int(len(ps))
    return float(level), na, int(len(ps))


def speech_frame_db_percentiles(x, sr, speech_mask, win_ms=20, hop_ms=10):
    p, centers = frame_power(x, sr, win_ms, hop_ms)
    n = min(len(p), len(speech_mask))
    sel = speech_mask[:n] & (centers[:n] < len(x))
    ps = p[:n][sel]
    if len(ps) == 0:
        return {}
    pd = db(ps)
    return {f"p{q}": round(float(np.percentile(pd, q)), 2) for q in (10, 25, 50, 75, 90)}


def window_levels(x, sr, speech_mask, win_s=300.0, hop_s=None, min_speech_s=30.0):
    """按 5min 窗口统计每窗的 ASL 与语音帧时长。返回 (times, levels, speech_secs)。"""
    hop_s = hop_s or win_s
    dur = len(x) / sr
    times, levels, secs = [], [], []
    t = 0.0
    while t < dur:
        e = min(dur, t + win_s)
        # 窗口内帧掩码：直接切片（帧与样本对齐到 10ms，近似即可）
        a = int(t * sr)
        b = int(e * sr)
        m = speech_mask[int(t * 100): int(e * 100)]  # 10ms 帧索引
        lv, na, ns = active_speech_level(x[a:b], sr, m)
        sp_s = ns * 0.01
        if ns > 0:
            times.append((t + e) / 2)
            levels.append(lv)
            secs.append(sp_s)
        t += hop_s
    return times, levels, secs


def fluctuation(levels):
    """返回 5min 窗口电平的 p5/p95 与半展宽（±dB）。"""
    if len(levels) < 2:
        return {"n_windows": len(levels or []), "p5": None, "p95": None, "spread": None, "half": None}
    a = np.asarray(levels, dtype=np.float64)
    p5, p95 = float(np.percentile(a, 5)), float(np.percentile(a, 95))
    return {"n_windows": int(len(a)), "p5": round(p5, 2), "p95": round(p95, 2),
            "spread": round(p95 - p5, 2), "half": round((p95 - p5) / 2, 2),
            "mean": round(float(a.mean()), 2), "std": round(float(a.std(ddof=0)), 2)}


# ---------------------------------------------------------------- EBU R128
_R128_RE = re.compile(r"\[Parsed_ebur128.*?Summary:.*?I:\s+(-?\d+\.?\d*)\s+LUFS.*?LRA:\s+(-?\d+\.?\d*)\s+LU.*?Peak:\s+(-?\d+\.?\d*)\s+dBFS",
                      re.S)


def measure_r128(path):
    """用 ffmpeg ebur128(peak=true) 测 integrated LUFS / LRA / true peak(dBTP)。"""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-i", path,
           "-af", "ebur128=peak=true", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = _R128_RE.search(r.stderr)
    if not m:
        return {"lufs": None, "lra": None, "tp_dbtp": None}
    return {"lufs": round(float(m.group(1)), 2), "lra": round(float(m.group(2)), 2),
            "tp_dbtp": round(float(m.group(3)), 2)}


# ---------------------------------------------------------------- FFmpeg
def ffmpeg_filter(src, dst, af, cwd=None, sample_fmt="pcm_f32le"):
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-af", af, "-c:a", sample_fmt, dst]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"ffmpeg failed: {af}\n{r.stderr[-2000:]}")
    return dst


def denoise_file(src, dst, mode="afftdn", log=None):
    """降噪：默认 highpass=f=80,afftdn=nr=12:nf=-30:tn=1；对照 arnndn。"""
    import time
    t0 = time.perf_counter()
    if mode == "afftdn":
        af = "highpass=f=80,afftdn=nr=12:nf=-30:tn=1"
        ffmpeg_filter(src, dst, af)
    elif mode == "arnndn":
        # arnndn 模型路径含冒号会破坏 filter 语法 -> cd 到模型目录用相对路径
        ffmpeg_filter(src, dst, "highpass=f=80,arnndn=m=sh.rnnn", cwd=RNN_DIR)
    elif mode == "highpass":
        ffmpeg_filter(src, dst, "highpass=f=80")
    else:
        raise ValueError(mode)
    if log is not None:
        log(f"[denoise:{mode}] {os.path.basename(src)} {time.perf_counter()-t0:.1f}s")
    return dst


def limiter_file(src, dst, limit_db=-2.0):
    """alimiter 限幅（关闭自动补偿增益）。limit 用 dBFS 换算为线性值。"""
    lin = 10 ** (limit_db / 20.0)
    af = f"alimiter=limit={lin:.6f}:attack=5:release=50:level=0"
    return ffmpeg_filter(src, dst, af, sample_fmt="pcm_f32le")


# ---------------------------------------------------------------- VAD (Silero)
_VAD = None


def _vad_model():
    global _VAD
    if _VAD is None:
        from faster_whisper.vad import VadOptions, get_speech_timestamps  # noqa: F401
        import faster_whisper.vad as fv
        _VAD = fv
    return _VAD


def detect_speech(x, sr=SR, threshold=0.5, min_silence_s=0.3, min_speech_s=0.25):
    """Silero VAD（faster-whisper 内置 ONNX 版，32ms 窗/16k）。

    返回 10ms 对齐的语音段 [(start,end), ...]，已按 min_silence 合并、min_speech 过滤。
    """
    fv = _vad_model()
    opts = fv.VadOptions(threshold=threshold,
                         min_speech_duration_ms=int(round(min_speech_s * 1000)),
                         min_silence_duration_ms=int(round(min_silence_s * 1000)),
                         speech_pad_ms=0)
    ts = fv.get_speech_timestamps(x, opts, sampling_rate=sr)
    segs = []
    for s in ts:
        a = round(s["start"] / sr / 0.01) * 0.01
        b = round(s["end"] / sr / 0.01) * 0.01
        if b > a:
            segs.append((round(a, 2), round(b, 2)))
    return segs


def classify_segments(segs, min_cont_s=3.0):
    """区分连续说话段落(>=3s) 与碎片(<3s)。"""
    cont = [s for s in segs if s[1] - s[0] >= min_cont_s - 1e-9]
    frag = [s for s in segs if s[1] - s[0] < min_cont_s - 1e-9]
    return cont, frag


# ---------------------------------------------------------------- 裁剪
def build_keep_intervals(dur, segs, cont, head_pad=0.3, tail_pad=0.3,
                         pause_min=3.0, guard=0.25):
    """生成保留区间与删除区间。

    - 头尾：以首个/末个 >=3s 连续语音段为界，各留 head_pad/tail_pad；
    - 内部：>=pause_min 秒的非语音段切除，两侧各留 guard 秒保护带。
    返回 (kept, removed)：kept=[(orig_s,orig_e),...] 有序不重叠；
    removed=[{"start","end","reason"},...]。
    """
    kept, removed = [], []
    if not cont:
        return [(0.0, dur)], []
    head = max(0.0, cont[0][0] - head_pad)
    tail = min(dur, cont[-1][1] + tail_pad)
    if head > 0:
        removed.append({"start": 0.0, "end": head, "reason": "head_trim"})
    if tail < dur:
        removed.append({"start": tail, "end": dur, "reason": "tail_trim"})
    # 在 [head, tail] 内遍历所有语音段（含碎片），找长非语音空隙
    inner = [s for s in segs if s[1] > head and s[0] < tail]
    cuts = []
    cur = head
    for s, e in inner:
        s2, e2 = max(s, head), min(e, tail)
        if s2 > cur:
            gap = s2 - cur
            if gap >= pause_min:
                cs, ce = cur + guard, s2 - guard
                if ce > cs:
                    cuts.append((cs, ce))
        cur = max(cur, e2)
    if tail - cur >= pause_min:
        cs, ce = cur + guard, tail - guard
        if ce > cs:
            cuts.append((cs, ce))
    kept = [head]
    for cs, ce in cuts:
        kept.append(cs)
        kept.append(ce)
        removed.append({"start": round(cs, 3), "end": round(ce, 3), "reason": "internal_pause"})
    kept.append(tail)
    kept = [(round(kept[i], 3), round(kept[i + 1], 3)) for i in range(0, len(kept), 2)
            if kept[i + 1] - kept[i] > 1e-6]
    removed.sort(key=lambda r: r["start"])
    return kept, removed


def mapping_from_kept(kept):
    """kept 区间 -> {kept:[[os,oe,cs,ce]], clean_dur}。"""
    out = []
    c = 0.0
    for s, e in kept:
        d = e - s
        out.append([round(s, 3), round(e, 3), round(c, 3), round(c + d, 3)])
        c += d
    return out, round(c, 3)


def apply_keep(x, kept, sr=SR):
    parts = [x[int(round(s * sr)): int(round(e * sr))] for s, e in kept]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def map_orig_to_clean(kept, s, e):
    """把 orig 区间 [s,e] 映射到 clean 时间；返回 [(clean_s, clean_e), ...]（可能被拆开）。"""
    out = []
    c = 0.0
    for ks, ke in kept:
        d = ke - ks
        a, b = max(s, ks), min(e, ke)
        if b > a:
            out.append((c + (a - ks), c + (b - ks)))
        c += d
    return out


# ---------------------------------------------------------------- 工具
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
