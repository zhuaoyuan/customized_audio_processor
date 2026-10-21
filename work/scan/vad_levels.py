# -*- coding: utf-8 -*-
"""Step 3: Silero VAD + 语音电平 ASL + 非语音结构。

VAD: faster-whisper 内置 Silero VAD v6 (onnxruntime, 512-sample 窗口, 批量推理)。
  选择理由：work/venv 已含该 onnx 模型；sherpa-onnx 的 SileroVadModelConfig 需要
  v4 状态接口，与本机 v5/v6 模型输入 (h,c) 不匹配。两者同属 Silero VAD 家族。
帧参数：25 ms 窗 / 10 ms 跳。语音帧 = 与 VAD 语音段时间重叠的帧。
ASL(活跃语音电平) = 语音帧能量的平均（ITU-T P.56 思路），dBFS。
输出 work/scan/vad/<short>.json
"""
import json
import os

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

ROOT = r"C:\projects\local\audio_processer"
SCAN = os.path.join(ROOT, "work", "scan")
VADDIR = os.path.join(SCAN, "vad")
os.makedirs(VADDIR, exist_ok=True)

SHORTS = ["f00_xifang20140227", "f20_lixiangguo20141219", "f24_xingershangxue",
          "f04_lixiangguo4", "f05_wulixue", "f06_lunyu"]

VAD_OPTS = dict(threshold=0.5, min_speech_duration_ms=250,
                min_silence_duration_ms=300, speech_pad_ms=100)
SR = 16000
WIN, HOP = 0.025, 0.010


def read_wav_f32(path):
    from scipy.io import wavfile
    sr, d = wavfile.read(path, mmap=True)
    assert sr == SR, sr
    return d  # float32 memmap


def db(v, eps=1e-12):
    return 20.0 * np.log10(np.maximum(v, eps))


def energy_db(seg):
    return 10.0 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-20)


def analyze(short):
    probe = json.load(open(os.path.join(SCAN, "probe.json"), encoding="utf-8"))
    wav = probe["files"][short]["wav"]
    x = read_wav_f32(wav)
    dur = len(x) / SR
    # VAD
    ts = get_speech_timestamps(x, VadOptions(**VAD_OPTS))
    segs = [(t["start"] / SR, t["end"] / SR) for t in ts]
    speech_s = sum(b - a for a, b in segs)

    # 25ms/10ms 帧 RMS（分块 cumsum，避免大文件内存爆掉）
    w = int(SR * WIN)
    h = int(SR * HOP)
    nf = max(0, (len(x) - w) // h + 1)
    e = np.empty(nf, dtype=np.float64)
    CH = 200000  # 200k 帧 = 2000s
    for s0 in range(0, nf, CH):
        s1 = min(nf, s0 + CH)
        a = s0 * h
        b = (s1 - 1) * h + w
        seg = np.asarray(x[a:b], dtype=np.float64)
        c = np.concatenate([[0.0], np.cumsum(seg * seg)])
        idx = np.arange(s0, s1) * h - a
        e[s0:s1] = (c[idx + w] - c[idx]) / w
    frms_db = 10 * np.log10(np.maximum(e, 1e-20))
    t_center = (np.arange(nf) * h + w / 2) / SR

    # 语音帧掩码：帧中心落在任一语音段内
    mask = np.zeros(nf, dtype=bool)
    for a, b in segs:
        mask |= (t_center >= a) & (t_center < b)
    sp = frms_db[mask]
    nsp = int(mask.sum())
    asl = float(10 * np.log10(np.mean(10 ** (sp / 10)) + 1e-20)) if nsp else float("nan")
    p = {f"p{k}": float(np.percentile(sp, k)) for k in (10, 50, 90)} if nsp else {}

    # 整段电平（含静音）与噪声底
    all_db = frms_db
    noise_floor = float(10 * np.log10(np.mean(10 ** (all_db[all_db <= np.percentile(all_db, 10)] / 10)) + 1e-20))

    # 5 分钟窗口 ASL
    win = 300.0
    wins = []
    t = 0.0
    while t < dur - 1e-6:
        a, b = t, min(t + win, dur)
        sel = (t_center >= a) & (t_center < b) & mask
        n = int(sel.sum())
        if n >= int(30 / HOP):  # 至少 30s 语音才统计
            wins.append({"t0": round(a, 1), "t1": round(b, 1), "speech_s": round(n * HOP, 1),
                         "asl": float(10 * np.log10(np.mean(10 ** (frms_db[sel] / 10)) + 1e-20))})
        t += win
    wa = np.array([w["asl"] for w in wins]) if wins else np.array([])
    wstats = {
        "n_windows": len(wins),
        "asl_p5": float(np.percentile(wa, 5)) if len(wa) else None,
        "asl_p50": float(np.percentile(wa, 50)) if len(wa) else None,
        "asl_p95": float(np.percentile(wa, 95)) if len(wa) else None,
        "asl_min": float(wa.min()) if len(wa) else None,
        "asl_max": float(wa.max()) if len(wa) else None,
        "asl_spread_p95_p5": float(np.percentile(wa, 95) - np.percentile(wa, 5)) if len(wa) >= 2 else None,
        "windows": wins,
    }

    # 头尾
    head_trim = segs[0][0] if segs else dur
    tail_trim = dur - segs[-1][1] if segs else dur

    # 内部 >=3s 非语音
    gaps = []
    for (a1, b1), (a2, b2) in zip(segs, segs[1:]):
        g = a2 - b1
        if g >= 3.0:
            gaps.append({"start": round(b1, 2), "dur": round(g, 2)})
    gaps.sort(key=lambda d: -d["dur"])
    gap_total = sum(g["dur"] for g in gaps)

    # 头部前 5 分钟语音占比（判断片头是否也是讲课）
    head5 = min(300.0, dur)
    head5_speech = sum(max(0.0, min(b, head5) - min(a, head5)) for a, b in segs if a < head5)

    res = {
        "short": short, "wav": wav, "dur_s": dur, "sr": SR,
        "vad": {"engine": "silero-vad-v6 (faster-whisper/onnxruntime)", **VAD_OPTS,
                "n_segments": len(segs), "speech_s": speech_s,
                "speech_ratio": speech_s / dur},
        "frames": {"win_ms": WIN * 1000, "hop_ms": HOP * 1000, "n_frames": nf,
                   "n_speech_frames": nsp},
        "level": {"asl_dbfs": asl,
                  "speech_frame_p10_dbfs": p.get("p10"),
                  "speech_frame_p50_dbfs": p.get("p50"),
                  "speech_frame_p90_dbfs": p.get("p90"),
                  "all_frames_rms_dbfs": float(10 * np.log10(np.mean(e) + 1e-20)),
                  "noise_floor_dbfs": noise_floor,
                  "snr_proxy_db": asl - noise_floor if nsp else None},
        "window5m": wstats,
        "edges": {"first_speech_s": round(head_trim, 2), "last_speech_end_s": round(segs[-1][1], 2),
                  "head_trim_s": round(head_trim, 2), "tail_trim_s": round(tail_trim, 2),
                  "head5m_speech_ratio": head5_speech / head5 if head5 else None},
        "nonspeech": {"gap_min_s": 3.0, "n_gaps": len(gaps), "total_s": round(gap_total, 1),
                      "longest_s": gaps[0]["dur"] if gaps else 0.0,
                      "top_gaps": gaps[:10]},
        "segments": [[round(a, 3), round(b, 3)] for a, b in segs],
    }
    json.dump(res, open(os.path.join(VADDIR, short + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[vad] {short}: dur={dur/60:.1f}min speech={speech_s/dur*100:.1f}% "
          f"ASL={asl:.2f} p10/p50/p90={p.get('p10',0):.1f}/{p.get('p50',0):.1f}/{p.get('p90',0):.1f} "
          f"noise={noise_floor:.1f} head={head_trim:.1f}s tail={tail_trim:.1f}s "
          f"gaps3+={len(gaps)}/{gap_total:.0f}s w5m={len(wins)}", flush=True)


def main():
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for s in SHORTS:
        if only and s != only:
            continue
        analyze(s)


if __name__ == "__main__":
    main()
