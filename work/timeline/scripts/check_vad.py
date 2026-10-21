"""VAD 自检：10ms 帧 silero 实现 vs faster-whisper 原生 32ms 窗实现，比较速度与分段差异。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    wav = sys.argv[1] if len(sys.argv) > 1 else os.path.join(common.ROOT, "work", "samples", "mono_600_60.wav")
    arr, sr = common.read_wav(wav)
    x = arr[:, 0].astype(np.float32)
    dur = len(x) / sr
    print(f"[audio] {common.rel(wav)} {sr}Hz {dur:.2f}s", flush=True)

    vad = common.SileroVAD(threads=4)
    t0 = time.perf_counter()
    p = vad.probs(x)
    t_vad = time.perf_counter() - t0
    print(f"[10ms] frames={len(p)} probs in {t_vad:.2f}s (RTF={t_vad/dur:.4f}) "
          f"p>=0.5 ratio={np.mean(p >= 0.5)*100:.1f}%", flush=True)

    # 原生 faster-whisper 32ms 不重叠窗
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    t0 = time.perf_counter()
    ref = get_speech_timestamps(x, VadOptions(threshold=0.5, min_speech_duration_ms=500,
                                              min_silence_duration_ms=350, speech_pad_ms=120),
                                sampling_rate=sr)
    t_ref = time.perf_counter() - t0
    print(f"[32ms] fw-vad {len(ref)} chunks in {t_ref:.2f}s (RTF={t_ref/dur:.4f})", flush=True)

    segs = common.probs_to_segments(p, hop=0.01, total_dur=dur)
    segs = common.post_merge(segs)
    print(f"[segs] 10ms method -> {len(segs)} segments")
    for s, e in segs[:10]:
        print(f"  {s:7.2f} - {e:7.2f}  ({e-s:5.2f}s)")

    # 与参照边界对齐误差
    ref_b = [(c["start"] / sr, c["end"] / sr) for c in ref]
    starts = np.array([s for s, _ in ref_b])
    ends = np.array([e for _, e in ref_b])
    ds, de = [], []
    for s, e in segs:
        i = int(np.argmin(np.abs(starts - s)))
        j = int(np.argmin(np.abs(ends - e)))
        ds.append(s - starts[i])
        de.append(e - ends[j])
    print(f"[diff vs 32ms] start median={np.median(np.abs(ds))*1000:.0f}ms "
          f"p90={np.percentile(np.abs(ds),90)*1000:.0f}ms | "
          f"end median={np.median(np.abs(de))*1000:.0f}ms p90={np.percentile(np.abs(de),90)*1000:.0f}ms")


if __name__ == "__main__":
    main()
