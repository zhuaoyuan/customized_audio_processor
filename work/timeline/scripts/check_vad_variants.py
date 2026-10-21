"""silero VAD 10ms 帧的几种实现变体对比（在 60s 讲课样本上）。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def streaming_probs(sess, x, hop, ctx=64, win=512, stateful=True, batch=4096):
    n = max(1, int(np.ceil(len(x) / hop)))
    pad = np.pad(x, (ctx, win + hop), mode="constant")
    h = np.zeros((1, 1, 128), dtype=np.float32)
    c = np.zeros((1, 1, 128), dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        idx = np.arange(i, min(n, i + batch))
        rows = np.empty((len(idx), ctx + win), dtype=np.float32)
        for k, j in enumerate(idx):
            base = j * hop + ctx
            rows[k, :ctx] = pad[base - ctx:base]
            rows[k, ctx:] = pad[base:base + win]
        p, hn, cn = sess.run(None, {"input": rows, "h": h, "c": c})
        out[i:i + len(idx)] = p.reshape(-1)[:len(idx)]
        if stateful:
            h, c = hn, cn
        else:
            h = np.zeros((1, 1, 128), dtype=np.float32)
            c = np.zeros((1, 1, 128), dtype=np.float32)
    return out


def stateless_percall(sess, x, hop=160, ctx=64, win=512):
    n = max(1, int(np.ceil(len(x) / hop)))
    pad = np.pad(x, (ctx, win + hop), mode="constant")
    h0 = np.zeros((1, 1, 128), dtype=np.float32)
    c0 = np.zeros((1, 1, 128), dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    for j in range(n):
        base = j * hop + ctx
        row = np.concatenate([pad[base - ctx:base], pad[base:base + win]]).reshape(1, -1)
        p, _, _ = sess.run(None, {"input": row, "h": h0, "c": c0})
        out[j] = p.reshape(-1)[0]
    return out


def main():
    wav = os.path.join(common.ROOT, "work", "samples", "mono_600_60.wav")
    arr, _ = common.read_wav(wav)
    x = arr[:, 0].astype(np.float32)
    dur = len(x) / 16000
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ref = get_speech_timestamps(x, VadOptions(threshold=0.5, min_speech_duration_ms=500,
                                              min_silence_duration_ms=350, speech_pad_ms=120))
    ref_b = [(c["start"] / 16000, c["end"] / 16000) for c in ref]
    print(f"[ref] fw 32ms streaming: {len(ref_b)} chunks, "
          f"speech={sum(e-s for s,e in ref_b):.1f}s/{dur:.0f}s", flush=True)

    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = 4; o.log_severity_level = 4
    sess = ort.InferenceSession(common.VAD_MODEL, providers=["CPUExecutionProvider"], sess_options=o)

    variants = [
        ("A hop512 streaming", dict(hop=512, stateful=True)),
        ("B hop160 streaming", dict(hop=160, stateful=True)),
    ]
    for name, kw in variants:
        t0 = time.perf_counter()
        p = streaming_probs(sess, x, **kw)
        dt = time.perf_counter() - t0
        segs = common.post_merge(common.probs_to_segments(p, hop=kw["hop"] / 16000, total_dur=dur))
        print(f"[{name}] {dt:.2f}s (RTF {dt/dur:.4f}) frames={len(p)} p>=.5={np.mean(p>=.5)*100:.0f}% "
              f"segs={len(segs)}", flush=True)

    # stateless 每窗一次调用（只跑前 30s 估速度）
    t0 = time.perf_counter()
    p = stateless_percall(sess, x[:16000 * 30])
    dt = time.perf_counter() - t0
    print(f"[C hop160 stateless] {dt:.2f}s for 30s (RTF {dt/30:.4f}) "
          f"p>=.5={np.mean(p>=.5)*100:.0f}%", flush=True)
    segs = common.post_merge(common.probs_to_segments(p, hop=0.01, total_dur=dur))
    print(f"    segs={len(segs)}: " + ", ".join(f"{s:.1f}-{e:.1f}" for s, e in segs[:8]))


if __name__ == "__main__":
    main()
