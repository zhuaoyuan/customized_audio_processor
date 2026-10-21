# -*- coding: utf-8 -*-
"""用 xes 的官方 diarization 结果检验 embedding 聚类（主/其他二分 + 时长占比）。"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import AE, ROOT, classify_segments, detect_speech, read_wav  # noqa: E402
import speakers  # noqa: E402


def main():
    start, end = 600.0, 1200.0
    sr, x = read_wav(os.path.join(AE, "wav", "xes.wav"))
    x = x[int(start * sr): int(end * sr)]
    t0 = time.perf_counter()
    segs = detect_speech(x, sr)
    cont, frag = classify_segments(segs, 3.0)
    print(f"[vad] segs={len(segs)} cont={len(cont)} time={time.perf_counter()-t0:.1f}s")

    import json
    d = json.load(open(os.path.join(ROOT, "work", "diar", "final_diar.json"), "r", encoding="utf-8"))
    rows = [(s["start"], s["end"], s["speaker"]) for s in d["segments"]]
    main_spk = d["main"]

    def official_label(s, e):
        # 与 diar 段重叠最多的 speaker
        ov = {}
        for a, b, spk in rows:
            lo, hi = max(s, a), min(e, b)
            if hi > lo:
                ov[spk] = ov.get(spk, 0.0) + (hi - lo)
        if not ov:
            return None
        return max(ov, key=ov.get)

    gold = []
    for s, e in segs:
        spk = official_label(s + start, e + start)
        gold.append(-1 if spk is None else (0 if spk == main_spk else (99 if spk == d["nonspeech_label"] else 1)))
    gold = np.array(gold)

    emb = speakers.Embedder(4)
    t0 = time.perf_counter()
    labels, info = speakers.cluster_segments(x, sr, segs, emb)
    print(f"[cluster] {time.perf_counter()-t0:.1f}s {info['method']} k={info['n_speakers']} "
          f"main_ratio={info['main_ratio']} thresh={info.get('dist_thresh_used')}")
    print(" cluster_sec:", info.get("cluster_sec"))

    # 二分类一致性（gold 0=主讲，1/99=非主讲，-1=无标注）
    g_main = gold == 0
    p_main = labels == 0
    valid = gold >= 0
    print(f"[gold] main_seg={int(g_main.sum())} other={int(((gold==1)|(gold==99)).sum())} none={int((gold==-1).sum())}")
    print(f"[pred] main_seg={int(p_main.sum())} other={int((~p_main).sum())}")
    tp = int((g_main & valid & p_main).sum())
    fn = int((g_main & valid & ~p_main).sum())
    fp = int(((~g_main) & valid & p_main).sum())
    print(f"[agreement] main recall={tp/max(1,tp+fn):.3f} 其他误判为主讲={fp} 主讲漏判={fn}")
    # 时长口径
    gm = sum(e - s for (s, e), g in zip(segs, gold) if g == 0)
    pm = sum(e - s for (s, e), p in zip(segs, labels) if p == 0)
    print(f"[duration] gold_main={gm:.1f}s pred_main={pm:.1f}s total={sum(e-s for s,e in segs):.1f}s")


if __name__ == "__main__":
    main()
