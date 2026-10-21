# -*- coding: utf-8 -*-
"""说话人标注：两条路线
A) 官方 sherpa-onnx OfflineSpeakerDiarization 结果（orig 时间轴）-> clean 帧标签；
B) VAD 段 + cam++ embedding 聚类（省掉分割模型，用于全部 6 个文件）。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import AE, ROOT, map_orig_to_clean  # noqa: E402

EMB_MODEL = os.path.join(ROOT, "work", "diar", "models",
                         "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
HOP = 0.01  # 10ms 帧


class Embedder:
    def __init__(self, num_threads=4):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL, num_threads=num_threads)
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, x):
        st = self.ex.create_stream()
        st.accept_waveform(16000, np.ascontiguousarray(x, dtype=np.float32))
        st.input_finished()
        e = np.asarray(self.ex.compute(st), dtype=np.float32)
        n = float(np.linalg.norm(e))
        return e / (n + 1e-9)


def cluster_segments(x, sr, segs, emb, sim_thresh=None, min_seg=1.5, chunk_s=6.0, hop_s=6.0,
                     main_ratio_target=0.98):
    """对 VAD 段做 embedding 聚类（平均链接 + “主讲人占绝对多数”先验选阈值）。

    cam++ 在 3s 短窗上的同人余弦只有 ~0.5-0.65，直接用固定 0.75 会把同一人切碎。
    因此用平均链接层次聚类，从最细（最大距离阈值）往粗扫，取“最大簇仍 >= main_ratio_target
    的语音时长”的最细一档；这样既保留真实差异，又不把主讲人切碎。
    返回 (seg_labels, info)；seg_labels 0=主讲（按时长自动编号）。
    """
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    chunks = []  # (seg_idx, duration_weight, emb)
    for i, (s, e) in enumerate(segs):
        d = e - s
        if d < min_seg:
            continue
        if d <= chunk_s:
            w = x[int(s * sr): int(e * sr)]
            chunks.append((i, d, emb.embed(w)))
        else:
            t = s
            while t + 1.0 < e:
                t2 = min(e, t + chunk_s)
                w = x[int(t * sr): int(t2 * sr)]
                if len(w) >= int(1.5 * sr):
                    chunks.append((i, min(t2 - t, chunk_s), emb.embed(w)))
                t += hop_s
    if not chunks:
        return np.zeros(len(segs), dtype=int), {"method": "none", "n_speakers": 1}
    E = np.stack([c[2] for c in chunks])
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    Z = linkage(pdist(E, metric="cosine"), method="average")

    def labels_at(t):
        return fcluster(Z, t=max(1e-6, t), criterion="distance")

    best = None
    cand = []
    scan = np.arange(0.20, 0.96, 0.02) if sim_thresh is None else np.array([1.0 - sim_thresh])
    for t in scan:
        lab = labels_at(t)
        wdur = {}
        nch = {}
        for (si, dw, _), lb in zip(chunks, lab):
            wdur[int(lb)] = wdur.get(int(lb), 0.0) + dw
            nch[int(lb)] = nch.get(int(lb), 0) + 1
        tot = sum(wdur.values())
        mratio = max(wdur.values()) / tot if tot else 1.0
        cand.append((float(t), len(wdur), round(mratio, 4)))
        if mratio >= main_ratio_target and best is None:
            best = (float(t), lab, wdur, mratio)
            break
    if best is None:
        t = 0.20
        lab = labels_at(t)
        wdur = {}
        for (si, dw, _), lb in zip(chunks, lab):
            wdur[int(lb)] = wdur.get(int(lb), 0.0) + dw
        best = (t, lab, wdur, max(wdur.values()) / sum(wdur.values()) if wdur else 1.0)
    tbest, lab, wdur, mratio = best
    # 段 -> 簇（其 chunk 的多数投票）
    seg_chunks = {}
    for (si, _, _), lb in zip(chunks, lab):
        seg_chunks.setdefault(si, []).append(int(lb))
    seg_labels = np.full(len(segs), -1, dtype=int)
    for si, lbs in seg_chunks.items():
        vals, cnt = np.unique(lbs, return_counts=True)
        seg_labels[si] = int(vals[np.argmax(cnt)])
    # 无 chunk 的段：与簇质心余弦最近
    uniq = sorted(set(int(v) for v in lab))
    cent = {}
    for u in uniq:
        idx = [k for k, lb in enumerate(lab) if int(lb) == u]
        v = E[idx].mean(axis=0)
        cent[u] = v / (np.linalg.norm(v) + 1e-9)
    for i, (s, e) in enumerate(segs):
        if seg_labels[i] >= 0:
            continue
        w = x[int(s * sr): int(e * sr)]
        if len(w) < int(0.3 * sr):
            seg_labels[i] = -2
            continue
        v = emb.embed(w)
        seg_labels[i] = max(cent, key=lambda u: float(np.dot(v, cent[u])))
    dur = {}
    for i, (s, e) in enumerate(segs):
        if seg_labels[i] >= 0:
            dur[seg_labels[i]] = dur.get(seg_labels[i], 0.0) + (e - s)
    main = max(dur, key=dur.get)
    seg_labels[seg_labels == -2] = main
    order = sorted(dur, key=dur.get, reverse=True)
    remap = {u: k for k, u in enumerate(order)}
    seg_labels = np.array([remap.get(int(v), 0) for v in seg_labels], dtype=int)
    total = sum(dur.values())
    info = {"method": "embedding_agglomerative", "sim_thresh_used": round(1.0 - tbest, 3),
            "dist_thresh_used": round(tbest, 3), "n_speakers": len(order),
            "n_chunks": len(chunks), "main_ratio": round(dur[order[0]] / total, 4) if total else 1.0,
            "cluster_sec": {str(remap[u]): round(v, 2) for u, v in
                            sorted(dur.items(), key=lambda kv: -kv[1])},
            "scan": [{"dist": round(a, 2), "k": b, "main_ratio": c} for a, b, c in cand]}
    return seg_labels, info


def _load_diar(path):
    """兼容两种 diarization JSON：{"segments":[{start,end,spk|speaker}]}。返回 (segs, main_spk, nonspeech)。"""
    d = json.load(open(path, "r", encoding="utf-8"))
    rows = []
    for s in d.get("segments", []):
        spk = s.get("spk", s.get("speaker", 0))
        rows.append({"start": float(s["start"]), "end": float(s["end"]),
                     "dur": float(s.get("dur", s["end"] - s["start"])), "spk": int(spk)})
    per = {}
    for r in rows:
        per[r["spk"]] = per.get(r["spk"], 0.0) + r["dur"]
    nonspeech = set(int(x) for x in d.get("nonspeech_label", []) if x is not None) \
        if isinstance(d.get("nonspeech_label"), list) else set()
    if "nonspeech_label" in d and not isinstance(d["nonspeech_label"], list):
        nonspeech = {int(d["nonspeech_label"])}
    main = d.get("main")
    main = int(main) if main is not None else (max(per, key=per.get) if per else 0)
    return rows, main, nonspeech, per


def labels_from_official(diar_json, kept, clean_n_samples, sr=16000):
    """官方 diarization（orig 时间轴）-> clean 10ms 帧 speaker 标签 + 段级信息。

    返回 (frame_labels, info)；speaker 0 = 主讲，非主讲 1..N；非语音簇并入主讲（-1 表示无标签）。
    """
    segs, main, nonspeech, per = _load_diar(diar_json)
    order = [main] + sorted([k for k in per if k != main], key=lambda k: -per[k])
    remap = {u: k for k, u in enumerate(order)}
    n_frames = int(np.ceil(clean_n_samples / (sr * HOP)))
    labels = np.full(n_frames, -1, dtype=int)
    for s in segs:
        lab = 0 if s["spk"] in nonspeech else remap[s["spk"]]
        for cs, ce in map_orig_to_clean(kept, s["start"], s["end"]):
            a = int(round(cs / HOP))
            b = int(round(ce / HOP))
            if b > a:
                labels[a:b] = lab
    info = {"method": "official_sherpa_offline", "json": os.path.abspath(diar_json),
            "n_speakers": len(per), "n_segments": len(segs),
            "main_label": main, "nonspeech_labels": sorted(nonspeech),
            "per_speaker_sec": {str(remap[k]): round(per[k], 2) for k in per},
            "speech_per_speaker_sec": {str(remap[k]): round(per[k], 2) for k in per
                                       if k not in nonspeech},
            "main_speech_ratio": (round(per[main] / sum(v for k, v in per.items() if k not in nonspeech), 4)
                                  if per else 1.0)}
    return labels, info


def seg_labels_from_official(diar_json, segs, kept):
    """把官方 diarization 映射到 VAD 段级标签（用于 V3 增益分组）。"""
    dsegs, main, nonspeech, per = _load_diar(diar_json)
    order = [main] + sorted([k for k in per if k != main], key=lambda k: -per[k])
    remap = {u: k for k, u in enumerate(order)}
    # clean 帧 -> orig 时间 的变换表
    clean2orig = []
    c = 0.0
    for ks, ke in kept:
        dlen = ke - ks
        clean2orig.append((c, c + dlen, ks))
        c += dlen

    def clean_to_orig(t):
        for cs, ce, ks in clean2orig:
            if cs <= t < ce:
                return ks + (t - cs)
        return None

    dsegs = sorted(dsegs, key=lambda s: s["start"])
    import bisect
    starts = [s["start"] for s in dsegs]

    def orig_to_spk(t):
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and dsegs[i]["start"] <= t < dsegs[i]["end"]:
            spk = dsegs[i]["spk"]
            return 0 if spk in nonspeech else remap[spk]
        return 0

    out = np.zeros(len(segs), dtype=int)
    for i, (s, e) in enumerate(segs):
        om = clean_to_orig((s + e) / 2.0)
        out[i] = 0 if om is None else orig_to_spk(om)
    return out, {"method": "official_sherpa_offline", "json": os.path.abspath(diar_json),
                 "n_speakers": len(per), "main_label": main}


def seg_labels_from_official(diar_json, segs, kept):
    """把官方 diarization 映射到 VAD 段级标签（用于 V3 增益分组）。"""
    rows, main, nonspeech, per = _load_diar(diar_json)
    order = [main] + sorted([k for k in per if k != main], key=lambda k: -per[k])
    remap = {u: k for k, u in enumerate(order)}
    # clean 帧 -> orig 时间 的变换表
    clean2orig = []
    c = 0.0
    for ks, ke in kept:
        dlen = ke - ks
        clean2orig.append((c, c + dlen, ks))
        c += dlen

    def clean_to_orig(t):
        for cs, ce, ks in clean2orig:
            if cs <= t < ce:
                return ks + (t - cs)
        return None

    # orig 时间 -> spk 标签（线性扫描，diar 段已按时间排序）
    dsegs = sorted(rows, key=lambda s: s["start"])
    import bisect
    starts = [s["start"] for s in dsegs]

    def orig_to_spk(t):
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and dsegs[i]["start"] <= t < dsegs[i]["end"]:
            spk = dsegs[i]["spk"]
            return 0 if spk in nonspeech else remap[spk]
        return 0

    out = np.zeros(len(segs), dtype=int)
    for i, (s, e) in enumerate(segs):
        om = clean_to_orig((s + e) / 2.0)
        out[i] = 0 if om is None else orig_to_spk(om)
    return out, {"method": "official_sherpa_offline", "json": os.path.abspath(diar_json),
                 "n_speakers": len(per), "main_label": main}
