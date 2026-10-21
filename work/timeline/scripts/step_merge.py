"""Step 3b: 二阶段说话人归并（对 threshold=0.7 的原始簇再做 cam++ 质心层次聚类）。

用途：threshold=0.7 直跑仍会把同一个人拆成若干 id（短段 embedding 不稳），
用段级 cam++ embedding 按时长加权求每簇质心，再做 average-linkage 余弦聚类，
把碎片并回同一人。一次算完可扫多个归并阈值，报告"合并前后"的说话人数差异。

用法:
  python -X utf8 step_merge.py --wav <16k wav> --diar diar.json --out merge.json
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--diar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb-model", default=common.DIAR_EMB_MODEL)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-seg", type=float, default=0.3)
    ap.add_argument("--max-seg", type=float, default=12.0)
    ap.add_argument("--sweep", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--merge-threshold", type=float, default=0.7)
    args = ap.parse_args()

    import sherpa_onnx
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    arr, sr = common.read_wav(args.wav)
    x = arr[:, 0].astype(np.float32)
    diar = common.read_json(args.diar)
    segs = diar["segments"]
    print(f"[in] {common.rel(args.diar)}: {len(segs)} segs, {len(x)/sr:.1f}s", flush=True)

    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.emb_model, num_threads=args.threads, debug=False, provider="cpu"))

    t0 = time.perf_counter()
    embs, weights, owner = [], [], []
    for i, s in enumerate(segs):
        d = float(s["end"] - s["start"])
        if d < args.min_seg:
            continue
        a = int(s["start"] * sr)
        b = min(len(x), int(s["end"] * sr))
        clip = x[a:min(b, a + int(args.max_seg * sr))]
        if len(clip) < int(0.2 * sr):
            continue
        st = ext.create_stream()
        st.accept_waveform(sr, clip)
        e = np.asarray(ext.compute(st), dtype=np.float32).reshape(-1)
        n = np.linalg.norm(e)
        if n > 0:
            e = e / n
        embs.append(e)
        weights.append(d)
        owner.append(s["speaker"])
        if (i + 1) % 400 == 0:
            print(f"  emb {i+1}/{len(segs)} {time.perf_counter()-t0:.0f}s", flush=True)
    t_emb = time.perf_counter() - t0
    E = np.stack(embs)
    W = np.array(weights, dtype=np.float32)
    print(f"[emb] {E.shape} in {t_emb:.1f}s", flush=True)

    # 每原始簇质心
    fine = {}
    for i, spk in enumerate(owner):
        fine.setdefault(int(spk), []).append(i)
    keys = sorted(fine.keys())
    C = []
    for k in keys:
        idx = fine[k]
        v = (E[idx] * W[idx, None]).sum(0)
        n = np.linalg.norm(v)
        C.append(v / n if n > 0 else v)
    C = np.stack(C)

    fine_dur = {}
    fine_n = {}
    for s in segs:
        fine_dur[int(s["speaker"])] = fine_dur.get(int(s["speaker"]), 0.0) + s["dur"]
        fine_n[int(s["speaker"])] = fine_n.get(int(s["speaker"]), 0) + 1
    fine_dur_v = np.array([fine_dur[k] for k in keys], dtype=np.float32)

    dmat = 1.0 - C @ C.T
    np.fill_diagonal(dmat, 0.0)
    dmat = np.clip(dmat, 0.0, None)
    Z = linkage(squareform(dmat, checks=False), method="average")

    sweep = []
    for th in [float(v) for v in args.sweep.split(",")]:
        lab = fcluster(Z, t=1.0 - th, criterion="distance")
        per = {}
        for k, l in zip(keys, lab):
            per[int(l)] = per.get(int(l), 0.0) + float(fine_dur[k])
        tot = sum(per.values())
        top = sorted(per.values(), reverse=True)
        sweep.append({"merge_threshold": th, "n_speakers": len(set(lab)),
                      "speech_s": round(tot, 1),
                      "main_s": round(top[0], 1) if top else 0,
                      "main_ratio": round(top[0] / tot, 4) if top else 0,
                      "n_ge_3s": int(sum(1 for v in per.values() if v >= 3)),
                      "n_ge_10s": int(sum(1 for v in per.values() if v >= 10))})
        print(f"[sweep] th={th:.2f} -> {len(set(lab))} speakers, "
              f"main={top[0]:.0f}s ({top[0]/tot*100:.1f}%)", flush=True)

    th = args.merge_threshold
    lab = fcluster(Z, t=1.0 - th, criterion="distance")
    lab_of = {k: int(l) for k, l in zip(keys, lab)}
    merged_dur = {}
    for s in segs:
        k = int(s["speaker"])
        if k in lab_of:
            merged_dur[lab_of[k]] = merged_dur.get(lab_of[k], 0.0) + s["dur"]
    main_l = max(merged_dur, key=merged_dur.get)
    others = [l for l in sorted(merged_dur, key=lambda z: -merged_dur[z]) if l != main_l]
    remap = {main_l: 0}
    for i, l in enumerate(others):
        remap[l] = i + 1

    out_segs = []
    for s in segs:
        k = int(s["speaker"])
        l = remap.get(lab_of.get(k), -1)
        out_segs.append({"start": s["start"], "end": s["end"], "dur": s["dur"],
                         "raw_speaker": k, "speaker": int(l)})

    main_C = C[keys.index(next(k for k in keys if lab_of[k] == main_l))]
    # 原始簇"抖动"诊断：多少小簇 / 多少簇其实与主讲人高相似（= 同一个人被拆开的证据）
    main_key = next(k for k in keys if lab_of[k] == main_l)
    raw_diag = []
    for k in keys:
        raw_diag.append({"raw_speaker": k, "dur_s": round(float(fine_dur[k]), 2),
                         "n_segments": int(fine_n[k]),
                         "cos_to_main": round(float(C[keys.index(k)] @ main_C), 3),
                         "merged_speaker": int(remap.get(lab_of.get(k), -1))})
    n_lt3 = sum(1 for r in raw_diag if r["dur_s"] < 3.0)
    dur_lt3 = sum(r["dur_s"] for r in raw_diag if r["dur_s"] < 3.0)
    n_cos06 = sum(1 for r in raw_diag if r["raw_speaker"] != main_key and r["cos_to_main"] >= 0.6)
    dur_cos06 = sum(r["dur_s"] for r in raw_diag
                    if r["raw_speaker"] != main_key and r["cos_to_main"] >= 0.6)
    print(f"[raw-diag] {len(keys)} raw clusters: <3s={n_lt3} ({dur_lt3:.0f}s), "
          f"cos_to_main>=0.6={n_cos06} ({dur_cos06:.0f}s)", flush=True)
    report = []
    for l in sorted(set(remap.values())):
        fk = [k for k in keys if lab_of.get(k) is not None and remap[lab_of[k]] == l]
        dur_s = sum(fine_dur[k] for k in fk)
        idxs = [i for k in fk for i in fine[k]]
        v = (E[idxs] * W[idxs, None]).sum(0)
        v /= max(np.linalg.norm(v), 1e-9)
        report.append({
            "speaker": l, "raw_clusters": len(fk), "dur_s": round(dur_s, 1),
            "n_segments": sum(1 for s in out_segs if s["speaker"] == l),
            "cos_to_main": round(float(v @ main_C), 3),
            "longest_seg_s": round(max(s["dur"] for s in segs if int(s["speaker"]) in fk), 2),
        })
    report.sort(key=lambda r: -r["dur_s"])

    res = {
        "wav": common.rel(args.wav), "diar": common.rel(args.diar),
        "raw_speakers": len(keys), "raw_segments": len(segs),
        "emb_s": round(t_emb, 1), "merge_threshold": th,
        "n_speakers": len(report), "sweep": sweep,
        "raw_diag": {"n_raw": len(keys), "n_raw_lt3s": n_lt3,
                     "dur_raw_lt3s": round(dur_lt3, 1),
                     "n_cos_to_main_ge_0.6": n_cos06,
                     "dur_cos_to_main_ge_0.6": round(dur_cos06, 1),
                     "clusters": raw_diag},
        "clusters": report,
        "segments": out_segs,
    }
    common.write_json(args.out, res)
    print(f"[merge] raw={len(keys)} -> merged={len(report)} (cos threshold {th}); "
          f"main={report[0]['dur_s']:.0f}s", flush=True)
    print("[clusters] " + " | ".join(
        f"S{r['speaker']}: {r['dur_s']:.0f}s raw={r['raw_clusters']} "
        f"cos={r['cos_to_main']:.2f} segs={r['n_segments']}" for r in report[:10]), flush=True)
    print(f"[out] {common.rel(args.out)}")


if __name__ == "__main__":
    main()
