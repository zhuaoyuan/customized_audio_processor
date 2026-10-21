"""二阶段说话人归并：用 sherpa 低阈值(过分割)结果 + cam++ 段级 embedding 重新层次聚类。

低 threshold 的 sherpa 输出会把同一个人拆成很多小簇（见 tune 表）。这里对每个小簇用
其全部语音段加权平均 embedding，再做一次 average-linkage 余弦层次聚类，把碎片并回同一人。
好处：一次分割 pass 就能扫多个归并阈值，且不受 sherpa FastClustering 内部行为限制。

输出:
  merged_th{TH}.json   segments: [{start,end,speaker,dur}] （主讲人=0）
  cluster_report.json  每个归并簇的时长/段数/到主讲人余弦/示例段
"""
import argparse
import json
import os
import sys
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "bench"))
from audiometrics import read_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(here, "..", "full_16k.wav"))
    ap.add_argument("--diar", default=os.path.join(here, "diar_full_th0.4.json"))
    ap.add_argument("--emb-model", default=os.path.join(here, "models",
                    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-seg", type=float, default=0.3, help="短于该时长的段不参与聚类")
    ap.add_argument("--max-seg", type=float, default=12.0, help="单段取前 N 秒算 embedding")
    ap.add_argument("--sweep", default="0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    ap.add_argument("--merge-threshold", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(here, "cluster_report.json"))
    args = ap.parse_args()

    import sherpa_onnx
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    wav_path = os.path.normpath(args.wav if os.path.isabs(args.wav)
                                else os.path.join(here, "..", "..", args.wav))
    diar_path = os.path.normpath(args.diar if os.path.isabs(args.diar)
                                 else os.path.join(here, "..", "..", args.diar))
    arr, sr = read_wav(wav_path)
    x = arr[:, 0].astype(np.float32)
    dur = len(x) / sr
    segs = [s for s in json.load(open(diar_path, encoding="utf-8"))["segments"]]
    print(f"[in] {diar_path}: {len(segs)} segs, {dur:.1f}s audio", flush=True)

    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=os.path.normpath(os.path.join(here, args.emb_model)),
            num_threads=args.threads, debug=False, provider="cpu"))

    # --- 段级 embedding + 小簇加权质心 ---
    fine = {}
    t0 = time.perf_counter()
    embs, weights = [], []
    for i, s in enumerate(segs):
        d = float(s["end"] - s["start"])
        if d < args.min_seg:
            continue
        a, b = int(s["start"] * sr), min(len(x), int(s["end"] * sr))
        clip = x[a:min(b, a + int(args.max_seg * sr))]
        st = ext.create_stream()
        st.accept_waveform(sr, clip)
        e = np.asarray(ext.compute(st), dtype=np.float32)
        n = np.linalg.norm(e)
        if n > 0:
            e = e / n
        embs.append(e)
        weights.append(d)
        fine.setdefault(s["speaker"], []).append(len(embs) - 1)
        if (i + 1) % 200 == 0:
            print(f"  emb {i+1}/{len(segs)} {time.perf_counter()-t0:.0f}s", flush=True)
    t_emb = time.perf_counter() - t0
    E = np.stack(embs)
    W = np.array(weights, dtype=np.float32)
    print(f"[emb] {E.shape} in {t_emb:.1f}s", flush=True)

    keys = sorted(fine.keys())
    C = []
    for k in keys:
        idx = fine[k]
        v = (E[idx] * W[idx, None]).sum(0)
        n = np.linalg.norm(v)
        C.append(v / n if n > 0 else v)
    C = np.stack(C)
    print(f"[clusters] fine={len(keys)}", flush=True)

    dmat = 1.0 - C @ C.T
    np.fill_diagonal(dmat, 0.0)
    dmat = np.clip(dmat, 0.0, None)
    Z = linkage(squareform(dmat, checks=False), method="average")

    fine_stats = {}
    for s in segs:
        fine_stats[s["speaker"]] = fine_stats.get(s["speaker"], 0.0) + (s["end"] - s["start"])
    fine_dur = np.array([fine_stats[k] for k in keys], dtype=np.float32)

    sweep = []
    for th in [float(v) for v in args.sweep.split(",")]:
        lab = fcluster(Z, t=1.0 - th, criterion="distance")
        n_spk = len(set(lab))
        per = {}
        for k, l in zip(keys, lab):
            per[int(l)] = per.get(int(l), 0.0) + float(fine_dur[keys.index(k)])
        tot = sum(per.values())
        top = sorted(per.items(), key=lambda z: -z[1])
        sweep.append({"merge_threshold": th, "n_speakers": n_spk,
                      "speech_s": round(tot, 1),
                      "main_s": round(top[0][1], 1),
                      "main_ratio": round(top[0][1] / tot, 4) if tot else 0,
                      "n_ge_10s": sum(1 for v in per.values() if v >= 10),
                      "top3_s": [round(v, 1) for _, v in top[:3]]})
        print(f"[sweep] th={th:.2f} -> {n_spk} speakers, main={top[0][1]:.0f}s "
              f"({top[0][1]/tot*100:.1f}% of speech), top3={[round(v,1) for _,v in top[:3]]}", flush=True)

    # --- 用指定阈值生成最终标注 ---
    th = args.merge_threshold
    lab = fcluster(Z, t=1.0 - th, criterion="distance")
    lab_of_fine = {k: int(l) for k, l in zip(keys, lab)}
    # 以时长最大簇为 0（主讲人）
    merged_dur = {}
    for s in segs:
        if s["speaker"] not in lab_of_fine:
            continue
        l = lab_of_fine[s["speaker"]]
        merged_dur[l] = merged_dur.get(l, 0.0) + (s["end"] - s["start"])
    main_l = max(merged_dur, key=merged_dur.get)
    remap = {l: (0 if l == main_l else i + 1) for i, l in enumerate(
        [l for l in merged_dur if l != main_l])}
    remap[main_l] = 0

    # 短段/未参与聚类的细簇：归到最近的已聚类细簇（按质心距离）
    out_segs = []
    for s in segs:
        f = s["speaker"]
        if f in lab_of_fine:
            l = remap[lab_of_fine[f]]
        else:
            # 找 embedding 最近的 fine cluster（无 embedding 就丢给主讲人）
            l = 0
        out_segs.append({"start": round(s["start"], 3), "end": round(s["end"], 3),
                         "dur": round(s["end"] - s["start"], 3), "speaker": int(l)})

    # 簇报告
    main_C = C[keys.index([k for k in keys if lab_of_fine[k] == main_l][0])] if any(
        lab_of_fine[k] == main_l for k in keys) else C[0]
    report = []
    for l in sorted(set(remap.values())):
        fk = [k for k in keys if lab_of_fine.get(k) is not None and remap[lab_of_fine[k]] == l]
        if not fk:
            continue
        dur_s = sum(fine_stats[k] for k in fk)
        # 该合并簇所有段的 embedding 平均，与主讲人质心余弦
        idxs = [i for k in fk for i in fine[k]]
        v = (E[idxs] * W[idxs, None]).sum(0)
        v /= max(np.linalg.norm(v), 1e-9)
        report.append({
            "speaker": l, "fine_clusters": len(fk), "dur_s": round(dur_s, 1),
            "ratio_speech": round(dur_s / sum(fine_stats.values()), 4),
            "cos_to_main": round(float(v @ main_C), 3),
            "n_segments": sum(1 for s in out_segs if s["speaker"] == l),
            "longest_seg": round(max((s["end"] - s["start"]) for s in segs
                                     if s["speaker"] in fk), 2) if fk else 0,
        })
    report.sort(key=lambda r: -r["dur_s"])

    res = {
        "wav": wav_path, "diar_in": diar_path, "dur": dur,
        "fine_clusters": len(keys), "n_segments": len(segs),
        "emb_s": round(t_emb, 1), "merge_threshold": th,
        "n_speakers": len(report), "sweep": sweep,
        "main_speaker": 0,
        "main_speech_s": round(report[0]["dur_s"], 1) if report else 0,
        "main_ratio_speech": report[0]["ratio_speech"] if report else 0,
        "clusters": report,
        "segments": out_segs,
    }
    outp = os.path.splitext(args.out)[0] + f"_th{th}.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in res.items() if k != "segments"}, f, ensure_ascii=False, indent=1)
    # make_clean 期望 {"segments":[...]}
    with open(os.path.splitext(args.out)[0] + f"_th{th}_diar.json", "w", encoding="utf-8") as f:
        json.dump({"segments": out_segs, "num_speakers": len(report),
                   "merge_threshold": th, "fine_clusters": len(keys)}, f, ensure_ascii=False)
    # 每个细簇的质心，便于复用
    np.savez_compressed(os.path.splitext(args.out)[0] + "_centroids.npz",
                        keys=np.array(keys), C=C, fine_dur=fine_dur)
    print(f"[out] {outp}")
    print(f"[out] {os.path.splitext(args.out)[0]}_th{th}_diar.json")
    print("[clusters] " + " | ".join(
        f"S{r['speaker']}: {r['dur_s']:.0f}s({r['ratio_speech']*100:.1f}%) cos={r['cos_to_main']:.2f}"
        f" segs={r['n_segments']}" for r in report[:12]))


if __name__ == "__main__":
    main()
