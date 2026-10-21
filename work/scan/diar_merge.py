# -*- coding: utf-8 -*-
"""对 diar 原始聚类做“同人合并”：用 CAM++ 说话人 embedding 计算每个簇的加权质心，
再按余弦相似度 > TH 合并过度聚类的簇。

输入 work/scan/diar/<short>.json（diar10.py 产出），输出 <short>_merged.json。
"""
import argparse
import json
import os

import numpy as np

ROOT = r"C:\projects\local\audio_processer\work"
SCAN = os.path.join(ROOT, "scan")
EMB = os.path.join(ROOT, "diar", "models", "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
SR = 16000
SHORTS = ["f00_xifang20140227", "f20_lixiangguo20141219", "f24_xingershangxue",
          "f04_lixiangguo4", "f05_wulixue", "f06_lunyu"]


def read_slice(path, start_s, dur_s):
    from scipy.io import wavfile
    sr, d = wavfile.read(path, mmap=True)
    a = int(start_s * sr)
    b = min(len(d), a + int(dur_s * sr))
    return np.ascontiguousarray(d[a:b], dtype=np.float32), sr


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--th", type=float, default=0.70)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    todo = [s for s in (args.only.split(",") if args.only else SHORTS) if s]

    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=EMB, num_threads=4, debug=False, provider="cpu")
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    probe = json.load(open(os.path.join(SCAN, "probe.json"), encoding="utf-8"))
    for short in todo:
        path = os.path.join(SCAN, "diar", short + ".json")
        if not os.path.isfile(path):
            print("[skip] no diar json", short)
            continue
        d = json.load(open(path, encoding="utf-8"))
        wav = probe["files"][short]["wav"]
        x, sr = read_slice(wav, d["start_s"], d["dur_s"])

        # 每个 segment 的 embedding（只取 >=0.4s 的段）
        embs, wts, raws, durs = [], [], [], []
        for s in d["segments"]:
            du = s["dur"]
            if du < 0.4:
                continue
            a = int(max(0.0, s["start"]) * sr)
            b = min(len(x), int(s["end"] * sr))
            if b - a < int(0.4 * sr):
                continue
            st = ext.create_stream()
            st.accept_waveform(sr, np.ascontiguousarray(x[a:b]))
            st.input_finished()
            e = unit(ext.compute(st))
            embs.append(e)
            wts.append(du)
            raws.append(s["speaker"])
            durs.append(du)
        if not embs:
            print("[warn] no embeddings", short)
            continue
        E = np.array(embs)
        W = np.array(wts)
        R = np.array(raws)

        # 原始簇的加权质心
        uniq = sorted(set(R.tolist()))
        cents, weights, dur_sum, members = [], [], [], []
        for k in uniq:
            sel = R == k
            w = W[sel]
            c = unit(np.sum(E[sel] * w[:, None], axis=0))
            cents.append(c)
            weights.append(float(w.sum()))
            dur_sum.append(float(W[sel].sum()))
            members.append([int(k)])
        C = np.array(cents)
        Wt = np.array(weights)
        Dur = np.array(dur_sum)
        Mem = members

        # 迭代合并：质心余弦 > TH
        while len(C) > 1:
            sim = C @ C.T
            np.fill_diagonal(sim, -2)
            i, j = np.unravel_index(np.argmax(sim), sim.shape)
            if sim[i, j] <= args.th:
                break
            wsum = Wt[i] + Wt[j]
            c = unit((C[i] * Wt[i] + C[j] * Wt[j]) / wsum)
            keep, drop = (i, j) if i < j else (j, i)
            C[keep] = c
            Wt[keep] = wsum
            Dur[keep] = Dur[keep] + Dur[drop]
            Mem[keep] = Mem[keep] + Mem[drop]
            C = np.delete(C, drop, axis=0)
            Wt = np.delete(Wt, drop)
            Dur = np.delete(Dur, drop)
            Mem.pop(drop)

        order = np.argsort(-Dur)
        total = float(Dur.sum())
        merged = []
        for idx in order:
            merged.append({"raw_clusters": Mem[idx], "dur_s": round(float(Dur[idx]), 1),
                           "share": float(Dur[idx]) / total if total else 0.0})
        main = merged[0] if merged else None
        nonmain = total - (main["dur_s"] if main else 0.0)
        effective = [m for m in merged if m["share"] >= 0.05]
        res = {
            "short": short, "start_s": d["start_s"], "dur_s": d["dur_s"],
            "merge_th_cosine": args.th,
            "raw_num_speakers": d["num_speakers"],
            "raw_n_segments": d["n_segments"],
            "merged_num_speakers": len(merged),
            "merged_speakers_ge1s": int(sum(1 for m in merged if m["dur_s"] >= 1.0)),
            "num_effective_speakers": len(effective),
            "effective_share_min": 0.05,
            "speech_total_s": round(total, 1),
            "main_dur_s": main["dur_s"] if main else None,
            "main_share_of_speech": main["share"] if main else None,
            "nonmain_speech_s": round(nonmain, 1),
            "merged_speakers": merged,
            "raw_per_speaker_s": d["per_speaker_s"],
        }
        json.dump(res, open(os.path.join(SCAN, "diar", short + "_merged.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"[merge] {short}: raw={d['num_speakers']} -> merged={len(merged)} "
              f"(>=1s: {res['merged_speakers_ge1s']}, eff>=5%: {len(effective)}) "
              f"main={res['main_share_of_speech']*100:.1f}% "
              f"nonmain={nonmain:.0f}s dur={[m['dur_s'] for m in merged[:6]]}", flush=True)


if __name__ == "__main__":
    main()
