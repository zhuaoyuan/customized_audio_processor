"""对归并后的每个说话人簇做"是否真人语音"判定，生成最终标注。

判据（客观、可复现）：
  1) 到主讲人质心的 cam++ 余弦（cos_to_main）——<0.7 视为不同人；
  2) 该簇语音的 ASR 字符率（chars / 解码秒数）——非语音（音乐/掌声/噪声）ASR 几乎无输出；
  3) 簇总时长（union，非段长相加）。

分类：
  main      主讲人（含 cos 高、有讲课文本的碎片簇，修复过分割）
  speech    其他真人说话（学生插话等）
  nonspeech 非语音（音乐/掌声/环境噪声）

输出 final_diar.json（main=0, speech=1.., nonspeech=99）+ cluster_classification.json
"""
import argparse
import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "bench"))
from audiometrics import read_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(here, "..", "full_16k.wav"))
    ap.add_argument("--diar", default=os.path.join(here, "cluster_report_th0.7_diar.json"))
    ap.add_argument("--report", default=os.path.join(here, "cluster_report_th0.7.json"))
    ap.add_argument("--model-dir", default=os.path.join(here, "..", "bench", "models", "sensevoice"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--asr-cap", type=float, default=120.0, help="每簇最多解码秒数（跨时间均匀取段）")
    ap.add_argument("--cos-main", type=float, default=0.70, help="cos >= 该值且有文本 -> 并入主讲人")
    ap.add_argument("--chars-per-s", type=float, default=0.5, help="char/s 低于该值 -> 疑似非语音")
    ap.add_argument("--nonspeech-min-dur", type=float, default=60.0,
                    help="只有 union 时长 >= 该值的低文本率簇才判为非语音")
    ap.add_argument("--out", default=os.path.join(here, "final_diar.json"))
    args = ap.parse_args()

    import sherpa_onnx

    wav_path = os.path.normpath(args.wav if os.path.isabs(args.wav)
                                else os.path.join(here, "..", "..", args.wav))
    diar_path = os.path.normpath(args.diar if os.path.isabs(args.diar)
                                 else os.path.join(here, "..", "..", args.diar))
    rep_path = os.path.normpath(args.report if os.path.isabs(args.report)
                                else os.path.join(here, "..", "..", args.report))
    arr, sr = read_wav(wav_path)
    x = arr[:, 0].astype(np.float32)
    n = len(x)
    dur = n / sr
    segs = json.load(open(diar_path, encoding="utf-8"))["segments"]
    rep = {r["speaker"]: r for r in json.load(open(rep_path, encoding="utf-8"))["clusters"]}

    md = os.path.normpath(args.model_dir if os.path.isabs(args.model_dir)
                          else os.path.join(here, "..", "..", args.model_dir))
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=os.path.join(md, "model.int8.onnx"), tokens=os.path.join(md, "tokens.txt"),
        num_threads=args.threads, use_itn=True, language="zh", debug=False)

    per = {}
    for s in segs:
        per.setdefault(s["speaker"], []).append(s)

    hop = int(sr * 0.01)
    rows = []
    for k, ss in sorted(per.items(), key=lambda kv: -sum(z["dur"] for z in kv[1])):
        m = np.zeros(int(dur / 0.01) + 1, dtype=bool)
        for s in ss:
            a, b = max(0, int(round(s["start"] / 0.01))), int(round(s["end"] / 0.01))
            m[a:b] = True
        union = float(m.sum() * 0.01)
        # 跨时间均匀取段做 ASR
        ss_sorted = sorted(ss, key=lambda z: z["start"])
        budget = args.asr_cap
        chosen, acc = [], 0.0
        if ss_sorted:
            # 均匀抽 ~12 段
            idxs = np.linspace(0, len(ss_sorted) - 1, min(12, len(ss_sorted))).astype(int)
            for i in idxs:
                s = ss_sorted[int(i)]
                a, b = int(s["start"] * sr), min(n, int(s["end"] * sr))
                b = min(b, a + int((budget - acc) * sr))
                if b <= a:
                    continue
                chosen.append(x[a:b])
                acc += (b - a) / sr
                if acc >= budget:
                    break
        text = ""
        if chosen:
            y = np.concatenate(chosen)
            st = rec.create_stream()
            st.accept_waveform(sr, y)
            rec.decode_stream(st)
            text = st.result.text.strip()
        chars = len("".join(text.split()))
        cps = chars / max(acc, 1e-6)
        row = {"speaker": k, "union_s": round(union, 1), "sum_s": round(sum(z["dur"] for z in ss), 1),
               "n_segments": len(ss), "asr_audio_s": round(acc, 1), "asr_chars": chars,
               "chars_per_s": round(cps, 3),
               "cos_to_main": rep.get(k, {}).get("cos_to_main"),
               "f0_hz": None, "label": "", "asr_text": text[:200]}
        rows.append(row)
        print(f"[S{k:>3}] union={union:7.1f}s segs={len(ss):3d} asr={acc:5.1f}s chars={chars:4d} "
              f"cps={cps:.2f} cos={row['cos_to_main']} | {text[:60]}", flush=True)

    main_labels = {0}
    for r in rows:
        if r["speaker"] == 0:
            r["label"] = "main"
        elif r["cos_to_main"] is not None and r["cos_to_main"] >= args.cos_main \
                and r["chars_per_s"] >= args.chars_per_s:
            r["label"] = "main"
            main_labels.add(r["speaker"])
        elif r["union_s"] >= args.nonspeech_min_dur and r["chars_per_s"] < args.chars_per_s:
            r["label"] = "nonspeech"
        else:
            r["label"] = "speech"

    # 重新编号：main=0, speech=1..N, nonspeech=99
    speech_labels = [r["speaker"] for r in rows if r["label"] == "speech"]
    remap = {k: 0 for k in main_labels}
    for i, k in enumerate(speech_labels, 1):
        remap[k] = i
    for r in rows:
        if r["label"] == "nonspeech":
            remap[r["speaker"]] = 99

    out_segs = []
    for s in segs:
        out_segs.append({"start": round(s["start"], 3), "end": round(s["end"], 3),
                         "dur": round(s["end"] - s["start"], 3),
                         "speaker": int(remap.get(s["speaker"], 99))})

    summary = {
        "wav": wav_path, "diar_in": diar_path, "dur": dur,
        "cos_main_th": args.cos_main, "chars_per_s_th": args.chars_per_s,
        "nonspeech_min_dur": args.nonspeech_min_dur,
        "n_main_clusters": len(main_labels), "n_speech_clusters": len(speech_labels),
        "n_nonspeech_clusters": sum(1 for r in rows if r["label"] == "nonspeech"),
        "clusters": rows,
    }
    with open(os.path.splitext(args.out)[0] + "_classification.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"segments": out_segs, "n_speakers": 1 + len(speech_labels),
                   "main": 0, "nonspeech_label": 99,
                   "main_clusters": sorted(main_labels),
                   "speech_speakers": speech_labels}, f, ensure_ascii=False)
    print(f"\n[final] main clusters={sorted(main_labels)} speech={len(speech_labels)} "
          f"nonspeech={summary['n_nonspeech_clusters']}")
    print(f"[out] {args.out}")
    print(f"[out] {os.path.splitext(args.out)[0]}_classification.json")

    # 记录人工可核对的分段
    cls = {r["speaker"]: r["label"] for r in rows}
    for line in out_segs[:3]:
        pass


if __name__ == "__main__":
    main()
