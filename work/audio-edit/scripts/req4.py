# -*- coding: utf-8 -*-
"""需求4 保守处理无关人声：在 xes（官方 diarization 可得）上给出“高置信非讲课内容”候选清单。

策略：默认保留；只标记两类可删候选：
 ① 非主讲说话人 且 时长>=5s 且 ASR 字符率<0.5 字/秒
 ② VAD 判语音但 ASR 无输出（远场/音乐/误检）
片头片尾由裁剪环节自动处理，本脚本只统计。
"""
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import AE, ROOT, read_wav  # noqa: E402

CJK = re.compile(r"[\u4e00-\u9fff]")
ALNUM = re.compile(r"[0-9A-Za-z]")
SENT = re.compile(r"[。，！？、,.!?;；:：]")
MODEL = os.path.join(ROOT, "work", "bench", "models", "sensevoice")
DIAR = os.path.join(ROOT, "work", "diar", "final_diar.json")


def count_chars(t):
    return len(CJK.findall(t)) + len(ALNUM.findall(t))


def load_recognizer():
    import sherpa_onnx
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=os.path.join(MODEL, "model.int8.onnx"),
        tokens=os.path.join(MODEL, "tokens.txt"),
        num_threads=4, use_itn=True, language="zh", debug=False)


def asr(rec, samples, sr):
    s = rec.create_stream()
    s.accept_waveform(sr, np.ascontiguousarray(samples, dtype=np.float32))
    rec.decode_stream(s)
    return s.result.text


def overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def main():
    sr, x = read_wav(os.path.join(AE, "wav", "xes.wav"))
    d = json.load(open(DIAR, "r", encoding="utf-8"))
    main_spk = d["main"]
    nonspeech = d.get("nonspeech_label", 99)
    per = {}
    for s in d["segments"]:
        per[s["speaker"]] = per.get(s["speaker"], 0.0) + s["dur"]
    print(f"[diar] speakers={len(per)} main={main_spk} ({per[main_spk]:.0f}s) "
          f"nonspeech={nonspeech} ({per.get(nonspeech, 0):.0f}s)")

    segs = [(s["start"], s["end"], s["speaker"], s["dur"]) for s in d["segments"]]
    # 候选 A：非主讲、非非语音簇、时长>=5s
    cand_a = [s for s in segs if s[2] != main_spk and s[2] != nonspeech and s[3] >= 5.0]
    print(f"[candA] 非主讲且>=5s: {len(cand_a)} 段, 共 {sum(s[3] for s in cand_a):.1f}s")

    # VAD 语音段（用于候选 B 与边界核对）
    from common import classify_segments, detect_speech
    den = os.path.join(AE, "denoised", "xes_afftdn.wav")
    if os.path.exists(den):
        _, xv = read_wav(den)
        if len(xv) > len(x):
            xv = xv[: len(x)]
        else:
            xv = np.pad(xv, (0, len(x) - len(xv)))
    else:
        xv = x
    vad = detect_speech(xv, sr)
    cont, frag = classify_segments(vad, 3.0)
    print(f"[vad] segs={len(vad)} cont={len(cont)}")

    rec = load_recognizer()
    rows = []
    t0 = time.perf_counter()

    def run_one(a, b, spk, reason_hint):
        a2 = max(0.0, a - 0.1)
        b2 = min(len(x) / sr, b + 0.1)
        clip = x[int(a2 * sr): int(b2 * sr)]
        txt = asr(rec, clip, sr).strip()
        chars = count_chars(txt)
        rate = chars / max(0.1, b - a)
        spk_name = ("非主讲S" + str(spk)) if spk != main_spk else ("主讲" if spk == main_spk else "非语音簇")
        if spk == nonspeech:
            spk_name = "非语音簇(音乐/噪声)"
        row = {"start_s": round(a, 2), "end_s": round(b, 2), "dur_s": round(b - a, 2),
               "speaker": spk, "speaker_label": spk_name, "asr_text": txt,
               "n_chars": chars, "char_rate": round(rate, 2), "reason_hint": reason_hint}
        # 判定
        if spk == nonspeech:
            row["decision"] = "keep(默认保留)"
            row["reason"] = "官方 diar 判非语音簇，属片头尾音乐/噪声；已在裁剪头尾处理"
        elif chars == 0:
            row["decision"] = "candidate_remove"
            row["reason"] = "VAD 判语音但 ASR 无输出（疑似远场/音乐/误检）"
        elif spk != main_spk and (b - a) >= 5.0 and rate < 0.5:
            row["decision"] = "candidate_remove"
            row["reason"] = f"非主讲 + >=5s + ASR字符率 {rate:.2f}<0.5 字/s"
        elif spk != main_spk:
            row["decision"] = "keep(默认保留)"
            row["reason"] = "非主讲但 ASR 有实质内容或时长/字符率不满足删除阈值"
        else:
            row["decision"] = "keep"
            row["reason"] = "主讲"
        rows.append(row)
        return row

    # A 组（全部 ≥5s 非主讲；通常数量不多）
    for (a, b, spk, dur) in cand_a:
        r = run_one(a, b, spk, "non_main_ge5s")
        print(f"  [A] {a:7.1f}-{b:7.1f} {r['speaker_label']:<18} chars={r['n_chars']:>2} "
              f"rate={r['char_rate']:.2f} -> {r['decision']} | {r['asr_text'][:40]}")
    # B 组：VAD 语音段中落在非语音簇里、或时长短的疑似远场段（抽样：取与 A 不重叠的前 8 段）
    seen = [(r["start_s"], r["end_s"]) for r in rows]
    n_b = 0
    for (a, b) in sorted(vad):
        if n_b >= 8:
            break
        if any(overlap(a, b, s, e) > 0.3 for s, e in seen):
            continue
        # 该 VAD 段对应的 diar 说话人（最大重叠）
        ov = {}
        for (ds, de, spk, dur) in segs:
            o = overlap(a, b, ds, de)
            if o > 0:
                ov[spk] = ov.get(spk, 0.0) + o
        spk = max(ov, key=ov.get) if ov else main_spk
        if spk == main_spk:
            continue
        r = run_one(a, b, spk, "vad_seg_non_main")
        print(f"  [B] {a:7.1f}-{b:7.1f} {r['speaker_label']:<18} chars={r['n_chars']:>2} "
              f"rate={r['char_rate']:.2f} -> {r['decision']} | {r['asr_text'][:40]}")
        n_b += 1

    rem = [r for r in rows if r["decision"] == "candidate_remove"]
    res = {"file": "24_形而上学(xes)", "diar_json": DIAR,
           "main_speaker": main_spk,
           "per_speaker_sec": {str(k): round(v, 1) for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:10]},
           "non_main_speech_sec": round(sum(v for k, v in per.items()
                                           if k != main_spk and k != nonspeech), 1),
           "nonspeech_cluster_sec": round(per.get(nonspeech, 0.0), 1),
           "n_checked": len(rows), "n_candidates": len(rem),
           "candidate_remove_sec": round(sum(r["dur_s"] for r in rem), 2),
           "policy": "默认全部保留；仅头尾(裁剪环节)自动删除；候选仅作人工复核清单",
           "rows": rows, "wall_s": round(time.perf_counter() - t0, 1)}
    out = os.path.join(AE, "req4_candidates.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[out] {out} candidates={len(rem)} sec={res['candidate_remove_sec']}")


if __name__ == "__main__":
    main()
