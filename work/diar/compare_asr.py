"""对比 原始 / 干净 / 被切除 三条音轨的同窗口 ASR 文本，量化"丢了什么"。"""
import argparse
import difflib
import json
import os
import re

here = os.path.dirname(os.path.abspath(__file__))


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def norm(t):
    return re.sub(r"[\s，。、？！…—\-—·,.?!;:；：“”‘’()（）《》【】]", "", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=os.path.join(here, "asr_orig.json"))
    ap.add_argument("--clean", default=os.path.join(here, "asr_clean.json"))
    ap.add_argument("--removed", default=os.path.join(here, "asr_removed.json"))
    ap.add_argument("--stats", default=os.path.join(here, "stats.json"))
    ap.add_argument("--out", default=os.path.join(here, "asr_compare.json"))
    args = ap.parse_args()
    o, c = load(args.orig), load(args.clean)
    r = load(args.removed) if os.path.exists(args.removed) else None
    st = load(args.stats) if os.path.exists(args.stats) else {}

    ko, kc = norm(o["text"]), norm(c["text"])
    sm = difflib.SequenceMatcher(None, ko, kc, autojunk=False)
    rows = []
    for wo, wc in zip(o["windows"], c["windows"]):
        a, b = norm(wo["text"]), norm(wc["text"])
        ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if (a or b) else 1.0
        rows.append({"start": wo["start"], "end": wo["end"],
                     "chars_orig": len(a), "chars_clean": len(b),
                     "ratio": round(ratio, 3),
                     "lost": len(a) - len(b),
                     "text_orig": wo["text"], "text_clean": wc["text"]})
    # 最需要看的窗口：原文有字、干净明显变少
    worst = sorted([x for x in rows if x["chars_orig"] >= 20 and x["lost"] > 5],
                   key=lambda x: -x["lost"])[:12]
    # 干净反而变多（可能是别的说话人残留被 ASR 补出来/幻觉）
    gain = sorted([x for x in rows if x["chars_clean"] - x["chars_orig"] > 5],
                  key=lambda x: -(x["chars_clean"] - x["chars_orig"]))[:8]

    out = {
        "orig_chars": len(ko), "clean_chars": len(kc),
        "char_ratio_clean_over_orig": round(len(kc) / max(len(ko), 1), 4),
        "seq_ratio": round(sm.ratio(), 4),
        "matched_chars": sum(b.size for b in sm.get_matching_blocks()),
        "windows": len(rows),
        "windows_clean_shorter": sum(1 for x in rows if x["chars_clean"] < x["chars_orig"]),
        "windows_clean_empty": sum(1 for x in rows if x["chars_clean"] == 0 and x["chars_orig"] > 0),
        "orig_asr": {"wall_s": o["wall_s"], "rtf": o["rtf"], "chars": o["chars"]},
        "clean_asr": {"wall_s": c["wall_s"], "rtf": c["rtf"], "chars": c["chars"]},
        "stats": {k: st.get(k) for k in ("n_speakers", "main_speaker", "main_speech_s",
                                         "other_speech_s", "kept_regions", "kept_dur_s",
                                         "removed_dur_s")},
        "worst_windows": worst, "gain_windows": gain, "rows": rows,
    }
    if r:
        kr = norm(r["text"])
        out["removed_asr"] = {"chars": len(kr), "rtf": r["rtf"],
                              "char_ratio_vs_orig": round(len(kr) / max(len(ko), 1), 4),
                              "text": r["text"][:4000]}
        # 被切除音轨里有文字、且原始也有文字 的窗口：说明确实切走了语音
        rr = []
        for wr, wo in zip(r["windows"], o["windows"]):
            a, b = norm(wr["text"]), norm(wo["text"])
            rr.append({"start": wr["start"], "end": wr["end"],
                       "chars_removed": len(a), "chars_orig": len(b),
                       "text_removed": wr["text"], "text_orig": wo["text"]})
        out["removed_rows"] = rr
        out["removed_windows_with_text"] = sum(1 for x in rr if x["chars_removed"] > 0)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"[cmp] orig_chars={len(ko)} clean_chars={len(kc)} ratio={len(kc)/max(len(ko),1):.4f} "
          f"seq_ratio={sm.ratio():.4f} windows_clean_shorter={out['windows_clean_shorter']}")
    if r:
        print(f"[cmp] removed track chars={len(kr)} ({len(kr)/max(len(ko),1)*100:.1f}% of orig), "
              f"windows_with_text={out['removed_windows_with_text']}")
    print("[cmp] 差异最大的窗口:")
    for x in worst[:8]:
        print(f"  {x['start']:7.1f}-{x['end']:7.1f} orig={x['chars_orig']:3d} clean={x['chars_clean']:3d} "
              f"| orig: {x['text_orig'][:60]}")
        print(f"                                   | clean: {x['text_clean'][:60]}")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
