import json, re
from pathlib import Path

OUT = Path(r"work/compose_proto")
text = (OUT / "manuscript.md").read_text(encoding="utf-8")
paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

SENT_RE = re.compile(r"[^。！？!?]*[。！？!?]?")

def sentences(p: str):
    return [s.strip() for s in re.findall(r"[^。！？!?]+[。！？!?]?", p) if s.strip()]

removed = []
new_paras = []
prev_sents = []
for p in paras:
    ts = re.match(r"^\[(\d{1,3}:\d{2})\]\s*", p)
    stamp = ts.group(1) if ts else ""
    body = p[ts.end():] if ts else p
    ss = sentences(body)
    # 去掉与上一段末尾重复的整句（块重叠导致；允许轻微用字差异）
    while ss:
        a = re.sub(r"[^\u4e00-\u9fff]", "", ss[0])
        import difflib
        hit = False
        for cand in prev_sents[-3:]:
            b = re.sub(r"[^\u4e00-\u9fff]", "", cand)
            if not a or not b:
                continue
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            if ratio >= 0.75 or ((a in b or b in a) and min(len(a), len(b)) >= 8):
                removed.append((stamp, round(ratio, 2), ss[0]))
                ss.pop(0)
                hit = True
                break
        if not hit:
            break
    if not ss:
        continue
    new_paras.append(f"[{stamp}]" + "".join(ss))
    prev_sents = ss

print("paragraphs before", len(paras), "after", len(new_paras), "removed sentences", len(removed))
for r in removed:
    print("removed:", r)

clean = "\n\n".join(new_paras) + "\n"
(OUT / "manuscript_clean.md").write_text(clean, encoding="utf-8")

# 时间单调/间隔检查
stamps = [int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in re.findall(r"^\[(\d{1,3}:\d{2})\]", clean, flags=re.M)]
bad = [(a, b) for a, b in zip(stamps, stamps[1:]) if b < a]
gaps = [(a, b) for a, b in zip(stamps, stamps[1:]) if b - a > 300]
print("stamps", len(stamps), "non_monotonic", bad[:5], "gaps>5min", gaps[:5], "last", stamps[-1] // 60, ":", stamps[-1] % 60)
print("chars", len(re.sub(r"\s+", "", clean)))
