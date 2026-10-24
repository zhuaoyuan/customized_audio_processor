"""成稿后处理：去掉分块重叠造成的段首重复（字符级最长公共子串）。"""
import json, re, difflib
from pathlib import Path

OUT = Path(r"work/compose_proto")
text = (OUT / "manuscript.md").read_text(encoding="utf-8")
paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
report = json.loads((OUT / "report_19chunks.json").read_text(encoding="utf-8"))
boundary = {c["start"] for c in report["chunk_report"] if c["chunk"] > 1}

def body_of(p: str):
    m = re.match(r"^\[(\d{1,3}:\d{2})\]\s*", p)
    return (m.group(1) if m else ""), (p[m.end():] if m else p)

def lcs_prefix_match(prev_tail: str, cur: str):
    """在 prev_tail 与 cur 前 80 字中找最长公共子串；返回 (cur 中的结束位置, 长度)。"""
    sm = difflib.SequenceMatcher(None, prev_tail, cur[:80], autojunk=False)
    best = (0, 0)
    for a, b, size in sm.get_matching_blocks():
        if size > best[1] and size >= 8 and b <= 10:
            best = (b + size, size)
    return best

out = []
prev_body = ""
removed_log = []
for p in paras:
    stamp, body = body_of(p)
    if body and prev_body and stamp in boundary:
        end, size = lcs_prefix_match(prev_body[-500:], body)
        if size >= 8:
            removed_log.append((stamp, size, body[:end]))
            body = re.sub(r"^[，。；：、\s]+", "", body[end:])
    # 句级去重（上一段末尾 3 句）
    prev_sents = [s for s in re.findall(r"[^。！？!?]+[。！？!?]?", prev_body) if s.strip()]
    ss = [s for s in re.findall(r"[^。！？!?]+[。！？!?]?", body) if s.strip()]
    while ss:
        a = re.sub(r"[^\u4e00-\u9fff]", "", ss[0])
        hit = False
        for cand in prev_sents[-3:]:
            b = re.sub(r"[^\u4e00-\u9fff]", "", cand)
            if not a or not b:
                continue
            if difflib.SequenceMatcher(None, a, b).ratio() >= 0.8 or ((a in b or b in a) and min(len(a), len(b)) >= 8):
                removed_log.append((stamp, "sent", ss[0][:40]))
                ss.pop(0); hit = True; break
        if not hit:
            break
    if not ss:
        continue
    out.append(f"[{stamp}]" + "".join(ss))
    prev_body = "".join(ss)

clean = "\n\n".join(out) + "\n"
(OUT / "manuscript_clean.md").write_text(clean, encoding="utf-8")
print("paragraphs", len(out), "removed", len(removed_log))
for r in removed_log:
    print(r)
print("chars", len(re.sub(r"\s+", "", clean)))
stamps = [int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in re.findall(r"^\[(\d{1,3}:\d{2})\]", clean, flags=re.M)]
print("stamps", len(stamps), "monotonic", all(b >= a for a, b in zip(stamps, stamps[1:])))
