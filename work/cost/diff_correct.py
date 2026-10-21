import difflib, json, re
a = open(r"C:\projects\local\audio_processer\work\cost\correct_test\input_chunk1.txt", encoding="utf-8").read()
b = open(r"C:\projects\local\audio_processer\work\cost\correct_test\corrected_chunk1.txt", encoding="utf-8").read()
print("orig chars:", len(a), " corrected chars:", len(b))
# 句级切分做差异，便于看改动上下文
sa = re.split(r"(?<=[。？！])", a)
sb = re.split(r"(?<=[。？！])", b)
print("orig sentences:", len(sa), "corrected sentences:", len(sb))
# 字级 diff
sm = difflib.SequenceMatcher(None, a, b)
ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
print("diff ops:", len(ops))
edits = []
for tag, i1, i2, j1, j2 in ops:
    edits.append((tag, a[i1:i2], b[j1:j2], a[max(0,i1-12):i1], a[i2:i2+12]))
for tag, old, new, pre, post in edits[:80]:
    print(f"[{tag}] ...{pre}【{old}】{post}...  →  【{new}】")
print("total edit ops:", len(edits))
changed_chars = sum(max(len(o), len(n)) for _, o, n, *_ in edits)
print("changed span chars ≈", changed_chars, f"({changed_chars/len(a)*100:.1f}% of text)")
