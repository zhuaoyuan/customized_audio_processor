import zstandard as zstd
import json, io, os, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Administrator\.dsh\sessions\--C-projects-local-audio_processer--\session-e7ab5b3f-cce9-4793-ba42-639bc7f8f040\session.v4.jsonl.zstd"
with open(path, "rb") as fh:
    data = fh.read()

# 多个 zstd frame 串联：用 stream_reader 全解
with zstd.ZstdDecompressor().stream_reader(io.BytesIO(data)) as r:
    raw = r.read()
text = raw.decode("utf-8", errors="replace")
lines = [l for l in text.splitlines() if l.strip()]
print("lines:", len(lines), "chars:", len(text))

kinds = collections.Counter()
objs = []
for l in lines:
    try:
        o = json.loads(l)
    except Exception:
        kinds["<bad>"] += 1
        continue
    objs.append(o)
    kinds[o.get("type") or o.get("kind") or "?"] += 1
print("types:", kinds.most_common(25))
for o in objs[:3]:
    print("---", json.dumps(o, ensure_ascii=False)[:600])
# 搜索 usage 字段
def walk(o, prefix=""):
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            if "usage" in k.lower() or "token" in k.lower():
                out.append((prefix + "/" + k, json.dumps(v, ensure_ascii=False)[:300]))
            out += walk(v, prefix + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]):
            out += walk(v, prefix + f"[{i}]")
    return out
seen = []
for o in objs[:200]:
    for p, v in walk(o):
        if p not in seen:
            seen.append(p)
print("usage-ish paths:", seen[:30])
