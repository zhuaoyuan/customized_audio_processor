import zstandard as zstd
import json, io, os, glob, collections, sys

BASE = r"C:\Users\Administrator\.dsh\sessions\--C-projects-local-audio_processer--"

def load(path):
    with open(path, "rb") as fh:
        data = fh.read()
    with zstd.ZstdDecompressor().stream_reader(io.BytesIO(data)) as r:
        raw = r.read()
    objs = []
    for l in raw.decode("utf-8", errors="replace").splitlines():
        if not l.strip():
            continue
        try:
            objs.append(json.loads(l))
        except Exception:
            pass
    return objs

def summarize(path, label):
    objs = load(path)
    tot = collections.Counter()
    n_msg = 0
    first_t = last_t = None
    for o in objs:
        t = o.get("type")
        if t == "assistant/message":
            u = (o.get("data") or {}).get("usage") or {}
            if u:
                n_msg += 1
                for k in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens", "totalTokens"):
                    tot[k] += int(u.get(k) or 0)
        if o.get("time"):
            first_t = first_t or o["time"]
            last_t = o["time"]
        if t == "tool/call":
            tot["tool_calls"] += 1
        if t == "web/deepseek-search-llm-request":
            tot["web_requests"] += 1
    dur_min = (last_t - first_t) / 60000 if first_t and last_t else 0
    print(f"\n=== {label} ===")
    print(f"  assistant msgs={n_msg}  tool_calls={tot['tool_calls']}  web={tot['web_requests']}  时长≈{dur_min:.0f} min")
    print(f"  input={tot['inputTokens']:,}  cacheRead={tot['cacheReadTokens']:,}  cacheWrite={tot['cacheWriteTokens']:,}")
    print(f"  output={tot['outputTokens']:,}  total={tot['totalTokens']:,}")
    return dict(label=label, msgs=n_msg, **{k: tot[k] for k in
                ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens", "totalTokens", "tool_calls", "web_requests")},
                dur_min=dur_min)

rows = []
for d in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, d)
    if not os.path.isdir(p):
        continue
    z = glob.glob(os.path.join(p, "*.zstd"))
    if not z:
        continue
    rows.append(summarize(z[0], d))

print("\n=== 合计 ===")
for k in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens", "totalTokens", "tool_calls", "web_requests"):
    print(f"  {k}: {sum(r[k] for r in rows):,}")
print(f"  会话数: {len(rows)}")
json.dump(rows, open(r"C:\projects\local\audio_processer\work\cost\session_usage.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
