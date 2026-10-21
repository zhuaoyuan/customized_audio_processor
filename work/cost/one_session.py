import zstandard as zstd, json, io, glob, os, sys

base = r"C:\Users\Administrator\.dsh\sessions\--C-projects-local-audio_processer--"
target = sys.argv[1] if len(sys.argv) > 1 else "e1b39627-fa00-45a0-8151-2dae11355728"
d = os.path.join(base, target)
z = glob.glob(os.path.join(d, "*.zstd"))
print("session file:", z)
with open(z[0], "rb") as fh:
    data = fh.read()
with zstd.ZstdDecompressor().stream_reader(io.BytesIO(data)) as r:
    raw = r.read()
tot = {"inputTokens": 0, "outputTokens": 0, "cacheReadTokens": 0, "cacheWriteTokens": 0, "totalTokens": 0}
n = 0
for line in raw.decode("utf-8", "replace").splitlines():
    if not line.strip():
        continue
    o = json.loads(line)
    if o.get("type") == "assistant/message":
        u = (o.get("data") or {}).get("usage") or {}
        for k in tot:
            tot[k] += int(u.get(k) or 0)
        n += 1
print("msgs:", n, json.dumps(tot, ensure_ascii=False))
cost_idle = tot["cacheReadTokens"] * 0.02 / 1e6 + tot["inputTokens"] * 1 / 1e6 + tot["outputTokens"] * 4 / 1e6
cost_peak = tot["cacheReadTokens"] * 0.04 / 1e6 + tot["inputTokens"] * 2 / 1e6 + tot["outputTokens"] * 8 / 1e6
print(f"cost idle=Y{cost_idle:.4f}  peak=Y{cost_peak:.4f}")
