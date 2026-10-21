# -*- coding: utf-8 -*-
"""f00 帧级诊断：元素 ID 直方图 + 单帧解码成功率。用于判断是整段码流损坏还是偏移。"""
import os
import subprocess
import tempfile

SCAN = r"C:\projects\local\audio_processer\work\scan"
AAC = os.path.join(SCAN, "f00_try", "sf5.aac")
d = open(AAC, "rb").read()


def frames(data):
    i = 0
    n = len(data)
    out = []
    while i + 7 <= n:
        if data[i] != 0xFF or (data[i + 1] & 0xF0) != 0xF0:
            j = data.find(b"\xff", i + 1)
            if j < 0:
                break
            i = j
            continue
        b = data[i:i + 7]
        flen = ((b[3] & 3) << 11) | (b[4] << 3) | ((b[5] >> 5) & 7)
        if flen < 7:
            i += 1
            continue
        out.append((i, flen))
        i += flen
    return out


fr = frames(d)
print(f"[frames] {len(fr)}")

# 元素 ID 直方图（每帧 payload 前 3 bit）
from collections import Counter
h = Counter()
for off, flen in fr:
    payload = d[off + 7: off + 7 + 8]
    eid = (payload[0] >> 5) & 7
    h[eid] += 1
print("[elem-id hist]", dict(sorted(h.items())), "  (0=SCE,1=CPE,5=PCE,6=FIL,7=END)")

# 对照：好文件 f24 的 ADTS
good = os.path.join(SCAN, "f24_ref.aac")
if not os.path.exists(good):
    src = os.path.join(r"C:\projects\local\audio_processer\data",
                       "24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-c:a", "copy",
                    "-f", "adts", good], capture_output=True)
gd = open(good, "rb").read()
gf = frames(gd)[:5000]
hg = Counter(((gd[o + 7] >> 5) & 7) for o, l in gf)
print("[elem-id hist good f24, first 5000]", dict(sorted(hg.items())))

# 单帧解码测试：均匀抽 40 帧
import random
random.seed(0)
idx = sorted(random.sample(range(len(fr)), 40))
ok = 0
fails = []
tmp = os.path.join(SCAN, "f00_try", "one.aac")
for k in idx:
    off, flen = fr[k]
    open(tmp, "wb").write(d[off: off + flen])
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "aac", "-i", tmp,
                        "-f", "null", "-"], capture_output=True)
    err = r.stderr.decode("utf-8", "replace").strip()
    if r.returncode == 0 and not err:
        ok += 1
    else:
        fails.append((k, err.splitlines()[:1]))
print(f"[single-frame] ok={ok}/{len(idx)}")
for f in fails[:6]:
    print("  fail frame", f)
