# -*- coding: utf-8 -*-
"""f00 损伤文件的采样率索引诊断：把 ADTS 头改成不同 sf_index 后解码，统计错误数。"""
import os
import subprocess
import sys

SCAN = r"C:\projects\local\audio_processer\work\scan"
SRC = os.path.join(SCAN, "f00_raw.aac")
DST = os.path.join(SCAN, "f00_try")
os.makedirs(DST, exist_ok=True)
d = open(SRC, "rb").read()


def frames(data):
    """遍历 ADTS 帧，返回 (offset, frame_len, sf_index, prof, ch)。"""
    i = 0
    n = len(data)
    out = []
    while i + 7 <= n:
        if data[i] != 0xFF or (data[i + 1] & 0xF0) != 0xF0:
            # 寻找下一个同步字
            j = data.find(b"\xff", i + 1)
            if j < 0:
                break
            i = j
            continue
        b = data[i:i + 7]
        prof = (b[2] >> 6) & 3
        sf = (b[2] >> 2) & 15
        ch = ((b[2] & 1) << 2) | ((b[3] >> 6) & 3)
        flen = ((b[3] & 3) << 11) | (b[4] << 3) | ((b[5] >> 5) & 7)
        if flen < 7:
            i += 1
            continue
        out.append((i, flen, sf, prof, ch))
        i += flen
    return out


fr = frames(d)
print(f"[frames] {len(fr)} frames, first sf={fr[0][2]} prof={fr[0][3]} ch={fr[0][4]} len={fr[0][1]}", flush=True)
# sf_index 一致?
from collections import Counter
print("[sf hist]", Counter(x[2] for x in fr).most_common(5), flush=True)

base = bytearray(d)


def rewrite(sf_target):
    out = bytearray(base)
    for (off, flen, sf, prof, ch) in fr:
        # ADTS: byte2 = profile(2) | sf_index(4) | private(1) | ch_config(bit2)
        out[off + 2] = (prof << 6) | (sf_target << 2) | ((ch >> 2) & 1)
    return bytes(out)


for sf in (4, 5, 7, 3, 6):
    p = os.path.join(DST, f"sf{sf}.aac")
    open(p, "wb").write(rewrite(sf))
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-i", p,
                        "-f", "null", "-"], capture_output=True)
    err = r.stderr.decode("utf-8", "replace")
    nlines = len([x for x in err.splitlines() if x.strip()])
    print(f"[sf={sf}] exit={r.returncode} errlines={nlines} head={err.splitlines()[:2]}", flush=True)
