# -*- coding: utf-8 -*-
"""f00 容器诊断：stsz/stco/stsc 与 mdat 一致性检查（判断样本边界是否整体错位）。"""
import struct

P = r"C:\projects\local\audio_processer\data\0_20140227-丁耘 西方哲学(20140227-丁耘 西方哲学)_normal..m4a"
d = open(P, "rb").read()
n = len(d)


def boxes(off, end, depth=0):
    out = []
    i = off
    while i + 8 <= end:
        sz = int.from_bytes(d[i:i + 4], "big")
        typ = d[i + 4:i + 8]
        hdr = 8
        if sz == 1:
            sz = int.from_bytes(d[i + 8:i + 16], "big")
            hdr = 16
        elif sz == 0:
            sz = end - i
        if sz < hdr:
            break
        out.append((typ.decode("latin1", "replace"), i, sz, hdr))
        if typ in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
            out += boxes(i + hdr, i + sz, depth + 1)
        i += sz
    return out


bs = boxes(0, n)
print("boxes:", [(t, i, s) for t, i, s, h in bs])
print("file len", n)

# 找 stsz / stco / stsc / mdat
stsz = next(b for b in bs if b[0] == "stsz")
stco = next(b for b in bs if b[0] == "stco")
stsc = next(b for b in bs if b[0] == "stsc")
mdat = next(b for b in bs if b[0] == "mdat")

i = stsz[1] + 8
ver_flags = d[i:i + 4]
sample_size = int.from_bytes(d[i + 4:i + 8], "big")
count = int.from_bytes(d[i + 8:i + 12], "big")
print(f"stsz: ver={ver_flags.hex()} const_size={sample_size} count={count}")
if sample_size == 0:
    sizes = list(struct.unpack(">%dI" % count, d[i + 12: i + 12 + 4 * count]))
else:
    sizes = [sample_size] * count
print(f"  sizes: min={min(sizes)} max={max(sizes)} sum={sum(sizes)}")

i = stco[1] + 8
nchunk = int.from_bytes(d[i + 4:i + 8], "big")
offs = list(struct.unpack(">%dI" % nchunk, d[i + 8: i + 8 + 4 * nchunk]))
print(f"stco: chunks={nchunk} first={offs[:5]} last={offs[-3:]}")
print(f"  mdat content start={mdat[1] + mdat[3]} mdat end={mdat[1] + mdat[2]}")

i = stsc[1] + 8
nent = int.from_bytes(d[i + 4:i + 8], "big")
ent = [tuple(int.from_bytes(d[i + 8 + 12 * k + 4 * j: i + 8 + 12 * k + 4 * j + 4], "big") for j in range(3))
       for k in range(nent)]
print(f"stsc: {ent}")

# 逐 chunk 计算样本区间，与 mdat 重叠检查
first_off = offs[0]
print(f"first chunk offset={first_off} (mdat payload start={mdat[1] + mdat[3]}), delta={first_off - (mdat[1] + mdat[3])}")
print("sum sizes vs mdat payload bytes:", sum(sizes), mdat[2] - mdat[3])
print("last chunk", offs[-1], "first sample bytes:", d[first_off:first_off + 24].hex())
