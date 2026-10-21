# -*- coding: utf-8 -*-
"""f00 修复：stco 偏移整体 +22118（moov 内 udta 21118/22216B 后未更新 stco 所致）。
写 work/scan/f00_fixed.m4a，不改 data/。"""
import shutil
import struct

SRC = r"C:\projects\local\audio_processer\data\0_20140227-丁耘 西方哲学(20140227-丁耘 西方哲学)_normal..m4a"
DST = r"C:\projects\local\audio_processer\work\scan\f00_fixed.m4a"
DELTA = 22118

shutil.copyfile(SRC, DST)
d = bytearray(open(DST, "rb").read())


def boxes(off, end, data, out):
    i = off
    while i + 8 <= end:
        sz = int.from_bytes(data[i:i + 4], "big")
        typ = data[i + 4:i + 8]
        hdr = 8
        if sz == 1:
            sz = int.from_bytes(data[i + 8:i + 16], "big")
            hdr = 16
        elif sz == 0:
            sz = end - i
        if sz < hdr:
            break
        out.append((typ, i, sz, hdr))
        if typ in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
            boxes(i + hdr, i + sz, data, out)
        i += sz
    return out


bs = boxes(0, len(d), d, [])
stco = next(b for b in bs if b[0] == b"stco")
i = stco[1] + 8
nchunk = int.from_bytes(d[i + 4:i + 8], "big")
offs = list(struct.unpack(">%dI" % nchunk, d[i + 8: i + 8 + 4 * nchunk]))
new = [o + DELTA for o in offs]
struct.pack_into(">%dI" % nchunk, d, i + 8, *new)
open(DST, "wb").write(d)
print(f"[fix] {nchunk} chunks shifted by {DELTA}: {offs[0]} -> {new[0]}, last {offs[-1]} -> {new[-1]}")
print("[fix] out:", DST)
