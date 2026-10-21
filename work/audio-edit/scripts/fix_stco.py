# -*- coding: utf-8 -*-
"""xfzx 容器 stco 偏移修复：moov 内 udta 插入后未更新 stco，导致采样偏移整体偏小。
原理：解析 moov/stbl/stco，取最小 chunk offset 与真实 mdat payload 起点比较求 delta，整体平移。
用法：python fix_stco.py <src.m4a> <dst.m4a>
"""
import struct
import sys


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


def main(src, dst):
    d = bytearray(open(src, "rb").read())
    bs = boxes(0, len(d), d, [])
    moov = next(b for b in bs if b[0] == b"moov")
    stco = next(b for b in bs if b[0] == b"stco")
    co64 = next((b for b in bs if b[0] == b"co64"), None)
    if co64 is not None:
        raise SystemExit("co64 unsupported")
    i = stco[1] + 8
    nchunk = int.from_bytes(d[i + 4:i + 8], "big")
    offs = list(struct.unpack(">%dI" % nchunk, d[i + 8:i + 8 + 4 * nchunk]))
    # mdat payload: 在 moov 之后找 mdat box
    mdat_payload = None
    j = moov[1] + moov[2]
    while j + 8 <= len(d):
        sz = int.from_bytes(d[j:j + 4], "big")
        typ = d[j + 4:j + 8]
        if typ == b"mdat":
            mdat_payload = j + 8
            break
        if sz < 8:
            break
        j += sz
    if mdat_payload is None:
        raise SystemExit("mdat not found")
    delta = mdat_payload - min(offs)
    print(f"[fix] chunks={nchunk} min_off={min(offs)} mdat_payload={mdat_payload} delta={delta}")
    if delta <= 0:
        raise SystemExit("delta not positive; maybe already ok")
    struct.pack_into(">%dI" % nchunk, d, i + 8, *[o + delta for o in offs])
    open(dst, "wb").write(d)
    print(f"[fix] -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
