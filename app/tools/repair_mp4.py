# -*- coding: utf-8 -*-
"""MP4 / M4A 容器 ``stco`` 偏移修复（移植并加固自 ``work/audio-edit/scripts/fix_stco.py``）。

损坏成因
--------
本批 ``0_20140227-丁耘 西方哲学…m4a`` 的 ``moov`` 里被插入了 ``udta`` 段却没更新
``stco``（chunk offset）表，导致所有采样偏移整体偏小 **22118 B**。ffmpeg 按错位偏移读
AAC 帧会报：

    [aac] Reserved bit set.
    [aac] Number of bands (41) exceeds limit (30).

修复原理
--------
1. 解析 box 树，取 ``mdat`` 的 payload 起点与 ``stco`` 的最小偏移；
2. ``delta = mdat_payload - min(stco_offsets)``；
3. ``delta != 0`` 时把 **全部** chunk 偏移 ``+ delta`` 后另存副本（源文件只读，绝不改写）；
4. 修完做一次边界自检：所有新偏移必须落在 ``[mdat_payload, mdat_end)`` 内，否则判定失败，
   交给上层 quarantine。

只处理 ISO BMFF（``ftyp`` 开头且含 ``moov/mdat``）；mp3/wav 等返回 ``applicable=False``。
``co64``（64 位偏移）暂不支持——遇到就明确报错，不猜。
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 需要递归下钻的容器 box（只关心 moov 到 stbl 的路径）
CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}
#: mdat 之后不再递归
MAX_PARSE_DEPTH = 8


class RepairError(RuntimeError):
    """容器结构非法、无法定位 stco/mdat 等。"""


@dataclass(frozen=True)
class Box:
    """一个 ISO BMFF box。``offset`` 是 box 起点，``header`` 是头长度（8 或 16）。"""

    type: bytes
    offset: int
    size: int
    header: int = 8
    depth: int = 0

    @property
    def type_name(self) -> str:
        return self.type.decode("latin-1", "replace")

    @property
    def payload_offset(self) -> int:
        return self.offset + self.header

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass
class RepairResult:
    """修复结果（``ok=False`` 时 ``error`` 一定非空）。"""

    ok: bool = False
    repaired: bool = False
    applicable: bool = True
    delta: int = 0
    n_chunks: int = 0
    min_offset: int = 0
    max_offset: int = 0
    mdat_payload: int = 0
    src: str = ""
    dst: str = ""
    error: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "repaired": self.repaired,
            "applicable": self.applicable,
            "delta": self.delta,
            "n_chunks": self.n_chunks,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "mdat_payload": self.mdat_payload,
            "src": self.src,
            "dst": self.dst,
            "error": self.error,
            "note": self.note,
        }


# ---------------------------------------------------------------- box 解析
def parse_boxes(data: bytes, start: int = 0, end: Optional[int] = None,
                depth: int = 0) -> List[Box]:
    """解析 [start, end) 范围内的 box；对容器 box 递归下钻。"""
    total = len(data)
    stop = total if end is None else min(end, total)
    boxes: List[Box] = []
    pos = start
    while pos + 8 <= stop:
        size = struct.unpack_from(">I", data, pos)[0]
        box_type = bytes(data[pos + 4:pos + 8])
        header = 8
        if size == 1:
            if pos + 16 > stop:
                break
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            # size==0 表示延伸到文件/父容器末尾
            size = stop - pos
        if size < header or pos + size > stop:
            # 结构断裂：停止解析（保留已解析的部分，交给上层判断）
            break
        box = Box(type=box_type, offset=pos, size=size, header=header, depth=depth)
        boxes.append(box)
        if box_type in CONTAINER_BOXES and depth < MAX_PARSE_DEPTH:
            boxes.extend(parse_boxes(data, box.payload_offset, box.end, depth + 1))
        pos += size
    return boxes


def find_boxes(boxes: Sequence[Box], box_type: bytes) -> List[Box]:
    return [b for b in boxes if b.type == box_type]


def find_box(boxes: Sequence[Box], box_type: bytes) -> Optional[Box]:
    for box in boxes:
        if box.type == box_type:
            return box
    return None


def is_iso_bmff(path: str | Path) -> bool:
    """按文件头判断是否 ISO BMFF（``....ftyp``）。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
    except OSError:
        return False
    return len(head) >= 8 and head[4:8] == b"ftyp"


# ---------------------------------------------------------------- stco / mdat
def read_stco_offsets(data: bytes, box: Box) -> List[int]:
    """读 stco 表的全部 chunk 偏移（自动校验条目数与 box 长度是否自洽）。"""
    payload = box.payload_offset
    if payload + 8 > box.end:
        raise RepairError("stco box 过短，结构异常")
    version_flags = struct.unpack_from(">I", data, payload)[0]
    if version_flags >> 24 != 0:
        raise RepairError(f"stco version={version_flags >> 24} 非 0，结构异常")
    count = struct.unpack_from(">I", data, payload + 4)[0]
    need = 8 + 4 * count
    if payload + need > box.end:
        raise RepairError(
            f"stco 条目数 {count} 超出 box 长度（需要 {need} B，可用 {box.end - payload} B）"
        )
    if count == 0:
        # 真实容器（含视频轨/封面轨的 mp4）允许存在空的 chunk 表，跳过即可
        return []
    return list(struct.unpack_from(">%dI" % count, data, payload + 8))


def write_stco_offsets(buf: bytearray, box: Box, offsets: Sequence[int]) -> None:
    """把偏移表写回 buffer（长度必须与原有条目数一致）。"""
    payload = box.payload_offset
    count = struct.unpack_from(">I", buf, payload + 4)[0]
    if len(offsets) != count:
        raise RepairError(f"偏移数量不符：{len(offsets)} != {count}")
    for value in offsets:
        if not (0 <= value <= 0xFFFFFFFF):
            raise RepairError(f"偏移 {value} 超出 32 位范围，需 co64")
    struct.pack_into(">%dI" % count, buf, payload + 8, *offsets)


@dataclass(frozen=True)
class MdatSpan:
    """顶层 mdat 的字节范围。"""

    box_offset: int
    payload: int
    end: int

    def contains(self, offset: int) -> bool:
        return self.payload <= offset < self.end


def mdat_spans(boxes: Sequence[Box]) -> List[MdatSpan]:
    return [
        MdatSpan(box_offset=b.offset, payload=b.payload_offset, end=b.end)
        for b in boxes
        if b.type == b"mdat" and b.depth == 0
    ]


def collect_stco_tables(data: bytes, boxes: Sequence[Box]) -> List[Tuple[Box, List[int]]]:
    """收集**非空**的 stco 表；空表（视频/封面轨常见）直接跳过。"""
    tables: List[Tuple[Box, List[int]]] = []
    for box in find_boxes(boxes, b"stco"):
        offsets = read_stco_offsets(data, box)
        if offsets:
            tables.append((box, offsets))
    return tables


def validate_alignment(
    tables: Sequence[Sequence[int]],
    spans: Sequence[MdatSpan],
    delta: int,
    file_len: int,
) -> Optional[str]:
    """检查「所有表整体平移 delta」后是否自洽；通过返回 None，否则返回失败原因。"""
    if not spans:
        return "找不到顶层 mdat box"
    for index, offsets in enumerate(tables):
        shifted = [o + delta for o in offsets]
        if shifted[0] < 0:
            return f"表#{index} 出现负偏移 {shifted[0]}"
        if shifted[-1] >= file_len:
            return f"表#{index} 偏移 {shifted[-1]} 超出文件长度 {file_len}"
        if any(b < a for a, b in zip(shifted, shifted[1:])):
            return f"表#{index} 平移后偏移非单调（容器可能多处损坏）"
        for offset in shifted:
            if not any(span.contains(offset) for span in spans):
                return f"表#{index} 的偏移 {offset} 不在任何 mdat 范围内"
    return None


def align_delta(
    data: bytes,
    boxes: Sequence[Box],
    tables: Sequence[Tuple[Box, List[int]]],
) -> Tuple[int, int]:
    """求把 stco 与 mdat 对齐所需的 delta，返回 ``(delta, 参考 mdat payload)``。

    候选顺序很重要：**先试 delta=0**（容器本就正确是最常见情况），再试
    「各 mdat payload − 各表最小偏移」。第一个能让全部偏移自洽的候选即采用。
    """
    spans = mdat_spans(boxes)
    if not spans:
        raise RepairError("找不到顶层 mdat box")
    all_offsets = [offset for _, offsets in tables for offset in offsets]
    candidates: List[int] = [0]
    for span in spans:
        candidates.append(span.payload - min(all_offsets))
        for _, offsets in tables:
            candidates.append(span.payload - min(offsets))
    unique: List[int] = []
    seen = set()
    for value in candidates:
        if value not in seen:
            seen.add(value)
            unique.append(value)

    errors: List[str] = []
    for delta in unique:
        reason = validate_alignment([offs for _, offs in tables], spans, delta, len(data))
        if reason is None:
            # 找出该 delta 下第一块 mdat（用于报告）
            reference = min(
                (span for span in spans if span.contains(min(all_offsets) + delta)),
                key=lambda span: span.payload,
                default=spans[0],
            )
            return int(delta), reference.payload
        errors.append(f"delta={delta}: {reason}")
    raise RepairError(
        "无法把 stco 偏移与 mdat 对齐（可能多处损坏）：" + "；".join(errors[:6])
    )


# ---------------------------------------------------------------- 修复
def repair_bytes(data: bytes, src: Optional[str | Path] = None,
                 dst: Optional[str | Path] = None) -> RepairResult:
    """对内存中的容器字节做修复（``repair_mp4`` 的纯函数内核，便于单测）。"""
    src_str = str(src) if src is not None else ""
    result = RepairResult(src=src_str, dst=str(dst) if dst is not None else "")
    if len(data) < 128:
        result.ok = False
        result.error = f"文件过小（{len(data)} B），不是有效的 MP4"
        return result
    if data[4:8] != b"ftyp":
        result.ok = True
        result.applicable = False
        result.error = ""
        return result

    boxes = parse_boxes(data)
    tables = collect_stco_tables(data, boxes)
    if not tables:
        # 情况 A：fragmented MP4（moof/trun，样例里 5/6 是这种）
        #   moov 里的 stco 表可以合法为空——采样偏移在 moof 的 tfhd/trun 里，
        #   不存在「整体平移」的损坏形态，直接判无需修复，交给解码器处理。
        if find_box(boxes, b"moof") is not None:
            result.ok = True
            result.repaired = False
            result.note = "fragmented mp4（moof/trun）：moov 无 chunk 偏移表，无需 stco 修复"
            return result
        if find_box(boxes, b"co64") is not None:
            result.ok = False
            result.error = "容器使用 co64（64 位 chunk offset），当前修复器不支持"
            return result
        result.ok = False
        result.error = "找不到非空 stco 表（容器结构不完整、被截断，或只有空轨）"
        return result

    all_offsets = [offset for _, offsets in tables for offset in offsets]
    result.n_chunks = len(all_offsets)
    result.min_offset = min(all_offsets)
    result.max_offset = max(all_offsets)

    try:
        delta, mdat_payload = align_delta(data, boxes, tables)
    except RepairError as exc:
        result.ok = False
        result.error = str(exc)
        return result
    result.mdat_payload = mdat_payload
    result.delta = int(delta)

    if delta == 0:
        # 容器本就自洽：不写副本（repaired=False）
        result.ok = True
        result.repaired = False
        return result

    # 所有 track 用同一个 delta（损坏源于 moov 整体插入，偏移是统一平移的）
    buf = bytearray(data)
    for box, offsets in tables:
        write_stco_offsets(buf, box, [o + delta for o in offsets])

    # 自检：重解析 + 用 delta=0 复验，确认补丁后偏移全部落在 mdat 内且单调
    patched = bytes(buf)
    patched_boxes = parse_boxes(patched)
    spans = mdat_spans(patched_boxes)
    verify_tables = [
        offsets for _, offsets in collect_stco_tables(patched, patched_boxes)
    ]
    reason = validate_alignment(verify_tables, spans, 0, len(patched))
    if reason is not None:
        result.ok = False
        result.error = f"修复后自检失败：{reason}（已放弃，建议 quarantine）"
        return result

    result.ok = True
    result.repaired = True
    if dst is not None:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(patched)
        result.dst = str(dst_path)
    return result


def repair_mp4(src: str | Path, dst: Optional[str | Path] = None) -> RepairResult:
    """修复 ``src`` 的 stco 偏移并写到 ``dst``（默认 ``<src>.fixed<ext>``）。

    源文件**只读**；``delta == 0`` 时不写 dst（``repaired=False``）。
    """
    src_path = Path(src)
    if not src_path.is_file():
        return RepairResult(ok=False, src=str(src_path), error=f"源文件不存在：{src_path}")
    try:
        data = src_path.read_bytes()
    except OSError as exc:
        return RepairResult(ok=False, src=str(src_path), error=f"读取源文件失败：{exc}")

    if dst is None:
        dst = src_path.with_name(f"{src_path.stem}.fixed{src_path.suffix}")
    dst_path = Path(dst)
    try:
        if dst_path.resolve() == src_path.resolve():
            return RepairResult(
                ok=False, src=str(src_path), error="dst 不能与 src 相同（源文件必须只读）"
            )
    except OSError:  # pragma: no cover - resolve 失败时按路径字符串比较
        if str(dst_path) == str(src_path):
            return RepairResult(ok=False, src=str(src_path), error="dst 不能与 src 相同")

    result = repair_bytes(data, src=src_path, dst=dst_path)
    result.src = str(src_path)
    return result


# ---------------------------------------------------------------- 解码自检（验收用）
def ffmpeg_check(path: str | Path, seconds: float = 60.0, timeout_s: float = 180.0) -> Dict[str, Any]:
    """用 ffprobe + ffmpeg 抽 ``seconds`` 秒解码，确认容器可正常解码（返回诊断 dict）。"""
    p = Path(path)
    out: Dict[str, Any] = {"path": str(p), "probe_ok": False, "decode_ok": False, "error": ""}
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        out["error"] = "PATH 中找不到 ffmpeg/ffprobe"
        return out
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-print_format", "json", str(p)],
        capture_output=True,
    )
    if probe.returncode != 0:
        out["error"] = probe.stderr.decode("utf-8", "replace")[-800:]
        return out
    out["probe_ok"] = True
    decode = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-t", str(float(seconds)), "-i", str(p),
         "-vn", "-f", "null", "-"],
        capture_output=True,
    )
    stderr = decode.stderr.decode("utf-8", "replace").strip()
    out["decode_ok"] = decode.returncode == 0 and not stderr
    if stderr:
        out["error"] = stderr[-800:]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行：``python -m app.tools.repair_mp4 <src> [dst] [--check]``。"""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="MP4/M4A stco 偏移修复（源文件只读）")
    parser.add_argument("src", help="输入 m4a/mp4")
    parser.add_argument("dst", nargs="?", default=None, help="输出副本（默认 <src>.fixed<ext>）")
    parser.add_argument("--check", action="store_true", help="修复后用 ffmpeg 解码 60 s 自检")
    parser.add_argument("--check-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)

    result = repair_mp4(args.src, args.dst)
    payload: Dict[str, Any] = {"repair": result.to_dict()}
    if args.check and result.ok and result.repaired:
        payload["ffmpeg"] = ffmpeg_check(result.dst, args.check_seconds)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not result.ok:
        return 2
    if args.check and result.repaired and not payload["ffmpeg"]["decode_ok"]:
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
