# -*- coding: utf-8 -*-
"""P1-T1 容器修复单测。

验收要求：
1. 对 ``data/0_20140227-丁耘 西方哲学…m4a`` 修复后 ffprobe 正常、解码 60 s 无错；
2. 构造最小 mp4 断言 delta / 修补结果（不依赖样例文件也能跑）。
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from app.models import make_job_id, slugify
from app.tools.repair_mp4 import (
    RepairError,
    align_delta,
    collect_stco_tables,
    ffmpeg_check,
    find_boxes,
    is_iso_bmff,
    mdat_spans,
    parse_boxes,
    read_stco_offsets,
    repair_bytes,
    repair_mp4,
    validate_alignment,
)

from conftest import requires_data, requires_ffmpeg


# ---------------------------------------------------------------- 构造最小 mp4
def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def _full_box(box_type: bytes, version_flags: int, payload: bytes) -> bytes:
    return _box(box_type, struct.pack(">I", version_flags) + payload)


def _stco_box(offsets: list[int]) -> bytes:
    payload = struct.pack(">I", len(offsets)) + struct.pack(">%dI" % len(offsets), *offsets)
    return _full_box(b"stco", 0, payload)


def _assemble(
    offsets: list[int],
    *,
    mdat_size: int = 256,
    with_co64: bool = False,
    empty_tracks: int = 0,
) -> bytes:
    """拼一个 ftyp + moov(trak…) + mdat 的最小 mp4；长度只与条目数有关，与偏移值无关。"""
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 0x200) + b"isomiso2mp41")
    stsd = _full_box(b"stsd", 0, struct.pack(">I", 0) + struct.pack(">I", 0))
    if with_co64:
        co64 = _full_box(b"co64", 0, struct.pack(">I", len(offsets)))
        stbl = _box(b"stbl", stsd + co64)
    else:
        stbl = _box(b"stbl", stsd + _stco_box(offsets))
    tracks = [_box(b"trak", _box(b"mdia", _box(b"minf", stbl)))]
    # 额外空轨（模拟带封面/视频轨的真实 mp4：stco 表存在但条目数为 0）
    for _ in range(empty_tracks):
        empty_stbl = _box(b"stbl", stsd + _stco_box([]))
        tracks.append(_box(b"trak", _box(b"mdia", _box(b"minf", empty_stbl))))
    moov = _box(b"moov", b"".join(tracks))
    mdat = _box(b"mdat", b"\x00" * mdat_size)
    return ftyp + moov + mdat


def build_mp4(
    n_chunks: int = 3,
    *,
    spacing: int = 16,
    mdat_size: int = 256,
    shift: int = 0,
    with_co64: bool = False,
    empty_tracks: int = 0,
) -> tuple[bytes, int]:
    """构造最小 mp4，返回 ``(bytes, mdat_payload_offset)``。

    ``shift``：stco 偏移相对正确值整体**偏小**多少字节（模拟 f00 的「moov 插入 udta
    后忘记更新 stco」损坏，实测 delta=22118）。
    """
    placeholder = _assemble([0] * n_chunks, mdat_size=mdat_size,
                            with_co64=with_co64, empty_tracks=empty_tracks)
    payload = len(placeholder) - mdat_size
    offsets = [payload - shift + i * spacing for i in range(n_chunks)]
    data = _assemble(offsets, mdat_size=mdat_size, with_co64=with_co64,
                     empty_tracks=empty_tracks)
    assert len(data) == len(placeholder)
    return data, payload


def _offsets_after_repair(data: bytes) -> list[int]:
    offsets: list[int] = []
    for _box_, table in collect_stco_tables(data, parse_boxes(data)):
        offsets.extend(table)
    return offsets


# ---------------------------------------------------------------- 解析
def test_parse_boxes_basic():
    data, payload = build_mp4(3)
    boxes = parse_boxes(data)
    top = [b for b in boxes if b.depth == 0]
    assert [b.type for b in top] == [b"ftyp", b"moov", b"mdat"]
    assert [b.type for b in boxes if b.depth == 1] == [b"trak"]
    # moov(0) > trak(1) > mdia(2) > minf(3) > stbl(4) > stsd/stco(5)
    stbl_children = [b.type for b in boxes if b.depth == 5]
    assert b"stsd" in stbl_children
    assert b"stco" in stbl_children
    assert payload == [b for b in top if b.type == b"mdat"][0].payload_offset
    assert [span.payload for span in mdat_spans(boxes)] == [payload]


def test_parse_boxes_handles_truncated_file():
    data, _ = build_mp4(1)
    boxes = parse_boxes(data[:20])
    assert find_boxes(boxes, b"stco") == []


def test_is_iso_bmff(tmp_path: Path):
    data, _ = build_mp4(1)
    mp4 = tmp_path / "a.m4a"
    mp4.write_bytes(data)
    assert is_iso_bmff(mp4)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)
    assert not is_iso_bmff(wav)
    assert not is_iso_bmff(tmp_path / "missing.mp4")


def test_empty_stco_tables_are_skipped():
    """真实 mp4 的视频/封面轨常有空 stco 表，不能被当成损坏。"""
    data, payload = build_mp4(3, empty_tracks=2)
    result = repair_bytes(data)
    assert result.ok is True
    assert result.repaired is False
    assert result.n_chunks == 3


# ---------------------------------------------------------------- 修复逻辑
def test_healthy_container_needs_no_repair():
    data, payload = build_mp4(3, shift=0)
    result = repair_bytes(data)
    assert result.ok is True
    assert result.repaired is False
    assert result.delta == 0
    assert result.dst == ""
    assert result.n_chunks == 3
    assert result.mdat_payload == payload
    assert result.min_offset == payload


def test_repair_shifts_offsets(tmp_path: Path):
    shift = 100
    data, payload = build_mp4(3, shift=shift)
    src = tmp_path / "broken.m4a"
    src.write_bytes(data)
    original_bytes = src.read_bytes()
    expected = [payload + i * 16 for i in range(3)]

    dst = tmp_path / "fixed.m4a"
    result = repair_mp4(src, dst)
    assert result.ok is True
    assert result.repaired is True
    assert result.delta == shift
    assert result.mdat_payload == payload
    assert result.min_offset == payload - shift
    assert Path(result.dst) == dst
    # 源文件只读：字节完全没变
    assert src.read_bytes() == original_bytes
    # 修补后偏移正好等于正确值，且容器大小不变
    fixed = dst.read_bytes()
    assert len(fixed) == len(original_bytes)
    assert _offsets_after_repair(fixed) == expected
    # 重复修复应识别为「已正确」
    again = repair_bytes(fixed)
    assert again.ok is True and again.repaired is False and again.delta == 0


def test_repair_multitrack_uses_same_delta():
    shift = 64
    data, payload = build_mp4(4, shift=shift, empty_tracks=1)
    result = repair_bytes(data)
    assert result.ok is True and result.repaired is True
    assert result.delta == shift
    assert _offsets_after_repair(data) == [payload - shift + i * 16 for i in range(4)]


def test_repair_rejects_co64():
    data, _ = build_mp4(2, with_co64=True)
    result = repair_bytes(data)
    assert result.ok is False
    assert "co64" in result.error


def test_repair_rejects_non_mp4():
    result = repair_bytes(b"RIFF" + b"\x00" * 200)
    assert result.ok is True
    assert result.applicable is False
    assert result.repaired is False


def test_repair_rejects_tiny_file():
    result = repair_bytes(b"\x00" * 16)
    assert result.ok is False
    assert "过小" in result.error


def test_repair_rejects_unalignable():
    """mdat 太小、平移后偏移越界 → 必须失败，不能写出坏文件。"""
    data, _ = build_mp4(3, mdat_size=16, shift=100)
    result = repair_bytes(data)
    assert result.ok is False
    assert "对齐" in result.error or "mdat" in result.error


def test_align_delta_and_validation_helpers():
    shift = 40
    data, payload = build_mp4(2, shift=shift)
    boxes = parse_boxes(data)
    tables = collect_stco_tables(data, boxes)
    delta, mdat_payload = align_delta(data, boxes, tables)
    assert delta == shift
    assert mdat_payload == payload
    spans = mdat_spans(boxes)
    assert validate_alignment([t for _, t in tables], spans, shift, len(data)) is None
    assert validate_alignment([t for _, t in tables], spans, -999, len(data)) is not None


def test_repair_refuses_dst_equals_src(tmp_path: Path):
    data, _ = build_mp4(1, shift=10)
    src = tmp_path / "a.m4a"
    src.write_bytes(data)
    result = repair_mp4(src, src)
    assert result.ok is False
    assert "只读" in result.error


def test_repair_missing_src(tmp_path: Path):
    result = repair_mp4(tmp_path / "nope.m4a")
    assert result.ok is False
    assert "不存在" in result.error


def test_repair_default_dst_name(tmp_path: Path):
    data, _ = build_mp4(1, shift=42)
    src = tmp_path / "lesson.m4a"
    src.write_bytes(data)
    result = repair_mp4(src)
    assert result.ok and result.repaired
    assert result.dst == str(tmp_path / "lesson.fixed.m4a")


def test_read_stco_offsets_rejects_bad_count():
    data = _box(b"ftyp", b"isom") + _box(
        b"stco", struct.pack(">I", 0) + struct.pack(">I", 99999)
    )
    boxes = parse_boxes(data)
    with pytest.raises(RepairError, match="超出 box 长度"):
        read_stco_offsets(data, find_boxes(boxes, b"stco")[0])


# ---------------------------------------------------------------- 真实样例（P1-T1 验收）
@requires_data
@requires_ffmpeg
def test_repair_real_f00_and_decode_60s(samples, tmp_path: Path):
    """f00：delta 应为 22118；修复副本 ffprobe 正常、解码 60 s 无错。"""
    src = samples["f00"]
    dst = tmp_path / "f00_fixed.m4a"
    result = repair_mp4(src, dst)
    assert result.ok is True, result.error
    assert result.repaired is True
    assert result.delta == 22118, f"实测 delta={result.delta}"
    assert result.dst == str(dst)
    assert dst.stat().st_size == src.stat().st_size, "修复只改 stco 偏移，文件大小应不变"

    check = ffmpeg_check(dst, seconds=60)
    assert check["probe_ok"] is True, check["error"]
    assert check["decode_ok"] is True, check["error"]

    # 原始文件确实解不了（否则这个用例没有意义）
    broken = ffmpeg_check(src, seconds=5)
    assert broken["decode_ok"] is False


@requires_data
def test_real_samples_have_no_offset_damage(samples, tmp_path: Path):
    """其余 5 个样例容器本就正常：修复应报告 repaired=False。

    注意它们多是 fragmented MP4（moof/trun，moov 里 stco 表为空），
    修复器必须识别为「无需修复」而不是报错。
    """
    fragmented = 0
    for key, path in samples.items():
        if key == "f00":
            continue
        result = repair_mp4(path, tmp_path / f"{key}.fixed.m4a")
        assert result.ok is True, f"{key}: {result.error}"
        assert result.repaired is False, f"{key} 出现意外 delta={result.delta}"
        assert result.delta == 0
        if "fragmented" in result.note:
            fragmented += 1
        else:
            assert result.n_chunks > 0, f"{key} 非碎片化容器却读不到 chunk"
    # 至少能识别出碎片化容器（本批实测：f04/f05/f06/f20/f24 均为 fragmented）
    assert fragmented >= 1


# ---------------------------------------------------------------- id 生成（P1-T3 依赖）
def test_make_job_id_deterministic_and_readable():
    name = "24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a"
    jid = make_job_id(name)
    assert jid == make_job_id(name)  # 稳定
    assert len(jid.split("_")[0]) == 8
    assert "亚里士多德" in jid
    assert " " not in jid and "《" not in jid and "/" not in jid
    # 不同目录同名不会撞（taken 里记录的是别的路径）
    other = make_job_id(
        Path("/x/24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a"),
        taken={jid: "/y/another.m4a"},
    )
    assert other != jid


def test_slugify_handles_noise_and_empty():
    assert slugify("0_20140227-丁耘 西方哲学(20140227)_normal..m4a") == "0_20140227_丁耘_西方哲学"
    assert slugify("___normal___") == "normal"  # 全是噪声词时保留原 token，避免空 id
    assert slugify("!!!") == "audio"
    assert len(slugify("a" * 100)) <= 20
