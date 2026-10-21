# -*- coding: utf-8 -*-
"""解码与缓存：源文件（m4a/mp3/wav/mp4）→ 16 kHz 单声道 float32 wav（分析轴），
按需再解 44.1 kHz（渲染轴），失败时自动走 stco 修复重试。

契约（P1-T2）
-------------
``decode(src, workdir) -> DecodeResult(wav16k, wav44k_or_None, duration, sr, repaired, ok)``
本模块的签名是 ``decode(src, cache_dir=None, job_id=None, need_44k=False, cfg=None, ...)``，
``cache_dir`` 即实施计划里的 ``workdir``（默认 ``work/cache``）。

关键约定
--------
- **16 kHz 用 float32**（``pcm_f32le``）：源存在 >0 dBFS 过冲，先削成 int16 会丢失判断依据；
- 缓存命中条件：源文件 size/mtime 未变 + wav 存在 + 时长自洽；命中时不调用 ffmpeg；
- 失败判定不只看 ffmpeg 返回码，还要看**输出时长**（ffmpeg 对损坏容器可能返回 0 却输出空文件）；
- 修复副本只在本次调用内存在，用完即删（``decode.repair.keep_fixed_file`` 可保留），
  避免 300 文件批量时 ``work/cache`` 膨胀。
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..logging_setup import get_logger
from ..models import DecodeResult, make_job_id
from ..tools.repair_mp4 import is_iso_bmff, repair_mp4

log = get_logger(__name__)

#: 契约要求的分析采样率
ANALYSIS_SR = 16000
#: 渲染采样率
RENDER_SR = 44100
#: 统一 float32，禁止 int16（见模块 docstring）
PCM_F32 = "pcm_f32le"

CACHE_SUFFIX_16K = ".16k.wav"
CACHE_SUFFIX_44K = ".44k.wav"
CACHE_SUFFIX_META = ".decode.json"


class DecodeError(RuntimeError):
    """解码/探测失败（错误信息面向人工排查）。"""


# ---------------------------------------------------------------- 媒体信息
@dataclass
class MediaInfo:
    """ffprobe 结果里我们关心的字段。"""

    duration_s: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bit_rate: int = 0
    codec: str = ""
    format_name: str = ""
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_rate": self.bit_rate,
            "codec": self.codec,
            "format_name": self.format_name,
            "size_bytes": self.size_bytes,
        }


@dataclass
class WavInfo:
    """WAV 头信息（只读头，不加载全部采样）。"""

    sample_rate: int = 0
    channels: int = 1
    sampwidth: int = 2
    n_frames: int = 0
    format_tag: int = 1
    data_bytes: int = 0

    @property
    def duration_s(self) -> float:
        return self.n_frames / float(self.sample_rate) if self.sample_rate else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sampwidth": self.sampwidth,
            "n_frames": self.n_frames,
            "format_tag": self.format_tag,
            "duration_s": round(self.duration_s, 3),
        }


def _require_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise DecodeError(f"PATH 中找不到 {name}，请先安装 ffmpeg 并加入 PATH")
    return path


def probe_media(path: str | Path, timeout_s: float = 60.0) -> MediaInfo:
    """ffprobe 读取时长/采样率/声道/码率；失败抛 DecodeError。"""
    _require_bin("ffprobe")
    src = Path(path)
    if not src.is_file():
        raise DecodeError(f"输入文件不存在：{src}")
    cmd = [
        "ffprobe", "-v", "error", "-show_format", "-show_streams",
        "-print_format", "json", str(src),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise DecodeError(f"ffprobe 超时（>{timeout_s}s）：{src}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise DecodeError(f"ffprobe 失败（exit={proc.returncode}）：{src}\n{err[-1500:]}")
    try:
        meta = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise DecodeError(f"ffprobe 输出不是合法 JSON：{src}") from exc

    fmt = meta.get("format") or {}
    streams = meta.get("streams") or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise DecodeError(f"文件里没有音频流：{src}")
    try:
        duration = float(fmt.get("duration") or audio.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise DecodeError(f"ffprobe 拿不到时长（容器可能损坏）：{src}")
    info = MediaInfo(
        duration_s=duration,
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        bit_rate=int(fmt.get("bit_rate") or audio.get("bit_rate") or 0),
        codec=str(audio.get("codec_name") or ""),
        format_name=str(fmt.get("format_name") or ""),
        size_bytes=int(fmt.get("size") or src.stat().st_size),
    )
    return info


# ---------------------------------------------------------------- WAV 头解析
def wav_info(path: str | Path) -> WavInfo:
    """解析 RIFF/WAVE 头，返回采样率/声道/位宽/帧数（不读 data 内容）。"""
    p = Path(path)
    info = WavInfo()
    if not p.is_file():
        raise DecodeError(f"WAV 文件不存在：{p}")
    with open(p, "rb") as fh:
        header = fh.read(12)
        if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise DecodeError(f"不是合法的 WAV（缺 RIFF/WAVE 头）：{p}")
        fmt_seen = False
        while True:
            chunk_head = fh.read(8)
            if len(chunk_head) < 8:
                break
            chunk_id = chunk_head[0:4]
            chunk_size = struct.unpack_from("<I", chunk_head, 4)[0]
            if chunk_id == b"fmt ":
                body = fh.read(chunk_size)
                if len(body) < 16:
                    raise DecodeError(f"WAV fmt 段过短：{p}")
                info.format_tag = struct.unpack_from("<H", body, 0)[0]
                info.channels = struct.unpack_from("<H", body, 2)[0]
                info.sample_rate = struct.unpack_from("<I", body, 4)[0]
                bits = struct.unpack_from("<H", body, 14)[0]
                info.sampwidth = max(1, bits // 8)
                if info.format_tag == 0xFFFE and len(body) >= 40:
                    # WAVE_FORMAT_EXTENSIBLE：真实格式在 subformat 前两字节
                    info.format_tag = struct.unpack_from("<H", body, 24)[0]
                fmt_seen = True
                if chunk_size % 2:
                    fh.seek(1, 1)
            elif chunk_id == b"data":
                if not fmt_seen:
                    raise DecodeError(f"WAV data 段出现在 fmt 段之前：{p}")
                info.data_bytes = chunk_size
                block = max(1, info.channels * info.sampwidth)
                info.n_frames = chunk_size // block
                break
            else:
                fh.seek(chunk_size + (chunk_size % 2), 1)
    if not fmt_seen:
        raise DecodeError(f"WAV 缺少 fmt 段：{p}")
    if info.n_frames == 0:
        raise DecodeError(f"WAV 没有音频数据（data 段为 0）：{p}")
    return info


def read_wav_float32(path: str | Path) -> Tuple[int, "Any"]:
    """读 WAV 为 (sr, float32 ndarray)；供后续 VAD/ASR/电平阶段使用。"""
    import numpy as np
    import scipy.io.wavfile as wavfile

    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        x = data
    elif data.dtype == np.float64:
        x = data.astype(np.float32)
    else:
        raise DecodeError(f"不支持的 WAV dtype={data.dtype}：{path}")
    if x.ndim > 1:
        x = x[:, 0]
    return int(sr), np.ascontiguousarray(x, dtype=np.float32)


# ---------------------------------------------------------------- ffmpeg 调用
def decode_audio(
    src: str | Path,
    dst: str | Path,
    sample_rate: int = ANALYSIS_SR,
    channels: int = 1,
    sample_fmt: str = PCM_F32,
    timeout_s: float = 1800.0,
) -> Path:
    """用 ffmpeg 解码成指定采样率/格式的 WAV；失败抛 DecodeError。"""
    _require_bin("ffmpeg")
    src_p, dst_p = Path(src), Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-i", str(src_p),
        "-map", "0:a:0",       # 显式取第一条音频流（部分 m4a 带封面视频流）
        "-vn", "-ac", str(int(channels)), "-ar", str(int(sample_rate)),
        "-c:a", sample_fmt, "-f", "wav", str(dst_p),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise DecodeError(f"ffmpeg 解码超时（>{timeout_s}s）：{src_p}") from exc
    if proc.returncode != 0 or not dst_p.is_file():
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise DecodeError(
            f"ffmpeg 解码失败（exit={proc.returncode}）：{src_p}\n{err[-1500:] or '(无 stderr)'}"
        )
    return dst_p


# ---------------------------------------------------------------- 缓存路径
def cache_paths(cache_dir: str | Path, job_id: str, src: Optional[str | Path] = None) -> Dict[str, Path]:
    """一个 job 的全部缓存文件路径。"""
    cdir = Path(cache_dir)
    ext = Path(src).suffix if src else ".m4a"
    return {
        "wav16k": cdir / f"{job_id}{CACHE_SUFFIX_16K}",
        "wav44k": cdir / f"{job_id}{CACHE_SUFFIX_44K}",
        "meta": cdir / f"{job_id}{CACHE_SUFFIX_META}",
        "fixed": cdir / f"{job_id}.fixed{ext}",
    }


def _load_meta(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_meta(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=1), encoding="utf-8")


def _source_signature(src: Path) -> Dict[str, Any]:
    stat = src.stat()
    return {"source": str(src), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _cache_hit(
    paths: Mapping[str, Path], src: Path, cfg: Any
) -> Optional[DecodeResult]:
    """判断能否直接复用缓存；能则返回已有 DecodeResult。"""
    meta = _load_meta(paths["meta"])
    if not meta or not meta.get("ok"):
        return None
    signature = _source_signature(src)
    for key in ("source", "size", "mtime_ns"):
        if meta.get(key) != signature[key]:
            return None
    wav16k = Path(meta.get("wav16k") or paths["wav16k"])
    if not wav16k.is_file() or wav16k.stat().st_size != int(meta.get("wav16k_bytes") or -1):
        return None
    try:
        info = wav_info(wav16k)
    except DecodeError:
        return None
    if abs(info.sample_rate - int(meta.get("sample_rate") or 0)) > 0:
        return None
    duration = float(meta.get("duration_s") or 0.0)
    if duration and abs(info.duration_s - duration) > 1.0:
        return None
    wav44k = meta.get("wav44k")
    wav44k_path = Path(wav44k) if wav44k and Path(wav44k).is_file() else None
    return DecodeResult(
        ok=True,
        job_id=str(meta.get("job_id") or ""),
        wav16k=str(wav16k),
        wav44k=str(wav44k_path) if wav44k_path else None,
        duration_s=duration,
        sample_rate=int(meta.get("sample_rate") or info.sample_rate),
        orig_sample_rate=int(meta.get("orig_sample_rate") or 0),
        channels=int(meta.get("channels") or 0),
        bit_rate=int(meta.get("bit_rate") or 0),
        repaired=bool(meta.get("repaired")),
        decode_source=str(meta.get("decode_source") or ""),
        codec=str(meta.get("codec") or ""),
        format_name=str(meta.get("format_name") or ""),
        from_cache=True,
        repair=dict(meta.get("repair") or {}),
    )


def cleanup_cache(
    cache_dir: str | Path,
    job_id: Optional[str] = None,
    keep_16k: bool = True,
    keep_44k: bool = False,
    keep_meta: bool = False,
) -> List[str]:
    """清理中间 wav / 修复副本（阶段结束调用）；返回删除的文件列表。"""
    cdir = Path(cache_dir)
    if not cdir.is_dir():
        return []
    if job_id is None:
        candidates = list(cdir.iterdir())
    else:
        candidates = [p for p in cdir.iterdir() if p.name.startswith(f"{job_id}.")]
    removed: List[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        name = path.name
        keep = (
            (keep_16k and name.endswith(CACHE_SUFFIX_16K))
            or (keep_44k and name.endswith(CACHE_SUFFIX_44K))
            or (keep_meta and name.endswith(CACHE_SUFFIX_META))
        )
        if keep:
            continue
        try:
            path.unlink()
            removed.append(str(path))
        except OSError:
            log.warning("清理缓存失败：%s", path)
    return removed


def cache_size_bytes(cache_dir: str | Path) -> int:
    cdir = Path(cache_dir)
    if not cdir.is_dir():
        return 0
    return sum(p.stat().st_size for p in cdir.iterdir() if p.is_file())


# ---------------------------------------------------------------- 主流程
def _validate_decoded(
    wav_path: Path,
    expected_duration: float,
    cfg: Any,
) -> Tuple[bool, str]:
    """校验解码产物：存在、非空、时长与 ffprobe 自洽。"""
    if not wav_path.is_file():
        return False, "ffmpeg 未产出 wav 文件"
    try:
        info = wav_info(wav_path)
    except DecodeError as exc:
        return False, f"wav 头异常：{exc}"
    if info.n_frames <= 0:
        return False, "解码产物没有采样（0 帧）"
    tol = float(cfg.get("decode.duration_tolerance_s", 5.0) if cfg else 5.0)
    ratio = float(cfg.get("decode.min_duration_ratio", 0.9) if cfg else 0.9)
    if expected_duration > 0:
        if info.duration_s < expected_duration * ratio or info.duration_s > expected_duration + tol:
            return False, (
                f"解码时长异常：wav={info.duration_s:.2f}s，"
                f"ffprobe={expected_duration:.2f}s（容差 {ratio:.0%}/{tol}s）"
            )
    return True, ""


def _analysis_sr(cfg: Any) -> int:
    return int(cfg.get("decode.analysis_sr", ANALYSIS_SR) if cfg else ANALYSIS_SR)


def _render_sr(cfg: Any) -> int:
    return int(cfg.get("decode.render_sr", RENDER_SR) if cfg else RENDER_SR)


def _channels(cfg: Any) -> int:
    return int(cfg.get("decode.channels", 1) if cfg else 1)


def _sample_fmt(cfg: Any) -> str:
    return str(cfg.get("decode.sample_fmt", PCM_F32) if cfg else PCM_F32)


def _timeout(cfg: Any) -> float:
    return float(cfg.get("decode.timeout_s", 1800.0) if cfg else 1800.0)


def decode(
    src: str | Path,
    cache_dir: Optional[str | Path] = None,
    job_id: Optional[str] = None,
    need_44k: bool = False,
    cfg: Any = None,
    use_cache: bool = True,
    force: bool = False,
    cleanup_fixed: Optional[bool] = None,
) -> DecodeResult:
    """解码源文件（带 stco 修复重试与缓存）。

    返回 :class:`DecodeResult`；失败时 ``ok=False`` 且 ``error`` 说明原因（不抛异常，
    便于批处理记录 stage 状态）。
    """
    started = time.perf_counter()
    timings: Dict[str, float] = {}
    src_path = Path(src)
    job_id = job_id or make_job_id(src_path)

    if cache_dir is None:
        cache_dir = _default_cache_dir(cfg)
    cache = cache_paths(cache_dir, job_id, src_path)
    cache["wav16k"].parent.mkdir(parents=True, exist_ok=True)

    if not src_path.is_file():
        return DecodeResult(ok=False, job_id=job_id, error=f"输入文件不存在：{src_path}",
                            decode_source=str(src_path))

    # 1) 探测
    t0 = time.perf_counter()
    try:
        info = probe_media(src_path, timeout_s=min(120.0, _timeout(cfg)))
    except DecodeError as exc:
        return DecodeResult(ok=False, job_id=job_id, error=str(exc), decode_source=str(src_path),
                            timings={"probe_s": time.perf_counter() - t0})
    timings["probe_s"] = time.perf_counter() - t0

    # 2) 缓存命中
    if use_cache and not force:
        hit = _cache_hit(cache, src_path, cfg)
        if hit is not None:
            hit.job_id = job_id
            hit.duration_s = hit.duration_s or info.duration_s
            if need_44k and not hit.wav44k:
                hit = _decode_44k(hit, source_for_render=src_path, cache=cache, info=info,
                                  cfg=cfg, timings=timings, src_path=src_path)
            hit.timings = {**timings, "total_s": time.perf_counter() - started}
            log.info("解码缓存命中：%s（%.1fs）", job_id, hit.timings["total_s"])
            return hit

    # 3) 16 kHz 解码（失败则修复容器后重试）
    repaired = False
    decode_source = src_path
    repair_info: Dict[str, Any] = {}
    errors: List[str] = []
    t0 = time.perf_counter()
    ok, reason = _try_decode_once(src_path, cache["wav16k"], info, cfg)
    timings["decode16k_s"] = time.perf_counter() - t0
    if not ok:
        errors.append(f"首轮解码失败：{reason}")
        repair_enabled = bool(cfg.get("decode.repair.enabled", True) if cfg else True)
        if repair_enabled and is_iso_bmff(src_path):
            t_repair = time.perf_counter()
            result = repair_mp4(src_path, cache["fixed"])
            timings["repair_s"] = time.perf_counter() - t_repair
            repair_info = result.to_dict()
            if result.ok and result.repaired:
                repaired = True
                decode_source = Path(result.dst)
                t0 = time.perf_counter()
                ok, reason = _try_decode_once(decode_source, cache["wav16k"], info, cfg)
                timings["decode16k_retry_s"] = time.perf_counter() - t0
                if not ok:
                    errors.append(f"修复 stco（delta={result.delta}）后仍解码失败：{reason}")
            elif result.ok:
                errors.append("容器无需修复（stco 偏移本就正确），不重试")
            else:
                errors.append(f"容器修复失败：{result.error}")
        elif not repair_enabled:
            errors.append("decode.repair.enabled=false，跳过修复重试")
        else:
            errors.append("非 MP4/M4A 容器，跳过 stco 修复")

    if not ok:
        # 清掉可能的半成品
        for key in ("wav16k", "wav44k"):
            try:
                cache[key].unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass
        _cleanup_fixed(cache, cfg, cleanup_fixed)
        elapsed = time.perf_counter() - started
        timings["total_s"] = elapsed
        msg = " | ".join(errors) or "未知解码失败"
        log.error("解码失败：%s：%s", job_id, msg)
        return DecodeResult(
            ok=False, job_id=job_id, repaired=repaired, decode_source=str(decode_source),
            orig_sample_rate=info.sample_rate, channels=info.channels, bit_rate=info.bit_rate,
            codec=info.codec, format_name=info.format_name, error=msg,
            timings=timings, repair=repair_info,
        )

    # 4) 按需 44.1 kHz（渲染轴）
    wav44k: Optional[Path] = None
    if need_44k:
        t0 = time.perf_counter()
        try:
            decode_audio(decode_source, cache["wav44k"], _render_sr(cfg), _channels(cfg),
                         _sample_fmt(cfg), _timeout(cfg))
            wav44k = cache["wav44k"]
        except DecodeError as exc:
            errors.append(f"44.1k 渲染解码失败：{exc}")
            log.warning("44.1k 解码失败（16k 可用）：%s", job_id)
        timings["decode44k_s"] = time.perf_counter() - t0

    # 5) 写缓存元数据 + 清理修复副本
    wav16k = cache["wav16k"]
    try:
        wav = wav_info(wav16k)
        duration = wav.duration_s
    except DecodeError:  # pragma: no cover - 上一步已校验
        duration = info.duration_s
    timings["total_s"] = time.perf_counter() - started
    meta = {
        **_source_signature(src_path),
        "job_id": job_id,
        "ok": True,
        "repaired": repaired,
        "decode_source": str(decode_source),
        "duration_s": round(duration, 3),
        "sample_rate": _analysis_sr(cfg),
        "orig_sample_rate": info.sample_rate,
        "channels": info.channels,
        "bit_rate": info.bit_rate,
        "codec": info.codec,
        "format_name": info.format_name,
        "wav16k": str(wav16k),
        "wav16k_bytes": wav16k.stat().st_size,
        "wav44k": str(wav44k) if wav44k else "",
        "repair": repair_info,
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    _write_meta(cache["meta"], meta)
    _cleanup_fixed(cache, cfg, cleanup_fixed)

    return DecodeResult(
        ok=True,
        job_id=job_id,
        wav16k=str(wav16k),
        wav44k=str(wav44k) if wav44k else None,
        duration_s=duration,
        sample_rate=_analysis_sr(cfg),
        orig_sample_rate=info.sample_rate,
        channels=info.channels,
        bit_rate=info.bit_rate,
        repaired=repaired,
        decode_source=str(decode_source),
        codec=info.codec,
        format_name=info.format_name,
        error="" if not errors else "；".join(errors),
        timings=timings,
        repair=repair_info,
    )


def _try_decode_once(
    src: Path, dst: Path, info: MediaInfo, cfg: Any
) -> Tuple[bool, str]:
    """一次解码 + 校验；返回 (ok, 失败原因)。"""
    try:
        decode_audio(src, dst, _analysis_sr(cfg), _channels(cfg), _sample_fmt(cfg), _timeout(cfg))
    except DecodeError as exc:
        return False, str(exc)
    return _validate_decoded(dst, info.duration_s, cfg)


def _decode_44k(
    hit: DecodeResult,
    source_for_render: Path,
    cache: Mapping[str, Path],
    info: MediaInfo,
    cfg: Any,
    timings: Dict[str, float],
    src_path: Path,
) -> DecodeResult:
    """缓存命中但缺 44.1k 时补解（修复过的源需要重新修复一次）。"""
    render_src = source_for_render
    fixed_path = cache["fixed"]
    if hit.repaired:
        result = repair_mp4(src_path, fixed_path)
        if result.ok and result.repaired:
            render_src = Path(result.dst)
        else:
            log.warning("补解 44.1k 时重新修复容器失败：%s", result.error)
            return hit
    t0 = time.perf_counter()
    try:
        decode_audio(render_src, cache["wav44k"], _render_sr(cfg), _channels(cfg),
                     _sample_fmt(cfg), _timeout(cfg))
        hit.wav44k = str(cache["wav44k"])
    except DecodeError as exc:
        log.warning("补解 44.1k 失败：%s", exc)
        return hit
    finally:
        timings["decode44k_s"] = time.perf_counter() - t0
        if hit.repaired:
            try:
                fixed_path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass
    meta = _load_meta(cache["meta"]) or {}
    meta["wav44k"] = hit.wav44k or ""
    _write_meta(cache["meta"], meta)
    return hit


def _cleanup_fixed(cache: Mapping[str, Path], cfg: Any, cleanup_fixed: Optional[bool]) -> None:
    keep = bool(cfg.get("decode.repair.keep_fixed_file", False) if cfg else False)
    if cleanup_fixed is not None:
        keep = not cleanup_fixed
    if keep:
        return
    try:
        cache["fixed"].unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        pass


def _default_cache_dir(cfg: Any) -> Path:
    if cfg is not None:
        try:
            return cfg.path("cache_dir")
        except Exception:  # pragma: no cover - 配置异常时兜底
            pass
    return Path(__file__).resolve().parents[2] / "work" / "cache"


def ensure_44k(
    src: str | Path,
    cache_dir: Optional[str | Path] = None,
    job_id: Optional[str] = None,
    cfg: Any = None,
) -> Optional[Path]:
    """确保 44.1k 渲染轴 wav 存在（P3 渲染阶段调用）；失败返回 None。"""
    result = decode(src, cache_dir=cache_dir, job_id=job_id, need_44k=True, cfg=cfg)
    if result.ok and result.wav44k:
        return Path(result.wav44k)
    return None
