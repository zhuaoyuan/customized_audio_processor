# -*- coding: utf-8 -*-
"""输出音频（P3-T4）+ P3 渲染编排（P6 完整编排器之前的可运行实现）。

职责
----
1. **编码**：44.1 kHz mono float32 wav → AAC 96 kbps m4a（``-c:a aac -b:a 96k -movflags +faststart``）；
   用 ffprobe 回读校验 44100/mono/aac/96k。
2. **真峰值闭环**：AAC 编码会在硬限幅的瞬态上产生 2–3 dB 的码流过冲（f00 实测：渲染 wav TP −2.7 dBTP
   → 解码 m4a 后 +0.4 dBTP），所以不能只看 wav 的限幅结果。这里的做法（与 P3-T3 的 wav 级限幅配合）：
   渲染 → 编码 → ``ebur128=peak=true`` 回读 m4a 真峰值 → 不达标就**下调 ceiling 重渲染**（最多
   ``loudness.limiter.max_iterations`` 次），同时回读 m4a 的 ASL，超过 ±0.5 dB 就调 gain_offset。
3. **产物**：``out/<id>/clean.m4a``、``edit.json``（kept/removed/gain 契约）、``report.json``
   （decode/vad/levels/trim/loudness/timing 段，字段名按 ``docs/实施计划.md`` §4.1）。
4. **单文件全流程** ``render_job()``：decode(need_44k) → analyze(缓存) → denoise → trim → loudness
   → encode/TP 闭环 → 写产物；P3 的 CLI ``render`` 与 ``app/tools/regress.py`` 共用它。

clean 时长口径见 ``app/pipeline/trim.py`` 与 ``docs/工程日志.md``「P3 开工第一件事」。
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_setup import get_logger
from ..models import (
    REMOVE_HEAD,
    REMOVE_INTERNAL_SILENCE,
    REMOVE_TAIL,
    SPEAKER_MAIN,
    AnalyzeResult,
    DecodeResult,
    GainInfo,
)
from . import loudness as loud_mod
from .analyze import (
    frame_power,
    measure_ebur128,
    p56_level,
    speech_mask,
    window_levels,
)
from .decode import (
    cache_paths as decode_cache_paths,
    decode as run_decode,
    read_wav_float32,
)
from .denoise import (
    DenoiseError,
    build_filter_chain,
    denoise_44k,
    resolve_nr,
)
from .trim import (
    TrimPlan,
    chain_clean_segments,
    edit_map,
    plan_trim,
    render_clean,
    write_edit_json,
)

log = get_logger(__name__)

SR44 = 44100
SR16 = 16000
CACHE_CLEAN44 = ".clean44k.wav"
CACHE_NORM44 = ".normalized44k.wav"
CACHE_DENOISED = ".denoised44k.wav"

#: 产物文件名（T4 契约）
PRODUCT_AUDIO = "clean.m4a"
PRODUCT_EDIT = "edit.json"
PRODUCT_REPORT = "report.json"

#: AAC 编码后真峰值回读的临时文件名
_TMP_ENCODE = ".clean.tmp.m4a"


class ExportError(RuntimeError):
    """输出/渲染编排失败（编码失败、字节数/格式不符、产物写入失败等）。"""


# ---------------------------------------------------------------- ffmpeg/ffprobe 封装
def _run(cmd: Sequence[str], timeout_s: float = 3600.0, what: str = "ffmpeg") -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExportError(f"{what} 超时（>{timeout_s}s）：{cmd[-1]}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise ExportError(f"{what} 失败（exit={proc.returncode}）：{cmd[-1]}\n{err[-1500:]}")
    return proc


def probe_audio(path: str | Path) -> Dict[str, Any]:
    """ffprobe 读第一条音频流 + 容器信息（编码/采样率/声道/码率/时长/字节数）。"""
    p = Path(path)
    if not p.is_file():
        raise ExportError(f"文件不存在：{p}")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bit_rate,channel_layout,duration",
        "-show_entries", "format=duration,bit_rate,size,format_name",
        "-of", "json", str(p),
    ]
    proc = _run(cmd, timeout_s=120.0, what="ffprobe")
    try:
        data = json.loads(proc.stdout.decode("utf-8", "replace"))
    except ValueError as exc:
        raise ExportError(f"ffprobe 输出无法解析：{p}") from exc
    streams = data.get("streams") or []
    if not streams:
        raise ExportError(f"ffprobe 找不到音频流：{p}")
    stream = streams[0]
    fmt = data.get("format") or {}
    return {
        "codec_name": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "stream_bit_rate": int(stream.get("bit_rate") or 0),
        "format_bit_rate": int(float(fmt.get("bit_rate") or 0)),
        "format_name": fmt.get("format_name"),
        "duration_s": float(stream.get("duration") or fmt.get("duration") or 0.0),
        "size_bytes": int(fmt.get("size") or p.stat().st_size),
    }


def decode_to_wav(src: str | Path, dst: str | Path, sample_rate: int) -> Path:
    """用 ffmpeg 把任意音频解码成 mono float32 wav（测量/回读用）。"""
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-i", str(src), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(int(sample_rate)),
        "-c:a", "pcm_f32le", "-f", "wav", str(out),
    ], what="ffmpeg 解码")
    return out


def wav_peak_check(path: str | Path, threshold: float = 0.999) -> Dict[str, float]:
    """分块读 wav，求 max|x| 与 |x|>=threshold 的样本数（削波检查，不整段进内存）。"""
    import scipy.io.wavfile as wavfile

    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data[:, 0]
    n = int(data.shape[0])
    max_abs = 0.0
    n_over = 0
    chunk = 1 << 22
    for a in range(0, n, chunk):
        part = np.abs(np.asarray(data[a:a + chunk], dtype=np.float32))
        if part.size:
            max_abs = max(max_abs, float(part.max()))
            n_over += int((part >= threshold).sum())
    return {"max_abs": max_abs, "n_over": n_over, "samples": n,
            "duration_s": n / float(sr) if sr else 0.0, "sample_rate": int(sr)}


# ---------------------------------------------------------------- 编码 + 测量
def encode_m4a(wav44: str | Path, m4a: str | Path, cfg: Any) -> Dict[str, Any]:
    """44.1k float32 wav → mono AAC m4a，并按契约回读校验格式。"""
    src, dst = Path(wav44), Path(m4a)
    if not src.is_file():
        raise ExportError(f"待编码 wav 不存在：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    codec = str(cfg.get("output.audio_codec", "aac") if cfg else "aac")
    bitrate = str(cfg.get("output.audio_bitrate", "96k") if cfg else "96k")
    channels = int(cfg.get("output.audio_channels", 1) if cfg else 1)
    t0 = time.perf_counter()
    _run([
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-i", str(src), "-map", "0:a:0", "-vn",
        "-c:a", codec, "-b:a", bitrate,
        "-ar", str(SR44), "-ac", str(channels),
        "-movflags", "+faststart", str(dst),
    ], what="ffmpeg 编码")
    elapsed = time.perf_counter() - t0
    info = probe_audio(dst)
    errors: List[str] = []
    if info["sample_rate"] != SR44:
        errors.append(f"采样率 {info['sample_rate']} != {SR44}")
    if info["channels"] != channels:
        errors.append(f"声道数 {info['channels']} != {channels}")
    if info["codec_name"] not in ("aac",):
        errors.append(f"编码 {info['codec_name']} != aac")
    target_bps = _parse_bitrate(bitrate)
    actual_bps = info["format_bit_rate"] or info["stream_bit_rate"]
    # 容器码率含封装开销，允许 -10% ~ +25%
    if target_bps and actual_bps and not (0.90 * target_bps <= actual_bps <= 1.25 * target_bps):
        errors.append(f"码率 {actual_bps} 偏离目标 {target_bps} 超过容差")
    if errors:
        raise ExportError("m4a 编码结果不合契约：" + "；".join(errors))
    info.update({"ok": True, "target_bit_rate": bitrate, "elapsed_s": round(elapsed, 3)})
    return info


def _parse_bitrate(text: str) -> int:
    t = str(text).strip().lower()
    try:
        if t.endswith("k"):
            return int(float(t[:-1]) * 1000)
        if t.endswith("m"):
            return int(float(t[:-1]) * 1_000_000)
        return int(float(t))
    except ValueError:
        return 0


def measure_m4a(m4a: str | Path, speech_intervals: Sequence[Sequence[float]], cfg: Any,
                tmp_dir: Optional[Path] = None) -> Dict[str, Any]:
    """回读 m4a：EBU R128 真峰值/LUFS + 44.1k 削波检查 + 16k ASL（与 P2 同口径）。"""
    src = Path(m4a)
    t0 = time.perf_counter()
    r128 = measure_ebur128(src)
    t_r128 = time.perf_counter() - t0

    tmp_dir = Path(tmp_dir) if tmp_dir else src.parent
    dec44 = tmp_dir / f".{src.stem}.dec44.wav"
    dec16 = tmp_dir / f".{src.stem}.dec16.wav"
    try:
        t0 = time.perf_counter()
        decode_to_wav(src, dec44, SR44)
        peak = wav_peak_check(dec44)
        t_peak = time.perf_counter() - t0

        t0 = time.perf_counter()
        decode_to_wav(src, dec16, SR16)
        sr, x16 = read_wav_float32(dec16)
        power, centers = frame_power(x16, sr)
        intervals = [(float(a), float(b)) for a, b in speech_intervals]
        mask = speech_mask(intervals, len(power), centers)
        rel_db = float(cfg.get("levels.rel_db", 15.0) if cfg else 15.0)
        asl = p56_level(power[mask], rel_db=rel_db) if mask.any() else float("nan")
        duration = len(x16) / float(sr) if sr else 0.0
        win = window_levels(
            power, mask, centers, duration,
            win_s=float(cfg.get("levels.window_s", 300.0) if cfg else 300.0),
            min_speech_s=float(cfg.get("levels.window_min_speech_s", 30.0) if cfg else 30.0),
            rel_db=rel_db,
        )
        t_asl = time.perf_counter() - t0
    finally:
        for p in (dec44, dec16):
            try:
                p.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass

    return {
        "lufs": r128.get("lufs"),
        "lra": r128.get("lra"),
        "tp_dbtp": r128.get("tp_dbtp"),
        "asl_out_dbfs": round(float(asl), 2) if asl == asl else None,
        "window5m": win,
        "window5m_spread_out_db": win.get("spread_db"),
        "max_abs": round(peak["max_abs"], 5),
        "n_over_threshold": int(peak["n_over"]),
        "decoded_duration_s": round(peak["duration_s"], 3),
        "timings": {"ebur128_s": round(t_r128, 2), "peak_s": round(t_peak, 2),
                    "asl_s": round(t_asl, 2)},
        "error": r128.get("error", ""),
    }


# ---------------------------------------------------------------- report/edit.json
def _clean_json(value: Any) -> Any:
    """递归把 NaN/Inf 换成 None，保证 report.json 是合法 JSON。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_clean_json(payload), ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    tmp.replace(p)
    return p


def speech_preserved_ratio(analyze: AnalyzeResult, plan: TrimPlan) -> float:
    """``[head, tail]`` 内的 VAD 语音保留率（内部切只切非语音，应恒为 1.0）。

    头尾裁剪**本来就会切掉**片头/片尾的 VAD 误判（音乐、掌声、课前闲聊），所以分母只算
    ``[head, tail]`` 内的 VAD 语音；把 0–head / tail–duration 的 VAD 段也算进来的
    ``speech_preserved_ratio_incl_head_tail`` 仅作参考（f24 约 0.94，属预期）。
    """
    kept = plan.kept_intervals  # TrimPlan 的 property：[(orig_start, orig_end), ...]
    head, tail = float(plan.head_s), float(plan.tail_s)
    total = 0.0
    kept_total = 0.0
    for seg in analyze.segments or []:
        s, e = float(seg[0]), float(seg[1])
        s, e = max(s, head), min(e, tail)
        if e <= s:
            continue
        total += e - s
        for ks, ke in kept:
            lo, hi = max(s, ks), min(e, ke)
            if hi > lo:
                kept_total += hi - lo
    return round(kept_total / total, 6) if total > 0 else 1.0


def speech_preserved_ratio_incl_head_tail(analyze: AnalyzeResult, plan: TrimPlan) -> float:
    """VAD 语音段（含 head/tail 裁剪区）与 kept 的交集比例，仅作参考。"""
    kept = plan.kept_intervals
    total = 0.0
    kept_total = 0.0
    for seg in analyze.segments or []:
        s, e = float(seg[0]), float(seg[1])
        if e <= s:
            continue
        total += e - s
        for ks, ke in kept:
            lo, hi = max(s, ks), min(e, ke)
            if hi > lo:
                kept_total += hi - lo
    return round(kept_total / total, 6) if total > 0 else 1.0


def build_report(
    *,
    job_id: str,
    source: str,
    decoded: DecodeResult,
    analyze: AnalyzeResult,
    plan: TrimPlan,
    loudness_result: Optional[loud_mod.LoudnessResult],
    m4a_info: Dict[str, Any],
    m4a_meas: Dict[str, Any],
    denoise_info: Dict[str, Any],
    timings: Dict[str, float],
    asl_target_dbfs: float = -20.0,
) -> Dict[str, Any]:
    """拼 ``report.json``（契约 §4.1 的字段名；额外段：denoise/loudness/output）。"""
    levels_in = analyze.levels or {}
    trim_in = analyze.trim or {}
    # P3-T4 之后补：回读（measure_m4a）自身耗时的分项，便于诊断 60s 级回读
    meas_timings = m4a_meas.get("timings") if isinstance(m4a_meas.get("timings"), dict) else {}
    report: Dict[str, Any] = {
        "id": job_id,
        "source": str(source),
        "duration_s": round(float(decoded.duration_s or analyze.duration_s), 3),
        "sample_rate": int(decoded.sample_rate or analyze.sample_rate or SR16),
        "bit_rate": int(decoded.bit_rate or 0),
        "decode": decoded.report_decode(),
        "vad": analyze.report_vad(),
        "levels": {
            # P2 原始口径（未降噪、orig 时间轴）
            "asl_in_dbfs": levels_in.get("asl_in_dbfs"),
            "asl_p56_in_dbfs": levels_in.get("asl_p56_dbfs"),
            "lufs_in": levels_in.get("lufs_in"),
            "true_peak_in_dbtp": levels_in.get("true_peak_in_dbtp"),
            "noise_floor_dbfs": levels_in.get("noise_floor_dbfs"),
            "snr_db": levels_in.get("snr_db"),
            "window5m_spread_in_db": levels_in.get("window5m_spread_in_db"),
            # P3 渲染口径（clean 16k / 输出 m4a）
            "asl_clean_dbfs": (round(float(loudness_result.asl_in_dbfs), 2)
                               if loudness_result and loudness_result.asl_in_dbfs == loudness_result.asl_in_dbfs
                               else None),
            "asl_out_dbfs": m4a_meas.get("asl_out_dbfs"),
            "asl_dev_db": (round(float(m4a_meas["asl_out_dbfs"]) - float(asl_target_dbfs), 2)
                           if m4a_meas.get("asl_out_dbfs") is not None else None),
            "lufs_out": m4a_meas.get("lufs"),
            "lra_out": m4a_meas.get("lra"),
            "true_peak_out_dbtp": m4a_meas.get("tp_dbtp"),
            "max_abs_out": m4a_meas.get("max_abs"),
            "clipped_samples_out": m4a_meas.get("n_over_threshold"),
            "window5m_spread_out_db": m4a_meas.get("window5m_spread_out_db"),
        },
        "trim": {
            "head_s": round(float(plan.head_s), 3),
            "tail_s": round(float(plan.tail_s), 3),
            "internal_cut_s": round(float(plan.internal_cut_s), 3),
            "clean_dur_s": round(float(plan.clean_duration_s), 3),
            "cut_ratio": plan.cut_ratio,
            "speech_preserved_ratio": speech_preserved_ratio(analyze, plan),
            "speech_preserved_ratio_incl_head_tail": speech_preserved_ratio_incl_head_tail(analyze, plan),
            "n_kept": len(plan.kept),
            "n_removed": len(plan.removed),
            "head_method": trim_in.get("head_method"),
            "flags": trim_in.get("flags", []),
        },
        "denoise": dict(denoise_info or {}),
        "output": {
            "file": PRODUCT_AUDIO,
            "codec": m4a_info.get("codec_name"),
            "sample_rate": m4a_info.get("sample_rate"),
            "channels": m4a_info.get("channels"),
            "bit_rate": m4a_info.get("format_bit_rate"),
            "size_bytes": m4a_info.get("size_bytes"),
            "duration_s": m4a_info.get("duration_s"),
            "target_bit_rate": m4a_info.get("target_bit_rate"),
        },
        "measure": {
            "decoded_duration_s": m4a_meas.get("decoded_duration_s"),
            "ebur128_s": meas_timings.get("ebur128_s"),
            "peak_s": meas_timings.get("peak_s"),
            "asl_s": meas_timings.get("asl_s"),
        },
        "timing": {k: (round(float(v), 2) if isinstance(v, (int, float)) else v)
                   for k, v in timings.items()},
    }
    if loudness_result is not None:
        report["loudness"] = {
            "gain_db": round(float(loudness_result.gain_db), 2),
            "gain": loudness_result.gain.as_dict(),
            "v2": loudness_result.v2,
            "limiter": loudness_result.limiter,
            "iterations": loudness_result.iterations,
            "tp_wav_dbtp": loudness_result.tp_wav_dbtp,
            "lufs_wav": loudness_result.lufs_out,
            "asl_dev_wav_db": loudness_result.asl_dev_db,
        }
    return _clean_json(report)


# ---------------------------------------------------------------- 单文件全流程
def _expected_static_gain(cfg: Any, asl_in_dbfs: float, target_dbfs: float) -> float:
    """loudness.render 里静态增益的期望值 ``clip(T - asl_in)``（用于跨轮传递 gain_offset）。"""
    gmin = float(cfg.get("loudness.gain_min_db", -24.0) if cfg else -24.0)
    gmax = float(cfg.get("loudness.gain_max_db", 24.0) if cfg else 24.0)
    base = float(target_dbfs) - float(asl_in_dbfs)
    return max(gmin, min(gmax, base))


def _cache_paths(cache_dir: Path, job_id: str) -> Dict[str, Path]:
    return {
        "denoised": cache_dir / f"{job_id}{CACHE_DENOISED}",
        "clean44": cache_dir / f"{job_id}{CACHE_CLEAN44}",
        "norm44": cache_dir / f"{job_id}{CACHE_NORM44}",
    }


def _load_analyze(decoded: DecodeResult, job_id: str, cfg: Any) -> AnalyzeResult:
    """读 analyze 缓存；缓存缺失/失效时**一次性**重跑完整分析（含 ASR 边界）。

    这里直接走 ``analyze()``（带 SenseVoice 转写器）而不是先调 ``analyze_wav``：后者在
    没有 transcriber 时会先跑一遍 VAD-only 边界并**覆盖缓存**（flags=asr_disabled），再被
    第二次带 ASR 的分析覆盖——既慢又会在中途留下错误缓存。``analyze()`` 的缓存命中路径
    在真正用转写器之前返回（只多花 ~1.2 s 加载模型），缓存失效时只跑一遍。
    """
    if not decoded.wav16k:
        raise ExportError("decode 结果缺少 16k wav，无法读取分析结果")
    from .analyze import analyze as run_analyze

    result = run_analyze(decoded.wav16k, cfg, job_id=job_id, run_boundary=True, use_cache=True)
    if result.from_cache:
        log.debug("analyze 缓存命中：%s", job_id)
    else:
        log.info("analyze 缓存失效，已重跑完整分析（含 ASR 边界）：%s（%.1fs）",
                 job_id, float(result.timings.get("total_s", 0.0)))
    return result


def render_job(
    job_id: str,
    source: str | Path,
    cfg: Any,
    *,
    out_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    force: bool = False,
    keep_intermediates: bool = False,
    skip_existing: bool = True,
    on_stage: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """对单个文件跑 P1–P3（decode→analyze→denoise→trim→loudness→export），返回结果字典。

    返回值：``{"ok", "job_id", "source", "out_dir", "skipped", "stage", "error",
    "report", "edit", "timings"}``；失败不抛异常（stage 标明失败阶段），便于批处理。
    """
    started = time.perf_counter()
    source_p = Path(source)
    out_root = Path(out_dir) if out_dir else cfg.path("out_dir")
    cache_root = Path(cache_dir) if cache_dir else cfg.path("cache_dir")
    job_out = out_root / job_id
    caches = _cache_paths(cache_root, job_id)
    # 回读临时 wav 的 per-job 目录（放 cache_dir，避免外置输出盘慢 + 多 job 并发时重名）
    measure_tmp = cache_root / f"{job_id}.measure"
    tmp_m4a = job_out / _TMP_ENCODE
    timings: Dict[str, float] = {}
    stage = "decode"
    result: Dict[str, Any] = {
        "ok": False, "job_id": job_id, "source": str(source_p), "out_dir": str(job_out),
        "skipped": False, "stage": stage, "error": "", "report": None, "edit": None,
        "timings": timings,
    }

    def notify(stage_name: str, status: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        if on_stage is not None:
            try:
                on_stage(stage_name, status, metrics or {})
            except Exception:  # pragma: no cover - 回调失败不影响主流程
                log.debug("on_stage 回调失败", exc_info=True)

    # 已有产物 → 跳过（--force 重做）
    if skip_existing and not force:
        report_path = job_out / PRODUCT_REPORT
        if (job_out / PRODUCT_AUDIO).is_file() and report_path.is_file():
            try:
                old = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = None
            if isinstance(old, dict) and old.get("ok", True) and old.get("id") == job_id:
                log.info("产物已存在，跳过：%s", job_out)
                result.update({
                    "ok": True, "skipped": True, "stage": "export",
                    "report": old, "timings": old.get("timing", {}),
                })
                for st in ("decode", "analyze", "denoise", "trim", "loudness", "export"):
                    notify(st, "done", {})
                return result

    try:
        # 1) 解码（16k 分析轴 + 44.1k 渲染轴，走 P1 缓存）
        stage = "decode"
        t0 = time.perf_counter()
        decoded = run_decode(
            source_p, cache_dir=cache_root, job_id=job_id, need_44k=True,
            cfg=cfg, force=False,
        )
        timings["decode_s"] = time.perf_counter() - t0
        if not decoded.ok or not decoded.wav44k:
            raise ExportError(f"解码失败：{decoded.error or '缺少 44.1k wav'}")
        notify("decode", "done", {"duration_s": decoded.duration_s, "repaired": decoded.repaired,
                                  "elapsed_s": timings["decode_s"]})

        # 2) 分析（缓存命中；P2 已产出边界/长静音/电平）
        stage = "analyze"
        t0 = time.perf_counter()
        analyze_res = _load_analyze(decoded, job_id, cfg)
        timings["analyze_s"] = time.perf_counter() - t0
        notify("analyze", "done", {"from_cache": analyze_res.from_cache,
                                   "elapsed_s": timings["analyze_s"]})

        # 3) 降噪（44.1k，输出 work/cache/<id>.denoised44k.wav）
        stage = "denoise"
        t0 = time.perf_counter()
        snr_db = analyze_res.levels.get("snr_db") if analyze_res.levels else None
        denoised = denoise_44k(Path(decoded.wav44k), cfg, job_id, snr_db=snr_db)
        timings["denoise_s"] = time.perf_counter() - t0
        nr_used = resolve_nr(cfg, snr_db=snr_db)
        try:
            af, _cwd = build_filter_chain(cfg, nr=nr_used, src_sr=SR44)
        except Exception:  # pragma: no cover - 仅用于报告
            af = ""
        denoise_info = {
            "mode": str(cfg.get("denoise.mode", "afftdn") if cfg else "afftdn"),
            "filter": af,
            "nr": nr_used,
            "snr_db": snr_db,
            "elapsed_s": round(timings["denoise_s"], 2),
        }
        notify("denoise", "done", dict(denoise_info))

        # 4) 裁剪（44.1k clean wav + kept 映射）
        stage = "trim"
        t0 = time.perf_counter()
        plan = plan_trim(analyze_res, cfg, duration_s=decoded.duration_s or analyze_res.duration_s)
        render_info = render_clean(Path(denoised), plan, caches["clean44"])
        timings["trim_s"] = time.perf_counter() - t0
        speech_intervals = chain_clean_segments(analyze_res, cfg)
        notify("trim", "done", {
            "clean_dur_s": plan.clean_duration_s, "clean_min": round(plan.clean_duration_s / 60.0, 2),
            "cut_ratio": plan.cut_ratio, "n_kept": len(plan.kept),
            "elapsed_s": timings["trim_s"],
        })

        # 5) 响度：wav 级真峰值限幅（出口）
        stage = "loudness"
        ceiling = float(cfg.get("loudness.limiter.ceiling_dbfs", -2.75) if cfg else -2.75)
        target_bitrate = str(cfg.get("output.audio_bitrate", "96k") if cfg else "96k")
        gain_offset = 0.0
        loud = loud_mod.render(
            caches["clean44"], speech_intervals, cfg,
            out44_wav=caches["norm44"], ceiling_dbtp=ceiling, gain_offset_db=gain_offset,
            drift_db=(analyze_res.levels or {}).get("window5m_spread_in_db"),
            analysis_asl_in=(analyze_res.levels or {}).get("asl_in_dbfs"),
        )
        timings["render_s"] = float(loud.timings.get("total_s", 0.0))
        notify("loudness", "done", {
            "asl_in_dbfs": loud.asl_in_dbfs, "asl_out_dbfs": loud.asl_out_dbfs,
            "gain_db": loud.gain_db, "iterations": loud.iterations,
            "elapsed_s": timings["render_s"],
        })

        # 6) 编码 + 真峰值/ASL 闭环（AAC 码流过冲 → 下调 ceiling 重渲染）
        stage = "export"
        job_out.mkdir(parents=True, exist_ok=True)
        # 回读的 dec44/dec16 临时 wav 约 1.2 GB，写到 cache_dir 的 per-job 子目录
        # （默认 C: 本地盘）而不是输出盘；外置/慢盘（实测 D: USB SSD 写 936 MB
        # 需 ~27 s）会把它拖成 60 s 级。per-job 子目录避免多 job 并发时临时文件重名。
        measure_tmp.mkdir(parents=True, exist_ok=True)
        m4a_path = job_out / PRODUCT_AUDIO
        target_tp = float(cfg.get("loudness.limiter.target_tp_dbtp", -1.0) if cfg else -1.0)
        max_iters = int(cfg.get("loudness.limiter.max_iterations", 3) if cfg else 3)
        margin = float(cfg.get("loudness.limiter.codec_margin_db", 0.5) if cfg else 0.5)
        asl_target = float(cfg.get("loudness.target_asl_dbfs", -20.0) if cfg else -20.0)
        encode_s = 0.0
        measure_s = 0.0
        m4a_info: Dict[str, Any] = {}
        m4a_meas: Dict[str, Any] = {}
        for attempt in range(1, max_iters + 1):
            t0 = time.perf_counter()
            encode_m4a(caches["norm44"], tmp_m4a, cfg)
            encode_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            m4a_meas = measure_m4a(tmp_m4a, speech_intervals, cfg, tmp_dir=measure_tmp)
            measure_s += time.perf_counter() - t0
            m4a_info = probe_audio(tmp_m4a)
            tp = m4a_meas.get("tp_dbtp")
            asl_out = m4a_meas.get("asl_out_dbfs")
            log.info("编码回读（第 %d/%d 次）：m4a TP=%s dBTP ASL=%s dBFS（ceiling=%.2f）",
                     attempt, max_iters, tp, asl_out, ceiling)
            tp_ok = tp is not None and tp <= target_tp
            asl_ok = asl_out is not None and abs(asl_out - asl_target) <= 0.5
            if tp_ok and asl_ok:
                break
            if attempt >= max_iters:
                log.warning("TP/ASL 闭环达到最大次数仍未完全达标：TP=%s（目标 ≤%.1f）ASL=%s（目标 %.1f±0.5）",
                            tp, target_tp, asl_out, asl_target)
                break
            if not asl_ok:
                gain_offset += asl_target - float(asl_out)
            if not tp_ok:
                # 以「回读 m4a TP 相对 wav TP 的过冲」精确下调 ceiling（不高于当前 wav TP）
                wav_tp = loud.tp_wav_dbtp
                base = min(float(wav_tp), ceiling) if wav_tp == wav_tp else ceiling
                ceiling = round(base - (float(tp) - target_tp + margin), 2)
                if ceiling < -12.0:
                    ceiling = -12.0
                log.info("m4a 真峰值 %.2f > %.2f，下调 ceiling 到 %.2f 重渲染", tp, target_tp, ceiling)
                # 换 ceiling 后限幅深度变化会让 ASL 掉一截；把本轮已收敛的静态增益
                # 直接作为下一轮起点，避免 ASL 校正从零重爬（f05 实测每遍 ~70 s）
                if loud.asl_in_dbfs == loud.asl_in_dbfs:
                    gain_offset = float(loud.gain_db) - _expected_static_gain(
                        cfg, loud.asl_in_dbfs, asl_target)
            loud = loud_mod.render(
                caches["clean44"], speech_intervals, cfg,
                out44_wav=caches["norm44"], ceiling_dbtp=ceiling, gain_offset_db=gain_offset,
                drift_db=(analyze_res.levels or {}).get("window5m_spread_in_db"),
                analysis_asl_in=(analyze_res.levels or {}).get("asl_in_dbfs"),
            )
            timings["render_s"] = float(timings.get("render_s", 0.0)) + float(loud.timings.get("total_s", 0.0))

        tmp_m4a.replace(m4a_path)
        m4a_info = probe_audio(m4a_path)
        m4a_info["target_bit_rate"] = target_bitrate
        # 循环内最后一次 measure_m4a 测的就是最终字节（replace 只改路径），不再对
        # clean.m4a 重复回读一遍——旧实现这里多花一次 ~20–68 s 且完全不计时。
        if not m4a_meas:  # 兜底：loudness.limiter.max_iterations 配成 0 时
            t0 = time.perf_counter()
            m4a_meas = measure_m4a(m4a_path, speech_intervals, cfg, tmp_dir=measure_tmp)
            measure_s += time.perf_counter() - t0
        timings["encode_s"] = encode_s
        timings["measure_s"] = measure_s

        # 7) edit.json / report.json
        t0 = time.perf_counter()
        gain_info: GainInfo = loud.gain
        write_edit_json(plan, job_out / PRODUCT_EDIT, gain=gain_info)
        total_s = time.perf_counter() - started
        timings["total_s"] = total_s
        report = build_report(
            job_id=job_id, source=str(source_p), decoded=decoded, analyze=analyze_res,
            plan=plan, loudness_result=loud, m4a_info=m4a_info, m4a_meas=m4a_meas,
            denoise_info=denoise_info, timings=timings, asl_target_dbfs=asl_target,
        )
        report["ok"] = True
        write_json(job_out / PRODUCT_REPORT, report)
        timings["write_s"] = time.perf_counter() - t0
        timings["total_s"] = time.perf_counter() - started
        # write_s/total_s 只能在首写之后得到；回填后重写一次（~ms），让 report 的
        # timing 与返回的 timings 一致（旧实现 report 里没有 write_s）。
        report["timing"] = {k: (round(float(v), 2) if isinstance(v, (int, float)) else v)
                            for k, v in timings.items()}
        write_json(job_out / PRODUCT_REPORT, report)

        result.update({
            "ok": True, "stage": "export", "report": report,
            "edit": edit_map(plan, gain=gain_info).to_dict(),
            "timings": {k: round(float(v), 2) for k, v in timings.items()},
        })
        notify("export", "done", {
            "tp_dbtp": m4a_meas.get("tp_dbtp"), "asl_out_dbfs": m4a_meas.get("asl_out_dbfs"),
            "size_bytes": m4a_info.get("size_bytes"), "elapsed_s": timings["total_s"],
        })
        log.info("渲染完成：%s clean=%.1fs 输出=%s TP=%s dBTP ASL=%s dBFS 总耗时 %.1fs",
                 job_id, plan.clean_duration_s, m4a_path.name,
                 m4a_meas.get("tp_dbtp"), m4a_meas.get("asl_out_dbfs"), timings["total_s"])
        return result

    except Exception as exc:
        msg = str(exc) or exc.__class__.__name__
        log.error("渲染失败（%s）：%s：%s", stage, job_id, msg)
        notify(stage, "failed", {"error": msg})
        result.update({"ok": False, "stage": stage, "error": msg})
        return result

    finally:
        # 8) 清理中间 wav 与临时 m4a（默认保留成品；--keep-intermediates 时保留 wav）
        try:
            tmp_m4a.unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            pass
        shutil.rmtree(measure_tmp, ignore_errors=True)
        if not keep_intermediates:
            for key in ("denoised", "clean44", "norm44"):
                try:
                    caches[key].unlink(missing_ok=True)
                except OSError:  # pragma: no cover
                    pass
            # 44.1k 解码缓存（渲染轴，体积大）与容器修复副本也删；
            # **保留** 16k wav、decode 元数据与 analyze 缓存（P3 明确要求别弄丢 analyze 缓存）
            try:
                decoded_paths = decode_cache_paths(cache_root, job_id, source_p)
                for key in ("wav44k", "fixed"):
                    try:
                        decoded_paths[key].unlink(missing_ok=True)
                    except OSError:  # pragma: no cover
                        pass
            except Exception:  # pragma: no cover - 路径解析失败不影响结果
                log.debug("清理 44.1k 解码缓存失败：%s", job_id, exc_info=True)
        else:
            log.info("保留中间文件：%s", ", ".join(str(caches[k]) for k in ("denoised", "clean44", "norm44")))


# ================================================================ 文本导出（P5-T4）
#
# ``transcript.json``（P4 产物，契约 §4.3，顶层是 **list**）→ ``transcript.md`` / ``.srt`` / ``.txt``。
# 时间轴约定（任务卡 P5-T4）：
# - md / txt 的 ``[mm:ss]`` 与 srt 的 ``HH:MM:SS,mmm`` 都取 **clean 轴**的 ``start``（向下取整到秒），
#   所以同一段的 md 行首时间与 srt 起始时间逐条一致；md 括号内才是 orig 轴（``orig_start``）。
# - md **逐段一行、不合并段落**（PoC 的 md 是合并过的，会把多段并到一行，导致条数与 srt 不一致）。
# - 空文本 / 纯标点段不写入三个文本产物（但仍计在 n_segments 里），srt 序号在过滤后连续。
# - 全部文件 UTF-8（无 BOM）/ LF；写入用临时文件 + 原子替换，与 ``write_json`` 一致。

#: 文本产物文件名（P5-T4 契约）
PRODUCT_TRANSCRIPT = "transcript.json"
PRODUCT_MD = "transcript.md"
PRODUCT_SRT = "transcript.srt"
PRODUCT_TXT = "transcript.txt"

#: export_text 默认导出的文本格式（transcript.json 由 P4 落盘，不重复写）
TEXT_FORMATS: Tuple[str, ...] = ("md", "srt", "txt")
_FORMAT_FILES = {"md": PRODUCT_MD, "srt": PRODUCT_SRT, "txt": PRODUCT_TXT}

#: SRT 单条最短时长兜底（start == end 时用 ``start + SRT_MIN_DURATION_S``）
SRT_MIN_DURATION_S = 0.3

#: 「有内容」判定：汉字/字母/数字；纯标点、空白、表情段跳过（与 asr.count_speech_chars 同口径）
_SPEECH_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def _as_seconds(value: Any, default: float = 0.0) -> float:
    """宽松转 float：None/非法/NaN/Inf 均返回 ``default``（导出不能因为脏时间戳崩掉）。"""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return default
    return seconds if math.isfinite(seconds) else default


def format_mmss(seconds: Any) -> str:
    """秒 → ``mm:ss``；**≥1 h 用 ``h:mm:ss``**（小时不补零，如 ``1:02:05``）。

    选择 ``h:mm:ss`` 而不是让分钟继续累加（``62:05``）：2 h 以上的长讲座
    （f04 clean 8131.8 s ≈ 2:15:32）更易读，也与 srt 的 ``HH:MM:SS`` 便于目视对照。
    向下取整到秒（63.9 → 63）；负数 / NaN / Inf 记 0。
    """
    total = int(max(0.0, _as_seconds(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_srt_time(seconds: Any) -> str:
    """秒 → SRT 时间戳 ``HH:MM:SS,mmm``（四舍五入到毫秒；负数 / NaN / Inf 记 0）。"""
    total_ms = int(round(max(0.0, _as_seconds(seconds)) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segment_field(seg: Any, key: str, default: Any = None) -> Any:
    """兼容 transcript.json 的 dict 条目与 ``TranscriptSegment`` dataclass。"""
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _has_speech_text(text: str) -> bool:
    return bool(_SPEECH_CHAR_RE.search(text or ""))


def _normalize_segments(segments: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """把 transcript 条目规整成导出用 dict，并跳过空文本 / 纯标点段。

    返回按输入顺序的列表，元素：``{start, end, orig_start, speaker, text}``。
    ``text`` 会压掉换行/多余空白，保证 md/txt 一段一行、srt 一条一块。
    """
    rows: List[Dict[str, Any]] = []
    for seg in segments or []:
        text = " ".join(str(_segment_field(seg, "text", "") or "").split())
        if not _has_speech_text(text):
            continue
        start = max(0.0, _as_seconds(_segment_field(seg, "start", 0.0)))
        end = max(0.0, _as_seconds(_segment_field(seg, "end", start), start))
        orig_start = _segment_field(seg, "orig_start", None)
        orig_start = start if orig_start is None else max(0.0, _as_seconds(orig_start, start))
        speaker = str(_segment_field(seg, "speaker", "") or "").strip() or SPEAKER_MAIN
        rows.append({
            "start": start,
            "end": end,
            "orig_start": orig_start,
            "speaker": speaker,
            "text": text,
        })
    return rows


def _resolve_title(title: Optional[str], source: Optional[str]) -> str:
    """md 标题：显式 title > 来源文件名的 stem > 兜底「转写」。"""
    if title and str(title).strip():
        return str(title).strip()
    if source:
        stem = Path(str(source)).stem
        if stem:
            return stem
    return "转写"


def transcript_to_md(
    segments: Optional[Sequence[Any]],
    *,
    source: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """transcript 段 → Markdown 文本：逐段一行 ``[mm:ss (orig mm:ss)] 发言人：text``。

    - clean 轴时间取自 ``start``，括号内 orig 轴取自 ``orig_start``；
    - **不合并段落**（一段一行），条数与 srt 逐条对应；
    - 文件头两行：``# <title> · 句级时间戳转写`` 与 ``> 时间轴：clean 音轨；括号内为原始音轨时间。``。
    """
    rows = _normalize_segments(segments)
    lines = [
        f"# {_resolve_title(title, source)} · 句级时间戳转写",
        "",
        "> 时间轴：clean 音轨；括号内为原始音轨时间。",
    ]
    body = [
        f"[{format_mmss(row['start'])} (orig {format_mmss(row['orig_start'])})] "
        f"{row['speaker']}：{row['text']}"
        for row in rows
    ]
    if body:
        lines.extend([""] + body)
    return "\n".join(lines) + "\n"


def transcript_to_srt(segments: Optional[Sequence[Any]]) -> str:
    """transcript 段 → SRT（clean 轴）：序号从 1，``HH:MM:SS,mmm --> HH:MM:SS,mmm``，正文 ``发言人：text``。

    ``end`` 用 ``max(end, start + 0.3)`` 兜底，保证 ``end > start``；块间空行；末尾单个换行。
    """
    rows = _normalize_segments(segments)
    blocks: List[str] = []
    for index, row in enumerate(rows, start=1):
        start = row["start"]
        end = max(row["end"], start + SRT_MIN_DURATION_S)
        blocks.append(
            f"{index}\n{format_srt_time(start)} --> {format_srt_time(end)}\n"
            f"{row['speaker']}：{row['text']}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def transcript_to_txt(segments: Optional[Sequence[Any]]) -> str:
    """transcript 段 → 纯文本：逐段一行 ``[mm:ss] 发言人：text``（clean 轴，无 orig 括号）。"""
    rows = _normalize_segments(segments)
    return "".join(
        f"[{format_mmss(row['start'])}] {row['speaker']}：{row['text']}\n" for row in rows
    )


def _normalize_formats(formats: Optional[Sequence[str] | str]) -> List[str]:
    """格式列表规整：去空白/点号/小写、保序去重。接受 ``"md,srt"`` 字符串。"""
    if isinstance(formats, str):
        raw: Sequence[str] = formats.split(",")
    else:
        raw = list(formats or ())
    out: List[str] = []
    for item in raw:
        name = str(item or "").strip().lower().lstrip(".")
        if name and name not in out:
            out.append(name)
    return out


def _read_json_dict(path: Path) -> Dict[str, Any]:
    """读 JSON 对象；文件缺失 / 非法 / 顶层不是 dict 时返回空 dict（导出不因此失败）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_text_atomic(path: Path, text: str) -> Path:
    """UTF-8（无 BOM）/ LF 原子写入：先写 ``*.tmp`` 再 replace。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    tmp.replace(target)
    return target


def _srt_monotonic_error(rows: Sequence[Dict[str, Any]]) -> str:
    """校验 srt 时间单调：段 ``start`` 必须非递减（end 由 max(end, start+0.3) 兜底恒 > start）。"""
    previous = -1.0
    for index, row in enumerate(rows):
        if row["start"] + 1e-6 < previous:
            return (
                f"transcript 段未按 start 单调排序（第 {index + 1} 段 start={row['start']:.3f} "
                f"< 上一段 {previous:.3f}）"
            )
        previous = max(previous, row["start"])
    return ""


def export_text(
    job_id: str,
    cfg: Any,
    *,
    out_dir: Optional[str | Path] = None,
    formats: Optional[Sequence[str] | str] = None,
    force: bool = False,
    on_stage: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """``out/<id>/transcript.json`` → ``transcript.md / .srt / .txt``，并回写 report.json。

    行为
    ----
    - 只读 ``out/<id>/transcript.json``（契约 §4.3，顶层 list）；不碰 clean.m4a / edit.json；
    - 默认 ``formats=["md","srt","txt"]``；``json`` 视为 P4 已落盘，不重复写；
    - 空文本 / 纯标点段跳过，不写入文本产物；
    - report.json **只读-改-写**：合并 ``output.formats``（保留 codec/size 等既有键）与
      ``timing.export_text_s``，其余字段原样保留（report 不存在时创建最小 dict）；
    - 目标文件全部存在且 ``force=False`` → 跳过（``skipped=True``）；
    - 失败不抛异常：返回 ``{"ok": False, "stage": "export", "error": ...}``，便于批处理；
    - ``on_stage("export", "done"/"failed", metrics)`` 供编排器落 stages 表。

    返回：``{"ok", "job_id", "out_dir", "skipped", "stage", "error", "formats", "files",
    "n_segments", "n_written", "n_skipped_empty", "elapsed_s", "timings", "report"}``。
    """
    started = time.perf_counter()
    job_id = str(job_id)
    job_out = Path(out_dir) / job_id if out_dir else None
    if job_out is None:
        if cfg is None:
            job_out = Path(".") / job_id
        else:
            job_out = cfg.path("out_dir") / job_id
    fmt_list = _normalize_formats(formats if formats is not None else TEXT_FORMATS)
    writable = [name for name in fmt_list if name in _FORMAT_FILES]

    def notify(stage_name: str, status: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        if on_stage is not None:
            try:
                on_stage(stage_name, status, metrics or {})
            except Exception:  # pragma: no cover - 回调失败不影响主流程
                log.debug("on_stage 回调失败", exc_info=True)

    def fail(message: str, n_segments: int = 0) -> Dict[str, Any]:
        elapsed = round(time.perf_counter() - started, 2)
        log.warning("文本导出失败：%s：%s", job_id, message)
        notify("export", "failed", {"error": message, "elapsed_s": elapsed})
        return {
            "ok": False, "job_id": job_id, "out_dir": str(job_out), "skipped": False,
            "stage": "export", "error": message, "formats": fmt_list, "files": {},
            "n_segments": n_segments, "n_written": 0, "n_skipped_empty": 0,
            "elapsed_s": elapsed, "timings": {"export_text_s": elapsed}, "report": None,
        }

    if not fmt_list:
        return fail("formats 为空，至少指定 md/srt/txt 之一")
    if not writable:
        # CLI 允许 ``--formats json``：json 就是 P4 已落盘的 transcript.json，无需再写文本产物
        if "json" in fmt_list:
            log.info("仅请求 json（transcript.json 已由 P4 落盘），无需文本导出：%s", job_id)
            notify("export", "done", {"skipped": True, "formats": fmt_list})
            return {
                "ok": True, "job_id": job_id, "out_dir": str(job_out), "skipped": True,
                "stage": "export", "error": "", "formats": fmt_list, "files": {},
                "n_segments": 0, "n_written": 0, "n_skipped_empty": 0,
                "elapsed_s": 0.0, "timings": {"export_text_s": 0.0}, "report": None,
            }
        return fail(f"formats={fmt_list} 中没有可导出的文本格式（支持 md/srt/txt；json 由 P4 落盘）")
    if not force:
        targets = [job_out / _FORMAT_FILES[name] for name in writable]
        if all(path.is_file() for path in targets):
            log.info("文本产物已存在，跳过：%s（formats=%s）", job_out, ",".join(writable))
            notify("export", "done", {"skipped": True, "formats": writable})
            return {
                "ok": True, "job_id": job_id, "out_dir": str(job_out), "skipped": True,
                "stage": "export", "error": "", "formats": fmt_list,
                "files": {name: str(job_out / _FORMAT_FILES[name]) for name in writable},
                "n_segments": 0, "n_written": 0, "n_skipped_empty": 0,
                "elapsed_s": 0.0, "timings": {"export_text_s": 0.0}, "report": None,
            }

    transcript_path = job_out / PRODUCT_TRANSCRIPT
    if not transcript_path.is_file():
        return fail(f"缺少 transcript.json：{transcript_path}")
    try:
        raw = json.loads(transcript_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return fail(f"transcript.json 读取/解析失败：{transcript_path}：{exc}")
    if not isinstance(raw, list):
        return fail(
            f"transcript.json 顶层必须是 list（契约 §4.3），实际是 {type(raw).__name__}：{transcript_path}"
        )

    rows = _normalize_segments(raw)
    error = _srt_monotonic_error(rows)
    if error:
        return fail(error, n_segments=len(raw))

    report_path = job_out / PRODUCT_REPORT
    report = _read_json_dict(report_path)
    source = report.get("source")
    title = report.get("title")
    contents: Dict[str, str] = {}
    if "md" in writable:
        contents["md"] = transcript_to_md(rows, source=source if isinstance(source, str) else None,
                                          title=title if isinstance(title, str) else None)
    if "srt" in writable:
        contents["srt"] = transcript_to_srt(rows)
    if "txt" in writable:
        contents["txt"] = transcript_to_txt(rows)

    files: Dict[str, str] = {}
    t_write = time.perf_counter()
    try:
        job_out.mkdir(parents=True, exist_ok=True)
        for name in writable:
            path = _write_text_atomic(job_out / _FORMAT_FILES[name], contents[name])
            files[name] = str(path)
    except OSError as exc:
        return fail(f"写入文本产物失败：{job_out}：{exc}", n_segments=len(raw))
    elapsed = time.perf_counter() - started
    write_elapsed = time.perf_counter() - t_write

    # report.json 只读-改-写：output.formats（合并）+ timing.export_text_s（保留其它 timing）
    output = report.get("output") if isinstance(report.get("output"), dict) else {}
    output["formats"] = fmt_list
    report["output"] = output
    timing = report.get("timing") if isinstance(report.get("timing"), dict) else {}
    timing["export_text_s"] = round(elapsed, 2)
    report["timing"] = timing
    report.setdefault("id", job_id)
    write_json(report_path, report)

    n_total = len(raw)
    n_written = len(rows)
    metrics = {
        "formats": fmt_list, "n_segments": n_total, "n_written": n_written,
        "n_skipped_empty": n_total - n_written,
        "bytes": {name: (job_out / _FORMAT_FILES[name]).stat().st_size for name in writable},
        "elapsed_s": round(elapsed, 2), "write_s": round(write_elapsed, 2),
    }
    notify("export", "done", metrics)
    log.info(
        "文本导出完成：%s formats=%s 段=%d（写入 %d，空/纯标点跳过 %d）耗时 %.2fs",
        job_id, ",".join(fmt_list), n_total, n_written, n_total - n_written, elapsed,
    )
    return {
        "ok": True, "job_id": job_id, "out_dir": str(job_out), "skipped": False,
        "stage": "export", "error": "", "formats": fmt_list, "files": files,
        "n_segments": n_total, "n_written": n_written,
        "n_skipped_empty": n_total - n_written,
        "elapsed_s": round(elapsed, 2), "timings": {"export_text_s": round(elapsed, 2)},
        "report": report,
    }
