# -*- coding: utf-8 -*-
"""时间戳精度测量（P4-T4）：句级边界 vs faster-whisper 词级时间戳。

口径（复现 ``work/timeline/scripts/step_eval.py``）
---------------------------------------------------
1. 取仓库产物同一 **clean 16k** 的 ``[start, start+dur)`` 窗口；
2. 用 faster-whisper **base**（CPU int8、``beam_size=1``、``word_timestamps=True``、
   ``language=zh``、``HF_HUB_OFFLINE=1``、模型缓存 ``work/bench/models/whisper``）解出词级
   start/end —— **只作参照，不写任何转写产物**；
3. 对窗口内每个 ``transcript.json`` 段，取「**词中点落在段内**」的首词 start / 末词 end
   作为参照边界，统计 ``我们 - 参照`` 的 signed median、abs median、P90 abs、n；
4. 输出 n_segments / chars 对照，以及 PoC 基线（600–1200 s）量级对照与验收结论。

坐标轴
------
``transcript.json`` 的 ``start/end`` 与 ``clean16k.wav`` 都是 **clean 轴**；本工具默认窗口
``--start 600 --dur 600`` 也指 clean 轴。PoC 基线（start abs med 0.21 / P90 0.654，
end 0.355 / 0.93）是在**原始音轨** 600–1200 s 上测的（见 docs/research/timeline-poc.md §3），
两者只作量级对照，不要求数字完全一致。

用法::

    work\\venv\\Scripts\\python app\\tools\\measure.py --job <id> [--start 600 --dur 600] [--json]
    work\\venv\\Scripts\\python app\\tools\\measure.py --all --json --out work/measure_all.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 支持任务卡里的直接运行方式：``python app/tools/measure.py ...``（PEP 366 补包上下文）。
# 用 ``python -m app.tools.measure`` 运行时 __package__ 已就绪，不会进这里。
if __package__ in (None, ""):  # pragma: no cover - 入口兼容
    import sys as _sys
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parents[2]
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))
    __package__ = "app.tools"

from ..config import PROJECT_ROOT, load_config
from ..logging_setup import get_logger, setup_logging

log = get_logger(__name__)

#: 分析/参照统一采样率
SR = 16000

#: 段边界判定时距窗口边缘的排除带（秒）——与 step_eval.py 的 ``win_start + 1.0`` 一致
EDGE_MARGIN_S = 1.0

#: 有效字符口径（汉字/字母/数字，不含标点；与 asr.count_speech_chars 一致）
_SPEECH_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

#: PoC 基线（docs/research/timeline-poc.md §3 / eval_ts_600_1200.json，原始音轨 600–1200 s）
#: 同时给出两次独立复跑里较优的「absolute」口径数字（见 timeline-poc §3 的复核说明）
POC_BASELINE: Dict[str, Any] = {
    "source": "docs/research/timeline-poc.md §3 / work/timeline/24_形而上学/eval_ts_600_1200.json",
    "axis": "original",
    "window_s": [600.0, 1200.0],
    "start_abs_median": 0.21,
    "start_p90_abs": 0.654,
    "end_abs_median": 0.355,
    "end_p90_abs": 0.93,
    "note": "PoC 在原始音轨 600–1200s 测；本工具窗口是 clean 轴，仅作量级对照",
}

#: 验收线（docs/实施计划.md P4-T2 / 工程日志 P4 开工）
ACCEPT_MEDIAN_S = 0.5
ACCEPT_P90_S = 1.5


# ---------------------------------------------------------------- 文本/统计工具
def count_chars(text: Optional[str]) -> int:
    """有效字符数（汉字/字母/数字；与 ``asr.count_speech_chars`` 同口径）。"""
    return len(_SPEECH_CHAR_RE.findall(text or ""))


def stat(values: Sequence[float]) -> Dict[str, Any]:
    """误差统计：n / signed median / abs median / mean / std / P90 abs / max abs。"""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 3),
        "abs_median": round(float(np.median(abs_arr)), 3),
        "mean": round(float(np.mean(arr)), 3),
        "std": round(float(np.std(arr)), 3),
        "p90_abs": round(float(np.percentile(abs_arr, 90)), 3),
        "max_abs": round(float(np.max(abs_arr)), 3),
    }


# ---------------------------------------------------------------- 纯函数：边界匹配
def match_boundaries(
    segments: Sequence[Dict[str, Any]],
    words: Sequence[Dict[str, Any]],
    *,
    win_start: float,
    win_end: float,
    margin_s: float = EDGE_MARGIN_S,
) -> Dict[str, Any]:
    """窗口内每个 transcript 段 → 参照词边界，返回误差统计。

    参数
    ----
    segments : ``transcript.json`` 段（``start/end/text``，clean 轴绝对秒）
    words    : 参照词（``start/end/word``，clean 轴绝对秒），需覆盖 ``[win_start, win_end]``
    win_start/win_end : 测量窗口（绝对秒）
    margin_s : 完全落在窗口边缘 ``margin_s`` 内的段不再参与（避免截断段误判）

    匹配口径：``seg.start <= (w.start+w.end)/2 <= seg.end`` 的词都算段内；
    参照起点 = 段内首词 start，参照终点 = 段内末词 end；``diff = seg - ref``。
    """
    words_sorted = sorted((w for w in words if str(w.get("word", "")).strip()),
                          key=lambda w: float(w["start"]))
    segs_in_window = 0
    matched = 0
    ds: List[float] = []
    de: List[float] = []
    chars_transcript = 0
    chars_words = 0
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end <= start:
            continue
        if start < win_start + float(margin_s) or end > win_end - float(margin_s):
            continue
        segs_in_window += 1
        inside = [w for w in words_sorted
                  if start <= (float(w["start"]) + float(w["end"])) / 2.0 <= end]
        if not inside:
            continue
        ref_start = min(float(w["start"]) for w in inside)
        ref_end = max(float(w["end"]) for w in inside)
        ds.append(start - ref_start)
        de.append(end - ref_end)
        chars_transcript += count_chars(seg.get("text") or seg.get("text_raw") or "")
        chars_words += sum(count_chars(w.get("word")) for w in inside)
        matched += 1
    start_abs = stat([abs(v) for v in ds])
    end_abs = stat([abs(v) for v in de])
    return {
        "n_segments_in_window": segs_in_window,
        "n_matched": matched,
        "start_signed": stat(ds),
        "end_signed": stat(de),
        "start_abs": start_abs,
        "end_abs": end_abs,
        "chars_transcript": int(chars_transcript),
        "chars_words": int(chars_words),
    }


def resolve_window(
    audio_duration_s: float, start: float = 600.0, dur: float = 600.0
) -> Tuple[float, float, bool]:
    """把 ``[start, start+dur)`` 裁到 ``[0, audio_duration_s]``，返回 ``(a, b, clamped)``。"""
    total = float(audio_duration_s)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("音频时长为空，无法测量")
    a = max(0.0, float(start))
    b = a + max(0.0, float(dur))
    if b > total:
        b = total
    if a >= total:
        raise ValueError(f"窗口起点 {a:.1f}s 超出音频时长 {total:.1f}s")
    return round(a, 3), round(b, 3), bool(a != float(start) or b != float(start) + float(dur))


# ---------------------------------------------------------------- whisper 参照
def transcribe_words(
    audio: np.ndarray,
    *,
    model_name: str = "base",
    download_root: Optional[str | Path] = None,
    threads: int = 8,
    beam_size: int = 1,
    language: str = "zh",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """跑 faster-whisper 词级时间戳（相对 ``audio`` 起点），返回 ``(words, run_info)``。

    时间戳**相对传入数组起点**；调用方负责加窗口偏移。只做测量，不写产物。
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from faster_whisper import WhisperModel

    x = np.ascontiguousarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    t0 = time.perf_counter()
    model = WhisperModel(
        str(model_name), device="cpu", compute_type="int8",
        download_root=str(download_root) if download_root else None,
        cpu_threads=max(1, int(threads)), num_workers=1,
    )
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    segments, _info = model.transcribe(
        x, language=language, beam_size=max(1, int(beam_size)),
        vad_filter=False, condition_on_previous_text=False, word_timestamps=True,
    )
    words: List[Dict[str, Any]] = []
    for seg in segments:
        for word in (seg.words or []):
            text = (word.word or "").strip()
            if not text:
                continue
            words.append({"start": round(float(word.start), 3),
                          "end": round(float(word.end), 3), "word": text})
    wall_s = time.perf_counter() - t0
    run_info = {
        "model": str(model_name), "device": "cpu", "compute_type": "int8",
        "beam_size": int(beam_size), "language": language, "word_timestamps": True,
        "vad_filter": False, "threads": max(1, int(threads)),
        "load_s": round(load_s, 2), "wall_s": round(wall_s, 2), "n_words": len(words),
        "rtf": round(wall_s / max(len(x) / float(SR), 1e-6), 4),
    }
    log.info("faster-whisper 参照：%s %d 词（load %.1fs / wall %.1fs）",
             model_name, len(words), load_s, wall_s)
    return words, run_info


# ---------------------------------------------------------------- 音频/产物定位
def _load_wav16k_audio(path: str | Path) -> np.ndarray:
    """读 16k wav 为 float32 单声道（复用 diarize 的读取/兼容逻辑）。"""
    from ..pipeline.diarize import load_wav16k

    return load_wav16k(path)


def _ensure_clean16k(
    job_id: str,
    cfg: Any,
    *,
    out_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """clean16k 缓存（缺时 ffmpeg 解 ``clean.m4a``），复用 ``diarize.ensure_clean16k``。"""
    from ..pipeline.diarize import ensure_clean16k

    return ensure_clean16k(job_id, cfg, out_dir=out_dir, cache_dir=cache_dir)


def _read_transcript(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("segments"), list):  # 兼容 PoC 旧格式
        data = data["segments"]
    if not isinstance(data, list):
        raise ValueError(f"transcript.json 顶层应为 list（契约 §4.3）：{path}")
    return [seg for seg in data if isinstance(seg, dict)]


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def collect_jobs(out_root: str | Path) -> List[str]:
    """列出 ``out/<id>/transcript.json`` 存在的 job id（按名字排序）。"""
    root = Path(out_root)
    if not root.is_dir():
        return []
    return sorted(p.parent.name for p in root.glob("*/transcript.json") if p.is_file())


# ---------------------------------------------------------------- 主测量
def measure_job(
    job_id: str,
    cfg: Any,
    *,
    start: float = 600.0,
    dur: float = 600.0,
    out_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    transcript_path: Optional[str | Path] = None,
    word_provider: Optional[Callable[[np.ndarray, float], List[Dict[str, Any]]]] = None,
    model_name: str = "base",
    threads: int = 8,
    beam_size: int = 1,
) -> Dict[str, Any]:
    """测量一个 job 的时间戳精度；失败返回 ``{"ok": False, "job_id", "error"}``，不抛。

    ``word_provider`` 仅供测试注入（``(window_audio, win_start) -> words(绝对时间)``），
    默认用 faster-whisper；生产/真实测量不要传。
    """
    started = time.perf_counter()
    out_root = Path(out_dir) if out_dir else cfg.path("out_dir")
    result: Dict[str, Any] = {"ok": False, "job_id": job_id, "window": None,
                              "method": None, "boundaries": None, "error": ""}
    try:
        tpath = Path(transcript_path) if transcript_path else out_root / job_id / "transcript.json"
        if not tpath.is_file():
            raise FileNotFoundError(f"transcript.json 不存在：{tpath}（先跑转写阶段）")
        segments = _read_transcript(tpath)

        wav_path = _ensure_clean16k(job_id, cfg, out_dir=out_root, cache_dir=cache_dir)
        x = _load_wav16k_audio(wav_path)
        audio_dur = len(x) / float(SR)
        win_start, win_end, clamped = resolve_window(audio_dur, start=start, dur=dur)
        a = int(round(win_start * SR))
        b = min(len(x), int(round(win_end * SR)))
        window = np.ascontiguousarray(x[a:b], dtype=np.float32)
        win_start, win_end = a / float(SR), b / float(SR)
        if window.size < int(1.0 * SR):
            raise ValueError(f"测量窗口过短：{win_end - win_start:.2f}s")

        if word_provider is None:
            words, run_info = transcribe_words(
                window, model_name=model_name,
                download_root=cfg.model_path("whisper_cache"),
                threads=threads, beam_size=beam_size,
            )
            words_abs = [{"start": round(w["start"] + win_start, 3),
                          "end": round(w["end"] + win_start, 3), "word": w["word"]}
                         for w in words]
            method_name = "faster_whisper_word_boundary"
        else:
            start_ts = time.perf_counter()
            words_abs = word_provider(window, win_start)
            run_info = {"provider": "injected", "n_words": len(words_abs),
                        "wall_s": round(time.perf_counter() - start_ts, 3)}
            method_name = "injected_word_boundary"

        boundaries = match_boundaries(segments, words_abs,
                                      win_start=win_start, win_end=win_end)
        start_abs = boundaries["start_abs"]
        end_abs = boundaries["end_abs"]
        acceptance = _acceptance(start_abs, end_abs)
        chars_tr = int(boundaries["chars_transcript"])
        chars_wr = int(boundaries["chars_words"])
        payload: Dict[str, Any] = {
            "ok": True,
            "job_id": job_id,
            "transcript": _rel(tpath),
            "wav16k": _rel(wav_path),
            "audio_duration_s": round(audio_dur, 3),
            "window": {"start": round(win_start, 3), "end": round(win_end, 3),
                       "dur": round(win_end - win_start, 3), "axis": "clean",
                       "requested": {"start": float(start), "dur": float(dur)},
                       "clamped": bool(clamped)},
            "method": {
                "name": method_name,
                "reference": "faster-whisper base word timestamps",
                "boundary_rule": "ref_start=min(word.start in seg), ref_end=max(word.end in seg)",
                "match_rule": "word midpoint (start+end)/2 within [seg.start, seg.end]",
                "edge_margin_s": EDGE_MARGIN_S,
                "window_axis": "clean",
                "ref_model": model_name if word_provider is None else "injected",
                "beam_size": int(beam_size), "language": "zh",
                "word_timestamps": True, "vad_filter": False,
            },
            "reference": run_info,
            "n_transcript_segments": len(segments),
            "n_segments_in_window": int(boundaries["n_segments_in_window"]),
            "n_matched": int(boundaries["n_matched"]),
            "boundaries": boundaries,
            "summary": {
                "n": int(boundaries["n_matched"]),
                "start_abs_median": start_abs.get("abs_median"),
                "start_p90_abs": start_abs.get("p90_abs"),
                "start_signed_median": boundaries["start_signed"].get("median"),
                "end_abs_median": end_abs.get("abs_median"),
                "end_p90_abs": end_abs.get("p90_abs"),
                "end_signed_median": boundaries["end_signed"].get("median"),
            },
            "chars": {
                "transcript": chars_tr,
                "reference_words": chars_wr,
                "ratio_reference_over_transcript": (round(chars_wr / chars_tr, 4)
                                                    if chars_tr else None),
            },
            "acceptance": acceptance,
            "baseline": dict(POC_BASELINE),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
        result = payload
        log.info("measure %s：window=%.1f-%.1fs n=%d start abs med=%s P90=%s | "
                 "end abs med=%s P90=%s → %s",
                 job_id, win_start, win_end, boundaries["n_matched"],
                 start_abs.get("abs_median"), start_abs.get("p90_abs"),
                 end_abs.get("abs_median"), end_abs.get("p90_abs"),
                 "PASS" if acceptance["pass"] else "FAIL")
        return result
    except Exception as exc:
        msg = str(exc) or exc.__class__.__name__
        log.error("measure 失败（%s）：%s", job_id, msg)
        result.update({"ok": False, "error": msg,
                       "elapsed_s": round(time.perf_counter() - started, 2)})
        return result


def _acceptance(start_abs: Dict[str, Any], end_abs: Dict[str, Any]) -> Dict[str, Any]:
    """按 P4-T2 验收线（abs median ≤0.5 s、P90 ≤1.5 s）给出结论。"""
    checks = {
        "start_abs_median_le_0.5": _le(start_abs.get("abs_median"), ACCEPT_MEDIAN_S),
        "start_p90_abs_le_1.5": _le(start_abs.get("p90_abs"), ACCEPT_P90_S),
        "end_abs_median_le_0.5": _le(end_abs.get("abs_median"), ACCEPT_MEDIAN_S),
        "end_p90_abs_le_1.5": _le(end_abs.get("p90_abs"), ACCEPT_P90_S),
    }
    checks["pass"] = all(bool(v) for k, v in checks.items() if k != "pass")
    checks["thresholds"] = {"abs_median_s": ACCEPT_MEDIAN_S, "p90_abs_s": ACCEPT_P90_S}
    return checks


def _le(value: Any, limit: float) -> Optional[bool]:
    if value is None:
        return None
    try:
        return bool(float(value) <= float(limit))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python app/tools/measure.py",
        description="句级时间戳精度测量（transcript clean 轴边界 vs faster-whisper 词级参照）",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", help="out/<id> 的 job id")
    group.add_argument("--all", action="store_true", help="测量 out/ 下所有有 transcript 的 job")
    ap.add_argument("--start", type=float, default=600.0, help="窗口起点（clean 轴秒，默认 600）")
    ap.add_argument("--dur", type=float, default=600.0, help="窗口时长（秒，默认 600）")
    ap.add_argument("--json", action="store_true", help="输出 JSON（默认人读摘要）")
    ap.add_argument("--out", default="", help="把 JSON 结果写到该路径（UTF-8，原子替换）")
    ap.add_argument("--transcript", default="", help="覆盖 transcript.json 路径（调试/对照用）")
    ap.add_argument("--out-dir", default="", help="覆盖 paths.out_dir")
    ap.add_argument("--model", default="base", help="faster-whisper 模型（默认 base）")
    ap.add_argument("--threads", type=int, default=8, help="faster-whisper CPU 线程数")
    ap.add_argument("--beam", type=int, default=1, help="beam_size（口径基线=1）")
    ap.add_argument("--log-level", default="INFO", help="日志级别")
    return ap


def _write_json(path: str | Path, payload: Any) -> Path:
    """UTF-8、无 NaN、原子替换。"""
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    setup_logging(cfg=cfg, level=str(args.log_level).upper(), console=True)
    if args.out_dir:
        cfg = cfg.replace({"paths.out_dir": str(args.out_dir)})
    out_root = cfg.path("out_dir")

    if args.all:
        job_ids = collect_jobs(out_root)
        if not job_ids:
            print(f"[警告] {out_root} 下没有 transcript.json", file=sys.stderr)
            payload: Dict[str, Any] = {"ok": True, "n": 0, "results": []}
        else:
            results = [measure_job(job_id, cfg, start=args.start, dur=args.dur,
                                   transcript_path=(args.transcript or None),
                                   model_name=args.model, threads=args.threads,
                                   beam_size=args.beam)
                       for job_id in job_ids]
            payload = {"ok": all(r.get("ok") for r in results), "n": len(results),
                       "results": results}
    else:
        payload = measure_job(str(args.job), cfg, start=args.start, dur=args.dur,
                              transcript_path=(args.transcript or None),
                              model_name=args.model, threads=args.threads,
                              beam_size=args.beam)

    if args.out:
        _write_json(args.out, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        rows = payload.get("results") if args.all else [payload]
        for row in rows or []:
            if not row.get("ok"):
                print(f"[FAIL] {row.get('job_id')}: {row.get('error')}")
                continue
            win = row["window"]
            s = row["summary"]
            print(f"[OK] {row['job_id']} window={win['start']:.1f}-{win['end']:.1f}s "
                  f"n_seg={row['n_segments_in_window']} n={row['n_matched']}")
            print(f"     start abs med={s['start_abs_median']}s P90={s['start_p90_abs']}s "
                  f"| end abs med={s['end_abs_median']}s P90={s['end_p90_abs']}s")
            print(f"     chars={row['chars']['transcript']} vs words={row['chars']['reference_words']} "
                  f"| 基线(orig 600–1200) start {POC_BASELINE['start_abs_median']}/"
                  f"{POC_BASELINE['start_p90_abs']} end {POC_BASELINE['end_abs_median']}/"
                  f"{POC_BASELINE['end_p90_abs']} | "
                  f"{'PASS' if row['acceptance']['pass'] else 'FAIL'}")
    ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
