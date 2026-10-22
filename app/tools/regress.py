# -*- coding: utf-8 -*-
"""六文件回归工具：P3 渲染回归（audio.csv）+ P8 产物回归（reuse / rerun）。

P8-T1 默认模式（``--reuse``，只读真实产物，**不跑流水线**）::

    work\\venv\\Scripts\\python -m app.tools.regress --reuse --json
    # 读 out/<id>/report.json + 8 件产物 → 断言 → work/regress/p8.json（6 行 + summary）

P8-T1 真跑模式（``--rerun``，独立目录，**绝不写 out/**）::

    work\\venv\\Scripts\\python -m app.tools.regress --rerun --limit 2 --json
    # 默认 --out out_p8 --cache-dir work/cache_p8；6 文件约 80 min（Lead 手册验收）

P3 旧模式（渲染回归，行为保留；只有显式 ``--force`` / ``--from-reports`` 才走）::

    work\\venv\\Scripts\\python -m app.tools.regress --force --limit 1
    work\\venv\\Scripts\\python -m app.tools.regress --from-reports

固定接口（P8-T1 契约，CLI 与测试共用）::

    run_regression(data_dir, cfg, *, mode="reuse"|"rerun", out_dir=None,
                   cache_dir=None, limit=None) -> dict
        {"ok", "mode", "n", "rows":[...], "summary":{...}}
    check_regression(result, *, baseline_csv=None, scan_dir="work/scan") -> dict
        {"ok", "failures":[...], "summary":{...}}

``check_regression`` 的 7 条断言（与 docs/开发启动Prompt_P8.md §5 P8-T1 对齐）:
1. 跨文件 out ASL 极差 ≤ :data:`ASL_SPREAD_MAX_DB`；
2. 全部 ebur128 真峰值 ≤ :data:`TP_MAX_DBTP` dBTP；
3. 削波样本合计 = 0；
4. 单文件 render 全链耗时（``report.timing.total_s``）≤ :data:`MAX_TOTAL_S` s；
5. 与 ``work/regress/audio.csv`` 逐文件比对：cut_ratio 偏差 ≤5 个百分点、
   ASL ≤0.5 dB、TP ≤1.0 dB（``baseline_csv=None`` 时跳过并在 summary 标注）；
6. 时间戳读 ``<scan_dir 上级>/measure_f24.json``：段起点 abs_median ≤0.5 s 且
   p90_abs ≤1.5 s（文件/字段缺失 → failure「时间戳基线缺失」）；
7. ASR 段数/字数与 :data:`P4_BASELINE` 偏差 ≤10%（基线缺该 job 跳过）。

另：reuse 模式任一 job 的 8 件产物缺失 → 该行 ``ok=False`` + ``error``，check 记 failure。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config, ConfigError, PROJECT_ROOT, load_config
from ..logging_setup import get_logger, setup_logging
from ..models import make_job_id
from ..pipeline.export import render_job
from ..pipeline.orchestrator import PRODUCT_FILES, batch_run
from ..storage import Storage

log = get_logger(__name__)

# ---------------------------------------------------------------- 常量（P3/P8 验收线）
#: 跨文件验收阈值（与 docs/实施计划.md P3-T5 / P8-T1 对齐）
ASL_SPREAD_MAX_DB = 1.0
ASL_DEV_MAX_DB = 0.5
TP_MAX_DBTP = -1.0
MAX_TOTAL_S = 900.0
EXPECTED_FILES = 6

#: P8-T1 第 5 条：与 P3 基线逐文件比对的容差
CUT_RATIO_TOL = 0.05          # cut_ratio 是 0..1 比例，5 个百分点 = 0.05
ASL_BASE_TOL_DB = 0.5
TP_BASE_TOL_DB = 1.0

#: P8-T1 第 6 条：时间戳基线阈值
TS_MEDIAN_MAX_S = 0.5
TS_P90_MAX_S = 1.5
#: 时间戳基线文件名（相对 ``scan_dir`` 的上级目录，即 ``work/measure_f24.json``）
MEASURE_BASELINE_NAME = "measure_f24.json"

#: P8-T1 第 7 条：ASR 段数/字数相对 P4 基线的最大偏差
ASR_TOL_RATIO = 0.10

#: rerun 模式默认独立目录（gitignore 已有 out_p8 / work/cache_p8）
DEFAULT_RERUN_OUT = "out_p8"
DEFAULT_RERUN_CACHE = "work/cache_p8"

#: rerun 从 data_dir 扫描的媒体后缀
MEDIA_EXTS = (
    ".m4a", ".m4b", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus",
    ".mp4", ".mkv", ".mov", ".wma",
)

#: out/ 下不是 job 目录的特殊目录（枚举 reuse job 时跳过）
SKIP_OUT_DIRS = frozenset({"logs", "quarantine", "review", "uploads", "locks", "__pycache__"})

#: P4 验收段数/字数基线（f24 已从 out/<id>/report.json 复核一致；见 docs/工程日志 P4-T2）。
#: 供 P8-T1 第 7 条「ASR 段数/字数偏差 ≤10%」使用；job_id → 基线。
P4_BASELINE: Dict[str, Dict[str, Any]] = {
    "7927721f_24_复旦哲学系丁耘_亚里士多德": {"key": "f24", "n_segments": 1175, "chars": 15524},
    "621cb91c_0_20140227_丁耘_西方哲学": {"key": "f00", "n_segments": 998, "chars": 9858},
    "9f3feb49_4_丁耘理想国_古希腊哲学_第四卷": {"key": "f04", "n_segments": 2299, "chars": 38060},
    "0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05": {"key": "f05", "n_segments": 1334, "chars": 28349},
    "b00e197d_复旦哲学系丁耘_孔子_论语_06_19": {"key": "f06", "n_segments": 1679, "chars": 22224},
    "4db18037_20_丁耘理想国_古希腊哲学": {"key": "f20", "n_segments": 1695, "chars": 26031},
}

#: ``report.timing`` 里属于「P3 render 内阶段」的键（total_s 已含，不可重复相加）
RENDER_TIMING_KEYS = ("decode_s", "analyze_s", "denoise_s", "trim_s",
                      "render_s", "encode_s", "measure_s")
#: 编排器后补的阶段（全链 = total_s + 这四项）
CHAIN_TAIL_TIMING_KEYS = ("asr_s", "diarize_s", "correct_s", "export_text_s")

#: CSV 列（P3 旧模式顺序固定，便于 diff 与归档）
CSV_COLUMNS = [
    "job_id", "source", "duration_s", "clean_s", "clean_min", "cut_ratio",
    "asl_in_dbfs", "asl_clean_dbfs", "asl_out_dbfs", "asl_dev_db",
    "lufs_in", "lufs_out", "true_peak_in_dbtp", "true_peak_out_dbtp",
    "gain_db", "gain_min_db", "gain_max_db", "max_rate_db_s", "limiter_triggers",
    "clipped_samples", "denoise_nr", "decode_s", "denoise_s", "trim_s", "render_s",
    "encode_s", "total_s", "ok", "skipped", "error",
]

#: P8 rows 契约字段（顺序即展示顺序；``duration_s``/``mode`` 为附加键，便于算 RTF）
ROW_KEYS = [
    "job_id", "source", "clean_s", "asl_out_dbfs", "asl_dev_db",
    "true_peak_out_dbtp", "clipped_samples", "cut_ratio", "render_total_s",
    "asr_s", "total_wall_s", "n_segments", "chars", "ok", "error",
]


# ================================================================ 小工具
def _num(value: Any) -> Optional[float]:
    """数值兜底：bool/None/NaN → None，其余 int/float 保留 3 位小数。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return round(float(value), 3) if value == value else None
        except (TypeError, ValueError):  # pragma: no cover - 防御
            return None
    return None


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value else None
    return None


def _float_or_none(value: Any) -> Optional[float]:
    """CSV 字符串 → float（空串/非法 → None）。"""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value == value else None
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _abs_path(value: Any, base: Optional[Path] = None) -> Path:
    """相对路径按项目根（或给定 base）解析成绝对路径。"""
    p = Path(value)
    if not p.is_absolute():
        p = (base or PROJECT_ROOT) / p
    return p.expanduser().resolve()


def _read_json(path: str | Path) -> Any:
    """读 JSON；文件不存在/非法返回 None（不抛异常，由调用方决定失败语义）。"""
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _full_chain_s(timing: Any) -> Optional[float]:
    """reuse 口径全链耗时 = render ``total_s`` + asr/diarize/correct/export_text。"""
    if not isinstance(timing, dict):
        return None
    total = _num(timing.get("total_s"))
    if total is None:
        return None
    tail = sum(_num(timing.get(key)) or 0.0 for key in CHAIN_TAIL_TIMING_KEYS)
    return round(total + tail, 3)


def _missing_products(job_dir: Path) -> List[str]:
    return [name for name in PRODUCT_FILES if not (Path(job_dir) / name).is_file()]


def _normalized_source(path: str | Path, base: Optional[Path] = None) -> str:
    """源路径归一化（相对路径按 base 解析 + Windows 大小写不敏感），用于反查 job id。"""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (base or PROJECT_ROOT) / p
    try:
        return str(p.resolve()).lower()
    except (OSError, ValueError):  # pragma: no cover - 非法路径
        return str(p).lower()


# ================================================================ 行数据（P3 旧模式）
def row_from_result(job_id: str, source: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """从 ``render_job`` 返回值抽取一行回归指标（P3 CSV 口径）。"""
    report = result.get("report") or {}
    trim = report.get("trim") or {}
    levels = report.get("levels") or {}
    loud = report.get("loudness") or {}
    gain = loud.get("gain") or {}
    timing = report.get("timing") or result.get("timings") or {}
    denoise = report.get("denoise") or {}
    clean_s = _num(trim.get("clean_dur_s"))
    asl_out = _num(levels.get("asl_out_dbfs"))
    asl_target = -20.0
    return {
        "job_id": job_id,
        "source": str(source),
        "duration_s": _num(report.get("duration_s")),
        "clean_s": clean_s,
        "clean_min": round(clean_s / 60.0, 3) if clean_s is not None else None,
        "cut_ratio": _num(trim.get("cut_ratio")),
        "asl_in_dbfs": _num(levels.get("asl_in_dbfs")),
        "asl_clean_dbfs": _num(levels.get("asl_clean_dbfs")),
        "asl_out_dbfs": asl_out,
        "asl_dev_db": round(asl_out - asl_target, 3) if asl_out is not None else None,
        "lufs_in": _num(levels.get("lufs_in")),
        "lufs_out": _num(levels.get("lufs_out")),
        "true_peak_in_dbtp": _num(levels.get("true_peak_in_dbtp")),
        "true_peak_out_dbtp": _num(levels.get("true_peak_out_dbtp")),
        "gain_db": _num(loud.get("gain_db")),
        "gain_min_db": _num(gain.get("min_db")),
        "gain_max_db": _num(gain.get("max_db")),
        "max_rate_db_s": _num(gain.get("max_rate_db_s")),
        "limiter_triggers": gain.get("limiter_triggers"),
        "clipped_samples": levels.get("clipped_samples_out"),
        "denoise_nr": _num(denoise.get("nr")),
        "decode_s": _num(timing.get("decode_s")),
        "denoise_s": _num(timing.get("denoise_s")),
        "trim_s": _num(timing.get("trim_s")),
        "render_s": _num(timing.get("render_s")),
        "encode_s": _num(timing.get("encode_s")),
        "total_s": _num(timing.get("total_s")),
        "ok": bool(result.get("ok")),
        "skipped": bool(result.get("skipped")),
        "error": (result.get("error") or "").replace("\n", " ")[:300],
    }


def row_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """从 ``out/<id>/report.json`` 构造同样的行（--from-reports）。"""
    fake = {"ok": bool(report.get("ok", True)), "skipped": False, "error": "",
            "report": report, "timings": report.get("timing") or {}}
    return row_from_result(report.get("id", ""), report.get("source", ""), fake)


# ================================================================ P8 行数据（reuse/rerun 共用）
def _row_from_report(job_id: Any, source: Any, report: Any, *,
                     ok: bool, error: str = "",
                     total_wall_s: Any = None) -> Dict[str, Any]:
    """从 ``report.json`` 构造 P8 回归行（契约字段 ``ROW_KEYS`` + ``duration_s``）。

    与 P3 CSV 行的区别：``render_total_s`` 只取 ``timing.total_s``（render 全链），
    ``total_wall_s`` 默认用 reuse 全链口径（total_s + asr/diarize/correct/export_text）；
    rerun 模式传入实测 wall（``run_job.timings.total_s``）。
    """
    report = report if isinstance(report, dict) else {}
    timing = report.get("timing") if isinstance(report.get("timing"), dict) else {}
    levels = report.get("levels") if isinstance(report.get("levels"), dict) else {}
    trim = report.get("trim") if isinstance(report.get("trim"), dict) else {}
    asr = report.get("asr") if isinstance(report.get("asr"), dict) else {}

    asl_out = _num(levels.get("asl_out_dbfs"))
    asl_dev = _num(levels.get("asl_dev_db"))
    if asl_dev is None and asl_out is not None:
        asl_dev = round(asl_out + 20.0, 3)  # 目标 −20 dBFS（旧 P3 口径）
    wall = _num(total_wall_s) if total_wall_s is not None else _full_chain_s(timing)
    return {
        "job_id": str(job_id or ""),
        "source": str(source or ""),
        "clean_s": _num(trim.get("clean_dur_s")),
        "asl_out_dbfs": asl_out,
        "asl_dev_db": asl_dev,
        "true_peak_out_dbtp": _num(levels.get("true_peak_out_dbtp")),
        "clipped_samples": _int_or_none(levels.get("clipped_samples_out")),
        "cut_ratio": _num(trim.get("cut_ratio")),
        "render_total_s": _num(timing.get("total_s")),
        "asr_s": _num(timing.get("asr_s")),
        "total_wall_s": wall,
        "n_segments": _int_or_none(asr.get("n_segments")),
        "chars": _int_or_none(asr.get("chars")),
        "ok": bool(ok),
        "error": (str(error) or "").replace("\n", " ")[:300],
        # 附加键：算 500 h 推算的 RTF 需要输入时长；不影响契约字段
        "duration_s": _num(report.get("duration_s")),
    }


def _summarize_rows(rows: Sequence[Dict[str, Any]], mode: str,
                    elapsed_s: float) -> Dict[str, Any]:
    """reuse/rerun 共用的运行汇总（含分阶段 RTF，供 docs/运行评估.md 引用）。"""
    ok_rows = [r for r in rows if r.get("ok")]

    def _values(key: str) -> List[float]:
        return [v for v in (_num(r.get(key)) for r in rows) if v is not None]

    durations = _values("duration_s")
    cleans = _values("clean_s")
    renders = _values("render_total_s")
    asrs = _values("asr_s")
    walls = _values("total_wall_s")
    asls = _values("asl_out_dbfs")
    tps = _values("true_peak_out_dbtp")
    clips = _values("clipped_samples")

    sum_input = round(sum(durations), 3) if durations else None
    sum_clean = round(sum(cleans), 3) if cleans else None
    sum_render = round(sum(renders), 3) if renders else None
    sum_asr = round(sum(asrs), 3) if asrs else None
    sum_wall = round(sum(walls), 3) if walls else None

    def _rtf(num: Optional[float], den: Optional[float]) -> Optional[float]:
        if num is None or den is None or den <= 0:
            return None
        return round(num / den, 4)

    return {
        "mode": mode,
        "n": len(rows),
        "n_ok": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "sum_input_s": sum_input,
        "sum_clean_s": sum_clean,
        "sum_render_total_s": sum_render,
        "sum_asr_s": sum_asr,
        "sum_wall_s": sum_wall,
        "render_rtf": _rtf(sum_render, sum_input),
        "asr_rtf": _rtf(sum_asr, sum_clean),
        "full_chain_rtf": _rtf(sum_wall, sum_input),
        "sum_wall_h": round(sum_wall / 3600.0, 3) if sum_wall is not None else None,
        "sum_input_h": round(sum_input / 3600.0, 3) if sum_input is not None else None,
        "elapsed_s": round(elapsed_s, 3),
        "asl_spread_db": (round(max(asls) - min(asls), 3) if len(asls) >= 2 else None),
        "asl_min_dbfs": round(min(asls), 2) if asls else None,
        "asl_max_dbfs": round(max(asls), 2) if asls else None,
        "tp_max_dbtp": round(max(tps), 2) if tps else None,
        "clipped_total": int(sum(clips)) if clips else None,
        "max_render_total_s": round(max(renders), 1) if renders else None,
    }


# ================================================================ reuse：枚举 out/<id>/
def _discover_job_dirs(out_root: Path) -> List[Path]:
    """枚举 ``out_root`` 下的 job 目录（含任意产物或 report.json，跳过特殊目录）。"""
    out_root = Path(out_root)
    if not out_root.is_dir():
        return []
    dirs: List[Path] = []
    for child in sorted(out_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name in SKIP_OUT_DIRS:
            continue
        has_product = any((child / name).is_file() for name in PRODUCT_FILES)
        if has_product:
            dirs.append(child)
    return dirs


def _run_reuse(data_dir: Any, cfg: Any, out_root: Path,
               limit: Optional[int]) -> Dict[str, Any]:
    """reuse：只读 ``out/<id>/report.json`` + 8 件产物做断言，不跑流水线。"""
    started = time.perf_counter()
    job_dirs = _discover_job_dirs(out_root)
    if limit is not None:
        job_dirs = job_dirs[: int(limit)]
    rows: List[Dict[str, Any]] = []
    for job_dir in job_dirs:
        report = _read_json(job_dir / "report.json")
        report_ok = isinstance(report, dict)
        source = report.get("source") if report_ok else ""
        job_id = (str(report.get("id") or job_dir.name) if report_ok else job_dir.name)
        missing = _missing_products(job_dir)
        errors: List[str] = []
        if not report_ok:
            errors.append("report.json 缺失或无法解析")
        if missing:
            errors.append("缺产物：" + ", ".join(missing))
        rows.append(_row_from_report(
            job_id, source, report,
            ok=report_ok and not missing, error="；".join(errors),
        ))
    elapsed = time.perf_counter() - started
    summary = _summarize_rows(rows, "reuse", elapsed)
    summary["out_dir"] = str(out_root)
    summary["data_dir"] = str(data_dir) if data_dir else ""
    if not job_dirs:
        summary["error"] = f"未发现 job 产物目录：{out_root}"
    return {
        "ok": bool(rows) and all(r.get("ok") for r in rows),
        "mode": "reuse",
        "n": len(rows),
        "rows": rows,
        "summary": summary,
    }


# ================================================================ rerun：独立目录真跑
def _iter_media_files(path: Path) -> List[Path]:
    """目录（递归）或单文件 → 媒体源列表（后缀白名单）。"""
    if path.is_file():
        return [path] if path.suffix.lower() in MEDIA_EXTS else []
    if not path.is_dir():
        return []
    return [f for f in sorted(path.rglob("*"))
            if f.is_file() and f.suffix.lower() in MEDIA_EXTS and not f.name.startswith(".")]


def _source_job_map(cfg: Any) -> Dict[str, str]:
    """``归一化源路径 → job_id``：只读扫原始 ``out/<id>/report.json`` 反查。

    回归约定「绝不写 out/」（连 ``Storage.init_db`` 的 WAL PRAGMA 都不做），所以这里
    **不打开** ``out/jobs.db``；6 个 job 的 report.json 始终存在。若 report 缺失，
    :func:`_score_inputs` 用确定性的 :func:`app.models.make_job_id` 生成同款 job id
    （6 个真实文件已核对一致），基线比对不受影响。
    """
    root = Path(getattr(cfg, "root", None) or PROJECT_ROOT)
    mapping: Dict[str, str] = {}
    for report in _jobs_from_reports(cfg):
        job_id, source = report.get("id"), report.get("source")
        if job_id and source:
            mapping[_normalized_source(source, base=root)] = str(job_id)
    return mapping


def _score_inputs(sources: Sequence[Path], job_map: Dict[str, str]) -> List[Any]:
    """源文件 → ``[(job_id, source)]``：能反查原 id 就保留，否则用 make_job_id 稳定生成。"""
    inputs: List[Any] = []
    taken: Dict[str, str] = {}
    for source in sources:
        job_id = job_map.get(_normalized_source(source))
        if job_id is None:
            try:
                job_id = make_job_id(source, taken)
            except Exception:  # pragma: no cover - 极端文件名
                job_id = source.stem
        taken[job_id] = str(source)
        inputs.append((job_id, source))
    return inputs


def _rerun_inputs(data_dir: Any, cfg: Any) -> List[Any]:
    """rerun 输入：``[(job_id, source)]``（尽量保留原 job id，便于与基线逐文件比对）。

    - 给了 ``data_dir``（文件/目录）→ 扫描媒体文件；
    - 没给或目录里没有媒体 → 从原始 ``out/<id>/report.json`` 的 source 反查（只读）。
    """
    root = Path(getattr(cfg, "root", None) or PROJECT_ROOT)
    sources: List[Path] = []
    if data_dir:
        sources = _iter_media_files(_abs_path(data_dir, root))
    if sources:
        return _score_inputs(sources, _source_job_map(cfg))
    jobs: List[Any] = []
    for report in _jobs_from_reports(cfg):
        job_id, source = report.get("id"), report.get("source")
        if job_id and source:
            jobs.append((str(job_id), Path(source)))
    return jobs


def _run_rerun(data_dir: Any, cfg: Any, out_root: Path, cache_root: Path,
               limit: Optional[int]) -> Dict[str, Any]:
    """rerun：独立 out/cache 真跑（orchestrator.batch_run），绝不写 out/。"""
    started = time.perf_counter()
    orig_out = cfg.path("out_dir")
    orig_db = cfg.path("db_path")
    if out_root.resolve() == orig_out.resolve():
        raise ValueError(
            f"rerun 模式禁止写 out/：out_dir({out_root}) 与 paths.out_dir({orig_out}) 相同；"
            f"请显式传独立 out_dir（默认 {DEFAULT_RERUN_OUT}）"
        )
    if (out_root / "jobs.db").resolve() == orig_db.resolve():
        raise ValueError(
            f"rerun 模式禁止写 out/：db_path 将落到原始 DB({orig_db})；请显式传独立 out_dir"
        )

    inputs = _rerun_inputs(data_dir, cfg)
    if limit is not None:
        inputs = list(inputs)[: int(limit)]
    summary: Dict[str, Any] = {
        "mode": "rerun",
        "out_dir": str(out_root),
        "cache_dir": str(cache_root),
    }
    if not inputs:
        summary["error"] = "未发现可重跑的输入（data_dir 为空且 out/<id>/report.json 无法反查 source）"
        summary.update(_summarize_rows([], "rerun", time.perf_counter() - started))
        return {"ok": False, "mode": "rerun", "n": 0, "rows": [], "summary": summary}

    # 参数必须真正生效：out/db/logs/cache/work/quarantine 全部指向独立目录
    overrides = {
        "paths.out_dir": str(out_root),
        "paths.db_path": str(out_root / "jobs.db"),
        "paths.logs_dir": str(out_root / "logs"),
        "paths.quarantine_dir": str(out_root / "quarantine"),
        "paths.cache_dir": str(cache_root),
        "paths.work_dir": str(cache_root.parent),
    }
    run_cfg = cfg.replace(overrides)
    summary["n_inputs"] = len(inputs)
    try:
        batch = batch_run(inputs, run_cfg, resume=False, force=True, limit=limit)
    except Exception as exc:  # pragma: no cover - batch_run 契约上不抛，兜底
        log.exception("rerun 编排失败")
        summary["error"] = f"编排失败：{exc}"
        summary.update(_summarize_rows([], "rerun", time.perf_counter() - started))
        return {"ok": False, "mode": "rerun", "n": 0, "rows": [], "summary": summary}

    rows: List[Dict[str, Any]] = []
    for result in (batch.get("results") or []):
        job_id = str(result.get("job_id") or "")
        report = result.get("report")
        if not isinstance(report, dict):
            report = _read_json(out_root / job_id / "report.json")
        ok = bool(result.get("ok"))
        error = str(result.get("error") or "")
        if ok and not isinstance(report, dict):
            missing = _missing_products(out_root / job_id)
            if missing:
                ok = False
                error = "缺产物：" + ", ".join(missing)
        wall = (result.get("timings") or {}).get("total_s")
        source = result.get("source") or (
            report.get("source") if isinstance(report, dict) else "")
        rows.append(_row_from_report(job_id, source, report, ok=ok,
                                     error=error, total_wall_s=wall))

    elapsed = time.perf_counter() - started
    row_summary = _summarize_rows(rows, "rerun", elapsed)
    summary["batch_ok"] = bool(batch.get("ok"))
    summary["stopped_reason"] = batch.get("stopped_reason") or ""
    summary["skipped"] = list(batch.get("skipped") or [])
    summary.update(row_summary)
    return {
        "ok": bool(rows) and all(r.get("ok") for r in rows),
        "mode": "rerun",
        "n": len(rows),
        "rows": rows,
        "summary": summary,
    }


# ================================================================ 固定接口 1：run_regression
def run_regression(data_dir: Any, cfg: Any, *, mode: str = "reuse",
                   out_dir: Any = None, cache_dir: Any = None,
                   limit: Optional[int] = None) -> Dict[str, Any]:
    """六文件回归：``mode="reuse"`` 只读产物；``mode="rerun"`` 独立目录真跑。

    参数
    ----
    data_dir : 媒体目录/文件（rerun 用来枚举输入；reuse 仅记录在 summary）
    cfg      : ``app.config.Config``
    mode     : ``"reuse"``（默认，秒级）或 ``"rerun"``（真跑约 80 min）
    out_dir  : reuse = 读取的产物根（默认 ``cfg.path("out_dir")``）；
               rerun = 写出的独立产物根（默认 ``out_p8``，禁止等于 out/）
    cache_dir: rerun 用独立缓存目录（默认 ``work/cache_p8``）
    limit    : 只处理前 N 个 job（冒烟）

    返回 ``{"ok", "mode", "n", "rows":[...], "summary":{...}}``；rows 每项含
    job_id/source/clean_s/asl_out_dbfs/asl_dev_db/true_peak_out_dbtp/clipped_samples/
    cut_ratio/render_total_s/asr_s/total_wall_s/n_segments/chars/ok/error（另附 duration_s）。
    """
    if mode not in ("reuse", "rerun"):
        raise ValueError(f"mode 只能是 'reuse' 或 'rerun'，当前：{mode!r}")
    if mode == "reuse":
        out_root = _abs_path(out_dir) if out_dir else cfg.path("out_dir")
        return _run_reuse(data_dir, cfg, out_root, limit)
    out_root = _abs_path(out_dir) if out_dir else _abs_path(DEFAULT_RERUN_OUT)
    cache_root = _abs_path(cache_dir) if cache_dir else _abs_path(DEFAULT_RERUN_CACHE)
    return _run_rerun(data_dir, cfg, out_root, cache_root, limit)


# ================================================================ 固定接口 2：check_regression
def _load_baseline_csv(path: Path) -> Dict[str, Dict[str, str]]:
    """读 P3 基线 ``audio.csv``（utf-8-sig；按 job_id 索引）。"""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {str(row.get("job_id") or ""): dict(row) for row in csv.DictReader(fh)}


def _check_row_level(rows: Sequence[Dict[str, Any]], failures: List[str]) -> None:
    for row in rows:
        if not row.get("ok"):
            failures.append(
                f"job: {row.get('job_id')} 未通过：{row.get('error') or '未知错误'}"
            )


def _check_aggregates(rows: Sequence[Dict[str, Any]], failures: List[str],
                      summary: Dict[str, Any]) -> None:
    """断言 1–4：ASL 极差 / 真峰值 / 削波 / 单文件 render 耗时。"""
    # 1) 跨文件 out ASL 极差 ≤ 1.0 dB
    asls = []
    missing_asl = []
    for row in rows:
        value = _num(row.get("asl_out_dbfs"))
        if value is None:
            missing_asl.append(str(row.get("job_id")))
        else:
            asls.append(value)
    spread = round(max(asls) - min(asls), 3) if len(asls) >= 2 else None
    summary["asl_spread_db"] = spread
    summary["asl_min_dbfs"] = round(min(asls), 2) if asls else None
    summary["asl_max_dbfs"] = round(max(asls), 2) if asls else None
    if missing_asl:
        failures.append(f"asl_spread: 缺 out ASL 的行：{', '.join(missing_asl)}")
    if spread is not None and spread > ASL_SPREAD_MAX_DB + 1e-9:
        failures.append(
            f"asl_spread: out ASL 极差 {spread} dB > {ASL_SPREAD_MAX_DB} dB"
        )

    # 2) 全部真峰值 ≤ −1.0 dBTP
    tp_values = []
    for row in rows:
        value = _num(row.get("true_peak_out_dbtp"))
        if value is None:
            failures.append(f"true_peak: {row.get('job_id')} 缺 out 真峰值")
            continue
        tp_values.append(value)
        if value > TP_MAX_DBTP + 1e-9:
            failures.append(
                f"true_peak: {row.get('job_id')} 真峰值 {value} > {TP_MAX_DBTP} dBTP"
            )
    summary["tp_max_dbtp"] = round(max(tp_values), 2) if tp_values else None

    # 3) 削波样本 = 0
    clip_values = []
    for row in rows:
        value = _int_or_none(row.get("clipped_samples"))
        if value is None:
            failures.append(f"clipped: {row.get('job_id')} 缺 clipped_samples")
            continue
        clip_values.append(value)
        if value != 0:
            failures.append(f"clipped: {row.get('job_id')} 削波样本 {value} != 0")
    summary["clipped_total"] = int(sum(clip_values)) if clip_values else None

    # 4) 单文件 render 全链 ≤ 900 s
    render_values = []
    for row in rows:
        value = _num(row.get("render_total_s"))
        if value is None:
            failures.append(f"render_total: {row.get('job_id')} 缺 render total_s")
            continue
        render_values.append(value)
        if value > MAX_TOTAL_S + 1e-9:
            failures.append(
                f"render_total: {row.get('job_id')} render total_s {value} > {MAX_TOTAL_S:.0f} s"
            )
    summary["max_render_total_s"] = round(max(render_values), 1) if render_values else None


def _check_baseline(rows: Sequence[Dict[str, Any]], baseline_csv: Any,
                    failures: List[str], summary: Dict[str, Any]) -> None:
    """断言 5：与 P3 基线（work/regress/audio.csv）逐文件比对。"""
    if baseline_csv is None:
        summary["baseline_checked"] = False
        summary["baseline_note"] = (
            "baseline_csv=None，跳过与 P3 基线（work/regress/audio.csv）的逐文件比对"
        )
        summary["cut_ratio_max_dev_pp"] = None
        summary["baseline_asl_max_dev_db"] = None
        summary["baseline_tp_max_dev_db"] = None
        return
    baseline_path = Path(baseline_csv)
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_ROOT / baseline_path
    summary["baseline_checked"] = True
    summary["baseline_csv"] = str(baseline_path)
    summary["cut_ratio_max_dev_pp"] = None
    summary["baseline_asl_max_dev_db"] = None
    summary["baseline_tp_max_dev_db"] = None
    if not baseline_path.is_file():
        failures.append(f"baseline: P3 基线文件缺失：{baseline_path}")
        return

    baseline = _load_baseline_csv(baseline_path)
    summary["baseline_rows"] = len(baseline)
    max_cut_pp = None
    max_asl_dev = None
    max_tp_dev = None
    for row in rows:
        job_id = str(row.get("job_id") or "")
        base = baseline.get(job_id)
        if base is None:
            failures.append(f"baseline: 基线缺该 job：{job_id}")
            continue
        # cut_ratio：比例差 → 百分点（×100）
        actual_cut = _num(row.get("cut_ratio"))
        base_cut = _float_or_none(base.get("cut_ratio"))
        if actual_cut is None:
            failures.append(f"baseline: {job_id} 缺当前 cut_ratio")
        elif base_cut is None:
            failures.append(f"baseline: {job_id} 基线缺 cut_ratio")
        else:
            dev_pp = abs(actual_cut - base_cut) * 100.0
            max_cut_pp = dev_pp if max_cut_pp is None else max(max_cut_pp, dev_pp)
            if dev_pp > CUT_RATIO_TOL * 100.0 + 1e-9:
                failures.append(
                    f"baseline: {job_id} cut_ratio {actual_cut:.4f} 与基线 {base_cut:.4f} "
                    f"偏差 {dev_pp:.1f} pp > 5 pp"
                )
        # out ASL ≤ 0.5 dB
        actual_asl = _num(row.get("asl_out_dbfs"))
        base_asl = _float_or_none(base.get("asl_out_dbfs"))
        if actual_asl is None:
            failures.append(f"baseline: {job_id} 缺当前 out ASL")
        elif base_asl is None:
            failures.append(f"baseline: {job_id} 基线缺 asl_out_dbfs")
        else:
            dev_asl = abs(actual_asl - base_asl)
            max_asl_dev = dev_asl if max_asl_dev is None else max(max_asl_dev, dev_asl)
            if dev_asl > ASL_BASE_TOL_DB + 1e-9:
                failures.append(
                    f"baseline: {job_id} out ASL {actual_asl} 与基线 {base_asl} "
                    f"偏差 {dev_asl:.2f} dB > {ASL_BASE_TOL_DB} dB"
                )
        # TP ≤ 1.0 dB
        actual_tp = _num(row.get("true_peak_out_dbtp"))
        base_tp = _float_or_none(base.get("true_peak_out_dbtp"))
        if actual_tp is None:
            failures.append(f"baseline: {job_id} 缺当前真峰值")
        elif base_tp is None:
            failures.append(f"baseline: {job_id} 基线缺 true_peak_out_dbtp")
        else:
            dev_tp = abs(actual_tp - base_tp)
            max_tp_dev = dev_tp if max_tp_dev is None else max(max_tp_dev, dev_tp)
            if dev_tp > TP_BASE_TOL_DB + 1e-9:
                failures.append(
                    f"baseline: {job_id} 真峰值 {actual_tp} 与基线 {base_tp} "
                    f"偏差 {dev_tp:.2f} dB > {TP_BASE_TOL_DB} dB"
                )
    summary["cut_ratio_max_dev_pp"] = round(max_cut_pp, 3) if max_cut_pp is not None else None
    summary["baseline_asl_max_dev_db"] = round(max_asl_dev, 3) if max_asl_dev is not None else None
    summary["baseline_tp_max_dev_db"] = round(max_tp_dev, 3) if max_tp_dev is not None else None


def _check_timestamps(scan_dir: Any, failures: List[str],
                      summary: Dict[str, Any]) -> None:
    """断言 6：读 ``work/measure_f24.json`` 的段起点时间戳基线。"""
    scan_path = Path(scan_dir)
    if not scan_path.is_absolute():
        scan_path = PROJECT_ROOT / scan_path
    measure_path = scan_path.parent / MEASURE_BASELINE_NAME
    info: Dict[str, Any] = {"path": str(measure_path)}
    summary["timestamp"] = info

    data = _read_json(measure_path)
    if not isinstance(data, dict):
        info["ok"] = False
        failures.append(f"timestamp: 时间戳基线缺失（{measure_path} 不存在或非法 JSON）")
        return
    boundaries = data.get("boundaries") if isinstance(data.get("boundaries"), dict) else {}
    start = boundaries.get("start_abs") if isinstance(boundaries.get("start_abs"), dict) else {}
    end = boundaries.get("end_abs") if isinstance(boundaries.get("end_abs"), dict) else {}
    fallback = data.get("summary") if isinstance(data.get("summary"), dict) else {}

    median = _num(start.get("abs_median"))
    p90 = _num(start.get("p90_abs"))
    if median is None:  # None 兜底：measure.py 旧输出可能只在 summary 里
        median = _num(fallback.get("start_abs_median"))
    if p90 is None:
        p90 = _num(fallback.get("start_p90_abs"))
    info.update({
        "start_abs_median": median,
        "start_p90_abs": p90,
        "end_abs_median": _num(end.get("abs_median")) or _num(fallback.get("end_abs_median")),
        "end_p90_abs": _num(end.get("p90_abs")) or _num(fallback.get("end_p90_abs")),
        "thresholds": {"abs_median_s": TS_MEDIAN_MAX_S, "p90_abs_s": TS_P90_MAX_S},
    })
    if median is None or p90 is None:
        info["ok"] = False
        failures.append(
            "timestamp: 时间戳基线缺失（measure_f24.json 缺 start_abs.abs_median / p90_abs）"
        )
        return
    passed = True
    if median > TS_MEDIAN_MAX_S + 1e-9:
        passed = False
        failures.append(
            f"timestamp: 段起点 abs_median {median} s > {TS_MEDIAN_MAX_S} s"
        )
    if p90 > TS_P90_MAX_S + 1e-9:
        passed = False
        failures.append(f"timestamp: 段起点 p90_abs {p90} s > {TS_P90_MAX_S} s")
    info["ok"] = passed


def _check_asr(rows: Sequence[Dict[str, Any]], failures: List[str],
               summary: Dict[str, Any]) -> None:
    """断言 7：ASR 段数/字数与 P4_BASELINE 偏差 ≤10%（基线缺该 job 跳过）。"""
    deviations: List[Dict[str, Any]] = []
    for row in rows:
        job_id = str(row.get("job_id") or "")
        base = P4_BASELINE.get(job_id)
        if base is None:
            continue  # 基线缺该 job → 跳过
        item: Dict[str, Any] = {"job_id": job_id, "key": base.get("key")}
        for metric in ("n_segments", "chars"):
            actual = _int_or_none(row.get(metric))
            base_value = int(base[metric])
            if actual is None:
                failures.append(f"asr: {job_id} 缺 ASR {metric}")
                item[f"{metric}_dev_ratio"] = None
                continue
            dev = abs(actual - base_value) / float(base_value)
            item[f"{metric}_dev_ratio"] = round(dev, 4)
            if dev > ASR_TOL_RATIO + 1e-9:
                failures.append(
                    f"asr: {job_id} ASR {metric}={actual} 与 P4 基线 {base_value} "
                    f"偏差 {dev:.1%} > {ASR_TOL_RATIO:.0%}"
                )
        deviations.append(item)
    summary["asr_baseline_checked"] = bool(deviations)
    summary["asr_deviation"] = deviations


def check_regression(result: Dict[str, Any], *, baseline_csv: Any = None,
                     scan_dir: Any = "work/scan") -> Dict[str, Any]:
    """对 :func:`run_regression` 的结果做 7 条断言。

    返回 ``{"ok", "failures":[...], "summary":{...}}``；failures 是中文可读字符串
    （``rule: detail`` 前缀便于定位），summary 含每条断言的关键指标与基线状态。
    """
    rows = list(result.get("rows") or [])
    failures: List[str] = []
    summary: Dict[str, Any] = {
        "n": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_failed": sum(1 for r in rows if not r.get("ok")),
        "mode": result.get("mode"),
    }
    _check_row_level(rows, failures)          # 0) 行级失败（缺产物/rerun 失败）
    if not rows:
        failures.append("rows: 没有可断言的 job 行（检查 out_dir / 产物 / limit）")
    _check_aggregates(rows, failures, summary)  # 1–4)
    _check_baseline(rows, baseline_csv, failures, summary)  # 5)
    _check_timestamps(scan_dir, failures, summary)          # 6)
    _check_asr(rows, failures, summary)                     # 7)
    summary["failures_count"] = len(failures)
    summary["ok"] = not failures
    return {"ok": not failures, "failures": failures, "summary": summary}


# ================================================================ 汇总/断言（P3 旧模式）
def evaluate(rows: Sequence[Dict[str, Any]], expected: int = EXPECTED_FILES) -> Dict[str, Any]:
    """P3 跨文件断言；返回指标与失败原因列表。"""
    ok_rows = [r for r in rows if r.get("ok")]
    asls = [r["asl_out_dbfs"] for r in ok_rows if r.get("asl_out_dbfs") is not None]
    devs = [abs(r["asl_dev_db"]) for r in ok_rows if r.get("asl_dev_db") is not None]
    tps = [r["true_peak_out_dbtp"] for r in ok_rows if r.get("true_peak_out_dbtp") is not None]
    clips = [r["clipped_samples"] for r in ok_rows if r.get("clipped_samples") is not None]
    totals = [r["total_s"] for r in ok_rows if r.get("total_s") is not None]
    rates = [r["max_rate_db_s"] for r in ok_rows if r.get("max_rate_db_s") is not None]

    metrics: Dict[str, Any] = {
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "asl_out_range_db": (round(max(asls) - min(asls), 3) if len(asls) >= 2 else None),
        "asl_out_min_dbfs": round(min(asls), 2) if asls else None,
        "asl_out_max_dbfs": round(max(asls), 2) if asls else None,
        "asl_dev_max_db": round(max(devs), 3) if devs else None,
        "tp_max_dbtp": round(max(tps), 2) if tps else None,
        "clipped_total": int(sum(clips)) if clips else None,
        "max_total_s": round(max(totals), 1) if totals else None,
        "max_rate_db_s": round(max(rates), 2) if rates else None,
    }
    reasons: List[str] = []
    if len(rows) < expected:
        reasons.append(f"文件数 {len(rows)} < 期望 {expected}")
    if metrics["n_failed"]:
        reasons.append(f"{metrics['n_failed']} 个文件渲染失败")
    if metrics["asl_out_range_db"] is None or metrics["asl_out_range_db"] > ASL_SPREAD_MAX_DB:
        reasons.append(f"out ASL 极差 {metrics['asl_out_range_db']} > {ASL_SPREAD_MAX_DB} dB")
    if metrics["asl_dev_max_db"] is None or metrics["asl_dev_max_db"] > ASL_DEV_MAX_DB:
        reasons.append(f"out ASL 最大偏差 {metrics['asl_dev_max_db']} > ±{ASL_DEV_MAX_DB} dB")
    if metrics["tp_max_dbtp"] is None or metrics["tp_max_dbtp"] > TP_MAX_DBTP:
        reasons.append(f"真峰值最大 {metrics['tp_max_dbtp']} > {TP_MAX_DBTP} dBTP")
    if metrics["clipped_total"] is None or metrics["clipped_total"] != 0:
        reasons.append(f"削波样本 {metrics['clipped_total']} != 0")
    if metrics["max_total_s"] is None or metrics["max_total_s"] > MAX_TOTAL_S:
        reasons.append(f"单文件最大耗时 {metrics['max_total_s']} > {MAX_TOTAL_S} s")
    if metrics["max_rate_db_s"] is not None and metrics["max_rate_db_s"] > 1.5 + 1e-6:
        reasons.append(f"V2 最大增益变化率 {metrics['max_rate_db_s']} > 1.5 dB/s")
    metrics["ok"] = not reasons
    metrics["reasons"] = reasons
    return metrics


# ================================================================ 输出
def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_COLUMNS})
    return p


def _fmt(value: Any, width: int = 8, spec: str = ".2f") -> str:
    if value is None:
        return " " * (width - 1) + "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:{width}{spec}}"
    return str(value)[:width]


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    print(f"\n{'job_id':44s} {'orig min':>8s} {'clean min':>9s} {'cut%':>6s} "
          f"{'asl_in':>7s} {'asl_out':>7s} {'TP':>6s} {'clip':>5s} {'total':>7s}")
    print("-" * 110)
    for r in rows:
        orig_min = (r["duration_s"] / 60.0) if r.get("duration_s") is not None else None
        cut = (r.get("cut_ratio") or 0) * 100 if r.get("cut_ratio") is not None else None
        print(f"{r['job_id']:44s} {_fmt(orig_min, 8, '.1f')} {_fmt(r.get('clean_min'), 9, '.1f')} "
              f"{_fmt(cut, 6, '.1f')} {_fmt(r.get('asl_in_dbfs'), 7, '.2f')} "
              f"{_fmt(r.get('asl_out_dbfs'), 7, '.2f')} {_fmt(r.get('true_peak_out_dbtp'), 6, '.2f')} "
              f"{_fmt(r.get('clipped_samples'), 5, '.0f')} {_fmt(r.get('total_s'), 7, '.1f')}")
    print("-" * 110)


def print_p8_table(rows: Sequence[Dict[str, Any]]) -> None:
    """P8 回归表：clean 分钟 / ASL / TP / 削波 / cut% / render / ASR / 全链 / 段数。"""
    print(f"\n{'job_id':44s} {'clean min':>9s} {'cut%':>6s} {'asl':>7s} "
          f"{'TP':>6s} {'clip':>5s} {'render':>7s} {'asr':>7s} {'wall':>7s} "
          f"{'seg':>5s} {'chars':>7s} ok")
    print("-" * 118)
    for r in rows:
        clean_min = (r["clean_s"] / 60.0) if r.get("clean_s") is not None else None
        cut = (r.get("cut_ratio") or 0) * 100 if r.get("cut_ratio") is not None else None
        print(f"{str(r.get('job_id'))[:44]:44s} {_fmt(clean_min, 9, '.1f')} "
              f"{_fmt(cut, 6, '.1f')} {_fmt(r.get('asl_out_dbfs'), 7, '.2f')} "
              f"{_fmt(r.get('true_peak_out_dbtp'), 6, '.2f')} "
              f"{_fmt(r.get('clipped_samples'), 5, '.0f')} "
              f"{_fmt(r.get('render_total_s'), 7, '.1f')} {_fmt(r.get('asr_s'), 7, '.1f')} "
              f"{_fmt(r.get('total_wall_s'), 7, '.1f')} {_fmt(r.get('n_segments'), 5, '.0f')} "
              f"{_fmt(r.get('chars'), 7, '.0f')} {'OK' if r.get('ok') else 'FAIL'}")
    print("-" * 118)


# ================================================================ 主流程
def _build_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m app.tools.regress",
        description="六文件回归：默认 --reuse 只读产物做 7 条断言（写 work/regress/p8.json）；"
                    "--rerun 独立目录真跑（out_p8 / work/cache_p8）",
    )
    ap.add_argument("data_dir", nargs="?", default=None,
                    help="媒体目录/文件（rerun 枚举输入；默认 paths.data_dir）")
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--reuse", dest="mode", action="store_const", const="reuse",
                            help="默认：只读 out/<id>/ 产物，不跑流水线")
    mode_group.add_argument("--rerun", dest="mode", action="store_const", const="rerun",
                            help="独立 out/cache 真跑（默认 out_p8 / work/cache_p8；禁止写 out/）")
    ap.set_defaults(mode="reuse")
    ap.add_argument("--config", default=None, help="配置文件（默认 configs/default.yaml）")
    ap.add_argument("--preset", default=None, choices=["fast", "standard", "quality"])
    ap.add_argument("--out", default=None,
                    help="reuse：读取的产物根（默认 out）；rerun：独立产物根（默认 out_p8）")
    ap.add_argument("--cache-dir", default=None,
                    help="rerun 独立缓存目录（默认 work/cache_p8）")
    ap.add_argument("--regress-dir", default="work/regress", help="p8.json/audio.csv 输出目录")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件（冒烟）")
    ap.add_argument("--json", action="store_true", help="只输出 JSON（日志不落 stdout）")
    # P3 旧模式的兼容开关（显式给出才走渲染/CSV 路径）
    ap.add_argument("--force", action="store_true",
                    help="[P3 旧模式] 已有产物也重渲染（写 audio.csv）")
    ap.add_argument("--from-reports", action="store_true",
                    help="[P3 旧模式] 不重跑，直接从 out/<id>/report.json 汇总写 audio.csv")
    return ap.parse_args(argv)


def _resolve_cfg(args: argparse.Namespace) -> Config:
    legacy = bool(args.from_reports or args.force)
    overrides: Dict[str, Any] = {}
    # rerun 的 out 由 run_regression 参数决定，不能先改 cfg（否则触发「禁止写 out/」）
    if args.out and (legacy or args.mode == "reuse"):
        overrides["paths.out_dir"] = args.out
        overrides["paths.db_path"] = str(Path(args.out) / "jobs.db")
        overrides["paths.logs_dir"] = str(Path(args.out) / "logs")
    if args.cache_dir:
        overrides["paths.cache_dir"] = args.cache_dir
    return load_config(
        config_path=args.config, preset=args.preset, overrides=overrides or None,
    )


def _iter_jobs(cfg: Config) -> List[Any]:
    store = Storage(cfg.path("db_path"))
    try:
        store.init_db()
        return store.list_jobs()
    finally:
        store.close()


def _jobs_from_reports(cfg: Config) -> List[Dict[str, Any]]:
    out_dir = cfg.path("out_dir")
    reports: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("*/" + "report.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("读取 report 失败：%s（%s）", path, exc)
            continue
        if isinstance(data, dict):
            reports.append(data)
    return reports


def _regress_dir(args: argparse.Namespace) -> Path:
    return _abs_path(args.regress_dir)


def _default_data_dir(cfg: Config) -> Optional[str]:
    try:
        return str(cfg.path("data_dir"))
    except ConfigError:
        return None


def _write_p8_payload(*, result: Dict[str, Any], check: Dict[str, Any],
                      out_path: Path) -> Dict[str, Any]:
    """P8 落盘 payload：6 行 + summary（run summary 与 check summary 合并）。"""
    summary: Dict[str, Any] = dict(result.get("summary") or {})
    summary.update(check.get("summary") or {})
    payload = {
        "ok": bool(check.get("ok")),
        "mode": result.get("mode"),
        "n": result.get("n"),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": result.get("rows") or [],
        "summary": summary,
        "failures": check.get("failures") or [],
    }
    _write_json(out_path, payload)
    return payload


def _print_check_summary(result: Dict[str, Any], check: Dict[str, Any],
                         out_path: Path) -> None:
    print_p8_table(result.get("rows") or [])
    print(f"\nP8 六文件回归（{result.get('mode')}，{result.get('n')} 文件）：")
    print(f"  out ASL 极差      : {_fmt(check['summary'].get('asl_spread_db'), 8, '.2f').strip() or '-'} dB  "
          f"(≤ {ASL_SPREAD_MAX_DB})")
    print(f"  真峰值最大        : {_fmt(check['summary'].get('tp_max_dbtp'), 8, '.2f').strip() or '-'} dBTP "
          f"(≤ {TP_MAX_DBTP})")
    print(f"  削波样本合计      : {check['summary'].get('clipped_total')}  (= 0)")
    print(f"  单文件最长 render : {check['summary'].get('max_render_total_s')} s  "
          f"(≤ {MAX_TOTAL_S:.0f})")
    ts = check["summary"].get("timestamp") or {}
    print(f"  时间戳段起点      : abs_median={ts.get('start_abs_median')} s / "
          f"p90={ts.get('start_p90_abs')} s  (≤ {TS_MEDIAN_MAX_S}/{TS_P90_MAX_S})")
    print(f"  全链 RTF          : {result['summary'].get('full_chain_rtf')} "
          f"（{result['summary'].get('sum_wall_h')} h / {result['summary'].get('sum_input_h')} h；"
          f"render RTF {result['summary'].get('render_rtf')}、ASR RTF {result['summary'].get('asr_rtf')}）")
    print(f"  结论              : {'PASS ✅' if check['ok'] else 'FAIL ❌'}")
    for reason in check.get("failures") or []:
        print(f"    - {reason}")
    print(f"\nP8 结果: {out_path}")


def _main_legacy(args: argparse.Namespace, cfg: Config, regress_dir: Path) -> int:
    """P3 旧模式：渲染/汇总 6 文件 → work/regress/audio.csv + summary.json。"""
    setup_logging(None, level="WARNING" if args.json else "INFO", console=not args.json,
                  force=True)
    csv_path = regress_dir / "audio.csv"
    summary_path = regress_dir / "summary.json"
    t_all = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    if args.from_reports:
        reports = _jobs_from_reports(cfg)
        if args.limit:
            reports = reports[: args.limit]
        if not reports:
            print("[警告] 没有 report.json 可供汇总", file=sys.stderr)
            return 2
        rows = [row_from_report(r) for r in reports]
    else:
        jobs = _iter_jobs(cfg)
        if args.limit:
            jobs = jobs[: args.limit]
        if not jobs:
            print("[警告] 数据库里没有 job（先跑 scan）", file=sys.stderr)
            return 2
        for job in jobs:
            print(f"[回归渲染] {job.id}（{job.source}）")
            result = render_job(
                job.id, job.source, cfg,
                force=args.force,
                skip_existing=not args.force,
            )
            row = row_from_result(job.id, job.source, result)
            rows.append(row)
            if not args.json:
                status = "OK" if row["ok"] else f"FAIL({result.get('stage')})"
                print(f"[{status:12s}] {job.id:44s} "
                      f"{row['clean_min'] if row['clean_min'] is not None else '-':>6} min "
                      f"asl_out={row['asl_out_dbfs']} TP={row['true_peak_out_dbtp']} "
                      f"total={row['total_s']}s")

    metrics = evaluate(rows, expected=args.limit or EXPECTED_FILES)
    metrics["wall_s"] = round(time.perf_counter() - t_all, 1)
    write_csv(csv_path, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.json:
        print_table(rows)
        print(f"\n跨文件验收（{len(rows)} 文件）：")
        print(f"  out ASL 极差      : {metrics['asl_out_range_db']} dB  (≤ {ASL_SPREAD_MAX_DB})")
        print(f"  out ASL 最大偏差  : {metrics['asl_dev_max_db']} dB  (≤ ±{ASL_DEV_MAX_DB})")
        print(f"  真峰值最大        : {metrics['tp_max_dbtp']} dBTP (≤ {TP_MAX_DBTP})")
        print(f"  削波样本合计      : {metrics['clipped_total']}  (= 0)")
        print(f"  V2 最大变化率     : {metrics['max_rate_db_s']} dB/s (≤ 1.5)")
        print(f"  单文件最长耗时    : {metrics['max_total_s']} s  (≤ {MAX_TOTAL_S:.0f})")
        print(f"  结论              : {'PASS ✅' if metrics['ok'] else 'FAIL ❌'}")
        for reason in metrics["reasons"]:
            print(f"    - {reason}")
        print(f"\nCSV: {csv_path}\n汇总: {summary_path}（wall {metrics['wall_s']}s）")
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["ok"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_args(argv if argv is not None else sys.argv[1:])
    try:
        cfg = _resolve_cfg(args)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    regress_dir = _regress_dir(args)

    # P3 旧模式只在显式 --force / --from-reports 时触发（默认路径不重写 audio.csv）
    if args.from_reports or args.force:
        return _main_legacy(args, cfg, regress_dir)

    setup_logging(None, level="CRITICAL" if args.json else "INFO",
                  console=not args.json, force=True)
    data_dir = args.data_dir or _default_data_dir(cfg)
    try:
        result = run_regression(
            data_dir, cfg, mode=args.mode,
            out_dir=args.out, cache_dir=args.cache_dir, limit=args.limit,
        )
    except ValueError as exc:  # rerun 目录安全校验等
        print(f"[参数错误] {exc}", file=sys.stderr)
        return 2

    baseline = regress_dir / "audio.csv"
    check = check_regression(
        result,
        baseline_csv=baseline if baseline.is_file() else None,
        scan_dir=regress_dir.parent / "scan",
    )
    out_path = regress_dir / ("p8_rerun.json" if args.mode == "rerun" else "p8.json")
    payload = _write_p8_payload(result=result, check=check, out_path=out_path)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_check_summary(result, check, out_path)
    return 0 if check["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
