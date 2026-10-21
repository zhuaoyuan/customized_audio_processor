# -*- coding: utf-8 -*-
"""P3-T5 六文件批量渲染回归：对 6 个样例跑 P1–P3，输出 ``work/regress/audio.csv`` 并做跨文件断言。

用法::

    work\\venv\\Scripts\\python -m app.tools.regress                  # 跑全部 job（产物已存在则跳过）
    work\\venv\\Scripts\\python -m app.tools.regress --force          # 强制重渲染
    work\\venv\\Scripts\\python -m app.tools.regress --limit 1        # 只跑 1 个（冒烟）
    work\\venv\\Scripts\\python -m app.tools.regress --from-reports   # 不重跑，只从 out/<id>/report.json 汇总

验收（全部硬断言，失败返回退出码 1）：
- 6 文件 out ASL 极差 ≤ 1.0 dB；
- 全部 ebur128 真峰值 ≤ −1.0 dBTP、解码后 0 个削波样本；
- 每个文件 out ASL 与 −20 dBFS 偏差 ≤ 0.5 dB；
- 单文件总耗时 ≤ 15 min（900 s）。

注意：本工具是 P8 回归的雏形，完整编排/断点续跑留给 P6 的 ``orchestrator``；这里只复用
``app.pipeline.export.render_job``（decode→analyze→denoise→trim→loudness→export 全链路）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config, ConfigError, PROJECT_ROOT, load_config
from ..logging_setup import get_logger, setup_logging
from ..models import STATUS_DONE
from ..pipeline.export import render_job
from ..storage import Storage

log = get_logger(__name__)

#: 跨文件验收阈值（与 docs/实施计划.md P3-T5 / P8-T1 对齐）
ASL_SPREAD_MAX_DB = 1.0
ASL_DEV_MAX_DB = 0.5
TP_MAX_DBTP = -1.0
MAX_TOTAL_S = 900.0
EXPECTED_FILES = 6

#: CSV 列（顺序固定，便于 diff 与归档）
CSV_COLUMNS = [
    "job_id", "source", "duration_s", "clean_s", "clean_min", "cut_ratio",
    "asl_in_dbfs", "asl_clean_dbfs", "asl_out_dbfs", "asl_dev_db",
    "lufs_in", "lufs_out", "true_peak_in_dbtp", "true_peak_out_dbtp",
    "gain_db", "gain_min_db", "gain_max_db", "max_rate_db_s", "limiter_triggers",
    "clipped_samples", "denoise_nr", "decode_s", "denoise_s", "trim_s", "render_s",
    "encode_s", "total_s", "ok", "skipped", "error",
]


# ---------------------------------------------------------------- 行数据
def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 3) if value == value else None
    return None


def row_from_result(job_id: str, source: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """从 ``render_job`` 返回值抽取一行回归指标。"""
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


# ---------------------------------------------------------------- 汇总/断言
def evaluate(rows: Sequence[Dict[str, Any]], expected: int = EXPECTED_FILES) -> Dict[str, Any]:
    """跨文件断言；返回指标与失败原因列表。"""
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


# ---------------------------------------------------------------- 输出
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


# ---------------------------------------------------------------- 主流程
def _build_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m app.tools.regress",
        description="P3-T5 六文件渲染回归（输出 work/regress/audio.csv + 跨文件断言）",
    )
    ap.add_argument("--config", default=None, help="配置文件（默认 configs/default.yaml）")
    ap.add_argument("--preset", default=None, choices=["fast", "standard", "quality"])
    ap.add_argument("--out", default=None, help="产物根目录（默认 out）")
    ap.add_argument("--cache-dir", default=None, help="中间缓存目录（默认 work/cache）")
    ap.add_argument("--regress-dir", default="work/regress", help="CSV/summary 输出目录")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个文件（冒烟）")
    ap.add_argument("--force", action="store_true", help="已有产物也重渲染")
    ap.add_argument("--from-reports", action="store_true",
                    help="不重跑，直接从 out/<id>/report.json 汇总")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出汇总")
    return ap.parse_args(argv)


def _resolve_cfg(args: argparse.Namespace) -> Config:
    overrides: Dict[str, Any] = {}
    if args.out:
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_args(argv if argv is not None else sys.argv[1:])
    try:
        cfg = _resolve_cfg(args)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg, level="INFO")
    regress_dir = Path(args.regress_dir)
    if not regress_dir.is_absolute():
        regress_dir = PROJECT_ROOT / regress_dir
    csv_path = regress_dir / "audio.csv"
    summary_path = regress_dir / "summary.json"

    t_all = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    if args.from_reports:
        reports = _jobs_from_reports(cfg)
        if args.limit:
            reports = reports[: args.limit]
        rows = [row_from_report(r) for r in reports]
    else:
        jobs = _iter_jobs(cfg)
        if args.limit:
            jobs = jobs[: args.limit]
        if not jobs:
            print("[警告] 数据库里没有 job（先跑 scan）", file=sys.stderr)
            return 2
        for job in jobs:
            log.info("回归渲染：%s（%s）", job.id, job.source)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
