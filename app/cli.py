# -*- coding: utf-8 -*-
"""命令行入口。

本轮（P0/P1）实现：``config show|check``、``scan``、``status``、``decode``。
P6 会在此基础上补齐 ``run`` / ``batch`` / ``retry`` / ``serve`` / ``export``。

约定
----
- 所有参数优先走 ``configs/default.yaml``；命令行只做**覆盖**，不改变默认值语义；
- ``--out <dir>`` 会同时把 db / logs 落到该目录下（除非显式给了 ``--db``），
  保证一份产物自包含、可直接拷走；
- 退出码：0 成功、1 运行失败、2 用法/配置错误、130 用户中断。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import __version__
from .config import PROJECT_ROOT, PRESETS, Config, ConfigError, load_config
from .logging_setup import get_logger, setup_logging
from .models import (
    DEFAULT_EXTENSIONS,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SKIPPED,
    STAGES,
    make_job_id,
)
from .pipeline.decode import decode as run_decode
from .pipeline.decode import DecodeError, cleanup_cache, wav_info
from .pipeline.analyze import analyze as run_analyze
from .pipeline.export import ExportError, render_job
from .storage import Storage, StorageError

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_INTERRUPT = 130

log = get_logger(__name__)


# ---------------------------------------------------------------- 公共工具
def build_parser() -> argparse.ArgumentParser:
    # 单一「全局选项」parent：顶层与每个子命令都挂它，于是
    # `cli --preset fast scan ...` 与 `cli scan ... --preset fast` 两种写法都成立。
    global_parent = _global_parent()
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description=f"audio_processer {__version__}：课程录音批处理（降噪/裁剪/响度/转写/纠错）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[global_parent],
        epilog=(
            "常用：\n"
            "  python -m app.cli config show\n"
            "  python -m app.cli scan data/ --out out/\n"
            "  python -m app.cli status\n"
            "  python -m app.cli decode --all\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"audio_processer {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # config
    p_config = subparsers.add_parser(
        "config", help="查看/校验合并后的配置", parents=[global_parent]
    )
    config_sub = p_config.add_subparsers(dest="config_command", metavar="<show|check>")
    p_show = config_sub.add_parser(
        "show", help="打印合并后的配置（含默认值与 preset）", parents=[global_parent]
    )
    p_show.add_argument("--json", action="store_true", help="以 JSON 输出（默认 YAML）")
    p_show.add_argument("--no-redact", action="store_true", help="不脱敏（会打印 API key，慎用）")
    p_show.add_argument("--save", default=None, help="同时另存为 YAML 文件")
    p_show.set_defaults(func=cmd_config_show)
    p_check = config_sub.add_parser(
        "check", help="检查配置引用的模型/目录是否存在", parents=[global_parent]
    )
    p_check.add_argument("--json", action="store_true")
    p_check.set_defaults(func=cmd_config_check)
    p_config.set_defaults(func=cmd_config_show, config_command="show")

    # scan
    p_scan = subparsers.add_parser(
        "scan", help="扫描输入目录，登记任务（pending）", parents=[global_parent]
    )
    p_scan.add_argument("path", help="输入目录或单个文件")
    p_scan.add_argument("--limit", type=int, default=None, help="最多登记 N 个文件（调试用）")
    p_scan.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    p_scan.add_argument("--dry-run", action="store_true", help="只打印将登记的文件，不写库")
    p_scan.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_scan.set_defaults(func=cmd_scan)

    # status
    p_status = subparsers.add_parser(
        "status", help="查看任务状态汇总", parents=[global_parent]
    )
    p_status.add_argument(
        "--status",
        choices=[STATUS_PENDING, "running", STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED],
        default=None,
        help="只看某个状态",
    )
    p_status.add_argument("--limit", type=int, default=200, help="最多显示 N 行（默认 200）")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    # decode
    p_decode = subparsers.add_parser(
        "decode", help="解码阶段：源文件 → 16k/44.1k float32 wav（含 stco 修复重试）",
        parents=[global_parent],
    )
    p_decode.add_argument("target", nargs="?", default=None, help="job id 或输入文件路径")
    p_decode.add_argument("--all", action="store_true", help="解码数据库里全部未完成的 job")
    p_decode.add_argument("--need-44k", action="store_true", help="同时解 44.1k 渲染轴")
    p_decode.add_argument("--no-cache", action="store_true", help="不复用缓存，强制重新解码")
    p_decode.add_argument("--cleanup", action="store_true", help="结束后清理中间 wav（仅保留 16k）")
    p_decode.add_argument("--limit", type=int, default=None, help="--all 时最多处理 N 个")
    p_decode.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本校验）")
    p_decode.set_defaults(func=cmd_decode)

    # analyze
    p_analyze = subparsers.add_parser(
        "analyze", help="分析阶段：VAD / 语音电平 / 5min 波动 / 内部长静音 / 头尾边界",
        parents=[global_parent],
    )
    p_analyze.add_argument("target", nargs="?", default=None, help="job id 或输入文件路径")
    p_analyze.add_argument("--all", action="store_true", help="分析数据库里全部 job")
    p_analyze.add_argument("--no-boundary", action="store_true",
                           help="跳过 ASR 边界判定（快，但头尾不裁并标 uncertain）")
    p_analyze.add_argument("--no-cache", action="store_true", help="不复用缓存，强制重算")
    p_analyze.add_argument("--limit", type=int, default=None, help="--all 时最多处理 N 个")
    p_analyze.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_analyze.set_defaults(func=cmd_analyze)

    # render（P3：降噪 → 裁剪 → 响度 → AAC 输出）
    p_render = subparsers.add_parser(
        "render",
        help="渲染阶段：降噪 → 裁剪 → 响度归一化 → 真峰值限幅 → clean.m4a / edit.json / report.json",
        parents=[global_parent],
    )
    p_render.add_argument("target", nargs="?", default=None, help="job id 或输入文件路径")
    p_render.add_argument("--all", action="store_true", help="渲染数据库里全部 job")
    p_render.add_argument("--limit", type=int, default=None, help="--all 时最多处理 N 个")
    p_render.add_argument("--force", action="store_true", help="已有产物也重做（仍复用 decode/analyze 缓存）")
    p_render.add_argument("--keep-intermediates", action="store_true",
                          help="保留中间 wav（denoised/clean/normalized，调试用）")
    p_render.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_render.set_defaults(func=cmd_render)

    return parser


def _global_parent() -> argparse.ArgumentParser:
    """子命令共用父 parser（用于 ``python -m app.cli scan ... --preset fast`` 这种写法）。"""
    parent = argparse.ArgumentParser(add_help=False)
    _add_global_options(parent)
    return parent


def _add_global_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/default.yaml）")
    parser.add_argument("--preset", choices=list(PRESETS), default=None, help="参数预设 fast|standard|quality")
    parser.add_argument("--out", dest="out_dir", default=None,
                        help="产物根目录（覆盖 paths.out_dir；同时把 db/logs 落到该目录）")
    parser.add_argument("--db", dest="db_path", default=None, help="SQLite 路径（覆盖 paths.db_path）")
    parser.add_argument("--cache-dir", default=None,
                        help="中间 wav 目录（覆盖 paths.cache_dir；默认 work/cache，不随 --out 变化）")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG 日志")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def load_cli_config(args: argparse.Namespace) -> Config:
    """根据命令行参数加载配置（命令行覆盖 > 环境变量 > preset > 默认值）。"""
    overrides: Dict[str, Any] = {}
    out_dir = getattr(args, "out_dir", None)
    if out_dir:
        out_path = Path(out_dir).expanduser()
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path)
        overrides["paths.out_dir"] = str(out_path)
        if not getattr(args, "db_path", None):
            overrides["paths.db_path"] = str(out_path / "jobs.db")
        overrides["paths.logs_dir"] = str(out_path / "logs")
    if getattr(args, "db_path", None):
        overrides["paths.db_path"] = args.db_path
    if getattr(args, "cache_dir", None):
        overrides["paths.cache_dir"] = args.cache_dir
    return load_config(
        config_path=getattr(args, "config", None),
        preset=getattr(args, "preset", None),
        overrides=overrides,
    )


def make_storage(cfg: Config) -> Storage:
    return Storage(cfg.path("db_path"), default_config_hash=cfg.config_hash)


def _json_dump(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _human_bytes(n: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def _fmt_ts(ts: Optional[int]) -> str:
    if not ts:
        return "-"
    import datetime

    return datetime.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _disp_width(text: str) -> int:
    """终端显示宽度：中日韩全角字符按 2 列算（否则表格会错位）。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, align: str = "left") -> str:
    text = "" if text is None else str(text)
    gap = max(0, width - _disp_width(text))
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def _print_table(headers: List[str], rows: List[List[str]], min_widths: List[int]) -> None:
    """打印 CJK 对齐的表格（最后一列（错误信息）不补空格）。"""
    widths = [
        max([min_widths[i]] + [_disp_width(row[i]) for row in rows])
        for i in range(len(headers))
    ]
    line = "  ".join(_pad(h, widths[i]) for i, h in enumerate(headers))
    print(line.rstrip())
    print("-" * (_disp_width(line.rstrip()) + 2))
    for row in rows:
        cells = [
            _pad(row[i], widths[i]) if i < len(headers) - 1 else row[i]
            for i in range(len(headers))
        ]
        print("  ".join(cells).rstrip())


# ---------------------------------------------------------------- config
def cmd_config_show(args: argparse.Namespace, cfg: Config) -> int:
    if not args.no_redact and cfg.section("correct").get("api_key"):
        print("[提示] correct.api_key 已脱敏；需要明文请加 --no-redact", file=sys.stderr)
    payload = cfg.to_dict(redact=not args.no_redact)
    if getattr(args, "json", False):
        _json_dump(payload)
    else:
        import yaml

        print(
            f"# preset={cfg.preset}  source={cfg.source}  config_hash={cfg.config_hash[:16]}",
            file=sys.stderr,
        )
        print(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False))
    save_path = getattr(args, "save", None)
    if save_path:
        from .config import write_yaml

        write_yaml(save_path, payload)
        print(f"[ok] 已另存：{Path(save_path).resolve()}", file=sys.stderr)
    return EXIT_OK


def cmd_config_check(args: argparse.Namespace, cfg: Config) -> int:
    """检查关键路径/模型是否就绪（不阻塞 config show）。"""
    checks: List[Tuple[str, str, bool, str]] = []

    def add(kind: str, label: str, path: Path, want_dir: bool) -> None:
        exists = path.is_dir() if want_dir else path.is_file()
        checks.append((kind, label, exists, str(path)))

    add("path", "data_dir", cfg.path("data_dir"), True)
    add("path", "cache_dir", cfg.path("cache_dir"), True)
    model = cfg.section("paths").get("models") or {}
    add("model", "sensevoice_dir", cfg.model_path("sensevoice_dir"), True) if model.get("sensevoice_dir") else None
    if model.get("rnnoise_model"):
        add("model", "rnnoise_model", cfg.model_path("rnnoise_model"), False)
    if model.get("whisper_cache"):
        add("model", "whisper_cache", cfg.model_path("whisper_cache"), True)
    if model.get("diar_dir"):
        add("model", "diar_dir", cfg.model_path("diar_dir"), True)

    missing = [c for c in checks if not c[2]]
    payload = {
        "config_hash": cfg.config_hash,
        "preset": cfg.preset,
        "checks": [
            {"kind": k, "name": n, "ok": ok, "path": p} for k, n, ok, p in checks
        ],
        "missing": [c[1] for c in missing],
    }
    if args.json:
        _json_dump(payload)
    else:
        print(f"preset={cfg.preset} config_hash={cfg.config_hash[:16]}")
        for kind, name, ok, path in checks:
            print(f"  [{'OK ' if ok else 'MISS'}] {kind:5s} {name:16s} {path}")
        print("全部就绪" if not missing else f"缺失 {len(missing)} 项（运行前请补齐）")
    return EXIT_OK if not missing else EXIT_RUNTIME


# ---------------------------------------------------------------- scan
def discover_files(
    root: str | Path,
    extensions: Optional[Iterable[str]] = None,
    recursive: bool = True,
    min_bytes: int = 0,
    max_bytes: int = 0,
) -> List[Path]:
    """发现待处理媒体文件（排序稳定，便于分批与断点续跑）。"""
    root_path = Path(root)
    exts = {e.lower() if e.startswith(".") else "." + e.lower()
            for e in (extensions or DEFAULT_EXTENSIONS)}
    if root_path.is_file():
        candidates = [root_path]
    elif root_path.is_dir():
        iterator = root_path.rglob("*") if recursive else root_path.glob("*")
        candidates = [p for p in iterator if p.is_file()]
    else:
        raise FileNotFoundError(f"输入路径不存在：{root_path}")

    selected: List[Path] = []
    for path in candidates:
        if path.suffix.lower() not in exts:
            continue
        if any(part.startswith(".") for part in path.relative_to(root_path if root_path.is_dir() else path.parent).parts):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if min_bytes and size < min_bytes:
            log.warning("跳过过小文件（%s < %s）：%s", _human_bytes(size), _human_bytes(min_bytes), path)
            continue
        if max_bytes and size > max_bytes:
            log.warning("跳过过大文件（> %s）：%s", _human_bytes(max_bytes), path)
            continue
        selected.append(path)
    selected.sort(key=lambda p: str(p).lower())
    return selected


def cmd_scan(args: argparse.Namespace, cfg: Config) -> int:
    scan_cfg = cfg.section("scan")
    try:
        files = discover_files(
            args.path,
            extensions=scan_cfg.get("extensions"),
            recursive=scan_cfg.get("recursive", True) and not args.no_recursive,
            min_bytes=int(scan_cfg.get("min_bytes", 0) or 0),
            max_bytes=int(scan_cfg.get("max_bytes", 0) or 0),
        )
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[警告] {args.path} 下没有发现可处理的媒体文件（扩展名：{scan_cfg.get('extensions')}）",
              file=sys.stderr)
        return EXIT_RUNTIME

    if args.dry_run:
        for path in files:
            print(f"{make_job_id(path)}\t{path}")
        return EXIT_OK

    store = make_storage(cfg)
    store.init_db()
    created = updated = 0
    rows: List[Dict[str, Any]] = []
    for path in files:
        job, is_new = store.upsert_job(path, config_hash=cfg.config_hash)
        created += int(is_new)
        updated += int(not is_new)
        rows.append({"id": job.id, "source": job.source, "status": job.status, "new": is_new})

    counts = store.count_by_status()
    if args.json:
        _json_dump({"db": store.db_path, "created": created, "existing": updated,
                    "jobs": rows, "counts": counts})
    else:
        print(f"扫描 {len(files)} 个文件 → db={store.db_path}")
        print(f"  新增 {created}，已存在 {updated}")
        for row in rows:
            flag = "new " if row["new"] else "have"
            print(f"  [{flag}] {row['id']}  {Path(row['source']).name}")
        print("状态：" + "，".join(f"{k}={v}" for k, v in counts.items() if k != "total"))
    store.close()
    return EXIT_OK


# ---------------------------------------------------------------- status
def cmd_status(args: argparse.Namespace, cfg: Config) -> int:
    db_path = cfg.path("db_path")
    if not Path(db_path).exists():
        print(f"[错误] 数据库不存在：{db_path}\n先执行：python -m app.cli scan <输入目录> --out <输出目录>",
              file=sys.stderr)
        return EXIT_USAGE
    with make_storage(cfg) as store:
        store.init_db()
        if args.json:
            jobs = store.jobs_snapshot(limit=args.limit)
            if args.status:
                jobs = [j for j in jobs if j["status"] == args.status]
            _json_dump({"summary": store.summary(), "jobs": jobs})
            return EXIT_OK
        counts = store.count_by_status()
        jobs = store.list_jobs(status=args.status, limit=args.limit)
        print(f"数据库：{db_path}")
        print("状态汇总：" + "，".join(f"{k}={v}" for k, v in counts.items()))
        if not jobs:
            print("（没有匹配的任务）")
            return EXIT_OK
        rows: List[List[str]] = []
        for job in jobs:
            err = (job.error or "").replace("\n", " ")
            if len(err) > 60:
                err = err[:57] + "..."
            rows.append([
                job.id, job.status, job.stage or "-", str(job.attempts),
                _fmt_ts(job.updated_at), err,
            ])
        _print_table(
            ["id", "状态", "阶段", "尝试", "更新时间", "错误"],
            rows,
            [20, 7, 8, 4, 19, 0],
        )
        print(f"共 {len(jobs)} 条（显示上限 {args.limit}）")
    return EXIT_OK


# ---------------------------------------------------------------- decode
def _resolve_decode_targets(
    args: argparse.Namespace, cfg: Config, store: Storage
) -> List[Tuple[str, Path]]:
    """把命令行 target 解析成 [(job_id, source_path)]。"""
    if args.all:
        jobs = store.list_jobs(statuses=["pending", "running", "failed"])
        if args.limit:
            jobs = jobs[: args.limit]
        return [(job.id, Path(job.source)) for job in jobs]

    if not args.target:
        raise ConfigError("请给出 job id / 文件路径，或用 --all 解码全部待处理任务")
    target = args.target
    job = store.get_job(target)
    if job is not None:
        return [(job.id, Path(job.source))]
    path = Path(target)
    if path.is_file():
        existing = store.get_job_by_source(path)
        if existing is not None:
            return [(existing.id, path)]
        return [(make_job_id(path), path)]
    raise ConfigError(f"找不到 job 或文件：{target}")


def _record_stage(
    store: Storage, job_id: str, ok: bool, result: Any, elapsed: float
) -> None:
    """把 decode 阶段结果落库（阶段级 error 写进 metrics）。"""
    if store.get_job(job_id) is None:
        return
    store.start_stage(job_id, "decode", reset=not ok)
    metrics = {
        "ok": bool(ok),
        "repaired": bool(getattr(result, "repaired", False)),
        "duration_s": round(float(getattr(result, "duration_s", 0.0) or 0.0), 3),
        "sample_rate": int(getattr(result, "sample_rate", 0) or 0),
        "orig_sample_rate": int(getattr(result, "orig_sample_rate", 0) or 0),
        "bit_rate": int(getattr(result, "bit_rate", 0) or 0),
        "from_cache": bool(getattr(result, "from_cache", False)),
        "elapsed_s": round(elapsed, 3),
    }
    if ok:
        store.finish_stage(job_id, "decode", STATUS_DONE, metrics)
        # 只跑了解码这一阶段：job 仍待后续（stage 记为 decode 便于状态展示）
        store.update_job(job_id, status=STATUS_PENDING, stage="decode")
    else:
        store.fail_stage(job_id, "decode", getattr(result, "error", "") or "解码失败", metrics)
        store.fail_job(job_id, getattr(result, "error", "") or "解码失败")


def cmd_decode(args: argparse.Namespace, cfg: Config) -> int:
    store = make_storage(cfg)
    store.init_db()
    try:
        targets = _resolve_decode_targets(args, cfg, store)
    except ConfigError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        store.close()
        return EXIT_USAGE
    if not targets:
        print("[警告] 没有待解码的任务", file=sys.stderr)
        store.close()
        return EXIT_OK

    results: List[Dict[str, Any]] = []
    failures = 0
    for job_id, source in targets:
        import time

        t0 = time.perf_counter()
        result = run_decode(
            source,
            cache_dir=cfg.path("cache_dir"),
            job_id=job_id,
            need_44k=args.need_44k,
            cfg=cfg,
            use_cache=not args.no_cache,
            force=args.no_cache,
        )
        elapsed = time.perf_counter() - t0
        _record_stage(store, job_id, result.ok, result, elapsed)
        if not result.ok:
            failures += 1
        item = {"job_id": job_id, "source": str(source), "ok": result.ok,
                "elapsed_s": round(elapsed, 2), **result.to_dict()}
        item["elapsed_s"] = round(elapsed, 2)
        item["repair"] = result.repair
        results.append(item)
        if not args.json:
            status = "OK  " if result.ok else "FAIL"
            extra = f"repaired={result.repaired}" if result.repaired else f"cache={result.from_cache}"
            note = "" if result.ok else f"  {result.error.splitlines()[0][:120]}"
            print(f"[{status}] {_pad(job_id, 40)} {result.duration_s:8.2f}s "
                  f"{result.sample_rate:>6d}Hz {extra:16s} {elapsed:6.2f}s{note}")

        if args.cleanup:
            # 仅清 44k/fixed 等中间文件；保留 16k + 缓存元数据，否则下次又得重新解码
            removed = cleanup_cache(
                cfg.path("cache_dir"), job_id, keep_16k=True, keep_44k=False, keep_meta=True
            )
            if removed and not args.json:
                print(f"       清理 {len(removed)} 个中间文件")

    store.close()
    if args.json:
        _json_dump({"ok": failures == 0, "n": len(results), "failures": failures,
                    "results": results})
    else:
        ok_n = len(results) - failures
        print(f"\n解码完成：成功 {ok_n} / {len(results)}"
              + (f"，失败 {failures}" if failures else ""))
    return EXIT_OK if failures == 0 else EXIT_RUNTIME


# ---------------------------------------------------------------- analyze
def _resolve_analyze_targets(
    args: argparse.Namespace, store: Storage
) -> List[Tuple[str, Path]]:
    if args.all:
        jobs = store.list_jobs()
        if args.limit:
            jobs = jobs[: args.limit]
        return [(job.id, Path(job.source)) for job in jobs]
    if not args.target:
        raise ConfigError("请给出 job id / 文件路径，或用 --all 分析全部任务")
    job = store.get_job(args.target)
    if job is not None:
        return [(job.id, Path(job.source))]
    path = Path(args.target)
    if path.is_file():
        existing = store.get_job_by_source(path)
        return [(existing.id if existing else make_job_id(path), path)]
    raise ConfigError(f"找不到 job 或文件：{args.target}")


def cmd_analyze(args: argparse.Namespace, cfg: Config) -> int:
    store = make_storage(cfg)
    store.init_db()
    try:
        targets = _resolve_analyze_targets(args, store)
    except ConfigError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        store.close()
        return EXIT_USAGE
    if not targets:
        print("[警告] 没有待分析的任务", file=sys.stderr)
        store.close()
        return EXIT_OK

    results: List[Dict[str, Any]] = []
    failures = 0
    for job_id, source in targets:
        import time

        t0 = time.perf_counter()
        # 1) 确保 16k wav（缓存命中时 <0.1s）
        decoded = run_decode(
            source, cache_dir=cfg.path("cache_dir"), job_id=job_id, cfg=cfg,
        )
        if not decoded.ok or not decoded.wav16k:
            failures += 1
            message = f"解码失败，无法分析：{decoded.error}"
            if store.get_job(job_id) is not None:
                store.fail_stage(job_id, "analyze", message)
                store.fail_job(job_id, message)
            if not args.json:
                print(f"[FAIL] {_pad(job_id, 40)} {message[:110]}")
            results.append({"job_id": job_id, "ok": False, "error": message})
            continue

        # 2) 分析
        try:
            result = run_analyze(
                decoded.wav16k, cfg, job_id=job_id,
                run_boundary=not args.no_boundary,
                use_cache=not args.no_cache,
                force=args.no_cache,
            )
        except Exception as exc:  # 分析失败不阻塞其它文件
            elapsed = time.perf_counter() - t0
            message = f"{type(exc).__name__}: {exc}"
            if store.get_job(job_id) is not None:
                store.fail_stage(job_id, "analyze", message, {"elapsed_s": round(elapsed, 2)})
                store.fail_job(job_id, message)
            failures += 1
            if not args.json:
                print(f"[FAIL] {_pad(job_id, 40)} {message[:110]}")
            results.append({"job_id": job_id, "ok": False, "error": message})
            continue
        elapsed = time.perf_counter() - t0

        # 3) 落库（阶段级 error 走 metrics；状态机只在有 job 时更新）
        if store.get_job(job_id) is not None:
            store.start_stage(job_id, "analyze", reset=True)
            store.finish_stage(job_id, "analyze", STATUS_DONE, _analyze_metrics(result, elapsed))
            store.update_job(job_id, status=STATUS_PENDING, stage="analyze")

        item = {
            "job_id": job_id,
            "ok": True,
            "elapsed_s": round(elapsed, 2),
            **result.to_dict(),
        }
        item.pop("segments", None)  # 段表太长，json 输出只给摘要
        results.append(item)
        if not args.json:
            print(_analyze_row(job_id, result, elapsed))

    store.close()
    if args.json:
        _json_dump({"ok": failures == 0, "n": len(results), "failures": failures,
                    "results": results})
    else:
        ok_n = len(results) - failures
        print(f"\n分析完成：成功 {ok_n} / {len(results)}"
              + (f"，失败 {failures}" if failures else ""))
    return EXIT_OK if failures == 0 else EXIT_RUNTIME


def _analyze_metrics(result: Any, elapsed: float) -> Dict[str, Any]:
    """落库用的精简指标（完整结果在 work/cache/<id>.analyze.json）。"""
    vad = result.vad or {}
    levels = result.levels or {}
    trim = result.trim or {}
    boundary = result.boundary or {}
    return {
        "ok": True,
        "duration_s": round(float(result.duration_s), 3),
        "speech_ratio": vad.get("speech_ratio"),
        "n_segments": vad.get("n_segments"),
        "asl_in_dbfs": levels.get("asl_in_dbfs"),
        "asl_all_vad_dbfs": levels.get("asl_all_vad_dbfs"),
        "noise_floor_dbfs": levels.get("noise_floor_dbfs"),
        "snr_db": levels.get("snr_db"),
        "lufs_in": levels.get("lufs_in"),
        "window5m_half_spread_db": levels.get("window5m_half_spread_db"),
        "head_s": trim.get("head_s"),
        "tail_s": trim.get("tail_s"),
        "internal_cut_s": trim.get("internal_cut_s"),
        "internal_gap_total_s": trim.get("internal_gap_total_s"),
        "flags": trim.get("flags"),
        "manual_review": trim.get("manual"),
        "boundary_method": boundary.get("method"),
        "from_cache": bool(result.from_cache),
        "elapsed_s": round(elapsed, 2),
    }


def _analyze_row(job_id: str, result: Any, elapsed: float) -> str:
    levels = result.levels
    trim = result.trim
    flags = ",".join(trim.get("flags") or []) or "-"
    return (
        f"[OK  ] {_pad(job_id, 40)} dur={result.duration_s:7.1f}s "
        f"speech={result.vad['speech_ratio'] * 100:5.1f}% "
        f"ASL={levels['asl_in_dbfs']:7.2f} noise={levels['noise_floor_dbfs']:6.1f} "
        f"SNR={levels['snr_db']:5.1f} lufs={levels.get('lufs_in')} "
        f"head={trim['head_s']:6.1f}s tail={trim['tail_s']:6.1f}s "
        f"int_gap={trim['internal_cut_s']:6.1f}s flags={flags:28s} "
        f"{'cache' if result.from_cache else 'calc '} {elapsed:6.1f}s"
    )


# ---------------------------------------------------------------- render（P3）
def _resolve_render_targets(
    args: argparse.Namespace, store: Storage
) -> List[Tuple[str, Path]]:
    if args.all:
        jobs = store.list_jobs()
        if args.limit:
            jobs = jobs[: args.limit]
        return [(job.id, Path(job.source)) for job in jobs]
    if not args.target:
        raise ConfigError("请给出 job id / 文件路径，或用 --all 渲染全部任务")
    job = store.get_job(args.target)
    if job is not None:
        return [(job.id, Path(job.source))]
    path = Path(args.target)
    if path.is_file():
        existing = store.get_job_by_source(path)
        return [(existing.id if existing else make_job_id(path), path)]
    raise ConfigError(f"找不到 job 或文件：{args.target}")


def _stage_recorder(store: Storage, job_id: str):
    """render_job 的阶段回调：denoise/trim/loudness/export 落 stages 表。"""

    def on_stage(stage: str, status: str, metrics: Dict[str, Any]) -> None:
        if store.get_job(job_id) is None:
            return
        try:
            if status == "done":
                store.start_stage(job_id, stage, reset=True)
                store.finish_stage(job_id, stage, STATUS_DONE, metrics)
            elif status == "failed":
                store.fail_stage(
                    job_id, stage, str(metrics.get("error") or "渲染失败"), metrics
                )
        except Exception:  # pragma: no cover - 落库失败不阻塞渲染
            log.debug("阶段落库失败：%s/%s", job_id, stage, exc_info=True)

    return on_stage


def _render_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """JSON 输出用的精简摘要（完整报告在 out/<id>/report.json）。"""
    report = result.get("report") or {}
    trim = report.get("trim") or {}
    levels = report.get("levels") or {}
    out = report.get("output") or {}
    loud = report.get("loudness") or {}
    timing = report.get("timing") or result.get("timings") or {}
    return {
        "job_id": result.get("job_id"),
        "ok": bool(result.get("ok")),
        "skipped": bool(result.get("skipped")),
        "stage": result.get("stage"),
        "error": result.get("error", ""),
        "clean_dur_s": trim.get("clean_dur_s"),
        "cut_ratio": trim.get("cut_ratio"),
        "asl_in_dbfs": levels.get("asl_in_dbfs"),
        "asl_out_dbfs": levels.get("asl_out_dbfs"),
        "asl_dev_db": levels.get("asl_dev_db"),
        "lufs_out": levels.get("lufs_out"),
        "true_peak_out_dbtp": levels.get("true_peak_out_dbtp"),
        "clipped_samples_out": levels.get("clipped_samples_out"),
        "gain_db": loud.get("gain_db"),
        "gain_min_db": (loud.get("gain") or {}).get("min_db"),
        "gain_max_db": (loud.get("gain") or {}).get("max_db"),
        "max_rate_db_s": (loud.get("gain") or {}).get("max_rate_db_s"),
        "limiter_triggers": (loud.get("gain") or {}).get("limiter_triggers"),
        "bit_rate": out.get("bit_rate"),
        "size_bytes": out.get("size_bytes"),
        "timing": timing,
    }


def _render_row(job_id: str, summary: Dict[str, Any]) -> str:
    if not summary["ok"]:
        return (f"[FAIL] {_pad(job_id, 40)} {str(summary.get('stage')):9s} "
                f"{(summary.get('error') or '')[:110]}")
    tag = "SKIP" if summary.get("skipped") else "OK  "
    timing = summary.get("timing") or {}
    total = timing.get("total_s")
    total_s = f"{float(total):7.1f}s" if isinstance(total, (int, float)) else "     -  "
    return (
        f"[{tag}] {_pad(job_id, 40)} "
        f"asl_in={summary.get('asl_in_dbfs') if summary.get('asl_in_dbfs') is not None else float('nan'):7.2f} "
        f"out={summary.get('asl_out_dbfs') if summary.get('asl_out_dbfs') is not None else float('nan'):7.2f} "
        f"gain={summary.get('gain_db') if summary.get('gain_db') is not None else float('nan'):+6.1f}dB "
        f"TP={summary.get('true_peak_out_dbtp') if summary.get('true_peak_out_dbtp') is not None else float('nan'):6.2f}dBTP "
        f"clip={summary.get('clipped_samples_out')} "
        f"clean={float(summary.get('clean_dur_s') or 0.0) / 60.0:6.1f}min "
        f"{total_s}"
    )


def cmd_render(args: argparse.Namespace, cfg: Config) -> int:
    store = make_storage(cfg)
    store.init_db()
    try:
        targets = _resolve_render_targets(args, store)
    except ConfigError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        store.close()
        return EXIT_USAGE
    if not targets:
        print("[警告] 没有待渲染的任务", file=sys.stderr)
        store.close()
        return EXIT_OK

    summaries: List[Dict[str, Any]] = []
    failures = 0
    for job_id, source in targets:
        result = render_job(
            job_id, source, cfg,
            force=args.force,
            keep_intermediates=args.keep_intermediates,
            on_stage=_stage_recorder(store, job_id),
        )
        summary = _render_summary(result)
        summaries.append(summary)
        if result.get("ok"):
            if store.get_job(job_id) is not None:
                store.update_job(job_id, status=STATUS_DONE, stage="export")
            if not args.json:
                print(_render_row(job_id, summary))
                if result.get("skipped"):
                    print(f"       产物已存在，跳过（--force 可重做）")
        else:
            failures += 1
            message = result.get("error") or "渲染失败"
            if store.get_job(job_id) is not None:
                store.fail_job(job_id, message)
            if not args.json:
                print(_render_row(job_id, summary))

    store.close()
    if args.json:
        _json_dump({"ok": failures == 0, "n": len(summaries), "failures": failures,
                    "results": summaries})
    else:
        ok_n = len(summaries) - failures
        print(f"\n渲染完成：成功 {ok_n} / {len(summaries)}"
              + (f"，失败 {failures}" if failures else ""))
    return EXIT_OK if failures == 0 else EXIT_RUNTIME


# ---------------------------------------------------------------- 入口
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None) and not getattr(args, "config_command", None):
        parser.print_help()
        return EXIT_USAGE

    try:
        cfg = load_cli_config(args)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return EXIT_USAGE

    level = "DEBUG" if getattr(args, "verbose", False) else getattr(args, "log_level", None)
    setup_logging(cfg, level=level)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(func(args, cfg))
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (StorageError, DecodeError, ExportError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    except KeyboardInterrupt:
        print("[中断] 用户取消", file=sys.stderr)
        return EXIT_INTERRUPT


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
