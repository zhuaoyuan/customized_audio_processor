# -*- coding: utf-8 -*-
"""命令行入口。

P0–P5：``config show|check``、``scan``、``status``、``decode``、``analyze``、``render``、
``transcribe``、``correct``、``export``。
P6/P7 新增：``run`` / ``batch`` / ``retry``（全链路编排，见 ``app/pipeline/orchestrator.py``）、
``status --watch``、``serve``（本地 Web UI，见 ``app/server.py``）。
P10 新增：``compose``（全文成稿：整讲课程档案 + 分块整理为书面稿；独立于 ``run`` 全链路）。

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
            "  python -m app.cli run data\\24_...m4a --out out\n"
            "  python -m app.cli batch data/ --out out --resume\n"
            "  python -m app.cli retry <job id>\n"
            "  python -m app.cli status --watch\n"
            "  python -m app.cli serve --port 7860\n"
            "  python -m app.cli regress --reuse --json\n"
            "  python -m app.cli transcribe --all\n"
            "  python -m app.cli correct --all\n"
            "  python -m app.cli export --all\n"
            "  python -m app.cli compose <job id>   # 全文成稿（可选，独立于全链路）\n"
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
    p_status.add_argument("--watch", action="store_true", help="每 2 s 刷新（Ctrl-C 退出）")
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

    # transcribe（P4：VAD 段逐段 ASR + clean/orig 双时间轴）
    p_transcribe = subparsers.add_parser(
        "transcribe",
        help="转写阶段：VAD 段逐段 SenseVoice + clean/orig 句级时间戳 → transcript.json",
        parents=[global_parent],
    )
    p_transcribe.add_argument("target", nargs="?", default=None, help="job id 或输入文件路径")
    p_transcribe.add_argument("--all", action="store_true", help="转写数据库里全部 job")
    p_transcribe.add_argument("--limit", type=int, default=None, help="--all 时最多处理 N 个")
    p_transcribe.add_argument("--no-cache", action="store_true",
                              help="已有 transcript.json 也重做")
    p_transcribe.add_argument("--merge-gap", type=float, default=None,
                              help="可选：合并 gap<=N 秒的相邻段（默认不合并；>30s 不合并）")
    p_transcribe.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_transcribe.set_defaults(func=cmd_transcribe)

    # correct（P5：DeepSeek 分块纠错 + 校验回退）
    p_correct = subparsers.add_parser(
        "correct",
        help="纠错阶段：deepseek-flash 按主题/热词分块纠错，校验不过整块回退 → corrections.json",
        parents=[global_parent],
    )
    p_correct.add_argument("target", nargs="?", default=None, help="job id 或输入文件路径")
    p_correct.add_argument("--all", action="store_true", help="纠错数据库里全部 job")
    p_correct.add_argument("--limit", type=int, default=None, help="--all 时最多处理 N 个")
    p_correct.add_argument("--no-cache", action="store_true",
                           help="已有 corrections.json 也重做")
    p_correct.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_correct.set_defaults(func=cmd_correct)

    # export（P5-T4：md/srt/txt 文本导出；音频导出仍走 render）
    p_export = subparsers.add_parser(
        "export",
        help="文本导出：transcript.json → transcript.md / .srt / .txt（clean 轴，句级）",
        parents=[global_parent],
    )
    p_export.add_argument("target", nargs="?", default=None, help="job id 或输入文件路径")
    p_export.add_argument("--all", action="store_true", help="导出数据库里全部 job")
    p_export.add_argument("--limit", type=int, default=None, help="--all 时最多处理 N 个")
    p_export.add_argument("--formats", default="md,srt,txt",
                          help="逗号分隔的格式（默认 md,srt,txt）")
    p_export.add_argument("--force", action="store_true", help="已有文本也覆盖")
    p_export.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_export.set_defaults(func=cmd_export)

    # compose（P10：全文成稿；独立命令，不在 run 全链路里）
    p_compose = subparsers.add_parser(
        "compose",
        help="全文成稿：通读整讲生成课程档案，再分块把 ASR 转写整理为书面讲稿",
        parents=[global_parent],
    )
    p_compose.add_argument("target", nargs="?", default=None, help="job id 或输入文件路径")
    p_compose.add_argument("--all", action="store_true", help="整理数据库里全部 job")
    p_compose.add_argument("--limit", type=int, default=None, help="--all 时最多处理 N 个")
    p_compose.add_argument("--force", action="store_true",
                           help="已有 manuscript.json 也重做（档案/分块缓存仍复用；重做档案请删 compose_card.json）")
    p_compose.add_argument("--no-cache", action="store_true", help="不读分块缓存")
    p_compose.add_argument("--dry-run", action="store_true",
                           help="只显示分块方案，不调用 API")
    p_compose.add_argument("--chunk-chars", type=int, default=None,
                           help="覆盖 compose.chunk_chars")
    p_compose.add_argument("--concurrency", type=int, default=None,
                           help="覆盖 compose.concurrency（1–16）")
    p_compose.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_compose.set_defaults(func=cmd_compose)

    # run（P6-T2：单 job 全链路 decode→...→export）
    p_run = subparsers.add_parser(
        "run",
        help="单文件全链路：decode→analyze→denoise→trim→loudness→asr→diarize→correct→export",
        parents=[global_parent],
    )
    p_run.add_argument("target", help="job id 或输入文件路径（未登记的路径会先入库）")
    p_run.add_argument("--force", action="store_true",
                       help="已有产物也重跑（仍复用 decode/analyze 缓存）")
    p_run.add_argument("--resume", action="store_true",
                       help="断点续跑（默认行为，显式声明用）")
    p_run.add_argument("--keep-intermediates", action="store_true", help="保留中间 wav")
    p_run.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_run.set_defaults(func=cmd_run)

    # batch（P6-T2：扫描/登记后批量跑全链路）
    p_batch = subparsers.add_parser(
        "batch",
        help="批量：扫描/登记目录下全部媒体文件，逐个跑全链路（可断点续跑）",
        parents=[global_parent],
    )
    p_batch.add_argument("path", help="输入目录或单个文件")
    p_batch.add_argument("--resume", action="store_true",
                         help="跳过已完成任务（默认开；--force 才重跑）")
    p_batch.add_argument("--limit", type=int, default=None, help="最多跑 N 个")
    p_batch.add_argument("--force", action="store_true", help="已完成任务也重跑")
    p_batch.add_argument("--jobs", type=int, default=1,
                         help="并发数（本版只支持单进程 1，参数保留）")
    p_batch.add_argument("--keep-intermediates", action="store_true", help="保留中间 wav")
    p_batch.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_batch.set_defaults(func=cmd_batch)

    # retry（P6-T2：重置失败阶段及下游后重跑）
    p_retry = subparsers.add_parser(
        "retry", help="重试失败任务：重置失败阶段及其下游后重跑",
        parents=[global_parent],
    )
    p_retry.add_argument("target", help="job id 或输入文件路径")
    p_retry.add_argument("--force", action="store_true", help="重置全部阶段后整体重跑")
    p_retry.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_retry.set_defaults(func=cmd_retry)

    # serve（P7：本地 Web UI）
    p_serve = subparsers.add_parser(
        "serve", help="启动本地 Web UI（FastAPI + uvicorn，只 bind 127.0.0.1）",
        parents=[global_parent],
    )
    p_serve.add_argument("--port", type=int, default=7860, help="端口（默认 7860）")
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="监听地址（默认 127.0.0.1；本工具无鉴权，勿改 0.0.0.0）")
    p_serve.set_defaults(func=cmd_serve)

    # regress（P8-T1：6 文件回归）
    p_regress = subparsers.add_parser(
        "regress",
        help="6 文件回归：默认 --reuse 只读真实产物断言（秒级）；--rerun 独立目录真跑（慢）",
        parents=[global_parent],
    )
    p_regress.add_argument("data_dir", nargs="?", default=None,
                           help="输入目录或文件（rerun 枚举输入；默认 data/）")
    regress_mode = p_regress.add_mutually_exclusive_group()
    regress_mode.add_argument("--reuse", dest="regress_mode", action="store_const",
                              const="reuse",
                              help="只读 out/<id>/report.json + 8 件产物做断言（默认，不跑流水线）")
    regress_mode.add_argument("--rerun", dest="regress_mode", action="store_const",
                              const="rerun",
                              help="用独立 out_p8/work/cache_p8 真跑后断言（慢；禁止写 out/）")
    p_regress.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件")
    p_regress.add_argument("--regress-dir", default=None,
                           help="p8.json/p8_rerun.json 输出目录（默认 work/regress）")
    p_regress.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_regress.set_defaults(func=cmd_regress, regress_mode="reuse")

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
def _status_once(args: argparse.Namespace, cfg: Config, title: str = "") -> int:
    """打印一次状态（``--watch`` 循环体复用）。"""
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
        if title:
            print(title)
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


def cmd_status(args: argparse.Namespace, cfg: Config) -> int:
    if not getattr(args, "watch", False) or getattr(args, "json", False):
        return _status_once(args, cfg)

    import datetime
    import time

    interval = 2.0
    tty = sys.stdout.isatty()
    if tty:
        sys.stdout.write("\x1b[?25l")  # 隐藏光标，避免闪
    try:
        while True:
            if tty:
                sys.stdout.write("\x1b[2J\x1b[H")  # 清屏 + 回左上角
            _status_once(
                args, cfg,
                title=f"[watch] {datetime.datetime.now().strftime('%H:%M:%S')}"
                      f"  每 {interval:g}s 刷新，Ctrl-C 退出",
            )
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[watch] 已退出")
        return EXIT_OK
    finally:
        if tty:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()


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


# ---------------------------------------------------------------- 转写/纠错/导出（P4/P5）
def _resolve_stage_targets(
    args: argparse.Namespace, store: Storage, action: str
) -> List[Tuple[str, Path]]:
    """P4/P5 子命令共用的目标解析（``<id|path>`` / ``--all`` / ``--limit``）。"""
    if getattr(args, "all", False):
        jobs = store.list_jobs()
        if getattr(args, "limit", None):
            jobs = jobs[: args.limit]
        return [(job.id, Path(job.source)) for job in jobs]
    if not getattr(args, "target", None):
        raise ConfigError(f"请给出 job id / 文件路径，或用 --all {action}全部任务")
    target = args.target
    job = store.get_job(target)
    if job is not None:
        return [(job.id, Path(job.source))]
    path = Path(target)
    if path.is_file():
        existing = store.get_job_by_source(path)
        return [(existing.id if existing else make_job_id(path), path)]
    raise ConfigError(f"找不到 job 或文件：{target}")


def _stage_job_done(store: Storage, job_id: str, stage: str) -> None:
    """阶段成功后更新 job（P3 已把 job 置 done，这里只推进 stage）。"""
    if store.get_job(job_id) is not None:
        store.update_job(job_id, status=STATUS_DONE, stage=stage)


def _transcribe_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    report = result.get("report") or {}
    asr = result.get("asr") or report.get("asr") or {}
    speakers = result.get("speakers") or report.get("speakers") or {}
    return {
        "job_id": result.get("job_id"),
        "ok": bool(result.get("ok")),
        "skipped": bool(result.get("skipped")),
        "stage": result.get("stage"),
        "error": result.get("error", ""),
        "n_segments": asr.get("n_segments", result.get("n_segments")),
        "chars": asr.get("chars", result.get("chars")),
        "n_empty": asr.get("n_empty", result.get("n_empty")),
        "rtf": asr.get("rtf", result.get("rtf")),
        "elapsed_s": asr.get("elapsed_s", result.get("elapsed_s")),
        "speakers": speakers,
        "diarize": result.get("diarize"),
        "transcript": result.get("transcript"),
    }


def _transcribe_row(job_id: str, summary: Dict[str, Any]) -> str:
    if not summary["ok"]:
        return (f"[FAIL] {_pad(job_id, 40)} {str(summary.get('stage')):9s} "
                f"{(summary.get('error') or '')[:110]}")
    tag = "SKIP" if summary.get("skipped") else "OK  "
    n = summary.get("n_segments")
    chars = summary.get("chars")
    empty = summary.get("n_empty")
    rtf = summary.get("rtf")
    elapsed = summary.get("elapsed_s")
    spk = summary.get("speakers") or {}
    return (
        f"[{tag}] {_pad(job_id, 40)} "
        f"segs={n if n is not None else '-':>5} "
        f"chars={chars if chars is not None else '-':>6} "
        f"empty={empty if empty is not None else '-':>3} "
        f"rtf={float(rtf) if rtf is not None else float('nan'):.4f} "
        f"spk={spk.get('method', '-')}/{spk.get('n', '-')} "
        f"{float(elapsed) if elapsed is not None else float('nan'):6.1f}s"
    )


def cmd_transcribe(args: argparse.Namespace, cfg: Config) -> int:
    from .pipeline.asr import transcribe_job

    store = make_storage(cfg)
    store.init_db()
    try:
        targets = _resolve_stage_targets(args, store, "转写")
    except ConfigError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        store.close()
        return EXIT_USAGE
    if not targets:
        print("[警告] 没有待转写的任务", file=sys.stderr)
        store.close()
        return EXIT_OK

    summaries: List[Dict[str, Any]] = []
    failures = 0
    diar_mode = str(cfg.get("diar.mode", "none") or "none")
    for job_id, source in targets:
        result = transcribe_job(
            job_id, cfg,
            force=args.no_cache,
            merge_gap_s=args.merge_gap,
            on_stage=_stage_recorder(store, job_id),
        )
        if result.get("ok") and diar_mode != "none":
            try:
                from .pipeline.diarize import diarize_job

                result["diarize"] = diarize_job(
                    job_id, cfg, force=args.no_cache,
                    on_stage=_stage_recorder(store, job_id),
                )
            except Exception as exc:  # diar 失败不阻塞转写产物
                log.warning("说话人阶段失败（不影响 transcript）：%s", exc)
                result["diarize"] = {"ok": False, "error": str(exc)}
        summary = _transcribe_summary(result)
        summaries.append(summary)
        if result.get("ok"):
            _stage_job_done(store, job_id, "diarize" if diar_mode != "none" else "asr")
            if not args.json:
                print(_transcribe_row(job_id, summary))
        else:
            failures += 1
            if store.get_job(job_id) is not None:
                store.fail_job(job_id, result.get("error") or "转写失败")
            if not args.json:
                print(_transcribe_row(job_id, summary))

    store.close()
    if args.json:
        _json_dump({"ok": failures == 0, "n": len(summaries), "failures": failures,
                    "results": summaries})
    else:
        ok_n = len(summaries) - failures
        print(f"\n转写完成：成功 {ok_n} / {len(summaries)}"
              + (f"，失败 {failures}" if failures else ""))
    return EXIT_OK if failures == 0 else EXIT_RUNTIME


def _correct_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    report = result.get("report") or {}
    corr = result.get("correct") or report.get("correct") or {}
    return {
        "job_id": result.get("job_id"),
        "ok": bool(result.get("ok")),
        "skipped": bool(result.get("skipped")),
        "stage": result.get("stage"),
        "error": result.get("error", ""),
        "enabled": corr.get("enabled", result.get("enabled")),
        "chunks": corr.get("chunks", result.get("chunks")),
        "edits": corr.get("edits", result.get("edits")),
        "rejected": corr.get("rejected", result.get("rejected")),
        "cost_cny": corr.get("cost_cny", result.get("cost_cny")),
        "elapsed_s": corr.get("elapsed_s", result.get("elapsed_s")),
    }


def _correct_row(job_id: str, summary: Dict[str, Any]) -> str:
    if not summary["ok"]:
        return (f"[FAIL] {_pad(job_id, 40)} {str(summary.get('stage')):9s} "
                f"{(summary.get('error') or '')[:110]}")
    tag = "SKIP" if summary.get("skipped") else "OK  "
    return (
        f"[{tag}] {_pad(job_id, 40)} enabled={str(bool(summary.get('enabled'))):5s} "
        f"chunks={summary.get('chunks') if summary.get('chunks') is not None else '-':>3} "
        f"edits={summary.get('edits') if summary.get('edits') is not None else '-':>4} "
        f"rejected={summary.get('rejected') if summary.get('rejected') is not None else '-':>2} "
        f"cost=¥{float(summary.get('cost_cny') or 0.0):.4f} "
        f"{float(summary.get('elapsed_s') or 0.0):6.1f}s"
    )


def cmd_correct(args: argparse.Namespace, cfg: Config) -> int:
    from .pipeline.correct import correct_job

    store = make_storage(cfg)
    store.init_db()
    try:
        targets = _resolve_stage_targets(args, store, "纠错")
    except ConfigError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        store.close()
        return EXIT_USAGE
    if not targets:
        print("[警告] 没有待纠错的任务", file=sys.stderr)
        store.close()
        return EXIT_OK

    summaries: List[Dict[str, Any]] = []
    failures = 0
    for job_id, source in targets:
        result = correct_job(
            job_id, cfg,
            force=args.no_cache,
            on_stage=_stage_recorder(store, job_id),
        )
        summary = _correct_summary(result)
        summaries.append(summary)
        if result.get("ok"):
            _stage_job_done(store, job_id, "correct")
            if not args.json:
                print(_correct_row(job_id, summary))
        else:
            failures += 1
            if store.get_job(job_id) is not None:
                store.fail_job(job_id, result.get("error") or "纠错失败")
            if not args.json:
                print(_correct_row(job_id, summary))

    store.close()
    if args.json:
        _json_dump({"ok": failures == 0, "n": len(summaries), "failures": failures,
                    "results": summaries})
    else:
        ok_n = len(summaries) - failures
        print(f"\n纠错完成：成功 {ok_n} / {len(summaries)}"
              + (f"，失败 {failures}" if failures else ""))
    return EXIT_OK if failures == 0 else EXIT_RUNTIME


# ---------------------------------------------------------------- 全文成稿（P10）
def cmd_compose(args: argparse.Namespace, cfg: Config) -> int:
    from .pipeline.compose import ComposeError, compose_job

    store = make_storage(cfg)
    store.init_db()
    try:
        targets = _resolve_stage_targets(args, store, "成稿")
    except ConfigError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        store.close()
        return EXIT_USAGE
    store.close()
    if not targets:
        print("[警告] 没有待成稿的任务", file=sys.stderr)
        return EXIT_OK

    summaries: List[Dict[str, Any]] = []
    failures = 0
    for job_id, _source in targets:
        progress = None
        if not args.json:
            def progress(done: int, total: int, result: Dict[str, Any],
                         _job: str = job_id) -> None:
                tag = "OK" if result.get("ok") else ("FB" if result.get("fallback") else "FAIL")
                print(f"  {_pad(_job, 36)} chunk {done:>2}/{total:<2} "
                      f"{result.get('start', '-')}-{result.get('end', '-')} "
                      f"[{tag}] {float(result.get('elapsed_s') or 0):5.1f}s", flush=True)
        try:
            result = compose_job(
                job_id, cfg,
                force=args.force,
                no_cache=args.no_cache,
                dry_run=args.dry_run,
                chunk_chars=args.chunk_chars,
                concurrency=args.concurrency,
                on_progress=progress,
            )
        except ComposeError as exc:
            result = {"ok": False, "job_id": job_id, "error": str(exc)}
        except KeyboardInterrupt:
            raise
        summary = _compose_summary(result)
        summaries.append(summary)
        if not result.get("ok"):
            failures += 1
        if not args.json:
            print(_compose_row(job_id, summary))
        if args.dry_run:
            continue

    if args.json:
        _json_dump({"ok": failures == 0, "n": len(summaries), "failures": failures,
                    "results": summaries})
    else:
        ok_n = len(summaries) - failures
        print(f"\n成稿完成：成功 {ok_n} / {len(summaries)}"
              + (f"，失败 {failures}" if failures else ""))
    return EXIT_OK if failures == 0 else EXIT_RUNTIME


def _compose_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    stats = result.get("stats") or {}
    return {
        "job_id": result.get("job_id"),
        "ok": bool(result.get("ok")),
        "skipped": bool(result.get("skipped")),
        "dry_run": bool(result.get("dry_run")),
        "error": result.get("error", ""),
        "manuscript": result.get("manuscript"),
        "n_chunks": stats.get("n_chunks", result.get("n_chunks")),
        "n_paragraphs": stats.get("n_paragraphs"),
        "chars": stats.get("manuscript_chars"),
        "ratio": stats.get("char_ratio"),
        "fallback": stats.get("n_fallback", 0),
        "cost_cny": result.get("cost_cny", stats.get("cost_cny", 0.0)),
        "elapsed_s": result.get("elapsed_s", stats.get("elapsed_s")),
    }


def _compose_row(job_id: str, summary: Dict[str, Any]) -> str:
    if not summary["ok"]:
        return (f"[FAIL] {_pad(job_id, 40)} {str(summary.get('error') or '')[:110]}")
    if summary.get("dry_run"):
        return f"[DRY ] {_pad(job_id, 40)} chunks={summary.get('n_chunks') or 0}"
    tag = "SKIP" if summary.get("skipped") else "OK  "
    return (
        f"[{tag}] {_pad(job_id, 40)} chunks={summary.get('n_chunks') or 0:>2} "
        f"paras={summary.get('n_paragraphs') or 0:>3} "
        f"chars={summary.get('chars') or 0:>6} "
        f"ratio={float(summary.get('ratio') or 0):.2f} "
        f"fallback={summary.get('fallback') or 0} "
        f"cost=¥{float(summary.get('cost_cny') or 0.0):.2f} "
        f"{float(summary.get('elapsed_s') or 0.0):6.1f}s"
    )


def _export_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": result.get("job_id"),
        "ok": bool(result.get("ok")),
        "skipped": bool(result.get("skipped")),
        "stage": result.get("stage"),
        "error": result.get("error", ""),
        "formats": result.get("formats"),
        "files": result.get("files"),
        "n_segments": result.get("n_segments"),
        "elapsed_s": result.get("elapsed_s"),
    }


def _export_row(job_id: str, summary: Dict[str, Any]) -> str:
    if not summary["ok"]:
        return (f"[FAIL] {_pad(job_id, 40)} {str(summary.get('stage')):9s} "
                f"{(summary.get('error') or '')[:110]}")
    tag = "SKIP" if summary.get("skipped") else "OK  "
    formats = summary.get("formats")
    if isinstance(formats, (list, tuple)):
        formats = ",".join(str(x) for x in formats)
    return (
        f"[{tag}] {_pad(job_id, 40)} "
        f"segs={summary.get('n_segments') if summary.get('n_segments') is not None else '-':>5} "
        f"fmt={formats or '-':14s} "
        f"{float(summary.get('elapsed_s') or 0.0):6.2f}s"
    )


def cmd_export(args: argparse.Namespace, cfg: Config) -> int:
    from .pipeline.export import export_text

    formats = [f.strip().lower() for f in str(args.formats or "").split(",") if f.strip()]
    if not formats:
        print("[错误] --formats 不能为空（可选 md,srt,txt）", file=sys.stderr)
        return EXIT_USAGE
    unknown = [f for f in formats if f not in ("md", "srt", "txt", "json")]
    if unknown:
        print(f"[错误] 不支持的格式：{', '.join(unknown)}（可选 md,srt,txt）", file=sys.stderr)
        return EXIT_USAGE

    store = make_storage(cfg)
    store.init_db()
    try:
        targets = _resolve_stage_targets(args, store, "导出")
    except ConfigError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        store.close()
        return EXIT_USAGE
    if not targets:
        print("[警告] 没有待导出的任务", file=sys.stderr)
        store.close()
        return EXIT_OK

    summaries: List[Dict[str, Any]] = []
    failures = 0
    for job_id, source in targets:
        result = export_text(
            job_id, cfg,
            formats=formats,
            force=args.force,
            on_stage=_stage_recorder(store, job_id),
        )
        summary = _export_summary(result)
        summaries.append(summary)
        if result.get("ok"):
            _stage_job_done(store, job_id, "export")
            if not args.json:
                print(_export_row(job_id, summary))
        else:
            failures += 1
            if store.get_job(job_id) is not None:
                store.fail_job(job_id, result.get("error") or "导出失败")
            if not args.json:
                print(_export_row(job_id, summary))

    store.close()
    if args.json:
        _json_dump({"ok": failures == 0, "n": len(summaries), "failures": failures,
                    "results": summaries})
    else:
        ok_n = len(summaries) - failures
        print(f"\n导出完成：成功 {ok_n} / {len(summaries)}"
              + (f"，失败 {failures}" if failures else ""))
    return EXIT_OK if failures == 0 else EXIT_RUNTIME


# ---------------------------------------------------------------- run/batch/retry（P6-T2）
def _short_id(job_id: str, width: int = 28) -> str:
    """终端展示用短 id（超长截断，保留可读短名前缀）。"""
    text = str(job_id)
    return text if len(text) <= width else text[: width - 1] + "…"


def _resolve_job_target(
    args: argparse.Namespace, store: Storage, cfg: Config, action: str
) -> Tuple[str, Path]:
    """``<job id|path>`` → ``(job_id, source)``；未登记的**文件**路径先入库。

    只认 DB 里已有的 job id；目录请用 ``batch``。
    """
    target = str(getattr(args, "target", "") or "").strip()
    if not target:
        raise ConfigError(f"请给出 job id 或输入文件路径（{action}）")
    job = store.get_job(target)
    if job is not None:
        return job.id, Path(job.source)
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_dir():
        raise ConfigError(f"{action} 只接受单个文件；目录请用 batch：{path}")
    if not path.is_file():
        raise ConfigError(f"找不到 job id 或输入文件：{target}")
    job, created = store.upsert_job(path, config_hash=cfg.config_hash)
    if created:
        log.info("新登记 job：%s ← %s", job.id, path)
    return job.id, Path(job.source)


def _event_detail(stage: str, status: str, metrics: Dict[str, Any]) -> str:
    """进度行的附加信息（无则空串）。"""
    data = metrics or {}
    if status == "failed":
        return str(data.get("error") or "")[:120]
    if stage == "asr":
        return f"segs={data.get('n_segments')} chars={data.get('chars')}"
    if stage == "diarize":
        return f"mode={data.get('mode')} speakers={data.get('n')}"
    if stage == "correct":
        return f"enabled={data.get('enabled')} edits={data.get('edits')} cost=¥{data.get('cost_cny')}"
    if stage == "export":
        tp = data.get("tp_dbtp")
        asl = data.get("asl_out_dbfs")
        if tp is not None or asl is not None:
            return f"asl_out={asl} tp={tp}"
        return f"fmt={','.join(str(x) for x in (data.get('formats') or []))}"
    return ""


def _run_progress(prefix: str = "  "):
    """run_job 的 on_event(stage, status, metrics) → 终端进度行。"""

    def on_event(stage: str, status: str, metrics: Dict[str, Any]) -> None:
        mark = "ok  " if status == "done" else "FAIL"
        print(f"{prefix}{stage:9s} {mark} {_event_detail(stage, status, metrics)}".rstrip(),
              flush=True)

    return on_event


def _batch_progress():
    """batch_run 的 on_event(job_id, stage, status, metrics) → 终端进度行。"""

    def on_event(job_id: str, stage: str, status: str, metrics: Dict[str, Any]) -> None:
        mark = "ok  " if status == "done" else "FAIL"
        print(f"  [{_short_id(job_id, 24):24s}] {stage:9s} {mark} "
              f"{_event_detail(stage, status, metrics)}".rstrip(), flush=True)

    return on_event


def _run_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """run_job 结果的精简摘要（完整报告在 out/<id>/report.json）。"""
    report = result.get("report") or {}
    asr = report.get("asr") or {}
    trim = report.get("trim") or {}
    timing = result.get("timings") or report.get("timing") or {}
    artifacts = result.get("artifacts") or []
    return {
        "job_id": result.get("job_id"),
        "ok": bool(result.get("ok")),
        "skipped": bool(result.get("skipped")),
        "locked": bool(result.get("locked")),
        "stage": result.get("stage"),
        "error": result.get("error", ""),
        "clean_dur_s": trim.get("clean_dur_s"),
        "n_segments": asr.get("n_segments"),
        "chars": asr.get("chars"),
        "n_artifacts": len(artifacts),
        "total_s": timing.get("total_s"),
        "timings": timing,
    }


def _run_row(job_id: str, summary: Dict[str, Any]) -> str:
    if summary.get("locked"):
        return f"[LOCK] {_pad(job_id, 40)} 另一个进程正在跑该 job（work/locks）"
    if not summary.get("ok"):
        return (f"[FAIL] {_pad(job_id, 40)} {str(summary.get('stage')):9s} "
                f"{(summary.get('error') or '')[:110]}")
    tag = "SKIP" if summary.get("skipped") else "OK  "
    total = summary.get("total_s")
    total_s = f"{float(total):8.1f}s" if isinstance(total, (int, float)) else "       -"
    return (
        f"[{tag}] {_pad(job_id, 40)} "
        f"clean={float(summary.get('clean_dur_s') or 0.0) / 60.0:6.1f}min "
        f"segs={summary.get('n_segments') if summary.get('n_segments') is not None else '-':>5} "
        f"chars={summary.get('chars') if summary.get('chars') is not None else '-':>7} "
        f"artifacts={summary.get('n_artifacts')}/8 {total_s}"
    )


def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    """单 job 全链路（编排器 run_job）。"""
    from .pipeline.orchestrator import run_job

    store = make_storage(cfg)
    store.init_db()
    try:
        job_id, source = _resolve_job_target(args, store, cfg, "run")
    except (ConfigError, StorageError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        store.close()
        return EXIT_USAGE
    store.close()  # 关掉自己的连接：run_job 内部会新建 Storage（SQLite 不跨线程复用）

    out_dir = cfg.path("out_dir")
    cache_dir = cfg.path("cache_dir")
    if not args.json:
        print(f"run {job_id}")
        print(f"  source={source}")
        print(f"  out={out_dir}  cache={cache_dir}")
    result = run_job(
        job_id, source, cfg,
        force=args.force,
        out_dir=out_dir,
        cache_dir=cache_dir,
        keep_intermediates=args.keep_intermediates,
        on_event=None if args.json else _run_progress(),
    )
    summary = _run_summary(result)
    if args.json:
        _json_dump({"ok": bool(result.get("ok")), "summary": summary, "result": result})
    else:
        print(_run_row(job_id, summary))
    return EXIT_OK if result.get("ok") else EXIT_RUNTIME


def cmd_batch(args: argparse.Namespace, cfg: Config) -> int:
    """目录批量（编排器 batch_run）：先 discover/upsert，再逐个跑全链路。"""
    from .pipeline.orchestrator import batch_run

    if int(getattr(args, "jobs", 1) or 1) > 1:
        print("[提示] 本版单进程执行（--jobs 参数保留，不做进程池）", file=sys.stderr)
    scan_cfg = cfg.section("scan")
    try:
        files = discover_files(
            args.path,
            extensions=scan_cfg.get("extensions"),
            recursive=bool(scan_cfg.get("recursive", True)),
            min_bytes=int(scan_cfg.get("min_bytes", 0) or 0),
            max_bytes=int(scan_cfg.get("max_bytes", 0) or 0),
        )
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not files:
        print(f"[警告] {args.path} 下没有发现可处理的媒体文件（扩展名：{scan_cfg.get('extensions')}）",
              file=sys.stderr)
        return EXIT_OK

    store = make_storage(cfg)
    store.init_db()
    targets: List[Tuple[str, str]] = []
    created = existing = 0
    for path in files:
        job, is_new = store.upsert_job(path, config_hash=cfg.config_hash)
        created += int(is_new)
        existing += int(not is_new)
        targets.append((job.id, job.source))
    store.close()
    if args.limit:
        targets = targets[: int(args.limit)]

    resume = bool(args.resume or cfg.get("batch.resume", True)) and not args.force
    if not args.json:
        print(f"batch {args.path}：发现 {len(files)} 个文件（新登记 {created}，已存在 {existing}），"
              f"本轮处理 {len(targets)} 个（resume={resume} force={args.force}）")
        sys.stdout.flush()

    result = batch_run(
        targets, cfg,
        resume=resume,
        limit=args.limit,
        force=args.force,
        keep_intermediates=args.keep_intermediates,
        on_event=None if args.json else _batch_progress(),
    )
    failures = result.get("failures") or []
    skipped = result.get("skipped") or []
    stopped = result.get("stopped_reason") or ""
    results = result.get("results") or []

    if args.json:
        _json_dump(result)
    else:
        for job_id in skipped:
            reason = (result.get("skipped_reasons") or {}).get(job_id, "skipped")
            print(f"[SKIP] {_pad(job_id, 40)} {reason}")
        for item in results:
            if not item.get("ok"):
                print(_run_row(str(item.get("job_id") or ""), _run_summary(item)))
        done_n = sum(1 for r in results if r.get("ok") and not r.get("skipped"))
        print(f"\nbatch 完成：成功 {done_n}，跳过 {len(skipped)}，失败 {len(failures)}")
        if stopped:
            print(f"[停止] stopped_reason={stopped}")
    return EXIT_OK if not failures and not stopped else EXIT_RUNTIME


def cmd_retry(args: argparse.Namespace, cfg: Config) -> int:
    """失败任务重试：重置失败阶段及其下游后重跑（--force 则整体重跑）。"""
    from .pipeline.orchestrator import run_job

    store = make_storage(cfg)
    store.init_db()
    try:
        job_id, source = _resolve_job_target(args, store, cfg, "retry")
        job = store.require_job(job_id)
        if args.force:
            reset_from = STAGES[0]
        else:
            pending = [st for st in STAGES
                       if store.stage_map(job_id).get(st) not in (STATUS_DONE, STATUS_SKIPPED)]
            reset_from = pending[0] if pending else STAGES[-1]
        reset_n = store.reset_stage(job_id, reset_from, downstream=True)
    except (ConfigError, StorageError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        store.close()
        return EXIT_USAGE
    store.close()
    if not args.json:
        print(f"retry {job_id}：从 {reset_from} 阶段重跑（重置 {reset_n} 个阶段；"
              f"原状态 {job.status}/{job.stage or '-'}）")
        sys.stdout.flush()

    result = run_job(
        job_id, source, cfg,
        force=bool(args.force),
        out_dir=cfg.path("out_dir"),
        cache_dir=cfg.path("cache_dir"),
        on_event=None if args.json else _run_progress(),
    )
    summary = _run_summary(result)
    if args.json:
        _json_dump({"ok": bool(result.get("ok")), "summary": summary, "result": result})
    else:
        print(_run_row(job_id, summary))
    return EXIT_OK if result.get("ok") else EXIT_RUNTIME


def cmd_serve(args: argparse.Namespace, cfg: Config) -> int:
    """启动本地 Web UI（FastAPI + uvicorn，默认 127.0.0.1:7860）。"""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("[错误] 缺少 uvicorn：work\\venv\\Scripts\\python -m pip install uvicorn",
              file=sys.stderr)
        return EXIT_USAGE
    from .server import create_app

    host = str(getattr(args, "host", None) or "127.0.0.1")
    port = int(getattr(args, "port", 7860) or 7860)
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("Web UI 无鉴权，%s 不是回环地址，请确认网络环境安全", host)
    app = create_app(cfg)
    url = f"http://{host}:{port}"
    print(f"Web UI: {url}", flush=True)
    print(f"数据库：{cfg.path('db_path')}", flush=True)
    print(f"上传目录：{cfg.path('out_dir') / 'uploads'}", flush=True)
    print("Ctrl-C 退出", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return EXIT_OK


def cmd_regress(args: argparse.Namespace, cfg: Config) -> int:
    """P8-T1 六文件回归：默认 reuse 只读真实产物；rerun 用独立目录真跑（绝不写 out/）。"""
    from .tools.regress import _print_check_summary, _write_p8_payload
    from .tools.regress import check_regression, run_regression

    mode = str(getattr(args, "regress_mode", None) or "reuse")
    data_dir = getattr(args, "data_dir", None) or None
    out_dir = getattr(args, "out_dir", None) or None
    cache_dir = getattr(args, "cache_dir", None) or None
    regress_dir = Path(getattr(args, "regress_dir", None) or "work/regress")
    if not regress_dir.is_absolute():
        regress_dir = (Path.cwd() / regress_dir)

    base_cfg = cfg
    if mode == "rerun" and (out_dir or getattr(args, "db_path", None)):
        # rerun 的 out_dir 不能同时被全局 --out 覆盖进 cfg，否则会被安全校验误判为「写 out/」；
        # 这里按 config+preset 重新加载一份未覆盖 out 的 cfg，再由 out_dir/cache_dir 参数显式指定。
        base_cfg = load_config(
            config_path=getattr(args, "config", None),
            preset=getattr(args, "preset", None),
        )

    try:
        result = run_regression(
            data_dir, base_cfg, mode=mode,
            out_dir=out_dir, cache_dir=cache_dir, limit=args.limit,
        )
    except ValueError as exc:  # rerun 目录安全校验 / 非法 mode
        print(f"[参数错误] {exc}", file=sys.stderr)
        return EXIT_USAGE

    baseline = regress_dir / "audio.csv"
    check = check_regression(
        result,
        baseline_csv=baseline if baseline.is_file() else None,
        scan_dir=regress_dir.parent / "scan",
    )
    out_path = regress_dir / ("p8_rerun.json" if mode == "rerun" else "p8.json")
    payload = _write_p8_payload(result=result, check=check, out_path=out_path)
    if args.json:
        _json_dump(payload)
    else:
        _print_check_summary(result, check, out_path)
    return EXIT_OK if check.get("ok") else EXIT_RUNTIME


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
    # --json 模式：日志只进文件，stdout 保持纯 JSON（脚本可直接解析）
    json_mode = bool(getattr(args, "json", False))
    setup_logging(cfg, level=level, console=False if json_mode else None, force=json_mode)

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
