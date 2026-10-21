# -*- coding: utf-8 -*-
"""日志：同时写文件（UTF-8、按大小轮转）与 stdout，重复调用不叠加 handler。

用法
----
>>> from app.logging_setup import setup_logging, get_logger
>>> log = setup_logging(cfg)          # 依据 cfg.logging 配置
>>> get_logger(__name__).info("开始")

坑
--
Windows 控制台编码不是 UTF-8 时中文会乱码：这里主动 reconfigure stdout/stderr，
文件 handler 一律 ``encoding="utf-8"``。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

#: 我们自己的 handler 打这个标记，便于幂等移除
_HANDLER_TAG = "_audio_processer_handler"

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

ROOT_LOGGER_NAME = "app"

_configured: dict[str, Any] = {"file": None, "level": None}


class _Tagger(logging.Filter):
    """给 handler 打标记（也可以直接用 setattr，这里用 Filter 保证 stdout 也一致）。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - logging 协议
        return True


def _ensure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            # 被重定向到不支持 reconfigure 的对象时忽略即可
            pass


def _remove_ours(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - 关闭失败不影响主流程
                pass


def _make_formatter(fmt: Optional[str] = None, datefmt: Optional[str] = None) -> logging.Formatter:
    return logging.Formatter(fmt or DEFAULT_FORMAT, datefmt or DEFAULT_DATEFMT)


def attach_file_handler(
    logger: logging.Logger,
    path: str | Path,
    level: Optional[int | str] = None,
    rotate_mb: float = 10.0,
    backups: int = 5,
    fmt: Optional[str] = None,
    tagged: bool = True,
) -> RotatingFileHandler:
    """给 logger 挂一个 UTF-8 轮转文件 handler；调用方负责 detach。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(p),
        maxBytes=max(0, int(rotate_mb * 1024 * 1024)),
        backupCount=max(0, int(backups)),
        encoding="utf-8",
        delay=False,
    )
    if level is not None:
        handler.setLevel(_to_level(level))
    handler.setFormatter(_make_formatter(fmt))
    if tagged:
        setattr(handler, _HANDLER_TAG, True)
        handler.addFilter(_Tagger())
    logger.addHandler(handler)
    return handler


def detach_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    logger.removeHandler(handler)
    try:
        handler.close()
    except Exception:  # pragma: no cover
        pass


def _to_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    value = logging.getLevelName(str(level).upper())
    if isinstance(value, int):
        return value
    raise ValueError(f"未知日志级别：{level!r}")


def setup_logging(
    cfg: Any = None,
    log_file: Optional[str | Path] = None,
    level: Optional[int | str] = None,
    console: Optional[bool] = None,
    force: bool = False,
) -> logging.Logger:
    """配置 ``app`` 根 logger（子 logger 通过 propagate 继承）。

    参数
    ----
    cfg      : ``app.config.Config``；为 None 时只输出到 stdout
    log_file : 显式指定日志文件（覆盖 cfg.logging.file_name）
    level    : 日志级别（覆盖 cfg.logging.level）
    console  : 是否同时输出 stdout（覆盖 cfg.logging.console）
    force    : True 时强制重建 handler（测试/重配用）

    返回 ``app`` logger。
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # 由各 handler 过滤级别
    logger.propagate = False

    file_path: Optional[Path] = None
    rotate_mb = 10.0
    backups = 5
    effective_level: Any = level
    effective_console = True

    if cfg is not None:
        logging_sec = cfg.section("logging") if hasattr(cfg, "section") else (cfg.get("logging") or {})
        effective_level = level or logging_sec.get("level", "INFO")
        effective_console = logging_sec.get("console", True) if console is None else console
        rotate_mb = float(logging_sec.get("rotate_mb", 10) or 10)
        backups = int(logging_sec.get("rotate_backups", 5) or 5)
        if log_file is not None:
            file_path = Path(log_file)
        elif logging_sec.get("file_name"):
            file_path = cfg.path("logs_dir") / str(logging_sec["file_name"])
    else:
        effective_level = effective_level or "INFO"
        effective_console = True if console is None else console
        file_path = Path(log_file) if log_file else None

    level_int = _to_level(effective_level)

    # 幂等：同配置直接复用
    key_file = str(file_path) if file_path else None
    if (
        not force
        and _configured["file"] == key_file
        and _configured["level"] == level_int
        and any(getattr(h, _HANDLER_TAG, False) for h in logger.handlers)
    ):
        return logger

    _remove_ours(logger)
    _ensure_utf8_streams()

    if effective_console:
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setLevel(level_int)
        stream.setFormatter(_make_formatter())
        setattr(stream, _HANDLER_TAG, True)
        logger.addHandler(stream)

    if file_path is not None:
        handler = attach_file_handler(
            logger, file_path, level=level_int, rotate_mb=rotate_mb, backups=backups
        )
        handler.setLevel(level_int)

    _configured["file"] = key_file
    _configured["level"] = level_int
    return logger


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """取 logger；``__name__`` 以 app 开头的保持原名，其它归到 app 子树下。"""
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith("app") or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def shutdown_logging() -> None:
    """移除并关闭本模块挂上去的全部 handler（测试收尾用）。"""
    _remove_ours(logging.getLogger(ROOT_LOGGER_NAME))
    _configured["file"] = None
    _configured["level"] = None
