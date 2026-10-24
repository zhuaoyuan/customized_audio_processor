# -*- coding: utf-8 -*-
"""P7-T1 本地 Web API：FastAPI 应用工厂 + 固定端点（扫描 / 上传 / 运行 / 详情 / 音频 /
SSE / 改字 / 导出 / 静态资源）。

设计要点
--------
- **应用工厂**：``create_app(cfg=None)`` 便于测试注入临时配置；模块级 ``app = create_app()``
  在导入时执行 ``load_config()``，供 ``python -m app.cli serve`` 与 ``uvicorn app.server:app``
  直接使用。FastAPI 实例只 bind 127.0.0.1 由启动方（CLI/uvicorn）负责，本模块不监听端口。
- **每个请求新建 Storage**：SQLite 是 WAL 模式，连接**不跨请求/线程复用**；后台线程自己再开
  一个 Storage（P6 编排器内部亦如此约定）。
- **延迟 import 编排器与 CLI**：``app.pipeline.orchestrator``（P6 并行开发）与
  ``app.cli.discover_files`` 都只在调用点 import；服务模块级 import 它们会拖慢启动、
  并让 P7 在 P6 未落地时完全不可用。测试可 monkeypatch
  ``app.pipeline.orchestrator.run_job``（本模块取的是**模块属性**，不是函数别名）。
- **单 worker + per-job 锁**：进程内 ``running`` 集合 + 每个 job 一把 ``threading.Lock``，
  重复 run 返回 409；跨进程互斥由 ``orchestrator.run_job`` 自己抢 ``work/locks/<id>.lock``。
- **安全**：所有 job id 先查 DB（未知一律 404），产物/导出只允许 8 件已知文件名，禁止路径
  穿越；上传文件名取 basename、替换 ``<>:"/\\|?*`` 与控制字符、去前导点，上限 2 GiB。
- **JSON 中文**：Starlette ``JSONResponse`` 默认 ``ensure_ascii=False``，中文不转义。

端点一览（契约固定，不得改名）
------------------------------
``GET  /api/health``                         健康检查
``GET  /api/jobs``                           汇总 + 任务列表
``POST /api/jobs``                           JSON ``{"path": ...}`` 扫描登记 / 裸流上传
``POST /api/jobs/{id}/run``                  后台线程跑编排器（202 / 404 / 409）
``POST /api/jobs/{id}/retry``                重置失败阶段后重跑（202 / 404 / 409）
``GET  /api/jobs/{id}``                      详情（job/report/stages/artifacts/transcript/corrections）
``GET  /api/jobs/{id}/artifacts``            8 件已知产物（uploads 不列）
``GET  /api/jobs/{id}/audio``                clean.m4a（Starlette FileResponse 原生 Range）
``GET  /api/events``                         SSE：snapshot + 15 s ``: ping`` 心跳
``POST /api/jobs/{id}/transcript``           人工改字（只改 text）后强制重导出 md/srt/txt
``GET  /api/jobs/{id}/export?format=md|srt|txt``  文本产物下载（UTF-8）
``GET  /`` ``GET /static/{file}``            app/web 静态资源（无前端构建链）
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from . import __version__
from .config import Config, load_config
from .logging_setup import get_logger
from .models import STATUS_RUNNING, STAGES, Job
from .storage import Storage, StorageError

log = get_logger(__name__)

# ---------------------------------------------------------------- 常量
#: 上传裸流大小上限（2 GiB；契约 §5 P7-T1）
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

#: 上传文件名里必须替换掉的字符（Windows 非法字符 + 路径分隔符 + 控制字符）
_UPLOAD_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Windows 保留设备名（仅用于避免 out/uploads 下写出非法文件名）
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

#: 8 件已知产物（契约 §4；uploads 不在此列，绝不列任何目录）
PRODUCT_FILES: Tuple[str, ...] = (
    "clean.m4a",
    "edit.json",
    "report.json",
    "transcript.json",
    "transcript.md",
    "transcript.srt",
    "transcript.txt",
    "corrections.json",
)

#: 文本导出格式 → 下载 MIME（md 用 markdown，srt/txt 用纯文本；均 UTF-8）
_EXPORT_FORMATS: Dict[str, str] = {
    "md": "text/markdown; charset=utf-8",
    "srt": "text/plain; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
}

#: SSE 轮询 DB / 心跳间隔（秒）
SSE_POLL_S = 1.5
SSE_PING_S = 15.0

#: 人工改字单段文本上限（字符）；防止误贴整篇文档
MAX_SEGMENT_TEXT_LEN = 5000

#: 前端静态资源目录（不引入构建链，直接读文件）
WEB_DIR = Path(__file__).resolve().parent / "web"

#: 已确认建过表的 db 路径（Storage.init_db 幂等，但 SSE 每 1.5 s 轮询没必要反复执行）
_DB_READY: Set[str] = set()
_DB_READY_LOCK = threading.Lock()


# ---------------------------------------------------------------- 小工具
def _storage(cfg: Config) -> Storage:
    """新建一个 Storage 并确保表存在（WAL；调用方负责 ``close``）。"""
    store = Storage(cfg.path("db_path"), default_config_hash=cfg.config_hash)
    store.connect()
    key = store.db_path
    with _DB_READY_LOCK:
        ready = key in _DB_READY
    if not ready:
        # 双重检查锁：并发首请求只初始化一次（init_db 幂等，这里只是省重复 DDL）
        with _DB_READY_LOCK:
            if key not in _DB_READY:
                store.init_db()
                _DB_READY.add(key)
    return store


@contextmanager
def _open_storage(cfg: Config) -> Iterator[Storage]:
    """请求级 Storage 上下文：用后即关，绝不跨请求复用连接。"""
    store = _storage(cfg)
    try:
        yield store
    finally:
        store.close()


def _job_dir(cfg: Config, job_id: str) -> Path:
    """``out/<id>``；job_id 已由 DB 校验，这里再做一次防穿越断言。"""
    return cfg.path("out_dir") / job_id


def _set_job_dir_guard(job_id: str) -> None:
    """job id 只允许 DB 里取到的普通名字（防 ``..``/分隔符/控制字符）。"""
    if (
        not job_id
        or job_id in (".", "..")
        or any(ch in job_id for ch in ("/", "\\", "\x00"))
        or any(ord(ch) < 0x20 for ch in job_id)
    ):
        raise HTTPException(status_code=404, detail="非法 job id")


def _require_job(store: Storage, job_id: str) -> Job:
    """所有 job id 必须来自 DB；未知一律 404。"""
    _set_job_dir_guard(job_id)
    try:
        return store.require_job(job_id)
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _read_json(path: Path, default: Any = None, *, expect: Optional[type] = None) -> Any:
    """宽松读 JSON：缺失/非法/类型不符都返回 default（详情页不因坏文件 500）。"""
    if not path.is_file():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        log.warning("读取 JSON 失败（按缺失处理）：%s：%s", path, exc)
        return default
    if expect is not None and not isinstance(data, expect):
        log.warning("JSON 类型不符（按缺失处理）：%s：%s", path, type(data).__name__)
        return default
    return data


def _atomic_write_json(path: Path, payload: Any) -> None:
    """先写同目录临时文件再 ``os.replace``（Windows 同卷原子替换）。

    换行风格沿用原文件（P6/P7 遗留 #2）：读既有文件判断是否 CRLF；原文件是 CRLF 就继续
    写 CRLF，避免改字把 ``transcript.json`` 从 CRLF 规范成 LF。文件不存在时用 LF。
    """
    newline = "\n"
    try:
        if path.is_file() and b"\r\n" in path.read_bytes()[:65536]:
            newline = "\r\n"
    except OSError:  # pragma: no cover - 读失败不影响写入
        newline = "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline=newline) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - 清理失败不影响主错误
            pass
        raise


def _sanitize_upload_name(filename: str) -> str:
    """上传文件名 → 安全 basename。

    规则（契约 §5 P7-T1）：取 basename（同时处理 ``/`` 与 ``\\``）→ 替换
    ``<>:"/\\|?*`` 与控制字符为 ``_`` → 去前导点/空白 → 避开 Windows 保留名。
    """
    raw = (filename or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="上传缺少 filename 参数")
    name = Path(raw.replace("\\", "/")).name
    name = _UPLOAD_BAD_CHARS.sub("_", name)
    name = name.strip().strip(".").strip()
    if len(name) > 180:
        name = name[:180].rstrip(". ") or "upload"
    if not name:
        raise HTTPException(status_code=400, detail=f"上传文件名非法：{filename!r}")
    if Path(name).stem.upper() in _WINDOWS_RESERVED:
        name = f"_{name}"
    return name


def _content_type(request: Request) -> str:
    """取主 Content-Type（去 charset 等参数，转小写）。"""
    raw = request.headers.get("content-type") or ""
    return raw.split(";", 1)[0].strip().lower()


def _orchestrator():
    """延迟 import P6 编排器模块（测试 monkeypatch 模块属性即可生效）。"""
    from .pipeline import orchestrator

    return orchestrator


def _fallback_artifacts(cfg: Config, job_id: str) -> List[Dict[str, Any]]:
    """编排器不可用时的兜底：只扫描 8 件已知产物（不列目录、不列 uploads）。

    ``clean.m4a`` 认可 finalize 后的交付位置（``<out>/output/audio/``）。
    """
    job_dir = _job_dir(cfg, job_id)
    out_root = cfg.path("out_dir")
    report = _read_json(job_dir / "report.json", None, expect=dict)
    from .pipeline.finalize import resolve_product

    items: List[Dict[str, Any]] = []
    for name in PRODUCT_FILES:
        path = resolve_product(job_dir, out_root, name, report=report)
        if path is None:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        items.append(
            {"name": name, "size": int(stat.st_size), "mtime": round(float(stat.st_mtime), 3)}
        )
    return items


def _job_artifacts(cfg: Config, job_id: str) -> List[Dict[str, Any]]:
    """产物列表：优先 ``orchestrator.artifacts``，异常时回退为已知产物扫描。

    只透传白名单里的 8 个名字——编排器/损坏文件都不可能让服务端列出任意路径。
    """
    try:
        raw = _orchestrator().artifacts(job_id, cfg)
    except Exception as exc:  # ImportError / 未实现 / 编排器内部错误
        log.warning("orchestrator.artifacts 不可用，回退扫描已知产物：%s", exc)
        return _fallback_artifacts(cfg, job_id)
    items: List[Dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "")
        if name not in PRODUCT_FILES:
            continue
        mtime = item.get("mtime")
        items.append(
            {
                "name": name,
                "size": int(item.get("size") or 0),
                "mtime": round(float(mtime), 3) if isinstance(mtime, (int, float)) else 0.0,
            }
        )
    return items


def _snapshot(cfg: Config) -> Dict[str, Any]:
    """SSE / 列表共用的任务快照（在 ``asyncio.to_thread`` 里跑，不阻塞事件循环）。"""
    with _open_storage(cfg) as store:
        return {
            "ts": int(time.time()),
            "summary": store.summary(),
            "jobs": store.jobs_snapshot(),
        }


def _summarize_result(result: Any) -> str:
    """后台运行日志用的一行结果摘要。"""
    if isinstance(result, Mapping):
        return (
            f"ok={result.get('ok')} stage={result.get('stage')} "
            f"skipped={result.get('skipped')} error={str(result.get('error') or '')[:200]}"
        )
    return repr(result)[:300]


# ---------------------------------------------------------------- 后台运行
def _make_event_logger(job_id: str):
    """把编排器的 on_event 转成 DEBUG 日志（进度靠 DB/SSE 展示，不在回调里写库）。"""

    def on_event(stage: str, status: str, metrics: Optional[Mapping[str, Any]] = None) -> None:
        log.debug("job %s 事件：stage=%s status=%s", job_id, stage, status)

    return on_event


def _run_worker(
    application: FastAPI,
    cfg: Config,
    job_id: str,
    source: str,
    run_fn: Any,
) -> None:
    """后台线程执行 ``orchestrator.run_job``；异常兜底落 failed，最后释放 per-job 锁。"""
    state = application.state
    try:
        result = run_fn(job_id, source, cfg, on_event=_make_event_logger(job_id))
        log.info("job %s 运行结束：%s", job_id, _summarize_result(result))
    except Exception as exc:  # 编排器契约是「失败不抛」，这里只兜底实现缺陷
        log.exception("job %s 后台运行异常", job_id)
        try:
            with _open_storage(cfg) as store:
                store.fail_job(job_id, f"后台运行异常：{exc}")
        except Exception:  # pragma: no cover - 落库都失败只能记日志
            log.exception("job %s 失败状态落库也失败", job_id)
    finally:
        with state.run_lock:
            state.running.discard(job_id)
            lock = state.job_locks.pop(job_id, None)
        if lock is not None:
            try:
                lock.release()
            except RuntimeError:  # pragma: no cover - 理论上不会重复释放
                pass


def _start_run(application: FastAPI, cfg: Config, job: Job) -> Dict[str, Any]:
    """登记 running + 抢 per-job 锁 + 起后台线程；冲突返回 409，编排器缺失返回 503。"""
    job_id = str(job.id)
    state = application.state
    with state.run_lock:
        if job_id in state.running:
            raise HTTPException(status_code=409, detail=f"job 正在运行：{job_id}")
        lock = state.job_locks.get(job_id)
        if lock is None:
            lock = threading.Lock()
            state.job_locks[job_id] = lock
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail=f"job 正在运行：{job_id}")
        state.running.add(job_id)
    try:
        # 取模块属性（不是函数别名）：测试 monkeypatch run_job 能生效
        run_fn = _orchestrator().run_job
    except Exception as exc:
        with state.run_lock:
            state.running.discard(job_id)
        lock.release()
        raise HTTPException(
            status_code=503,
            detail=f"编排器 app.pipeline.orchestrator 未就绪：{exc}",
        ) from exc
    thread = threading.Thread(
        target=_run_worker,
        args=(application, cfg, job_id, str(job.source), run_fn),
        name=f"job-{job_id}",
        daemon=True,
    )
    try:
        thread.start()
    except Exception as exc:  # pragma: no cover - 线程创建失败极少见
        with state.run_lock:
            state.running.discard(job_id)
        lock.release()
        raise HTTPException(status_code=500, detail=f"后台线程启动失败：{exc}") from exc
    log.info("job %s 已提交后台运行（thread=%s）", job_id, thread.name)
    return {"ok": True, "job_id": job_id, "status": STATUS_RUNNING, "accepted": True}


# ---------------------------------------------------------------- 请求体解析
async def _read_json_body(request: Request) -> Any:
    """POST body → JSON；非法 JSON 统一 400（不暴露 pydantic 422 细节）。"""
    try:
        return await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"请求体必须是合法 JSON：{exc}") from exc


def _parse_transcript_updates(body: Any, n_segments: int) -> List[Tuple[int, str]]:
    """解析改字请求 → [(index, text)]，并校验 index 范围、重复与文本长度。

    支持两种 body：
    - ``{"segments": [{"index": 3, "text": "..."}]}``：按 index 局部更新；
    - 完整 list（与 transcript.json 同长）：按位置更新；
    只取 ``text``，其它字段（text_raw/start/end/orig_*/speaker/flags）一律保留。
    """
    if isinstance(body, list):
        items = body
        full_list = True
    elif isinstance(body, Mapping) and isinstance(body.get("segments"), list):
        items = body["segments"]
        full_list = False
    else:
        raise HTTPException(
            status_code=400,
            detail='请求体应为 {"segments":[{"index":0,"text":"..."}]} 或完整 transcript list',
        )
    if full_list and len(items) != n_segments:
        raise HTTPException(
            status_code=400,
            detail=f"完整 list 长度 {len(items)} 与现有 transcript 段数 {n_segments} 不一致",
        )
    if not items:
        raise HTTPException(status_code=400, detail="没有可更新的段")

    updates: List[Tuple[int, str]] = []
    seen: Set[int] = set()
    for position, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise HTTPException(status_code=400, detail=f"第 {position + 1} 项不是对象")
        if "text" not in item:
            raise HTTPException(status_code=400, detail=f"第 {position + 1} 项缺少 text")
        text = item["text"]
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail=f"第 {position + 1} 项 text 必须是字符串")
        if len(text) > MAX_SEGMENT_TEXT_LEN:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"第 {position + 1} 项 text 长度 {len(text)} 超过上限 "
                    f"{MAX_SEGMENT_TEXT_LEN} 字符"
                ),
            )
        if full_list:
            index = position
        else:
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise HTTPException(
                    status_code=400, detail=f"第 {position + 1} 项 index 必须是整数"
                )
        if not (0 <= index < n_segments):
            raise HTTPException(
                status_code=400, detail=f"第 {position + 1} 项 index={index} 越界（0..{n_segments - 1}）"
            )
        if index in seen:
            raise HTTPException(status_code=400, detail=f"index={index} 重复")
        seen.add(index)
        updates.append((index, text))
    return updates


# ---------------------------------------------------------------- 应用工厂
def create_app(cfg: Optional[Config] = None) -> FastAPI:
    """构造 FastAPI 应用。

    ``cfg`` 为 None 时执行 ``load_config()``（模块级 ``app`` 即此路径）；测试传临时 cfg
    即可把所有产物/DB 指向 tmp_path。每个 handler 用**闭包里的 cfg**，不读全局配置。
    """
    resolved = load_config() if cfg is None else cfg

    application = FastAPI(
        title="audio_processer",
        version=__version__,
        description="课程录音批处理本地 Web API（P7-T1）：扫描/上传/运行/播放/改字/导出",
    )
    # 进程内运行状态：running 集合 + per-job 锁 + 状态锁（不跨 app 实例共享）
    application.state.cfg = resolved
    application.state.running = set()
    application.state.job_locks = {}
    application.state.run_lock = threading.Lock()

    # ------------------------------------------------------------ health / 列表
    @application.get("/api/health")
    def health() -> Dict[str, Any]:
        """健康检查：服务存活即可（不触碰 DB，便于启动探活）。"""
        return {"ok": True, "version": __version__}

    @application.get("/api/jobs")
    def list_jobs() -> Dict[str, Any]:
        """任务汇总 + 列表（直接复用 Storage.summary / jobs_snapshot）。"""
        with _open_storage(resolved) as store:
            return {"summary": store.summary(), "jobs": store.jobs_snapshot()}

    # ------------------------------------------------------------ 扫描 / 上传
    @application.post("/api/jobs")
    async def create_job(request: Request) -> Dict[str, Any]:
        """JSON ``{"path": ...}`` 扫描登记；``application/octet-stream`` 裸流上传。"""
        ctype = _content_type(request)
        if ctype == "application/octet-stream":
            return await _handle_upload(resolved, request)
        if ctype in ("", "application/json"):
            return await _handle_scan(resolved, request)
        raise HTTPException(
            status_code=415,
            detail=f"不支持的 Content-Type：{ctype or '(缺失)'}（JSON 或 application/octet-stream）",
        )

    async def _handle_scan(active_cfg: Config, request: Request) -> Dict[str, Any]:
        """扫描目录/单文件 → discover_files → upsert job（函数内 import app.cli）。"""
        body = await _read_json_body(request)
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail='请求体应为 {"path": "data/"}')
        raw_path = body.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise HTTPException(status_code=400, detail="缺少 path（目录或文件路径）")
        target = Path(raw_path.strip()).expanduser()
        if not target.is_absolute():
            target = active_cfg.root / target
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"路径不存在：{target}")

        from .cli import discover_files  # 禁止模块级 import app.cli（见模块 docstring）

        scan_cfg = active_cfg.section("scan")
        try:
            files = discover_files(
                target,
                extensions=scan_cfg.get("extensions"),
                recursive=bool(scan_cfg.get("recursive", True)),
                min_bytes=int(scan_cfg.get("min_bytes", 0) or 0),
                max_bytes=int(scan_cfg.get("max_bytes", 0) or 0),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"扫描失败：{exc}") from exc

        created = 0
        jobs: List[Dict[str, Any]] = []
        with _open_storage(active_cfg) as store:
            for path in files:
                job, is_new = store.upsert_job(path, config_hash=active_cfg.config_hash)
                created += 1 if is_new else 0
                jobs.append({**job.to_dict(), "stages": store.stage_map(job.id)})
        existing = len(jobs) - created
        log.info("扫描登记：path=%s created=%d existing=%d", target, created, existing)
        return {"created": created, "existing": existing, "jobs": jobs}

    async def _handle_upload(active_cfg: Config, request: Request) -> Dict[str, Any]:
        """裸流上传：``?filename=xxx.m4a`` → ``out/uploads/<sanitized>`` 后 upsert job。

        用 ``request.stream()`` 分块落盘，不把 2 GiB 读进内存；超过上限 413。
        """
        filename = request.query_params.get("filename") or request.query_params.get("name")
        name = _sanitize_upload_name(filename or "")
        uploads = active_cfg.path("out_dir") / "uploads"
        try:
            uploads.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"创建上传目录失败：{exc}") from exc
        target = uploads / name
        tmp = uploads / f".{name}.{os.getpid()}.{threading.get_ident()}.uploading"

        total = 0
        try:
            with open(tmp, "wb") as handle:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"上传超过上限 {MAX_UPLOAD_BYTES} 字节",
                        )
                    handle.write(chunk)
            if total <= 0:
                raise HTTPException(status_code=400, detail="上传内容为空")
            os.replace(tmp, target)
        except HTTPException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass
            raise
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass
            raise HTTPException(status_code=500, detail=f"写入上传文件失败：{exc}") from exc
        except BaseException:  # 客户端中断等
            try:
                tmp.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass
            raise

        with _open_storage(active_cfg) as store:
            job, created = store.upsert_job(target, config_hash=active_cfg.config_hash)
        log.info("上传完成：%s（%d 字节）→ job=%s", target, total, job.id)
        return {
            "job_id": job.id,
            "source": job.source,
            "size": total,
            "created": created,
        }

    # ------------------------------------------------------------ 运行 / 重试
    @application.post("/api/jobs/{job_id}/run", status_code=202)
    def run_job_endpoint(job_id: str) -> Dict[str, Any]:
        """后台线程跑 ``orchestrator.run_job``；已在跑 409，未知 id 404。"""
        with _open_storage(resolved) as store:
            job = _require_job(store, job_id)
        return _start_run(application, resolved, job)

    @application.post("/api/jobs/{job_id}/retry", status_code=202)
    def retry_job_endpoint(job_id: str) -> Dict[str, Any]:
        """把失败阶段及其下游置回 pending 后重跑（``Storage.reset_stage``）。

        运行中的 job 先判 409，避免把正在跑的阶段表/状态改脏（与 run 同一把进程内锁）。
        """
        with _open_storage(resolved) as store:
            job = _require_job(store, job_id)
            if job_id in application.state.running:
                raise HTTPException(status_code=409, detail=f"job 正在运行：{job_id}")
            stage = job.stage if job.stage in STAGES else "decode"
            try:
                reset = store.reset_stage(job_id, stage, downstream=True)
            except StorageError as exc:  # pragma: no cover - 上面已校验 stage 合法
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            # reset 后 job 状态/pending 以 DB 为准，重新取一次用于响应
            job = store.require_job(job_id)
        payload = _start_run(application, resolved, job)
        payload["reset"] = reset
        log.info("job %s retry：reset_stage(%s, downstream=True) 重置 %d 行", job_id, stage, reset)
        return payload

    # ------------------------------------------------------------ 详情 / 产物
    @application.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> Dict[str, Any]:
        """详情：job/report/stages/artifacts/transcript/corrections。

        transcript 直接内嵌（契约允许几千条）；非 JSON 产物（clean.m4a）只出现在
        artifacts 列表里，不内嵌。
        """
        with _open_storage(resolved) as store:
            job = _require_job(store, job_id)
            stages = [item.to_dict() for item in store.list_stages(job_id)]
            job_dict = job.to_dict()
            job_dict["stages"] = store.stage_map(job_id)
        job_dir = _job_dir(resolved, job_id)
        return {
            "job": job_dict,
            "report": _read_json(job_dir / "report.json", None, expect=dict),
            "stages": stages,
            "artifacts": _job_artifacts(resolved, job_id),
            "transcript": _read_json(job_dir / "transcript.json", None, expect=list),
            "corrections": _read_json(job_dir / "corrections.json", None, expect=dict),
        }

    @application.get("/api/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str) -> List[Dict[str, Any]]:
        """8 件已知产物 ``[{name,size,mtime}]``；uploads 不列、目录不列。"""
        with _open_storage(resolved) as store:
            _require_job(store, job_id)
        return _job_artifacts(resolved, job_id)

    @application.get("/api/jobs/{job_id}/audio")
    def job_audio(job_id: str) -> FileResponse:
        """``clean.m4a``；Starlette FileResponse 原生支持 Range（206）。

        finalize 后音频在 ``<out>/output/audio/<原始文件名>``，这里自动回退。
        """
        with _open_storage(resolved) as store:
            _require_job(store, job_id)
        from .pipeline.finalize import resolve_job_audio

        job_dir = _job_dir(resolved, job_id)
        report = _read_json(job_dir / "report.json", None, expect=dict) or {}
        path = resolve_job_audio(job_dir, resolved.path("out_dir"), str(report.get("source") or ""))
        if path is None:
            raise HTTPException(status_code=404, detail="音频产物不存在：clean.m4a")
        return FileResponse(path, media_type="audio/mp4")

    # ------------------------------------------------------------ SSE
    @application.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        """SSE：每 1.5 s 轮询 DB 推 ``event: snapshot``，15 s 一次 ``: ping`` 心跳。

        DB 查询走 ``asyncio.to_thread`` 不阻塞事件循环；客户端断开时生成器自然退出。
        """

        async def generate():
            last_ping = time.monotonic()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    snapshot = await asyncio.to_thread(_snapshot, resolved)
                    data = json.dumps(
                        snapshot, ensure_ascii=False, separators=(",", ":"), default=str
                    )
                    yield f"event: snapshot\ndata: {data}\n\n"
                    now = time.monotonic()
                    if now - last_ping >= SSE_PING_S:
                        yield ": ping\n\n"
                        last_ping = now
                    await asyncio.sleep(SSE_POLL_S)
            except asyncio.CancelledError:  # 服务器关闭/客户端断开
                raise
            except Exception:  # pragma: no cover - 轮询异常不应拖垮连接
                log.exception("SSE 轮询异常，连接结束")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------ 人工改字
    @application.post("/api/jobs/{job_id}/transcript")
    async def update_transcript(job_id: str, request: Request) -> Dict[str, Any]:
        """只改 ``text``（保留 text_raw/时间/说话人/flags）→ 原子写 transcript.json →
        强制重导出 md/srt/txt；跑任务中 409，未知 id 404。"""
        body = await _read_json_body(request)
        with _open_storage(resolved) as store:
            job = _require_job(store, job_id)
        if job.status == STATUS_RUNNING or job_id in application.state.running:
            raise HTTPException(status_code=409, detail=f"job 正在运行，暂不能改字：{job_id}")

        transcript_path = _job_dir(resolved, job_id) / "transcript.json"
        transcript = _read_json(transcript_path, None, expect=list)
        if transcript is None:
            raise HTTPException(
                status_code=404, detail=f"尚无 transcript.json：{transcript_path.name}"
            )
        updates = _parse_transcript_updates(body, len(transcript))
        for index, text in updates:
            entry = transcript[index]
            if not isinstance(entry, dict):  # pragma: no cover - 契约保证是对象
                raise HTTPException(status_code=500, detail=f"第 {index} 段结构异常")
            entry["text"] = text  # 只覆盖 text，其余字段原样保留

        _atomic_write_json(transcript_path, transcript)

        def _run_export() -> Dict[str, Any]:
            from .pipeline.export import export_text  # 延迟 import，保持服务启动轻量

            return export_text(job_id, resolved, force=True)

        result = await asyncio.to_thread(_run_export)
        if not isinstance(result, Mapping) or not result.get("ok"):
            error = result.get("error") if isinstance(result, Mapping) else result
            raise HTTPException(status_code=500, detail=f"文本重导出失败：{error}")
        log.info("job %s 人工改字 %d 段，已重导出 %s", job_id, len(updates), result.get("formats"))
        return {
            "ok": True,
            "n_updated": len(updates),
            "formats": result.get("formats"),
            "artifacts": _job_artifacts(resolved, job_id),
        }

    # ------------------------------------------------------------ 文本导出
    @application.get("/api/jobs/{job_id}/export")
    def export_text_file(job_id: str, format: str = "") -> FileResponse:
        """``?format=md|srt|txt`` → transcript.<fmt>（UTF-8）；未知格式 400、缺文件 404。"""
        fmt = (format or "").strip().lower().lstrip(".")
        if fmt not in _EXPORT_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的导出格式：{format!r}（可选：md / srt / txt）",
            )
        with _open_storage(resolved) as store:
            _require_job(store, job_id)
        path = _job_dir(resolved, job_id) / f"transcript.{fmt}"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"导出文件不存在：{path.name}")
        return FileResponse(
            path, media_type=_EXPORT_FORMATS[fmt], filename=path.name
        )

    # ------------------------------------------------------------ 静态资源
    @application.get("/")
    def index_page() -> FileResponse:
        """前端入口 ``app/web/index.html``（文件由 P7-T2 提供，缺失时 404）。"""
        path = WEB_DIR / "index.html"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="前端页面尚未生成：app/web/index.html")
        return FileResponse(path, media_type="text/html; charset=utf-8")

    @application.get("/static/{filename}")
    def static_file(filename: str) -> FileResponse:
        """``/static/app.js`` / ``/static/style.css``；只允许 web 目录下的普通文件名。"""
        if (
            not filename
            or filename.startswith(".")
            or "/" in filename
            or "\\" in filename
            or Path(filename).name != filename
        ):
            raise HTTPException(status_code=404, detail="非法静态资源名")
        path = WEB_DIR / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"静态资源不存在：{filename}")
        media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if media_type.startswith("text/") or media_type in ("application/javascript",):
            media_type += "; charset=utf-8"
        return FileResponse(path, media_type=media_type)

    return application


#: 模块级应用：导入即 ``load_config()``，供 ``uvicorn app.server:app`` / ``cli serve`` 使用。
app = create_app()
