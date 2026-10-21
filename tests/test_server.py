# -*- coding: utf-8 -*-
"""P7-T1 Web API 集成测试：真实 uvicorn 线程 + 标准库 urllib。

约定
----
- 不引入 httpx / TestClient / python-multipart：服务用 ``uvicorn.Server`` 在测试线程内
  bind ``127.0.0.1:0``（从 socket 取实际端口），客户端只用 ``urllib``；测试结束优雅关闭。
- 每个用例用 ``conftest.cfg``（所有路径指向 ``tmp_path``）+ 独立 ``create_app(cfg)``，
  DB / 产物互不污染。
- **不跑真实流水线**：``run`` / ``retry`` 用例把 ``app.pipeline.orchestrator.run_job``
  monkeypatch 成快速假实现；真实 ``orchestrator.py`` 尚不存在时由夹具注入轻量占位模块
  （``app/server.py`` 里是延迟 import，行为一致）。
- 需要 transcript 的用例自己造 ``out/<id>/transcript.json``（契约 §4.3 顶层 list）与
  ``report.json``，不依赖真实 decode/ASR 产物。
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import pytest
import uvicorn
from fastapi import FastAPI

from app import __version__
from app.config import Config
from app.models import STATUS_DONE, STATUS_FAILED, STATUS_PENDING, STATUS_RUNNING
from app.server import PRODUCT_FILES, create_app
from app.storage import Storage

ROOT = Path(__file__).resolve().parents[1]
WEB_INDEX = ROOT / "app" / "web" / "index.html"
WEB_JS = ROOT / "app" / "web" / "app.js"

#: 3 段 fixture transcript（契约 §4.3 字段齐全；第 3 段带 flags 用于验证「只改 text」）
SAMPLE_SEGMENTS: List[Dict[str, Any]] = [
    {"start": 0.0, "end": 1.5, "orig_start": 7.8, "orig_end": 9.3,
     "speaker": "主讲", "text_raw": "我去做饭。", "text": "我去做饭。", "flags": []},
    {"start": 23.4, "end": 27.0, "orig_start": 30.1, "orig_end": 33.7,
     "speaker": "主讲", "text_raw": "俩就个。", "text": "俩就个。", "flags": []},
    {"start": 63.0, "end": 65.0, "orig_start": 555.0, "orig_end": 557.0,
     "speaker": "主讲", "text_raw": "嗯。", "text": "嗯。", "flags": ["rejected_chunk"]},
]

#: 4 KiB 假音频（内容无需合法 m4a，Range/文件存在性测试足够）
AUDIO_BYTES = bytes((index * 37 + 11) % 256 for index in range(4096))

#: 契约端点（方法, 路径）——P7-T1 固定，不得改名
EXPECTED_ROUTES = {
    ("/api/health", "GET"),
    ("/api/jobs", "GET"),
    ("/api/jobs", "POST"),
    ("/api/jobs/{job_id}/run", "POST"),
    ("/api/jobs/{job_id}/retry", "POST"),
    ("/api/jobs/{job_id}", "GET"),
    ("/api/jobs/{job_id}/artifacts", "GET"),
    ("/api/jobs/{job_id}/audio", "GET"),
    ("/api/events", "GET"),
    ("/api/jobs/{job_id}/transcript", "POST"),
    ("/api/jobs/{job_id}/export", "GET"),
    ("/", "GET"),
    ("/static/{filename}", "GET"),
}

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------- HTTP 客户端
def _request(
    url: str,
    *,
    method: str = "GET",
    data: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 20.0,
):
    """发请求，返回 ``(status, headers, body)``；4xx/5xx 不抛异常。"""
    request = urllib.request.Request(
        url, data=data, headers=dict(headers or {}), method=method
    )
    try:
        response = _OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()
    try:
        return response.status, response.headers, response.read()
    finally:
        response.close()


def _json_body(body: bytes) -> Any:
    return json.loads(body.decode("utf-8")) if body else None


def _json_request(
    url: str,
    payload: Any,
    *,
    method: str = "POST",
    timeout: float = 20.0,
):
    """发 JSON 请求（Content-Type: application/json，UTF-8 不转义）。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return _request(
        url, method=method, data=data,
        headers={"Content-Type": "application/json"}, timeout=timeout,
    )


def _header(headers: Any, name: str) -> str:
    """大小写无关地取响应头。"""
    if headers is None:
        return ""
    try:
        return str(headers.get(name) or "")
    except AttributeError:  # pragma: no cover - dict 兜底
        return ""


# ---------------------------------------------------------------- 服务夹具
class LiveServer:
    """真实 uvicorn 实例句柄：``str(handle)`` 是 base URL，``.app`` 是 FastAPI 实例。

    用 ``f"{live_server}/api/health"`` 拼 URL；需要断言进程内运行态时读 ``.app.state``。
    """

    def __init__(self, base_url: str, application: FastAPI) -> None:
        self.base_url = base_url
        self.app = application

    def __str__(self) -> str:
        return self.base_url

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"


@pytest.fixture()
def live_server(cfg: Config) -> Iterator[LiveServer]:
    """真实 uvicorn 实例（127.0.0.1:0），yield 句柄，结束优雅关闭。"""
    application = create_app(cfg)
    config = uvicorn.Config(
        application,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn-test", daemon=True)
    thread.start()
    deadline = time.time() + 20.0
    while not server.started:
        if time.time() > deadline or not thread.is_alive():
            raise RuntimeError("uvicorn 测试服务器启动失败")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield LiveServer(f"http://127.0.0.1:{port}", application)
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        if thread.is_alive():  # pragma: no cover - SSE 未关时兜底
            server.force_exit = True
            thread.join(timeout=5.0)


def _wait_not_running(live_server: LiveServer, job_id: str, timeout: float = 10.0) -> None:
    """等后台线程把 job 从进程内 running 集合移除（finally 已执行完）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job_id not in live_server.app.state.running:
            return
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} 仍停留在 running 集合：{live_server.app.state.running}")


@pytest.fixture()
def fake_orchestrator(monkeypatch) -> Dict[str, Any]:
    """注入并替换编排器模块属性（真实模块存在与否都能用）。

    返回可变 state：``calls`` 记录 run_job 调用；``started`` 在假 run 进入时 set；
    ``release`` 默认已 set（立即完成），需要「运行中」状态时测试里 clear 再放行。
    """
    import app.pipeline as pipeline_pkg

    try:
        import app.pipeline.orchestrator as orchestrator
    except Exception:  # ImportError/SyntaxError：P6 并行开发中，注入占位模块
        orchestrator = types.ModuleType("app.pipeline.orchestrator")
        monkeypatch.setitem(sys.modules, "app.pipeline.orchestrator", orchestrator)
        monkeypatch.setattr(pipeline_pkg, "orchestrator", orchestrator, raising=False)

    state: Dict[str, Any] = {
        "calls": [],
        "started": threading.Event(),
        "release": threading.Event(),
    }
    state["release"].set()

    def fake_run_job(job_id: str, source: str, cfg: Config, **kwargs: Any) -> Dict[str, Any]:
        state["calls"].append(job_id)
        store = Storage(cfg.path("db_path"), default_config_hash=cfg.config_hash)
        try:
            store.init_db()
            store.begin_attempt(job_id)
            store.finish_job(job_id, STATUS_DONE)
        finally:
            store.close()
        state["started"].set()
        state["release"].wait(10.0)
        return {"ok": True, "job_id": job_id, "stage": "export", "skipped": False}

    def fake_artifacts(job_id: str, cfg: Config, **kwargs: Any) -> List[Dict[str, Any]]:
        job_dir = cfg.path("out_dir") / job_id
        items: List[Dict[str, Any]] = []
        for name in PRODUCT_FILES:
            path = job_dir / name
            if path.is_file():
                stat = path.stat()
                items.append({"name": name, "size": stat.st_size, "mtime": stat.st_mtime})
        return items

    monkeypatch.setattr(orchestrator, "run_job", fake_run_job, raising=False)
    monkeypatch.setattr(orchestrator, "artifacts", fake_artifacts, raising=False)
    monkeypatch.setattr(orchestrator, "is_complete", lambda *a, **k: False, raising=False)
    return state


# ---------------------------------------------------------------- DB / 产物夹具
def _new_store(cfg: Config) -> Storage:
    store = Storage(cfg.path("db_path"), default_config_hash=cfg.config_hash)
    store.init_db()
    return store


def _register_job(
    cfg: Config,
    job_id: str = "job_test",
    *,
    status: str = STATUS_PENDING,
    stage: str = "",
) -> Dict[str, Any]:
    store = _new_store(cfg)
    try:
        source = str(cfg.path("work_dir") / "inputs" / f"{job_id}.m4a")
        job, _ = store.upsert_job(source, job_id=job_id, status=status)
        if stage:
            store.update_job(job_id, stage=stage)
        return job.to_dict()
    finally:
        store.close()


def _write_job_files(
    cfg: Config,
    job_id: str,
    *,
    transcript: bool = True,
    report: bool = True,
    audio: bool = False,
    all_products: bool = False,
) -> Path:
    job_dir = cfg.path("out_dir") / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if transcript:
        (job_dir / "transcript.json").write_text(
            json.dumps(SAMPLE_SEGMENTS, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if report:
        (job_dir / "report.json").write_text(
            json.dumps(
                {
                    "id": job_id,
                    "source": str(cfg.path("work_dir") / "inputs" / f"{job_id}.m4a"),
                    "duration_s": 65.0,
                    "asr": {"n_segments": 3},
                    "speakers": {"n": 1, "method": "none"},
                    "correct": {"total": 0, "rejected_chunks": 0},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if audio:
        (job_dir / "clean.m4a").write_bytes(AUDIO_BYTES)
    if all_products:
        for name in PRODUCT_FILES:
            path = job_dir / name
            if not path.is_file():
                path.write_bytes(b"product")
    return job_dir


def _wait_job_status(cfg: Config, job_id: str, expected: str, timeout: float = 10.0) -> Dict[str, Any]:
    """轮询 DB 直到 job 到达期望状态（后台线程是异步的）。"""
    deadline = time.time() + timeout
    last: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        store = _new_store(cfg)
        try:
            job = store.get_job(job_id)
            if job is None:  # pragma: no cover - 登记后不会消失
                last = None
            else:
                last = job.to_dict()
                if job.status == expected:
                    return last
        finally:
            store.close()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} 未在 {timeout}s 内到达 {expected}，最后状态={last}")


# ================================================================ 基础 / 工厂
def test_module_level_app_is_fastapi():
    """模块级 ``app = create_app()`` 必须存在（uvicorn app.server:app 入口）。"""
    from app.server import app as module_app

    assert isinstance(module_app, FastAPI)


def test_create_app_exposes_contract_routes(cfg: Config):
    """契约端点齐全，且模块级 app 创建不写 DB/产物（这里只断言路由集合）。"""
    application = create_app(cfg)
    routes = {
        (route.path, method)
        for route in application.routes
        if hasattr(route, "methods")
        for method in route.methods
    }
    missing = EXPECTED_ROUTES - routes
    assert not missing, f"缺少契约端点：{missing}"


def test_health(live_server: LiveServer):
    status, headers, body = _request(f"{live_server}/api/health")
    payload = _json_body(body)
    assert status == 200
    assert payload == {"ok": True, "version": __version__}
    assert "application/json" in _header(headers, "content-type")


def test_jobs_list(live_server: LiveServer, cfg: Config):
    _register_job(cfg, "job_list")
    status, _, body = _request(f"{live_server}/api/jobs")
    payload = _json_body(body)
    assert status == 200
    assert payload["summary"]["counts"]["total"] == 1
    assert [item["id"] for item in payload["jobs"]] == ["job_list"]
    assert "stages" in payload["jobs"][0]


# ================================================================ 扫描 / 上传
def test_create_job_by_scanning_directory(live_server: LiveServer, cfg: Config, tmp_path: Path):
    """POST JSON path → discover_files 扫描登记（>65536 字节的假 .m4a 才不会被 min_bytes 过滤）。"""
    inputs = tmp_path / "scan_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    media = inputs / "第一讲 测试.m4a"
    media.write_bytes(b"\x00\x00\x00\x20ftypM4A " + b"x" * 70000)

    status, _, body = _json_request(f"{live_server}/api/jobs", {"path": str(inputs)})
    payload = _json_body(body)
    assert status == 200
    assert payload["created"] == 1
    assert payload["existing"] == 0
    assert payload["jobs"][0]["source"] == str(media)

    job_id = payload["jobs"][0]["id"]
    store = _new_store(cfg)
    try:
        assert store.get_job(job_id) is not None
    finally:
        store.close()

    # 重复扫描：existing=1、created=0（幂等）
    status, _, body = _json_request(f"{live_server}/api/jobs", {"path": str(inputs)})
    payload = _json_body(body)
    assert status == 200
    assert (payload["created"], payload["existing"]) == (0, 1)


def test_create_job_path_missing_404(live_server: LiveServer, tmp_path: Path):
    status, _, body = _json_request(
        f"{live_server}/api/jobs", {"path": str(tmp_path / "不存在的目录")}
    )
    assert status == 404
    assert "路径不存在" in _json_body(body)["detail"]


def test_upload_octet_stream(live_server: LiveServer, cfg: Config):
    """裸流上传：文件名去目录/替换危险字符，落 out/uploads 并 upsert job。"""
    payload = b"\x00\x00\x00\x20ftypM4A " + b"upload-body" * 128
    filename = "..\\..\\a<b>.m4a"  # 路径穿越 + 非法字符
    query = urllib.parse.quote(filename, safe="")
    status, _, body = _request(
        f"{live_server}/api/jobs?filename={query}",
        method="POST",
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
    )
    result = _json_body(body)
    assert status == 200
    assert result["size"] == len(payload)

    uploads = cfg.path("out_dir") / "uploads"
    target = uploads / "a_b_.m4a"  # basename + <> 替换为 _
    assert target.is_file()
    assert target.read_bytes() == payload
    assert result["source"] == str(target)

    store = _new_store(cfg)
    try:
        job = store.get_job(result["job_id"])
        assert job is not None
        assert job.source == str(target)
    finally:
        store.close()


def test_upload_requires_filename(live_server: LiveServer):
    status, _, body = _request(
        f"{live_server}/api/jobs",
        method="POST",
        data=b"abc",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert status == 400
    assert "filename" in _json_body(body)["detail"]


def test_upload_too_large_413(live_server: LiveServer, cfg: Config, monkeypatch):
    """上限用 monkeypatch 缩小，避免真的传 2 GiB；校验临时文件已清理。"""
    import app.server as server_module

    monkeypatch.setattr(server_module, "MAX_UPLOAD_BYTES", 64)
    status, _, body = _request(
        f"{live_server}/api/jobs?filename=big.m4a",
        method="POST",
        data=b"x" * 200,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert status == 413
    uploads = cfg.path("out_dir") / "uploads"
    leftovers = [p.name for p in uploads.glob("*") if p.is_file()] if uploads.is_dir() else []
    assert leftovers == []


# ================================================================ 详情 / 产物 / Range
def test_unknown_job_returns_404(live_server: LiveServer):
    """未知 id 的所有端点都 404（job id 必须来自 DB）。"""
    cases: Tuple[Tuple[str, str], ...] = (
        ("GET", "/api/jobs/nope"),
        ("GET", "/api/jobs/nope/artifacts"),
        ("GET", "/api/jobs/nope/audio"),
        ("GET", "/api/jobs/nope/export?format=md"),
        ("POST", "/api/jobs/nope/run"),
        ("POST", "/api/jobs/nope/retry"),
    )
    for method, path in cases:
        status, _, _ = _request(f"{live_server}{path}", method=method)
        assert status == 404, f"{method} {path} 应为 404，实际 {status}"
    status, _, _ = _json_request(
        f"{live_server}/api/jobs/nope/transcript",
        {"segments": [{"index": 0, "text": "x"}]},
    )
    assert status == 404


def test_job_detail(live_server: LiveServer, cfg: Config, fake_orchestrator: Dict[str, Any]):
    _register_job(cfg, "job_detail")
    _write_job_files(cfg, "job_detail")

    status, _, body = _request(f"{live_server}/api/jobs/job_detail")
    payload = _json_body(body)
    assert status == 200
    assert set(payload) == {
        "job", "report", "stages", "artifacts", "transcript", "corrections"
    }
    assert payload["job"]["id"] == "job_detail"
    assert payload["report"]["duration_s"] == 65.0
    assert isinstance(payload["stages"], list)
    assert [seg["text"] for seg in payload["transcript"]] == [seg["text"] for seg in SAMPLE_SEGMENTS]
    assert payload["corrections"] is None
    # ensure_ascii=False：原始字节里就是 UTF-8 中文，不是 \uXXXX
    assert "我去做饭。".encode("utf-8") in body


def test_artifacts_lists_known_products(live_server: LiveServer, cfg: Config, fake_orchestrator):
    _register_job(cfg, "job_art")
    _write_job_files(cfg, "job_art", all_products=True)

    status, _, body = _request(f"{live_server}/api/jobs/job_art/artifacts")
    items = _json_body(body)
    assert status == 200
    assert isinstance(items, list)
    assert {item["name"] for item in items} == set(PRODUCT_FILES)
    assert all(isinstance(item["size"], int) and item["size"] > 0 for item in items)
    assert all("mtime" in item for item in items)
    # uploads 不列、目录不列
    assert all("/" not in item["name"] for item in items)


def test_artifacts_uses_orchestrator_interface(
    live_server: LiveServer, cfg: Config, fake_orchestrator: Dict[str, Any], monkeypatch
):
    """端点必须走 orchestrator.artifacts（模块属性），而不是自己拼路径。"""
    import app.pipeline.orchestrator as orchestrator

    _register_job(cfg, "job_art2")
    _write_job_files(cfg, "job_art2", transcript=False, report=False)
    calls: List[str] = []

    def custom_artifacts(job_id: str, cfg_arg: Config, **kwargs: Any):
        calls.append(job_id)
        return [{"name": "clean.m4a", "size": 123, "mtime": 1700000000.0}]

    monkeypatch.setattr(orchestrator, "artifacts", custom_artifacts)
    status, _, body = _request(f"{live_server}/api/jobs/job_art2/artifacts")
    items = _json_body(body)
    assert status == 200
    assert calls == ["job_art2"]
    assert items == [{"name": "clean.m4a", "size": 123, "mtime": 1700000000.0}]


def test_audio_supports_range_206(live_server: LiveServer, cfg: Config):
    """Starlette FileResponse 原生 Range：bytes=0-1023 → 206 + Content-Range。"""
    _register_job(cfg, "job_audio")
    _write_job_files(cfg, "job_audio", transcript=False, report=False, audio=True)

    status, headers, body = _request(
        f"{live_server}/api/jobs/job_audio/audio",
        headers={"Range": "bytes=0-1023"},
    )
    assert status == 206
    assert body == AUDIO_BYTES[:1024]
    assert _header(headers, "content-range") == f"bytes 0-1023/{len(AUDIO_BYTES)}"
    assert _header(headers, "accept-ranges") == "bytes"

    status, _, body = _request(f"{live_server}/api/jobs/job_audio/audio")
    assert status == 200
    assert body == AUDIO_BYTES


def test_audio_missing_404(live_server: LiveServer, cfg: Config):
    _register_job(cfg, "job_no_audio")
    _write_job_files(cfg, "job_no_audio", transcript=False, report=False)
    status, _, _ = _request(f"{live_server}/api/jobs/job_no_audio/audio")
    assert status == 404


# ================================================================ SSE
def test_events_first_snapshot(live_server: LiveServer, cfg: Config):
    """读到首个 ``event: snapshot`` 后关闭连接；data 是 summary+jobs JSON。"""
    _register_job(cfg, "job_sse")
    request = urllib.request.Request(
        f"{live_server}/api/events", headers={"Accept": "text/event-stream"}
    )
    response = _OPENER.open(request, timeout=20.0)
    try:
        assert "text/event-stream" in _header(response.headers, "content-type")
        lines: List[str] = []
        for _ in range(20):
            line = response.readline().decode("utf-8")
            if not line:
                break
            lines.append(line.rstrip("\r\n"))
            if lines[-1].startswith("data:"):
                break
        assert lines and lines[0] == "event: snapshot", f"首个事件不对：{lines}"
        data_line = next(line for line in lines if line.startswith("data:"))
        payload = json.loads(data_line[len("data:"):].strip())
        assert payload["summary"]["counts"]["total"] == 1
        assert [item["id"] for item in payload["jobs"]] == ["job_sse"]
    finally:
        response.close()


# ================================================================ 改字 / 导出
def test_transcript_edit_updates_products(live_server: LiveServer, cfg: Config, fake_orchestrator):
    """只改 text、保留其它字段；原子写 transcript.json 并强制重导出 md/srt/txt。"""
    _register_job(cfg, "job_edit", status=STATUS_DONE, stage="export")
    job_dir = _write_job_files(cfg, "job_edit")

    status, _, body = _json_request(
        f"{live_server}/api/jobs/job_edit/transcript",
        {"segments": [{"index": 1, "text": "改过的文本"}]},
    )
    payload = _json_body(body)
    assert status == 200
    assert payload["ok"] is True
    assert payload["n_updated"] == 1
    names = {item["name"] for item in payload["artifacts"]}
    assert {"transcript.json", "transcript.md", "transcript.srt", "transcript.txt"} <= names

    segments = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
    edited = segments[1]
    assert edited["text"] == "改过的文本"
    assert edited["text_raw"] == "俩就个。"          # 原文保留
    assert (edited["start"], edited["end"]) == (23.4, 27.0)
    assert (edited["orig_start"], edited["orig_end"]) == (30.1, 33.7)
    assert edited["speaker"] == "主讲"
    assert edited["flags"] == []
    assert segments[0]["text"] == "我去做饭。"        # 其它段不动
    assert segments[2]["flags"] == ["rejected_chunk"]

    md = (job_dir / "transcript.md").read_text(encoding="utf-8")
    srt = (job_dir / "transcript.srt").read_text(encoding="utf-8")
    txt = (job_dir / "transcript.txt").read_text(encoding="utf-8")
    assert "改过的文本" in md and "改过的文本" in srt and "改过的文本" in txt
    assert "俩就个。" not in md                       # 旧 text 已被覆盖
    # md/srt 时间轴仍按原段 start 生成
    assert "[00:23 (orig 00:30)]" in md
    assert "00:00:23,400 --> 00:00:27,000" in srt

    # 详情接口能读到新 text（且原始字节是 UTF-8 中文）
    status, _, detail_body = _request(f"{live_server}/api/jobs/job_edit")
    assert status == 200
    assert "改过的文本".encode("utf-8") in detail_body


def test_transcript_full_list(live_server: LiveServer, cfg: Config):
    """完整 list（与 transcript 同长）按位置更新 text。"""
    _register_job(cfg, "job_full", status=STATUS_DONE, stage="export")
    job_dir = _write_job_files(cfg, "job_full")
    updated = [dict(segment) for segment in SAMPLE_SEGMENTS]
    for index, segment in enumerate(updated):
        segment["text"] = f"第{index + 1}段新文本"
        segment["text_raw"] = "调用方不应生效"

    status, _, body = _json_request(f"{live_server}/api/jobs/job_full/transcript", updated)
    payload = _json_body(body)
    assert status == 200
    assert payload["n_updated"] == 3

    segments = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
    assert [segment["text"] for segment in segments] == ["第1段新文本", "第2段新文本", "第3段新文本"]
    # 完整 list 里只取 text：text_raw 仍是磁盘上的原文
    assert [segment["text_raw"] for segment in segments] == [
        segment["text_raw"] for segment in SAMPLE_SEGMENTS
    ]


def test_transcript_validation_400(live_server: LiveServer, cfg: Config):
    _register_job(cfg, "job_bad", status=STATUS_DONE, stage="export")
    _write_job_files(cfg, "job_bad")
    url = f"{live_server}/api/jobs/job_bad/transcript"

    status, _, body = _json_request(url, {"segments": [{"index": 99, "text": "越界"}]})
    assert status == 400
    assert "越界" in _json_body(body)["detail"]

    status, _, body = _json_request(url, {"segments": [{"index": 0, "text": "x" * 5001}]})
    assert status == 400
    assert "上限" in _json_body(body)["detail"]

    status, _, _ = _json_request(url, [{"text": "长度不对"}])  # 完整 list 但长度不符
    assert status == 400

    status, _, _ = _request(
        url, method="POST", data=b"not json", headers={"Content-Type": "application/json"}
    )
    assert status == 400


def test_transcript_during_run_409(live_server: LiveServer, cfg: Config):
    _register_job(cfg, "job_running", status=STATUS_RUNNING, stage="asr")
    _write_job_files(cfg, "job_running")
    status, _, body = _json_request(
        f"{live_server}/api/jobs/job_running/transcript",
        {"segments": [{"index": 0, "text": "运行中"}]},
    )
    assert status == 409
    assert "正在运行" in _json_body(body)["detail"]


def test_transcript_missing_transcript_404(live_server: LiveServer, cfg: Config):
    _register_job(cfg, "job_notrans", status=STATUS_DONE, stage="export")
    _write_job_files(cfg, "job_notrans", transcript=False, report=True)
    status, _, _ = _json_request(
        f"{live_server}/api/jobs/job_notrans/transcript",
        {"segments": [{"index": 0, "text": "x"}]},
    )
    assert status == 404


def test_export_md_and_unknown_format(live_server: LiveServer, cfg: Config):
    _register_job(cfg, "job_export", status=STATUS_DONE, stage="export")
    job_dir = _write_job_files(cfg, "job_export")
    # 先让真正的 export_text 生成 md/srt/txt（走改字端点）
    status, _, _ = _json_request(
        f"{live_server}/api/jobs/job_export/transcript",
        {"segments": [{"index": 0, "text": "导出内容测试"}]},
    )
    assert status == 200

    status, headers, body = _request(f"{live_server}/api/jobs/job_export/export?format=md")
    assert status == 200
    assert "text/markdown" in _header(headers, "content-type")
    text = body.decode("utf-8")
    assert "导出内容测试" in text
    assert text == (job_dir / "transcript.md").read_text(encoding="utf-8")

    status, _, _ = _request(f"{live_server}/api/jobs/job_export/export?format=pdf")
    assert status == 400

    status, _, _ = _request(f"{live_server}/api/jobs/job_export/export?format=srt")
    assert status == 200


# ================================================================ run / retry
def test_run_202_and_duplicate_409(
    live_server: LiveServer, cfg: Config, fake_orchestrator: Dict[str, Any]
):
    """run 返回 202；同一 job 运行中重复 run → 409（进程内 running + per-job 锁）。"""
    _register_job(cfg, "job_run")
    fake_orchestrator["release"].clear()  # 让假 run 卡住，制造「运行中」

    status, _, body = _request(f"{live_server}/api/jobs/job_run/run", method="POST")
    payload = _json_body(body)
    assert status == 202
    assert payload["job_id"] == "job_run"
    assert payload["status"] == STATUS_RUNNING

    status, _, body = _request(f"{live_server}/api/jobs/job_run/run", method="POST")
    assert status == 409
    assert "正在运行" in _json_body(body)["detail"]

    # 运行中 retry 也必须 409（避免把运行中的 stages 改脏）
    status, _, body = _request(f"{live_server}/api/jobs/job_run/retry", method="POST")
    assert status == 409
    assert "正在运行" in _json_body(body)["detail"]

    fake_orchestrator["release"].set()
    fake_orchestrator["started"].wait(5.0)
    job = _wait_job_status(cfg, "job_run", STATUS_DONE)
    assert fake_orchestrator["calls"] == ["job_run"]
    assert job["status"] == STATUS_DONE
    _wait_not_running(live_server, "job_run")

    # 运行结束后可以再次提交（不残留锁）
    status, _, _ = _request(f"{live_server}/api/jobs/job_run/run", method="POST")
    assert status == 202
    _wait_job_status(cfg, "job_run", STATUS_DONE)
    _wait_not_running(live_server, "job_run")


def test_retry_resets_downstream_stages_202(
    live_server: LiveServer, cfg: Config, fake_orchestrator: Dict[str, Any]
):
    """retry：reset failed 及下游阶段 → 202；decode 已完成保留，asr 回到 pending。"""
    store = _new_store(cfg)
    try:
        source = str(cfg.path("work_dir") / "inputs" / "job_retry.m4a")
        store.upsert_job(source, job_id="job_retry", status=STATUS_FAILED)
        store.start_stage("job_retry", "decode")
        store.finish_stage("job_retry", "decode", STATUS_DONE)
        store.start_stage("job_retry", "asr")
        store.fail_stage("job_retry", "asr", "识别失败")
        store.start_stage("job_retry", "correct")
        store.fail_stage("job_retry", "correct", "纠错失败")
        store.update_job("job_retry", status=STATUS_FAILED, stage="asr", error="识别失败")
    finally:
        store.close()

    status, _, body = _request(f"{live_server}/api/jobs/job_retry/retry", method="POST")
    payload = _json_body(body)
    assert status == 202
    assert payload["job_id"] == "job_retry"
    assert payload["reset"] >= 2  # asr + correct(+export 未建行不计数)

    _wait_job_status(cfg, "job_retry", STATUS_DONE)
    store = _new_store(cfg)
    try:
        assert store.is_stage_done("job_retry", "decode") is True
        assert store.get_stage("job_retry", "asr").status == STATUS_PENDING
        assert store.get_stage("job_retry", "correct").status == STATUS_PENDING
    finally:
        store.close()


# ================================================================ 静态资源
@pytest.mark.skipif(not WEB_INDEX.is_file(), reason="app/web/index.html 尚未生成（P7-T2 并行开发中）")
def test_static_index(live_server: LiveServer):
    status, headers, body = _request(f"{live_server}/")
    assert status == 200
    assert "text/html" in _header(headers, "content-type")
    assert b"<" in body


@pytest.mark.skipif(not WEB_JS.is_file(), reason="app/web/app.js 尚未生成（P7-T2 并行开发中）")
def test_static_assets(live_server: LiveServer):
    status, headers, body = _request(f"{live_server}/static/app.js")
    assert status == 200
    assert "javascript" in _header(headers, "content-type")
    assert len(body) > 0

    status, _, _ = _request(f"{live_server}/static/nonexistent.css")
    assert status == 404
