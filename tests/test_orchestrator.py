# -*- coding: utf-8 -*-
"""P6-T1/T3 编排器单测：阶段顺序/落库、export 合并、resume 以产物为准、失败隔离、
磁盘水位、锁、清理、quarantine、attempts 上限 / limit / force。

约定
----
- **全部用 monkeypatch 假阶段**（render_job / transcribe_job / diarize_job / correct_job /
  export_text），不跑真实音频 / ASR / ffmpeg；
- 假阶段会按真实模块的契约写产物 + 回调 ``on_stage(stage, status, metrics)``，这样
  ``is_complete`` / 合并指标 / 补记 stages 的行都能被真实验证；
- 所有落盘路径走 ``tests/conftest.py`` 的 ``cfg`` fixture（全部指向 ``tmp_path``），
  不写 ``out/``、不碰 ``work/cache``。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import app.pipeline.orchestrator as orch
from app.models import STAGES, STATUS_DONE, STATUS_FAILED
from app.storage import Storage


# ================================================================ 工具
def _job_out(cfg: Any, job_id: str) -> Path:
    return cfg.path("out_dir") / job_id


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _update_report(job_out: Path, **sections: Any) -> Dict[str, Any]:
    """report.json 读-改-写（与真实阶段模块一致，保留既有字段）。"""
    path = job_out / "report.json"
    report = _read_json(path) if path.is_file() else {}
    report.update(sections)
    _write_json(path, report)
    return report


def _make_source(cfg: Any, job_id: str, suffix: str = ".m4a") -> Path:
    """在 tmp 下造一个假输入文件（内容任意，假阶段不真的解码）。"""
    src = cfg.path("out_dir").parent / "src" / f"{job_id}{suffix}"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"FAKE-AUDIO" * 16)
    return src


def _register(cfg: Any, job_id: str, source: Path, *, status: Optional[str] = None,
              attempts: int = 0) -> None:
    store = Storage(cfg.path("db_path"))
    store.init_db()
    store.upsert_job(source, job_id=job_id, config_hash=cfg.config_hash)
    if status is not None or attempts:
        store.update_job(job_id, status=status, attempts=attempts)
    store.close()


def _stage_rows(cfg: Any, job_id: str):
    store = Storage(cfg.path("db_path"))
    store.init_db()
    try:
        return store.stage_map(job_id), store.list_stages(job_id)
    finally:
        store.close()


def _get_job(cfg: Any, job_id: str):
    store = Storage(cfg.path("db_path"))
    store.init_db()
    try:
        return store.get_job(job_id)
    finally:
        store.close()


def _seed_stages(cfg: Any, job_id: str, stages: List[str], metrics: Dict[str, Any]) -> None:
    store = Storage(cfg.path("db_path"))
    store.init_db()
    try:
        for stage in stages:
            store.finish_stage(job_id, stage, STATUS_DONE, dict(metrics))
    finally:
        store.close()


def _make_complete_artifacts(cfg: Any, job_id: str, source: Path,
                             *, report_sections: bool = True) -> Path:
    """造 8 件齐全的产物（report 默认含 asr/correct/speakers）。"""
    job_out = _job_out(cfg, job_id)
    job_out.mkdir(parents=True, exist_ok=True)
    (job_out / "clean.m4a").write_bytes(b"FAKE-M4A-CONTENT")
    _write_json(job_out / "edit.json", {"kept": [], "removed": [], "reason_enum": []})
    report: Dict[str, Any] = {
        "id": job_id,
        "source": str(source),
        "ok": True,
        "levels": {"true_peak_out_dbtp": -1.4, "asl_out_dbfs": -20.0},
        "output": {"size_bytes": 999},
    }
    if report_sections:
        report.update({
            "asr": {"n_segments": 3, "chars": 9, "elapsed_s": 1.0},
            "speakers": {"n": 1, "method": "none"},
            "correct": {"enabled": False, "edits": 0, "rejected": 0},
        })
    _write_json(job_out / "report.json", report)
    _write_json(job_out / "transcript.json", [
        {"start": 0.0, "end": 1.0, "orig_start": 0.0, "orig_end": 1.0,
         "speaker": "主讲", "text_raw": "你好", "text": "你好", "flags": []},
    ])
    (job_out / "transcript.md").write_text("# t\n\n[00:00] 主讲：你好\n", encoding="utf-8")
    (job_out / "transcript.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n主讲：你好\n", encoding="utf-8"
    )
    (job_out / "transcript.txt").write_text("[00:00] 主讲：你好\n", encoding="utf-8")
    _write_json(job_out / "corrections.json",
                {"total": 0, "items": [], "rejected_chunks": []})
    return job_out


# ================================================================ 假阶段
def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    cfg: Any,
    *,
    fail: Optional[str] = None,
    raise_at: Optional[str] = None,
    raise_job: Optional[str] = None,
    render_skip: bool = False,
) -> Dict[str, Any]:
    """用假阶段替换编排器 import 的 5 个阶段函数，返回调用记录 state。

    - ``fail``：指定阶段返回 ``ok=False`` 并 ``on_stage(..., "failed")``；
    - ``raise_at``：指定阶段直接抛异常（测「意外异常也要失败隔离」）；
    - ``raise_job``：异常只对某个 job 抛（batch 失败隔离用）；
    - ``render_skip``：模拟 render_job 产物已存在的 skip 分支（回调全部阶段、metrics={}）。
    """
    state: Dict[str, Any] = {
        "calls": [],
        "fail": fail,
        "raise_at": raise_at,
        "raise_job": raise_job,
        "render_skip": render_skip,
    }

    def _out(job_id: str, out_dir: Any) -> Path:
        return (Path(out_dir) if out_dir else cfg.path("out_dir")) / job_id

    def _raise(job_id: str, stage: str) -> None:
        if state["raise_at"] == stage and state["raise_job"] in (None, job_id):
            raise RuntimeError(f"{stage}-boom")

    def fake_render(job_id, source, cfg_, *, out_dir=None, cache_dir=None, force=False,
                    keep_intermediates=False, on_stage=None):
        state["calls"].append(("render", job_id))
        _raise(job_id, "decode")
        job_out = _out(job_id, out_dir)
        job_out.mkdir(parents=True, exist_ok=True)
        if state["fail"] == "decode":
            if on_stage:
                on_stage("decode", "failed", {"error": "decode-fail"})
            return {"ok": False, "stage": "decode", "error": "decode-fail", "skipped": False}
        (job_out / "clean.m4a").write_bytes(b"FAKE-M4A")
        _write_json(job_out / "edit.json", {"kept": [], "removed": [], "reason_enum": []})
        report = {
            "id": job_id, "source": str(source), "ok": True,
            "levels": {"true_peak_out_dbtp": -1.25, "asl_out_dbfs": -20.02},
            "output": {"size_bytes": 4096},
            "trim": {"clean_dur_s": 123.4},
        }
        _write_json(job_out / "report.json", report)
        if state["render_skip"]:
            for stage in ("decode", "analyze", "denoise", "trim", "loudness", "export"):
                if on_stage:
                    on_stage(stage, "done", {})
            return {"ok": True, "skipped": True, "stage": "export",
                    "report": report, "timings": {"total_s": 0.01}}
        for stage, metrics in (
            ("decode", {"duration_s": 123.4}),
            ("analyze", {"from_cache": False}),
            ("denoise", {"mode": "afftdn"}),
            ("trim", {"clean_dur_s": 123.4}),
            ("loudness", {"gain_db": -9.9}),
        ):
            if on_stage:
                on_stage(stage, "done", metrics)
        if on_stage:
            on_stage("export", "done",
                     {"tp_dbtp": -1.30, "asl_out_dbfs": -20.01, "size_bytes": 4096})
        return {"ok": True, "skipped": False, "stage": "export",
                "report": report, "timings": {"total_s": 1.0}}

    def fake_asr(job_id, cfg_, *, out_dir=None, cache_dir=None, force=False, on_stage=None):
        state["calls"].append(("asr", job_id))
        _raise(job_id, "asr")
        if state["fail"] == "asr":
            if on_stage:
                on_stage("asr", "failed", {"error": "asr-fail"})
            return {"ok": False, "stage": "asr", "error": "asr-fail"}
        job_out = _out(job_id, out_dir)
        job_out.mkdir(parents=True, exist_ok=True)
        cache = Path(cache_dir) if cache_dir else cfg.path("cache_dir")
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{job_id}.clean16k.wav").write_bytes(b"CLEAN16K")  # 模拟 ASR 解出的中间文件
        _write_json(job_out / "transcript.json", [
            {"start": 0.0, "end": 1.5, "orig_start": 2.0, "orig_end": 3.5,
             "speaker": "主讲", "text_raw": "你好", "text": "你好", "flags": []},
        ])
        metrics = {"n_segments": 1, "chars": 2, "n_empty": 0, "elapsed_s": 0.1}
        _update_report(job_out, asr=dict(metrics))
        if on_stage:
            on_stage("asr", "done", dict(metrics))
        return {"ok": True, "skipped": False, "stage": "asr", "asr": dict(metrics),
                "timings": {"asr_s": 0.1}}

    def fake_diarize(job_id, cfg_, *, out_dir=None, cache_dir=None, force=False, on_stage=None):
        state["calls"].append(("diarize", job_id))
        _raise(job_id, "diarize")
        job_out = _out(job_id, out_dir)
        job_out.mkdir(parents=True, exist_ok=True)
        speakers = {"n": 1, "method": "none", "diar_version": "fake"}
        _update_report(job_out, speakers=dict(speakers))
        if on_stage:
            on_stage("diarize", "done", {"mode": "none", "n": 1, "changed": 0})
        return {"ok": True, "skipped": False, "stage": "diarize",
                "speakers": dict(speakers)}

    def fake_correct(job_id, cfg_, *, out_dir=None, force=False, on_stage=None):
        state["calls"].append(("correct", job_id))
        _raise(job_id, "correct")
        job_out = _out(job_id, out_dir)
        job_out.mkdir(parents=True, exist_ok=True)
        _write_json(job_out / "corrections.json", {
            "total": 2,
            "items": [{"orig": "你好", "fixed": "您好", "reason": "x", "chunk": 1},
                      {"orig": "啊", "fixed": "呀", "reason": "y", "chunk": 1}],
            "rejected_chunks": [{"chunk": 2, "diff_ratio": 0.9, "reason": "diff_too_large"}],
        })
        metrics = {"enabled": False, "chunks": 1, "edits": 2, "rejected": 1, "cost_cny": 0.0}
        _update_report(job_out, correct=dict(metrics))
        if on_stage:
            on_stage("correct", "done", dict(metrics))
        return {"ok": True, "skipped": False, "stage": "correct", "timings": {"correct_s": 0.05}}

    def fake_export_text(job_id, cfg_, *, out_dir=None, formats=None, force=False, on_stage=None):
        state["calls"].append(("export_text", job_id))
        _raise(job_id, "export")
        job_out = _out(job_id, out_dir)
        job_out.mkdir(parents=True, exist_ok=True)
        (job_out / "transcript.md").write_text(
            "# t\n\n[00:00] 主讲：你好\n", encoding="utf-8")
        (job_out / "transcript.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,500\n主讲：你好\n", encoding="utf-8")
        (job_out / "transcript.txt").write_text("[00:00] 主讲：你好\n", encoding="utf-8")
        metrics = {"formats": ["md", "srt", "txt"], "n_segments": 1, "n_written": 1,
                   "elapsed_s": 0.02}
        if on_stage:
            on_stage("export", "done", {"formats": ["md", "srt", "txt"],
                                        "n_segments": 1, "n_written": 1})
        return {"ok": True, "skipped": False, "stage": "export",
                "formats": ["md", "srt", "txt"], "n_segments": 1, "n_written": 1,
                "elapsed_s": 0.02}

    monkeypatch.setattr(orch, "render_job", fake_render)
    monkeypatch.setattr(orch, "transcribe_job", fake_asr)
    monkeypatch.setattr(orch, "diarize_job", fake_diarize)
    monkeypatch.setattr(orch, "correct_job", fake_correct)
    monkeypatch.setattr(orch, "export_text", fake_export_text)
    return state


def _export_rows(rows: List[Any]) -> List[Any]:
    return [row for row in rows if row.stage == "export"]


# ================================================================ 产物 / 完成判定
def test_artifacts_lists_only_existing_products(cfg):
    job_id = "job-artifacts"
    job_out = _job_out(cfg, job_id)
    job_out.mkdir(parents=True, exist_ok=True)
    assert orch.artifacts(job_id, cfg) == []
    (job_out / "clean.m4a").write_bytes(b"x" * 10)
    (job_out / "transcript.md").write_text("md\n", encoding="utf-8")
    (job_out / "unknown.bin").write_bytes(b"ignored")
    items = orch.artifacts(job_id, cfg)
    assert [item["name"] for item in items] == ["clean.m4a", "transcript.md"]
    assert items[0]["size"] == 10
    assert items[0]["mtime"] > 0
    assert orch.artifacts("../evil", cfg) == []  # 路径穿越直接拒绝


def test_is_complete_requires_all_files_and_report_sections(cfg):
    job_id = "job-complete"
    source = _make_source(cfg, job_id)
    assert orch.is_complete(job_id, cfg) is False
    _make_complete_artifacts(cfg, job_id, source)
    assert orch.is_complete(job_id, cfg) is True
    # report 缺 speakers → 不全
    report_path = _job_out(cfg, job_id) / "report.json"
    report = _read_json(report_path)
    report.pop("speakers")
    _write_json(report_path, report)
    assert orch.is_complete(job_id, cfg) is False
    # 恢复后删一个文本产物 → 不全
    report["speakers"] = {"n": 1, "method": "none"}
    _write_json(report_path, report)
    (_job_out(cfg, job_id) / "transcript.srt").unlink()
    assert orch.is_complete(job_id, cfg) is False


# ================================================================ run_job：成功路径
def test_run_job_success_order_metrics_and_single_finish(monkeypatch, cfg):
    job_id = "job-ok"
    source = _make_source(cfg, job_id)
    state = install_fakes(monkeypatch, cfg)

    finish_calls = {"n": 0}
    original_finish = Storage.finish_job

    def spy_finish(self, job_id_, status=STATUS_DONE, error=""):
        finish_calls["n"] += 1
        return original_finish(self, job_id_, status, error)

    monkeypatch.setattr(Storage, "finish_job", spy_finish)

    result = orch.run_job(job_id, source, cfg)

    assert result["ok"] is True and result["skipped"] is False and result["locked"] is False
    assert [call[0] for call in state["calls"]] == [
        "render", "asr", "diarize", "correct", "export_text",
    ]
    assert result["stages"] == {stage: STATUS_DONE for stage in STAGES}
    assert finish_calls["n"] == 1  # finish_job 只调一次
    assert len(result["artifacts"]) == 8
    assert result["timings"]["total_s"] >= 0
    job = _get_job(cfg, job_id)
    assert job is not None and job.status == STATUS_DONE and job.attempts == 1

    # export 只有一条，且 metrics 是 audio + text + transcript + corrections + speakers 合并
    _statuses, rows = _stage_rows(cfg, job_id)
    export_rows = _export_rows(rows)
    assert len(export_rows) == 1
    metrics = export_rows[0].metrics
    assert metrics["tp_dbtp"] == -1.30
    assert metrics["asl_out_dbfs"] == -20.01
    assert metrics["size_bytes"] == 4096
    assert metrics["md"] and metrics["srt"] and metrics["txt"]
    assert metrics["n_written"] == 1
    assert metrics["n_segments"] == 1 and metrics["chars"] == 2
    assert metrics["total"] == 2 and metrics["rejected"] == 1
    assert metrics["n"] == 1 and metrics["method"] == "none"


def test_render_skip_callbacks_do_not_block_downstream(monkeypatch, cfg):
    """render_job 的 skip 分支回调全部阶段且 metrics={}：必须继续跑 ASR 等后续阶段。"""
    job_id = "job-render-skip"
    source = _make_source(cfg, job_id)
    state = install_fakes(monkeypatch, cfg, render_skip=True)

    result = orch.run_job(job_id, source, cfg)

    assert result["ok"] is True
    assert [call[0] for call in state["calls"]] == [
        "render", "asr", "diarize", "correct", "export_text",
    ]
    statuses, rows = _stage_rows(cfg, job_id)
    assert statuses == {stage: STATUS_DONE for stage in STAGES}
    # P1–P3 的旧 metrics 保留（skip 回调的 {} 不覆盖已有行；这里首次落库就是 {}）
    decode_row = next(row for row in rows if row.stage == "decode")
    assert decode_row.metrics == {}
    # export 的 audio 指标从 report 兜底（render skip 没给 audio_export）
    export_row = _export_rows(rows)[0]
    assert export_row.metrics["tp_dbtp"] == -1.25
    assert export_row.metrics["asl_out_dbfs"] == -20.02


# ================================================================ run_job：resume
def test_run_job_complete_skips_without_calling_stages(monkeypatch, cfg):
    job_id = "job-done"
    source = _make_source(cfg, job_id)
    _register(cfg, job_id, source, status=STATUS_DONE)
    _make_complete_artifacts(cfg, job_id, source)
    state = install_fakes(monkeypatch, cfg)

    before = _get_job(cfg, job_id)
    result = orch.run_job(job_id, source, cfg)

    assert result["ok"] is True and result["skipped"] is True
    assert state["calls"] == []  # 不调用任何阶段（含 render/ASR）
    after = _get_job(cfg, job_id)
    assert after.status == STATUS_DONE and after.attempts == before.attempts
    # 缺的 stages 行被补记（backfill），export 行用合并指标
    statuses, rows = _stage_rows(cfg, job_id)
    assert statuses == {stage: STATUS_DONE for stage in STAGES}
    export_row = _export_rows(rows)[0]
    assert export_row.metrics.get("backfilled") is True
    assert export_row.metrics["n_segments"] == 3
    assert export_row.metrics["chars"] == 9


def test_run_job_backfills_only_missing_diarize_row(monkeypatch, cfg):
    """P4 遗留：done 任务产物齐全但 stages 没有 diarize 行 → 只补该行，不整条重跑。"""
    job_id = "job-p4"
    source = _make_source(cfg, job_id)
    _register(cfg, job_id, source, status=STATUS_DONE)
    _make_complete_artifacts(cfg, job_id, source)
    seeded = [stage for stage in STAGES if stage != "diarize"]
    _seed_stages(cfg, job_id, seeded, {"seed": True})
    state = install_fakes(monkeypatch, cfg)

    result = orch.run_job(job_id, source, cfg)

    assert result["skipped"] is True
    assert state["calls"] == []
    statuses, rows = _stage_rows(cfg, job_id)
    assert statuses["diarize"] == STATUS_DONE
    diarize_row = next(row for row in rows if row.stage == "diarize")
    assert diarize_row.metrics.get("backfilled") is True
    # 其它行（含 export）保持原 metrics 不变
    export_row = _export_rows(rows)[0]
    assert export_row.metrics == {"seed": True}


# ================================================================ 锁
def test_job_lock_is_exclusive_and_reusable(cfg):
    lock1 = orch.JobLock("job-lock", cfg)
    lock2 = orch.JobLock("job-lock", cfg)
    assert lock1.path == cfg.path("work_dir") / "locks" / "job-lock.lock"
    assert lock1.held is False
    assert lock1.acquire(blocking=False) is True
    assert lock1.held is True
    assert lock2.acquire(blocking=False) is False  # 同 job 第二个抢不到
    lock1.release()
    assert lock1.held is False
    assert lock2.acquire(blocking=False) is True
    lock2.release()
    assert lock2.held is False and not lock2.path.exists()
    # with 语句
    with orch.JobLock("job-lock-ctx", cfg) as held:
        assert held.held is True
        with pytest.raises(RuntimeError):
            with orch.JobLock("job-lock-ctx", cfg):
                pass
    assert not held.path.exists()


def test_job_lock_reclaims_stale_pid(cfg):
    lock = orch.JobLock("job-stale", cfg)
    lock.lock_dir.mkdir(parents=True, exist_ok=True)
    # 一个不可能存在的 pid（Windows OpenProcess 报 87 / POSIX ProcessLookupError）
    lock.path.write_text("999999999\t0.0\n", encoding="utf-8")
    assert lock.acquire(blocking=False) is True
    assert lock.held is True
    pid_text = lock.path.read_text(encoding="utf-8").split("\t")[0]
    assert pid_text == str(os.getpid())  # stale 锁已被替换成自己的 pid
    lock.release()


def test_pid_alive_false_for_terminated_process_object():
    """回归：进程已退出但内核进程对象未回收时，PID 仍可被 OpenProcess 成功打开。

    修复前 ``_pid_alive`` 只凭 OpenProcess 成功就判活 → 强杀（taskkill）后 stale 锁
    永远清不掉、resume 直接返回 locked=True。修复后看 GetExitCodeProcess（≠259 → 死）。
    """
    if sys.platform != "win32":
        pytest.skip("依赖 Windows「已终止但进程对象未回收」的语义")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.05)"])
    try:
        deadline = time.time() + 15.0
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.02)
        assert proc.returncode is not None, "子进程未在 15s 内退出"
        # 此刻 Popen 仍持有进程 handle（未 wait/未回收），PID 在进程表里但进程已死
        assert orch._pid_alive(proc.pid) is False
    finally:
        proc.wait()


def test_job_lock_reclaims_pid_of_terminated_process(cfg):
    """强杀场景的端到端回归：锁文件 pid 已终止 → acquire 清 stale 并重建。"""
    if sys.platform != "win32":
        pytest.skip("依赖 Windows「已终止但进程对象未回收」的语义")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.05)"])
    try:
        deadline = time.time() + 15.0
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.02)
        assert proc.returncode is not None
        lock = orch.JobLock("job-zombie", cfg)
        lock.lock_dir.mkdir(parents=True, exist_ok=True)
        lock.path.write_text(f"{proc.pid}\t0.0\n", encoding="utf-8")
        assert lock.acquire(blocking=False) is True  # 修复前这里是 False
        assert lock.held is True
        assert lock.path.read_text(encoding="utf-8").split("\t")[0] == str(os.getpid())
        lock.release()
    finally:
        proc.wait()


def test_job_lock_reclaims_stale_pid_mocked(cfg, monkeypatch):
    """mock ``_pid_alive=False``（等价于持锁进程已死）：stale 锁被删并重建。"""
    lock = orch.JobLock("job-stale-mock", cfg)
    lock.lock_dir.mkdir(parents=True, exist_ok=True)
    lock.path.write_text("424242\t0.0\n", encoding="utf-8")
    monkeypatch.setattr(orch, "_pid_alive", lambda pid: False)
    assert lock.acquire(blocking=False) is True
    assert lock.held is True
    assert lock.path.read_text(encoding="utf-8").split("\t")[0] == str(os.getpid())
    lock.release()
    assert not lock.path.exists()


def test_job_lock_live_pid_is_not_stolen(cfg, monkeypatch):
    """活进程（本进程 pid / mock True）的锁不能被抢。"""
    # 真实探活：本进程 pid 恒为真；不存在的 pid 恒为假（不会误判活）
    assert orch._pid_alive(os.getpid()) is True
    assert orch._pid_alive(999999999) is False

    lock = orch.JobLock("job-live", cfg)
    lock.lock_dir.mkdir(parents=True, exist_ok=True)
    lock.path.write_text(f"{os.getpid()}\t0.0\n", encoding="utf-8")
    monkeypatch.setattr(orch, "_pid_alive", lambda pid: True)
    assert lock.acquire(blocking=False) is False
    assert lock.path.read_text(encoding="utf-8").split("\t")[0] == str(os.getpid())


def test_run_job_locked_returns_and_does_not_touch_db(monkeypatch, cfg):
    job_id = "job-busy"
    source = _make_source(cfg, job_id)
    _register(cfg, job_id, source)  # pending / attempts=0
    before = _get_job(cfg, job_id).to_dict()
    state = install_fakes(monkeypatch, cfg)

    lock = orch.JobLock(job_id, cfg)
    assert lock.acquire(blocking=False) is True
    try:
        result = orch.run_job(job_id, source, cfg)
    finally:
        lock.release()

    assert result["ok"] is False and result["locked"] is True and result["stage"] == "lock"
    assert "正在运行" in result["error"]
    assert state["calls"] == []
    after = _get_job(cfg, job_id).to_dict()
    assert after == before  # 不写 DB（status/stage/attempts 全不变）


# ================================================================ 失败 / quarantine
def test_decode_failure_quarantines_source(monkeypatch, cfg):
    job_id = "job-bad"
    source = _make_source(cfg, job_id)
    install_fakes(monkeypatch, cfg, fail="decode")

    result = orch.run_job(job_id, source, cfg)

    assert result["ok"] is False and result["stage"] == "decode"
    assert "decode-fail" in result["error"]
    job = _get_job(cfg, job_id)
    assert job is not None and job.status == STATUS_FAILED
    quarantine = cfg.path("quarantine_dir") / f"{job_id}.m4a"
    assert quarantine.is_file()
    assert quarantine.read_bytes() == source.read_bytes()
    assert source.is_file()  # 不删 data/ 原文件
    _statuses, rows = _stage_rows(cfg, job_id)
    decode_row = next(row for row in rows if row.stage == "decode")
    assert decode_row.status == STATUS_FAILED


def test_asr_failure_marks_failed_and_stops_downstream(monkeypatch, cfg):
    job_id = "job-asr-fail"
    source = _make_source(cfg, job_id)
    state = install_fakes(monkeypatch, cfg, fail="asr")

    result = orch.run_job(job_id, source, cfg)

    assert result["ok"] is False and result["stage"] == "asr"
    assert [call[0] for call in state["calls"]] == ["render", "asr"]
    job = _get_job(cfg, job_id)
    assert job is not None and job.status == STATUS_FAILED and "asr-fail" in job.error
    statuses, _rows = _stage_rows(cfg, job_id)
    assert statuses.get("asr") == STATUS_FAILED
    assert "diarize" not in statuses and "correct" not in statuses


# ================================================================ 清理
def test_success_cleanup_keeps_analyze_and_16k(monkeypatch, cfg):
    job_id = "job-clean"
    source = _make_source(cfg, job_id)
    cache = cfg.path("cache_dir")
    cache.mkdir(parents=True, exist_ok=True)
    keep = ["job-clean.16k.wav", "job-clean.analyze.json", "job-clean.decode.json"]
    for name in keep:
        (cache / name).write_bytes(b"keep-me")
    install_fakes(monkeypatch, cfg)

    result = orch.run_job(job_id, source, cfg)

    assert result["ok"] is True
    assert not (cache / f"{job_id}.clean16k.wav").exists()  # 成功后删 clean16k
    for name in keep:  # 16k / analyze / decode 必须保留
        assert (cache / name).is_file(), name


def test_keep_intermediates_preserves_clean16k(monkeypatch, cfg):
    job_id = "job-keep"
    source = _make_source(cfg, job_id)
    install_fakes(monkeypatch, cfg)

    result = orch.run_job(job_id, source, cfg, keep_intermediates=True)

    assert result["ok"] is True
    clean16k = cfg.path("cache_dir") / f"{job_id}.clean16k.wav"
    assert clean16k.is_file()


# ================================================================ batch_run
def test_batch_isolates_failures_and_continues(monkeypatch, cfg):
    source_a = _make_source(cfg, "job-a")
    source_b = _make_source(cfg, "job-b")
    state = install_fakes(monkeypatch, cfg, raise_at="asr", raise_job="job-a")
    events: List[Any] = []

    result = orch.batch_run(
        [("job-a", source_a), ("job-b", source_b)], cfg, resume=False,
        on_event=lambda job_id, stage, status, metrics: events.append(
            (job_id, stage, status)),
    )

    assert result["ok"] is False
    assert result["n"] == 2
    assert len(result["failures"]) == 1
    failure = result["failures"][0]
    assert failure["job_id"] == "job-a" and failure["stage"] == "asr"
    assert "asr-boom" in failure["error"]
    assert len(result["results"]) == 2  # 两个 job 都被尝试（一个失败一个成功）
    assert _get_job(cfg, "job-a").status == STATUS_FAILED
    assert _get_job(cfg, "job-b").status == STATUS_DONE
    assert len([call for call in state["calls"] if call[0] == "render"]) == 2
    # batch 的 on_event 多一个 job_id 参数（契约），且 export 只在最后合并后发一次
    assert {event[0] for event in events} == {"job-a", "job-b"}
    assert ("job-a", "asr", "failed") in events  # 失败也走 on_event
    assert sum(1 for event in events if event[:3] == ("job-b", "diarize", "done")) == 1
    assert sum(1 for event in events if event[:3] == ("job-b", "export", "done")) == 1


def test_batch_resume_skips_complete_jobs_without_stage_calls(monkeypatch, cfg):
    entries = []
    for index in (1, 2):
        job_id = f"job-resume-{index}"
        source = _make_source(cfg, job_id)
        _register(cfg, job_id, source, status=STATUS_DONE)
        _make_complete_artifacts(cfg, job_id, source)
        entries.append((job_id, source))
    state = install_fakes(monkeypatch, cfg)

    result = orch.batch_run(entries, cfg)  # resume 默认 True

    assert result["ok"] is True
    assert result["n"] == 2
    assert result["results"] == []
    assert sorted(result["skipped"]) == ["job-resume-1", "job-resume-2"]
    assert result["skipped_reasons"]["job-resume-1"] == "complete"
    assert result["stopped_reason"] == ""
    assert state["calls"] == []  # 不调用任何阶段（不重跑 render/ASR）
    # 缺的 stages 行补记（P4 遗留 diarize 行）
    statuses, _rows = _stage_rows(cfg, "job-resume-1")
    assert statuses == {stage: STATUS_DONE for stage in STAGES}


def test_batch_disk_low_stops_before_starting_new_job(monkeypatch, cfg):
    low_cfg = cfg.replace({"batch.disk_free_min_gb": 1e9})
    source = _make_source(low_cfg, "job-disk")
    state = install_fakes(monkeypatch, low_cfg)

    result = orch.batch_run([("job-disk", source)], low_cfg)

    assert result["stopped_reason"] == "disk_low"
    assert result["n"] == 0
    assert result["results"] == [] and result["failures"] == [] and result["skipped"] == []
    assert state["calls"] == []  # 没有启动新 job
    # mark_interrupted_pending 仍在启动时执行过；这里只确认没有登记/运行这个 job
    assert _get_job(low_cfg, "job-disk") is None


def test_batch_max_attempts_skips_failed_job(monkeypatch, cfg):
    job_id = "job-max"
    source = _make_source(cfg, job_id)
    _register(cfg, job_id, source, status=STATUS_FAILED, attempts=2)
    state = install_fakes(monkeypatch, cfg)

    result = orch.batch_run([(job_id, source)], cfg)  # 默认 max_attempts=2

    assert result["skipped"] == [job_id]
    assert result["skipped_reasons"][job_id] == "max_attempts"
    assert result["results"] == [] and result["failures"] == []
    assert state["calls"] == []
    job = _get_job(cfg, job_id)
    assert job.status == STATUS_FAILED and job.attempts == 2


def test_batch_limit_stops_after_n_jobs(monkeypatch, cfg):
    sources = [_make_source(cfg, f"job-limit-{index}") for index in range(3)]
    entries = [(f"job-limit-{index}", sources[index]) for index in range(3)]
    state = install_fakes(monkeypatch, cfg)

    result = orch.batch_run(entries, cfg, resume=False, limit=2)

    assert result["stopped_reason"] == "limit"
    assert result["n"] == 2
    assert len(result["results"]) == 2
    assert len([call for call in state["calls"] if call[0] == "render"]) == 2
    assert _get_job(cfg, "job-limit-2") is None  # 第三个没被取到


def test_batch_registers_path_inputs(monkeypatch, cfg):
    job_id = "job-path"
    source = _make_source(cfg, job_id)
    install_fakes(monkeypatch, cfg)

    result = orch.batch_run([source], cfg, resume=False)  # 纯路径，未登记

    assert len(result["results"]) == 1 and result["results"][0]["ok"] is True
    store = Storage(cfg.path("db_path"))
    store.init_db()
    try:
        job = store.get_job_by_source(source)
    finally:
        store.close()
    assert job is not None and job.status == STATUS_DONE


def test_batch_force_reruns_complete_job(monkeypatch, cfg):
    job_id = "job-force"
    source = _make_source(cfg, job_id)
    _register(cfg, job_id, source, status=STATUS_DONE)
    _make_complete_artifacts(cfg, job_id, source)
    _seed_stages(cfg, job_id, list(STAGES), {"seed": True})
    state = install_fakes(monkeypatch, cfg)

    result = orch.batch_run([(job_id, source)], cfg, force=True)

    assert result["skipped"] == []
    assert len(result["results"]) == 1 and result["results"][0]["ok"] is True
    assert state["calls"][0][0] == "render"  # 强制重跑
    job = _get_job(cfg, job_id)
    assert job.status == STATUS_DONE and job.attempts == 1
    _statuses, rows = _stage_rows(cfg, job_id)
    decode_row = next(row for row in rows if row.stage == "decode")
    assert decode_row.metrics.get("duration_s") == 123.4  # 旧 seed metrics 被覆盖


# ================================================================ P6 遗留 #1：模块阶段实时
def test_run_job_module_stages_are_live_in_jobs_stage(monkeypatch, cfg):
    """ASR/diarize/correct 跑动期间 ``jobs.stage`` 就是当前阶段（不再滞后到 loudness）。"""
    job_id = "job-live-stage"
    source = _make_source(cfg, job_id)
    _register(cfg, job_id, source)
    install_fakes(monkeypatch, cfg)

    seen: Dict[str, tuple] = {}

    def _spy(name: str):
        fn = getattr(orch, name)

        def wrapper(job_id_: str, cfg_: Any, **kwargs: Any) -> Any:
            store = Storage(cfg_.path("db_path"))
            store.init_db()
            try:
                job = store.get_job(job_id_)
                seen[name] = (job.stage, job.status)
            finally:
                store.close()
            return fn(job_id_, cfg_, **kwargs)

        return wrapper

    monkeypatch.setattr(orch, "transcribe_job", _spy("transcribe_job"))
    monkeypatch.setattr(orch, "diarize_job", _spy("diarize_job"))
    monkeypatch.setattr(orch, "correct_job", _spy("correct_job"))

    result = orch.run_job(job_id, source, cfg)

    assert result["ok"] is True
    assert seen["transcribe_job"] == ("asr", "running")
    assert seen["diarize_job"] == ("diarize", "running")
    assert seen["correct_job"] == ("correct", "running")


def test_run_job_does_not_reset_done_module_stage(monkeypatch, cfg):
    """已 done 的模块阶段（非 force）不被实时置 running，旧 metrics 保留。"""
    job_id = "job-keep-done-stage"
    source = _make_source(cfg, job_id)
    _register(cfg, job_id, source)
    _seed_stages(cfg, job_id, ["asr"], {"n_segments": 999, "chars": 1})
    install_fakes(monkeypatch, cfg)

    result = orch.run_job(job_id, source, cfg)

    assert result["ok"] is True
    _statuses, rows = _stage_rows(cfg, job_id)
    asr_row = next(row for row in rows if row.stage == "asr")
    assert asr_row.metrics.get("n_segments") == 999  # 旧指标未被 fake 的 1 覆盖
    assert asr_row.metrics.get("chars") == 1
