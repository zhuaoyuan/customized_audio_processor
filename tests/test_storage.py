# -*- coding: utf-8 -*-
"""P0-T2 存储层单测：建表/迁移、job 增删查改、stage 状态机、断点续跑查询。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.models import (
    STAGES,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    Job,
    StageResult,
)
from app.storage import SCHEMA_VERSION, Storage, StorageError


@pytest.fixture()
def store(tmp_path: Path) -> Storage:
    storage = Storage(tmp_path / "jobs.db", default_config_hash="hash-default")
    storage.init_db()
    yield storage
    storage.close()


# ---------------------------------------------------------------- 建表 / 迁移
def test_init_db_creates_schema_and_version(tmp_path: Path):
    db_path = tmp_path / "sub" / "jobs.db"
    assert not db_path.parent.exists()
    store = Storage(db_path)
    version = store.init_db()
    assert version == SCHEMA_VERSION
    assert db_path.is_file()
    # 可重复调用（幂等）
    assert store.init_db() == SCHEMA_VERSION
    tables = {
        row["name"]
        for row in store.connect().execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"jobs", "stages", "meta"} <= tables
    # 表结构与契约 §4.5 一致
    job_cols = {r["name"] for r in store.connect().execute("PRAGMA table_info(jobs)")}
    assert job_cols == {
        "id", "source", "status", "stage", "error", "attempts",
        "created_at", "updated_at", "config_hash",
    }
    stage_cols = {r["name"] for r in store.connect().execute("PRAGMA table_info(stages)")}
    assert stage_cols == {
        "job_id", "stage", "status", "started_at", "ended_at", "metrics_json",
    }
    assert store.connect().execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    store.close()


def test_open_newer_schema_rejected(tmp_path: Path):
    db = tmp_path / "future.db"
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION + 5}")
    conn.commit()
    conn.close()
    with pytest.raises(StorageError, match="高于程序支持"):
        Storage(db).init_db()


def test_persistence_across_connections(tmp_path: Path):
    db = tmp_path / "jobs.db"
    first = Storage(db)
    first.init_db()
    job, _ = first.upsert_job(tmp_path / "a.m4a")
    first.close()

    second = Storage(db)
    second.init_db()
    assert second.get_job(job.id) is not None
    assert second.count_by_status()["total"] == 1
    second.close()


# ---------------------------------------------------------------- jobs
def test_upsert_creates_pending_job(store: Storage, tmp_path: Path):
    job, created = store.upsert_job(tmp_path / "courses" / "第一讲.m4a")
    assert created
    assert job.status == STATUS_PENDING
    assert job.stage == ""
    assert job.attempts == 0
    assert job.config_hash == "hash-default"
    assert job.created_at > 0 and job.updated_at >= job.created_at
    # id 规则：hash8_可读短名
    assert job.id.startswith(job.id[:8])
    assert "第一讲" in job.id
    # 第二次 scan 不应重置已有进度
    store.update_job(job.id, status=STATUS_DONE, stage="export")
    again, created2 = store.upsert_job(tmp_path / "courses" / "第一讲.m4a")
    assert not created2
    assert again.id == job.id
    assert again.status == STATUS_DONE


def test_upsert_same_name_different_dir_gets_distinct_id(store: Storage, tmp_path: Path):
    a, _ = store.upsert_job(tmp_path / "d1" / "lecture.m4a")
    b, _ = store.upsert_job(tmp_path / "d2" / "lecture.m4a")
    assert a.id != b.id
    assert len(store.list_jobs()) == 2


def test_get_by_source_and_require(store: Storage, tmp_path: Path):
    src = tmp_path / "x.m4a"
    job, _ = store.upsert_job(src)
    assert store.get_job_by_source(src).id == job.id
    assert store.get_job_by_source(tmp_path / "nope.m4a") is None
    with pytest.raises(StorageError, match="job 不存在"):
        store.require_job("no-such-id")
    with pytest.raises(StorageError, match="job 不存在"):
        store.update_job("no-such-id", status=STATUS_RUNNING)


def test_list_filter_and_counts(store: Storage, tmp_path: Path):
    jobs = [store.upsert_job(tmp_path / f"s{i}.m4a")[0] for i in range(4)]
    store.update_job(jobs[0].id, status=STATUS_DONE)
    store.update_job(jobs[1].id, status=STATUS_FAILED, error="boom")
    assert len(store.list_jobs()) == 4
    assert len(store.list_jobs(status=STATUS_PENDING)) == 2
    assert len(store.list_jobs(statuses=[STATUS_DONE, STATUS_FAILED])) == 2
    assert len(store.list_jobs(limit=1)) == 1
    assert len(store.list_jobs(limit=2, offset=2)) == 2
    counts = store.count_by_status()
    assert counts[STATUS_DONE] == 1
    assert counts[STATUS_FAILED] == 1
    assert counts[STATUS_PENDING] == 2
    assert counts["total"] == 4
    with pytest.raises(StorageError, match="非法 job 状态"):
        store.update_job(jobs[0].id, status="weird")


def test_begin_attempt_and_fail(store: Storage, tmp_path: Path):
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    store.start_stage(job.id, "decode")
    store.fail_stage(job.id, "decode", "ffmpeg 失败")
    job = store.fail_job(job.id, "ffmpeg 失败")
    assert job.status == STATUS_FAILED
    job = store.begin_attempt(job.id, stage="decode")
    assert job.attempts == 1
    assert job.status == STATUS_RUNNING
    assert job.error == ""
    job = store.begin_attempt(job.id)
    assert job.attempts == 2


def test_delete_job_cascades_stages(store: Storage, tmp_path: Path):
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    store.start_stage(job.id, "decode")
    assert len(store.list_stages(job.id)) == 1
    assert store.delete_job(job.id) is True
    assert store.get_job(job.id) is None
    assert store.list_stages(job.id) == []
    assert store.delete_job(job.id) is False


# ---------------------------------------------------------------- stages
def test_stage_lifecycle_and_metrics(store: Storage, tmp_path: Path):
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    started = store.start_stage(job.id, "decode")
    assert started.status == STATUS_RUNNING
    assert started.started_at is not None
    assert started.ended_at is None
    # 开始阶段会把 job 置为 running
    assert store.get_job(job.id).status == STATUS_RUNNING
    assert store.get_job(job.id).stage == "decode"

    metrics = {"ok": True, "repaired": False, "duration_s": 3048.6, "nested": {"a": [1, 2]}}
    finished = store.finish_stage(job.id, "decode", STATUS_DONE, metrics)
    assert finished.status == STATUS_DONE
    assert finished.ended_at is not None
    assert finished.metrics["duration_s"] == pytest.approx(3048.6)
    assert finished.metrics["nested"]["a"] == [1, 2]
    assert finished.duration_s is not None and finished.duration_s >= 0
    # 往返读回一致
    again = store.get_stage(job.id, "decode")
    assert again.metrics == finished.metrics
    assert store.is_stage_done(job.id, "decode")
    assert not store.is_stage_done(job.id, "analyze")


def test_stage_error_roundtrip(store: Storage, tmp_path: Path):
    """stages 表按契约没有 error 列：阶段级错误存在 metrics_json.error 里。"""
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    store.start_stage(job.id, "decode")
    result = store.fail_stage(job.id, "decode", "Reserved bit set", {"attempt": 1})
    assert result.status == STATUS_FAILED
    assert result.error == "Reserved bit set"
    assert result.metrics["attempt"] == 1
    reloaded = store.get_stage(job.id, "decode")
    assert reloaded.error == "Reserved bit set"
    assert reloaded.metrics["error"] == "Reserved bit set"


def test_start_stage_idempotent_unless_reset(store: Storage, tmp_path: Path):
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    store.start_stage(job.id, "decode")
    store.finish_stage(job.id, "decode", STATUS_DONE, {"v": 1})
    # 断点续跑的安全网：已完成阶段不会被重新置为 running
    again = store.start_stage(job.id, "decode")
    assert again.status == STATUS_DONE
    assert again.metrics == {"v": 1}
    # 显式 reset 才会重跑
    reset = store.start_stage(job.id, "decode", reset=True)
    assert reset.status == STATUS_RUNNING
    assert reset.metrics == {}


def test_stage_order_and_next_incomplete(store: Storage, tmp_path: Path):
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    for stage in ("denoise", "decode"):
        store.start_stage(job.id, stage)
        store.finish_stage(job.id, stage, STATUS_DONE)
    ordered = [s.stage for s in store.list_stages(job.id)]
    assert ordered == ["decode", "denoise"]  # 按流水线顺序，不是写入顺序
    assert store.next_incomplete_stage(job.id) == "analyze"
    for stage in STAGES[1:]:
        store.finish_stage(job.id, stage, STATUS_SKIPPED)
    assert store.next_incomplete_stage(job.id) is None


def test_stage_validation(store: Storage, tmp_path: Path):
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    with pytest.raises(StorageError, match="非法阶段"):
        store.start_stage(job.id, "transcode")
    with pytest.raises(StorageError, match="非法阶段状态"):
        store.finish_stage(job.id, "decode", "ok-ish")
    with pytest.raises(StorageError, match="job 不存在"):
        store.start_stage("ghost", "decode")


def test_finish_unknown_stage_inserts_backfill(store: Storage, tmp_path: Path):
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    result = store.finish_stage(job.id, "export", STATUS_DONE, {"note": "backfill"})
    assert result.started_at is None
    assert result.ended_at is not None
    assert result.metrics["note"] == "backfill"


def test_reset_stage_downstream(store: Storage, tmp_path: Path):
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    for stage in STAGES:
        store.start_stage(job.id, stage)
        store.finish_stage(job.id, stage, STATUS_DONE)
    assert store.next_incomplete_stage(job.id) is None
    assert store.get_job(job.id).status == STATUS_RUNNING
    reset = store.reset_stage(job.id, "asr")
    assert reset == len(STAGES[STAGES.index("asr"):])
    statuses = store.stage_map(job.id)
    assert statuses["decode"] == STATUS_DONE
    assert statuses["loudness"] == STATUS_DONE
    assert statuses["asr"] == STATUS_PENDING
    assert statuses["export"] == STATUS_PENDING
    assert store.get_job(job.id).status == STATUS_PENDING
    assert store.next_incomplete_stage(job.id) == "asr"


# ---------------------------------------------------------------- 断点续跑
def test_resume_candidates_and_interrupted_recovery(store: Storage, tmp_path: Path):
    pending, _ = store.upsert_job(tmp_path / "p.m4a")
    done, _ = store.upsert_job(tmp_path / "d.m4a")
    store.update_job(done.id, status=STATUS_DONE)
    failed, _ = store.upsert_job(tmp_path / "f.m4a")
    store.fail_job(failed.id, "boom")
    running, _ = store.upsert_job(tmp_path / "r.m4a")
    store.update_job(running.id, status=STATUS_RUNNING, stage="denoise")
    store.start_stage(running.id, "denoise")

    candidates = {job.id for job in store.resume_candidates()}
    assert candidates == {pending.id, failed.id, running.id}

    # 进程被 kill 后重启：running → pending，running 阶段 → failed
    assert store.mark_interrupted_pending() == 1
    assert store.get_job(running.id).status == STATUS_PENDING
    assert store.get_stage(running.id, "denoise").status == STATUS_FAILED


def test_jobs_snapshot(store: Storage, tmp_path: Path):
    job, _ = store.upsert_job(tmp_path / "a.m4a")
    store.start_stage(job.id, "decode")
    store.finish_stage(job.id, "decode", STATUS_DONE)
    snapshot = store.jobs_snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["id"] == job.id
    assert snapshot[0]["stages"]["decode"] == STATUS_DONE
    assert store.summary()["counts"]["total"] == 1


def test_models_job_roundtrip():
    job = Job.new("abc12345_lesson", "data/a.m4a", config_hash="h")
    assert job.status == STATUS_PENDING
    assert job.to_dict()["id"] == "abc12345_lesson"
    stage = StageResult(job_id=job.id, stage="decode", status=STATUS_DONE,
                        started_at=100, ended_at=110, metrics={"ok": True})
    assert stage.duration_s == 10.0
    assert stage.to_dict()["metrics"] == {"ok": True}
