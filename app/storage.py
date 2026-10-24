# -*- coding: utf-8 -*-
"""SQLite 存储层：``jobs`` / ``stages`` 两张表（契约 §4.5），支持断点续跑与阶段状态机。

设计要点
--------
- **单进程**使用（与流水线并发策略一致），内部用 ``RLock`` 串行化写操作；
- ``stages`` 表严格按契约列定义（没有 error 列），**阶段级错误写进 metrics_json 的
  ``error`` 键**，同时由 :class:`StageResult` 还原成 ``.error`` 属性；
- ``start_stage`` 是幂等的：已 ``done`` 的阶段默认不会被重新置为 ``running``
  （断点续跑的安全网），需要重跑请显式 ``reset=True``；
- 数据库只存状态与指标，不存音频；中间产物路径由 ``work/cache`` 约定。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .models import (
    JOB_STATUSES,
    RESUMABLE_JOB_STATUSES,
    STAGES,
    STAGE_STATUSES,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    Job,
    StageResult,
    make_job_id,
    now_ts,
)

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    status      TEXT NOT NULL,
    stage       TEXT NOT NULL DEFAULT '',
    error       TEXT NOT NULL DEFAULT '',
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    config_hash TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);

CREATE TABLE IF NOT EXISTS stages (
    job_id       TEXT NOT NULL,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL,
    started_at   INTEGER,
    ended_at     INTEGER,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (job_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_stages_status ON stages(job_id, status);
"""


class StorageError(RuntimeError):
    """存储层错误（数据库版本不匹配、记录不存在等）。"""


def _dump_metrics(metrics: Optional[Mapping[str, Any]]) -> str:
    """metrics → JSON 字符串（numpy 标量等用 str 兜底，保证总能落库）。"""
    if not metrics:
        return "{}"
    return json.dumps(dict(metrics), ensure_ascii=False, default=str)


def _stage_order(stage: str) -> int:
    try:
        return STAGES.index(stage)
    except ValueError:
        return len(STAGES)


class Storage:
    """jobs/stages 读写入口。

    典型用法::

        with Storage(db_path) as store:
            store.init_db()
            job, created = store.upsert_job("data/a.m4a", config_hash=cfg.config_hash)
            store.start_stage(job.id, "decode")
            store.finish_stage(job.id, "decode", "done", metrics={"duration_s": 12.3})
    """

    def __init__(
        self,
        db_path: str | Path,
        default_config_hash: str = "",
        timeout_s: float = 30.0,
    ) -> None:
        self.db_path = str(db_path)
        self.default_config_hash = default_config_hash
        self.timeout_s = float(timeout_s)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------ 生命周期
    @property
    def is_memory(self) -> bool:
        return self.db_path in (":memory:", "")

    def connect(self) -> sqlite3.Connection:
        """建立连接（幂等）；父目录不存在会自动创建。"""
        with self._lock:
            if self._conn is not None:
                return self._conn
            if not self.is_memory:
                path = Path(self.db_path)
                path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout_s,
                isolation_level="DEFERRED",
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            if not self.is_memory:
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                except sqlite3.DatabaseError:  # pragma: no cover - 特殊文件系统兜底
                    pass
            self._conn = conn
            return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def __enter__(self) -> "Storage":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """串行化的事务上下文。"""
        with self._lock:
            conn = self.connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ------------------------------------------------------------ 建表/迁移
    def init_db(self) -> int:
        """建表 + 迁移；返回当前 schema 版本。可重复调用。"""
        with self._tx() as conn:
            conn.executescript(_SCHEMA_SQL)
            version = self._read_version(conn)
            if version == 0:
                self._set_version(conn, SCHEMA_VERSION)
                version = SCHEMA_VERSION
            version = self._migrate(conn, version)
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(version),),
            )
        return version

    @staticmethod
    def _read_version(conn: sqlite3.Connection) -> int:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])

    @staticmethod
    def _set_version(conn: sqlite3.Connection, version: int) -> None:
        conn.execute(f"PRAGMA user_version={int(version)}")

    def _migrate(self, conn: sqlite3.Connection, version: int) -> int:
        """按版本号逐级升级（当前只有 v1，留好扩展点）。"""
        if version > SCHEMA_VERSION:
            raise StorageError(
                f"数据库 schema 版本 {version} 高于程序支持的 {SCHEMA_VERSION}，"
                f"请升级程序或换一个 db 文件：{self.db_path}"
            )
        # 未来新增迁移：if version < 2: conn.execute("ALTER TABLE ...")
        self._set_version(conn, SCHEMA_VERSION)
        return SCHEMA_VERSION

    # ------------------------------------------------------------ jobs：写
    def _id_sources(self) -> Dict[str, str]:
        rows = self.connect().execute("SELECT id, source FROM jobs").fetchall()
        return {row["id"]: row["source"] for row in rows}

    def upsert_job(
        self,
        source: str | Path,
        job_id: Optional[str] = None,
        config_hash: str = "",
        status: Optional[str] = None,
    ) -> tuple[Job, bool]:
        """登记一个输入文件（scan 用）。已存在时**不改状态/不重置进度**，只刷新 source 与 hash。

        返回 ``(job, created)``。
        """
        src = str(Path(source))
        with self._tx() as conn:
            if job_id is None:
                job_id = make_job_id(src, taken=self._id_sources())
            if status is not None and status not in JOB_STATUSES:
                raise StorageError(f"非法 job 状态：{status}，可选：{JOB_STATUSES}")
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                job = Job(
                    id=job_id,
                    source=src,
                    status=status or STATUS_PENDING,
                    config_hash=config_hash or self.default_config_hash,
                )
                conn.execute(
                    "INSERT INTO jobs(id, source, status, stage, error, attempts, "
                    "created_at, updated_at, config_hash) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        job.id, job.source, job.status, job.stage, job.error,
                        job.attempts, job.created_at, job.updated_at, job.config_hash,
                    ),
                )
                return job, True
            existing = Job.from_row(row)
            new_hash = config_hash or existing.config_hash
            if existing.source != src or new_hash != existing.config_hash:
                conn.execute(
                    "UPDATE jobs SET source=?, config_hash=?, updated_at=? WHERE id=?",
                    (src, new_hash, now_ts(), job_id),
                )
                existing.source = src
                existing.config_hash = new_hash
                existing.updated_at = now_ts()
            return existing, False

    def add_job(self, job: Job) -> Job:
        """显式插入/覆盖一条 job（测试与导入用；已存在则整体覆盖）。"""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO jobs(id, source, status, stage, error, attempts, "
                "created_at, updated_at, config_hash) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET source=excluded.source, status=excluded.status, "
                "stage=excluded.stage, error=excluded.error, attempts=excluded.attempts, "
                "updated_at=excluded.updated_at, config_hash=excluded.config_hash",
                (
                    job.id, job.source, job.status, job.stage, job.error, job.attempts,
                    job.created_at, job.updated_at, job.config_hash,
                ),
            )
        return job

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        error: Optional[str] = None,
        attempts: Optional[int] = None,
        config_hash: Optional[str] = None,
        source: Optional[str] = None,
        touch: bool = True,
    ) -> Job:
        """按字段更新 job；返回更新后的 Job（不存在抛 StorageError）。"""
        if status is not None and status not in JOB_STATUSES:
            raise StorageError(f"非法 job 状态：{status}，可选：{JOB_STATUSES}")
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise StorageError(f"job 不存在：{job_id}")
            job = Job.from_row(row)
            sets: List[str] = []
            values: List[Any] = []
            for column, value in (
                ("status", status), ("stage", stage), ("error", error),
                ("attempts", attempts), ("config_hash", config_hash), ("source", source),
            ):
                if value is not None:
                    sets.append(f"{column}=?")
                    values.append(value)
                    setattr(job, column, value)
            if touch:
                sets.append("updated_at=?")
                values.append(now_ts())
                job.updated_at = now_ts()
            if sets:
                values.append(job_id)
                conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", values)
        return job

    def begin_attempt(self, job_id: str, stage: Optional[str] = None) -> Job:
        """开始一次执行：attempts+1、status=running、清空旧 error。"""
        with self._tx() as conn:
            row = conn.execute("SELECT attempts FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise StorageError(f"job 不存在：{job_id}")
            attempts = int(row["attempts"]) + 1
            conn.execute(
                "UPDATE jobs SET status=?, error='', attempts=?, stage=COALESCE(?, stage), "
                "updated_at=? WHERE id=?",
                (STATUS_RUNNING, attempts, stage, now_ts(), job_id),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def finish_job(self, job_id: str, status: str = STATUS_DONE, error: str = "") -> Job:
        return self.update_job(job_id, status=status, error=error)

    def fail_job(self, job_id: str, error: str) -> Job:
        return self.update_job(job_id, status=STATUS_FAILED, error=str(error))

    def mark_interrupted_pending(self) -> int:
        """把上次异常退出遗留的 ``running`` job 改回 ``pending``（启动时调用）。"""
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE status=?",
                (STATUS_PENDING, now_ts(), STATUS_RUNNING),
            )
            count = cur.rowcount or 0
            conn.execute(
                "UPDATE stages SET status=?, ended_at=? WHERE status=?",
                (STATUS_FAILED, now_ts(), STATUS_RUNNING),
            )
        return int(count)

    def delete_job(self, job_id: str) -> bool:
        with self._tx() as conn:
            conn.execute("DELETE FROM stages WHERE job_id=?", (job_id,))
            cur = conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            return bool(cur.rowcount)

    def rename_job(
        self,
        old_id: str,
        new_id: str,
        source: Optional[str] = None,
    ) -> bool:
        """把 job 的 id 改成 ``new_id``（同步 stages 表），finalize 整改目录名时使用。

        - 旧 id 不存在：若新 id 已存在视为已迁移（返回 False），否则抛 :class:`StorageError`；
        - 新旧都存在：抛 :class:`StorageError`（避免误合并两条任务）；
        - ``source`` 非空时顺带刷新（源文件路径不变，一般不需要）。
        """
        old_id, new_id = str(old_id), str(new_id)
        if old_id == new_id:
            return False
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (old_id,)).fetchone()
            if row is None:
                if conn.execute("SELECT 1 FROM jobs WHERE id=?", (new_id,)).fetchone() is not None:
                    return False  # 已迁移
                raise StorageError(f"job 不存在：{old_id}")
            if conn.execute("SELECT 1 FROM jobs WHERE id=?", (new_id,)).fetchone() is not None:
                raise StorageError(f"目标 job id 已存在：{new_id}（旧：{old_id}）")
            conn.execute("UPDATE stages SET job_id=? WHERE job_id=?", (new_id, old_id))
            if source is not None:
                conn.execute(
                    "UPDATE jobs SET id=?, source=?, updated_at=? WHERE id=?",
                    (new_id, str(Path(source)), now_ts(), old_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET id=?, updated_at=? WHERE id=?",
                    (new_id, now_ts(), old_id),
                )
            return True

    # ------------------------------------------------------------ jobs：读
    def get_job(self, job_id: str) -> Optional[Job]:
        row = self.connect().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job.from_row(row) if row is not None else None

    def get_job_by_source(self, source: str | Path) -> Optional[Job]:
        row = self.connect().execute(
            "SELECT * FROM jobs WHERE source=?", (str(Path(source)),)
        ).fetchone()
        return Job.from_row(row) if row is not None else None

    def require_job(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise StorageError(f"job 不存在：{job_id}（先执行 scan 或检查 id）")
        return job

    def list_jobs(
        self,
        status: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "created_at, id",
    ) -> List[Job]:
        sql = "SELECT * FROM jobs"
        where: List[str] = []
        params: List[Any] = []
        if status is not None:
            where.append("status=?")
            params.append(status)
        if statuses:
            where.append("status IN (" + ",".join("?" * len(statuses)) + ")")
            params.extend(statuses)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order}"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])
        rows = self.connect().execute(sql, params).fetchall()
        return [Job.from_row(row) for row in rows]

    def resume_candidates(self) -> List[Job]:
        """断点续跑候选：pending / running（上次中断）/ failed。"""
        return self.list_jobs(statuses=list(RESUMABLE_JOB_STATUSES))

    def pending_jobs(self) -> List[Job]:
        return self.list_jobs(status=STATUS_PENDING)

    def count_by_status(self) -> Dict[str, int]:
        counts = {status: 0 for status in JOB_STATUSES}
        rows = self.connect().execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        for row in rows:
            counts[row["status"]] = int(row["n"])
        counts["total"] = sum(counts[s] for s in JOB_STATUSES)
        return counts

    # ------------------------------------------------------------ stages
    def start_stage(self, job_id: str, stage: str, reset: bool = False) -> StageResult:
        """把阶段置为 running；已完成阶段默认不重跑（返回原记录）。"""
        if stage not in STAGES:
            raise StorageError(f"非法阶段：{stage}，可选：{STAGES}")
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone() is None:
                raise StorageError(f"job 不存在：{job_id}")
            row = conn.execute(
                "SELECT * FROM stages WHERE job_id=? AND stage=?", (job_id, stage)
            ).fetchone()
            if row is not None and not reset and row["status"] in (STATUS_DONE, STATUS_SKIPPED):
                return StageResult.from_row(row)
            now = now_ts()
            conn.execute(
                "INSERT INTO stages(job_id, stage, status, started_at, ended_at, metrics_json) "
                "VALUES(?,?,?,?,NULL,'{}') ON CONFLICT(job_id, stage) DO UPDATE SET "
                "status=excluded.status, started_at=excluded.started_at, ended_at=NULL, "
                "metrics_json='{}'",
                (job_id, stage, STATUS_RUNNING, now),
            )
            conn.execute(
                "UPDATE jobs SET stage=?, status=?, updated_at=? WHERE id=?",
                (stage, STATUS_RUNNING, now, job_id),
            )
            row = conn.execute(
                "SELECT * FROM stages WHERE job_id=? AND stage=?", (job_id, stage)
            ).fetchone()
        return StageResult.from_row(row)

    def finish_stage(
        self,
        job_id: str,
        stage: str,
        status: str = STATUS_DONE,
        metrics: Optional[Mapping[str, Any]] = None,
        error: str = "",
    ) -> StageResult:
        """结束阶段并落库指标（阶段级 error 一并写进 metrics_json）。"""
        if stage not in STAGES:
            raise StorageError(f"非法阶段：{stage}，可选：{STAGES}")
        if status not in STAGE_STATUSES:
            raise StorageError(f"非法阶段状态：{status}，可选：{STAGE_STATUSES}")
        payload: Dict[str, Any] = dict(metrics or {})
        if error:
            payload["error"] = str(error)
        now = now_ts()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM stages WHERE job_id=? AND stage=?", (job_id, stage)
            ).fetchone()
            if row is None:
                # 允许「先结束后开始」的补记（例如导入既有结果）
                conn.execute(
                    "INSERT INTO stages(job_id, stage, status, started_at, ended_at, metrics_json) "
                    "VALUES(?,?,?,NULL,?,?)",
                    (job_id, stage, status, now, _dump_metrics(payload)),
                )
            else:
                conn.execute(
                    "UPDATE stages SET status=?, ended_at=?, metrics_json=? "
                    "WHERE job_id=? AND stage=?",
                    (status, now, _dump_metrics(payload), job_id, stage),
                )
            conn.execute("UPDATE jobs SET updated_at=? WHERE id=?", (now, job_id))
            row = conn.execute(
                "SELECT * FROM stages WHERE job_id=? AND stage=?", (job_id, stage)
            ).fetchone()
        return StageResult.from_row(row)

    def fail_stage(
        self,
        job_id: str,
        stage: str,
        error: str,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> StageResult:
        return self.finish_stage(job_id, stage, STATUS_FAILED, metrics, error=str(error))

    def get_stage(self, job_id: str, stage: str) -> Optional[StageResult]:
        row = self.connect().execute(
            "SELECT * FROM stages WHERE job_id=? AND stage=?", (job_id, stage)
        ).fetchone()
        return StageResult.from_row(row) if row is not None else None

    def list_stages(self, job_id: str) -> List[StageResult]:
        rows = self.connect().execute(
            "SELECT * FROM stages WHERE job_id=?", (job_id,)
        ).fetchall()
        results = [StageResult.from_row(row) for row in rows]
        results.sort(key=lambda s: _stage_order(s.stage))
        return results

    def stage_map(self, job_id: str) -> Dict[str, str]:
        return {s.stage: s.status for s in self.list_stages(job_id)}

    def is_stage_done(self, job_id: str, stage: str) -> bool:
        result = self.get_stage(job_id, stage)
        return result is not None and result.status in (STATUS_DONE, STATUS_SKIPPED)

    def next_incomplete_stage(self, job_id: str) -> Optional[str]:
        """第一个未完成（非 done/skipped）的阶段名；全部完成返回 None。"""
        statuses = self.stage_map(job_id)
        for stage in STAGES:
            if statuses.get(stage) not in (STATUS_DONE, STATUS_SKIPPED):
                return stage
        return None

    def reset_stage(self, job_id: str, stage: str, downstream: bool = True) -> int:
        """把某阶段（及其后续阶段）置回 pending，供 retry 使用；返回重置条数。"""
        if stage not in STAGES:
            raise StorageError(f"非法阶段：{stage}，可选：{STAGES}")
        index = STAGES.index(stage)
        targets = STAGES[index:] if downstream else (stage,)
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE stages SET status=?, started_at=NULL, ended_at=NULL "
                f"WHERE job_id=? AND stage IN ({','.join('?' * len(targets))})",
                (STATUS_PENDING, job_id, *targets),
            )
            conn.execute(
                "UPDATE jobs SET status=?, error='', updated_at=?, stage=? WHERE id=?",
                (STATUS_PENDING, now_ts(), stage, job_id),
            )
            return int(cur.rowcount or 0)

    # ------------------------------------------------------------ 汇总
    def summary(self) -> Dict[str, Any]:
        """``app.cli status`` 用的汇总。"""
        return {
            "db": self.db_path,
            "counts": self.count_by_status(),
            "stages": [s for s in STAGES],
        }

    def jobs_snapshot(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """job 列表 + 阶段状态（供 status/Web UI 展示）。"""
        out: List[Dict[str, Any]] = []
        for job in self.list_jobs(limit=limit):
            item = job.to_dict()
            item["stages"] = self.stage_map(job.id)
            out.append(item)
        return out
