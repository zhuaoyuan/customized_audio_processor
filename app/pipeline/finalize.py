# -*- coding: utf-8 -*-
"""交付目录整理（流水线尾部 ``finalize``）。

把 ``<out_root>`` 下流水线产出的 job 目录整理成**可交付布局**::

    <out_root>/
      output/
        audio/<源原始文件名>                 # clean.m4a（移动，不留冗余副本）
        manuscript/<源原始文件名主体>.md      # manuscript.md 的副本（job 目录保留原稿）
      <hash8>_<源原始文件名主体>/            # job 目录：report/transcript/manuscript.md/...
      jobs.db  logs/

设计要点
--------
- **命名**：job 目录名与 ``jobs.db`` 的 id 统一改为
  ``<hash8>_<原始文件名主体>``（``app.models.make_job_id``），不再按 20 字截断；
  原始文件名取自 job 目录 ``report.json`` 的 ``source``。
- **幂等**：重复执行只补缺失/刷新 JSON，不重复搬移；``dry_run=True`` 只规划不动盘。
- **失败隔离**：单个 job 出错只记 error，不影响其它 job；返回汇总里 ``ok=False``。
- **可追溯**：``report.json`` 新增 ``delivery`` 段（相对 out_root 的 audio / manuscript 路径）；
  ``manuscript.json`` / ``compose_report.json`` 内的 job_id / 路径同步改写。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..logging_setup import get_logger
from ..models import make_job_id, source_basename, source_stem
from ..storage import Storage, StorageError

log = get_logger(__name__)

#: 交付根下的子目录
OUTPUT_DIRNAME = "output"
AUDIO_DIRNAME = "audio"
MANUSCRIPT_DIRNAME = "manuscript"

#: report.json 里的交付段（新增字段，旧程序忽略即可）
DELIVERY_KEY = "delivery"

JOB_REPORT = "report.json"
JOB_AUDIO = "clean.m4a"
JOB_MANUSCRIPT = "manuscript.md"
JOB_MANUSCRIPT_JSON = "manuscript.json"
JOB_COMPOSE_REPORT = "compose_report.json"

ProgressFn = Callable[[Dict[str, Any]], None]


class FinalizeError(RuntimeError):
    """交付目录整理失败（out_root 不存在等）。"""


# ================================================================ 路径帮助（也被 orchestrator/server 复用）
def _safe_relative(out_root: str | Path, rel: Any) -> Optional[Path]:
    """把 report 里记录的相对路径解析成 out_root 下的绝对路径（拒绝越界/绝对路径）。"""
    if not isinstance(rel, str) or not rel.strip():
        return None
    p = Path(rel.strip())
    if p.is_absolute() or ".." in p.parts:
        return None
    return Path(out_root) / p


def delivery_audio_path(out_root: str | Path, source: str | Path) -> Path:
    """``<out_root>/output/audio/<源原始文件名>``。"""
    return (
        Path(out_root) / OUTPUT_DIRNAME / AUDIO_DIRNAME / source_basename(source)
    )


def delivery_manuscript_path(out_root: str | Path, source: str | Path) -> Path:
    """``<out_root>/output/manuscript/<源原始文件名主体>.md``。"""
    return (
        Path(out_root) / OUTPUT_DIRNAME / MANUSCRIPT_DIRNAME / f"{source_stem(source)}.md"
    )


def _read_json_dict(path: str | Path) -> Optional[Dict[str, Any]]:
    """宽松读 JSON 对象；缺失/非法/非对象返回 None。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_delivery_paths(
    out_root: str | Path, report: Optional[Dict[str, Any]]
) -> Tuple[Optional[Path], Optional[Path]]:
    """从 report.json 的 ``delivery`` 段取 (audio, manuscript) 绝对路径；缺失返回 None。"""
    if not isinstance(report, dict):
        return None, None
    delivery = report.get(DELIVERY_KEY)
    if not isinstance(delivery, dict):
        return None, None
    return (
        _safe_relative(out_root, delivery.get("audio")),
        _safe_relative(out_root, delivery.get("manuscript")),
    )


def resolve_product(
    job_dir: str | Path,
    out_root: str | Path,
    name: str,
    *,
    source: str = "",
    report: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """找一个产物的**实际路径**：job 目录优先；``clean.m4a`` 回退到 finalize 后的交付目录。

    返回存在的路径；找不到返回 None。用于完成判定 / Web 产物列表 / 音频下载。
    """
    path = Path(job_dir) / name
    if path.is_file():
        return path
    if name != JOB_AUDIO:
        return None
    if report is None:
        report = _read_json_dict(Path(job_dir) / JOB_REPORT)
    audio, _ = read_delivery_paths(out_root, report)
    if audio is not None and audio.is_file():
        return audio
    src = source
    if not src and isinstance(report, dict):
        raw = report.get("source")
        src = raw if isinstance(raw, str) else ""
    if src:
        candidate = delivery_audio_path(out_root, src)
        if candidate.is_file():
            return candidate
    return None


def resolve_job_audio(
    job_dir: str | Path, out_root: str | Path, source: str = ""
) -> Optional[Path]:
    """``clean.m4a`` 实际位置（job 目录优先，其次 finalize 交付目录）。"""
    report = _read_json_dict(Path(job_dir) / JOB_REPORT)
    return resolve_product(job_dir, out_root, JOB_AUDIO, source=source, report=report)


# ================================================================ 内部工具
def _rel_posix(out_root: Path, path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(out_root).as_posix()
    except ValueError:  # pragma: no cover - path 不在 out_root 下
        return path.as_posix()


def _write_json(path: Path, payload: Any) -> bool:
    """写 JSON（UTF-8 / indent=2）；内容未变则不写、返回 False。

    换行风格沿用既有文件（流水线在本机写的是 CRLF；若是 LF 则保持 LF），
    避免 finalize 只改 job_id 却把整个文件的行尾刷成另一种。
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        raw = path.read_bytes()
    except OSError:
        raw = b""
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    newline = "\r\n" if b"\r\n" in raw[:65536] else ("\r\n" if not raw and os.linesep == "\r\n" else "\n")
    try:
        old_text = raw.decode("utf-8")
    except UnicodeDecodeError:  # pragma: no cover - 坏文件按不存在处理
        old_text = ""
    if old_text.replace("\r\n", "\n").rstrip("\n") == text:
        return False
    data = text.replace("\n", newline).encode("utf-8") if newline != "\n" else text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            pass
    return True


def _copy_if_changed(src: Path, dst: Path) -> bool:
    """仅当内容/大小/时间不同才复制（先写临时文件再替换，避免半截文件）。返回是否复制。"""
    try:
        st_src = src.stat()
        st_dst = dst.stat()
        if st_src.st_size == st_dst.st_size and int(st_src.st_mtime) == int(st_dst.st_mtime):
            return False
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            pass
    return True


def _replace_dir_in_text(text: str, old_dir: Path, new_dir: Path) -> str:
    """把 JSON 里记录的旧 job 目录路径改成新目录（先绝对路径，再裸目录名兜底）。"""
    if not isinstance(text, str):
        return text
    old_abs = str(old_dir)
    if old_abs in text:
        return text.replace(old_abs, str(new_dir))
    old_name = old_dir.name
    if old_name and old_name in text:
        return text.replace(old_name, new_dir.name)
    return text


def _claim_name(name: str, taken: Dict[str, str], source: str) -> str:
    """交付文件名去重：同名不同源时追加源路径 hash4（正常情况即原始文件名）。"""
    owner = taken.get(name)
    if owner is None or owner == source:
        taken[name] = source
        return name
    extra = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:4]
    head, dot, ext = name.rpartition(".")
    alt = f"{head}_{extra}{dot}{ext}" if dot else f"{name}_{extra}"
    taken[alt] = source
    return alt


def _discover_job_dirs(out_root: Path) -> Tuple[List[Path], List[str]]:
    """枚举 job 目录（含 report.json），跳过交付子目录/hidden，并给出可疑目录提示。"""
    job_dirs: List[Path] = []
    warnings: List[str] = []
    for child in sorted(out_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name == OUTPUT_DIRNAME:
            continue
        if (child / JOB_REPORT).is_file():
            job_dirs.append(child)
        elif any(
            (child / name).is_file()
            for name in (JOB_AUDIO, JOB_MANUSCRIPT, "transcript.json")
        ):
            warnings.append(f"跳过缺少 {JOB_REPORT} 的目录：{child.name}")
    return job_dirs, warnings


# ================================================================ 计划
@dataclass
class _JobPlan:
    old_dir: Path
    old_id: str
    source: str
    report: Dict[str, Any]
    new_id: str = ""
    audio_dst: Optional[Path] = None
    manuscript_dst: Optional[Path] = None
    error: str = ""


def _plan_jobs(
    out_root: Path, job_dirs: List[Path], warnings: List[str]
) -> List[_JobPlan]:
    taken: Dict[str, str] = {}
    audio_taken: Dict[str, str] = {}
    md_taken: Dict[str, str] = {}
    plans: List[_JobPlan] = []
    for job_dir in job_dirs:
        report = _read_json_dict(job_dir / JOB_REPORT)
        if report is None:
            warnings.append(f"跳过无法读取的 {JOB_REPORT}：{job_dir.name}")
            continue
        raw_source = report.get("source")
        source = str(raw_source).strip() if isinstance(raw_source, str) else ""
        if not source:
            warnings.append(f"跳过缺少 source 的 job 目录：{job_dir.name}")
            continue
        plan = _JobPlan(
            old_dir=job_dir, old_id=job_dir.name, source=source, report=report
        )
        plan.new_id = make_job_id(source, taken=taken)
        taken[plan.new_id] = source
        target = out_root / plan.new_id
        if plan.new_id != plan.old_id and target.exists():
            plan.error = f"目标目录已存在，未迁移：{plan.new_id}"
        plan.audio_dst = (
            out_root
            / OUTPUT_DIRNAME
            / AUDIO_DIRNAME
            / _claim_name(source_basename(source), audio_taken, source)
        )
        plan.manuscript_dst = (
            out_root
            / OUTPUT_DIRNAME
            / MANUSCRIPT_DIRNAME
            / _claim_name(f"{source_stem(source)}.md", md_taken, source)
        )
        plans.append(plan)
    return plans


# ================================================================ 执行
def _update_report(
    job_dir: Path,
    plan: _JobPlan,
    out_root: Path,
    audio_ok: bool,
    manuscript_ok: bool,
) -> None:
    report = _read_json_dict(job_dir / JOB_REPORT) or dict(plan.report)
    report["id"] = plan.new_id
    report[DELIVERY_KEY] = {
        "dir": plan.new_id,
        "audio": _rel_posix(out_root, plan.audio_dst) if audio_ok else "",
        "manuscript": (
            _rel_posix(out_root, plan.manuscript_dst) if manuscript_ok else ""
        ),
    }
    _write_json(job_dir / JOB_REPORT, report)


def _update_json_ids(
    job_dir: Path, plan: _JobPlan, filename: str, *, replace_source: bool
) -> None:
    path = job_dir / filename
    data = _read_json_dict(path)
    if data is None:
        return
    changed = False
    if data.get("job_id") != plan.new_id:
        data["job_id"] = plan.new_id
        changed = True
    if replace_source and isinstance(data.get("source"), str):
        new_src = _replace_dir_in_text(
            data["source"], plan.old_dir, job_dir
        )
        if new_src != data["source"]:
            data["source"] = new_src
            changed = True
    if changed:
        _write_json(path, data)


def _run_one(
    plan: _JobPlan,
    out_root: Path,
    *,
    dry_run: bool,
    store: Optional[Storage],
    summary: Dict[str, Any],
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "old_id": plan.old_id,
        "new_id": plan.new_id,
        "source": plan.source,
        "dir_renamed": False,
        "audio": "",
        "manuscript": "",
        "error": plan.error,
    }
    old_dir = plan.old_dir
    new_dir = out_root / plan.new_id
    job_dir = old_dir if dry_run else new_dir
    audio_dst = plan.audio_dst
    md_dst = plan.manuscript_dst

    if dry_run:
        item["dir_renamed"] = plan.new_id != plan.old_id
        item["audio"] = (
            "would-move"
            if (old_dir / JOB_AUDIO).is_file()
            else ("present" if audio_dst is not None and audio_dst.is_file() else "missing")
        )
        item["manuscript"] = (
            "would-copy"
            if (old_dir / JOB_MANUSCRIPT).is_file()
            else (
                "present"
                if md_dst is not None and md_dst.is_file()
                else "missing"
            )
        )
        summary["renamed"] += int(item["dir_renamed"])
        summary["audio_moved"] += int(item["audio"] == "would-move")
        summary["audio_present"] += int(item["audio"] == "present")
        summary["manuscript_copied"] += int(item["manuscript"] == "would-copy")
        summary["jobs"].append(item)
        if on_progress is not None:
            on_progress(item)
        return item

    try:
        if plan.error:
            raise FinalizeError(plan.error)
        if plan.new_id != plan.old_id:
            target = out_root / plan.new_id
            if target.exists():
                raise FinalizeError(f"目标目录已存在：{target.name}")
            old_dir.rename(target)
            item["dir_renamed"] = True
            summary["renamed"] += 1
        job_dir = new_dir

        audio_src = job_dir / JOB_AUDIO
        if audio_src.is_file():
            assert audio_dst is not None
            audio_dst.parent.mkdir(parents=True, exist_ok=True)
            if audio_dst.exists():
                audio_dst.unlink()
            shutil.move(str(audio_src), str(audio_dst))
            item["audio"] = "moved"
            summary["audio_moved"] += 1
        elif audio_dst is not None and audio_dst.is_file():
            item["audio"] = "present"
            summary["audio_present"] += 1
        else:
            item["audio"] = "missing"
            if (job_dir / JOB_COMPOSE_REPORT).is_file():
                summary["warnings"].append(
                    f"{plan.new_id}：{JOB_AUDIO} 缺失（audio 未归档）"
                )

        audio_ok = item["audio"] in ("moved", "present")
        md_src = job_dir / JOB_MANUSCRIPT
        if md_src.is_file():
            assert md_dst is not None
            copied = _copy_if_changed(md_src, md_dst)
            item["manuscript"] = "copied" if copied else "present"
            if copied:
                summary["manuscript_copied"] += 1
        elif md_dst is not None and md_dst.is_file():
            item["manuscript"] = "present"
        else:
            item["manuscript"] = "missing"
        manuscript_ok = item["manuscript"] in ("copied", "present")

        _update_report(job_dir, plan, out_root, audio_ok, manuscript_ok)
        _update_json_ids(
            job_dir, plan, JOB_MANUSCRIPT_JSON, replace_source=True
        )
        _update_json_ids(
            job_dir, plan, JOB_COMPOSE_REPORT, replace_source=False
        )

        if store is not None and plan.new_id != plan.old_id:
            store.rename_job(plan.old_id, plan.new_id)
    except (OSError, ValueError, StorageError, FinalizeError) as exc:
        item["error"] = str(exc)
        summary["ok"] = False
        summary["errors"].append(f"{plan.old_id}: {exc}")
        log.warning("finalize job 失败：%s：%s", plan.old_id, exc)

    summary["jobs"].append(item)
    if on_progress is not None:
        on_progress(item)
    return item


def finalize_run(
    out_root: str | Path,
    *,
    db_path: Optional[str | Path] = None,
    dry_run: bool = False,
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """整理 ``out_root`` 下的产物目录（幂等）。

    参数
    ----
    out_root : 产物根（如 ``out/1``；也可直接传单次运行的 out 根）
    db_path  : jobs.db 路径；缺省/文件不存在则跳过 DB id 更新
    dry_run  : 只规划打印，不改盘
    on_progress : 每个 job 处理完回调一次（dict，与返回 ``jobs`` 元素同构）

    返回形如::

        {"ok": True, "out_dir": "...", "dry_run": False, "n_jobs": 14,
         "renamed": 14, "audio_moved": 14, "audio_present": 0,
         "manuscript_copied": 14, "jobs": [...], "warnings": [], "errors": []}
    """
    root = Path(out_root).expanduser()
    summary: Dict[str, Any] = {
        "ok": True,
        "out_dir": str(root),
        "dry_run": bool(dry_run),
        "n_jobs": 0,
        "renamed": 0,
        "audio_moved": 0,
        "audio_present": 0,
        "manuscript_copied": 0,
        "jobs": [],
        "warnings": [],
        "errors": [],
    }
    if not root.is_dir():
        raise FinalizeError(f"产物根目录不存在：{root}")
    job_dirs, warnings = _discover_job_dirs(root)
    summary["warnings"].extend(warnings)
    plans = _plan_jobs(root, job_dirs, summary["warnings"])
    summary["n_jobs"] = len(plans)
    if not plans:
        return summary

    store: Optional[Storage] = None
    if db_path is not None and Path(str(db_path)).is_file():
        store = Storage(str(db_path))
    try:
        for plan in plans:
            _run_one(
                plan,
                root,
                dry_run=dry_run,
                store=store,
                summary=summary,
                on_progress=on_progress,
            )
    finally:
        if store is not None:
            store.close()
    if summary["errors"]:
        summary["ok"] = False
    return summary
