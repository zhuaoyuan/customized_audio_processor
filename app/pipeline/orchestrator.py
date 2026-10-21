# -*- coding: utf-8 -*-
"""P6-T1 编排器：把 P1–P5 串成**可断点续跑、失败隔离**的单 job / 批量流水线。

固定契约（``app/cli.py`` 与 ``app/server.py`` 依赖，**不得改名/改返回结构**）::

    PRODUCT_FILES: tuple[str, ...]
    artifacts(job_id, cfg, *, out_dir=None) -> list[dict]
    is_complete(job_id, cfg, *, out_dir=None) -> bool
    class JobLock: JobLock(job_id, cfg, lock_dir=None); acquire/release/held/with
    run_job(job_id, source, cfg, *, force=False, out_dir=None, cache_dir=None,
            on_event=None, keep_intermediates=False) -> dict
    batch_run(inputs, cfg, *, resume=True, limit=None, force=False, on_event=None,
              keep_intermediates=False) -> dict

设计决定（见 ``docs/工程日志.md``「P6/P7 开工基准与决定」）
----------------------------------------------------------
1. **阶段顺序**：``decode → analyze → denoise → trim → loudness → asr → diarize →
   correct → export``（``app.models.STAGES``）。``export`` 的语义是「最终全部产物落盘」
   （audio + transcript + corrections + md/srt/txt），在最后**统一落一条**。
2. **export 名复用**：P3 的 audio 导出与 P5-T4 的 text 导出回调都叫 ``export``，
   这里都把它们的 metrics 只**缓存**（``audio_export`` / ``text_export``），不单独落库；
   全部阶段结束后 ``start_stage("export", reset=True)`` + ``finish_stage("export", done,
   merged)``，merged 为 audio（tp_dbtp/asl_out_dbfs/size_bytes）+ text（md/srt/txt +
   n_written）+ transcript（n_segments/chars）+ corrections（total/rejected）+
   speakers（n/method）的**扁平**指标（与 P5-T4 落库风格一致）。
3. **resume 以产物为准**：``force=False`` 且 :func:`is_complete` 为真 → 整 job ``skipped``，
   **不调用任何阶段**；只把 stages 表里缺失的行按 done 补记（``metrics.backfilled=True``，
   例如 P4 done 任务缺 ``diarize`` 行），不整条重跑。产物不全时逐阶段幂等续跑：
   render/ASR/diarize/correct/export_text 各自会跳过已完成部分。
4. **锁**：``work/locks/<job_id>.lock``（``O_CREAT|O_EXCL`` 写 ``pid\t<ts>``）；持锁进程
   已死（Windows ``OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`` / POSIX ``os.kill(pid,0)``）
   → stale 清除后重抢。``run_job`` 抢不到直接返回 ``{ok:false, locked:true, stage:"lock"}``，
   **不写 DB**（Storage 在抢锁之后才创建）。
5. **清理**：成功后只删 ``<id>.clean16k.wav``；``<id>.16k.wav`` / ``<id>.analyze.json`` /
   ``<id>.decode.json`` **必须保留**（否则 resume 要重跑分析）；``keep_intermediates=True``
   时连 clean16k 也保留；render 自己的 44.1k/denoise/normalized 中间 wav 由 ``render_job``
   自己清；**不调用** ``decode.cleanup_cache``。
6. **batch**：启动时 ``Storage.init_db()`` + ``mark_interrupted_pending()``（``run_job``
   不调用）；取新 job 前检查 ``batch.disk_free_min_gb`` 磁盘水位（不足 → 已开始的做完、
   ``stopped_reason="disk_low"``）；``attempts >= batch.max_attempts`` 且 failed 且非 force
   → 跳过（``skipped_reasons[job_id]="max_attempts"``）；单个 job 失败/异常只记 failures，
   继续下一个；decode 阶段失败把源文件复制到 ``quarantine_dir/<job_id><原后缀>``（不删 data/）。

返回结构补充（契约允许的附加键）
--------------------------------
- ``run_job`` 的 ``stages`` 是 ``{stage: status}`` 字典；``artifacts`` 元素为
  ``{"name","size","mtime"}``（mtime 为 Unix 秒浮点）。
- ``batch_run`` 额外返回 ``skipped_reasons``：``{job_id: "complete"|"max_attempts"|"locked"}``，
  ``skipped`` 仍是 job id 列表（契约字段）；``ok`` 表示「没有 failures」（磁盘水位/limit
  导致的提前收工不算失败，用 ``stopped_reason`` 表达）。
- ``locked`` 的 job 在 batch 里按 skipped 处理（另一个进程正在跑，不是失败）。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..models import (
    STAGE_EXPORT,
    STAGES,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
)
from ..storage import Storage
from .asr import CLEAN16K_SUFFIX, transcribe_job
from .correct import correct_job
from .diarize import diarize_job
from .export import export_text, render_job

log = logging.getLogger(__name__)

#: 完整产物清单（契约：8 件齐全才算 done；顺序即展示/校验顺序）
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

#: 文本产物文件名（合并指标用）
_TEXT_FILES: Dict[str, str] = {
    "md": "transcript.md",
    "srt": "transcript.srt",
    "txt": "transcript.txt",
}

#: report.json 里必须是对象的三个段（缺一不可判完成）
_REPORT_SECTIONS: Tuple[str, ...] = ("asr", "correct", "speakers")

#: 锁轮询间隔（秒）
_LOCK_POLL_S = 0.05


# ================================================================ 小工具
def _out_root(cfg: Any, out_dir: str | Path | None = None) -> Path:
    """输出根目录：显式 ``out_dir`` 优先，否则 ``cfg.path("out_dir")``。"""
    return Path(out_dir) if out_dir is not None else Path(cfg.path("out_dir"))


def _cache_root(cfg: Any, cache_dir: str | Path | None = None) -> Path:
    """缓存根目录：显式 ``cache_dir`` 优先，否则 ``cfg.path("cache_dir")``。"""
    return Path(cache_dir) if cache_dir is not None else Path(cfg.path("cache_dir"))


def _job_dir(out_root: Path, job_id: str) -> Path:
    """``<out_root>/<job_id>``；拒绝带目录分隔符的 id（防路径穿越）。"""
    name = str(job_id or "")
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError(f"非法 job id：{job_id!r}")
    return Path(out_root) / name


def _file_size(path: Path) -> Optional[int]:
    try:
        return int(Path(path).stat().st_size)
    except OSError:
        return None


def _pick(*values: Any) -> Any:
    """取第一个非 None 的值（``0`` / ``0.0`` 也算有效值）。"""
    for value in values:
        if value is not None:
            return value
    return None


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json_dict(path: Path) -> Dict[str, Any]:
    """读 JSON 对象；缺失/非法/顶层非 dict → 空 dict（只读，不抛）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_report(job_out: Path) -> Dict[str, Any]:
    return _read_json_dict(Path(job_out) / "report.json")


def _emit(on_event: Optional[Callable[[str, str, Dict[str, Any]], None]],
          stage: str, status: str, metrics: Optional[Dict[str, Any]] = None) -> None:
    """安全调用 ``on_event(stage, status, metrics)``（回调异常只打 debug）。"""
    if on_event is None:
        return
    try:
        on_event(str(stage), str(status), dict(metrics or {}))
    except Exception:  # pragma: no cover - 回调失败不影响主流程
        log.debug("on_event 回调失败", exc_info=True)


# ================================================================ 产物 / 完成判定
def artifacts(job_id: str, cfg: Any, *, out_dir: str | Path | None = None) -> List[Dict[str, Any]]:
    """列出 ``out/<id>/`` 下**存在**的 :data:`PRODUCT_FILES`。

    返回 ``[{"name", "size", "mtime"}]``（mtime 为 Unix 秒浮点），顺序同
    :data:`PRODUCT_FILES`；``out_dir`` 缺省取 ``cfg.path("out_dir")``。
    """
    try:
        job_out = _job_dir(_out_root(cfg, out_dir), job_id)
    except ValueError:
        return []
    items: List[Dict[str, Any]] = []
    for name in PRODUCT_FILES:
        path = job_out / name
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        items.append({
            "name": name,
            "size": int(stat.st_size),
            "mtime": round(float(stat.st_mtime), 3),
        })
    return items


def _missing_products(job_id: str, cfg: Any,
                      out_dir: str | Path | None = None) -> Tuple[List[str], List[str]]:
    """返回 ``(缺失文件名, report 缺失/非对象的段名)``；用于完成判定与错误信息。"""
    try:
        job_out = _job_dir(_out_root(cfg, out_dir), job_id)
    except ValueError:
        return list(PRODUCT_FILES), list(_REPORT_SECTIONS)
    missing = [name for name in PRODUCT_FILES if not (job_out / name).is_file()]
    report = _read_report(job_out)
    missing_sections = [key for key in _REPORT_SECTIONS
                        if not isinstance(report.get(key), dict)]
    return missing, missing_sections


def is_complete(job_id: str, cfg: Any, *, out_dir: str | Path | None = None) -> bool:
    """完成判定：8 件产物齐全 **且** ``report.asr/correct/speakers`` 都是对象。

    只看产物、不看 stages 表（P3/P5 共用 ``export`` 名，旧 metrics 不可信）。
    """
    missing, missing_sections = _missing_products(job_id, cfg, out_dir)
    return not missing and not missing_sections


# ================================================================ 文件锁
def _default_lock_dir(cfg: Any) -> Path:
    """默认锁目录：``<work_dir>/locks``（缺 work_dir 时退回 out/cache 同级 locks）。"""
    try:
        return Path(cfg.path("work_dir")) / "locks"
    except Exception:  # pragma: no cover - 非标准配置对象兜底
        pass
    for key in ("cache_dir", "out_dir"):
        try:
            base = cfg.path(key)
        except Exception:
            continue
        if base:
            return Path(base).parent / "locks"
    return Path.cwd() / "work" / "locks"


def _safe_lock_name(job_id: str) -> str:
    """job id → 安全文件名（保留中日韩/字母数字/``_-``，其余换 ``_``）。"""
    name = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(job_id))
    return name.strip("._") or "job"


def _parse_lock_pid(text: str) -> Optional[int]:
    """锁文件内容 ``pid\\tts`` → pid；非法返回 None。"""
    first = (text or "").strip().splitlines()
    if not first:
        return None
    token = first[0].split("\t")[0].strip()
    try:
        return int(token)
    except ValueError:
        return None


#: 懒加载的 kernel32（已设好函数原型；模块级缓存，避免每次探活都重设 argtypes）
_KERNEL32: Any = None

#: Windows API 常量
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def _kernel32() -> Any:
    """取已设置正确原型的 ``kernel32``（``ctypes.WinDLL(use_last_error=True)``）。

    必须显式设 restype/argtypes：``OpenProcess`` 默认 restype 是 32 位 ``c_int``，
    64 位 HANDLE 会被截断导致误判；``GetExitCodeProcess`` 需要 ``POINTER(c_uint32)``。
    """
    global _KERNEL32
    if _KERNEL32 is not None:
        return _KERNEL32
    import ctypes

    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.OpenProcess.restype = ctypes.c_void_p
    lib.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    lib.CloseHandle.restype = ctypes.c_int
    lib.CloseHandle.argtypes = (ctypes.c_void_p,)
    lib.GetExitCodeProcess.restype = ctypes.c_int
    lib.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    _KERNEL32 = lib
    return lib


def _windows_pid_alive(pid: int) -> bool:
    """Windows 探活：``OpenProcess`` 成功 **且** 退出码 == STILL_ACTIVE 才算存活。

    坑：进程已终止但内核进程对象尚未回收（例如崩溃后由其它仍打开的 handle 持有）时，
    PID 仍能 ``OpenProcess`` 成功（``Get-CimInstance Win32_Process`` 已查不到它），此时
    必须看 ``GetExitCodeProcess``：返回真实退出码（≠259）→ 已死。只凭 OpenProcess 成功
    判活会让 stale 锁永远清不掉（真实 kill 后无法 resume）。
    """
    import ctypes

    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
    if not handle:
        # ERROR_INVALID_PARAMETER（pid 不存在）等：按已死处理（本地单用户工具）
        return False
    try:
        exit_code = ctypes.c_uint32(0)
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            # 查询失败：仅「无权限」保守判活（不误抢活进程的锁），其余按已死处理
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        return int(exit_code.value) == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    """探测进程是否存活。

    - Windows：``OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`` + ``GetExitCodeProcess``
      （已终止但对象未回收的 PID 会被判死，见 :func:`_windows_pid_alive`）；
    - POSIX：``os.kill(pid, 0)``（``ProcessLookupError``=死；``PermissionError``=活）。
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform.startswith("win"):
        try:
            return _windows_pid_alive(pid)
        except Exception:  # pragma: no cover - ctypes/API 不可用时保守判活
            log.debug("Windows 进程探活失败，保守判活：pid=%s", pid, exc_info=True)
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class JobLock:
    """单 job 文件锁：``<lock_dir>/<job_id>.lock``，跨进程防重复跑。

    - ``acquire(blocking=False)``：立即尝试，失败返回 False；
    - ``acquire(blocking=True)``：轮询到成功；``timeout_s > 0`` 时超时返回 False，
      ``timeout_s <= 0`` 时一直等；
    - stale 判定：锁文件里的 pid 已死 → 删除后重抢（``O_EXCL`` 保证只有一个赢家）；
    - 支持 ``with``：非阻塞抢锁，抢不到抛 ``RuntimeError``（不静默）。
    """

    def __init__(self, job_id: str, cfg: Any, lock_dir: str | Path | None = None) -> None:
        self.job_id = str(job_id)
        self.cfg = cfg
        self.lock_dir = Path(lock_dir) if lock_dir is not None else _default_lock_dir(cfg)
        self.path = self.lock_dir / f"{_safe_lock_name(self.job_id)}.lock"
        self._held = False
        self._pid = os.getpid()

    @property
    def held(self) -> bool:
        return self._held

    # -- 内部
    def _try_create(self) -> bool:
        try:
            self.lock_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except OSError as exc:  # pragma: no cover - 权限/磁盘异常
            log.warning("创建锁文件失败：%s：%s", self.path, exc)
            return False
        try:
            os.write(fd, f"{self._pid}\t{time.time():.3f}\n".encode("utf-8"))
        finally:
            os.close(fd)
        log.debug("已获取 job 锁：%s（pid=%s）", self.path, self._pid)
        return True

    def _read_pid(self) -> Optional[int]:
        try:
            return _parse_lock_pid(self.path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return None

    def _clear_if_stale(self) -> bool:
        """持锁 pid 已死则删除锁文件并返回 True。"""
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return True  # 文件已消失：让调用方重试创建
        pid = _parse_lock_pid(text)
        if pid is None:
            log.warning("锁文件内容非法，按 stale 处理：%s", self.path)
        elif _pid_alive(pid):
            return False
        try:
            self.path.unlink()
            log.info("清理 stale 锁（pid=%s 已退出）：%s", pid, self.path)
        except OSError:  # pragma: no cover - 并发删除
            return False
        return True

    # -- 对外
    def acquire(self, blocking: bool = False, timeout_s: float = 0.0) -> bool:
        """抢锁；成功 True。规则见类 docstring。"""
        deadline: Optional[float] = None
        timeout = float(timeout_s or 0.0)
        if timeout > 0:
            deadline = time.monotonic() + timeout
        while True:
            if self._try_create() or (self._clear_if_stale() and self._try_create()):
                self._held = True
                return True
            if not blocking and deadline is None:
                return False
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            time.sleep(_LOCK_POLL_S if remaining is None else min(_LOCK_POLL_S, remaining))

    def release(self) -> None:
        """释放锁（只删内容仍是自己 pid 的锁文件，避免误删别人的锁）。"""
        if not self._held:
            return
        self._held = False
        try:
            if self._read_pid() in (None, self._pid):
                self.path.unlink(missing_ok=True)
                log.debug("已释放 job 锁：%s", self.path)
        except OSError:  # pragma: no cover
            log.debug("释放锁失败：%s", self.path, exc_info=True)

    def __enter__(self) -> "JobLock":
        if not self.acquire(blocking=False):
            raise RuntimeError(f"job 正在运行（锁被占用）：{self.path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


# ================================================================ 阶段落库
def _record_stage_done(store: Storage, job_id: str, stage: str,
                       metrics: Optional[Dict[str, Any]], *, force: bool) -> None:
    """阶段 done 落库：已 done 且非 force 时**不覆盖**旧 metrics（P3 的 5 行空 metrics 保留）。"""
    if not force and store.is_stage_done(job_id, stage):
        log.debug("阶段已 done，保留旧指标：%s/%s", job_id, stage)
        return
    store.start_stage(job_id, stage, reset=True)
    store.finish_stage(job_id, stage, STATUS_DONE, dict(metrics or {}))


def _mark_failed(store: Storage, job_id: str, stage: str, error: str,
                 state: Dict[str, Any],
                 on_event: Optional[Callable[[str, str, Dict[str, Any]], None]]) -> None:
    """失败落库：``fail_stage`` + ``fail_job``（整个 run 只调一次 fail_job）+ 回调。"""
    message = str(error or f"{stage} 阶段失败")
    store.fail_stage(job_id, stage, message)
    if not state.get("fail_job_done"):
        store.fail_job(job_id, message)
        state["fail_job_done"] = True
    state["failed"] = stage
    state["failed_error"] = message
    _emit(on_event, stage, "failed", {"error": message})


def _invoke(stage: str, fn: Callable[[], Any], *, store: Storage, job_id: str,
            state: Dict[str, Any], on_event: Optional[Callable[[str, str, Dict[str, Any]], None]],
            timings: Dict[str, float], timings_key: str) -> Dict[str, Any]:
    """执行一个阶段（兜底捕获异常并落失败），返回阶段结果 dict；耗时写进 ``timings``。"""
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # 阶段函数约定失败不抛，这里兜底保证 batch 不被单个 job 打断
        message = str(exc) or exc.__class__.__name__
        log.error("阶段 %s 抛出异常：%s：%s", stage, job_id, message)
        _mark_failed(store, job_id, stage, message, state, on_event)
        result = {"ok": False, "stage": stage, "error": message}
    timings[timings_key] = round(time.perf_counter() - started, 2)
    if not isinstance(result, dict):
        message = f"{stage} 阶段返回非 dict：{type(result).__name__}"
        _mark_failed(store, job_id, stage, message, state, on_event)
        result = {"ok": False, "stage": stage, "error": message}
    return result


# ================================================================ 指标合并
def _text_file_metrics(job_out: Path) -> Dict[str, Any]:
    """文本产物的落盘快照：``formats`` + ``md/srt/txt`` 字节数 + ``n_written``（txt 非空行数）。"""
    sizes: Dict[str, Optional[int]] = {fmt: _file_size(Path(job_out) / name)
                                       for fmt, name in _TEXT_FILES.items()}
    n_written = 0
    try:
        with open(Path(job_out) / "transcript.txt", "r", encoding="utf-8",
                  errors="replace") as handle:
            n_written = sum(1 for line in handle if line.strip())
    except OSError:
        n_written = 0
    return {
        "formats": [fmt for fmt in ("md", "srt", "txt") if sizes[fmt] is not None],
        "md": sizes["md"],
        "srt": sizes["srt"],
        "txt": sizes["txt"],
        "n_written": int(n_written),
    }


def _build_merged_metrics(*, job_id: str, job_out: Path,
                          audio_export: Dict[str, Any], text_export: Dict[str, Any],
                          report: Dict[str, Any],
                          asr_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """拼最终 ``export`` 行的合并指标（audio + text + transcript + corrections + speakers）。

    取值优先级：阶段回调 metrics → report.json 既有字段 → 磁盘产物兜底。
    """
    report = report if isinstance(report, dict) else {}
    levels = report.get("levels") if isinstance(report.get("levels"), dict) else {}
    output = report.get("output") if isinstance(report.get("output"), dict) else {}
    asr = report.get("asr") if isinstance(report.get("asr"), dict) else {}
    speakers = report.get("speakers") if isinstance(report.get("speakers"), dict) else {}
    correct = report.get("correct") if isinstance(report.get("correct"), dict) else {}
    audio_export = audio_export if isinstance(audio_export, dict) else {}
    text_export = text_export if isinstance(text_export, dict) else {}
    asr_metrics = asr_metrics if isinstance(asr_metrics, dict) else {}

    # audio（P3 的 export 回调；render 被 skip / 旧产物时退回 report）
    tp_dbtp = _pick(audio_export.get("tp_dbtp"), levels.get("true_peak_out_dbtp"))
    asl_out = _pick(audio_export.get("asl_out_dbfs"), levels.get("asl_out_dbfs"))
    size_bytes = _pick(audio_export.get("size_bytes"), output.get("size_bytes"),
                       _file_size(job_out / "clean.m4a"))

    # text（P5-T4 的 export 回调；skip 时用磁盘快照兜底）
    text_files = _text_file_metrics(job_out)
    n_written = _as_int(text_export.get("n_written"), 0) or 0
    if n_written <= 0:
        n_written = text_files["n_written"]
    formats = text_export.get("formats")
    if not isinstance(formats, (list, tuple)) or not formats:
        formats = text_files["formats"]

    # transcript / corrections / speakers
    n_segments = _pick(asr_metrics.get("n_segments"), asr.get("n_segments"),
                       text_export.get("n_segments"))
    chars = _pick(asr_metrics.get("chars"), asr.get("chars"))
    corrections = _read_json_dict(job_out / "corrections.json")
    rejected_chunks = corrections.get("rejected_chunks")
    total = _pick(corrections.get("total"), correct.get("edits"))
    rejected = _pick(len(rejected_chunks) if isinstance(rejected_chunks, list) else None,
                     correct.get("rejected"))
    n_speakers = _pick(speakers.get("n"), 1)
    method = _pick(speakers.get("method"), "none")

    return {
        # audio
        "tp_dbtp": tp_dbtp,
        "asl_out_dbfs": asl_out,
        "size_bytes": _as_int(size_bytes, size_bytes if isinstance(size_bytes, int) else None),
        # text
        "formats": list(formats),
        "md": text_files["md"],
        "srt": text_files["srt"],
        "txt": text_files["txt"],
        "n_written": int(n_written),
        # transcript
        "n_segments": _as_int(n_segments),
        "chars": _as_int(chars),
        # corrections
        "total": _as_int(total, 0),
        "rejected": _as_int(rejected, 0),
        # speakers
        "n": _as_int(n_speakers, 1),
        "method": str(method),
    }


def _backfill_stage_rows(store: Storage, job_id: str, job_out: Path) -> List[str]:
    """产物齐全但 stages 缺行时补记 done（不执行任何阶段，例如缺 ``diarize`` 的 P4 任务）。

    已有 done/skipped 的行**不动**；补记行带 ``metrics.backfilled=True``；``export`` 行
    用合并指标（可读 report/产物算出）补记。返回补记的阶段名列表。
    """
    statuses = store.stage_map(job_id)
    report = _read_report(job_out)
    filled: List[str] = []
    for stage in STAGES:
        if statuses.get(stage) in (STATUS_DONE, STATUS_SKIPPED):
            continue
        if stage == STAGE_EXPORT:
            metrics = _build_merged_metrics(
                job_id=job_id, job_out=job_out,
                audio_export={}, text_export={}, report=report,
            )
        else:
            metrics = {}
        metrics["backfilled"] = True
        store.finish_stage(job_id, stage, STATUS_DONE, metrics)
        filled.append(stage)
    if filled:
        log.info("产物齐全，补记 stage 行：%s → %s", job_id, ", ".join(filled))
    return filled


def _quarantine_source(job_id: str, source: str | Path, cfg: Any) -> Optional[Path]:
    """把 decode 失败（修不好）的源文件复制到 ``quarantine_dir/<job_id><原后缀>``。

    只复制、**不删** data/ 原文件；失败不影响主流程（记 error 日志）。
    """
    src = Path(source)
    if not src.is_file():
        log.warning("源文件不存在，跳过隔离：%s", src)
        return None
    try:
        quarantine_dir = Path(cfg.path("quarantine_dir"))
    except Exception:  # pragma: no cover - 非标准配置对象兜底
        quarantine_dir = Path(cfg.path("out_dir")) / "quarantine"
    try:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        dst = quarantine_dir / f"{_safe_lock_name(job_id)}{src.suffix}"
        shutil.copy2(src, dst)
    except OSError as exc:  # pragma: no cover - 磁盘/权限异常
        log.error("隔离源文件失败：%s：%s", src, exc)
        return None
    log.warning("decode 失败，源文件已隔离副本：%s", dst)
    return dst


def _cleanup_clean16k(cache_root: Path, job_id: str) -> None:
    """删 ``<id>.clean16k.wav``（保留 16k/analyze/decode 缓存；不调用 decode.cleanup_cache）。"""
    path = Path(cache_root) / f"{job_id}{CLEAN16K_SUFFIX}"
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        log.warning("清理 clean16k 缓存失败：%s", path, exc_info=True)


# ================================================================ 单 job
def _ensure_job(store: Storage, job_id: str, source: str | Path, cfg: Any) -> Any:
    """确保 job 已登记；source 以调用方传入的为准（DB 里旧路径会被刷新）。"""
    job = store.get_job(job_id)
    if job is None:
        job, _created = store.upsert_job(Path(source), job_id=job_id,
                                         config_hash=getattr(cfg, "config_hash", ""))
        log.info("新登记 job：%s ← %s", job_id, source)
        return job
    if str(Path(job.source)) != str(Path(source)):
        log.warning("job %s 的 DB source 与传入不一致，按传入刷新：%s → %s",
                    job_id, job.source, source)
        job = store.update_job(job_id, source=str(Path(source)))
    return job


def _base_result(job_id: str, source: str | Path, job_out: Path) -> Dict[str, Any]:
    """run_job 的固定返回骨架（契约字段一个不少）。"""
    return {
        "ok": False,
        "job_id": str(job_id),
        "source": str(source),
        "out_dir": str(job_out),
        "stage": "",
        "error": "",
        "skipped": False,
        "locked": False,
        "timings": {},
        "report": None,
        "artifacts": [],
        "stages": {},
    }


def _finish_timings(timings: Dict[str, float], started: float) -> Dict[str, float]:
    timings["total_s"] = round(time.perf_counter() - started, 2)
    return dict(timings)


def run_job(
    job_id: str,
    source: str | Path,
    cfg: Any,
    *,
    force: bool = False,
    out_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    on_event: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    keep_intermediates: bool = False,
) -> Dict[str, Any]:
    """单 job 全链路：decode→…→export，失败不抛异常，返回状态字典。

    参数
    ----
    job_id : job id（未登记时会自动 ``upsert_job``；已登记则以传入 ``source`` 为准）
    source : 输入媒体文件路径
    cfg    : ``app.config.Config``
    force  : True 时重跑全部阶段（忽略已 done 的 stages 与产物）；False 时幂等续跑
    out_dir / cache_dir : 覆盖 ``paths.out_dir`` / ``paths.cache_dir``
    on_event : 阶段回调 ``on_event(stage, status, metrics)``（与 render_job 同签名）
    keep_intermediates : True 时保留 ``<id>.clean16k.wav``（render 的中间 wav 也透传保留）

    行为
    ----
    1. 先抢 :class:`JobLock`；抢不到返回 ``{ok:false, locked:true, stage:"lock"}``，不写 DB；
    2. ``force=False`` 且 :func:`is_complete` → ``skipped=True``：不调用任何阶段，只补记
       stages 缺失行（``metrics.backfilled=True``）；
    3. 否则 ``store.begin_attempt`` 后跑 render_job（P1–P3，其 ``export`` 回调只缓存）→
       transcribe_job → diarize_job（总是调用，mode=none 也落 stage）→ correct_job →
       export_text（各自幂等跳过）；每个阶段 done/failed 都落 stages + on_event；
    4. 最后统一落**一条** ``export``（合并 metrics），再 :func:`is_complete` 复核：
       不全 → ``fail_job`` 并返回 ``ok=False``；齐全 → ``finish_job(done)``（只调一次）；
    5. 成功且非 keep_intermediates → 删 ``<id>.clean16k.wav``；decode 阶段失败 →
       源文件复制到 ``quarantine_dir``（不删 data/）。

    返回
    ----
    ``{"ok", "job_id", "source", "out_dir", "stage", "error", "skipped", "locked",
    "timings", "report", "artifacts", "stages"}``；``stages`` 为 ``{stage: status}``，
    ``timings`` 含 ``render_s/asr_s/diarize_s/correct_s/export_text_s/total_s``。
    """
    started = time.perf_counter()
    source_path = Path(source)
    out_root = _out_root(cfg, out_dir)
    cache_root = _cache_root(cfg, cache_dir)
    job_out = out_root / str(job_id)
    result = _base_result(job_id, source, job_out)

    lock = JobLock(job_id, cfg)
    if not lock.acquire(blocking=False):
        result.update({
            "locked": True,
            "stage": "lock",
            "error": f"job 正在运行（锁被占用）：{lock.path}",
        })
        log.warning("job 正在运行，跳过：%s（%s）", job_id, lock.path)
        return result

    store: Optional[Storage] = None
    state: Dict[str, Any] = {
        "audio_export": {},
        "text_export": {},
        "failed": None,
        "failed_error": "",
        "fail_job_done": False,
    }
    timings: Dict[str, float] = {}

    def render_handler(stage: str, status: str, metrics: Dict[str, Any]) -> None:
        """render_job 阶段回调：P1–P3 落库；audio 的 export 只缓存。"""
        data = dict(metrics or {})
        if status == "done":
            if stage == STAGE_EXPORT:
                state["audio_export"].update(data)
                return
            _record_stage_done(store, job_id, stage, data, force=force)
            _emit(on_event, stage, "done", data)
        elif status == "failed":
            _mark_failed(store, job_id, stage, str(data.get("error") or "渲染失败"),
                         state, on_event)

    def module_handler(stage: str, status: str, metrics: Dict[str, Any]) -> None:
        """asr/diarize/correct 阶段回调。"""
        data = dict(metrics or {})
        if status == "done":
            _record_stage_done(store, job_id, stage, data, force=force)
            _emit(on_event, stage, "done", data)
        elif status == "failed":
            _mark_failed(store, job_id, stage, str(data.get("error") or f"{stage} 失败"),
                         state, on_event)

    def text_handler(stage: str, status: str, metrics: Dict[str, Any]) -> None:
        """export_text 回调：text metrics 只缓存，export 行最后统一落。"""
        data = dict(metrics or {})
        if status == "done":
            state["text_export"].update(data)
        elif status == "failed":
            _mark_failed(store, job_id, STAGE_EXPORT,
                         str(data.get("error") or "文本导出失败"), state, on_event)

    def finish_failure(stage: str, error: str) -> Dict[str, Any]:
        """失败收尾：保证 fail_job 已调 + 返回契约字典。"""
        message = str(error or f"{stage} 阶段失败")
        if store is not None and not state.get("fail_job_done"):
            store.fail_job(job_id, message)
            state["fail_job_done"] = True
        result.update({
            "ok": False,
            "stage": stage,
            "error": message,
            "skipped": False,
            "locked": False,
            "timings": _finish_timings(timings, started),
            "report": _read_report(job_out),
            "artifacts": artifacts(job_id, cfg, out_dir=out_root),
            "stages": store.stage_map(job_id) if store is not None else {},
        })
        return result

    try:
        store = Storage(cfg.path("db_path"))
        store.init_db()
        _ensure_job(store, job_id, source_path, cfg)

        # 1) 产物齐全 → 整 job skipped（不调用任何阶段，只补记 stages 缺行）
        if not force and is_complete(job_id, cfg, out_dir=out_root):
            missing, _sections = _missing_products(job_id, cfg, out_dir=out_root)
            _backfill_stage_rows(store, job_id, job_out)
            job = store.get_job(job_id)
            if job is not None and job.status != STATUS_DONE:
                store.finish_job(job_id, STATUS_DONE)
            result.update({
                "ok": True,
                "skipped": True,
                "stage": STAGE_EXPORT,
                "error": "",
                "timings": _finish_timings(timings, started),
                "report": _read_report(job_out),
                "artifacts": artifacts(job_id, cfg, out_dir=out_root),
                "stages": store.stage_map(job_id),
            })
            log.info("产物齐全（%d/8），跳过：%s", len(PRODUCT_FILES) - len(missing), job_id)
            return result

        # 2) 记一次 attempt（attempts+1 / status=running），再校验源文件
        store.begin_attempt(job_id)
        if not source_path.is_file():
            store.fail_stage(job_id, "decode", f"源文件不存在：{source_path}")
            return finish_failure("decode", f"源文件不存在：{source_path}")

        # 3) P1–P3：render_job（decode→analyze→denoise→trim→loudness→audio export）
        render_result = _invoke(
            "decode",
            lambda: render_job(
                job_id, source_path, cfg,
                out_dir=out_root,
                cache_dir=cache_root,
                force=force,
                keep_intermediates=keep_intermediates,
                on_stage=render_handler,
            ),
            store=store, job_id=job_id, state=state, on_event=on_event,
            timings=timings, timings_key="render_s",
        )
        if not render_result.get("ok"):
            failed_stage = str(render_result.get("stage") or "decode")
            if failed_stage == "decode":
                _quarantine_source(job_id, source_path, cfg)
            return finish_failure(failed_stage, str(render_result.get("error") or "渲染失败"))

        # 4) ASR（clean 轴分段转写；已有产物会 skip）
        asr_result = _invoke(
            "asr",
            lambda: transcribe_job(
                job_id, cfg,
                out_dir=out_root,
                cache_dir=cache_root,
                force=force,
                on_stage=module_handler,
            ),
            store=store, job_id=job_id, state=state, on_event=on_event,
            timings=timings, timings_key="asr_s",
        )
        if not asr_result.get("ok"):
            return finish_failure("asr", str(asr_result.get("error") or "转写失败"))

        # 5) 说话人（总是调用；mode=none 也落 stage）
        diarize_result = _invoke(
            "diarize",
            lambda: diarize_job(
                job_id, cfg,
                out_dir=out_root,
                cache_dir=cache_root,
                force=force,
                on_stage=module_handler,
            ),
            store=store, job_id=job_id, state=state, on_event=on_event,
            timings=timings, timings_key="diarize_s",
        )
        if not diarize_result.get("ok"):
            return finish_failure("diarize", str(diarize_result.get("error") or "说话人标注失败"))

        # 6) 纠错（无 key 时 enabled=false 也要落 stage + corrections.json）
        correct_result = _invoke(
            "correct",
            lambda: correct_job(
                job_id, cfg,
                out_dir=out_root,
                force=force,
                on_stage=module_handler,
            ),
            store=store, job_id=job_id, state=state, on_event=on_event,
            timings=timings, timings_key="correct_s",
        )
        if not correct_result.get("ok"):
            return finish_failure("correct", str(correct_result.get("error") or "纠错失败"))

        # 7) 文本导出（force=False 时缺文件也会补）
        export_result = _invoke(
            STAGE_EXPORT,
            lambda: export_text(
                job_id, cfg,
                out_dir=out_root,
                force=force,
                on_stage=text_handler,
            ),
            store=store, job_id=job_id, state=state, on_event=on_event,
            timings=timings, timings_key="export_text_s",
        )
        if not export_result.get("ok"):
            return finish_failure(STAGE_EXPORT, str(export_result.get("error") or "文本导出失败"))

        # 8) 统一落一条 export（audio + text + transcript + corrections + speakers）
        report = _read_report(job_out)
        merged = _build_merged_metrics(
            job_id=job_id, job_out=job_out,
            audio_export=state["audio_export"],
            text_export=state["text_export"],
            report=report,
            asr_metrics=asr_result.get("asr") if isinstance(asr_result.get("asr"), dict) else {},
        )
        store.start_stage(job_id, STAGE_EXPORT, reset=True)
        store.finish_stage(job_id, STAGE_EXPORT, STATUS_DONE, merged)
        _emit(on_event, STAGE_EXPORT, "done", merged)

        # 9) 产物复核：不全 → fail_job（export 行改正为 failed，避免「done job 无产物」）
        missing, missing_sections = _missing_products(job_id, cfg, out_dir=out_root)
        if missing or missing_sections:
            message = (f"产物不全：缺文件 {missing or '无'}；"
                       f"report 缺段 {missing_sections or '无'}（{job_out}）")
            store.fail_stage(job_id, STAGE_EXPORT, message)
            return finish_failure(STAGE_EXPORT, message)

        store.finish_job(job_id, STATUS_DONE)  # 只调一次
        if not keep_intermediates:
            _cleanup_clean16k(cache_root, job_id)

        result.update({
            "ok": True,
            "skipped": False,
            "stage": STAGE_EXPORT,
            "error": "",
            "timings": _finish_timings(timings, started),
            "report": _read_report(job_out),
            "artifacts": artifacts(job_id, cfg, out_dir=out_root),
            "stages": store.stage_map(job_id),
        })
        log.info("编排完成：%s 阶段=%d 产物=%d/8 总耗时 %.1fs",
                 job_id, len(result["stages"]), len(result["artifacts"]),
                 result["timings"].get("total_s", 0.0))
        return result

    except Exception as exc:  # 兜底：编排器自身异常也按失败返回，不再向上抛
        message = str(exc) or exc.__class__.__name__
        log.error("编排器异常：%s：%s", job_id, message, exc_info=True)
        if store is not None:
            try:
                if not state.get("fail_job_done"):
                    store.fail_job(job_id, message)
                    state["fail_job_done"] = True
            except Exception:  # pragma: no cover - DB 也坏了
                log.debug("异常收尾 fail_job 失败", exc_info=True)
        result.update({
            "ok": False,
            "stage": str(state.get("failed") or ""),
            "error": message,
            "timings": _finish_timings(timings, started),
            "report": _read_report(job_out),
            "artifacts": artifacts(job_id, cfg, out_dir=out_root),
            "stages": store.stage_map(job_id) if store is not None else {},
        })
        return result

    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # pragma: no cover
                log.debug("关闭 Storage 失败", exc_info=True)
        lock.release()


# ================================================================ 批量
def _normalize_batch_entry(store: Storage, entry: Any, cfg: Any) -> Tuple[str, Path]:
    """batch 输入元素 → ``(job_id, source)``；``(job_id, source)`` 二元组或路径。

    纯路径会先 ``store.upsert_job`` 登记（返回已有/新建 job 的稳定 id）。
    """
    if isinstance(entry, (str, Path)):
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        job, _created = store.upsert_job(path, config_hash=getattr(cfg, "config_hash", ""))
        return job.id, Path(job.source)
    try:
        job_id, source = entry[0], entry[1]
    except (TypeError, IndexError) as exc:
        raise ValueError(f"batch 输入元素非法（需 (job_id, source) 或路径）：{entry!r}") from exc
    job_id = str(job_id)
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    job = store.get_job(job_id)
    if job is None:
        job, _created = store.upsert_job(source_path, job_id=job_id,
                                         config_hash=getattr(cfg, "config_hash", ""))
    return job_id, Path(job.source)


def batch_run(
    inputs: Iterable[Any],
    cfg: Any,
    *,
    resume: bool = True,
    limit: Optional[int] = None,
    force: bool = False,
    on_event: Optional[Callable[[str, str, str, Dict[str, Any]], None]] = None,
    keep_intermediates: bool = False,
) -> Dict[str, Any]:
    """批量跑多个 job（单进程串行；失败隔离、磁盘水位、attempts 上限、断点续跑）。

    参数
    ----
    inputs : 可迭代元素，``(job_id, source)`` 二元组或 ``Path``/``str`` 路径
             （路径会先 ``upsert_job`` 登记；二元组未登记时也会补登记）
    cfg    : ``app.config.Config``
    resume : True 时产物齐全的 job 直接 skipped（不启动任何阶段）；False 时仍调
             ``run_job``（各阶段模块自身幂等跳过，真正重跑需要 ``force=True``）
    limit  : 最多处理多少个 job（含 skipped；到上限 ``stopped_reason="limit"``）
    force  : True 时重跑全部阶段（忽略 resume/attempts 上限）
    on_event : ``on_event(job_id, stage, status, metrics)``（batch 比 run_job 多一个
               ``job_id`` 参数；回调异常不影响主流程）
    keep_intermediates : 透传 ``run_job``（保留 clean16k 等中间缓存）

    启动时新建独立 ``Storage`` + ``init_db`` + ``mark_interrupted_pending()``
    （把上次崩溃遗留的 running job 改回 pending）；``run_job`` 不调用它。

    返回
    ----
    ``{"ok", "n", "results", "failures", "skipped", "stopped_reason", "timings"}``
    （另附 ``skipped_reasons``：``{job_id: "complete"|"max_attempts"|"locked"}``）。
    - ``results``：真正调用 ``run_job`` 的结果列表（含 ok=False 的尝试，便于 CLI 展示）；
    - ``failures``：``[{"job_id","source","stage","error"}]``；
    - ``skipped``：被跳过的 job id 列表（产物齐全 / attempts 上限 / 抢锁失败）；
    - ``stopped_reason``：``""`` / ``"disk_low"`` / ``"limit"``；
    - ``ok``：没有 failures（磁盘水位/limit 提前收工不算失败）；
    - ``n``：本轮处理的输入数（含 skipped / 失败的尝试；因磁盘水位或 limit 未取的不计）。
    """
    started = time.perf_counter()
    entries = list(inputs or ())
    out_root = _out_root(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = _cache_root(cfg)

    disk_min_gb = float(cfg.get("batch.disk_free_min_gb", 10.0) or 0.0)
    max_attempts = int(cfg.get("batch.max_attempts", 2) or 2)

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    skipped: List[str] = []
    skipped_reasons: Dict[str, str] = {}
    stopped_reason = ""
    processed = 0

    store = Storage(cfg.path("db_path"))
    try:
        store.init_db()
        marked = store.mark_interrupted_pending()
        if marked:
            log.warning("启动时把 %d 个残留 running job 改回 pending", marked)
        log.info("batch 开始：%d 个输入，resume=%s force=%s limit=%s",
                 len(entries), resume, force, limit)

        for entry in entries:
            if limit is not None and processed >= int(limit):
                stopped_reason = "limit"
                log.info("达到 limit=%s，停止取新 job", limit)
                break

            # 磁盘水位：不足则停止取新 job（已开始的 job 已做完）
            if disk_min_gb > 0:
                try:
                    free = shutil.disk_usage(out_root).free
                except OSError:  # pragma: no cover - 特殊文件系统
                    free = None
                if free is not None and free < disk_min_gb * 1e9:
                    stopped_reason = "disk_low"
                    log.warning("磁盘水位不足（剩余 %.2f GB < %.2f GB），停止取新 job",
                                free / 1e9, disk_min_gb)
                    break

            try:
                job_id, source = _normalize_batch_entry(store, entry, cfg)
            except (ValueError, OSError) as exc:
                failures.append({
                    "job_id": "",
                    "source": str(entry),
                    "stage": "input",
                    "error": str(exc) or "batch 输入元素非法",
                })
                processed += 1
                continue

            job = store.get_job(job_id)

            # resume 以产物为准：齐全 → 直接 skipped（不调用任何阶段）
            if resume and not force and is_complete(job_id, cfg, out_dir=out_root):
                _backfill_stage_rows(store, job_id, out_root / job_id)
                if job is not None and job.status != STATUS_DONE:
                    store.finish_job(job_id, STATUS_DONE)
                skipped.append(job_id)
                skipped_reasons[job_id] = "complete"
                processed += 1
                log.info("产物齐全，跳过：%s", job_id)
                continue

            # attempts 上限：失败次数到顶且非 force → 跳过，不无限重试
            if not force and job is not None and job.status == STATUS_FAILED \
                    and int(job.attempts or 0) >= max_attempts:
                skipped.append(job_id)
                skipped_reasons[job_id] = "max_attempts"
                processed += 1
                log.warning("attempts=%s >= max_attempts=%s，跳过：%s",
                            job.attempts, max_attempts, job_id)
                continue

            def _batch_event(stage: str, status: str, metrics: Dict[str, Any],
                             _job_id: str = job_id) -> None:
                if on_event is None:
                    return
                try:
                    on_event(_job_id, str(stage), str(status), dict(metrics or {}))
                except Exception:  # pragma: no cover - 回调失败不影响批处理
                    log.debug("batch on_event 回调失败", exc_info=True)

            result = run_job(
                job_id, source, cfg,
                force=force,
                out_dir=out_root,
                cache_dir=cache_root,
                on_event=_batch_event,
                keep_intermediates=keep_intermediates,
            )
            processed += 1

            if result.get("locked"):
                skipped.append(job_id)
                skipped_reasons[job_id] = "locked"
                log.warning("job 正在运行（锁被占用），batch 跳过：%s", job_id)
                continue
            if result.get("ok"):
                results.append(result)
                if result.get("skipped"):
                    skipped.append(job_id)
                    skipped_reasons.setdefault(job_id, "complete")
            else:
                results.append(result)
                failures.append({
                    "job_id": job_id,
                    "source": str(source),
                    "stage": str(result.get("stage") or ""),
                    "error": str(result.get("error") or ""),
                })
                log.error("batch 中 job 失败（%s）：%s：%s",
                          result.get("stage"), job_id, result.get("error"))
    finally:
        store.close()

    timings = {"total_s": round(time.perf_counter() - started, 2)}
    n = processed
    log.info("batch 结束：处理 %d（成功 %d，失败 %d，跳过 %d），stopped_reason=%s，总耗时 %.1fs",
             n, sum(1 for r in results if r.get("ok")), len(failures), len(skipped),
             stopped_reason or "-", timings["total_s"])
    return {
        "ok": not failures,
        "n": n,
        "results": results,
        "failures": failures,
        "skipped": skipped,
        "skipped_reasons": skipped_reasons,
        "stopped_reason": stopped_reason,
        "timings": timings,
    }
