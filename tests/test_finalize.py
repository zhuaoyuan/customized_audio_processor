# -*- coding: utf-8 -*-
"""P11 交付整理（finalize）+ 新命名契约测试。

覆盖
----
1. **命名契约**（``app.models``）：``safe_stem``（Windows 非法字符/尾部点空格/保留中文《》/长度上限/
   空与保留设备名）、``make_job_id``（``<hash8>_<完整原始文件名主体>``，04下/08上/08下 必须可区分）、
   ``source_basename`` / ``source_stem``（``..._normal..m4a`` 不留尾点）。
2. **交付路径**（``app.pipeline.finalize``）：``delivery_audio_path`` / ``delivery_manuscript_path`` /
   ``resolve_job_audio``（job 目录优先、finalize 后回退 ``output/audio``）。
3. **finalize_run 集成**：临时 out 根 + 2 个假 job 目录（report.json / clean.m4a / manuscript.md /
   manuscript.json / compose_report.json）+ ``app.storage.Storage`` 建的 jobs.db（旧 id + stage 行）：
   目录改名、音频移动、成稿复制、report.id/delivery、DB jobs/stages id 改写、其它 JSON job_id 同步。
4. **幂等 / dry-run / 缺 report.json 告警 / 异常入参**。
5. **CLI** ``finalize --out <root> --json``：返回 0，stdout 是纯 JSON（日志进文件）。
6. **下游兼容**：orchestrator 的 ``is_complete`` / ``_missing_products`` / ``artifacts`` 认可
   finalize 后的 ``output/audio`` 布局。

所有落盘都在 ``tmp_path`` 内，不读写真实 ``out/``。
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

import app.pipeline.orchestrator as orchestrator
from app.cli import EXIT_OK, main
from app.models import make_job_id, safe_stem, source_basename, source_stem
from app.pipeline.finalize import (
    FinalizeError,
    delivery_audio_path,
    delivery_manuscript_path,
    finalize_run,
    resolve_job_audio,
)
from app.storage import Storage


# ================================================================ 测试数据
PHYSICS = "【复旦哲学系丁耘】 亚里士多德《物理学》"

#: 新版 id 必须能区分这三个同名课程的批次（旧版 20 字 slug 会全部撞成同一个 id）
SOURCE_04 = f"data/{PHYSICS}(04下-20100325)_5_normal..m4a"
SOURCE_08_UP = f"data/{PHYSICS}(08上-20101111)_5_normal..m4a"
SOURCE_08_DOWN = f"data/{PHYSICS}(08下-20101112)_5_normal..m4a"

#: finalize 集成用的两个源文件（一个带《》与中文，一个带空格）
SOURCE_A = SOURCE_04
SOURCE_B = "data/0_20140227-丁耘 西方哲学(20140227-丁耘 西方哲学)_normal..m4a"

#: 旧版 slug 风格 job id（finalize 前 jobs.db / 目录名都是它）
OLD_ID_A = "0_20140227_丁耘_西方哲学"
OLD_ID_B = "24_复旦哲学系丁耘_亚里士多德_形而上学"

FAKE_M4A = b"FAKE-M4A-CONTENT" * 32
MANUSCRIPT_TEXT = "# 成稿\n\n[00:00] 主讲：第一段正文。\n"


# ================================================================ 小工具
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _hash8(job_id: str) -> str:
    return job_id.split("_", 1)[0]


def _is_hex(text: str) -> bool:
    return bool(text) and all(ch in string.hexdigits for ch in text)


# ================================================================ 命名契约
def test_safe_stem_replaces_windows_illegal_chars():
    """``<>:"|?*`` 与控制字符 → ``_``；分隔符只取 basename。"""
    assert safe_stem('a<b>c:d"e|f?g*h') == "a_b_c_d_e_f_g_h"
    assert safe_stem("tab\there") == "tab_here"
    assert safe_stem("line\nbreak") == "line_break"
    # 斜杠/反斜杠只当路径分隔符（只取最后一段），不会留在目录名里
    assert safe_stem("dir/sub/file:1") == "file_1"
    assert safe_stem(r"dir\sub\file:1") == "file_1"


def test_safe_stem_strips_trailing_dots_and_spaces():
    """Windows 会静默丢掉尾部点/空格：必须提前剥掉，保证 id 与目录名一致。"""
    assert safe_stem("课程录音...") == "课程录音"
    assert safe_stem("课程录音 . . ") == "课程录音"
    assert safe_stem("  前后空白 课程  ") == "前后空白 课程"
    assert safe_stem("...hidden") == "hidden"  # 前导点 → 避免隐藏目录
    assert safe_stem("normal..").endswith("normal")


def test_safe_stem_keeps_cjk_and_legal_symbols():
    """中文、空格、《》() 与 04下/08上 等区分后缀原样保留（这是新契约的关键）。"""
    name = "【复旦哲学系丁耘】 亚里士多德《物理学》(04下-20100325)_5_normal"
    assert safe_stem(name) == name
    assert safe_stem("中文 空格 (括号) 《书名》") == "中文 空格 (括号) 《书名》"


def test_safe_stem_length_cap():
    """默认 120 字上限；极端长文件名才硬截断，且截断后不留尾部点/空格。"""
    long_name = "字" * 500
    assert len(safe_stem(long_name)) == 120
    assert safe_stem(long_name) == "字" * 120
    # 自定义上限
    assert len(safe_stem("x" * 50, 10)) == 10
    # 截断位置刚好是点：不能残留尾点
    truncated = safe_stem("a" * 119 + "." * 10)
    assert truncated == "a" * 119
    assert not truncated.endswith((".", " "))


def test_safe_stem_empty_and_reserved_names():
    assert safe_stem("") == "audio"
    assert safe_stem("   ") == "audio"
    assert safe_stem(" ... ") == "audio"
    # Windows 保留设备名（大小写无关）加前缀，避免无法建目录
    for reserved in ("CON", "con", "NUL", "Com1", "lpt9"):
        assert safe_stem(reserved) == f"_{reserved}"


def test_source_basename_and_stem_for_normal_double_dot():
    """``..._normal..m4a``：basename 保留原样，stem 去掉尾点（finalize 用它生成成稿名）。"""
    source = f"data/{PHYSICS}(05-20100401)_5_normal..m4a"
    assert source_basename(source) == f"{PHYSICS}(05-20100401)_5_normal..m4a"
    stem = source_stem(source)
    assert stem == f"{PHYSICS}(05-20100401)_5_normal"
    assert not stem.endswith((".", " "))
    # 反斜杠路径（Windows 写法）也要正确
    assert source_stem(r"C:\data\课程 01..m4a") == "课程 01"


def test_make_job_id_distinguishes_physics_batches():
    """04下 / 08上 / 08下 三个源文件必须得到互不相同的 id（旧版截断后会混淆）。"""
    sources = (SOURCE_04, SOURCE_08_UP, SOURCE_08_DOWN)
    ids = [make_job_id(source) for source in sources]
    assert len(set(ids)) == 3, ids

    expected_suffixes = ("(04下-20100325)", "(08上-20101111)", "(08下-20101112)")
    for job_id, suffix in zip(ids, expected_suffixes):
        assert "物理学" in job_id
        assert suffix in job_id, f"{job_id} 丢了区分后缀 {suffix}"
        assert len(job_id) > 20, "新契约不再截断到 20 字"
        assert len(_hash8(job_id)) == 8 and _is_hex(_hash8(job_id))
    # 每个 id 都以 8 位 hash 开头（同名不同目录可区分、移动目录后稳定）
    for source, job_id in zip(sources, ids):
        assert job_id.startswith(_hash8(job_id) + "_")


def test_make_job_id_keeps_spaces_and_cjk_and_handles_taken():
    name = "24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a"
    job_id = make_job_id(name)
    assert job_id == make_job_id(Path("data") / name)  # 稳定：只依赖 basename
    assert " 亚里士多德" in job_id  # 空格保留
    assert "《形而上学》" in job_id  # 《》保留
    assert "(01)" in job_id
    assert job_id.count("_") >= 2  # hash8_ 与原始文件名里的 _
    assert not any(ch in job_id for ch in '<>:"/\\|?*')

    # taken 里是**同一个**源文件 → 不追加后缀
    assert make_job_id(name, taken={job_id: name}) == job_id
    # taken 里是另一个源文件 → 追加 4 位路径 hash 区分
    other = make_job_id(
        Path(r"C:\elsewhere") / name,
        taken={job_id: "data/another.m4a"},
    )
    assert other != job_id
    assert other.startswith(job_id + "_")
    assert len(other.rsplit("_", 1)[1]) == 4


# ================================================================ 交付路径
def test_delivery_paths_and_resolve_job_audio(tmp_path: Path):
    out_root = tmp_path / "out"
    source = str(tmp_path / "in" / "课程 01..m4a")

    assert delivery_audio_path(out_root, source) == (
        out_root / "output" / "audio" / "课程 01..m4a"
    )
    assert delivery_manuscript_path(out_root, source) == (
        out_root / "output" / "manuscript" / "课程 01.md"
    )

    job_dir = out_root / "60814669_课程 01"
    job_dir.mkdir(parents=True)
    job_audio = job_dir / "clean.m4a"
    job_audio.write_bytes(b"job-dir-audio")
    delivery = delivery_audio_path(out_root, source)
    delivery.parent.mkdir(parents=True, exist_ok=True)
    delivery.write_bytes(b"delivery-audio")

    # job 目录优先
    assert resolve_job_audio(job_dir, out_root, source) == job_audio
    # job 目录没有 → 回退 finalize 交付目录
    job_audio.unlink()
    assert resolve_job_audio(job_dir, out_root, source) == delivery

    # report.delivery.audio 指向的实际位置优先于 source 推导
    custom = out_root / "output" / "audio" / "自定义 名称.m4a"
    custom.write_bytes(b"custom")
    _write_json(job_dir / "report.json", {
        "id": job_dir.name,
        "source": source,
        "delivery": {"dir": job_dir.name, "audio": "output/audio/自定义 名称.m4a"},
    })
    assert resolve_job_audio(job_dir, out_root, source) == custom

    # 两边都没有 → None
    custom.unlink()
    delivery.unlink()
    assert resolve_job_audio(job_dir, out_root, source) is None


# ================================================================ finalize 夹具
def _make_job_dir(
    root: Path,
    old_id: str,
    source: str,
    *,
    with_audio: bool = True,
    with_manuscript: bool = True,
) -> Path:
    """造一个 pre-finalize 的 job 目录（含 5 个 finalize 关心的文件）。"""
    job_dir = root / old_id
    job_dir.mkdir(parents=True)
    _write_json(job_dir / "report.json", {
        "id": old_id,
        "source": source,
        "ok": True,
        "levels": {"asl_in_dbfs": -23.0},
    })
    if with_audio:
        (job_dir / "clean.m4a").write_bytes(FAKE_M4A)
    if with_manuscript:
        (job_dir / "manuscript.md").write_text(MANUSCRIPT_TEXT, encoding="utf-8")
    _write_json(job_dir / "manuscript.json", {
        "version": "test",
        "job_id": old_id,
        "source": str(job_dir / "transcript.json"),
        "paragraphs": [],
    })
    _write_json(job_dir / "compose_report.json", {
        "version": "test",
        "job_id": old_id,
        "ok": True,
        "stats": {"chars": 12},
    })
    return job_dir


def _make_db(root: Path) -> Path:
    """用真实 Storage 建 jobs.db：两条旧 id + 各自 stage 行。"""
    db_path = root / "jobs.db"
    store = Storage(db_path, default_config_hash="hash-test")
    store.init_db()
    try:
        store.upsert_job(SOURCE_A, job_id=OLD_ID_A, config_hash="hash-test")
        store.upsert_job(SOURCE_B, job_id=OLD_ID_B, config_hash="hash-test")
        store.start_stage(OLD_ID_A, "decode")
        store.finish_stage(OLD_ID_A, "decode", "done", {"duration_s": 12.5})
        store.start_stage(OLD_ID_A, "asr")
        store.finish_stage(OLD_ID_A, "asr", "done", {"n_segments": 2})
        store.start_stage(OLD_ID_B, "decode")
        store.finish_stage(OLD_ID_B, "decode", "done", {"duration_s": 3.0})
    finally:
        store.close()
    return db_path


def _build_out_root(root: Path, *, broken_dir: bool = False) -> Dict[str, Any]:
    """造完整 pre-finalize 布局，返回测试需要的路径与期望 id。"""
    root.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {
        "root": root,
        "job_a": _make_job_dir(root, OLD_ID_A, SOURCE_A),
        "job_b": _make_job_dir(root, OLD_ID_B, SOURCE_B),
        "db": _make_db(root),
        "broken": None,
        "new_a": make_job_id(SOURCE_A),
        "new_b": make_job_id(SOURCE_B),
    }
    if broken_dir:
        broken = root / "orphan_old_job"
        broken.mkdir()
        (broken / "clean.m4a").write_bytes(FAKE_M4A)  # 有产物、没 report.json
        info["broken"] = broken
    return info


def _snapshot(root: Path, *, include_db: bool = False) -> Dict[str, Tuple[int, int]]:
    """文件树快照：相对路径 → (size, mtime_ns)；默认跳过 sqlite 文件（连接会碰 WAL）。"""
    result: Dict[str, Tuple[int, int]] = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        if not include_db and (path.name == "jobs.db" or path.name.startswith("jobs.db-")):
            continue
        stat = path.stat()
        result[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return result


def _job_ids(db_path: Path) -> set:
    store = Storage(db_path)
    try:
        store.init_db()
        return {job.id for job in store.list_jobs()}
    finally:
        store.close()


@pytest.fixture()
def finalized(tmp_path: Path):
    """跑过一次 finalize 的 out 根（幂等用例复用）。"""
    root = tmp_path / "out"
    info = _build_out_root(root)
    summary = finalize_run(root, db_path=info["db"])
    assert summary["ok"] is True, summary
    return info, summary


# ================================================================ finalize 集成
def test_finalize_run_renames_dirs_and_archives_products(finalized):
    info, summary = finalized
    root = info["root"]
    new_a, new_b = info["new_a"], info["new_b"]

    # ---- 汇总计数
    assert summary["dry_run"] is False
    assert summary["n_jobs"] == 2
    assert (summary["renamed"], summary["audio_moved"], summary["manuscript_copied"]) == (2, 2, 2)
    assert summary["errors"] == []
    by_old = {item["old_id"]: item for item in summary["jobs"]}
    assert by_old[OLD_ID_A]["new_id"] == new_a
    assert by_old[OLD_ID_A]["dir_renamed"] is True
    assert by_old[OLD_ID_A]["audio"] == "moved"
    assert by_old[OLD_ID_A]["manuscript"] == "copied"
    assert by_old[OLD_ID_A]["error"] == ""

    # ---- 目录改名：旧目录消失，新目录名 = hash8 + 完整原始文件名主体
    assert not info["job_a"].exists() and not info["job_b"].exists()
    assert (root / new_a).is_dir() and (root / new_b).is_dir()
    assert "《物理学》" in new_a and "(04下-20100325)" in new_a
    assert len(new_a) > 20
    assert {path.name for path in root.iterdir() if path.is_dir()} == {
        "output", new_a, new_b,
    }

    # ---- 音频归档（移动、不留冗余副本）
    audio_a = root / "output" / "audio" / source_basename(SOURCE_A)
    assert audio_a.is_file()
    assert audio_a.read_bytes() == FAKE_M4A
    assert not (root / new_a / "clean.m4a").exists()
    assert (root / "output" / "audio" / source_basename(SOURCE_B)).is_file()

    # ---- 成稿归档（复制；job 目录保留原稿）
    manuscript_a = root / "output" / "manuscript" / f"{source_stem(SOURCE_A)}.md"
    assert manuscript_a.is_file()
    assert manuscript_a.read_text(encoding="utf-8") == MANUSCRIPT_TEXT
    assert (root / new_a / "manuscript.md").read_text(encoding="utf-8") == MANUSCRIPT_TEXT
    assert manuscript_a.name.endswith("_normal.md")
    assert not manuscript_a.name.endswith("..md")

    # ---- report.json：id + delivery（相对 out_root 的 posix 路径）
    report = _read_json(root / new_a / "report.json")
    assert report["id"] == new_a
    assert report["delivery"] == {
        "dir": new_a,
        "audio": (
            Path("output") / "audio" / source_basename(SOURCE_A)
        ).as_posix(),
        "manuscript": (
            Path("output") / "manuscript" / f"{source_stem(SOURCE_A)}.md"
        ).as_posix(),
    }
    # 原有业务字段不丢
    assert report["source"] == SOURCE_A
    assert report["levels"] == {"asl_in_dbfs": -23.0}

    # ---- manuscript.json / compose_report.json 的 job_id（与 source 里的旧目录）同步改写
    manuscript_json = _read_json(root / new_a / "manuscript.json")
    assert manuscript_json["job_id"] == new_a
    assert manuscript_json["source"] == str(root / new_a / "transcript.json")
    assert _read_json(root / new_a / "compose_report.json")["job_id"] == new_a

    # ---- DB：jobs/stages 的 id 全部改写（旧 id 不再存在）
    store = Storage(info["db"])
    try:
        store.init_db()
        assert store.get_job(OLD_ID_A) is None
        assert store.get_job(OLD_ID_B) is None
        job = store.get_job(new_a)
        # Storage.upsert_job 会把 source 规整成平台路径（Windows 是反斜杠）
        assert job is not None and job.source == str(Path(SOURCE_A))
        assert store.get_job(new_b) is not None
        stages = store.list_stages(new_a)
        assert [stage.stage for stage in stages] == ["decode", "asr"]
        assert all(stage.job_id == new_a for stage in stages)
        assert [stage.job_id for stage in store.list_stages(new_b)] == [new_b]
    finally:
        store.close()
    assert _job_ids(info["db"]) == {new_a, new_b}


def test_finalize_run_is_idempotent(finalized):
    """第二次执行：renamed/audio_moved/manuscript_copied 全为 0，交付文件 mtime/size 不变。"""
    info, _first = finalized
    root = info["root"]
    before = _snapshot(root)

    second = finalize_run(root, db_path=info["db"])

    after = _snapshot(root)
    assert second["ok"] is True
    assert second["n_jobs"] == 2
    assert second["renamed"] == 0
    assert second["audio_moved"] == 0
    assert second["manuscript_copied"] == 0
    assert second["audio_present"] == 2
    assert second["errors"] == []
    assert [item["audio"] for item in second["jobs"]] == ["present", "present"]
    assert [item["manuscript"] for item in second["jobs"]] == ["present", "present"]
    assert before == after, "幂等 second run 不应改动任何交付文件（size/mtime 必须一致）"
    # 目录名不变
    assert {path.name for path in root.iterdir() if path.is_dir()} == {
        "output", info["new_a"], info["new_b"],
    }


def test_finalize_run_dry_run_changes_nothing(tmp_path: Path):
    root = tmp_path / "out"
    info = _build_out_root(root)
    before = _snapshot(root)
    db_before = info["db"].read_bytes()

    summary = finalize_run(root, db_path=info["db"], dry_run=True)

    assert summary["ok"] is True and summary["dry_run"] is True
    assert summary["n_jobs"] == 2
    assert summary["renamed"] == 2
    assert summary["audio_moved"] == 2
    assert summary["manuscript_copied"] == 2
    assert all(item["dir_renamed"] is True for item in summary["jobs"])
    assert {item["audio"] for item in summary["jobs"]} == {"would-move"}
    assert {item["manuscript"] for item in summary["jobs"]} == {"would-copy"}

    # 盘上零改动：目录、文件、DB 都没变
    assert not (root / "output").exists()
    assert info["job_a"].is_dir() and info["job_b"].is_dir()
    assert _snapshot(root) == before
    assert info["db"].read_bytes() == db_before
    assert _job_ids(info["db"]) == {OLD_ID_A, OLD_ID_B}


def test_finalize_run_warns_on_dir_without_report(tmp_path: Path):
    root = tmp_path / "out"
    info = _build_out_root(root, broken_dir=True)

    summary = finalize_run(root, db_path=info["db"])

    assert summary["ok"] is True
    assert summary["n_jobs"] == 2  # 只处理有 report.json 的
    assert any(
        "缺少 report.json" in warning and info["broken"].name in warning
        for warning in summary["warnings"]
    ), summary["warnings"]
    # 可疑目录原样保留（不猜、不删）
    assert (info["broken"] / "clean.m4a").is_file()


def test_finalize_run_edge_cases(tmp_path: Path):
    # 不存在的产物根 → 明确报错
    with pytest.raises(FinalizeError, match="不存在"):
        finalize_run(tmp_path / "nope")

    # 空目录 → 0 个 job、不报错
    empty = tmp_path / "empty"
    empty.mkdir()
    summary = finalize_run(empty)
    assert summary["ok"] is True
    assert summary["n_jobs"] == 0
    assert summary["renamed"] == 0
    assert summary["jobs"] == []
    # 没有 manifest 的目录不会凭空生成 output/
    assert not (empty / "output").exists()


# ================================================================ CLI
def test_finalize_cli_json(tmp_path: Path, capsys):
    """``finalize --out <root> --json``：返回 0，stdout 纯 JSON（日志只进文件）。"""
    from app.logging_setup import shutdown_logging

    root = tmp_path / "out"
    info = _build_out_root(root)

    try:
        code = main(["finalize", "--out", str(root), "--json"])
    finally:
        # main() 会把日志重配到 --out 下（file-only），用完还原，避免污染后续用例
        shutdown_logging()

    captured = capsys.readouterr()
    assert code == EXIT_OK, captured.err
    payload = json.loads(captured.out)  # 若有日志混进 stdout 这里直接失败
    assert payload["ok"] is True
    assert payload["dry_run"] is False
    assert payload["n_jobs"] == 2
    assert payload["renamed"] == 2
    assert payload["audio_moved"] == 2
    assert payload["manuscript_copied"] == 2
    assert payload["errors"] == []
    # 确实按新布局落盘（不是只打印）
    assert (root / info["new_a"]).is_dir()
    assert (root / "output" / "audio" / source_basename(SOURCE_A)).is_file()
    assert _job_ids(info["db"]) == {info["new_a"], info["new_b"]}
    # --json 模式日志写文件（stdout 才可能保持纯 JSON）
    logs_dir = root / "logs"
    assert logs_dir.is_dir() and any(logs_dir.glob("*.log"))


# ================================================================ 下游兼容（orchestrator）
def test_orchestrator_accepts_finalized_audio_layout(cfg, tmp_path: Path):
    """finalize 后 job 目录没有 clean.m4a：完成判定/产物列表必须仍认可 output/audio 布局。"""
    out_root = cfg.path("out_dir")
    out_root.mkdir(parents=True, exist_ok=True)
    source = str(
        tmp_path / "inputs" / "24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a"
    )
    old_id = "24_复旦哲学系丁耘_亚里士多德_形而上学"
    job_dir = out_root / old_id
    job_dir.mkdir()
    for name in orchestrator.PRODUCT_FILES:
        (job_dir / name).write_bytes(b"x" * 8)
    _write_json(job_dir / "report.json", {
        "id": old_id,
        "source": source,
        "asr": {"n_segments": 1, "chars": 4, "elapsed_s": 0.1},
        "correct": {"enabled": False, "edits": 0, "rejected": 0},
        "speakers": {"n": 1, "method": "none"},
    })
    assert orchestrator.is_complete(old_id, cfg) is True

    summary = finalize_run(out_root)
    new_id = make_job_id(source)
    assert summary["ok"] is True
    assert summary["renamed"] == 1 and summary["audio_moved"] == 1
    assert not job_dir.exists()
    assert not (out_root / new_id / "clean.m4a").exists()

    missing, missing_sections = orchestrator._missing_products(new_id, cfg)
    assert (missing, missing_sections) == ([], [])
    assert orchestrator.is_complete(new_id, cfg) is True
    # 产物列表仍列出 8 件（clean.m4a 从 delivery 解析）
    assert [item["name"] for item in orchestrator.artifacts(new_id, cfg)] == list(
        orchestrator.PRODUCT_FILES
    )
    # 音频解析到 output/audio/<源原始文件名>
    assert resolve_job_audio(out_root / new_id, out_root, source) == delivery_audio_path(
        out_root, source
    )
