# -*- coding: utf-8 -*-
"""CLI 冒烟测试：config show / scan / status / decode 的参数解析与入库行为。

用临时目录与假文件，不依赖 data/；真实六文件走 ``python -m app.cli decode --all``（见工程日志）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cli import EXIT_OK, EXIT_USAGE, discover_files, main


def _fake_media(root: Path, names: list[str], size: int = 70000) -> list[Path]:
    """造一批「足够大」的假媒体文件（scan 有 min_bytes 过滤）。"""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * size)
        paths.append(path)
    return paths


def test_config_show_json(capsys, tmp_path: Path):
    code = main(["config", "show", "--json",
                 "--out", str(tmp_path / "out")])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["levels"]["target_asl_dbfs"] == -20.0
    assert payload["trim"]["boundary_window_s"] == 120.0
    assert payload["paths"]["cache_dir"].endswith("cache")


def test_config_show_yaml_default(capsys, tmp_path: Path):
    code = main(["config", "show", "--preset", "fast", "--out", str(tmp_path / "out")])
    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert "name: fast" in captured.out
    assert "target_asl_dbfs: -20.0" in captured.out
    # 默认值必须出现在输出里（验收点：打印合并后配置）
    assert "boundary_min_chars_per_sec: 2.0" in captured.out
    # fast preset 覆盖项生效
    assert "enabled: false" in captured.out


def test_config_check_json(capsys):
    code = main(["config", "check", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code in (EXIT_OK, 1)
    assert "checks" in payload and "missing" in payload
    assert any(c["name"] == "data_dir" for c in payload["checks"])


def test_config_unknown_preset(capsys):
    with pytest.raises(SystemExit):
        main(["config", "show", "--preset", "turbo"])


def test_discover_files_filters(tmp_path: Path):
    _fake_media(tmp_path / "in", ["a.m4a", "sub/b.mp3", "c.txt", "d.wav"])
    (tmp_path / "in" / "small.m4a").write_bytes(b"\x00" * 10)
    (tmp_path / "in" / ".hidden").mkdir()
    (tmp_path / "in" / ".hidden" / "e.m4a").write_bytes(b"\x00" * 70000)
    found = discover_files(tmp_path / "in", min_bytes=1024)
    names = [p.name for p in found]
    assert "a.m4a" in names and "b.mp3" in names and "d.wav" in names
    assert "c.txt" not in names
    assert "small.m4a" not in names
    assert "e.m4a" not in names  # 隐藏目录跳过
    # 非递归
    flat = discover_files(tmp_path / "in", recursive=False, min_bytes=1024)
    assert "b.mp3" not in [p.name for p in flat]
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "nope")


def test_scan_then_status(capsys, tmp_path: Path):
    data = tmp_path / "data"
    _fake_media(data, [
        "0_20140227-丁耘 西方哲学(20140227)_normal..m4a",
        "24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a",
        "4_丁耘理想国.古希腊哲学(第四卷)_normal..m4a",
    ])
    out = tmp_path / "out"
    assert main(["scan", str(data), "--out", str(out)]) == EXIT_OK
    scan_out = capsys.readouterr().out
    assert "新增 3" in scan_out
    assert (out / "jobs.db").is_file()

    # 再扫一次：不重复新增、不重置状态
    assert main(["scan", str(data), "--out", str(out)]) == EXIT_OK
    assert "新增 0" in capsys.readouterr().out

    assert main(["status", "--out", str(out), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["counts"]["pending"] == 3
    assert len(payload["jobs"]) == 3
    assert all(job["status"] == "pending" for job in payload["jobs"])
    ids = [job["id"] for job in payload["jobs"]]
    assert len(set(ids)) == 3
    assert all(len(i.split("_")[0]) == 8 for i in ids)

    # 人读表格
    assert main(["status", "--out", str(out)]) == EXIT_OK
    table = capsys.readouterr().out
    assert "pending" in table and "状态汇总" in table


def test_scan_dry_run_and_empty(capsys, tmp_path: Path):
    data = tmp_path / "data"
    _fake_media(data, ["a.m4a"])
    out = tmp_path / "out"
    assert main(["scan", str(data), "--out", str(out), "--dry-run"]) == EXIT_OK
    assert "a.m4a" in capsys.readouterr().out
    assert not (out / "jobs.db").exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["scan", str(empty), "--out", str(out)]) == 1
    assert "没有发现" in capsys.readouterr().err


def test_status_without_db(capsys, tmp_path: Path):
    code = main(["status", "--out", str(tmp_path / "nope")])
    assert code == EXIT_USAGE
    assert "数据库不存在" in capsys.readouterr().err


def test_scan_limit(capsys, tmp_path: Path):
    data = tmp_path / "data"
    _fake_media(data, [f"f{i}.m4a" for i in range(5)])
    out = tmp_path / "out"
    assert main(["scan", str(data), "--out", str(out), "--limit", "2"]) == EXIT_OK
    assert "新增 2" in capsys.readouterr().out


def test_help_available():
    for argv in (["decode", "--help"], ["scan", "--help"], ["status", "--help"],
                 ["config", "--help"], ["run", "--help"], ["batch", "--help"],
                 ["retry", "--help"], ["serve", "--help"]):
        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == 0


# ---------------------------------------------------------------- P6-T2：run/batch/retry
def test_batch_json_and_skipped_rows(capsys, tmp_path: Path, monkeypatch):
    """batch：目标解析成 (job_id, source) → batch_run；--resume 透传；SKIP 行与 JSON 输出。"""
    import app.pipeline.orchestrator as orch

    data = tmp_path / "data"
    _fake_media(data, ["a.m4a", "b.m4a"])
    out = tmp_path / "out"
    seen: dict = {}

    def fake_batch(inputs, cfg, *, resume=True, limit=None, force=False, on_event=None,
                   keep_intermediates=False):
        targets = list(inputs)
        seen.update({"n": len(targets), "resume": resume, "force": force,
                     "ids": [jid for jid, _ in targets]})
        return {"ok": True, "n": len(targets), "results": [],
                "failures": [], "skipped": [jid for jid, _ in targets],
                "skipped_reasons": {jid: "complete" for jid, _ in targets},
                "stopped_reason": "", "timings": {"total_s": 0.01}}

    monkeypatch.setattr(orch, "batch_run", fake_batch)
    assert main(["batch", str(data), "--out", str(out), "--resume"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "跳过 2" in text and text.count("[SKIP]") == 2
    assert seen["n"] == 2 and seen["resume"] is True and seen["force"] is False

    assert main(["batch", str(data), "--out", str(out), "--resume", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["skipped"]) == 2 and payload["ok"] is True


def test_batch_failure_exit_code(capsys, tmp_path: Path, monkeypatch):
    """batch 有失败 → 退出码 1 且打印失败行。"""
    import app.pipeline.orchestrator as orch

    data = tmp_path / "data"
    _fake_media(data, ["a.m4a"])
    out = tmp_path / "out"

    def fake_batch(inputs, cfg, **kwargs):
        job_id, source = list(inputs)[0]
        return {
            "ok": False, "n": 1,
            "results": [{"ok": False, "job_id": job_id, "source": source,
                         "stage": "asr", "error": "boom", "locked": False,
                         "timings": {}, "report": {}, "artifacts": []}],
            "failures": [{"job_id": job_id, "source": source, "stage": "asr", "error": "boom"}],
            "skipped": [], "stopped_reason": "", "timings": {},
        }

    monkeypatch.setattr(orch, "batch_run", fake_batch)
    assert main(["batch", str(data), "--out", str(out)]) == 1
    assert "失败 1" in capsys.readouterr().out


def test_run_command_registers_unknown_path(capsys, tmp_path: Path, monkeypatch):
    """run <路径>：未登记路径先 upsert 入库，再调 run_job。"""
    import app.pipeline.orchestrator as orch
    from app.storage import Storage

    data = tmp_path / "data"
    _fake_media(data, ["x.m4a"])
    out = tmp_path / "out"
    captured: dict = {}

    def fake_run(job_id, source, cfg, *, force=False, out_dir=None, cache_dir=None,
                 on_event=None, keep_intermediates=False):
        captured.update({"job_id": job_id, "source": str(source), "force": force})
        return {"ok": True, "job_id": job_id, "source": str(source), "stage": "export",
                "error": "", "skipped": False, "locked": False,
                "timings": {"total_s": 1.0}, "report": {}, "artifacts": [], "stages": {}}

    monkeypatch.setattr(orch, "run_job", fake_run)
    assert main(["run", str(data / "x.m4a"), "--out", str(out), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["result"]["skipped"] is False
    with Storage(out / "jobs.db") as store:
        job = store.get_job(captured["job_id"])
    assert job is not None and Path(job.source) == data / "x.m4a"


def test_run_command_locked_exit_code(capsys, tmp_path: Path, monkeypatch):
    """run 被锁占用（locked=True）→ 退出码 1，打印 LOCK 行。"""
    import app.pipeline.orchestrator as orch

    data = tmp_path / "data"
    _fake_media(data, ["z.m4a"])
    out = tmp_path / "out"

    def fake_run(job_id, source, cfg, **kwargs):
        return {"ok": False, "job_id": job_id, "source": str(source), "stage": "lock",
                "error": "job 正在运行", "skipped": False, "locked": True,
                "timings": {}, "report": None, "artifacts": [], "stages": {}}

    monkeypatch.setattr(orch, "run_job", fake_run)
    assert main(["run", str(data / "z.m4a"), "--out", str(out)]) == 1
    assert "[LOCK]" in capsys.readouterr().out


def test_retry_resets_failed_stage(capsys, tmp_path: Path, monkeypatch):
    """retry：从第一个未完成阶段（asr）重置下游；run_job 收到 force=False。"""
    import app.pipeline.orchestrator as orch

    data = tmp_path / "data"
    _fake_media(data, ["y.m4a"])
    out = tmp_path / "out"
    from app.models import Job
    from app.storage import Storage

    with Storage(out / "jobs.db") as store:
        store.init_db()
        store.add_job(Job.new("job_retry", str(data / "y.m4a")))
        for name in ("decode", "analyze", "denoise", "trim", "loudness"):
            store.finish_stage("job_retry", name, "done", {"duration_s": 0.1})
        store.fail_stage("job_retry", "asr", "boom")
        store.fail_job("job_retry", "boom")

    seen: dict = {}

    def fake_run(job_id, source, cfg, *, force=False, out_dir=None, cache_dir=None,
                 on_event=None, keep_intermediates=False):
        with Storage(out / "jobs.db") as store:
            seen["stages"] = store.stage_map(job_id)
        seen["force"] = force
        return {"ok": True, "job_id": job_id, "source": str(source), "stage": "export",
                "error": "", "skipped": False, "locked": False, "timings": {},
                "report": {}, "artifacts": [], "stages": {}}

    monkeypatch.setattr(orch, "run_job", fake_run)
    assert main(["retry", "job_retry", "--out", str(out)]) == EXIT_OK
    assert "从 asr 阶段重跑" in capsys.readouterr().out
    assert seen["stages"]["decode"] == "done"      # 上游已完成阶段不动
    assert seen["stages"]["asr"] == "pending"      # 失败阶段及下游被重置
    assert seen["force"] is False


def test_decode_command_stage_and_cleanup(capsys, tmp_path: Path, monkeypatch):
    """decode 子命令：阶段结果落库 + --cleanup 只删中间文件、保留 16k 与缓存元数据。

    用 monkeypatch 顶掉真实解码，避免用例依赖 ffmpeg（真实解码由 test_decode.py 覆盖）。
    """
    from app.models import DecodeResult
    from app.storage import Storage

    data = tmp_path / "data"
    _fake_media(data, ["第一讲.m4a"])
    out = tmp_path / "out"
    assert main(["scan", str(data), "--out", str(out)]) == EXIT_OK
    capsys.readouterr()

    with Storage(out / "jobs.db") as store:
        job_id = store.list_jobs()[0].id

    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    wav16k = cache / f"{job_id}.16k.wav"
    wav44k = cache / f"{job_id}.44k.wav"
    meta = cache / f"{job_id}.decode.json"
    for path in (wav16k, wav44k, meta):
        path.write_bytes(b"\x00" * 16)

    fake = DecodeResult(
        ok=True, job_id=job_id, wav16k=str(wav16k), duration_s=12.5,
        sample_rate=16000, orig_sample_rate=44100, channels=1, bit_rate=96000,
        repaired=False, decode_source=str(data / "第一讲.m4a"),
    )
    monkeypatch.setattr("app.cli.run_decode", lambda *a, **k: fake)

    assert main([
        "decode", job_id, "--out", str(out), "--cache-dir", str(cache), "--cleanup",
    ]) == EXIT_OK
    output = capsys.readouterr().out
    assert "成功 1 / 1" in output

    # 44k 被清掉；16k 与元数据保留（否则缓存校验失效）
    assert wav16k.is_file()
    assert meta.is_file()
    assert not wav44k.is_file()

    with Storage(out / "jobs.db") as store:
        job = store.get_job(job_id)
        stage = store.get_stage(job_id, "decode")
    assert stage is not None and stage.status == "done"
    assert stage.metrics["duration_s"] == pytest.approx(12.5)
    assert job.stage == "decode"
    assert job.status == "pending"  # 只跑了解码，等后续阶段


# ================================================================ P8 regress 子命令
def _fake_regress_result(mode: str = "reuse") -> dict:
    return {
        "ok": True,
        "mode": mode,
        "n": 1,
        "rows": [{"job_id": "j1", "clean_s": 100.0, "asl_out_dbfs": -20.0,
                  "asl_dev_db": 0.0, "true_peak_out_dbtp": -2.0, "clipped_samples": 0,
                  "cut_ratio": 0.05, "render_total_s": 10.0, "asr_s": 5.0,
                  "total_wall_s": 15.0, "n_segments": 3, "chars": 9, "ok": True,
                  "error": ""}],
        "summary": {"mode": mode, "n": 1, "full_chain_rtf": 0.1, "ok": True},
    }


def test_regress_command_reuse_json(capsys, tmp_path: Path, monkeypatch):
    """`cli regress --reuse --json`：跑 run_regression + check_regression 并落 p8.json。"""
    import app.tools.regress as regress_mod

    seen: dict = {}

    def fake_run(data_dir, cfg, *, mode="reuse", out_dir=None, cache_dir=None, limit=None):
        seen.update(mode=mode, out_dir=out_dir, cache_dir=cache_dir, limit=limit)
        return _fake_regress_result(mode)

    def fake_check(result, *, baseline_csv=None, scan_dir=None):
        seen["baseline_csv"] = str(baseline_csv) if baseline_csv else None
        return {"ok": True, "failures": [], "summary": {"ok": True}}

    monkeypatch.setattr(regress_mod, "run_regression", fake_run)
    monkeypatch.setattr(regress_mod, "check_regression", fake_check)
    regress_dir = tmp_path / "regress"

    code = main(["regress", "--reuse", "--limit", "2", "--json",
                 "--regress-dir", str(regress_dir)])

    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["mode"] == "reuse"
    assert seen["mode"] == "reuse" and seen["limit"] == 2
    assert (regress_dir / "p8.json").is_file()
    assert not (regress_dir / "p8_rerun.json").exists()


def test_regress_command_failure_exit_1(capsys, tmp_path: Path, monkeypatch):
    """断言失败（check_regression.ok=False）→ 退出码 1，p8.json 仍落盘（含 failures）。"""
    import app.tools.regress as regress_mod

    monkeypatch.setattr(regress_mod, "run_regression",
                        lambda *a, **k: _fake_regress_result("reuse"))
    monkeypatch.setattr(regress_mod, "check_regression",
                        lambda *a, **k: {"ok": False, "failures": ["tp: 超限"],
                                         "summary": {"ok": False}})
    regress_dir = tmp_path / "regress"

    code = main(["regress", "--json", "--regress-dir", str(regress_dir)])

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["failures"] == ["tp: 超限"]


def test_regress_command_rerun_uses_independent_dirs(capsys, tmp_path: Path, monkeypatch):
    """`--rerun` 默认 out_p8/work/cache_p8；安全校验 ValueError → 退出码 2。"""
    import app.tools.regress as regress_mod

    seen: dict = {}

    def fake_run(data_dir, cfg, *, mode="reuse", out_dir=None, cache_dir=None, limit=None):
        seen.update(mode=mode, out_dir=out_dir, cache_dir=cache_dir)
        return _fake_regress_result(mode)

    monkeypatch.setattr(regress_mod, "run_regression", fake_run)
    monkeypatch.setattr(regress_mod, "check_regression",
                        lambda *a, **k: {"ok": True, "failures": [], "summary": {}})
    regress_dir = tmp_path / "regress"
    assert main(["regress", "--rerun", "--json",
                 "--regress-dir", str(regress_dir)]) == EXIT_OK
    assert seen["mode"] == "rerun"
    assert (regress_dir / "p8_rerun.json").is_file()

    def raising_run(*a, **k):
        raise ValueError("rerun 模式禁止写 out/")

    monkeypatch.setattr(regress_mod, "run_regression", raising_run)
    capsys.readouterr()
    code = main(["regress", "--rerun", "--out", str(tmp_path / "out"),
                 "--regress-dir", str(regress_dir)])
    assert code == EXIT_USAGE
    assert "禁止写 out/" in capsys.readouterr().err
