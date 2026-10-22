# -*- coding: utf-8 -*-
"""P8-T1 回归工具单测：``run_regression`` / ``check_regression`` 的 7 条断言 + 两种模式 + CLI。

约定
----
- 快速用例全部用 ``tmp_path`` 造 6 个假 ``out/<id>/report.json`` + 8 件产物，**不跑**真实
  流水线 / ASR / ffmpeg；``rerun`` 用例 monkeypatch ``app.tools.regress.batch_run``，
  不写真实 ``out_p8/``。
- 真实 6 文件只读回归放 ``@pytest.mark.slow``（秒级、只读 ``out/``，允许写
  ``work/regress/p8.json`` 作为 P8-T1 交付物）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import pytest

import app.tools.regress as regress
from app.config import load_config
from app.pipeline.orchestrator import PRODUCT_FILES
from app.tools.regress import check_regression, run_regression

ROOT = Path(__file__).resolve().parents[1]
REAL_OUT = ROOT / "out"
REAL_BASELINE = ROOT / "work" / "regress" / "audio.csv"

F24 = "7927721f_24_复旦哲学系丁耘_亚里士多德"
F00 = "621cb91c_0_20140227_丁耘_西方哲学"

#: 6 个假 job 的近似真实指标（job_id 与 P4_BASELINE / audio.csv 对齐，断言 5/7 可比）
JOB_SPECS: List[Dict[str, Any]] = [
    {"job_id": "0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05", "source": "data/f05.m4a",
     "duration_s": 6325.743, "clean_s": 6201.13, "cut_ratio": 0.0197, "asl_out": -20.07,
     "tp": -5.1, "clip": 0, "render_total": 668.71, "asr_s": 592.69,
     "n_segments": 1334, "chars": 28349},
    {"job_id": "4db18037_20_丁耘理想国_古希腊哲学", "source": "data/f20.m4a",
     "duration_s": 6151.407, "clean_s": 5869.60, "cut_ratio": 0.0458, "asl_out": -20.06,
     "tp": -1.9, "clip": 0, "render_total": 735.54, "asr_s": 224.60,
     "n_segments": 1695, "chars": 26031},
    {"job_id": F00, "source": "data/f00.m4a",
     "duration_s": 3047.2, "clean_s": 2745.28, "cut_ratio": 0.0991, "asl_out": -20.02,
     "tp": -1.3, "clip": 0, "render_total": 192.16, "asr_s": 107.83,
     "n_segments": 998, "chars": 9858},
    {"job_id": F24, "source": "data/f24.m4a",
     "duration_s": 4673.294, "clean_s": 3806.46, "cut_ratio": 0.1855, "asl_out": -20.04,
     "tp": -2.5, "clip": 0, "render_total": 230.15, "asr_s": 242.56,
     "n_segments": 1175, "chars": 15524},
    {"job_id": "9f3feb49_4_丁耘理想国_古希腊哲学_第四卷", "source": "data/f04.m4a",
     "duration_s": 8294.98, "clean_s": 8131.83, "cut_ratio": 0.0197, "asl_out": -20.03,
     "tp": -2.1, "clip": 0, "render_total": 496.26, "asr_s": 367.73,
     "n_segments": 2299, "chars": 38060},
    {"job_id": "b00e197d_复旦哲学系丁耘_孔子_论语_06_19", "source": "data/f06.m4a",
     "duration_s": 6070.625, "clean_s": 5327.85, "cut_ratio": 0.1224, "asl_out": -20.02,
     "tp": -1.6, "clip": 0, "render_total": 323.39, "asr_s": 249.38,
     "n_segments": 1679, "chars": 22224},
]

#: P3 基线（work/regress/audio.csv 的真实值；cut_ratio 是 0..1 比例）
BASELINE_VALUES: Dict[str, Dict[str, float]] = {
    spec["job_id"]: {"cut_ratio": base_cut, "asl_out_dbfs": spec["asl_out"],
                     "true_peak_out_dbtp": spec["tp"]}
    for spec, base_cut in zip(
        JOB_SPECS, (0.02, 0.046, 0.099, 0.185, 0.02, 0.122),
    )
}


# ================================================================ 造数据
def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _report_payload(spec: Dict[str, Any]) -> Dict[str, Any]:
    """按真实 report.json 结构造一份（字段足够覆盖 7 条断言 + P3 旧 evaluate）。"""
    return {
        "id": spec["job_id"],
        "source": spec["source"],
        "duration_s": spec["duration_s"],
        "levels": {
            "asl_in_dbfs": -12.0,
            "asl_clean_dbfs": -11.0,
            "asl_out_dbfs": spec["asl_out"],
            "asl_dev_db": round(spec["asl_out"] + 20.0, 3),
            "true_peak_out_dbtp": spec["tp"],
            "clipped_samples_out": spec["clip"],
        },
        "trim": {"clean_dur_s": spec["clean_s"], "cut_ratio": spec["cut_ratio"]},
        "timing": {
            "decode_s": 2.89, "analyze_s": 1.21, "denoise_s": 17.49, "trim_s": 2.93,
            "render_s": round(spec["render_total"] - 200.0, 2),
            "encode_s": 100.0, "measure_s": 50.0, "total_s": spec["render_total"],
            "asr_s": spec["asr_s"], "diarize_s": 0.01,
            "correct_s": 0.01, "export_text_s": 0.03,
        },
        "loudness": {"gain": {"type": "v2", "gain_db": -1.0, "min_db": -3.0,
                              "max_db": 2.0, "max_rate_db_s": 0.4,
                              "limiter_triggers": 0}},
        "asr": {"n_segments": spec["n_segments"], "chars": spec["chars"],
                "elapsed_s": spec["asr_s"], "rtf": 0.05},
        "speakers": {"n": 1, "method": "none"},
        "correct": {"enabled": False, "chunks": 0, "edits": 0,
                    "rejected": 0, "cost_cny": 0.0},
        "ok": True,
    }


def _write_job(out_root: Path, spec: Dict[str, Any],
               missing: Iterable[str] = ()) -> Path:
    """写一个假 job 目录（8 件产物；``missing`` 里的文件不写）。"""
    job_out = out_root / spec["job_id"]
    job_out.mkdir(parents=True, exist_ok=True)
    skip = set(missing)
    for name in PRODUCT_FILES:
        if name in skip:
            continue
        if name == "report.json":
            _write_json(job_out / name, _report_payload(spec))
        else:
            (job_out / name).write_bytes(b"fake-" + name.encode("utf-8"))
    return job_out


def _write_baseline(path: Path, specs: Optional[List[Dict[str, Any]]] = None) -> Path:
    """写 P3 基线 CSV（utf-8-sig，列与 work/regress/audio.csv 子集一致）。"""
    specs = JOB_SPECS if specs is None else specs
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["job_id", "cut_ratio", "asl_out_dbfs", "true_peak_out_dbtp"]
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow({"job_id": spec["job_id"], **BASELINE_VALUES[spec["job_id"]]})
    return path


def _write_measure(path: Path, *, start_median: float = 0.18,
                   start_p90: float = 0.47, end_median: float = 0.26,
                   end_p90: float = 0.59) -> Path:
    """写时间戳基线（字段与真实 work/measure_f24.json 一致）。"""
    return _write_json(path, {
        "ok": True,
        "boundaries": {
            "start_abs": {"n": 153, "abs_median": start_median, "p90_abs": start_p90},
            "end_abs": {"n": 153, "abs_median": end_median, "p90_abs": end_p90},
        },
        "summary": {
            "n": 153,
            "start_abs_median": start_median,
            "start_p90_abs": start_p90,
            "end_abs_median": end_median,
            "end_p90_abs": end_p90,
        },
        "acceptance": {"pass": True},
    })


@pytest.fixture()
def env(tmp_path: Path, cfg: Any) -> SimpleNamespace:
    """6 个假 job + P3 基线 CSV + 时间戳基线的临时回归环境（全部在 tmp_path）。"""
    out = cfg.path("out_dir")  # conftest 的 cfg 已把 out/cache/work 指到 tmp_path
    data = tmp_path / "data"
    data.mkdir()
    work = tmp_path / "work"
    scan = work / "scan"
    scan.mkdir(parents=True)
    regress_dir = tmp_path / "regress"
    regress_dir.mkdir()
    for spec in JOB_SPECS:
        _write_job(out, spec)
    baseline = _write_baseline(regress_dir / "audio.csv")
    measure = _write_measure(work / "measure_f24.json")
    return SimpleNamespace(
        cfg=cfg, tmp=tmp_path, out=out, data=data, work=work, scan=scan,
        regress_dir=regress_dir, baseline=baseline, measure=measure,
    )


# ================================================================ 测试辅助
def _read_report(env: SimpleNamespace, job_id: str) -> Dict[str, Any]:
    return json.loads((env.out / job_id / "report.json").read_text(encoding="utf-8"))


def _write_report(env: SimpleNamespace, job_id: str, report: Dict[str, Any]) -> None:
    _write_json(env.out / job_id / "report.json", report)


def _patch_report(env: SimpleNamespace, job_id: str,
                  mutate: Any) -> Dict[str, Any]:
    report = _read_report(env, job_id)
    mutate(report)
    _write_report(env, job_id, report)
    return report


def _run_and_check(env: SimpleNamespace, *, baseline: Any = "default",
                   scan: Any = None, limit: Optional[int] = None):
    result = run_regression(env.data, env.cfg, mode="reuse", limit=limit)
    baseline_csv = env.baseline if baseline == "default" else baseline
    check = check_regression(result, baseline_csv=baseline_csv,
                             scan_dir=scan if scan is not None else env.scan)
    return result, check


def _failure_rules(check: Dict[str, Any]) -> List[str]:
    return [str(item).split(":", 1)[0] for item in check["failures"]]


# ================================================================ reuse 正常路径
def test_reuse_reads_six_jobs_and_contract(env: SimpleNamespace) -> None:
    result, check = _run_and_check(env)

    assert result["mode"] == "reuse"
    assert result["n"] == 6
    assert len(result["rows"]) == 6
    assert all(row["ok"] for row in result["rows"])
    # rows 契约字段一个不少
    required = {
        "job_id", "source", "clean_s", "asl_out_dbfs", "asl_dev_db",
        "true_peak_out_dbtp", "clipped_samples", "cut_ratio", "render_total_s",
        "asr_s", "total_wall_s", "n_segments", "chars", "ok", "error",
    }
    for row in result["rows"]:
        assert required <= set(row), required - set(row)
    # reuse 全链口径 = render total_s + asr + diarize + correct + export_text
    f24 = next(row for row in result["rows"] if row["job_id"] == F24)
    expected_wall = 230.15 + 242.56 + 0.01 + 0.01 + 0.03
    assert f24["total_wall_s"] == pytest.approx(expected_wall, abs=0.02)
    # 汇总字段
    summary = result["summary"]
    assert summary["asl_spread_db"] == pytest.approx(0.05, abs=1e-6)
    assert summary["tp_max_dbtp"] == -1.3
    assert summary["clipped_total"] == 0
    assert summary["max_render_total_s"] == pytest.approx(735.54, abs=0.1)
    assert summary["full_chain_rtf"] == pytest.approx(
        summary["sum_wall_s"] / summary["sum_input_s"], rel=1e-3
    )
    # 7 条断言全过
    assert check["ok"] is True, check["failures"]
    assert check["failures"] == []
    assert check["summary"]["baseline_checked"] is True
    assert check["summary"]["timestamp"]["start_abs_median"] == 0.18


def test_reuse_limit(env: SimpleNamespace) -> None:
    result = run_regression(env.data, env.cfg, mode="reuse", limit=2)
    assert result["n"] == 2
    assert len(result["rows"]) == 2


def test_reuse_limit_zero(env: SimpleNamespace) -> None:
    result = run_regression(env.data, env.cfg, mode="reuse", limit=0)
    assert result["n"] == 0


def test_reuse_empty_out_dir_fails(env: SimpleNamespace) -> None:
    empty = env.tmp / "empty_out"
    empty.mkdir()
    result = run_regression(env.data, env.cfg, mode="reuse", out_dir=empty)
    assert result["n"] == 0
    assert result["ok"] is False
    check = check_regression(result, baseline_csv=None, scan_dir=env.scan)
    assert check["ok"] is False
    assert "rows" in _failure_rules(check)


def test_reuse_missing_report_row_error(env: SimpleNamespace) -> None:
    job_out = env.out / F24
    (job_out / "report.json").unlink()
    result, check = _run_and_check(env)
    row = next(row for row in result["rows"] if row["job_id"] == F24)
    assert row["ok"] is False
    assert "report.json" in row["error"]
    assert check["ok"] is False
    assert "job" in _failure_rules(check)


# ================================================================ 断言 1–4（跨文件）
def test_assert_asl_spread_fail(env: SimpleNamespace) -> None:
    _patch_report(env, F24, lambda r: r["levels"].update(asl_out_dbfs=-21.5))
    result, check = _run_and_check(env)
    assert check["ok"] is False
    assert "asl_spread" in _failure_rules(check)
    assert result["ok"] is True  # 产物齐全，只是数值断言失败


def test_assert_true_peak_fail(env: SimpleNamespace) -> None:
    _patch_report(env, F00, lambda r: r["levels"].update(true_peak_out_dbtp=-0.5))
    _, check = _run_and_check(env)
    assert "true_peak" in _failure_rules(check)


def test_assert_clipped_fail(env: SimpleNamespace) -> None:
    _patch_report(env, "9f3feb49_4_丁耘理想国_古希腊哲学_第四卷",
                  lambda r: r["levels"].update(clipped_samples_out=7))
    _, check = _run_and_check(env)
    assert "clipped" in _failure_rules(check)
    assert check["summary"]["clipped_total"] == 7


def test_assert_render_total_fail(env: SimpleNamespace) -> None:
    def mutate(report: Dict[str, Any]) -> None:
        report["timing"]["total_s"] = 950.0
        report["trim"]["clean_dur_s"] = 1.0

    _patch_report(env, "4db18037_20_丁耘理想国_古希腊哲学", mutate)
    _, check = _run_and_check(env)
    assert "render_total" in _failure_rules(check)
    assert check["summary"]["max_render_total_s"] == 950.0


# ================================================================ 断言 5（P3 基线）
def test_assert_cut_ratio_drift_fail(env: SimpleNamespace) -> None:
    # 基线 0.185 → 0.30 = 11.5 pp > 5 pp
    _patch_report(env, F24, lambda r: r["trim"].update(cut_ratio=0.30))
    _, check = _run_and_check(env)
    assert "baseline" in _failure_rules(check)
    assert any("cut_ratio" in item for item in check["failures"])


def test_assert_baseline_asl_and_tp_dev_fail(env: SimpleNamespace) -> None:
    def mutate(report: Dict[str, Any]) -> None:
        report["levels"]["asl_out_dbfs"] = -20.7   # 基线 −20.02 → 偏差 0.68 > 0.5
        report["levels"]["asl_dev_db"] = -0.7
        report["levels"]["true_peak_out_dbtp"] = -2.4  # 基线 −1.3 → 偏差 1.1 > 1.0

    _patch_report(env, F00, mutate)
    _, check = _run_and_check(env)
    baseline_failures = [item for item in check["failures"] if item.startswith("baseline:")]
    assert len(baseline_failures) == 2, check["failures"]
    assert any("ASL" in item for item in baseline_failures)
    assert any("真峰值" in item for item in baseline_failures)
    assert "asl_spread" not in _failure_rules(check)  # 极差 0.68 ≤ 1.0，不误报


def test_assert_baseline_missing_job_fail(env: SimpleNamespace) -> None:
    specs = [spec for spec in JOB_SPECS if spec["job_id"] != F24]
    _write_baseline(env.baseline, specs)
    _, check = _run_and_check(env)
    assert "baseline" in _failure_rules(check)
    assert any(F24 in item and "基线缺该 job" in item for item in check["failures"])


def test_baseline_none_skips_assertion(env: SimpleNamespace) -> None:
    _, check = _run_and_check(env, baseline=None)
    assert check["ok"] is True, check["failures"]
    assert check["summary"]["baseline_checked"] is False
    assert "跳过" in check["summary"]["baseline_note"]


def test_assert_missing_artifact_fail(env: SimpleNamespace) -> None:
    (env.out / F24 / "transcript.txt").unlink()
    result, check = _run_and_check(env)
    row = next(row for row in result["rows"] if row["job_id"] == F24)
    assert row["ok"] is False
    assert "transcript.txt" in row["error"]
    assert "job" in _failure_rules(check)


# ================================================================ 断言 6（时间戳）
def test_assert_measure_missing_fail(env: SimpleNamespace) -> None:
    env.measure.unlink()
    _, check = _run_and_check(env)
    assert "timestamp" in _failure_rules(check)
    assert any("时间戳基线缺失" in item for item in check["failures"])


def test_assert_measure_missing_field_fail(env: SimpleNamespace) -> None:
    _write_json(env.measure, {"ok": True, "boundaries": {"start_abs": {}}})
    _, check = _run_and_check(env)
    assert "timestamp" in _failure_rules(check)
    assert any("时间戳基线缺失" in item for item in check["failures"])


def test_assert_timestamp_threshold_fail(env: SimpleNamespace) -> None:
    _write_measure(env.measure, start_median=0.8, start_p90=1.7)
    _, check = _run_and_check(env)
    rules = _failure_rules(check)
    assert rules.count("timestamp") == 2, check["failures"]
    assert check["summary"]["timestamp"]["ok"] is False


# ================================================================ 断言 7（ASR 基线）
def test_assert_asr_deviation_fail(env: SimpleNamespace) -> None:
    _patch_report(env, F24, lambda r: r["asr"].update(n_segments=1400, chars=18000))
    _, check = _run_and_check(env)
    asr_failures = [item for item in check["failures"] if item.startswith("asr:")]
    assert len(asr_failures) == 2, check["failures"]
    assert check["summary"]["asr_baseline_checked"] is True


def test_asr_baseline_missing_job_skipped(env: SimpleNamespace) -> None:
    # 造一个不在 P4_BASELINE 里的 job：断言 7 应跳过（其它断言仍按数据判断）
    spec = {"job_id": "unknown_job_for_asr", "source": "data/unknown.m4a",
            "duration_s": 100.0, "clean_s": 100.0, "cut_ratio": 0.0,
            "asl_out": -20.0, "tp": -2.0, "clip": 0, "render_total": 10.0,
            "asr_s": 5.0, "n_segments": 10, "chars": 100}
    _write_job(env.out, spec)
    result = run_regression(env.data, env.cfg, mode="reuse", limit=1)  # limit=1 只读 0a7
    assert result["n"] == 1
    # 直接构造含 unknown job 的 result 验证跳过逻辑
    result["rows"].append(regress._row_from_report(
        spec["job_id"], spec["source"], _report_payload(spec), ok=True))
    result["rows"][0], result["rows"][1] = result["rows"][1], result["rows"][0]
    check = check_regression(result, baseline_csv=None, scan_dir=env.scan)
    assert not any(item.startswith("asr:") for item in check["failures"]), check["failures"]
    known = [item for item in check["summary"]["asr_deviation"]]
    assert all(item["job_id"] != spec["job_id"] for item in known)


# ================================================================ rerun（mock 编排器）
def test_rerun_uses_independent_dirs(env: SimpleNamespace, monkeypatch: Any) -> None:
    data = env.tmp / "rerun_data"
    data.mkdir()
    for index in range(2):
        (data / f"sample{index}.m4a").write_bytes(b"FAKE-AUDIO")
    out_p8 = env.tmp / "out_p8"
    cache_p8 = env.tmp / "cache_p8"
    captured: Dict[str, Any] = {}

    def fake_batch(inputs: Any, run_cfg: Any, **kwargs: Any) -> Dict[str, Any]:
        captured["inputs"] = list(inputs)
        captured["cfg"] = run_cfg
        captured.update(kwargs)
        results = []
        for entry in inputs:
            if isinstance(entry, tuple):
                job_id, source = str(entry[0]), Path(entry[1])
            else:
                source = Path(entry)
                job_id = source.stem
            report = _report_payload(JOB_SPECS[0])
            report["id"] = job_id
            report["source"] = str(source)
            results.append({"ok": True, "job_id": job_id, "source": str(source),
                            "report": report, "timings": {"total_s": 111.1},
                            "stage": "export", "error": ""})
        return {"ok": True, "n": len(results), "results": results,
                "failures": [], "skipped": [], "stopped_reason": ""}

    monkeypatch.setattr(regress, "batch_run", fake_batch)
    result = run_regression(data, env.cfg, mode="rerun",
                            out_dir=out_p8, cache_dir=cache_p8)

    run_cfg = captured["cfg"]
    assert run_cfg.path("out_dir") == out_p8.resolve()
    assert run_cfg.path("db_path") == (out_p8 / "jobs.db").resolve()
    assert run_cfg.path("logs_dir") == (out_p8 / "logs").resolve()
    assert run_cfg.path("cache_dir") == cache_p8.resolve()
    assert run_cfg.path("quarantine_dir") == (out_p8 / "quarantine").resolve()
    # 参数真正生效：绝不指向原始 out/
    assert run_cfg.path("out_dir") != env.cfg.path("out_dir")
    assert captured["resume"] is False and captured["force"] is True
    assert len(captured["inputs"]) == 2
    assert result["mode"] == "rerun" and result["n"] == 2
    assert all(row["ok"] for row in result["rows"])
    # rerun 的 total_wall_s 用实测 run_job wall（mock 为 111.1）
    assert result["rows"][0]["total_wall_s"] == 111.1
    assert result["summary"]["full_chain_rtf"] is not None


def test_rerun_default_dirs(env: SimpleNamespace, monkeypatch: Any) -> None:
    data = env.tmp / "rerun_default_data"
    data.mkdir()
    (data / "one.m4a").write_bytes(b"FAKE-AUDIO")
    captured: Dict[str, Any] = {}

    def fake_batch(inputs: Any, run_cfg: Any, **kwargs: Any) -> Dict[str, Any]:
        captured["cfg"] = run_cfg
        return {"ok": True, "n": 0, "results": [], "failures": [],
                "skipped": [], "stopped_reason": ""}

    monkeypatch.setattr(regress, "batch_run", fake_batch)
    run_regression(data, env.cfg, mode="rerun")  # 默认 out_p8 / work/cache_p8
    assert captured["cfg"].path("out_dir") == (regress.PROJECT_ROOT / "out_p8").resolve()
    assert captured["cfg"].path("cache_dir") == (
        regress.PROJECT_ROOT / "work" / "cache_p8").resolve()


def test_rerun_refuses_writing_original_out(env: SimpleNamespace,
                                            monkeypatch: Any) -> None:
    def fake_batch(*args: Any, **kwargs: Any):  # pragma: no cover - 不应被调用
        raise AssertionError("目录安全校验失败时不应调用 batch_run")

    monkeypatch.setattr(regress, "batch_run", fake_batch)
    out_target = env.tmp / "out_p8"
    cfg2 = env.cfg.replace({"paths.out_dir": str(out_target)})
    with pytest.raises(ValueError, match="禁止写 out/"):
        run_regression(env.data, cfg2, mode="rerun", out_dir=out_target)


def test_rerun_limit(env: SimpleNamespace, monkeypatch: Any) -> None:
    data = env.tmp / "rerun_limit_data"
    data.mkdir()
    for index in range(3):
        (data / f"clip{index}.m4a").write_bytes(b"FAKE-AUDIO")
    captured: Dict[str, Any] = {}

    def fake_batch(inputs: Any, run_cfg: Any, **kwargs: Any) -> Dict[str, Any]:
        captured["inputs"] = list(inputs)
        return {"ok": True, "n": 0, "results": [], "failures": [],
                "skipped": [], "stopped_reason": ""}

    monkeypatch.setattr(regress, "batch_run", fake_batch)
    result = run_regression(data, env.cfg, mode="rerun", out_dir=env.tmp / "out_p8",
                            cache_dir=env.tmp / "cache_p8", limit=1)
    assert len(captured["inputs"]) == 1
    assert result["n"] == 0


# ================================================================ CLI
def test_cli_reuse_json_writes_p8(env: SimpleNamespace, capsys: Any) -> None:
    _write_measure(env.regress_dir.parent / "measure_f24.json")  # scan_dir = regress_dir 上级/scan
    rc = regress.main(["--reuse", "--json", "--out", str(env.out),
                       "--regress-dir", str(env.regress_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["mode"] == "reuse"
    assert payload["n"] == 6
    p8 = env.regress_dir / "p8.json"
    assert p8.is_file()
    saved = json.loads(p8.read_text(encoding="utf-8"))
    assert len(saved["rows"]) == 6
    assert saved["summary"]["asl_spread_db"] is not None
    assert saved["summary"]["full_chain_rtf"] is not None
    # 默认路径不重写 P3 基线 audio.csv
    assert (env.regress_dir / "audio.csv").is_file()


def test_cli_legacy_from_reports_keeps_audio_csv(env: SimpleNamespace,
                                                 capsys: Any) -> None:
    rc = regress.main(["--from-reports", "--json", "--out", str(env.out),
                       "--regress-dir", str(env.regress_dir)])
    assert rc == 0
    assert (env.regress_dir / "audio.csv").is_file()
    with (env.regress_dir / "audio.csv").open(encoding="utf-8-sig") as fh:
        assert len(list(csv.DictReader(fh))) == 6
    assert not (env.regress_dir / "p8.json").exists()
    # 旧模式 --json 输出 metrics（有 ok 字段）
    assert json.loads(capsys.readouterr().out)["ok"] is True


# ================================================================ slow：真实 6 文件
@pytest.mark.slow
def test_reuse_real_six_jobs_writes_p8_json() -> None:
    """真实 --reuse：读 out/ 6 个 job 的 report + 8 件产物，7 条断言全过 → p8.json。"""
    assert REAL_BASELINE.is_file(), f"P3 基线缺失：{REAL_BASELINE}"
    cfg = load_config()

    result = run_regression(cfg.path("data_dir"), cfg, mode="reuse")
    assert result["mode"] == "reuse"
    assert result["n"] == 6, [row["job_id"] for row in result["rows"]]
    assert all(row["ok"] for row in result["rows"]), [
        (row["job_id"], row["error"]) for row in result["rows"] if not row["ok"]
    ]
    # 8 件产物齐全（reuse 行 ok 已覆盖，这里再显式核对一次）
    for row in result["rows"]:
        job_out = cfg.path("out_dir") / row["job_id"]
        assert all((job_out / name).is_file() for name in PRODUCT_FILES)

    check = check_regression(result, baseline_csv=REAL_BASELINE,
                             scan_dir=ROOT / "work" / "scan")
    assert check["ok"] is True, check["failures"]
    assert check["failures"] == []

    summary = {**result["summary"], **check["summary"]}
    assert summary["asl_spread_db"] <= regress.ASL_SPREAD_MAX_DB
    assert summary["tp_max_dbtp"] <= regress.TP_MAX_DBTP
    assert summary["clipped_total"] == 0
    assert summary["max_render_total_s"] <= regress.MAX_TOTAL_S
    # 实测全链 RTF ≈0.128（docs 计划写 0.129）；给宽容区间防机器抖动
    assert 0.10 <= summary["full_chain_rtf"] <= 0.16
    assert summary["sum_input_s"] == pytest.approx(34563.25, abs=1.0)
    assert summary["sum_clean_s"] == pytest.approx(32082.15, abs=1.0)

    out_path = ROOT / "work" / "regress" / "p8.json"
    payload = {
        "ok": check["ok"],
        "mode": result["mode"],
        "n": result["n"],
        "generated_at": "",
        "rows": result["rows"],
        "summary": summary,
        "failures": check["failures"],
    }
    _write_json(out_path, payload)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert saved["n"] == 6
    assert len(saved["rows"]) == 6
    assert saved["summary"]["asl_spread_db"] <= 1.0
    assert saved["summary"]["full_chain_rtf"] > 0
