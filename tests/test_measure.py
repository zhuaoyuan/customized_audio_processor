# -*- coding: utf-8 -*-
"""P4-T4 时间戳测量的单元测试（不加载 faster-whisper）。

覆盖：
- ``stat`` / ``count_chars`` 口径；
- 边界匹配规则（词中点落在段内、首词 start / 末词 end、窗口边缘排除带）；
- 窗口裁剪 ``resolve_window``；
- ``measure_job`` 全链路（注入词级参照，不跑 whisper）：窗口/统计/字符对照/基线/验收；
- job 收集、CLI 参数与 ``--json --out`` 落盘；
- 错误路径（transcript 缺失、窗口越界）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.tools import measure as M


# ---------------------------------------------------------------- 夹具
def _seg(start, end, text="测试"):
    return {"start": float(start), "end": float(end), "orig_start": float(start) + 480.0,
            "orig_end": float(end) + 480.0, "speaker": "主讲",
            "text_raw": text, "text": text, "flags": []}


def _word(start, end, text="词"):
    return {"start": float(start), "end": float(end), "word": text}


def _write_job(tmp_path: Path, segments) -> Path:
    job_out = tmp_path / "out" / "job1"
    job_out.mkdir(parents=True, exist_ok=True)
    (job_out / "transcript.json").write_text(
        json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    return job_out


def _write_clean16k(tmp_path: Path, seconds: float = 60.0) -> Path:
    from scipy.io import wavfile

    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    wav = cache / "job1.clean16k.wav"
    wavfile.write(str(wav), M.SR, np.zeros(int(seconds * M.SR), dtype=np.float32))
    return wav


# ---------------------------------------------------------------- 基础口径
def test_count_chars_counts_han_letters_digits_only():
    assert M.count_chars("主讲：亚里士多德 Mete 123，。！") == 14
    assert M.count_chars("") == 0


def test_stat_basic():
    s = M.stat([0.1, -0.2, 0.3, -0.4])
    assert s["n"] == 4
    assert s["median"] == pytest.approx(-0.05, abs=1e-9)
    assert s["abs_median"] == pytest.approx(0.25, abs=1e-9)
    assert s["max_abs"] == pytest.approx(0.4, abs=1e-9)
    assert s["p90_abs"] > 0


def test_stat_empty():
    assert M.stat([]) == {"n": 0}


# ---------------------------------------------------------------- 边界匹配
def test_match_boundaries_exact():
    segments = [_seg(10.0, 20.0, "甲乙丙"), _seg(30.0, 40.0, "丁戊")]
    words = [
        _word(10.0, 10.6, "甲"), _word(11.0, 11.6, "乙"),
        _word(19.0, 20.0, "丙"),
        _word(30.0, 30.6, "丁"), _word(39.0, 40.0, "戊"),
    ]
    out = M.match_boundaries(segments, words, win_start=0.0, win_end=100.0)
    assert out["n_segments_in_window"] == 2
    assert out["n_matched"] == 2
    assert out["start_signed"]["median"] == pytest.approx(0.0, abs=1e-9)
    assert out["end_signed"]["median"] == pytest.approx(0.0, abs=1e-9)
    assert out["chars_transcript"] == 5
    assert out["chars_words"] == 5


def test_match_boundaries_uses_word_midpoint():
    segments = [_seg(10.0, 20.0)]
    words = [_word(9.0, 10.4, "跨"),  # 中点 9.7 在段外 → 不算
             _word(10.5, 11.0, "内"),
             _word(19.6, 20.6, "尾")]  # 中点 20.1 在段外 → 不算
    out = M.match_boundaries(segments, words, win_start=0.0, win_end=100.0)
    assert out["n_matched"] == 1
    assert out["start_signed"]["median"] == pytest.approx(10.0 - 10.5, abs=1e-9)
    assert out["end_signed"]["median"] == pytest.approx(20.0 - 11.0, abs=1e-9)


def test_match_boundaries_edge_margin_excluded():
    segments = [_seg(0.5, 5.0), _seg(1.0, 6.0), _seg(94.0, 99.5)]
    words = [_word(1.2, 1.8), _word(2.0, 2.6), _word(94.5, 95.0)]
    out = M.match_boundaries(segments, words, win_start=0.0, win_end=100.0,
                             margin_s=1.0)
    # 0.5 < win_start+1 排除；94.0 段 end=99.5 > win_end-1 排除；只剩 (1.0,6.0)
    assert out["n_segments_in_window"] == 1
    assert out["n_matched"] == 1


def test_match_boundaries_no_words_matched():
    out = M.match_boundaries([_seg(10.0, 20.0)], [], win_start=0.0, win_end=100.0)
    assert out["n_segments_in_window"] == 1
    assert out["n_matched"] == 0
    assert out["start_abs"] == {"n": 0}
    assert out["end_abs"] == {"n": 0}


def test_match_boundaries_ignores_invalid_and_empty_segments():
    segments = [_seg(10.0, 10.0), _seg(20.0, 25.0)]
    out = M.match_boundaries(segments, [_word(20.5, 21.0)], win_start=0.0, win_end=100.0)
    assert out["n_segments_in_window"] == 1
    assert out["n_matched"] == 1


# ---------------------------------------------------------------- 窗口
def test_resolve_window_clamps_to_audio():
    a, b, clamped = M.resolve_window(100.0, start=90.0, dur=60.0)
    assert (a, b) == (90.0, 100.0)
    assert clamped is True
    a, b, clamped = M.resolve_window(2000.0, start=600.0, dur=600.0)
    assert (a, b, clamped) == (600.0, 1200.0, False)


def test_resolve_window_invalid():
    with pytest.raises(ValueError):
        M.resolve_window(0.0)
    with pytest.raises(ValueError):
        M.resolve_window(100.0, start=100.0, dur=10.0)


# ---------------------------------------------------------------- measure_job
def test_measure_job_with_injected_words(cfg, tmp_path):
    segments = [_seg(12.0, 18.0, "甲乙丙"), _seg(30.0, 36.0, "丁戊")]
    _write_job(tmp_path, segments)
    _write_clean16k(tmp_path, seconds=60.0)

    def provider(window, win_start):
        assert abs(win_start - 10.0) < 1e-6
        return [
            _word(11.8, 12.2),   # 中点 12.0 在段内；ref 起点 11.8，我们 12.0 → +0.2
            _word(17.2, 17.9),   # ref 终点 17.9，我们 18.0 → +0.1
            _word(29.9, 30.2),   # +0.1
            _word(35.0, 35.7),   # +0.3
        ]

    result = M.measure_job("job1", cfg, start=10.0, dur=30.0, word_provider=provider)

    assert result["ok"] is True
    assert result["window"]["axis"] == "clean"
    assert result["window"]["start"] == pytest.approx(10.0, abs=1e-6)
    assert result["window"]["end"] == pytest.approx(40.0, abs=1e-6)
    assert result["n_matched"] == 2
    assert result["n_segments_in_window"] == 2
    assert result["summary"]["start_abs_median"] == pytest.approx(0.15, abs=1e-9)
    assert result["summary"]["end_abs_median"] == pytest.approx(0.2, abs=1e-9)
    assert result["chars"]["transcript"] == 5
    assert result["chars"]["reference_words"] == 4
    assert result["baseline"]["start_abs_median"] == 0.21
    assert result["acceptance"]["pass"] is True
    assert result["method"]["name"] == "injected_word_boundary"
    assert "boundaries" in result and "start_abs" in result["boundaries"]


def test_measure_job_missing_transcript(cfg, tmp_path):
    (tmp_path / "out" / "job1").mkdir(parents=True)
    result = M.measure_job("job1", cfg, start=0.0, dur=10.0)
    assert result["ok"] is False
    assert "transcript" in result["error"]


def test_measure_job_window_out_of_range(cfg, tmp_path):
    _write_job(tmp_path, [_seg(1.0, 2.0)])
    _write_clean16k(tmp_path, seconds=20.0)
    result = M.measure_job("job1", cfg, start=30.0, dur=10.0)
    assert result["ok"] is False
    assert "超出" in result["error"]


# ---------------------------------------------------------------- job 收集 / CLI
def test_collect_jobs_lists_dirs_with_transcript(tmp_path):
    out = tmp_path / "out"
    (out / "b").mkdir(parents=True)
    (out / "a").mkdir()
    (out / "c").mkdir()
    (out / "b" / "transcript.json").write_text("[]", encoding="utf-8")
    (out / "a" / "transcript.json").write_text("[]", encoding="utf-8")
    assert M.collect_jobs(out) == ["a", "b"]
    assert M.collect_jobs(tmp_path / "nope") == []


def test_build_parser_job_and_all_required():
    parser = M.build_parser()
    args = parser.parse_args(["--job", "x"])
    assert args.job == "x" and args.start == 600.0 and args.dur == 600.0
    assert args.beam == 1 and args.model == "base"
    args = parser.parse_args(["--all", "--start", "10", "--dur", "20", "--json"])
    assert args.all is True and args.start == 10.0 and args.dur == 20.0 and args.json is True
    with pytest.raises(SystemExit):
        parser.parse_args([])  # --job/--all 必选


def test_acceptance_thresholds():
    good_start = {"n": 10, "abs_median": 0.3, "p90_abs": 1.0}
    bad_end = {"n": 10, "abs_median": 0.9, "p90_abs": 2.0}
    acc = M._acceptance(good_start, bad_end)
    assert acc["start_abs_median_le_0.5"] is True
    assert acc["end_abs_median_le_0.5"] is False
    assert acc["pass"] is False
    assert M._acceptance(good_start, good_start)["pass"] is True


def test_main_json_out_writes_payload(cfg, tmp_path, monkeypatch):
    fake = {"ok": True, "job_id": "job1", "window": {"start": 0.0, "end": 10.0},
            "summary": {"start_abs_median": 0.1, "start_p90_abs": 0.2,
                        "end_abs_median": 0.1, "end_p90_abs": 0.2},
            "n_segments_in_window": 1, "n_matched": 1,
            "chars": {"transcript": 2, "reference_words": 2},
            "acceptance": {"pass": True}}
    monkeypatch.setattr(M, "measure_job", lambda *a, **k: fake)
    monkeypatch.setattr(M, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(M, "setup_logging", lambda *a, **k: None)
    out = tmp_path / "measure.json"
    code = M.main(["--job", "job1", "--json", "--out", str(out)])
    assert code == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["job_id"] == "job1" and saved["ok"] is True


def test_measure_json_is_utf8_and_no_nan(tmp_path):
    out = tmp_path / "m.json"
    M._write_json(out, {"ok": True, "v": float("nan"), "中文": "亚里士多德"})
    text = out.read_text(encoding="utf-8")
    assert "亚里士多德" in text
    assert "NaN" not in text
    assert json.loads(text)["v"] is None


def test_direct_script_invocation_help(project_root):
    """任务卡口径是 `python app/tools/measure.py ...`：直接运行必须可用（PEP 366 包上下文）。"""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(project_root / "app" / "tools" / "measure.py"), "--help"],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--job" in proc.stdout and "--all" in proc.stdout
