# -*- coding: utf-8 -*-
"""P4-T3 说话人标注的单元测试（不依赖真实模型/真实 transcript）。

覆盖：
- 采样窗口规划（正文/预算/短音频退化）；
- 纯函数聚类归并（average-linkage 余弦）、稳定说话人判定、段级标签映射；
- 全量 diar 的身份门槛（最长=主讲、碎片不造身份）；
- ``diarize_job`` 的 none 模式（全主讲 + report 读-改-写 + 幂等跳过）；
- sampled=单人的保守回退（绝不臆造）、sampled→full 的标签落盘、模型缺失失败；
- ``ensure_clean16k`` 复用缓存 / 缺 clean.m4a 报错。

真实模型相关的端到端（f24/f00 正文抽样）在 ``work/diar_p4_check.json`` 留证据，
不在 pytest 里跑，避免 CI/无模型环境失败。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.pipeline import diarize as D


# ---------------------------------------------------------------- 夹具
def _seg(start, end, speaker="主讲", text="测试文本", flags=None):
    return {
        "start": float(start), "end": float(end),
        "orig_start": float(start) + 480.0, "orig_end": float(end) + 480.0,
        "speaker": speaker, "text_raw": text, "text": text,
        "flags": list(flags or []),
    }


def _write_job(tmp_path: Path, segments, report=None) -> Path:
    job_out = tmp_path / "out" / "job1"
    job_out.mkdir(parents=True, exist_ok=True)
    (job_out / "transcript.json").write_text(
        json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    if report is not None:
        (job_out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return job_out


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tiny_wav(tmp_path: Path, seconds: float = 2.0) -> Path:
    from scipy.io import wavfile

    path = tmp_path / "clean16k.wav"
    wavfile.write(str(path), D.SR,
                  np.zeros(int(seconds * D.SR), dtype=np.float32))
    return path


# ---------------------------------------------------------------- 采样窗口
def test_plan_sample_windows_default_budget():
    windows = D.plan_sample_windows(3806.46, sample_minutes=10, sample_segments=3,
                                    body_margin_s=120)
    assert len(windows) == 3
    assert windows == sorted(windows)
    for (a, b) in windows:
        assert b - a == pytest.approx(200.0, abs=0.02)  # 10 min / 3
        assert 120.0 - 1e-6 <= a and b <= 3806.46 - 120.0 + 1e-6
    # 互不重叠
    for (a1, b1), (a2, b2) in zip(windows, windows[1:]):
        assert b1 <= a2


def test_plan_sample_windows_three_five_minute_segments():
    windows = D.plan_sample_windows(3806.46, sample_minutes=15, sample_segments=3,
                                    body_margin_s=120)
    assert len(windows) == 3
    for a, b in windows:
        assert b - a == pytest.approx(300.0, abs=0.02)


def test_plan_sample_windows_short_audio_falls_back_to_whole_file():
    windows = D.plan_sample_windows(100.0)
    assert len(windows) == 3
    assert windows[0][0] == pytest.approx(0.0, abs=1e-6)
    assert windows[-1][1] == pytest.approx(100.0, abs=0.02)
    # 窗口不重叠
    for (a1, b1), (a2, b2) in zip(windows, windows[1:]):
        assert b1 <= a2 + 1e-6


@pytest.mark.parametrize("duration", [0.0, -5.0, float("nan")])
def test_plan_sample_windows_invalid_duration(duration):
    assert D.plan_sample_windows(duration) == []


def test_plan_sample_windows_single_segment():
    windows = D.plan_sample_windows(1000.0, sample_minutes=5, sample_segments=1,
                                    body_margin_s=100)
    assert len(windows) == 1
    assert windows[0][1] - windows[0][0] == pytest.approx(300.0, abs=0.02)
    assert windows[0][0] >= 100.0


# ---------------------------------------------------------------- 质心归并
def _unit(*values):
    v = np.asarray(values, dtype=np.float64)
    return v / np.linalg.norm(v)


def hist_ok(history):
    return isinstance(history, list) and all({"i", "j", "sim"} <= set(h) for h in history)


def test_merge_cluster_centroids_identical_vectors_merge():
    c = np.stack([_unit(1, 0), _unit(1, 0), _unit(0, 1)])
    labels, history = D.merge_cluster_centroids(c, threshold=0.7)
    assert len(set(labels.tolist())) == 2
    assert labels[0] == labels[1]
    assert hist_ok(history)
    # 确定性：同样输入同样结果
    labels2, _ = D.merge_cluster_centroids(c, threshold=0.7)
    assert labels.tolist() == labels2.tolist()


def test_merge_cluster_centroids_threshold_monotonic():
    c = np.stack([_unit(1.0, 0.0), _unit(0.9, 0.43589), _unit(0.0, 1.0)])  # cos(1,2)≈0.9
    loose, _ = D.merge_cluster_centroids(c, threshold=0.5)
    strict, _ = D.merge_cluster_centroids(c, threshold=0.95)
    assert len(set(loose.tolist())) == 2
    assert len(set(strict.tolist())) == 3


def test_merge_cluster_centroids_edge_cases():
    empty, hist = D.merge_cluster_centroids(np.zeros((0, 4)))
    assert empty.tolist() == [] and hist == []
    one, _ = D.merge_cluster_centroids(np.ones((1, 4)))
    assert one.tolist() == [0]
    # 质心未归一化也能正确工作
    c = np.stack([[2.0, 0.0], [5.0, 0.0], [0.0, 3.0]])
    labels, _ = D.merge_cluster_centroids(c, threshold=0.9)
    assert labels[0] == labels[1] != labels[2]


def test_merge_cluster_centroids_average_linkage_chain():
    # 相邻两两 cos=0.8；合并后与第三簇平均 cos=0.54 → th=0.5 时成链并成一簇
    c = np.stack([_unit(1.0, 0.0), _unit(0.8, 0.6), _unit(0.28, 0.96)])
    labels, _ = D.merge_cluster_centroids(c, threshold=0.5)
    assert len(set(labels.tolist())) == 1
    # 提高阈值到 0.6：第一次合并后平均 cos=0.54 < 0.6 → 只并前两个
    labels_strict, _ = D.merge_cluster_centroids(c, threshold=0.6)
    assert len(set(labels_strict.tolist())) == 2


# ---------------------------------------------------------------- 稳定说话人
def test_judge_stable_speakers_single_speaker_file():
    judge = D.judge_stable_speakers({0: 590.0, 1: 4.5, 2: 1.0})
    assert judge["is_multi"] is False
    assert judge["n_stable"] == 1
    assert judge["stable"][0]["cluster"] == 0
    assert judge["main_ratio"] == pytest.approx(590.0 / 595.5, abs=1e-4)


def test_judge_stable_speakers_two_real_speakers():
    judge = D.judge_stable_speakers({0: 500.0, 1: 100.0, 2: 4.0})
    assert judge["is_multi"] is True
    assert judge["n_stable"] == 2
    assert [s["cluster"] for s in judge["stable"]] == [0, 1]


def test_judge_stable_speakers_boundaries_inclusive():
    # 第二人恰好 5 s / 3%：算稳定（>= 边界）
    judge = D.judge_stable_speakers({0: 161.0, 1: 5.0})  # 5/166≈3.01%
    assert judge["n_stable"] == 2
    judge2 = D.judge_stable_speakers({0: 170.0, 1: 5.0})  # 5/175≈2.86% < 3%
    assert judge2["n_stable"] == 1


def test_judge_stable_speakers_empty():
    judge = D.judge_stable_speakers({})
    assert judge["n"] == 0 and judge["n_stable"] == 0 and judge["is_multi"] is False
    assert judge["speech_s"] == 0.0


def test_cluster_durations():
    assert D.cluster_durations([0, 0, 1], [1.0, 2.0, 3.5]) == {0: 3.0, 1: 3.5}
    assert D.cluster_durations([], []) == {}


# ---------------------------------------------------------------- 段映射
def test_assign_labels_max_overlap():
    segments = [_seg(0.0, 5.0), _seg(5.5, 10.0), _seg(10.5, 15.0)]
    diar_segments = [
        {"start": 0.0, "end": 5.0, "speaker": 0},
        {"start": 5.0, "end": 10.0, "speaker": 1},
        {"start": 10.0, "end": 15.0, "speaker": 1},
    ]
    assignments, stats = D.assign_labels(
        segments, diar_segments,
        speaker_by_cluster={0: "主讲", 1: "其他1"},
        main_cluster=0, min_overlap_ratio=0.3)
    assert [a["speaker"] for a in assignments] == ["主讲", "其他1", "其他1"]
    assert stats["n"] == 3 and stats["uncertain"] == 0


def test_assign_labels_low_overlap_falls_back_to_main():
    segments = [_seg(0.0, 10.0)]
    diar_segments = [{"start": 0.0, "end": 2.0, "speaker": 1}]  # 只覆盖 20%
    assignments, stats = D.assign_labels(
        segments, diar_segments, speaker_by_cluster={0: "主讲", 1: "其他1"},
        main_cluster=0, min_overlap_ratio=0.3)
    assert assignments[0]["speaker"] == "主讲"
    assert assignments[0]["uncertain"] is True
    assert stats["low_overlap"] == 1


def test_assign_labels_unknown_fragment_cluster_falls_back_to_main():
    segments = [_seg(0.0, 5.0)]
    diar_segments = [{"start": 0.0, "end": 5.0, "speaker": 7}]
    assignments, stats = D.assign_labels(
        segments, diar_segments, speaker_by_cluster={0: "主讲", 1: "其他1"},
        main_cluster=0)
    assert assignments[0]["speaker"] == "主讲"
    assert assignments[0]["conservative"] is True
    assert stats["fragment"] == 1


def test_assign_labels_no_overlap_and_empty_diar():
    segments = [_seg(0.0, 5.0)]
    assignments, stats = D.assign_labels(
        segments, [], speaker_by_cluster={0: "主讲"}, main_cluster=0)
    assert assignments[0]["speaker"] == "主讲"
    assert assignments[0]["uncertain"] is True
    assert stats["no_overlap"] == 1


def test_apply_full_labels_sets_speaker_and_flag_without_touching_text():
    segments = [_seg(0.0, 5.0), _seg(10.0, 15.0, flags=["keep"])]
    full = {
        "main_cluster": 0,
        "speaker_by_cluster": {0: "主讲", 1: "其他1"},
        "segments": [{"start": 0.0, "end": 5.0, "speaker": 0},
                     {"start": 10.0, "end": 15.0, "speaker": 1}],
    }
    changed, stats = D.apply_full_labels(segments, full, min_overlap_ratio=0.5)
    assert changed == 1
    assert [s["speaker"] for s in segments] == ["主讲", "其他1"]
    assert segments[0]["text"] == "测试文本"
    assert segments[1]["flags"] == ["keep"]  # 高重叠，不加 uncertain flag
    assert stats["n"] == 2


def test_other_speaker_name():
    assert D.other_speaker_name(1) == "其他1"
    assert D.other_speaker_name(2) == "其他2"


# ---------------------------------------------------------------- 全量身份门槛
def test_full_diarize_identity_threshold(monkeypatch):
    fake = {
        "ok": True, "duration_s": 600.0, "raw_speakers": 5, "merged_speakers": 4,
        "main_ratio": 0.8, "timings": {},
        "per_speaker_s": {"0": 480.0, "1": 100.0, "2": 4.0, "3": 1.0},
        "segments": [
            {"start": 0.0, "end": 480.0, "dur": 480.0, "raw_speaker": [0, 0],
             "merged_speaker": 0},
            {"start": 480.0, "end": 580.0, "dur": 100.0, "raw_speaker": [0, 1],
             "merged_speaker": 1},
            {"start": 580.0, "end": 584.0, "dur": 4.0, "raw_speaker": [0, 2],
             "merged_speaker": 2},
        ],
    }
    monkeypatch.setattr(D, "diarize_windows", lambda *a, **k: fake)
    full = D.full_diarize(np.zeros(1600, dtype=np.float32), None)
    # 4s 碎片簇（< max(fragment 3s, min_speaker 5s)）不造身份 → 只有主讲 + 其他1
    assert full["n_speakers"] == 2
    assert full["main_cluster"] == 0
    assert full["speaker_by_cluster"] == {0: "主讲", 1: "其他1"}
    assert len(full["clusters"]) == 4


# ---------------------------------------------------------------- sample_speakers
def _fake_sampled(is_multi: bool, n_stable: int = 1):
    per = {"0": 560.0, "1": 30.0, "2": 10.0} if is_multi else {"0": 590.0, "1": 10.0}
    return {
        "ok": True, "duration_s": 600.0, "windows": [[0.0, 200.0]], "n_windows": 1,
        "raw_speakers": 3, "raw_n_segments": 40, "merged_speakers": len(per),
        "per_speaker_s": per, "main_ratio": 0.9,
        "segments": [], "timings": {"total_s": 1.0},
        "threshold": 0.7, "merge_threshold": 0.7,
        "sampled": True, "min_speaker_s": 5.0, "min_speaker_ratio": 0.03,
        "n_speakers": len(per), "n_stable": n_stable,
        "stable": [{"cluster": 0, "dur_s": 560.0, "ratio": 0.93}],
        "is_multi": is_multi, "verdict": "multi" if is_multi else "single",
    }


def test_sample_speakers_judges_single(monkeypatch):
    monkeypatch.setattr(D, "diarize_windows", lambda *a, **k: _fake_sampled(False))
    result = D.sample_speakers(np.zeros(1600, dtype=np.float32), None,
                               windows=[(0.0, 200.0)])
    assert result["sampled"] is True
    assert result["is_multi"] is False
    assert result["verdict"] == "single"
    assert result["n_stable"] == 1


def test_sample_speakers_judges_multi(monkeypatch):
    monkeypatch.setattr(D, "diarize_windows", lambda *a, **k: _fake_sampled(True, 2))
    result = D.sample_speakers(np.zeros(1600, dtype=np.float32), None,
                               windows=[(0.0, 200.0)])
    assert result["is_multi"] is True
    assert result["n_stable"] == 2


# ---------------------------------------------------------------- diarize_job: none
def test_diarize_job_none_marks_all_main_and_updates_report(cfg, tmp_path):
    segments = [
        _seg(0.0, 3.0, speaker="其他1", text="甲"),
        _seg(3.0, 6.0, speaker="主讲", text="乙", flags=["keep"]),
    ]
    report = {
        "id": "job1", "ok": True,
        "decode": {"repaired": False, "ok": True},
        "trim": {"head_s": 480.0, "clean_dur_s": 3400.0},
        "timing": {"decode_s": 2.0},
    }
    job_out = _write_job(tmp_path, segments, report)

    result = D.diarize_job("job1", cfg)

    assert result["ok"] is True
    assert result["mode"] == "none"
    assert result["changed"] == 1
    saved = _read(job_out / "transcript.json")
    assert [s["speaker"] for s in saved] == ["主讲", "主讲"]
    assert saved[0]["text"] == "甲" and saved[1]["text"] == "乙"
    assert saved[1]["flags"] == ["keep"]  # none 模式绝不改 flags
    saved_report = _read(job_out / "report.json")
    assert saved_report["speakers"]["n"] == 1
    assert saved_report["speakers"]["method"] == "none"
    assert saved_report["decode"] == {"repaired": False, "ok": True}  # 既有字段保留
    assert saved_report["trim"]["clean_dur_s"] == 3400.0
    assert saved_report["timing"]["decode_s"] == 2.0
    assert "diarize_s" in saved_report["timing"]
    assert saved_report["ok"] is True


def test_diarize_job_none_is_idempotent_skip(cfg, tmp_path):
    segments = [_seg(0.0, 3.0)]
    job_out = _write_job(tmp_path, segments, {"id": "job1", "ok": True})
    first = D.diarize_job("job1", cfg)
    assert first["ok"] and not first["skipped"]
    second = D.diarize_job("job1", cfg)
    assert second["ok"] and second["skipped"] is True
    forced = D.diarize_job("job1", cfg, force=True)
    assert forced["ok"] and not forced["skipped"]


def test_diarize_job_none_missing_transcript_fails(cfg, tmp_path):
    (tmp_path / "out" / "job1").mkdir(parents=True)
    result = D.diarize_job("job1", cfg)
    assert result["ok"] is False
    assert result["stage"] == "diarize"
    assert "transcript" in result["error"]


def test_diarize_job_on_stage_callback(cfg, tmp_path):
    segments = [_seg(0.0, 3.0)]
    _write_job(tmp_path, segments, {"id": "job1", "ok": True})
    calls = []
    result = D.diarize_job("job1", cfg, on_stage=lambda st, status, m: calls.append((st, status, m)))
    assert result["ok"]
    assert calls and calls[0][0] == "diarize" and calls[0][1] == "done"


# ---------------------------------------------------------------- diarize_job: sampled
def test_diarize_job_sampled_single_keeps_main(cfg, tmp_path, monkeypatch):
    segments = [_seg(0.0, 3.0, speaker="其他1"), _seg(3.0, 6.0)]
    report = {"id": "job1", "ok": True, "trim": {"clean_dur_s": 2.0}}
    job_out = _write_job(tmp_path, segments, report)
    wav = _tiny_wav(tmp_path)
    monkeypatch.setattr(D, "ensure_clean16k", lambda *a, **k: wav)
    monkeypatch.setattr(D, "sample_speakers", lambda *a, **k: _fake_sampled(False))
    sampled_cfg = cfg.replace({"diar.mode": "sampled"})

    result = D.diarize_job("job1", sampled_cfg)

    assert result["ok"] is True and result["mode"] == "sampled"
    saved = _read(job_out / "transcript.json")
    assert all(s["speaker"] == "主讲" for s in saved)
    saved_report = _read(job_out / "report.json")
    assert saved_report["speakers"]["n"] == 1
    assert saved_report["speakers"]["method"] == "sampled"
    assert saved_report["speakers"]["sampled_judged_multi"] is False
    assert saved_report["speakers"]["raw_clusters"] == 3
    assert saved_report["trim"]["clean_dur_s"] == 2.0
    # 不写逐段 diar 明细到 report（体积控制）
    assert "segments" not in result["diar"]["sampled"]


def test_diarize_job_sampled_multi_applies_full_labels(cfg, tmp_path, monkeypatch):
    segments = [_seg(0.0, 3.0), _seg(6.0, 9.0)]
    job_out = _write_job(tmp_path, segments, {"id": "job1", "ok": True})
    wav = _tiny_wav(tmp_path)
    fake_full = {
        "ok": True, "n_speakers": 2, "main_cluster": 0,
        "speaker_by_cluster": {0: "主讲", 1: "其他1"},
        "segments": [{"start": 0.0, "end": 3.0, "speaker": 0},
                     {"start": 6.0, "end": 9.0, "speaker": 1}],
        "clusters": [{"cluster": 0, "dur_s": 300.0, "ratio": 0.9, "speaker": "主讲"},
                     {"cluster": 1, "dur_s": 30.0, "ratio": 0.1, "speaker": "其他1"}],
        "raw_speakers": 4, "merged_speakers": 2, "main_ratio": 0.9,
        "identity_min_s": 5.0, "timings": {},
    }
    monkeypatch.setattr(D, "ensure_clean16k", lambda *a, **k: wav)
    monkeypatch.setattr(D, "sample_speakers", lambda *a, **k: _fake_sampled(True, 2))
    monkeypatch.setattr(D, "full_diarize", lambda *a, **k: fake_full)
    sampled_cfg = cfg.replace({"diar.mode": "sampled"})

    result = D.diarize_job("job1", sampled_cfg)

    assert result["ok"] is True
    saved = _read(job_out / "transcript.json")
    assert [s["speaker"] for s in saved] == ["主讲", "其他1"]
    saved_report = _read(job_out / "report.json")
    assert saved_report["speakers"]["n"] == 2
    assert saved_report["speakers"]["method"] == "full"
    assert saved_report["speakers"]["trigger"] == "sampled"
    assert saved_report["speakers"]["raw_clusters"] == 4  # 全量跑的原始簇数


def test_diarize_job_full_mode_without_second_speaker_is_conservative(cfg, tmp_path, monkeypatch):
    segments = [_seg(0.0, 3.0, speaker="其他1")]
    job_out = _write_job(tmp_path, segments, {"id": "job1", "ok": True})
    wav = _tiny_wav(tmp_path)
    fake_full = {
        "ok": True, "n_speakers": 1, "main_cluster": 0,
        "speaker_by_cluster": {0: "主讲"},
        "segments": [{"start": 0.0, "end": 3.0, "speaker": 0}],
        "clusters": [{"cluster": 0, "dur_s": 600.0, "ratio": 1.0, "speaker": "主讲"}],
        "raw_speakers": 6, "merged_speakers": 1, "main_ratio": 1.0,
        "identity_min_s": 5.0, "timings": {},
    }
    monkeypatch.setattr(D, "ensure_clean16k", lambda *a, **k: wav)
    monkeypatch.setattr(D, "full_diarize", lambda *a, **k: fake_full)
    full_cfg = cfg.replace({"diar.mode": "full"})

    result = D.diarize_job("job1", full_cfg)

    assert result["ok"] is True and result["mode"] == "full"
    saved = _read(job_out / "transcript.json")
    assert all(s["speaker"] == "主讲" for s in saved)
    saved_report = _read(job_out / "report.json")
    assert saved_report["speakers"]["method"] == "full"
    assert saved_report["speakers"]["n"] == 1


def test_diarize_job_sampled_missing_model_fails_without_touching_transcript(cfg, tmp_path):
    segments = [_seg(0.0, 3.0)]
    job_out = _write_job(tmp_path, segments, {"id": "job1", "ok": True})
    sampled_cfg = cfg.replace({
        "diar.mode": "sampled",
        "paths.models.diar_dir": str(tmp_path / "no_such_models"),
    })
    result = D.diarize_job("job1", sampled_cfg)
    assert result["ok"] is False
    assert "模型" in result["error"]
    saved = _read(job_out / "transcript.json")
    assert saved[0]["speaker"] == "主讲"


# ---------------------------------------------------------------- clean16k 缓存
def test_ensure_clean16k_reuses_existing_cache(cfg, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    wav = cache / f"job1{D.CLEAN16K_SUFFIX}"
    from scipy.io import wavfile

    wavfile.write(str(wav), D.SR, np.zeros(D.SR, dtype=np.float32))
    out_root = tmp_path / "out"
    (out_root / "job1").mkdir(parents=True)
    path = D.ensure_clean16k("job1", cfg, out_dir=out_root, cache_dir=cache)
    assert path == wav


def test_ensure_clean16k_missing_clean_m4a_raises(cfg, tmp_path):
    cache = tmp_path / "cache"
    out_root = tmp_path / "out"
    (out_root / "job1").mkdir(parents=True)
    with pytest.raises(D.DiarError):
        D.ensure_clean16k("job1", cfg, out_dir=out_root, cache_dir=cache)


# ---------------------------------------------------------------- report 读写
def test_update_report_preserves_other_sections(tmp_path):
    path = tmp_path / "report.json"
    report = {"id": "x", "ok": True, "decode": {"ok": True},
              "timing": {"decode_s": 1.0, "asr_s": 2.0}}
    path.write_text(json.dumps(report), encoding="utf-8")
    D._update_report(path, report, {"speakers": {"n": 1, "method": "none"},
                                    "timing": {"diarize_s": 0.5}})
    saved = _read(path)
    assert saved["ok"] is True
    assert saved["decode"] == {"ok": True}
    assert saved["timing"] == {"decode_s": 1.0, "asr_s": 2.0, "diarize_s": 0.5}
    assert saved["speakers"] == {"n": 1, "method": "none"}


# ---------------------------------------------------------------- 声学引擎不可用
def test_diar_available_false_for_missing_models(cfg, tmp_path):
    bad = cfg.replace({"paths.models.diar_dir": str(tmp_path / "missing")})
    assert D.diar_available(bad) is False


def test_diarize_windows_empty_windows():
    result = D.diarize_windows(np.zeros(1600, dtype=np.float32), None, windows=[])
    assert result["ok"] is False
    assert result["segments"] == []
    assert result["merged_speakers"] == 0
