# -*- coding: utf-8 -*-
"""P0-T1 配置与日志单测。

覆盖验收点：`python -m app.cli config show` 打印的是「合并后的配置（含默认值）」
——即 preset/环境变量/命令行覆盖确实生效，且锁定参数值没有被误改。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import (
    PRESETS,
    PROJECT_ROOT,
    Config,
    ConfigError,
    apply_env,
    compute_config_hash,
    deep_merge,
    load_config,
    read_yaml,
    validate_config,
)
from app.logging_setup import get_logger, setup_logging, shutdown_logging


# ---------------------------------------------------------------- 默认值（锁定决策）
def test_locked_defaults():
    """§2 锁定参数必须与计划一致，防止后人「顺手改默认值」。"""
    cfg = load_config()
    assert cfg.preset == "standard"
    assert cfg.get("levels.target_asl_dbfs") == -20.0
    assert cfg.get("loudness.target_asl_dbfs") == -20.0
    assert cfg.get("levels.cross_file_tolerance_db") == 0.5
    assert cfg.get("levels.window_half_spread_tolerance_db") == 1.5
    # V2 慢变：窗 20 s / hop 5 s / clamp ±3 dB / 速率 ≤1.5 dB/s / 20 ms 交叉淡化
    assert cfg.get("loudness.v2.enabled") is True
    assert cfg.get("loudness.v2.win_s") == 20.0
    assert cfg.get("loudness.v2.hop_s") == 5.0
    assert cfg.get("loudness.v2.clamp_db") == 3.0
    assert cfg.get("loudness.v2.high_drift_clamp_db") == 6.0
    assert cfg.get("loudness.v2.max_rate_db_s") == 1.5
    assert cfg.get("loudness.v2.crossfade_ms") == 20
    # limiter 必须按真峰值
    assert cfg.get("loudness.limiter.target_tp_dbtp") == -1.0
    assert cfg.get("loudness.limiter.oversample") == 4
    assert cfg.get("loudness.limiter.ceiling_dbfs") <= -2.5
    # 头尾边界：120 s 窗口、字符率 ≥2.0、连续 2 窗口、软窗口 1.0（P2 实测细化）
    assert cfg.get("trim.boundary_window_s") == 120.0
    assert cfg.get("trim.boundary_min_chars_per_sec") == 2.0
    assert cfg.get("trim.boundary_consecutive_windows") == 2
    assert cfg.get("trim.boundary_soft_chars_per_sec") == 1.0
    assert cfg.get("trim.boundary_pad_s") == 0.5
    # 分析段（P2）
    assert cfg.get("analyze.cache_enabled") is True
    assert cfg.get("analyze.boundary_use_asr") is True
    assert cfg.get("levels.noise_floor_percentile") == 10.0
    assert cfg.get("levels.window_min_speech_s") == 30.0
    assert cfg.get("levels.percentiles") == [10, 50, 90]
    # 内部长静音 ≥3 s，保护带 0.25 s，碎片保留
    assert cfg.get("trim.internal_silence_min_s") == 3.0
    assert cfg.get("trim.internal_guard_s") == 0.25
    assert cfg.get("trim.keep_fragments") is True
    # 转写：16k float32、VAD 段 0.5–30 s、8 线程
    assert cfg.get("decode.analysis_sr") == 16000
    assert cfg.get("decode.render_sr") == 44100
    assert cfg.get("decode.sample_fmt") == "pcm_f32le"
    assert cfg.get("asr.segment_min_s") == 0.5
    assert cfg.get("asr.segment_max_s") == 30.0
    assert cfg.get("asr.num_threads") == 8
    # 纠错
    assert cfg.get("correct.model") == "deepseek-flash"
    assert cfg.get("correct.chunk_min_chars") == 2000
    assert cfg.get("correct.chunk_max_chars") == 2500
    assert cfg.get("correct.max_diff_ratio") == 0.12
    # 无关人声保守策略
    assert cfg.get("unrelated_speech.mode") == "conservative"
    assert cfg.get("unrelated_speech.max_chars_per_sec") == 0.5
    assert cfg.get("unrelated_speech.min_duration_s") == 5.0
    # 并发与磁盘水位
    assert cfg.get("batch.jobs") == 1
    assert cfg.get("batch.disk_free_min_gb") == 10.0


def test_paths_resolved_absolute(tmp_path: Path):
    cfg = load_config()
    for key in ("data_dir", "out_dir", "work_dir", "cache_dir", "db_path", "logs_dir"):
        path = cfg.path(key)
        assert path.is_absolute(), f"paths.{key} 未解析为绝对路径：{path}"
    assert cfg.path("data_dir") == PROJECT_ROOT / "data"
    assert cfg.path("cache_dir") == PROJECT_ROOT / "work" / "cache"
    # 模型路径由 ${paths.models_root} 占位符解析
    assert cfg.model_path("sensevoice_dir") == PROJECT_ROOT / "work" / "bench" / "models" / "sensevoice"
    assert cfg.model_path("rnnoise_model").name == "sh.rnnn"


def test_preset_override_applies():
    standard = load_config()
    fast = load_config(preset="fast")
    quality = load_config(preset="quality")
    # fast：轻降噪、关 V2、关纠错
    assert fast.preset == "fast"
    assert fast.get("denoise.afftdn.nr") == 8.0
    assert fast.get("loudness.v2.enabled") is False
    assert fast.get("correct.enabled") is False
    # quality：强降噪
    assert quality.get("denoise.afftdn.nr") == 16.0
    assert quality.get("correct.chunk_max_chars") == 2000
    # 未覆盖的键仍取默认值
    assert fast.get("levels.target_asl_dbfs") == standard.get("levels.target_asl_dbfs")
    assert fast.get("trim.boundary_window_s") == 120.0


def test_unknown_preset_rejected():
    with pytest.raises(ConfigError, match="未知 preset"):
        load_config(preset="turbo")
    assert "standard" in PRESETS


def test_env_overrides_api_key_and_model_dir(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    monkeypatch.setenv("MODEL_DIR", r"D:\models")
    cfg = load_config()
    assert cfg.get("correct.api_key") == "sk-test-123"
    assert cfg.model_path("sensevoice_dir") == Path(r"D:\models\sensevoice")
    # api_key 不参与 config_hash，也不出现在脱敏输出里
    assert "sk-test-123" not in json.dumps(cfg.to_dict(), ensure_ascii=False)
    assert "***" in cfg.to_dict()["correct"]["api_key"]


def test_env_ap_out_dir(tmp_path: Path):
    target = tmp_path / "custom_out"
    cfg = load_config(env={"AP_OUT_DIR": str(target)})
    assert cfg.path("out_dir") == target.resolve()


def test_cli_overrides_win(tmp_path: Path):
    cfg = load_config(overrides={"paths.cache_dir": str(tmp_path / "c")})
    assert cfg.path("cache_dir") == (tmp_path / "c").resolve()
    # replace 保留 preset、重新解析路径
    replaced = cfg.replace({"levels.target_asl_dbfs": -18.0})
    assert replaced.get("levels.target_asl_dbfs") == -18.0
    assert replaced.path("cache_dir") == cfg.path("cache_dir")


def test_config_hash_stability_and_sensitivity(tmp_path: Path):
    a = load_config()
    b = load_config()
    assert a.config_hash == b.config_hash
    # 与路径无关（换工作目录不应改变指纹）
    c = load_config(overrides={"paths.out_dir": str(tmp_path / "x")})
    assert c.config_hash == a.config_hash
    # 改参数则指纹变化
    d = a.replace({"loudness.v2.clamp_db": 6.0})
    assert d.config_hash != a.config_hash
    # api_key 不敏感
    e = load_config(env={"DEEPSEEK_API_KEY": "sk-abc"})
    assert e.config_hash == a.config_hash
    assert compute_config_hash(a.to_dict())  # 纯函数可独立调用


def test_deep_merge_and_read_yaml(tmp_path: Path):
    base = {"a": {"b": 1, "c": {"d": 2}}, "e": [1, 2]}
    merged = deep_merge(base, {"a": {"c": {"d": 3}}, "e": [9]})
    assert merged == {"a": {"b": 1, "c": {"d": 3}}, "e": [9]}
    # 原始 dict 不被修改
    assert base["a"]["c"]["d"] == 2

    path = tmp_path / "c.yaml"
    path.write_text("a:\n  b: 中文\n", encoding="utf-8")
    assert read_yaml(path) == {"a": {"b": "中文"}}
    with pytest.raises(ConfigError, match="不存在"):
        read_yaml(tmp_path / "nope.yaml")


def test_placeholder_unknown_rejected(tmp_path: Path):
    bad = tmp_path / "default.yaml"
    bad.write_text("paths:\n  x: ${paths.nope}/a\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="占位符"):
        load_config(config_path=bad)


def test_validate_rejects_illegal_values():
    cfg = load_config()
    data = cfg.to_dict(redact=False)
    data["levels"]["target_asl_dbfs"] = 5.0
    data["diar"]["mode"] = "sometimes"
    with pytest.raises(ConfigError) as info:
        validate_config(data)
    msg = str(info.value)
    assert "target_asl_dbfs" in msg and "diar.mode" in msg


def test_to_dict_yaml_is_utf8_and_contains_defaults():
    cfg = load_config()
    text = cfg.to_yaml()
    assert "目标语音电平" not in text  # 注释不参与合并输出
    data = cfg.to_dict()
    # 合并后必须包含各段默认值，而不是只包含被覆盖的部分
    for section in ("paths", "logging", "decode", "vad", "levels", "trim", "denoise",
                    "loudness", "asr", "diar", "correct", "unrelated_speech", "output", "batch"):
        assert section in data, f"合并配置缺少段：{section}"
    assert data["decode"]["analysis_sr"] == 16000
    # yaml 中中文不转义
    assert "\\u" not in text


def test_config_object_accessors():
    cfg = load_config()
    assert "levels.target_asl_dbfs" in cfg
    assert "levels.not_exists" not in cfg
    assert cfg.get("levels.not_exists", "fallback") == "fallback"
    with pytest.raises(ConfigError, match="缺少配置项"):
        cfg.require("levels.not_exists")
    with pytest.raises(ConfigError, match="缺少路径配置"):
        cfg.path("not_a_path")
    assert cfg.section("logging")["level"] == "INFO"
    assert repr(cfg).startswith("Config(")


# ---------------------------------------------------------------- 日志
def test_logging_writes_utf8_file_and_console(tmp_path: Path, capsys):
    log_file = tmp_path / "logs" / "run.log"
    shutdown_logging()
    logger = setup_logging(log_file=log_file, level="INFO", console=True, force=True)
    logger.info("中文日志测试 %s", "ok")
    for handler in logger.handlers:
        handler.flush()

    text = log_file.read_text(encoding="utf-8")
    assert "中文日志测试 ok" in text
    captured = capsys.readouterr()
    assert "中文日志测试 ok" in captured.out
    # 子 logger 通过 "app." 前缀继承
    get_logger("app.pipeline.decode").info("子模块日志")
    for handler in logger.handlers:
        handler.flush()
    assert "子模块日志" in log_file.read_text(encoding="utf-8")
    shutdown_logging()


def _our_handlers(logger) -> list:
    """pytest 的 logging 插件会往非 propagate 的 logger 上挂自己的 handler，这里只数我们的。"""
    return [h for h in logger.handlers if getattr(h, "_audio_processer_handler", False)]


def test_logging_idempotent_no_duplicate(tmp_path: Path):
    log_file = tmp_path / "run.log"
    shutdown_logging()
    setup_logging(log_file=log_file, level="INFO", console=False, force=True)
    setup_logging(log_file=log_file, level="INFO", console=False)  # 同配置：不重复挂 handler
    logger = get_logger("app")
    assert len(_our_handlers(logger)) == 1
    logger.info("只应出现一次")
    for handler in _our_handlers(logger):
        handler.flush()
    assert log_file.read_text(encoding="utf-8").count("只应出现一次") == 1
    # 换级别会重建（旧的被移除，不会叠加）
    setup_logging(log_file=log_file, level="DEBUG", console=False)
    assert len(_our_handlers(logger)) == 1
    assert _our_handlers(logger)[0].level == 10
    # force=True 也保持单份
    setup_logging(log_file=log_file, level="DEBUG", console=False, force=True)
    assert len(_our_handlers(logger)) == 1
    shutdown_logging()


def test_logging_from_config(cfg, tmp_path: Path):
    shutdown_logging()
    logger = setup_logging(cfg)
    logger.info("按配置落盘")
    expected = cfg.path("logs_dir") / cfg.get("logging.file_name")
    for handler in logger.handlers:
        handler.flush()
    assert expected.is_file()
    assert "按配置落盘" in expected.read_text(encoding="utf-8")
    shutdown_logging()
