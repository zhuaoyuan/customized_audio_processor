# -*- coding: utf-8 -*-
"""P3-T1 降噪单测：滤镜链纯函数 + ffmpeg 封装 + 错误路径 + 44.1k 渲染轴契约。

分两层：
- **纯函数用例**（默认跑）：``build_filter_chain`` 各 mode、arnndn 的 cwd/相对路径/48k
  往返、``resolve_nr`` 的低 SNR 升 nr、参数校验；
- **ffmpeg 用例**（``requires_ffmpeg``）：afftdn/off 真跑、产物采样率与时长校验、超时；
- **denoise_44k 契约用例**：用 monkeypatch 替换 ``denoise_wav`` 捕获 nr 决策（不依赖真实
  rnnoise 模型），另有一条真跑 afftdn 验证产物 44.1k/mono/float32。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from app.config import Config
from app.pipeline import denoise as denoise_mod
from app.pipeline.decode import read_wav_float32, wav_info
from app.pipeline.denoise import (
    CACHE_SUFFIX_DENOISED44K,
    DenoiseError,
    build_filter_chain,
    denoise_44k,
    denoise_wav,
    resolve_nf,
    resolve_nr,
)

from conftest import requires_ffmpeg

DEFAULT_AFFTDN = "highpass=f=80,afftdn=nr=12:nf=-30:tn=1"


# ---------------------------------------------------------------- 造测试音频
def _write_wav(
    path: Path,
    seconds: float = 2.0,
    sr: int = 44100,
    freq: float = 440.0,
    amp: float = 0.3,
    noise: float = 0.01,
) -> Path:
    """正弦 + 白噪的 float32 单声道 wav（降噪输入的最小样例）。"""
    rng = np.random.default_rng(0)
    t = np.arange(int(round(sr * seconds))) / float(sr)
    x = amp * np.sin(2 * np.pi * freq * t) + noise * rng.normal(size=t.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, x.astype(np.float32))
    return path


# ---------------------------------------------------------------- 滤镜链（纯函数）
def test_build_filter_chain_default_afftdn(cfg):
    af, cwd = build_filter_chain(cfg)
    assert af == DEFAULT_AFFTDN
    assert cwd is None


def test_build_filter_chain_accepts_plain_defaults_without_cfg():
    # cfg=None 也要给出计划 §2 锁定的默认链（缺键全部走 cfg.get 默认值）
    af, cwd = build_filter_chain(None)
    assert af == DEFAULT_AFFTDN
    assert cwd is None


def test_build_filter_chain_nr_override(cfg):
    af, _ = build_filter_chain(cfg, nr=16)
    assert af == "highpass=f=80,afftdn=nr=16:nf=-30:tn=1"
    af, _ = build_filter_chain(cfg, nr=12.5)
    assert "nr=12.5" in af


def test_build_filter_chain_nf_override(cfg):
    # 噪声底显式覆盖（默认走配置 −30；这里只改 nf，nr 仍取配置 12）
    af, _ = build_filter_chain(cfg, nf=-25)
    assert af == "highpass=f=80,afftdn=nr=12:nf=-25:tn=1"
    af, _ = build_filter_chain(cfg, nr=16, nf=-22)
    assert af == "highpass=f=80,afftdn=nr=16:nf=-22:tn=1"


def test_build_filter_chain_highpass_mode(cfg):
    af, cwd = build_filter_chain(cfg, mode="highpass")
    assert af == "highpass=f=80"
    assert cwd is None


def test_build_filter_chain_off_mode(cfg):
    af, cwd = build_filter_chain(cfg, mode="off")
    assert af == ""
    assert cwd is None


def test_build_filter_chain_custom_params(cfg):
    custom = cfg.replace({
        "denoise.highpass_hz": 100,
        "denoise.afftdn.nf": -25.0,
        "denoise.afftdn.tn": 0,
        "denoise.afftdn.nr": 9.5,
    })
    af, _ = build_filter_chain(custom)
    assert af == "highpass=f=100,afftdn=nr=9.5:nf=-25:tn=0"


def test_build_filter_chain_arnndn_cwd_and_relative_model(cfg, tmp_path: Path):
    # 用假模型文件：只验证「cwd=模型目录 + m=相对文件名 + 48k 进出」，不跑 arnndn
    model = tmp_path / "models" / "rnnoise" / "sh.rnnn"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fake rnnoise model")
    custom = cfg.replace({"paths.models.rnnoise_model": str(model)})

    af, cwd = build_filter_chain(custom, mode="arnndn")
    assert af == (
        "highpass=f=80,aresample=48000,arnndn=m=sh.rnnn,aresample=44100"
    )
    assert cwd == model.parent
    assert ":" not in af.split("arnndn=m=")[1]  # 盘符冒号不能出现在 filter 里
    assert "\\" not in af.split("arnndn=m=")[1]


def test_build_filter_chain_arnndn_follows_src_sr(cfg, tmp_path: Path):
    model = tmp_path / "sh.rnnn"
    model.write_bytes(b"fake")
    custom = cfg.replace({
        "paths.models.rnnoise_model": str(model),
        "denoise.arnndn.resample_sr": 48000,
    })
    af, cwd = build_filter_chain(custom, mode="arnndn", src_sr=16000)
    assert af.endswith(",aresample=16000")
    assert af.startswith("highpass=f=80,aresample=48000,arnndn=m=sh.rnnn")
    assert cwd == tmp_path

    af48, _ = build_filter_chain(custom, mode="arnndn", src_sr=48000)
    assert af48.endswith(",aresample=48000")


def test_build_filter_chain_custom_resample_sr(cfg, tmp_path: Path):
    model = tmp_path / "sh.rnnn"
    model.write_bytes(b"fake")
    custom = cfg.replace({
        "paths.models.rnnoise_model": str(model),
        "denoise.arnndn.resample_sr": 16000,
    })
    af, _ = build_filter_chain(custom, mode="arnndn", src_sr=44100)
    assert af == "highpass=f=80,aresample=16000,arnndn=m=sh.rnnn,aresample=44100"


def test_build_filter_chain_invalid_mode(cfg):
    with pytest.raises(DenoiseError, match="未知降噪模式"):
        build_filter_chain(cfg, mode="banana")
    bad = cfg.replace({"denoise.mode": "noise-gate"})
    with pytest.raises(DenoiseError, match="未知降噪模式"):
        build_filter_chain(bad)


def test_build_filter_chain_invalid_params(cfg):
    for nr in (0, -1, 98, "abc", True):
        with pytest.raises(DenoiseError):
            build_filter_chain(cfg, nr=nr)
    for key, value in (
        ("denoise.highpass_hz", 0),
        ("denoise.highpass_hz", 30000),
        ("denoise.afftdn.nf", -100.0),
        ("denoise.afftdn.nf", -10.0),
        ("denoise.afftdn.tn", 2),
    ):
        bad = cfg.replace({key: value})
        with pytest.raises(DenoiseError):
            build_filter_chain(bad)


def test_build_filter_chain_arnndn_without_model_path():
    # 最小 Config：没有 paths.models.rnnoise_model → 明确报错而不是拼出坏滤镜
    minimal = Config({"denoise": {"mode": "arnndn"}})
    with pytest.raises(DenoiseError, match="rnnoise_model"):
        build_filter_chain(minimal)


# ---------------------------------------------------------------- nr 决策（低 SNR）
def test_resolve_nr_default_and_explicit(cfg):
    assert resolve_nr(cfg) == 12.0
    assert resolve_nr(cfg, mode="afftdn") == 12.0
    assert resolve_nr(cfg, mode="afftdn", nr=16) == 16.0
    assert resolve_nr(cfg, mode="afftdn", nr=20) == 20.0
    # 显式 nr 优先级最高：即使 SNR 很低也不被 low_snr_nr 覆盖
    assert resolve_nr(cfg, mode="afftdn", snr_db=3.0, nr=18) == 18.0


def test_resolve_nr_low_snr_upgrades(cfg):
    # f20 的 SNR 25.6 → 不升；低 SNR（<12）→ 16
    assert resolve_nr(cfg, mode="afftdn", snr_db=25.6) == 12.0
    assert resolve_nr(cfg, mode="afftdn", snr_db=12.0) == 12.0  # 等于阈值不升
    assert resolve_nr(cfg, mode="afftdn", snr_db=11.99) == 16.0
    assert resolve_nr(cfg, mode="afftdn", snr_db=-5.0) == 16.0


def test_resolve_nr_custom_thresholds_and_missing_keys(cfg):
    custom = cfg.replace({"denoise.low_snr_db": 30.0, "denoise.low_snr_nr": 18.0})
    assert resolve_nr(custom, mode="afftdn", snr_db=20.0) == 18.0
    # 缺 low_snr_* / afftdn.nr 键 → 用代码默认（12 / 16），不依赖 default.yaml
    minimal = Config({"denoise": {"mode": "afftdn"}})
    assert resolve_nr(minimal, mode="afftdn", snr_db=30.0) == 12.0
    assert resolve_nr(minimal, mode="afftdn", snr_db=5.0) == 16.0


def test_resolve_nr_non_afftdn_modes(cfg):
    assert resolve_nr(cfg, mode="highpass", snr_db=1.0) is None
    assert resolve_nr(cfg, mode="off", nr=20) is None
    assert resolve_nr(cfg, mode="arnndn", snr_db=1.0) is None


def test_resolve_nf_default_and_override(cfg):
    # 默认 = 配置锁定值 −30（不改 configs/default.yaml 就能回归）
    assert resolve_nf(cfg) == -30.0
    assert resolve_nf(cfg, mode="afftdn") == -30.0
    # 显式覆盖优先（给「用户同意后逐文件调噪声底」留的口子）
    assert resolve_nf(cfg, mode="afftdn", nf=-25) == -25.0
    assert resolve_nf(cfg, mode="afftdn", nf=-22.5) == -22.5
    custom = cfg.replace({"denoise.afftdn.nf": -28.0})
    assert resolve_nf(custom, mode="afftdn") == -28.0
    assert resolve_nf(custom, mode="afftdn", nf=-25) == -25.0
    # 非 afftdn 模式 nf 无意义
    assert resolve_nf(cfg, mode="highpass") is None
    assert resolve_nf(cfg, mode="off", nf=-25) is None
    # 缺键 → 代码默认 −30
    assert resolve_nf(Config({"denoise": {"mode": "afftdn"}}), mode="afftdn") == -30.0
    with pytest.raises(DenoiseError):
        resolve_nf(cfg, mode="afftdn", nf=-10)


# ---------------------------------------------------------------- denoise_wav
def test_denoise_wav_missing_input(cfg, tmp_path: Path):
    with pytest.raises(DenoiseError, match="不存在"):
        denoise_wav(tmp_path / "nope.wav", tmp_path / "out.wav", cfg)


def test_denoise_wav_rejects_bad_wav(cfg, tmp_path: Path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a real wav file")
    with pytest.raises(DenoiseError, match="WAV"):
        denoise_wav(bad, tmp_path / "out.wav", cfg)


def test_denoise_wav_unknown_mode(cfg, tmp_path: Path):
    src = _write_wav(tmp_path / "in.wav", seconds=0.5)
    with pytest.raises(DenoiseError, match="未知降噪模式"):
        denoise_wav(src, tmp_path / "out.wav", cfg, mode="banana")


def test_denoise_wav_arnndn_missing_model(cfg, tmp_path: Path):
    src = _write_wav(tmp_path / "in.wav", seconds=0.5)
    custom = cfg.replace({"paths.models.rnnoise_model": str(tmp_path / "missing" / "sh.rnnn")})
    with pytest.raises(DenoiseError, match="模型不存在"):
        denoise_wav(src, tmp_path / "out.wav", custom, mode="arnndn")


@requires_ffmpeg
def test_denoise_wav_afftdn_roundtrip(cfg, tmp_path: Path):
    src = _write_wav(tmp_path / "in 中文 源.wav", seconds=2.0)
    dst = tmp_path / "out 降噪 结果.wav"

    result = denoise_wav(src, dst, cfg)

    assert result["ok"] is True
    assert result["filter"] == DEFAULT_AFFTDN
    assert result["mode"] == "afftdn"
    assert result["nr"] == 12.0
    assert result["nf"] == -30.0
    assert result["src_sr"] == 44100 and result["dst_sr"] == 44100
    assert result["elapsed_s"] > 0 and result["elapsed_s"] < 60
    assert Path(result["dst"]) == dst.resolve() and dst.is_file()

    info = wav_info(dst)
    assert info.sample_rate == 44100
    assert info.channels == 1
    assert info.sampwidth == 4  # pcm_f32le
    assert info.duration_s == pytest.approx(2.0, abs=0.05)


@requires_ffmpeg
def test_denoise_wav_afftdn_nf_override(cfg, tmp_path: Path):
    src = _write_wav(tmp_path / "in.wav", seconds=1.0)
    dst = tmp_path / "nf25.wav"
    result = denoise_wav(src, dst, cfg, nf=-25)
    assert result["nf"] == -25.0
    assert result["nr"] == 12.0
    assert ":nf=-25:" in result["filter"]
    assert dst.is_file()
    assert wav_info(dst).sample_rate == 44100


@requires_ffmpeg
def test_denoise_wav_off_mode_is_passthrough(cfg, tmp_path: Path):
    src = _write_wav(tmp_path / "in.wav", seconds=1.0)
    dst = tmp_path / "off.wav"
    result = denoise_wav(src, dst, cfg, mode="off")
    assert result["filter"] == ""
    assert result["nr"] is None
    assert result["nf"] is None
    sr_in, x_in = read_wav_float32(src)
    sr_out, x_out = read_wav_float32(dst)
    assert sr_out == sr_in
    assert len(x_out) == len(x_in)
    assert np.allclose(x_in, x_out, atol=1e-6)


@requires_ffmpeg
def test_denoise_wav_timeout_raises(cfg, tmp_path: Path):
    src = _write_wav(tmp_path / "in.wav", seconds=1.0)
    with pytest.raises(DenoiseError, match="超时"):
        denoise_wav(src, tmp_path / "out.wav", cfg, timeout_s=1e-3)


# ---------------------------------------------------------------- denoise_44k
def _fake_denoise_wav(recorder: Dict[str, Any], seconds: float = 0.5):
    """替换 denoise_wav：记录参数并写出合法的 44.1k/mono/float32 产物。"""

    def _fake(src, dst, cfg, *, mode=None, nr=None, nf=None, timeout_s=None):
        recorder.update(src=str(src), dst=str(dst), mode=mode, nr=nr, nf=nf)
        _write_wav(Path(dst), seconds=seconds, sr=44100)
        return {"ok": True, "mode": mode, "nr": nr, "nf": nf, "dst": str(dst)}

    return _fake


@pytest.mark.parametrize(
    "snr_db, expected_nr",
    [
        (25.6, 12.0),   # 正常 SNR：默认 nr=12
        (12.0, 12.0),   # 等于阈值不升
        (8.0, 16.0),    # 低 SNR：升到 16（f20 类文件）
        (-3.0, 16.0),
    ],
)
def test_denoise_44k_low_snr_policy(cfg, tmp_path: Path, monkeypatch, snr_db, expected_nr):
    src = _write_wav(tmp_path / "src44k.wav", seconds=0.5)
    recorder: Dict[str, Any] = {}
    monkeypatch.setattr(denoise_mod, "denoise_wav", _fake_denoise_wav(recorder))

    dst = denoise_44k(src, cfg, "joblow", snr_db=snr_db)

    assert recorder["mode"] == "afftdn"
    assert recorder["nr"] == pytest.approx(expected_nr)
    assert dst == cfg.path("cache_dir") / f"joblow{CACHE_SUFFIX_DENOISED44K}"
    assert dst.is_file()


def test_denoise_44k_explicit_nr_wins(cfg, tmp_path: Path, monkeypatch):
    src = _write_wav(tmp_path / "src44k.wav", seconds=0.5)
    recorder: Dict[str, Any] = {}
    monkeypatch.setattr(denoise_mod, "denoise_wav", _fake_denoise_wav(recorder))

    denoise_44k(src, cfg, "jobf20", snr_db=3.0, nr=16)
    assert recorder["nr"] == 16.0
    denoise_44k(src, cfg, "jobf20b", snr_db=30.0, nr=20)
    assert recorder["nr"] == 20.0


def test_denoise_44k_off_mode_no_nr(cfg, tmp_path: Path, monkeypatch):
    src = _write_wav(tmp_path / "src44k.wav", seconds=0.5)
    recorder: Dict[str, Any] = {}
    monkeypatch.setattr(denoise_mod, "denoise_wav", _fake_denoise_wav(recorder))
    custom = cfg.replace({"denoise.mode": "off"})

    denoise_44k(src, custom, "joboff", snr_db=1.0)
    assert recorder["mode"] == "off"
    assert recorder["nr"] is None
    assert recorder["nf"] is None


def test_denoise_44k_nf_passthrough(cfg, tmp_path: Path, monkeypatch):
    # 默认 nf=None（走配置）；显式 nf 原样传给 denoise_wav，供「用户同意后」逐文件调 nf
    src = _write_wav(tmp_path / "src44k.wav", seconds=0.5)
    recorder: Dict[str, Any] = {}
    monkeypatch.setattr(denoise_mod, "denoise_wav", _fake_denoise_wav(recorder))

    denoise_44k(src, cfg, "jobnf_default", snr_db=25.0)
    assert recorder["nf"] is None
    denoise_44k(src, cfg, "jobnf25", snr_db=25.0, nf=-25)
    assert recorder["nf"] == -25.0


def test_denoise_44k_rejects_non_44k_input(cfg, tmp_path: Path):
    src = _write_wav(tmp_path / "src16k.wav", seconds=0.5, sr=16000)
    with pytest.raises(DenoiseError, match="渲染轴"):
        denoise_44k(src, cfg, "job16k")


def test_denoise_44k_rejects_missing_input_and_empty_job_id(cfg, tmp_path: Path):
    with pytest.raises(DenoiseError, match="不存在"):
        denoise_44k(tmp_path / "nope.wav", cfg, "jobx")
    src = _write_wav(tmp_path / "src44k.wav", seconds=0.5)
    with pytest.raises(DenoiseError, match="job_id"):
        denoise_44k(src, cfg, "")


@requires_ffmpeg
def test_denoise_44k_contract_real_afftdn(cfg, tmp_path: Path):
    src = _write_wav(tmp_path / "src44k.wav", seconds=1.0)
    dst = denoise_44k(src, cfg, "jobcontract", snr_db=25.0)

    assert dst == cfg.path("cache_dir") / f"jobcontract{CACHE_SUFFIX_DENOISED44K}"
    assert dst.is_file()
    info = wav_info(dst)
    assert info.sample_rate == 44100
    assert info.channels == 1
    assert info.sampwidth == 4
    assert info.duration_s == pytest.approx(1.0, abs=0.05)
