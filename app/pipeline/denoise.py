# -*- coding: utf-8 -*-
"""降噪阶段（P3-T1）：ffmpeg 滤镜链封装（highpass / afftdn / arnndn）+ GTCRN 神经网络降噪。

契约（供 trim / loudness / 批量管线调用）
----------------------------------------
- ``build_filter_chain(cfg, *, mode=None, nr=None, nf=None, src_sr=None) -> (af, cwd)``：
  **只拼滤镜字符串**、不碰文件，可纯函数单测；``cwd`` 非空时调用方必须把 ffmpeg
  的工作目录切过去（只有 arnndn 需要，见下）。``nr`` / ``nf`` 是显式覆盖（默认走配置值）。
- ``denoise_wav(src, dst, cfg, *, mode=None, nr=None, nf=None, timeout_s=None) -> dict``：
  调 ffmpeg 产出 float32 WAV，**采样率跟随输入**；返回滤镜链/模式/nr/nf/输入输出采样率/
  耗时等，产物再做存在性与时长校验（与输入差 <50 ms）。
- ``denoise_44k(decoded44k, cfg, job_id, *, snr_db=None, nr=None, nf=None) -> Path``：
  渲染轴入口，产出 ``<paths.cache_dir>/<job_id>.denoised44k.wav``，保持 44.1 kHz/mono/float32。
- ``resolve_mode(cfg, snr_db=None) -> str``：决定**生效**降噪模式（含 ``auto`` 规则），供
  export 分支调用；``denoise_44k`` 本身行为不变（只跑滤镜链）。
- ``gtcrn_denoise_wav16k(src, dst, cfg, *, segments=None) -> dict``：16 kHz 输入 →
  GTCRN + VAD 引导增益下限 → float32 16 kHz WAV（见下「GTCRN 降噪」）。
- ``gain_floor_curve(n_frames, hop_s, frame_ms, segments, ...) -> np.ndarray``：纯函数，
  生成逐 STFT 帧的增益下限曲线（dB），单测直接覆盖。
- ``denoise_gtcrn_44k(wav16k, cfg, job_id, *, segments=None) -> Path``：渲染轴入口，先写
  ``<job_id>.gtcrn16k.wav`` 再 aresample 到 ``<job_id>.denoised44k.wav``，**输出契约与
  ``denoise_44k`` 完全一致**（44.1 kHz / mono / float32 / 时长差 <50 ms）。

为什么默认 afftdn（计划 §2 锁定）
-------------------------------
``highpass=f=80,afftdn=nr=12:nf=-30:tn=1`` @44.1 kHz 是 PoC 里「语音基本不动、底噪明显降」的
折中：PoC ``work/audio-edit/denoise_compare.json`` 实测 afftdn 平均语音电平变化仅 −0.23 dB、
RTF≈0.0024（10 min 音频约 1.5 s，满足 P3-T1「≤5 s/10 min」）；arnndn 降噪更狠（噪声 −12 dB）
但语音也被拉低 −2.2 dB 量级，会与 P3-T3 的响度归一化相互纠缠，故只作可选模式。

arnndn 的两个硬约束（本机实测 + 计划 §7 坑 #1）
---------------------------------------------
1. **rnnoise 只支持 48 kHz**：16k 直接送入会得到 NaN/静音（实测见
   ``work/audio-edit/arnndn_rate_test.json``）。所以链路必须 ``aresample=48000`` 进、
   再 ``aresample=<源采样率>`` 出——本模块按**输入采样率**写最后一级，保证
   ``denoise_wav`` 的产物与输入同采样率（渲染轴 44.1k 不会被改成 48k）。
2. **ffmpeg filtergraph 用 ``:`` 分隔参数**，Windows 绝对路径的盘符冒号会被当成参数分隔符
   （``m=C:/...`` 直接解析失败）。因此 ``m=`` 只写模型目录下的**相对文件名**，并把
   subprocess 的 ``cwd`` 切到模型目录（``build_filter_chain`` 返回的第二个值）。

低 SNR 升 nr 的策略（P3-T1 + P2 日志「f20 伴随杂音」）
----------------------------------------------------
- 默认全库 ``nr=12``（计划 §2 锁定），不为单个文件写死逻辑；
- 仅当 ``mode=afftdn`` 且调用方传入的 ``snr_db`` 低于 ``denoise.low_snr_db``（默认 12 dB）时
  升到 ``denoise.low_snr_nr``（默认 16 dB）——数据驱动，见 :func:`resolve_nr`；
- 需要逐文件覆盖时，给 :func:`denoise_44k` 传显式 ``nr``（显式值优先级最高）。

GTCRN 降噪（16 kHz 神经网络 + VAD 引导增益下限）
-----------------------------------------------
实测（3047 s 真文件，SNR 10.8 dB）：现用 afftdn 链把停顿噪声只压低 ~10.6 dB 且 nr 已饱和
（16/20/24/30 结果一致），100–300 Hz 隆隆声几乎不动；GTCRN（``sherpa_onnx``
``OfflineSpeechDenoiser``，只支持 16 kHz，RTF≈0.05）能把噪声压低 ~22 dB、听感接近无噪，
但会把轻音字「门」掉，听感「人声不连续」。因此 GTCRN 路径不是直接出模型结果，而是：

1. 分块跑 GTCRN（``denoise.gtcrn.chunk_s``，默认 600 s，块间 0.5 s 重叠线性交叉淡化）——
   50 min 文件也不会一次把模型输入塞爆；
2. **整段**把 GTCRN 输出按中位帧电平对齐回输入（帧 25 ms / 帧移 10 ms）；
3. STFT（n_fft=512 / hop=128 / Hann / boundary="even"）里用 **GTCRN 幅度 + 输入相位**，
   逐 bin 增益 ``clip(|Y|/|X|, 10^(floor(t)/20), 2.0)``，再做 3 bin 频率平滑与
   ``denoise.gtcrn.smooth_ms``（默认 60 ms）时间平滑；
4. ``floor(t)`` 由 VAD 段生成（:func:`gain_floor_curve`）：语音段 ``floor_speech_db``
   （默认 −8 dB）、停顿 ``floor_pause_db``（默认 −24 dB），边界用滑动平均过渡。

离线验证（``tmp/ana2/gtcrn_floor_vad.py``，同一文件）：停顿噪声 p50 保持纯 GTCRN 的
−55 dBFS，语音帧 1st-percentile 增益从 −17.7 dB 改善到 −11.2 dB，被压制 >10 dB 的语音帧
占比从 4.6 % 降到 0.8 %。STFT 也按 ``denoise.gtcrn.stft_chunk_s``（默认 120 s，0.5 s
重叠交叉淡化）分块，保证 50 min 文件峰值内存 < ~1.5 GB。

``denoise.gtcrn.*`` 配置键均**可选**（缺键走代码默认值，便于单测/灰度）：
``chunk_s`` / ``overlap_s`` / ``stft_chunk_s`` / ``stft_overlap_s`` / ``floor_speech_db`` /
``floor_pause_db`` / ``smooth_ms`` / ``num_threads``；模型路径优先
``paths.models.gtcrn_model``，否则 ``paths.models_root/gtcrn/gtcrn_simple.onnx``。
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import ConfigError
from ..logging_setup import get_logger
from .decode import DecodeError, RENDER_SR, read_wav_float32, wav_info

log = get_logger(__name__)

#: 降噪产物的缓存后缀（与 decode 的 ``.44k.wav`` 区分）
CACHE_SUFFIX_DENOISED44K = ".denoised44k.wav"
#: GTCRN 16 kHz 产物缓存后缀（先落 16k，再 aresample 成 44.1k）
CACHE_SUFFIX_GTCRN16K = ".gtcrn16k.wav"
#: 统一 float32（与 decode 一致，禁止 int16）
PCM_F32 = "pcm_f32le"
#: 输出与输入时长允许偏差（秒）；超过即视为 ffmpeg 异常
DURATION_TOLERANCE_S = 0.05
#: 合法降噪模式（与 configs/default.yaml 的 denoise.mode 对齐；``auto`` 由 resolve_mode 解析）
VALID_MODES = ("afftdn", "arnndn", "highpass", "off", "gtcrn")

#: GTCRN（sherpa-onnx）输入的固定采样率（模型只支持 16 kHz）
GTCRN_SR = 16000
#: 模型缺省相对路径（相对 ``paths.models_root``）
GTCRN_MODEL_REL = Path("gtcrn") / "gtcrn_simple.onnx"
#: STFT 参数（与离线验证脚本 tmp/ana2/gtcrn_floor_vad.py 保持一致）
GTCRN_NFFT = 512
GTCRN_HOP = 128
#: 频率方向平滑窗（bin 数）：3 bin 只抹掉孤立尖峰，不糊掉共振峰
GTCRN_FREQ_SMOOTH_BINS = 3
#: 单 bin 增益上限：GTCRN 幅度最多允许 2 倍（+6 dB）补偿，避免相位重建爆音
GTCRN_MAX_GAIN = 2.0


class DenoiseError(RuntimeError):
    """降噪失败（输入缺失/不是 WAV/ffmpeg 失败/产物校验不通过）。"""


# ---------------------------------------------------------------- 参数校验
def _cfg_get(cfg: Any, dotted: str, default: Any) -> Any:
    """``cfg.get`` 的安全版本（允许 cfg=None，便于纯函数测试）。"""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if getter is None:
        return default
    try:
        return getter(dotted, default)
    except Exception:  # pragma: no cover - 只防御异常配置对象
        return default


def _as_float(value: Any, name: str) -> float:
    """转 float；bool 视为非法（避免 True 被当成 1.0）。"""
    if isinstance(value, bool):
        raise DenoiseError(f"{name} 必须是数值，当前={value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DenoiseError(f"{name} 必须是数值，当前={value!r}") from exc


def _as_int(value: Any, name: str) -> int:
    number = _as_float(value, name)
    if number != int(number):
        raise DenoiseError(f"{name} 必须是整数，当前={value!r}")
    return int(number)


def _fmt_num(value: Any) -> str:
    """数值 → 滤镜参数文本：12.0→"12"、-30.0→"-30"、12.5→"12.5"。"""
    number = _as_float(value, "滤镜参数")
    if number == int(number):
        return str(int(number))
    return f"{number:g}"


def _mode_value(cfg: Any, mode: Optional[str]) -> str:
    """确定降噪模式（显式参数 > 配置 > 默认 afftdn）并校验。

    ``denoise.mode=auto`` 在这里**保守落成 afftdn**：auto 的正式判定在
    :func:`resolve_mode`（要看 SNR，export 分支用）；而 ``denoise_wav`` /
    ``denoise_44k`` 这类只看配置的入口遇到 auto 时不能去加载神经网络模型，
    回落 afftdn 既保持既有行为，也保证 ``build_filter_chain(cfg)`` 仍可单测。
    """
    if mode is None:
        value = str(_cfg_get(cfg, "denoise.mode", "afftdn"))
    else:
        value = str(mode)
    value = value.strip().lower()
    if value == "auto":
        return "afftdn"
    if value not in VALID_MODES:
        raise DenoiseError(
            f"未知降噪模式：{value!r}（可选：{', '.join(VALID_MODES)}）"
        )
    return value


def _as_nr(value: Any, name: str = "denoise.afftdn.nr") -> float:
    """afftdn 的 nr：ffmpeg 允许 (0, 97]。"""
    number = _as_float(value, name)
    if not (0.0 < number <= 97.0):
        raise DenoiseError(f"{name} 超出 afftdn 允许范围 (0, 97]：{number}")
    return number


def _as_noise_floor(value: Any, name: str = "denoise.afftdn.nf") -> float:
    """afftdn 的 nf：ffmpeg 允许 [−80, −20] dBFS。"""
    number = _as_float(value, name)
    if not (-80.0 <= number <= -20.0):
        raise DenoiseError(f"{name} 超出 afftdn 允许范围 [-80, -20]：{number}")
    return number


def _as_track_noise(value: Any, name: str = "denoise.afftdn.tn") -> int:
    """afftdn 的 tn：0/1 开关。"""
    number = _as_int(value, name)
    if number not in (0, 1):
        raise DenoiseError(f"{name} 必须是 0 或 1，当前={value!r}")
    return number


def _as_highpass(value: Any, name: str = "denoise.highpass_hz") -> float:
    number = _as_float(value, name)
    if not (0.0 < number <= 20000.0):
        raise DenoiseError(f"{name} 超出 highpass 允许范围 (0, 20000]：{number}")
    return number


def _decode_render_sr(cfg: Any) -> int:
    """渲染轴采样率（默认 44100）。"""
    sr = _as_int(_cfg_get(cfg, "decode.render_sr", RENDER_SR), "decode.render_sr")
    if sr <= 0:
        raise DenoiseError(f"decode.render_sr 非法：{sr}")
    return sr


def _rnnoise_model_path(cfg: Any) -> Path:
    """``paths.models.rnnoise_model`` 的绝对路径（arnndn 模式必需）。"""
    if cfg is None:
        raise DenoiseError("arnndn 模式需要 cfg（paths.models.rnnoise_model）")
    try:
        return Path(cfg.model_path("rnnoise_model"))
    except ConfigError as exc:
        raise DenoiseError(
            f"arnndn 缺少模型路径配置 paths.models.rnnoise_model：{exc}"
        ) from exc
    except AttributeError as exc:  # pragma: no cover - 传入非 Config 对象
        raise DenoiseError(f"cfg 不支持 model_path()：{exc}") from exc


# ---------------------------------------------------------------- 滤镜链
def resolve_nr(
    cfg: Any,
    *,
    mode: Optional[str] = None,
    snr_db: Optional[float] = None,
    nr: Optional[float] = None,
) -> Optional[float]:
    """决定 afftdn 实际使用的 ``nr``（非 afftdn 返回 ``None``）。

    优先级：**显式 nr** > **低 SNR 提升** > 配置默认（``denoise.afftdn.nr``）。

    - 显式 ``nr`` 永远优先：逐文件覆盖（如 f20 想用 nr=16）不能被 SNR 规则覆盖；
    - ``snr_db < denoise.low_snr_db``（默认 12 dB）时用 ``denoise.low_snr_nr``（默认 16）；
    - 缺键一律走默认值，不在代码里写死具体文件。
    """
    mode_value = _mode_value(cfg, mode)
    if mode_value != "afftdn":
        return None
    if nr is not None:
        return _as_nr(nr)
    base = _as_nr(_cfg_get(cfg, "denoise.afftdn.nr", 12.0))
    if snr_db is None:
        return base
    snr_value = _as_float(snr_db, "snr_db")
    low_snr_threshold = _as_float(_cfg_get(cfg, "denoise.low_snr_db", 12.0), "denoise.low_snr_db")
    if snr_value < low_snr_threshold:
        return _as_nr(_cfg_get(cfg, "denoise.low_snr_nr", 16.0))
    return base


def resolve_nf(
    cfg: Any,
    *,
    mode: Optional[str] = None,
    nf: Optional[float] = None,
) -> Optional[float]:
    """决定 afftdn 实际使用的 ``nf``（噪声底 dBFS；非 afftdn 返回 ``None``）。

    优先级：**显式 nf** > 配置 ``denoise.afftdn.nf``（默认 −30）。
    显式参数留给「用户同意逐文件调噪声底」的场景（如 f24 教学区 nf=−25 实测底噪
    p10 多降 2.1 dB），默认路径仍严格走配置锁定值 −30。
    """
    mode_value = _mode_value(cfg, mode)
    if mode_value != "afftdn":
        return None
    if nf is not None:
        return _as_noise_floor(nf)
    return _as_noise_floor(_cfg_get(cfg, "denoise.afftdn.nf", -30.0))


def resolve_mode(cfg: Any, snr_db: Optional[float] = None) -> str:
    """决定**生效**降噪模式（供 export 分支；不改 ``denoise_44k`` 的既有行为）。

    规则
    ----
    - ``denoise.mode`` 是 ``afftdn`` / ``arnndn`` / ``highpass`` / ``off`` / ``gtcrn`` 时
      **原样返回**（显式配置永远优先，便于逐文件覆盖/灰度）；
    - ``denoise.mode == "auto"``：``snr_db`` 非空且低于
      ``denoise.auto_gtcrn_below_snr_db``（默认 12.0 dB）时返回 ``"gtcrn"``，否则 ``"afftdn"``；
      ``snr_db`` 为空（没跑 P2 电平统计）时保守回落到 ``"afftdn"``；
    - 其他值抛 :class:`DenoiseError`（与 :func:`_mode_value` 行为一致，不做静默兜底）。

    为什么阈值默认 12 dB：实测文件 SNR 10.8 dB 落在阈值下（该文件正是 GTCRN 显胜的场景），
    而 SNR 正常的文件继续走 afftdn（RTF≈0.002，比 GTCRN 便宜两个数量级）。
    """
    value = str(_cfg_get(cfg, "denoise.mode", "afftdn")).strip().lower()
    if value == "auto":
        if snr_db is None:
            return "afftdn"
        threshold = _as_float(
            _cfg_get(cfg, "denoise.auto_gtcrn_below_snr_db", 12.0),
            "denoise.auto_gtcrn_below_snr_db",
        )
        return "gtcrn" if _as_float(snr_db, "snr_db") < threshold else "afftdn"
    if value not in VALID_MODES:
        raise DenoiseError(
            f"未知降噪模式：{value!r}（可选：{', '.join(VALID_MODES)}；"
            f"或用 auto 按 SNR 自动选择）"
        )
    return value


def build_filter_chain(
    cfg: Any,
    *,
    mode: Optional[str] = None,
    nr: Optional[float] = None,
    nf: Optional[float] = None,
    src_sr: Optional[int] = None,
) -> Tuple[str, Optional[Path]]:
    """拼 ffmpeg 滤镜链，返回 ``(af, cwd)``；``af`` 为空串表示不挂滤镜（mode=off）。

    - afftdn：``highpass=f=<hz>,afftdn=nr=<nr>:nf=<nf>:tn=<tn>``
    - highpass：``highpass=f=<hz>``
    - arnndn：``highpass=f=<hz>,aresample=<resample_sr>,arnndn=m=<相对模型名>,aresample=<src_sr>``，
      返回的 ``cwd`` = 模型目录（规避 Windows 盘符冒号坑），测试只断言字符串、不需要真实模型。
    - off：``("", None)``
    - gtcrn：**不是滤镜链**，抛 :class:`DenoiseError` 指向
      :func:`denoise_gtcrn_44k` / :func:`gtcrn_denoise_wav16k`（避免拼出沉默的坏链路）。

    ``nr`` / ``nf`` 为显式覆盖（默认 ``None`` = 用配置值，不在代码里写死具体文件）；
    ``src_sr`` 未给时按 ``decode.render_sr``（默认 44100）；arnndn 的最后一级 aresample
    必须回到输入采样率，否则 ``denoise_wav`` 的产物采样率会变成 48k。
    """
    mode_value = _mode_value(cfg, mode)
    highpass_hz = _as_highpass(_cfg_get(cfg, "denoise.highpass_hz", 80.0))
    if mode_value == "off":
        return "", None
    if mode_value == "gtcrn":
        raise DenoiseError(
            "gtcrn 不是 ffmpeg 滤镜链：请调用 denoise_gtcrn_44k / gtcrn_denoise_wav16k"
            "（export 用 resolve_mode 分支）"
        )
    if mode_value == "highpass":
        return f"highpass=f={_fmt_num(highpass_hz)}", None
    if mode_value == "afftdn":
        nr_value = _as_nr(nr) if nr is not None else _as_nr(
            _cfg_get(cfg, "denoise.afftdn.nr", 12.0)
        )
        noise_floor = resolve_nf(cfg, mode="afftdn", nf=nf)
        track_noise = _as_track_noise(_cfg_get(cfg, "denoise.afftdn.tn", 1))
        chain = (
            f"highpass=f={_fmt_num(highpass_hz)},"
            f"afftdn=nr={_fmt_num(nr_value)}:nf={_fmt_num(noise_floor)}:tn={_fmt_num(track_noise)}"
        )
        return chain, None

    # ---- arnndn：48k 进出 + cwd 到模型目录用相对文件名
    model_path = _rnnoise_model_path(cfg)
    configured_name = _cfg_get(cfg, "denoise.arnndn.model", model_path.name)
    model_name = Path(str(configured_name)).name  # 只保留文件名，路径由 cwd 承担
    if not model_name:
        raise DenoiseError(f"arnndn 模型文件名非法：{configured_name!r}")
    resample_sr = _as_int(_cfg_get(cfg, "denoise.arnndn.resample_sr", 48000), "denoise.arnndn.resample_sr")
    if resample_sr <= 0:
        raise DenoiseError(f"denoise.arnndn.resample_sr 非法：{resample_sr}")
    out_sr = _as_int(src_sr if src_sr is not None else _decode_render_sr(cfg), "src_sr")
    if out_sr <= 0:
        raise DenoiseError(f"输入采样率非法：{out_sr}")
    chain = (
        f"highpass=f={_fmt_num(highpass_hz)},"
        f"aresample={resample_sr},"
        f"arnndn=m={model_name},"
        f"aresample={out_sr}"
    )
    return chain, model_path.parent


# ---------------------------------------------------------------- ffmpeg 封装
def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise DenoiseError("PATH 中找不到 ffmpeg，请先安装并加入 PATH")


def _read_wav_info(path: Path, what: str) -> Any:
    try:
        return wav_info(path)
    except DecodeError as exc:
        raise DenoiseError(f"{what}不是可解析的 WAV：{path}（{exc}）") from exc


def _default_timeout(cfg: Any, duration_s: float) -> float:
    """超时默认值：配置优先；否则按「1.5×时长 + 30 s」且不少于 120 s。"""
    configured = _cfg_get(cfg, "denoise.timeout_s", None)
    if configured is not None:
        return max(1.0, _as_float(configured, "denoise.timeout_s"))
    return max(120.0, float(duration_s) * 1.5 + 30.0)


def denoise_wav(
    src: str | Path,
    dst: str | Path,
    cfg: Any,
    *,
    mode: Optional[str] = None,
    nr: Optional[float] = None,
    nf: Optional[float] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """对 WAV 跑一遍降噪，输出 pcm_f32le WAV（采样率与声道数跟随输入）。

    返回 ``{ok, filter, mode, nr, nf, src_sr, dst_sr, elapsed_s, dst, ...}``。
    ``nr`` / ``nf`` 为显式覆盖（默认 ``None`` = 用配置值；``nf`` 只在 afftdn 模式生效）。
    失败（输入缺失/不是 WAV/ffmpeg 非 0/产物缺失/采样率变化/时长差 ≥50 ms）抛
    :class:`DenoiseError`——不做「静默返回半成品」。

    注意：``src``/``dst`` 会先 ``resolve()`` 成绝对路径，因为 arnndn 模式会把
    subprocess 的 cwd 切到模型目录，相对路径会写到错误的地方。
    """
    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.is_file():
        raise DenoiseError(f"降噪输入不存在：{src_p}（先跑 decode.ensure_44k）")
    _require_ffmpeg()

    info = _read_wav_info(src_p, "降噪输入")
    src_sr = int(info.sample_rate)
    if src_sr <= 0:
        raise DenoiseError(f"降噪输入采样率非法：{src_sr}（{src_p}）")
    duration_s = float(info.duration_s)

    af, cwd = build_filter_chain(cfg, mode=mode, nr=nr, nf=nf, src_sr=src_sr)
    if cwd is not None:  # arnndn：模型必须在 cwd 下真实存在，否则 ffmpeg 报错难读
        model_path = _rnnoise_model_path(cfg)
        if not model_path.is_file():
            raise DenoiseError(
                f"arnndn 模型不存在：{model_path}（cwd={cwd}，m={model_path.name}）"
            )

    resolved_src = src_p.resolve()
    resolved_dst = dst_p.resolve()
    resolved_dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-i", str(resolved_src),
    ]
    if af:  # mode=off 不挂 -af，只做 float32 重封装
        cmd += ["-af", af]
    cmd += ["-c:a", PCM_F32, "-f", "wav", str(resolved_dst)]

    timeout = (
        _default_timeout(cfg, duration_s)
        if timeout_s is None
        else max(1e-3, _as_float(timeout_s, "timeout_s"))
    )
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise DenoiseError(
            f"ffmpeg 降噪超时（>{timeout:.1f}s）：{resolved_src}（mode={_mode_value(cfg, mode)}）"
        ) from exc
    elapsed_s = time.perf_counter() - started

    if proc.returncode != 0 or not resolved_dst.is_file():
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise DenoiseError(
            f"ffmpeg 降噪失败（exit={proc.returncode}，mode={_mode_value(cfg, mode)}）："
            f"{resolved_src}\nfilter: {af or '(off)'}\n{stderr[-1500:] or '(无 stderr)'}"
        )

    out_info = _read_wav_info(resolved_dst, "降噪产物")
    dst_sr = int(out_info.sample_rate)
    if dst_sr != src_sr:
        raise DenoiseError(
            f"降噪产物采样率 {dst_sr} ≠ 输入 {src_sr}：{resolved_dst}"
            f"（arnndn 的最后一级 aresample 应回到输入采样率）"
        )
    duration_diff = abs(float(out_info.duration_s) - duration_s)
    if duration_diff >= DURATION_TOLERANCE_S:
        raise DenoiseError(
            f"降噪产物时长与输入差 {duration_diff * 1000.0:.1f} ms"
            f"（要求 <{DURATION_TOLERANCE_S * 1000.0:.0f} ms）：{resolved_dst}"
        )

    effective_nr = resolve_nr(cfg, mode=mode, nr=nr) if af else None
    effective_nf = resolve_nf(cfg, mode=mode, nf=nf) if af else None
    result: Dict[str, Any] = {
        "ok": True,
        "filter": af,
        "mode": _mode_value(cfg, mode),
        "nr": effective_nr,
        "nf": effective_nf,
        "src_sr": src_sr,
        "dst_sr": dst_sr,
        "elapsed_s": round(elapsed_s, 3),
        "rtf": round(elapsed_s / duration_s, 4) if duration_s > 0 else None,
        "duration_s": round(duration_s, 3),
        "dst_duration_s": round(float(out_info.duration_s), 3),
        "src": str(resolved_src),
        "dst": str(resolved_dst),
        "size_bytes": resolved_dst.stat().st_size,
        "cwd": str(cwd) if cwd is not None else "",
    }
    log.info(
        "降噪完成：%s mode=%s nr=%s nf=%s %.1fs（RTF=%.4f，filter=%s）→ %s",
        resolved_src.name, result["mode"], result["nr"], result["nf"], elapsed_s,
        result["rtf"] or 0.0, af or "(off)", resolved_dst.name,
    )
    return result


# ---------------------------------------------------------------- GTCRN：纯函数曲线
def gain_floor_curve(
    n_frames: int,
    hop_s: float,
    frame_ms: float,
    segments: Optional[Sequence[Sequence[float]]],
    *,
    floor_speech_db: float = -8.0,
    floor_pause_db: float = -24.0,
    smooth_ms: float = 60.0,
) -> np.ndarray:
    """逐 STFT 帧的增益下限曲线（dB，长度 ``n_frames``）；**纯函数**，不做文件 IO。

    参数
    ----
    n_frames : STFT 帧数（``scipy.signal.stft(boundary="even")`` 对应 ``1 + n_samples//hop``）
    hop_s    : STFT 帧移（秒）
    frame_ms : STFT 窗长（毫秒）；只用来把帧中心放在 ``i*hop + frame_ms/2`` 处
    segments : VAD 语音段 ``[[start_s, end_s], ...]``（与输入同一时间轴，可空/None）
    floor_speech_db : 语音帧下限（默认 −8 dB：保住轻音字）
    floor_pause_db  : 停顿帧下限（默认 −24 dB：让停顿噪声接近纯 GTCRN）
    smooth_ms       : 时间方向滑动平均窗，给段边界一个线性过渡（默认 60 ms）

    实现（保持简单、确定性）：帧中心落在某段内 → 语音下限，否则停顿下限；再对曲线做
    ``smooth_ms`` 的滑动平均（边界按 edge 复制，避免首尾被拉向 0 dB 而失去下限作用）。
    ``smooth_ms <= 0`` 时返回未平滑的硬阶梯，便于单测对比「平滑确实减小了跳变」。
    """
    n = int(n_frames)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    hop = _as_float(hop_s, "hop_s")
    if hop <= 0:
        raise DenoiseError(f"STFT 帧移必须 > 0：{hop_s}")
    frame_s = _as_float(frame_ms, "frame_ms") / 1000.0
    if frame_s < 0:
        raise DenoiseError(f"帧长不能为负：{frame_ms}")
    speech_db = _as_float(floor_speech_db, "floor_speech_db")
    pause_db = _as_float(floor_pause_db, "floor_pause_db")
    if speech_db > 0.0:
        raise DenoiseError(f"语音帧增益下限不能 > 0 dB（不允许放大）：{speech_db}")
    if pause_db > speech_db:
        raise DenoiseError(
            f"停顿帧下限 {pause_db} dB 不能高于语音帧下限 {speech_db} dB"
            "（停顿必须压得更狠）"
        )

    centers = np.arange(n, dtype=np.float64) * hop + 0.5 * frame_s
    curve = np.full(n, pause_db, dtype=np.float64)
    for seg in segments or ():
        try:
            start, end = float(seg[0]), float(seg[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise DenoiseError(f"VAD 段非法（需要 [start, end]）：{seg!r}") from exc
        if not (np.isfinite(start) and np.isfinite(end)):
            raise DenoiseError(f"VAD 段含非有限值：{seg!r}")
        if end <= start:
            continue
        curve[(centers >= start) & (centers < end)] = speech_db

    smooth = _as_float(smooth_ms, "smooth_ms")
    if smooth < 0:
        raise DenoiseError(f"smooth_ms 不能为负：{smooth_ms}")
    k = max(1, int(round(smooth / 1000.0 / hop)))
    if k > 1:
        left = k // 2
        padded = np.pad(curve, (left, k - 1 - left), mode="edge")
        curve = np.convolve(padded, np.ones(k, dtype=np.float64) / k, mode="valid")
    return curve


# ---------------------------------------------------------------- GTCRN：内部工具
def _frame_db(
    x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0
) -> np.ndarray:
    """逐帧 dBFS（默认 25 ms 帧 / 10 ms 帧移）；分块计算避免长音频生成巨临时数组。"""
    n = int(round(sr * frame_ms / 1000.0))
    h = int(round(sr * hop_ms / 1000.0))
    if n <= 0 or h <= 0:
        raise DenoiseError(f"帧长/帧移非法：frame_ms={frame_ms} hop_ms={hop_ms}")
    x = np.asarray(x, dtype=np.float32)
    if x.size < n:
        power = float(np.mean(np.square(x, dtype=np.float64))) if x.size else 0.0
        return np.array([10.0 * np.log10(max(power, 1e-20))])
    n_frames = 1 + (x.size - n) // h
    windows = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    powers = np.empty(n_frames, dtype=np.float64)
    block = 4096  # 一批约 13 MB 临时数组（4096×400 float64）
    for start in range(0, n_frames, block):
        part = windows[start:start + block]
        powers[start:start + block] = np.mean(np.square(part, dtype=np.float64), axis=1)
    return 10.0 * np.log10(np.maximum(powers, 1e-20))


def _level_match_gain_db(x: np.ndarray, y: np.ndarray, sr: int) -> float:
    """GTCRN 输出对齐输入所需增益：整段**中位帧电平**之差（dB）。

    只给 GTCRN 输出用（不是逐帧动态增益），因此不会把停顿噪声抬回来——对齐后的
    语音电平与输入一致，停顿仍由后续增益下限控制。
    """
    ref_db = _frame_db(x, sr)
    out_db = _frame_db(y, sr)
    n = min(ref_db.size, out_db.size)
    if n == 0:
        return 0.0
    return float(np.median(ref_db[:n]) - np.median(out_db[:n]))


def _chunk_spans(
    n_samples: int,
    chunk_s: float,
    overlap_s: float,
    sr: int,
    *,
    align: int = 1,
) -> List[Tuple[int, int]]:
    """把 ``[0, n_samples)`` 切成带重叠的分块，返回 ``[(start, end), ...]``（样本下标）。

    ``align``：块长/重叠/起点都对齐到该样本数的整数倍。STFT 分块必须对齐到 hop，
    这样第 k 块的帧下标 = ``start // hop + 块内帧号``，可直接切片整段下限曲线。
    """
    n = int(n_samples)
    if n <= 0:
        return []
    sr = int(sr)
    chunk = max(1, int(round(_as_float(chunk_s, "chunk_s") * sr)))
    overlap = max(0, int(round(_as_float(overlap_s, "overlap_s") * sr)))
    if align > 1:
        chunk = max(int(align), (chunk // int(align)) * int(align))
        overlap = (overlap // int(align)) * int(align)
    if overlap >= chunk:  # 重叠不能吃掉整块，否则 stride=0 死循环
        overlap = max(0, chunk // 2)
    stride = chunk - overlap
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + chunk)
        spans.append((start, end))
        if end >= n:
            break
        start += stride
    return spans


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """把一维信号裁/补（尾部补零）到恰好 ``n`` 个采样，float32。"""
    x = np.asarray(x, dtype=np.float32)
    n = int(n)
    if x.size >= n:
        return np.ascontiguousarray(x[:n], dtype=np.float32)
    return np.pad(x, (0, n - x.size)).astype(np.float32)


def _stitch_fade(
    out: np.ndarray, part: np.ndarray, start: int, end: int, prev_end: int
) -> None:
    """分块结果按线性交叉淡化写进整段缓冲：重叠区用 0→1 斜坡（增益之和恒为 1）。"""
    m = int(end) - int(start)
    if m <= 0:
        return
    part = _fit_length(part, m)
    if prev_end <= start:
        out[start:end] = part
        return
    overlap = min(int(prev_end) - int(start), m)
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    out[start:start + overlap] = (
        out[start:start + overlap] * (1.0 - ramp) + part[:overlap] * ramp
    )
    out[start + overlap:end] = part[overlap:]


def _moving_average(arr: np.ndarray, k: int, axis: int) -> np.ndarray:
    """沿 ``axis`` 做 k 点滑动平均；边界按 edge 复制，避免频带两端被 0 稀释。"""
    k = int(k)
    if k <= 1 or arr.size == 0:
        return arr
    width = arr.shape[axis]
    if width <= 0:
        return arr
    left = k // 2
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (left, k - 1 - left)
    padded = np.pad(arr, pad, mode="edge")
    acc = np.zeros(arr.shape, dtype=np.float64)
    for offset in range(k):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(offset, offset + width)
        acc += padded[tuple(sl)]
    return acc / float(k)


def _rolling_sum_1d(mask: np.ndarray, k: int) -> np.ndarray:
    """一维滚动窗口计数（边界 edge 复制），供形态学膨胀/腐蚀使用。"""
    mask = np.asarray(mask)
    k = int(k)
    if k <= 1:
        return mask.astype(np.int64)
    left = k // 2
    padded = np.pad(mask.astype(np.int64), (left, k - 1 - left), mode="edge")
    cumsum = np.cumsum(padded)
    return cumsum[k - 1:] - np.concatenate(([0], cumsum[:-k]))


def _normalize_segments(
    segments: Optional[Sequence[Sequence[float]]],
) -> List[Tuple[float, float]]:
    """把外部 VAD 段规整成 ``(start, end)`` 浮点对；丢弃空段、非法段直接报错。"""
    out: List[Tuple[float, float]] = []
    for seg in segments or ():
        try:
            start, end = float(seg[0]), float(seg[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise DenoiseError(f"VAD 段非法（需要 [start, end]）：{seg!r}") from exc
        if not (np.isfinite(start) and np.isfinite(end)):
            raise DenoiseError(f"VAD 段含非有限值：{seg!r}")
        if end <= start:
            continue
        out.append((start, end))
    return out


def _energy_vad_segments(
    x: np.ndarray,
    sr: int = GTCRN_SR,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    threshold_db: float = 6.0,
    smooth_ms: float = 100.0,
) -> List[Tuple[float, float]]:
    """兜底「能量 VAD」（不依赖 Silero，调用方没给 analyze 段时用）。

    规则：25 ms 帧 / 10 ms 帧移，帧功率高于「中位帧电平 + 6 dB」记为语音；再做 100 ms
    形态学平滑（闭 → 开）补小洞、去毛刺；最后把连续 True 帧并成 ``[start, end]`` 段。
    只求「别把整段当停顿」的粗粒度判断，不追求精确边界（正式路径始终用 analyze 的 VAD）。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return []
    hop_s = hop_ms / 1000.0
    db = _frame_db(x, sr, frame_ms, hop_ms)
    mask = db > (float(np.median(db)) + float(threshold_db))
    k = max(1, int(round(_as_float(smooth_ms, "smooth_ms") / hop_ms)))
    if k > 1:
        mask = _rolling_sum_1d(mask, k) > 0      # 膨胀
        mask = _rolling_sum_1d(mask, k) >= k     # 腐蚀 → 闭运算（补小洞）
        mask = _rolling_sum_1d(mask, k) >= k     # 再腐蚀
        mask = _rolling_sum_1d(mask, k) > 0      # 再膨胀 → 开运算（去孤立毛刺）
    # 连续 True 段：索引差分（[0, mask, 0] 的 +1/-1 边界）
    edges = np.flatnonzero(
        np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    )
    duration_s = x.size / float(sr)
    frame_s = frame_ms / 1000.0
    segments: List[Tuple[float, float]] = []
    for i in range(0, edges.size, 2):
        start_s = float(edges[i]) * hop_s
        end_s = float(edges[i + 1] - 1) * hop_s + frame_s
        start_s = max(0.0, min(start_s, duration_s))
        end_s = max(start_s, min(end_s, duration_s))
        segments.append((round(start_s, 3), round(end_s, 3)))
    return segments


def _gtcrn_model_path(cfg: Any) -> Path:
    """GTCRN 模型路径：优先 ``paths.models.gtcrn_model``，否则 ``models_root/gtcrn/gtcrn_simple.onnx``。"""
    if cfg is None:
        raise DenoiseError(
            "gtcrn 模式需要 cfg（paths.models.gtcrn_model 或 paths.models_root）"
        )
    try:
        return Path(cfg.model_path("gtcrn_model"))
    except ConfigError:
        pass
    except AttributeError:  # pragma: no cover - 传入非 Config 对象时走下面的 path()
        pass
    try:
        models_root = Path(cfg.path("models_root"))
    except (ConfigError, AttributeError) as exc:
        raise DenoiseError(
            f"gtcrn 缺少模型路径配置（paths.models.gtcrn_model / paths.models_root）：{exc}"
        ) from exc
    return models_root / GTCRN_MODEL_REL


#: 进程内 GTCRN denoiser 缓存：key=(模型绝对路径, num_threads)；加载一次几秒钟，别每块重建
_GTCRN_DENOISERS: Dict[Tuple[str, int], Any] = {}
_GTCRN_LOCK = threading.Lock()


def _get_gtcrn_denoiser(model_path: Path, num_threads: int) -> Any:
    """按模型路径缓存 ``sherpa_onnx.OfflineSpeechDenoiser``（懒加载 sherpa_onnx）。"""
    key = (str(Path(model_path).resolve()), int(num_threads))
    with _GTCRN_LOCK:
        cached = _GTCRN_DENOISERS.get(key)
        if cached is not None:
            return cached
        try:
            import sherpa_onnx
        except ModuleNotFoundError as exc:  # pragma: no cover - 环境缺依赖
            raise DenoiseError(
                "缺少 sherpa-onnx：请执行 "
                "work\\venv\\Scripts\\python -m pip install -r requirements.txt"
            ) from exc
        started = time.perf_counter()
        try:
            config = sherpa_onnx.OfflineSpeechDenoiserConfig(
                model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                    gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                        model=str(Path(model_path).resolve())
                    ),
                    num_threads=int(num_threads),
                )
            )
            denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
        except Exception as exc:  # onnxruntime 加载/opset 不匹配等
            raise DenoiseError(
                f"GTCRN 模型加载失败：{model_path}（{exc}）"
            ) from exc
        _GTCRN_DENOISERS[key] = denoiser
        log.info(
            "GTCRN 已加载：%s threads=%d（%.2fs）",
            Path(model_path).name, int(num_threads), time.perf_counter() - started,
        )
        return denoiser


def reset_gtcrn_cache() -> None:
    """释放缓存里的 GTCRN denoiser（测试/换模型时用）。"""
    with _GTCRN_LOCK:
        _GTCRN_DENOISERS.clear()


def _gtcrn_params(
    cfg: Any,
    *,
    floor_speech_db: Optional[float] = None,
    floor_pause_db: Optional[float] = None,
    smooth_ms: Optional[float] = None,
    chunk_s: Optional[float] = None,
    overlap_s: Optional[float] = None,
) -> Dict[str, Any]:
    """GTCRN 路径的全部参数（显式入参 > 配置 ``denoise.gtcrn.*`` > 代码默认值）。"""
    def pick(explicit: Optional[float], key: str, default: float, name: str) -> float:
        value = explicit if explicit is not None else _cfg_get(cfg, key, default)
        return _as_float(value, name)

    speech_db = pick(floor_speech_db, "denoise.gtcrn.floor_speech_db", -8.0,
                     "denoise.gtcrn.floor_speech_db")
    pause_db = pick(floor_pause_db, "denoise.gtcrn.floor_pause_db", -24.0,
                    "denoise.gtcrn.floor_pause_db")
    if speech_db > 0.0:
        raise DenoiseError(f"语音帧增益下限不能 > 0 dB（不允许放大）：{speech_db}")
    if pause_db > speech_db:
        raise DenoiseError(
            f"停顿帧下限 {pause_db} dB 不能高于语音帧下限 {speech_db} dB"
        )
    smooth_value = (
        _as_float(smooth_ms, "denoise.gtcrn.smooth_ms") if smooth_ms is not None
        else _as_float(
            # 兼容两种键名：``smooth_ms``（本模块约定）与 ``gain_smooth_ms``（默认配置用名）
            _cfg_get(
                cfg,
                "denoise.gtcrn.smooth_ms",
                _cfg_get(cfg, "denoise.gtcrn.gain_smooth_ms", 60.0),
            ),
            "denoise.gtcrn.smooth_ms",
        )
    )
    if smooth_value < 0:
        raise DenoiseError(f"denoise.gtcrn.smooth_ms 不能为负：{smooth_value}")
    chunk_value = pick(chunk_s, "denoise.gtcrn.chunk_s", 600.0, "denoise.gtcrn.chunk_s")
    if chunk_value <= 0:
        raise DenoiseError(f"denoise.gtcrn.chunk_s 必须 > 0：{chunk_value}")
    overlap_value = pick(overlap_s, "denoise.gtcrn.overlap_s", 0.5,
                         "denoise.gtcrn.overlap_s")
    if overlap_value < 0 or overlap_value >= chunk_value:
        raise DenoiseError(
            f"denoise.gtcrn.overlap_s={overlap_value} 必须落在 [0, chunk_s={chunk_value})"
        )
    stft_chunk = pick(None, "denoise.gtcrn.stft_chunk_s", 120.0,
                      "denoise.gtcrn.stft_chunk_s")
    if stft_chunk <= 0:
        raise DenoiseError(f"denoise.gtcrn.stft_chunk_s 必须 > 0：{stft_chunk}")
    stft_overlap = pick(None, "denoise.gtcrn.stft_overlap_s", 0.5,
                        "denoise.gtcrn.stft_overlap_s")
    if stft_overlap < 0 or stft_overlap >= stft_chunk:
        raise DenoiseError(
            f"denoise.gtcrn.stft_overlap_s={stft_overlap} 必须落在 [0, {stft_chunk})"
        )
    threads = _as_int(_cfg_get(cfg, "denoise.gtcrn.num_threads", 4),
                      "denoise.gtcrn.num_threads")
    if threads <= 0:
        raise DenoiseError(f"denoise.gtcrn.num_threads 必须 > 0：{threads}")
    return {
        "floor_speech_db": speech_db,
        "floor_pause_db": pause_db,
        "smooth_ms": smooth_value,
        "chunk_s": chunk_value,
        "overlap_s": overlap_value,
        "stft_chunk_s": stft_chunk,
        "stft_overlap_s": stft_overlap,
        "num_threads": threads,
    }


def _write_wav_float32(path: Path, sr: int, x: np.ndarray) -> None:
    """写 float32 单声道 WAV；失败时删掉半成品再抛 DenoiseError。"""
    import scipy.io.wavfile as wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(x, dtype=np.float32)
    try:
        wavfile.write(str(path), int(sr), array)
    except (OSError, ValueError) as exc:
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - 删除失败不影响报错
            pass
        raise DenoiseError(f"写 float32 WAV 失败：{path}（{exc}）") from exc


def _floor_stft_chunk(
    x: np.ndarray,
    y: np.ndarray,
    floor_curve_db: np.ndarray,
    frame_offset: int,
    *,
    sr: int,
    smooth_ms: float,
) -> np.ndarray:
    """单块 STFT 增益下限：GTCRN 幅度 + **输入相位**，增益 clip 到 ``[floor(t), 2.0]``。

    ``g = |Y| / |X|`` 先按 ``10^(floor(t)/20)`` 抬下限、按 2.0 封上限，再做 3 bin
    频率平滑与 ``smooth_ms`` 时间平滑（都是 ``np.convolve`` 滑动平均），最后
    ``S = g * |X| * e^{j·angle(X)}`` 重建——即用 GTCRN 的幅度、原始输入的相位。
    """
    from scipy.signal import istft, stft

    noverlap = GTCRN_NFFT - GTCRN_HOP
    _, _, spec_x = stft(
        np.asarray(x, dtype=np.float64), fs=float(sr), window="hann",
        nperseg=GTCRN_NFFT, noverlap=noverlap, boundary="even",
    )
    _, _, spec_y = stft(
        np.asarray(y, dtype=np.float64), fs=float(sr), window="hann",
        nperseg=GTCRN_NFFT, noverlap=noverlap, boundary="even",
    )
    n = min(spec_x.shape[1], spec_y.shape[1])
    spec_x, spec_y = spec_x[:, :n], spec_y[:, :n]
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    floor_db = np.asarray(
        floor_curve_db[int(frame_offset):int(frame_offset) + n], dtype=np.float64
    )
    if floor_db.size < n:  # 末块可能多 1 帧（曲线长度按整段算），用末值补齐
        tail = floor_db[-1] if floor_db.size else float(np.min(floor_curve_db))
        floor_db = np.concatenate([floor_db, np.full(n - floor_db.size, tail)])

    magnitude_x = np.abs(spec_x)
    gain = np.abs(spec_y) / np.maximum(magnitude_x, 1e-10)
    gain = np.maximum(gain, 10.0 ** (floor_db[None, :] / 20.0))
    gain = np.clip(gain, None, GTCRN_MAX_GAIN)
    gain = _moving_average(gain, GTCRN_FREQ_SMOOTH_BINS, 0)  # 频率 3 bin
    time_k = max(1, int(round(float(smooth_ms) / (GTCRN_HOP / float(sr) * 1000.0))))
    gain = _moving_average(gain, time_k, 1)                  # 时间 ~60 ms

    spec = gain * magnitude_x * np.exp(1j * np.angle(spec_x))  # 输入相位
    _, z = istft(
        spec, fs=float(sr), window="hann",
        nperseg=GTCRN_NFFT, noverlap=noverlap, boundary="even",
    )
    return _fit_length(np.asarray(z, dtype=np.float32), len(x))


def _apply_gain_floor(
    x: np.ndarray, y: np.ndarray, floor_curve_db: np.ndarray, *, params: Dict[str, Any]
) -> np.ndarray:
    """分块套增益下限（默认 120 s / 0.5 s 重叠交叉淡化），控制 50 min 文件的 STFT 内存。"""
    n = int(np.asarray(x).size)
    spans = _chunk_spans(
        n, params["stft_chunk_s"], params["stft_overlap_s"], GTCRN_SR, align=GTCRN_HOP
    )
    out = np.zeros(n, dtype=np.float32)
    prev_end = 0
    for start, end in spans:
        part = _floor_stft_chunk(
            x[start:end], y[start:end], floor_curve_db, start // GTCRN_HOP,
            sr=GTCRN_SR, smooth_ms=params["smooth_ms"],
        )
        _stitch_fade(out, part, start, end, prev_end)
        prev_end = end
    return out


def _run_gtcrn_chunks(
    denoiser: Any, x: np.ndarray, *, params: Dict[str, Any]
) -> np.ndarray:
    """分块跑 GTCRN（默认 600 s / 0.5 s 重叠），块间线性交叉淡化后拼回整段。"""
    n = int(np.asarray(x).size)
    spans = _chunk_spans(n, params["chunk_s"], params["overlap_s"], GTCRN_SR)
    out = np.zeros(n, dtype=np.float32)
    prev_end = 0
    for index, (start, end) in enumerate(spans):
        chunk = np.ascontiguousarray(x[start:end], dtype=np.float32)
        try:
            denoised = denoiser.run(chunk, sample_rate=GTCRN_SR)
        except Exception as exc:  # onnxruntime 推理异常
            raise DenoiseError(
                f"GTCRN 推理失败（第 {index + 1}/{len(spans)} 块 {start}-{end}）：{exc}"
            ) from exc
        samples = getattr(denoised, "samples", None)
        out_sr = int(getattr(denoised, "sample_rate", GTCRN_SR) or GTCRN_SR)
        if samples is None:
            raise DenoiseError("GTCRN 返回结果缺少 samples 字段")
        if out_sr != GTCRN_SR:
            raise DenoiseError(f"GTCRN 输出采样率 {out_sr} ≠ {GTCRN_SR}")
        part = _fit_length(np.asarray(samples, dtype=np.float32), end - start)
        _stitch_fade(out, part, start, end, prev_end)
        prev_end = end
    return out


# ---------------------------------------------------------------- GTCRN：主入口
def gtcrn_denoise_wav16k(
    src: str | Path,
    dst: str | Path,
    cfg: Any,
    *,
    segments: Optional[Sequence[Sequence[float]]] = None,
    floor_speech_db: Optional[float] = None,
    floor_pause_db: Optional[float] = None,
    smooth_ms: Optional[float] = None,
    chunk_s: Optional[float] = None,
    overlap_s: Optional[float] = None,
    level_match: bool = True,
) -> Dict[str, Any]:
    """16 kHz/mono/float32 WAV → GTCRN + VAD 引导增益下限 → float32 16 kHz WAV。

    处理顺序（每一步都可解释，见模块 docstring「GTCRN 降噪」）：
    GTCRN 分块推理 → 整段中位帧电平对齐 → STFT 幅度下限（输入相位）→ 写 WAV。

    参数
    ----
    segments       : analyze 的 VAD 段 ``[[start_s, end_s], ...]``；``None`` 时退化成
                     :func:`_energy_vad_segments`（能量 VAD，仅兜底）；显式 ``[]`` 表示
                     「整段都按停顿处理」（下限 −24 dB），便于测试/纯噪声场景
    floor_speech_db / floor_pause_db / smooth_ms / chunk_s / overlap_s :
                     显式覆盖 ``denoise.gtcrn.*``（默认走配置或代码默认值）
    level_match    : 是否做中位帧电平对齐（默认 True，生产路径）；``False`` 便于单测
                     直接观察 GTCRN+下限的原始降噪量（对齐会把纯噪声信号整体抬回输入电平）

    返回 ``{ok, mode:"gtcrn", model, src_sr, dst_sr, duration_s, elapsed_s, rtf,
    floor_speech_db, floor_pause_db, smooth_ms, dst, ...}``；输入缺失/采样率不是 16 kHz/
    非单声道/模型缺失/sherpa 缺依赖/产物校验失败一律抛 :class:`DenoiseError`。
    """
    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.is_file():
        raise DenoiseError(f"GTCRN 降噪输入不存在：{src_p}（先跑 decode）")
    info = _read_wav_info(src_p, "GTCRN 降噪输入")
    if int(info.sample_rate) != GTCRN_SR:
        raise DenoiseError(
            f"GTCRN 只支持 {GTCRN_SR} Hz 输入，当前 {int(info.sample_rate)} Hz：{src_p}"
            "（先 aresample 到 16k）"
        )
    if int(info.channels) != 1:
        raise DenoiseError(
            f"GTCRN 只支持单声道输入，当前 {int(info.channels)} 声道：{src_p}"
        )
    duration_s = float(info.duration_s)
    sr, samples = read_wav_float32(src_p)
    if int(sr) != GTCRN_SR:  # 双保险：头信息与 scipy 读出的采样率必须一致
        raise DenoiseError(f"GTCRN 输入采样率不一致：wav 头 {GTCRN_SR}，实际 {sr}：{src_p}")
    x = np.ascontiguousarray(samples, dtype=np.float32)
    if x.size == 0:
        raise DenoiseError(f"GTCRN 降噪输入为空（0 采样）：{src_p}")

    model_path = _gtcrn_model_path(cfg)
    if not model_path.is_file():
        raise DenoiseError(
            f"GTCRN 模型不存在：{model_path}"
            "（把 gtcrn_simple.onnx 放到 paths.models_root/gtcrn/，或配 paths.models.gtcrn_model）"
        )
    params = _gtcrn_params(
        cfg,
        floor_speech_db=floor_speech_db,
        floor_pause_db=floor_pause_db,
        smooth_ms=smooth_ms,
        chunk_s=chunk_s,
        overlap_s=overlap_s,
    )
    resolved_src = src_p.resolve()
    resolved_dst = dst_p.resolve()
    started = time.perf_counter()

    denoiser = _get_gtcrn_denoiser(model_path, params["num_threads"])
    # 1) GTCRN 分块推理
    y = _run_gtcrn_chunks(denoiser, x, params=params)
    # 2) 整段电平对齐（只对齐中位帧电平，不做逐帧动态增益）
    level_gain_db = 0.0
    if level_match:
        level_gain_db = _level_match_gain_db(x, y, GTCRN_SR)
        if np.isfinite(level_gain_db) and abs(level_gain_db) > 1e-9:
            y = (y * np.float32(10.0 ** (level_gain_db / 20.0))).astype(np.float32)
    # 3) VAD 段 → 逐帧下限曲线 → STFT 幅度下限（输入相位）
    seg_list = (
        _normalize_segments(segments) if segments is not None
        else _energy_vad_segments(x, GTCRN_SR)
    )
    n_frames = 1 + x.size // GTCRN_HOP
    floor_curve = gain_floor_curve(
        n_frames,
        GTCRN_HOP / GTCRN_SR,
        GTCRN_NFFT * 1000.0 / GTCRN_SR,
        seg_list,
        floor_speech_db=params["floor_speech_db"],
        floor_pause_db=params["floor_pause_db"],
        smooth_ms=params["smooth_ms"],
    )
    z = _apply_gain_floor(x, y, floor_curve, params=params)

    # 4) 写 float32 WAV + 产物校验（存在、采样率、帧数）
    _write_wav_float32(resolved_dst, GTCRN_SR, z)
    elapsed_s = time.perf_counter() - started
    out_info = _read_wav_info(resolved_dst, "GTCRN 降噪产物")
    if int(out_info.sample_rate) != GTCRN_SR:
        raise DenoiseError(
            f"GTCRN 产物采样率 {int(out_info.sample_rate)} ≠ {GTCRN_SR}：{resolved_dst}"
        )
    duration_diff = abs(float(out_info.duration_s) - duration_s)
    if duration_diff >= DURATION_TOLERANCE_S:
        raise DenoiseError(
            f"GTCRN 产物时长与输入差 {duration_diff * 1000.0:.1f} ms"
            f"（要求 <{DURATION_TOLERANCE_S * 1000.0:.0f} ms）：{resolved_dst}"
        )

    rtf = elapsed_s / duration_s if duration_s > 0 else None
    result: Dict[str, Any] = {
        "ok": True,
        "mode": "gtcrn",
        "model": str(model_path),
        "src_sr": int(sr),
        "dst_sr": GTCRN_SR,
        "duration_s": round(duration_s, 3),
        "dst_duration_s": round(float(out_info.duration_s), 3),
        "elapsed_s": round(elapsed_s, 3),
        "rtf": round(rtf, 4) if rtf is not None else None,
        "floor_speech_db": params["floor_speech_db"],
        "floor_pause_db": params["floor_pause_db"],
        "smooth_ms": params["smooth_ms"],
        "chunk_s": params["chunk_s"],
        "overlap_s": params["overlap_s"],
        "level_match_db": round(level_gain_db, 2),
        "level_match": bool(level_match),
        "n_segments": len(seg_list),
        "segments_source": "analyze" if segments is not None else "energy_vad",
        "src": str(resolved_src),
        "dst": str(resolved_dst),
        "size_bytes": resolved_dst.stat().st_size,
    }
    log.info(
        "GTCRN 降噪完成：%s mode=gtcrn floor=%.1f/%.1f dB smooth=%.0f ms "
        "%.1fs（RTF=%.4f，level_match=%+.1f dB）→ %s",
        resolved_src.name, params["floor_speech_db"], params["floor_pause_db"],
        params["smooth_ms"], elapsed_s, rtf or 0.0, level_gain_db, resolved_dst.name,
    )
    return result


# ---------------------------------------------------------------- 渲染轴入口
def denoise_44k(
    decoded44k: str | Path,
    cfg: Any,
    job_id: str,
    *,
    snr_db: Optional[float] = None,
    nr: Optional[float] = None,
    nf: Optional[float] = None,
) -> Path:
    """渲染轴降噪：44.1k/mono/float32 输入 → ``<cache_dir>/<job_id>.denoised44k.wav``。

    参数
    ----
    decoded44k : ``decode.ensure_44k`` 的产物（必须是 44.1k；渲染轴固定，不做隐式重采样）
    snr_db     : P2 的 ``levels.snr_db``；低于 ``denoise.low_snr_db`` 时 nr 升到
                 ``denoise.low_snr_nr``（仅 afftdn，见 :func:`resolve_nr`）
    nr         : 逐文件显式覆盖（优先级最高；默认全库仍走配置的 nr=12）
    nf         : 逐文件显式覆盖噪声底（仅 afftdn；默认 None=配置锁定值 −30，
                 留给「用户同意后才调」的场景，见 :func:`resolve_nf`）

    返回产物路径；输入采样率不是 44.1k、产物不是 44.1k/mono/float32 或 ffmpeg 失败时抛
    :class:`DenoiseError`。
    """
    if not job_id:
        raise DenoiseError("denoise_44k 需要 job_id 以命名缓存产物")
    src_p = Path(decoded44k)
    if not src_p.is_file():
        raise DenoiseError(f"44.1k 输入不存在：{src_p}（先跑 decode.ensure_44k）")
    render_sr = _decode_render_sr(cfg)
    info = _read_wav_info(src_p, "44.1k 输入")
    if int(info.sample_rate) != render_sr:
        raise DenoiseError(
            f"denoise_44k 只接受 {render_sr} Hz 渲染轴输入，当前 {int(info.sample_rate)} Hz：{src_p}"
        )
    if int(info.channels) != 1:
        raise DenoiseError(
            f"denoise_44k 只接受单声道输入，当前 {int(info.channels)} 声道：{src_p}"
        )

    mode_value = _mode_value(cfg, None)
    effective_nr = resolve_nr(cfg, mode=mode_value, snr_db=snr_db, nr=nr)
    base_nr = resolve_nr(cfg, mode=mode_value) if mode_value == "afftdn" else None
    if (
        effective_nr is not None
        and base_nr is not None
        and effective_nr > base_nr + 1e-9
        and nr is None
    ):
        log.info(
            "低 SNR（%s dB < %s dB）→ nr %s → %s：%s",
            _as_float(snr_db, "snr_db") if snr_db is not None else "?",
            _fmt_num(_cfg_get(cfg, "denoise.low_snr_db", 12.0)),
            _fmt_num(base_nr), _fmt_num(effective_nr), job_id,
        )

    try:
        cache_dir = Path(cfg.path("cache_dir"))
    except ConfigError as exc:
        raise DenoiseError(f"缺少路径配置 paths.cache_dir：{exc}") from exc
    dst = cache_dir / f"{job_id}{CACHE_SUFFIX_DENOISED44K}"

    denoise_wav(src_p, dst, cfg, mode=mode_value, nr=effective_nr, nf=nf)

    # 产物契约：44.1k / mono / float32（ffmpeg 输出看位宽 32bit 即 float32）
    out_info = _read_wav_info(dst, "降噪产物")
    if int(out_info.sample_rate) != render_sr:
        raise DenoiseError(
            f"降噪产物采样率 {int(out_info.sample_rate)} ≠ {render_sr}：{dst}"
        )
    if int(out_info.channels) != 1:
        raise DenoiseError(f"降噪产物不是单声道（{int(out_info.channels)}）：{dst}")
    if int(out_info.sampwidth) != 4:
        raise DenoiseError(
            f"降噪产物不是 32-bit float（sampwidth={int(out_info.sampwidth)}）：{dst}"
        )
    return dst


def _resample_to_render_sr(
    src: Path, dst: Path, render_sr: int, cfg: Any, *, duration_s: float
) -> None:
    """ffmpeg ``aresample`` 到渲染轴采样率，输出 pcm_f32le WAV；失败删半成品再抛错。"""
    _require_ffmpeg()
    resolved_src = src.resolve()
    resolved_dst = dst.resolve()
    resolved_dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-i", str(resolved_src),
        "-af", f"aresample={int(render_sr)}",
        "-c:a", PCM_F32, "-f", "wav", str(resolved_dst),
    ]
    timeout = _default_timeout(cfg, duration_s)
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        resolved_dst.unlink(missing_ok=True)
        raise DenoiseError(
            f"GTCRN 渲染轴 aresample 超时（>{timeout:.1f}s）：{resolved_src}"
        ) from exc
    if proc.returncode != 0 or not resolved_dst.is_file():
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        resolved_dst.unlink(missing_ok=True)
        raise DenoiseError(
            f"GTCRN 渲染轴 aresample 失败（exit={proc.returncode}）：{resolved_src}\n"
            f"{stderr[-1500:] or '(无 stderr)'}"
        )


def denoise_gtcrn_44k(
    wav16k: str | Path,
    cfg: Any,
    job_id: str,
    *,
    segments: Optional[Sequence[Sequence[float]]] = None,
) -> Path:
    """渲染轴 GTCRN 降噪：16k/mono 输入 → ``<cache_dir>/<job_id>.denoised44k.wav``。

    两步落盘（都带产物校验，不留半成品）：

    1. :func:`gtcrn_denoise_wav16k` 写 ``<job_id>.gtcrn16k.wav``（float32 16 kHz）；
    2. ffmpeg ``-af aresample=<decode.render_sr>`` 写 ``<job_id>.denoised44k.wav``
       （float32 / mono / 44.1 kHz）。

    输出契约与 :func:`denoise_44k` 完全一致（采样率/声道/位宽/时长差 <50 ms），
    所以 export 的 trim/loudness 两段可以原样复用。

    参数
    ----
    wav16k   : ``decode`` 的 16 kHz 产物（analyze 用的同一份，GTCRN 只吃 16k）
    segments : analyze 的 VAD 段 ``[[start_s, end_s], ...]``（同一时间轴）；``None`` 时
               用 :func:`_energy_vad_segments` 能量 VAD 兜底
    """
    if not job_id:
        raise DenoiseError("denoise_gtcrn_44k 需要 job_id 以命名缓存产物")
    src_p = Path(wav16k)
    if not src_p.is_file():
        raise DenoiseError(f"16k 输入不存在：{src_p}（先跑 decode）")
    info = _read_wav_info(src_p, "16k 输入")
    if int(info.sample_rate) != GTCRN_SR:
        raise DenoiseError(
            f"denoise_gtcrn_44k 只接受 {GTCRN_SR} Hz 输入，当前 {int(info.sample_rate)} Hz：{src_p}"
        )
    if int(info.channels) != 1:
        raise DenoiseError(
            f"denoise_gtcrn_44k 只接受单声道输入，当前 {int(info.channels)} 声道：{src_p}"
        )
    duration_s = float(info.duration_s)

    render_sr = _decode_render_sr(cfg)
    try:
        cache_dir = Path(cfg.path("cache_dir"))
    except (ConfigError, AttributeError) as exc:
        raise DenoiseError(f"缺少路径配置 paths.cache_dir：{exc}") from exc
    dst16 = cache_dir / f"{job_id}{CACHE_SUFFIX_GTCRN16K}"
    dst44 = cache_dir / f"{job_id}{CACHE_SUFFIX_DENOISED44K}"

    started = time.perf_counter()
    gtcrn_denoise_wav16k(src_p, dst16, cfg, segments=segments)
    _resample_to_render_sr(dst16, dst44, render_sr, cfg, duration_s=duration_s)

    # 产物契约：渲染轴采样率 / mono / float32 / 时长与输入一致（与 denoise_44k 相同口径）
    out_info = _read_wav_info(dst44, "GTCRN 渲染产物")
    if int(out_info.sample_rate) != render_sr:
        raise DenoiseError(
            f"GTCRN 产物采样率 {int(out_info.sample_rate)} ≠ {render_sr}：{dst44}"
        )
    if int(out_info.channels) != 1:
        raise DenoiseError(f"GTCRN 产物不是单声道（{int(out_info.channels)}）：{dst44}")
    if int(out_info.sampwidth) != 4:
        raise DenoiseError(
            f"GTCRN 产物不是 32-bit float（sampwidth={int(out_info.sampwidth)}）：{dst44}"
        )
    duration_diff = abs(float(out_info.duration_s) - duration_s)
    if duration_diff >= DURATION_TOLERANCE_S:
        raise DenoiseError(
            f"GTCRN 渲染产物时长与输入差 {duration_diff * 1000.0:.1f} ms"
            f"（要求 <{DURATION_TOLERANCE_S * 1000.0:.0f} ms）：{dst44}"
        )
    elapsed_s = time.perf_counter() - started
    log.info(
        "GTCRN 渲染轴完成：%s %.1fs（RTF=%.4f，%d Hz/mono/float32）→ %s",
        job_id, elapsed_s, elapsed_s / duration_s if duration_s > 0 else 0.0,
        render_sr, dst44.name,
    )
    return dst44
