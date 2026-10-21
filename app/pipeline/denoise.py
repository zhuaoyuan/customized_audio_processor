# -*- coding: utf-8 -*-
"""降噪阶段（P3-T1）：ffmpeg 滤镜链封装（highpass / afftdn / arnndn）。

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
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config import ConfigError
from ..logging_setup import get_logger
from .decode import DecodeError, RENDER_SR, wav_info

log = get_logger(__name__)

#: 降噪产物的缓存后缀（与 decode 的 ``.44k.wav`` 区分）
CACHE_SUFFIX_DENOISED44K = ".denoised44k.wav"
#: 统一 float32（与 decode 一致，禁止 int16）
PCM_F32 = "pcm_f32le"
#: 输出与输入时长允许偏差（秒）；超过即视为 ffmpeg 异常
DURATION_TOLERANCE_S = 0.05
#: 合法降噪模式（与 configs/default.yaml 的 denoise.mode 对齐）
VALID_MODES = ("afftdn", "arnndn", "highpass", "off")


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
    """确定降噪模式（显式参数 > 配置 > 默认 afftdn）并校验。"""
    if mode is None:
        value = str(_cfg_get(cfg, "denoise.mode", "afftdn"))
    else:
        value = str(mode)
    value = value.strip().lower()
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

    ``nr`` / ``nf`` 为显式覆盖（默认 ``None`` = 用配置值，不在代码里写死具体文件）；
    ``src_sr`` 未给时按 ``decode.render_sr``（默认 44100）；arnndn 的最后一级 aresample
    必须回到输入采样率，否则 ``denoise_wav`` 的产物采样率会变成 48k。
    """
    mode_value = _mode_value(cfg, mode)
    highpass_hz = _as_highpass(_cfg_get(cfg, "denoise.highpass_hz", 80.0))
    if mode_value == "off":
        return "", None
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
