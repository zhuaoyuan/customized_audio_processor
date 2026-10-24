# -*- coding: utf-8 -*-
"""配置加载：configs/default.yaml → preset → 环境变量 → 命令行覆盖，输出合并后的 Config。

设计要点
--------
1. 合并顺序（后者覆盖前者）：default.yaml → presets/<preset>.yaml → 环境变量 → overrides；
2. ``${paths.xxx}`` 占位符在**合并之后**统一解析，避免 preset 改了路径后模板失效；
3. 所有相对路径在这里一次性解析成绝对路径（``Path`` 对象），调用方不再关心项目根目录；
4. ``correct.api_key`` 是敏感项：默认脱敏输出，且**不参与** ``config_hash``；
5. 非法配置在加载时抛出 ``ConfigError``（带明确中文错误信息），不要拖到运行期才炸。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

# ---------------------------------------------------------------- 常量
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
PRESET_DIR = CONFIG_DIR / "presets"
PRESETS = ("fast", "standard", "quality")
DEFAULT_PRESET = "standard"

#: 需要从相对路径解析为绝对路径的 paths.* 键
_PATH_KEYS = (
    "data_dir", "out_dir", "work_dir", "cache_dir",
    "quarantine_dir", "logs_dir", "db_path", "models_root",
)

#: 敏感字段（不参与 config_hash、默认脱敏）
_SECRET_PATHS = (("correct", "api_key"),)

#: 环境变量覆盖表：env → 配置点号路径
_ENV_MAP = {
    "DEEPSEEK_API_KEY": "correct.api_key",
    "AP_PRESET": "preset.name",
    "AP_OUT_DIR": "paths.out_dir",
    "AP_CACHE_DIR": "paths.cache_dir",
    "AP_DB_PATH": "paths.db_path",
    "AP_LOG_LEVEL": "logging.level",
}

_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_DENOISE_MODES = ("afftdn", "arnndn", "highpass", "off")
_DIAR_MODES = ("none", "sampled", "full")
_UNRELATED_MODES = ("conservative", "off", "aggressive")
_SAMPLE_FMTS = ("pcm_f32le", "pcm_s16le", "pcm_s24le", "pcm_s32le")


class ConfigError(RuntimeError):
    """配置缺失、非法或占位符无法解析。"""


# ---------------------------------------------------------------- YAML 读写
def _yaml():
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - 环境缺依赖时给出可执行提示
        raise ConfigError(
            "缺少 PyYAML：请执行 "
            "work\\venv\\Scripts\\python -m pip install -r requirements.txt "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc
    return yaml


def read_yaml(path: str | Path) -> Dict[str, Any]:
    """读取 YAML；文件不存在或结构不是 mapping 时抛 ConfigError。"""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"配置文件不存在：{p}")
    try:
        data = _yaml().safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:  # yaml.YAMLError 等
        raise ConfigError(f"配置文件解析失败：{p}：{exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是 mapping：{p}")
    return data


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> Path:
    """把 dict 写成 UTF-8 YAML（中文不转义），返回路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = _yaml().safe_dump(
        dict(data), allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------- dict 工具
def deep_merge(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """递归合并：override 覆盖 base；两边都是 dict 时逐键递归，其余整体替换。"""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in (override or {}).items():
        if isinstance(out.get(key), dict) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = copy.deepcopy(value)
    return out


def _lookup(data: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _assign(data: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur: Any = data
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


# ---------------------------------------------------------------- 占位符 / 路径
def _interpolate(data: Mapping[str, Any]) -> Dict[str, Any]:
    """解析 ``${paths.xxx}`` 占位符（支持 ``${xxx}`` 简写与嵌套）。"""
    out: Dict[str, Any] = copy.deepcopy(dict(data))
    paths = out.get("paths") or {}
    if not isinstance(paths, dict):
        raise ConfigError("配置 paths 段必须是 mapping")

    def resolve_value(value: str, depth: int = 0) -> str:
        if depth > 8:
            raise ConfigError(f"占位符嵌套过深（可能循环引用）：{value}")
        if not isinstance(value, str) or "${" not in value:
            return value

        def repl(match: "re.Match[str]") -> str:
            key = match.group(1).strip()
            parts = key.split(".")
            if parts and parts[0] == "paths":
                parts = parts[1:]
            if not parts:
                raise ConfigError(f"非法占位符：${{{key}}}")
            try:
                target = _lookup(paths, ".".join(parts))
            except KeyError as exc:
                raise ConfigError(f"占位符 ${{{key}}} 引用了不存在的 paths 键") from exc
            return resolve_value(str(target), depth + 1)

        return _PLACEHOLDER_RE.sub(repl, value)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return resolve_value(node)

    return walk(out)


def _abs_path(value: str | Path, root: Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.expanduser().resolve()


def _absolutize(data: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """把 paths.* 与 paths.models.* 的相对路径解析为绝对路径（保留 str 形式便于序列化）。"""
    out = copy.deepcopy(data)
    paths = out.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ConfigError("配置 paths 段必须是 mapping")
    for key in _PATH_KEYS:
        value = paths.get(key)
        if value:
            paths[key] = str(_abs_path(value, root))
    models = paths.get("models")
    if isinstance(models, dict):
        for key, value in list(models.items()):
            if value:
                models[key] = str(_abs_path(value, root))
    return out


# ---------------------------------------------------------------- 脱敏 / hash
def _redact(data: Mapping[str, Any]) -> Dict[str, Any]:
    """复制并脱敏敏感字段。"""
    out = copy.deepcopy(dict(data))
    for sec_path in _SECRET_PATHS:
        node: Any = out
        for part in sec_path[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and node.get(sec_path[-1]):
            node[sec_path[-1]] = "***"
    return out


def _canonical_for_hash(data: Mapping[str, Any]) -> Dict[str, Any]:
    """参与 config_hash 的内容：去掉路径与密钥（换机器/换目录不应导致 hash 变化）。"""
    out = _redact(data)
    out.pop("paths", None)
    correct = out.get("correct")
    if isinstance(correct, dict):
        correct.pop("api_key", None)
    return out


def compute_config_hash(data: Mapping[str, Any]) -> str:
    """配置指纹（sha256，稳定排序），用于判断「同配置可复用产物」。"""
    payload = json.dumps(
        _canonical_for_hash(data), ensure_ascii=False, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 环境变量
def apply_env(data: Dict[str, Any], env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """把环境变量覆盖叠加到配置上（不做路径解析）。"""
    env = os.environ if env is None else env
    out = copy.deepcopy(data)
    for env_key, dotted in _ENV_MAP.items():
        value = env.get(env_key)
        if value:
            _assign(out, dotted, value)
    # MODEL_DIR 只改 models_root，模型子路径由 ${paths.models_root} 占位符自动跟随
    model_dir = env.get("MODEL_DIR")
    if model_dir:
        _assign(out, "paths.models_root", model_dir)
    return out


# ---------------------------------------------------------------- 校验
def validate_config(data: Mapping[str, Any]) -> None:
    """配置合法性检查；不通过抛 ConfigError（一次性列出全部问题）。"""
    errors: list[str] = []

    def section(name: str) -> Dict[str, Any]:
        value = data.get(name)
        if not isinstance(value, dict):
            errors.append(f"缺少配置段：{name}")
            return {}
        return value

    def num(sec: Mapping[str, Any], sec_name: str, key: str, lo=None, hi=None):
        value = sec.get(key)
        if not isinstance(value, (int, float)):
            errors.append(f"{sec_name}.{key} 必须是数值，当前={value!r}")
            return None
        if lo is not None and value < lo:
            errors.append(f"{sec_name}.{key}={value} 小于下限 {lo}")
        if hi is not None and value > hi:
            errors.append(f"{sec_name}.{key}={value} 大于上限 {hi}")
        return float(value)

    preset = data.get("preset") or {}
    if isinstance(preset, dict):
        name = preset.get("name", DEFAULT_PRESET)
        if name not in PRESETS:
            errors.append(f"preset.name={name!r} 非法，可选：{', '.join(PRESETS)}")

    logging_sec = section("logging")
    level = logging_sec.get("level", "INFO")
    if level not in _LOG_LEVELS:
        errors.append(f"logging.level={level!r} 非法，可选：{', '.join(_LOG_LEVELS)}")

    decode = section("decode")
    num(decode, "decode", "analysis_sr", lo=8000, hi=192000)
    num(decode, "decode", "render_sr", lo=8000, hi=192000)
    num(decode, "decode", "timeout_s", lo=1)
    num(decode, "decode", "min_duration_ratio", lo=0.0, hi=1.0)
    if decode.get("sample_fmt") not in _SAMPLE_FMTS:
        errors.append(f"decode.sample_fmt={decode.get('sample_fmt')!r} 非法，可选：{_SAMPLE_FMTS}")

    levels = section("levels")
    num(levels, "levels", "target_asl_dbfs", lo=-60.0, hi=0.0)
    num(levels, "levels", "window_s", lo=1.0)
    num(levels, "levels", "window_min_speech_s", lo=0.0)
    pct = num(levels, "levels", "noise_floor_percentile", lo=0.1, hi=99.9)
    if pct is not None and pct != int(pct):
        # 允许浮点，但分位必须能落在一个 0..100 的区间里
        pass
    quantiles = levels.get("percentiles")
    if quantiles is not None:
        if not isinstance(quantiles, (list, tuple)) or not quantiles:
            errors.append("levels.percentiles 必须是非空列表")
        else:
            if any(not isinstance(q, (int, float)) or not (0 <= q <= 100) for q in quantiles):
                errors.append(f"levels.percentiles 取值必须在 0..100：{quantiles}")
            elif list(quantiles) != sorted(quantiles):
                errors.append("levels.percentiles 必须升序")

    trim = section("trim")
    num(trim, "trim", "boundary_window_s", lo=1.0)
    num(trim, "trim", "boundary_min_chars_per_sec", lo=0.0)
    if num(trim, "trim", "boundary_consecutive_windows", lo=1) and (
        trim.get("boundary_consecutive_windows") % 1 != 0
    ):
        errors.append("trim.boundary_consecutive_windows 必须是整数")
    num(trim, "trim", "internal_guard_s", lo=0.0)

    denoise = section("denoise")
    if denoise.get("mode") not in _DENOISE_MODES:
        errors.append(
            f"denoise.mode={denoise.get('mode')!r} 非法，可选：{', '.join(_DENOISE_MODES)}"
        )

    loud = section("loudness")
    num(loud, "loudness", "target_asl_dbfs", lo=-60.0, hi=0.0)
    gain_min = num(loud, "loudness", "gain_min_db", lo=-60.0, hi=0.0)
    gain_max = num(loud, "loudness", "gain_max_db", lo=0.0, hi=60.0)
    if gain_min is not None and gain_max is not None and gain_min >= gain_max:
        errors.append("loudness.gain_min_db 必须小于 gain_max_db")
    v2 = loud.get("v2") or {}
    if isinstance(v2, dict) and v2.get("enabled", True):
        clamp = num(v2, "loudness.v2", "clamp_db", lo=0.0, hi=24.0)
        high = v2.get("high_drift_clamp_db")
        if clamp is not None and isinstance(high, (int, float)) and high < clamp:
            errors.append("loudness.v2.high_drift_clamp_db 不能小于 clamp_db")
        num(v2, "loudness.v2", "max_rate_db_s", lo=0.0, hi=24.0)
    limiter = loud.get("limiter") or {}
    if isinstance(limiter, dict):
        tp = num(limiter, "loudness.limiter", "target_tp_dbtp", lo=-24.0, hi=0.0)
        ceiling = num(limiter, "loudness.limiter", "ceiling_dbfs", lo=-24.0, hi=0.0)
        if tp is not None and ceiling is not None and ceiling > tp:
            errors.append(
                f"loudness.limiter.ceiling_dbfs({ceiling}) 必须不高于 target_tp_dbtp({tp})"
            )
        num(limiter, "loudness.limiter", "oversample", lo=1, hi=16)

    asr = section("asr")
    seg_min = num(asr, "asr", "segment_min_s", lo=0.0)
    seg_max = num(asr, "asr", "segment_max_s", lo=0.0)
    if seg_min is not None and seg_max is not None and seg_min >= seg_max:
        errors.append("asr.segment_min_s 必须小于 segment_max_s")
    num(asr, "asr", "num_threads", lo=1)

    analyze = data.get("analyze") or {}
    if not isinstance(analyze, dict):
        errors.append("analyze 段必须是 mapping")
    else:
        for key in ("cache_enabled", "boundary_use_asr", "ebur128"):
            value = analyze.get(key, True)
            if not isinstance(value, bool):
                errors.append(f"analyze.{key} 必须是 bool，当前={value!r}")
        review_file = analyze.get("manual_review_file", "")
        if review_file is not None and not isinstance(review_file, str):
            errors.append("analyze.manual_review_file 必须是字符串路径（可为空）")

    diar = section("diar")
    if diar.get("mode") not in _DIAR_MODES:
        errors.append(f"diar.mode={diar.get('mode')!r} 非法，可选：{', '.join(_DIAR_MODES)}")

    unrel = section("unrelated_speech")
    if unrel.get("mode") not in _UNRELATED_MODES:
        errors.append(
            f"unrelated_speech.mode={unrel.get('mode')!r} 非法，可选：{', '.join(_UNRELATED_MODES)}"
        )

    correct = section("correct")
    if isinstance(correct, dict) and correct.get("enabled", True):
        cmin = num(correct, "correct", "chunk_min_chars", lo=100)
        cmax = num(correct, "correct", "chunk_max_chars", lo=100)
        if cmin is not None and cmax is not None and cmin > cmax:
            errors.append("correct.chunk_min_chars 不能大于 chunk_max_chars")
        num(correct, "correct", "max_diff_ratio", lo=0.0, hi=1.0)
        num(correct, "correct", "max_retries", lo=0)

    compose = data.get("compose") or {}
    if not isinstance(compose, dict):
        errors.append("compose 段必须是 mapping")
    else:
        num(compose, "compose", "chunk_chars", lo=300)
        num(compose, "compose", "overlap_segments", lo=0, hi=5)
        lo = num(compose, "compose", "min_ratio", lo=0.0, hi=1.0)
        hi = num(compose, "compose", "max_ratio", lo=0.0, hi=2.0)
        if lo is not None and hi is not None and lo >= hi:
            errors.append("compose.min_ratio 必须小于 max_ratio")
        num(compose, "compose", "min_ratio_short", lo=0.0, hi=1.0)
        num(compose, "compose", "short_chars", lo=0)
        num(compose, "compose", "max_retries", lo=0, hi=5)
        num(compose, "compose", "concurrency", lo=1, hi=16)
        num(compose, "compose", "timeout_s", lo=1)
        num(compose, "compose", "analyze_timeout_s", lo=1)
        num(compose, "compose", "max_input_chars", lo=1000)
        num(compose, "compose", "max_glossary", lo=0)

    allowed_efforts = {"none", "low", "high", "max"}
    for sec_name, sec in (("correct", correct), ("compose", compose)):
        if not isinstance(sec, dict):
            continue
        for key in ("reasoning_effort", "analyze_reasoning_effort"):
            if key not in sec:
                continue
            value = sec.get(key)
            if value not in allowed_efforts:
                errors.append(
                    f"{sec_name}.{key}={value!r} 非法，可选：none/low/high/max"
                )

    batch = section("batch")
    num(batch, "batch", "jobs", lo=1)
    num(batch, "batch", "disk_free_min_gb", lo=0.0)

    if errors:
        raise ConfigError("配置非法：\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------- Config
class Config:
    """合并后的只读配置视图。

    - ``cfg.get("loudness.v2.clamp_db")`` 点号取值；
    - ``cfg.path("cache_dir")`` / ``cfg.model_path("sensevoice_dir")`` 取绝对路径；
    - ``cfg.to_dict()`` 转普通 dict（默认脱敏，可直接 yaml/json 打印）。
    """

    __slots__ = ("_data", "preset", "root", "source", "config_hash")

    def __init__(
        self,
        data: Mapping[str, Any],
        preset: str = DEFAULT_PRESET,
        root: str | Path = PROJECT_ROOT,
        source: Optional[str | Path] = None,
    ) -> None:
        self._data = copy.deepcopy(dict(data))
        self.preset = preset
        self.root = Path(root)
        self.source = Path(source) if source else None
        self.config_hash = compute_config_hash(self._data)

    # -- 取值
    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return _lookup(self._data, dotted)
        except KeyError:
            return default

    def require(self, dotted: str) -> Any:
        try:
            return _lookup(self._data, dotted)
        except KeyError as exc:
            raise ConfigError(f"缺少配置项：{dotted}") from exc

    def section(self, name: str) -> Dict[str, Any]:
        value = self._data.get(name, {})
        return value if isinstance(value, dict) else {}

    def path(self, key: str) -> Path:
        """取 paths.<key> 的绝对路径。"""
        value = self._data.get("paths", {}).get(key)
        if not value:
            raise ConfigError(f"缺少路径配置 paths.{key}")
        return _abs_path(value, self.root)

    def model_path(self, key: str) -> Path:
        """取 paths.models.<key> 的绝对路径。"""
        models = (self._data.get("paths") or {}).get("models") or {}
        value = models.get(key)
        if not value:
            raise ConfigError(f"缺少模型路径配置 paths.models.{key}")
        return _abs_path(value, self.root)

    # -- 输出
    def to_dict(self, redact: bool = True) -> Dict[str, Any]:
        return _redact(self._data) if redact else copy.deepcopy(self._data)

    def to_yaml(self, redact: bool = True) -> str:
        return _yaml().safe_dump(
            self.to_dict(redact=redact),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    # -- 复制
    def copy(self) -> "Config":
        return Config(self._data, preset=self.preset, root=self.root, source=self.source)

    def replace(self, overrides: Optional[Mapping[str, Any]] = None) -> "Config":
        """在已解析配置上叠加点号覆盖，返回新 Config（路径重新解析自查，不重复应用 preset）。"""
        if not overrides:
            return self.copy()
        data = copy.deepcopy(self._data)
        for dotted, value in overrides.items():
            _assign(data, dotted, value)
        data = _interpolate(data)
        data = _absolutize(data, self.root)
        return Config(data, preset=self.preset, root=self.root, source=self.source)

    def __contains__(self, dotted: str) -> bool:
        try:
            _lookup(self._data, dotted)
            return True
        except KeyError:
            return False

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Config(preset={self.preset!r}, source={self.source}, hash={self.config_hash[:8]})"


# ---------------------------------------------------------------- 加载入口
def load_config(
    config_path: Optional[str | Path] = None,
    preset: Optional[str] = None,
    root: Optional[str | Path] = None,
    env: Optional[Mapping[str, str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    """加载并合并配置。

    参数
    ----
    config_path : 默认 ``configs/default.yaml``
    preset      : fast | standard | quality；默认取 ``AP_PRESET`` 或 standard
    root        : 相对路径的解析基准，默认项目根目录
    env         : 环境变量 dict（默认 ``os.environ``），便于测试注入
    overrides   : 点号路径 → 值（命令行 --out/--db 等）
    """
    root_path = Path(root) if root else PROJECT_ROOT
    base_path = Path(config_path) if config_path else root_path / "configs" / "default.yaml"

    data = read_yaml(base_path)

    env_map = os.environ if env is None else env
    preset_name = preset or env_map.get("AP_PRESET") or DEFAULT_PRESET
    if preset_name not in PRESETS:
        raise ConfigError(
            f"未知 preset：{preset_name!r}，可选：{', '.join(PRESETS)}"
        )
    preset_path = (base_path.parent / "presets" / f"{preset_name}.yaml")
    if preset_name != DEFAULT_PRESET or preset_path.is_file():
        if preset_path.is_file():
            data = deep_merge(data, read_yaml(preset_path))
    # 记录 preset 名（供 config show 与 hash 使用）
    _assign(data, "preset.name", preset_name)
    data.setdefault("preset", {})["name"] = preset_name

    data = apply_env(data, env_map)
    for dotted, value in (overrides or {}).items():
        _assign(data, dotted, value)

    data = _interpolate(data)
    data = _absolutize(data, root_path)
    validate_config(data)
    return Config(data, preset=preset_name, root=root_path, source=base_path)


def default_config() -> Config:
    """便捷入口：按默认 preset 加载项目配置。"""
    return load_config()
