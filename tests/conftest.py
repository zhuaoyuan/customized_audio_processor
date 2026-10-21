# -*- coding: utf-8 -*-
"""pytest 公共夹具：项目根入 sys.path、临时目录、样例文件、ffmpeg 可用性。

约定：
- 依赖 ``data/`` 真实样例的用例标 ``@pytest.mark.data``，样例缺失时自动 skip；
- 依赖 ffmpeg/ffprobe 的用例用 ``requires_ffmpeg``；
- 所有用例只写 ``tmp_path``，不污染 ``out/`` 与 ``work/cache``。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 允许直接 `pytest tests/`（不安装包）
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"

#: 6 个样例的识别 glob（文件名带中文/空格/括号，只能用模式匹配）
SAMPLE_GLOBS: Dict[str, str] = {
    "f00": "0_20140227*",      # stco 偏移损坏，必须走修复路径
    "f04": "4_*",
    "f05": "*物理学*",
    "f06": "*论语*",
    "f20": "20_*",
    "f24": "24_*",
}


def find_samples() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not DATA_DIR.is_dir():
        return out
    for key, pattern in SAMPLE_GLOBS.items():
        matches = sorted(DATA_DIR.glob(pattern))
        if matches:
            out[key] = matches[0]
    return out


SAMPLES = find_samples()

requires_data = pytest.mark.skipif(not SAMPLES, reason="data/ 下没有可用的 m4a 样例")
requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="PATH 中缺 ffmpeg/ffprobe",
)


@pytest.fixture(scope="session")
def samples() -> Dict[str, Path]:
    return SAMPLES


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    path = tmp_path / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def cfg(tmp_path: Path):
    """把所有落盘路径指向 tmp_path 的配置（避免写脏仓库）。"""
    from app.config import load_config

    return load_config(
        overrides={
            "paths.out_dir": str(tmp_path / "out"),
            "paths.work_dir": str(tmp_path / "work"),
            "paths.cache_dir": str(tmp_path / "cache"),
            "paths.logs_dir": str(tmp_path / "logs"),
            "paths.quarantine_dir": str(tmp_path / "quarantine"),
            "paths.db_path": str(tmp_path / "jobs.db"),
        }
    )


@pytest.fixture(scope="session")
def session_cfg(tmp_path_factory):
    """会话级临时配置（给 session 级的数据用例用，避免 ScopeMismatch）。"""
    from app.config import load_config

    base = tmp_path_factory.mktemp("session_cfg")
    return load_config(
        overrides={
            "paths.out_dir": str(base / "out"),
            "paths.work_dir": str(base / "work"),
            "paths.cache_dir": str(base / "cache"),
            "paths.logs_dir": str(base / "logs"),
            "paths.quarantine_dir": str(base / "quarantine"),
            "paths.db_path": str(base / "jobs.db"),
        }
    )


@pytest.fixture()
def project_root() -> Path:
    return ROOT
