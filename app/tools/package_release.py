# -*- coding: utf-8 -*-
"""打交付包：只收代码/配置/文档/测试，排除 data/out/work/模型/密钥。

用法：
    work\\venv\\Scripts\\python -m app.tools.package_release
    work\\venv\\Scripts\\python -m app.tools.package_release --out dist --version 0.2.0

产物：dist/audio_processer-v<version>-<yyyymmdd>.zip
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from app import __version__

ROOT = Path(__file__).resolve().parents[2]

#: 整体收录的目录
INCLUDE_DIRS = ("app", "configs", "docs", "tests")
#: 根目录单文件
INCLUDE_FILES = (
    "README.md", "CHANGELOG.md", "requirements.txt", "pytest.ini",
    "start_web.bat", ".gitignore",
)
#: 路径片段/文件名排除（命中任一条即跳过）
EXCLUDE_PARTS = {
    "__pycache__", ".pytest_cache", ".git", "venv", "work", "out", "data",
    "dist", "node_modules",
}
#: 文件名/后缀排除
EXCLUDE_GLOBS = (
    "*.pyc", "*.pyo", "*.log", "*.wav", "*.m4a", "*.mp3", "*.flac", "*.onnx",
    "*.rnnn", "*.bin", "*.zip", "*.7z", "*.rar",
    "local.yaml", "*.local.yaml", "set_key.ps1",
    # 用户手工放进 app/ 的样例稿，不属于代码
    "app/transcript.md",
)


def _iter_files(root: Path, rel_dir: str) -> Iterable[Tuple[Path, str]]:
    base = root / rel_dir
    if not base.is_dir():
        return
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        rel_parts = set(Path(rel).parts)
        if rel_parts & EXCLUDE_PARTS:
            continue
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(path.name, pat)
               for pat in EXCLUDE_GLOBS):
            continue
        yield path, rel


def collect(root: Path) -> List[Tuple[Path, str]]:
    entries: List[Tuple[Path, str]] = []
    for directory in INCLUDE_DIRS:
        entries.extend(_iter_files(root, directory))
    for name in INCLUDE_FILES:
        path = root / name
        if path.is_file():
            entries.append((path, name))
    # 去重 + 稳定排序
    seen, unique = set(), []
    for path, rel in sorted(entries, key=lambda item: item[1]):
        if rel in seen:
            continue
        seen.add(rel)
        unique.append((path, rel))
    return unique


def build_zip(root: Path, out_dir: Path, version: str, entries: Sequence[Tuple[Path, str]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    target = out_dir / f"audio_processer-v{version}-{stamp}.zip"
    prefix = f"audio_processer-v{version}"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path, rel in entries:
            zf.write(path, f"{prefix}/{rel}")
    return target


def summarize(target: Path, entries: Sequence[Tuple[Path, str]]) -> None:
    raw = sum(path.stat().st_size for path, _ in entries)
    by_top: dict[str, int] = {}
    for path, rel in entries:
        top = rel.split("/", 1)[0]
        by_top[top] = by_top.get(top, 0) + path.stat().st_size
    print(f"包：{target}")
    print(f"文件数：{len(entries)}；原始大小：{raw / 1024 / 1024:.2f} MiB；"
          f"压缩后：{target.stat().st_size / 1024 / 1024:.2f} MiB")
    for top, size in sorted(by_top.items(), key=lambda item: -item[1]):
        print(f"  {top:<12s} {size / 1024 / 1024:8.2f} MiB")
    if raw:
        largest = sorted(entries, key=lambda item: -item[0].stat().st_size)[:5]
        print("最大的 5 个文件：")
        for path, rel in largest:
            print(f"  {path.stat().st_size / 1024:8.1f} KiB  {rel}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打 audio_processer 交付包（代码+配置+文档+测试）")
    parser.add_argument("--root", default=str(ROOT), help="项目根目录（默认当前仓库）")
    parser.add_argument("--out", default=None, help="输出目录（默认 <root>/dist）")
    parser.add_argument("--version", default=__version__, help=f"版本号（默认 {__version__}）")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    entries = collect(root)
    if not entries:
        print("[错误] 没有可打包文件，检查 --root", file=sys.stderr)
        return 2
    out_dir = Path(args.out).resolve() if args.out else root / "dist"
    target = build_zip(root, out_dir, args.version, entries)
    summarize(target, entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
