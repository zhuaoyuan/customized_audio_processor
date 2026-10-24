# -*- coding: utf-8 -*-
"""打包脚本单测：交付包不夹带 data/out/work/模型/密钥/用户样例稿。"""

from __future__ import annotations

import zipfile
from pathlib import Path

from app.tools.package_release import build_zip, collect


def test_collect_excludes_private_and_heavy(project_root: Path):
    rels = {rel for _path, rel in collect(project_root)}
    assert "app/cli.py" in rels
    assert "app/pipeline/compose.py" in rels
    assert "docs/交付说明.md" in rels
    assert "requirements.txt" in rels
    # 用户样例稿 / 私有配置 / 临时产物
    assert "app/transcript.md" not in rels
    assert not any("set_key" in rel for rel in rels)
    assert not any(rel.endswith("local.yaml") or ".local.yaml" in rel for rel in rels)
    assert not any(rel.startswith(("data/", "out/", "work/", "dist/", ".git/")) for rel in rels)
    assert not any("__pycache__" in rel or rel.endswith((".pyc", ".pyo")) for rel in rels)


def test_build_zip_is_scoped_and_prefixed(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "app" / "pipeline").mkdir(parents=True)
    (root / "app" / "pipeline" / "compose.py").write_text("x = 1\n", encoding="utf-8")
    (root / "app" / "transcript.md").write_text("user file\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "交付说明.md").write_text("# d\n", encoding="utf-8")
    (root / "work").mkdir()
    (root / "work" / "set_key.ps1").write_text("$env:X='sk-secret'\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "a.m4a").write_bytes(b"binary")
    (root / "README.md").write_text("# r\n", encoding="utf-8")

    entries = collect(root)
    target = build_zip(root, tmp_path / "dist", "9.9.9", entries)
    assert target.is_file()
    with zipfile.ZipFile(target) as zf:
        names = set(zf.namelist())
    prefix = "audio_processer-v9.9.9/"
    assert prefix + "app/pipeline/compose.py" in names
    assert prefix + "docs/交付说明.md" in names
    assert prefix + "README.md" in names
    assert all(name.startswith(prefix) for name in names)
    assert not any("transcript.md" in name or "set_key" in name or ".m4a" in name
                   for name in names)
