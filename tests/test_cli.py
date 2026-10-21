# -*- coding: utf-8 -*-
"""CLI 冒烟测试：config show / scan / status / decode 的参数解析与入库行为。

用临时目录与假文件，不依赖 data/；真实六文件走 ``python -m app.cli decode --all``（见工程日志）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cli import EXIT_OK, EXIT_USAGE, discover_files, main


def _fake_media(root: Path, names: list[str], size: int = 70000) -> list[Path]:
    """造一批「足够大」的假媒体文件（scan 有 min_bytes 过滤）。"""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * size)
        paths.append(path)
    return paths


def test_config_show_json(capsys, tmp_path: Path):
    code = main(["config", "show", "--json",
                 "--out", str(tmp_path / "out")])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["levels"]["target_asl_dbfs"] == -20.0
    assert payload["trim"]["boundary_window_s"] == 120.0
    assert payload["paths"]["cache_dir"].endswith("cache")


def test_config_show_yaml_default(capsys, tmp_path: Path):
    code = main(["config", "show", "--preset", "fast", "--out", str(tmp_path / "out")])
    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert "name: fast" in captured.out
    assert "target_asl_dbfs: -20.0" in captured.out
    # 默认值必须出现在输出里（验收点：打印合并后配置）
    assert "boundary_min_chars_per_sec: 2.0" in captured.out
    # fast preset 覆盖项生效
    assert "enabled: false" in captured.out


def test_config_check_json(capsys):
    code = main(["config", "check", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code in (EXIT_OK, 1)
    assert "checks" in payload and "missing" in payload
    assert any(c["name"] == "data_dir" for c in payload["checks"])


def test_config_unknown_preset(capsys):
    with pytest.raises(SystemExit):
        main(["config", "show", "--preset", "turbo"])


def test_discover_files_filters(tmp_path: Path):
    _fake_media(tmp_path / "in", ["a.m4a", "sub/b.mp3", "c.txt", "d.wav"])
    (tmp_path / "in" / "small.m4a").write_bytes(b"\x00" * 10)
    (tmp_path / "in" / ".hidden").mkdir()
    (tmp_path / "in" / ".hidden" / "e.m4a").write_bytes(b"\x00" * 70000)
    found = discover_files(tmp_path / "in", min_bytes=1024)
    names = [p.name for p in found]
    assert "a.m4a" in names and "b.mp3" in names and "d.wav" in names
    assert "c.txt" not in names
    assert "small.m4a" not in names
    assert "e.m4a" not in names  # 隐藏目录跳过
    # 非递归
    flat = discover_files(tmp_path / "in", recursive=False, min_bytes=1024)
    assert "b.mp3" not in [p.name for p in flat]
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "nope")


def test_scan_then_status(capsys, tmp_path: Path):
    data = tmp_path / "data"
    _fake_media(data, [
        "0_20140227-丁耘 西方哲学(20140227)_normal..m4a",
        "24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a",
        "4_丁耘理想国.古希腊哲学(第四卷)_normal..m4a",
    ])
    out = tmp_path / "out"
    assert main(["scan", str(data), "--out", str(out)]) == EXIT_OK
    scan_out = capsys.readouterr().out
    assert "新增 3" in scan_out
    assert (out / "jobs.db").is_file()

    # 再扫一次：不重复新增、不重置状态
    assert main(["scan", str(data), "--out", str(out)]) == EXIT_OK
    assert "新增 0" in capsys.readouterr().out

    assert main(["status", "--out", str(out), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["counts"]["pending"] == 3
    assert len(payload["jobs"]) == 3
    assert all(job["status"] == "pending" for job in payload["jobs"])
    ids = [job["id"] for job in payload["jobs"]]
    assert len(set(ids)) == 3
    assert all(len(i.split("_")[0]) == 8 for i in ids)

    # 人读表格
    assert main(["status", "--out", str(out)]) == EXIT_OK
    table = capsys.readouterr().out
    assert "pending" in table and "状态汇总" in table


def test_scan_dry_run_and_empty(capsys, tmp_path: Path):
    data = tmp_path / "data"
    _fake_media(data, ["a.m4a"])
    out = tmp_path / "out"
    assert main(["scan", str(data), "--out", str(out), "--dry-run"]) == EXIT_OK
    assert "a.m4a" in capsys.readouterr().out
    assert not (out / "jobs.db").exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["scan", str(empty), "--out", str(out)]) == 1
    assert "没有发现" in capsys.readouterr().err


def test_status_without_db(capsys, tmp_path: Path):
    code = main(["status", "--out", str(tmp_path / "nope")])
    assert code == EXIT_USAGE
    assert "数据库不存在" in capsys.readouterr().err


def test_scan_limit(capsys, tmp_path: Path):
    data = tmp_path / "data"
    _fake_media(data, [f"f{i}.m4a" for i in range(5)])
    out = tmp_path / "out"
    assert main(["scan", str(data), "--out", str(out), "--limit", "2"]) == EXIT_OK
    assert "新增 2" in capsys.readouterr().out


def test_help_available():
    for argv in (["decode", "--help"], ["scan", "--help"], ["status", "--help"],
                 ["config", "--help"]):
        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == 0
