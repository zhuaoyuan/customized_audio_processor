# -*- coding: utf-8 -*-
"""P4-T3 正文抽样重验脚本（证据生成，产品代码在 app/pipeline/diarize.py）。

已知旧证据的问题（docs/工程日志.md、task-2）：``work/scan/diar`` 的 600 s 样本起点
f24=233.7 s（< head 479.5 s）、f00=152.4 s（课前闲聊），都不在讲课正文里；本脚本按
clean 轴「正文（避头尾 body_margin）+ 均分窗口」重跑，输出原始簇/合并后有效人数/主占比/判定。

用法::

    work\\venv\\Scripts\\python work/diar_p4_check.py --minutes 15 --segments 3
    work\\venv\\Scripts\\python work/diar_p4_check.py --probe 60   # 快速自检（f24 正文 60s）

输出：``work/diar_p4_check.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.pipeline import diarize as D  # noqa: E402

#: 6 文件 job（本任务只看 f24 正文 + f00 正文）
JOBS = {
    "f24": "7927721f_24_复旦哲学系丁耘_亚里士多德",
    "f00": "621cb91c_0_20140227_丁耘_西方哲学",
}


def compact(sampled: dict) -> dict:
    """抽样结果压成可读证据（不写逐段列表，保留簇时长分布）。"""
    return {
        "windows": sampled.get("windows"),
        "n_windows": sampled.get("n_windows"),
        "threshold": sampled.get("threshold"),
        "merge_threshold": sampled.get("merge_threshold"),
        "raw_clusters": sampled.get("raw_speakers"),
        "raw_n_segments": sampled.get("raw_n_segments"),
        "merged_clusters": sampled.get("merged_speakers"),
        "per_speaker_s": sampled.get("per_speaker_s"),
        "per_raw_speaker_s": dict(list((sampled.get("per_raw_speaker_s") or {}).items())[:20]),
        "main_ratio": sampled.get("main_ratio"),
        "min_speaker_s": sampled.get("min_speaker_s"),
        "min_speaker_ratio": sampled.get("min_speaker_ratio"),
        "n_stable": sampled.get("n_stable"),
        "stable": sampled.get("stable"),
        "is_multi": sampled.get("is_multi"),
        "verdict": sampled.get("verdict"),
        "timings": sampled.get("timings"),
    }


def run_one(cfg, job_id: str, windows, label: str) -> dict:
    wav = D.ensure_clean16k(job_id, cfg)
    x = D.load_wav16k(wav)
    duration = len(x) / D.SR
    t0 = time.perf_counter()
    sampled = D.sample_speakers(x, cfg, windows=windows)
    elapsed = time.perf_counter() - t0
    print(f"[{label}] {job_id} dur={duration:.1f}s windows={windows}", flush=True)
    print(f"[{label}] raw={sampled.get('raw_speakers')} "
          f"merged={sampled.get('merged_speakers')} "
          f"main_ratio={sampled.get('main_ratio')} "
          f"n_stable={sampled.get('n_stable')} -> {sampled.get('verdict')} "
          f"({elapsed:.1f}s)", flush=True)
    return {
        "job_id": job_id,
        "clean16k": str(wav),
        "clean_dur_s": round(duration, 3),
        "elapsed_s": round(elapsed, 2),
        **compact(sampled),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0,
                    help="采样总分钟数（默认 15 = 3×5min，复现 P4-T3 任务卡）")
    ap.add_argument("--segments", type=int, default=3, help="采样窗口个数")
    ap.add_argument("--body-margin", type=float, default=120.0, help="正文避让秒数")
    ap.add_argument("--probe", type=float, default=0.0, help=">0 时只跑 f24 正文该秒数做自检")
    ap.add_argument("--out", default=str(ROOT / "work" / "diar_p4_check.json"))
    args = ap.parse_args(argv)

    cfg = load_config()
    output: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "sample_speakers（pyannote-seg + cam++ + FastClustering/threshold + 质心归并）",
        "config": {
            "sample_minutes": args.minutes,
            "sample_segments": args.segments,
            "body_margin_s": args.body_margin,
            "threshold": cfg.get("diar.threshold"),
            "min_speaker_s": cfg.get("diar.min_speaker_s"),
            "min_speaker_ratio": cfg.get("diar.min_speaker_ratio"),
        },
        "note": "旧 work/scan/diar 样本起点不在正文（f24 233.7<479.5 头裁；f00 152.4 课前闲聊），"
                "本文件全部用 clean 轴正文样本重跑；判定 n=1 则 6 文件全部 none/主讲是正确结果。",
        "files": {},
    }

    if args.probe > 0:
        job_id = JOBS["f24"]
        wav = D.ensure_clean16k(job_id, cfg)
        x = D.load_wav16k(wav)
        windows = [(600.0, 600.0 + args.probe)]
        output["files"]["probe_f24"] = {
            "job_id": job_id,
            **compact(D.sample_speakers(x, cfg, windows=windows)),
        }
    else:
        for key in ("f24", "f00"):
            job_id = JOBS[key]
            wav = D.ensure_clean16k(job_id, cfg)
            x = D.load_wav16k(wav)
            duration = len(x) / D.SR
            body_windows = D.plan_sample_windows(
                duration, sample_minutes=args.minutes, sample_segments=args.segments,
                body_margin_s=args.body_margin)
            output["files"][key] = {
                "body_sampled": run_one(cfg, job_id, body_windows, f"{key}-body"),
            }
            if key == "f24":
                # 任务要求：f24 clean 600–1200 s 附近正文单独跑一次（600 s 单窗）
                a = min(600.0, max(0.0, duration - 600.0))
                output["files"][key]["window_600_1200"] = run_one(
                    cfg, job_id, [(a, a + 600.0)], "f24-window")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"[out] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
