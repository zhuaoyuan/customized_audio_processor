"""并行吞吐测试：把 78 分钟音频切成 N 份，N 个进程并行跑 SenseVoice，测总墙钟时间。"""
import os
import subprocess
import sys
import time

import numpy as np

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PY = sys.executable
WAV = os.path.join(ROOT, "work", "full_16k.wav")
N = 4
THREADS = 4


def main():
    outdir = os.path.join(BENCH, "parallel")
    os.makedirs(outdir, exist_ok=True)
    # 切片
    total = 4673.294286
    part_dur = total / N
    parts = []
    for i in range(N):
        p = os.path.join(outdir, f"part{i}.wav")
        if not os.path.exists(p):
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", f"{i*part_dur:.3f}", "-t", f"{part_dur:.3f}",
                   "-i", WAV, "-c:a", "pcm_s16le", p]
            subprocess.run(cmd, check=True)
        parts.append(p)
        print(f"[split] {os.path.basename(p)} {os.path.getsize(p)/1024/1024:.1f} MB")

    procs = []
    t0 = time.perf_counter()
    for i, p in enumerate(parts):
        out = os.path.join(outdir, f"asr_part{i}.json")
        log = open(os.path.join(outdir, f"asr_part{i}.log"), "w", encoding="utf-8")
        pr = subprocess.Popen([PY, os.path.join(BENCH, "run_asr_sensevoice.py"),
                               "--wav", p, "--threads", str(THREADS), "--chunk", "30",
                               "--out", out], stdout=log, stderr=subprocess.STDOUT)
        procs.append((pr, log, p))
    for pr, log, _ in procs:
        rc = pr.wait()
        log.close()
        if rc != 0:
            print(f"[warn] 进程退出码 {rc}")
    wall = time.perf_counter() - t0
    print(f"[parallel] N={N} 进程 x {THREADS} 线程  总墙钟 = {wall:.1f}s  "
          f"RTF(整片) = {wall/total:.4f}  -> 78min 用时 {wall/60:.2f} min")
    print(f"[parallel] 串行外壳估计：{132.45:.0f}s（见 asr_sv_full78min 实测）; "
          f"加速比 = {132.45/wall:.2f}x")

    # 合并文本
    import json
    texts = []
    for i in range(N):
        p = os.path.join(outdir, f"asr_part{i}.json")
        with open(p, encoding="utf-8") as f:
            texts.append(json.load(f)["text"])
    full = "".join(texts)
    res = {"N": N, "threads": THREADS, "wall_s": wall, "rtf": wall / total,
           "chars": len(full), "text": full}
    with open(os.path.join(BENCH, "parallel_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] chars={len(full)}")


if __name__ == "__main__":
    main()
