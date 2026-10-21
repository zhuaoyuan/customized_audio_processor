# -*- coding: utf-8 -*-
"""排查 arnndn 在 16k 与 48k（再下采样）下的行为：rnnoise 模型要求 48kHz。"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import AE, db, frame_power, read_wav, write_wav  # noqa: E402

RNN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench", "models", "rnnoise")
OUT = os.path.join(AE, "denoise")


def run(src, dst, af, cwd=None):
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-af", af, "-c:a", "pcm_f32le", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return dst


def basic(path):
    _, x = read_wav(path)
    p, _ = frame_power(x, 16000)
    return {"n": len(x), "dur_s": round(len(x) / 16000, 2),
            "rms_dbfs": round(float(db(np.mean(x ** 2))), 2),
            "peak": round(float(np.max(np.abs(x))), 4),
            "nan": int(np.sum(~np.isfinite(x))),
            "zero_ratio": round(float(np.mean(np.abs(x) < 1e-9)), 4)}


def main():
    base = os.path.join(OUT, "xes_600_900_orig.wav")
    src48 = os.path.join(OUT, "xes_600_900_orig48.wav")
    run(base, src48, "aresample=48000")
    out48 = os.path.join(OUT, "xes_600_900_arnndn48_raw.wav")
    run(src48, out48, "highpass=f=80,arnndn=m=sh.rnnn", cwd=RNN)
    # 48k -> 16k
    out16 = os.path.join(OUT, "xes_600_900_arnndn48_16k.wav")
    run(out48, out16, "aresample=16000")
    ref = basic(base)
    a16 = basic(os.path.join(OUT, "xes_600_900_arnndn.wav"))
    a48 = basic(out16)
    res = {"orig_16k": ref, "arnndn_16k_direct": a16, "arnndn_48k_then_16k": a48,
           "note": "rnnoise 模型仅支持 48kHz；16k 直接送入会得到 NaN/静音"}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    with open(os.path.join(AE, "arnndn_rate_test.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
