# -*- coding: utf-8 -*-
"""Step 4: 每文件 60s 正文样本 SenseVoice ASR（字/秒、前两句、课程体系判断线索）。

样本窗选择：从 VAD 结果中滑动 60s 窗（步长 10s，避开片头前 5%），取语音覆盖最大者。
输出 work/scan/asr/<short>.json
"""
import json
import os
import re
import time
import wave

import numpy as np

ROOT = r"C:\projects\local\audio_processer\work"
SCAN = os.path.join(ROOT, "scan")
MODEL = os.path.join(ROOT, "bench", "models", "sensevoice", "model.int8.onnx")
TOKENS = os.path.join(ROOT, "bench", "models", "sensevoice", "tokens.txt")
OUTD = os.path.join(SCAN, "asr")
os.makedirs(OUTD, exist_ok=True)
SR = 16000

SHORTS = ["f00_xifang20140227", "f20_lixiangguo20141219", "f24_xingershangxue",
          "f04_lixiangguo4", "f05_wulixue", "f06_lunyu"]


def read_slice(path, start_s, dur_s):
    from scipy.io import wavfile
    sr, d = wavfile.read(path, mmap=True)
    a = int(start_s * sr)
    b = a + int(dur_s * sr)
    return np.ascontiguousarray(d[a:b], dtype=np.float32), sr


def pick_window(vad, dur):
    segs = vad["segments"]
    lo = max(60.0, vad["edges"]["first_speech_s"] + 30.0, dur * 0.05)
    hi = dur - 60.5
    best, best_s = lo, -1.0
    s = lo
    while s <= hi:
        cov = 0.0
        for a, b in segs:
            cov += max(0.0, min(b, s + 60) - max(a, s))
            if cov >= 59.0:
                break
        if cov > best_s:
            best_s, best = cov, s
        s += 10.0
    return best, best_s


def cn_len(t):
    """有效字数：CJK 汉字 + 拉丁字母数字，去标点空格。"""
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", t))


def split_sents(t):
    parts = [x.strip() for x in re.split(r"[。！？!?；;\n]+", t)]
    return [x for x in parts if x]


def main():
    import sherpa_onnx
    t0 = time.perf_counter()
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=MODEL, tokens=TOKENS, num_threads=8, use_itn=True, language="zh", debug=False)
    print(f"[asr] recognizer loaded in {time.perf_counter()-t0:.2f}s", flush=True)

    out = {}
    for short in SHORTS:
        vad = json.load(open(os.path.join(SCAN, "vad", short + ".json"), encoding="utf-8"))
        probe = json.load(open(os.path.join(SCAN, "probe.json"), encoding="utf-8"))
        wav = probe["files"][short]["wav"]
        dur = probe["files"][short]["wav_dur_s"]
        start, cov = pick_window(vad, dur)
        x, sr = read_slice(wav, start, 60.0)
        t0 = time.perf_counter()
        st = rec.create_stream()
        st.accept_waveform(sr, x)
        rec.decode_stream(st)
        dt = time.perf_counter() - t0
        text = st.result.text
        nl = cn_len(text)
        sents = split_sents(text)
        out[short] = {
            "wav": wav, "sample_start_s": round(start, 1), "sample_dur_s": round(len(x) / sr, 2),
            "vad_speech_in_window_s": round(cov, 2), "decode_s": round(dt, 3),
            "text": text, "chars": nl, "chars_per_s": nl / (len(x) / sr),
            "first2_sentences": sents[:2], "n_sentences": len(sents),
        }
        print(f"[asr] {short}: t={start:.0f}s chars={nl} cps={nl/(len(x)/sr):.2f} "
              f"| {text[:40]}", flush=True)

    json.dump(out, open(os.path.join(SCAN, "asr60.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for short, d in out.items():
        json.dump(d, open(os.path.join(OUTD, short + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print("[asr] done")


if __name__ == "__main__":
    main()
