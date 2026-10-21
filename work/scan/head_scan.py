# -*- coding: utf-8 -*-
"""片头/片尾 ASR 复核：30 s 分块转写，用“字符率”判断真实讲课起点（VAD 会把音乐误判为语音）。

输出 work/scan/head_scan.json
"""
import json
import os
import re

import numpy as np

ROOT = r"C:\projects\local\audio_processer\work"
SCAN = os.path.join(ROOT, "scan")
MODEL = os.path.join(ROOT, "bench", "models", "sensevoice", "model.int8.onnx")
TOKENS = os.path.join(ROOT, "bench", "models", "sensevoice", "tokens.txt")
SR = 16000
CH = 30.0
HEAD_SCAN = 600.0
TAIL_SCAN = 150.0
SHORTS = ["f00_xifang20140227", "f20_lixiangguo20141219", "f24_xingershangxue",
          "f04_lixiangguo4", "f05_wulixue", "f06_lunyu"]


def read_slice(path, start_s, dur_s):
    from scipy.io import wavfile
    sr, d = wavfile.read(path, mmap=True)
    a = int(max(0.0, start_s) * sr)
    b = min(len(d), a + int(dur_s * sr))
    return np.ascontiguousarray(d[a:b], dtype=np.float32), sr


def cn_len(t):
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", t))


def scan(rec, wav, a0, a1):
    out = []
    t = a0
    while t < a1 - 1.0:
        dur = min(CH, a1 - t)
        x, sr = read_slice(wav, t, dur)
        st = rec.create_stream()
        st.accept_waveform(sr, x)
        rec.decode_stream(st)
        txt = st.result.text.strip()
        n = cn_len(txt)
        out.append({"t0": round(t, 1), "t1": round(t + dur, 1), "chars": n,
                    "cps": round(n / dur, 2), "text": txt})
        t += CH
    return out


def first_teaching(chunks, need_chars=60, win=3):
    """最早满足：连续 win 个块（90 s）字符数 >= need_chars 的时刻。"""
    for i in range(0, len(chunks) - win + 1):
        s = sum(c["chars"] for c in chunks[i:i + win])
        if s >= need_chars:
            return chunks[i]["t0"]
    return None


def last_teaching(chunks, need_chars=40, win=2):
    for i in range(len(chunks) - win, -1, -1):
        s = sum(c["chars"] for c in chunks[i:i + win])
        if s >= need_chars:
            return chunks[min(i + win - 1, len(chunks) - 1)]["t1"]
    return None


def main():
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=MODEL, tokens=TOKENS, num_threads=8, use_itn=True, language="zh", debug=False)
    probe = json.load(open(os.path.join(SCAN, "probe.json"), encoding="utf-8"))
    out = {}
    for s in SHORTS:
        wav = probe["files"][s]["wav"]
        dur = probe["files"][s]["wav_dur_s"]
        head = scan(rec, wav, 0.0, min(HEAD_SCAN, dur))
        tail_start = max(0.0, dur - TAIL_SCAN)
        tail = scan(rec, wav, tail_start, dur)
        hs = first_teaching(head)
        te = last_teaching(tail)
        out[s] = {
            "dur_s": dur,
            "head_chunks": head,
            "tail_chunks": tail,
            "teaching_start_asr_s": hs,
            "teaching_end_asr_s": te,
            "head_trim_asr_s": hs,
            "tail_trim_asr_s": (dur - te) if te is not None else None,
        }
        A = half = [c["cps"] for c in head[:6]]
        print(f"[head] {s}: asr_start={hs} asr_end={te} "
              f"cps0-180s={[round(v,2) for v in A]}", flush=True)
    json.dump(out, open(os.path.join(SCAN, "head_scan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[head] done")


if __name__ == "__main__":
    main()
