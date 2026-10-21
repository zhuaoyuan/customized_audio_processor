# -*- coding: utf-8 -*-
"""头/尾边缘内容复核：对每文件片头/片尾被判定为非语音的区间，
给 5s RMS 曲线 + SenseVoice 转写，确认“可裁”不是 VAD 漏检。

输出 work/scan/edges.json
"""
import json
import os

import numpy as np

ROOT = r"C:\projects\local\audio_processer\work"
SCAN = os.path.join(ROOT, "scan")
MODEL = os.path.join(ROOT, "bench", "models", "sensevoice", "model.int8.onnx")
TOKENS = os.path.join(ROOT, "bench", "models", "sensevoice", "tokens.txt")
SR = 16000
SHORTS = ["f00_xifang20140227", "f20_lixiangguo20141219", "f24_xingershangxue",
          "f04_lixiangguo4", "f05_wulixue", "f06_lunyu"]


def read_slice(path, start_s, dur_s):
    from scipy.io import wavfile
    sr, d = wavfile.read(path, mmap=True)
    a = int(max(0.0, start_s) * sr)
    b = min(len(d), a + int(dur_s * sr))
    return np.ascontiguousarray(d[a:b], dtype=np.float32), sr


def rms_curve(x, sr, step=5.0):
    out = []
    n = int(step * sr)
    for a in range(0, len(x), n):
        seg = np.asarray(x[a:a + n], dtype=np.float64)
        out.append(round(float(20 * np.log10(max(np.sqrt(np.mean(seg ** 2)), 1e-9))), 1))
    return out


def main():
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=MODEL, tokens=TOKENS, num_threads=4, use_itn=True, language="zh", debug=False)
    probe = json.load(open(os.path.join(SCAN, "probe.json"), encoding="utf-8"))
    out = {}
    for s in SHORTS:
        vad = json.load(open(os.path.join(SCAN, "vad", s + ".json"), encoding="utf-8"))
        wav = probe["files"][s]["wav"]
        dur = probe["files"][s]["wav_dur_s"]
        head = vad["edges"]["head_trim_s"]
        tail = vad["edges"]["tail_trim_s"]
        rec_d = {"head_trim_s": head, "tail_trim_s": tail}
        if head >= 2.0:
            x, sr = read_slice(wav, 0, min(head + 5.0, 140.0))
            st = rec.create_stream(); st.accept_waveform(sr, x); rec.decode_stream(st)
            rec_d["head"] = {"dur_s": round(len(x) / sr, 1), "rms_5s": rms_curve(x, sr),
                             "text": st.result.text.strip()}
        if tail >= 2.0:
            a = max(0.0, dur - min(tail + 5.0, 140.0))
            x, sr = read_slice(wav, a, dur - a)
            st = rec.create_stream(); st.accept_waveform(sr, x); rec.decode_stream(st)
            rec_d["tail"] = {"start_s": round(a, 1), "dur_s": round(len(x) / sr, 1),
                             "rms_5s": rms_curve(x, sr), "text": st.result.text.strip()}
        out[s] = rec_d
        print(f"[edge] {s}: head={head:.0f}s tail={tail:.0f}s "
              f"head_text={rec_d.get('head', {}).get('text', '')[:60]!r} "
              f"tail_text={rec_d.get('tail', {}).get('text', '')[:60]!r}", flush=True)
    json.dump(out, open(os.path.join(SCAN, "edges.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[edge] done")


if __name__ == "__main__":
    main()
