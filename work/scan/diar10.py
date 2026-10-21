# -*- coding: utf-8 -*-
"""Step 5: 说话人结构抽样（sherpa-onnx OfflineSpeakerDiarization，10 分钟正文/文件）。

用法: python diar10.py --only f20_xifang20140219 --threads 4
模型复用 work/diar/models（只读）。输出 work/scan/diar/<short>.json
"""
import argparse
import json
import os
import time

import numpy as np

ROOT = r"C:\projects\local\audio_processer\work"
SCAN = os.path.join(ROOT, "scan")
DIAR = os.path.join(ROOT, "diar", "models")
SEG = os.path.join(DIAR, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
EMB = os.path.join(DIAR, "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
OUTD = os.path.join(SCAN, "diar")
os.makedirs(OUTD, exist_ok=True)
SR = 16000
WIN = 600.0
MIN_EFFECTIVE_SHARE = 0.05   # 有效讲话人：占语音 >=5% 的簇

SHORTS = ["f00_xifang20140227", "f20_lixiangguo20141219", "f24_xingershangxue",
          "f04_lixiangguo4", "f05_wulixue", "f06_lunyu"]


def read_slice(path, start_s, dur_s):
    from scipy.io import wavfile
    sr, d = wavfile.read(path, mmap=True)
    a = int(start_s * sr)
    b = a + int(dur_s * sr)
    return np.ascontiguousarray(d[a:b], dtype=np.float32), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    todo = [s for s in (args.only.split(",") if args.only else SHORTS) if s]

    import sherpa_onnx
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG),
            num_threads=args.threads, debug=False, provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=EMB, num_threads=args.threads, debug=False, provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=args.threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    assert cfg.validate()
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    probe = json.load(open(os.path.join(SCAN, "probe.json"), encoding="utf-8"))
    out = {}
    for short in todo:
        vad = json.load(open(os.path.join(SCAN, "vad", short + ".json"), encoding="utf-8"))
        wav = probe["files"][short]["wav"]
        dur = probe["files"][short]["wav_dur_s"]
        start = max(vad["edges"]["first_speech_s"] + 60.0, dur * 0.05)
        start = min(start, max(0.0, dur - WIN - 20.0))
        x, sr = read_slice(wav, start, WIN)
        t0 = time.perf_counter()
        result = sd.process(x)
        dt = time.perf_counter() - t0
        segs = result.sort_by_start_time()
        per = {}
        rows = []
        for s in segs:
            d = {"start": round(float(s.start), 3), "end": round(float(s.end), 3),
                 "dur": round(float(s.end - s.start), 3), "speaker": int(s.speaker)}
            rows.append(d)
            per[d["speaker"]] = per.get(d["speaker"], 0.0) + d["dur"]
        speech_total = sum(per.values())
        main_spk = max(per, key=per.get) if per else None
        nonmain = speech_total - (per.get(main_spk, 0.0) if main_spk is not None else 0.0)
        ordered = sorted(per.items(), key=lambda kv: -kv[1])
        effective = [k for k, v in ordered if speech_total and v / speech_total >= MIN_EFFECTIVE_SHARE]
        # 主讲人之外“有效”说话人的语音秒数
        nonmain_eff = speech_total - per.get(main_spk, 0.0) if main_spk is not None else 0.0
        res = {
            "short": short, "wav": wav, "start_s": round(start, 1),
            "dur_s": round(len(x) / sr, 2), "threshold": args.threshold,
            "num_speakers": int(result.num_speakers),
            "num_effective_speakers": len(effective),
            "effective_share_min": MIN_EFFECTIVE_SHARE,
            "n_segments": len(rows), "speech_total_s": round(speech_total, 1),
            "speech_ratio": speech_total / (len(x) / sr),
            "per_speaker_s": {str(k): round(v, 1) for k, v in ordered},
            "main_speaker": main_spk,
            "main_share_of_speech": (per[main_spk] / speech_total) if speech_total else None,
            "nonmain_speech_s": round(nonmain, 1),
            "process_s": round(dt, 1), "rtf": dt / (len(x) / sr),
            "segments": rows,
        }
        out[short] = res
        json.dump(res, open(os.path.join(OUTD, short + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"[diar] {short}: raw_spk={result.num_speakers} eff_spk={len(effective)} "
              f"main_share={res['main_share_of_speech']*100:.1f}% nonmain={nonmain:.0f}s "
              f"rtf={dt/(len(x)/sr):.3f} per={res['per_speaker_s']}", flush=True)
        with open(os.path.join(OUTD, short + ".rttm"), "w", encoding="utf-8") as f:
            for d in rows:
                f.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> S%d <NA> <NA>\n"
                        % (short, d["start"], d["dur"], d["speaker"]))
    # 汇总（并行运行时各进程只写自己的，合并由 build_report 读取 per-file json）
    print("[diar] done", list(out), flush=True)


if __name__ == "__main__":
    main()
