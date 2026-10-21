import json, wave, numpy as np, os, sys
os.environ.setdefault("HF_HOME", r"C:\projects\local\audio_processer\work\bench\models\hf_home")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

WAV = r"C:\projects\local\audio_processer\work\timeline\_cache\24_形而上学_16k.wav"
TR = r"C:\projects\local\audio_processer\work\timeline\24_形而上学\transcript.json"
T0, T1 = 600.0, 1200.0

with wave.open(WAV, "rb") as w:
    sr = w.getframerate()
    w.setpos(int(T0 * sr))
    n = int((T1 - T0) * sr)
    raw = w.readframes(n)
x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

from faster_whisper import WhisperModel
m = WhisperModel("base", device="cpu", compute_type="int8",
                 download_root=r"C:\projects\local\audio_processer\work\bench\models\whisper")
segs, info = m.transcribe(x / 1.0, language="zh", beam_size=5, word_timestamps=True)
words = []
for s in segs:
    for w in (s.words or []):
        words.append((w.start + T0, w.end + T0, w.word))
print("ref words:", len(words))
import json as _json
_json.dump(words, open(r"C:\projects\local\audio_processer\work\cost\whisper_words_600_1200.json", "w", encoding="utf-8"), ensure_ascii=False)

tr = json.load(open(TR, encoding="utf-8"))
if isinstance(tr, dict):
    tr = tr["segments"]
our = [s for s in tr if T0 <= s["start"] < T1]
print("our segments in window:", len(our))

def boundary_errors(ss, ee):
    errs_s, errs_e = [], []
    for s in our:
        mid = 0.5 * (s["start"] + s["end"])
        # 段内词：中点落在段内
        inside = [w for w in words if s["start"] - 0.5 <= 0.5 * (w[0] + w[1]) <= s["end"] + 0.5]
        if not inside:
            continue
        errs_s.append(s["start"] - inside[0][0])
        errs_e.append(s["end"] - inside[-1][1])
    return np.array(errs_s), np.array(errs_e)

es, ee = boundary_errors(None, None)
def stat(a):
    return dict(n=int(len(a)), med_s=float(np.median(a)), p90_abs=float(np.percentile(np.abs(a), 90)),
                mean=float(np.mean(a)), std=float(np.std(a)))
print("start err:", json.dumps(stat(es), ensure_ascii=False))
print("end   err:", json.dumps(stat(ee), ensure_ascii=False))
json.dump({"n_words": len(words), "n_segs": len(our), "start": stat(es), "end": stat(ee)},
          open(r"C:\projects\local\audio_processer\work\cost\ts_eval.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
