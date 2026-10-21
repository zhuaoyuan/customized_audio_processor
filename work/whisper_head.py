import wave
import numpy as np

MODEL_DIR = r"C:\projects\local\audio_processer\work\bench\models\whisper"
FULL = r"C:\projects\local\audio_processer\work\full_16k.wav"

with wave.open(FULL, "rb") as w:
    n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
    d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32768.0
x = d.reshape(-1, ch)[:, 0]

from faster_whisper import WhisperModel
m = WhisperModel("base", device="cpu", compute_type="int8", download_root=MODEL_DIR)
for t0, t1 in [(0, 60), (60, 120), (480, 540), (540, 600)]:
    seg = x[int(t0*sr):int(t1*sr)]
    segs, _ = m.transcribe(seg, language="zh", beam_size=5)
    txt = "".join(s.text for s in segs).strip()
    print(f"[{t0:>3}-{t1:>3}s] {txt!r}")
