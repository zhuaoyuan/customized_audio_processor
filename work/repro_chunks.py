import wave
import numpy as np
import sherpa_onnx

MODEL = r"C:\projects\local\audio_processer\work\bench\models\sensevoice\model.int8.onnx"
TOK = r"C:\projects\local\audio_processer\work\bench\models\sensevoice\tokens.txt"
FULL = r"C:\projects\local\audio_processer\work\full_16k.wav"

with wave.open(FULL, "rb") as w:
    n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
    d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32768.0
x = d.reshape(-1, ch)[:, 0]

rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=MODEL, tokens=TOK, num_threads=8, use_itn=True, language="zh", debug=False)

def asr(seg):
    s = rec.create_stream(); s.accept_waveform(sr, seg); rec.decode_stream(s)
    return s.result.text

# 复现 bench 的 30s 分块，逐块打印
print("=== 30s 分块 0-600s 逐块文本（复现 bench 的 full78 跑法）===")
for t0 in range(0, 600, 30):
    seg = x[t0*sr:(t0+30)*sr]
    print(f"[{t0:>3}-{t0+30:>3}] {asr(seg)!r}")
