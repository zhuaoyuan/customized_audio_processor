import wave
import numpy as np
import sherpa_onnx

MODEL = r"C:\projects\local\audio_processer\work\bench\models\sensevoice\model.int8.onnx"
TOK = r"C:\projects\local\audio_processer\work\bench\models\sensevoice\tokens.txt"
FULL = r"C:\projects\local\audio_processer\work\full_16k.wav"

def read(path):
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32768.0
    return d.reshape(-1, ch)[:, 0], sr

x, sr = read(FULL)
rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=MODEL, tokens=TOK, num_threads=8, use_itn=True, language="zh", debug=False)

def asr(seg):
    s = rec.create_stream(); s.accept_waveform(sr, seg); rec.decode_stream(s)
    return s.result.text.strip()

print("=== 0-180s 每 10s（只打印非空）===")
nonempty = 0
for t0 in range(0, 180, 10):
    seg = x[t0*sr:(t0+10)*sr]
    txt = asr(seg)
    if len(txt) > 1:
        nonempty += 1
        print(f"[{t0:>3}-{t0+10:>3}s] {txt}")
print(f"非空窗口: {nonempty}/18")
