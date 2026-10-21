import sys, json, wave, time
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

for t0, t1 in [(0, 15), (15, 30), (30, 60), (60, 90), (4560, 4600), (4620, 4673)]:
    seg = x[int(t0*sr):int(t1*sr)]
    st = time.time()
    s = rec.create_stream()
    s.accept_waveform(sr, seg)
    rec.decode_stream(s)
    txt = s.result.text.strip()
    rms = 20*np.log10(np.sqrt(np.mean(seg**2))+1e-10)
    print(f"[{t0:>4}-{t1:>4}s] rms={rms:6.1f}dBFS  asr={txt}")
