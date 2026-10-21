import wave, time
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

print("=== 片头 0-300s 每 30s ===")
for t0 in range(0, 300, 30):
    seg = x[t0*sr:(t0+30)*sr]
    rms = 20*np.log10(np.sqrt(np.mean(seg**2))+1e-10)
    print(f"[{t0:>3}-{t0+30:>3}s] rms={rms:6.1f}  {asr(seg)}")

print("\n=== 片尾 4400-4673s 每 30s ===")
for t0 in range(4400, 4673, 30):
    t1 = min(t0+30, 4673)
    seg = x[t0*sr:t1*sr]
    rms = 20*np.log10(np.sqrt(np.mean(seg**2))+1e-10)
    print(f"[{t0:>4}-{t1:>4}s] rms={rms:6.1f}  {asr(seg)}")
