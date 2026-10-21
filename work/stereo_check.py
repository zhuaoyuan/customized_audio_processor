import wave
import numpy as np

# 1) 逐段检查立体声 L/R 是否同源（判断片头是否来自不同采集）
def read_stereo(path):
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32768.0
    return d.reshape(-1, ch), sr

s, sr = read_stereo(r"C:\projects\local\audio_processer\work\full_44k_stereo.wav")
print("=== 分段 L/R 相关性（44.1k 立体声源）===")
for t0, t1, lab in [(0,300,"片头 0-300s"),(300,480,"片头 300-480s"),(480,900,"讲课 480-900s"),
                    (2400,2700,"讲课 2400-2700s"),(4500,4673,"片尾 4500-4673s")]:
    seg = s[int(t0*sr):int(t1*sr)]
    L, R = seg[:,0], seg[:,1]
    c = np.corrcoef(L, R)[0,1]
    rms = 20*np.log10(np.sqrt(np.mean(L**2))+1e-10)
    print(f"{lab:<20} corr={c:.6f}  L rms={rms:6.1f} dBFS")
