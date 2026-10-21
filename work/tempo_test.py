import wave
import numpy as np
import numpy.fft as fft

def read(path):
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32768.0
    return d.reshape(-1, ch)[:, 0], sr

x, sr = read(r"C:\projects\local\audio_processer\work\full_16k.wav")

def tempo_test(t0, t1, label):
    seg = x[int(t0*sr):int(t1*sr)]
    # 20ms 帧能量 → onset strength（谱通量简化版：能量正差分）
    hop = int(0.02*sr); win = int(0.04*sr)
    idx = np.arange(0, len(seg)-win, hop)
    e = np.array([np.sqrt(np.mean(seg[i:i+win]**2)) for i in idx])
    e = np.log(e + 1e-8)
    onset = np.maximum(np.diff(e), 0)
    onset -= onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset)-1:]
    ac /= (ac[0] + 1e-12)
    fps = 1.0/(hop/sr)   # 50 fps
    lag_min, lag_max = int(0.3*fps), int(2.0*fps)   # 0.3-2.0s 拍周期
    band = ac[lag_min:lag_max]
    lag = lag_min + int(np.argmax(band))
    # 周期性强度：自相关峰 vs 中位数
    strength = band.max() / (np.median(band) + 1e-9)
    print(f"{label:<20} 拍周期峰={lag/fps:.2f}s ({60/(lag/fps):5.1f} BPM)  峰/中位={strength:5.1f}  "
          f"onset均值={onset.mean():+.4f}")

for t0, t1, lab in [(0,120,"片头 0-120s"), (120,300,"片头 120-300s"), (300,480,"片头 300-480s"),
                    (480,600,"讲课 480-600s"), (600,900,"讲课 600-900s"),
                    (2400,2700,"讲课 2400-2700s"), (4560,4673,"片尾 4560-4673s")]:
    tempo_test(t0, t1, lab)
