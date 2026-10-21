import wave
import numpy as np
import numpy.fft as fft

def read(path):
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32768.0
    return d.reshape(-1, ch)[:, 0], sr

x, sr = read(r"C:\projects\local\audio_processer\work\full_16k.wav")

def stats(t0, t1, label):
    seg = x[int(t0*sr):int(t1*sr)]
    nfft, hop = 512, 160
    frames = np.lib.stride_tricks.sliding_window_view(seg, nfft)[::hop]
    S = np.abs(fft.rfft(frames*np.hanning(nfft), axis=1))**2 + 1e-12
    freqs = np.arange(S.shape[1])*sr/nfft
    band = (freqs >= 100) & (freqs <= 6000)
    Sb = S[:, band]
    # 谱平坦度
    flat = np.exp(np.mean(np.log(Sb), axis=1)) / np.mean(Sb, axis=1)
    # 谱质心
    cent = (Sb*freqs[band]).sum(axis=1) / Sb.sum(axis=1)
    # 4-8kHz 相对能量
    hf = S[:, (freqs>=4000)&(freqs<=7000)].sum(axis=1) / S[:, (freqs>=100)&(freqs<=7000)].sum(axis=1)
    rms = 20*np.log10(np.sqrt(np.mean(seg**2))+1e-10)
    print(f"{label:<22} rms={rms:6.1f}  平坦度中位={np.median(flat):.3f}  "
          f"谱质心={np.median(cent):6.0f}Hz  4-7k占比={np.median(hf)*100:4.1f}%  帧数={len(flat)}")

for t0, t1, lab in [(0,60,"片头 0-60s"), (60,300,"片头 60-300s"), (300,360,"讲课 300-360s"),
                    (600,900,"讲课 600-900s"), (1800,2100,"讲课 1800-2100s"),
                    (4500,4560,"讲课 4500-4560s"), (4560,4673,"片尾 4560-4673s")]:
    stats(t0, t1, lab)
