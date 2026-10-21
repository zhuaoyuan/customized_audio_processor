import numpy as np, wave
import numpy.fft as fft

def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32768.0
    return d.reshape(-1, ch), sr

x, sr = read_wav(r"C:\projects\local\audio_processer\work\full_16k.wav")
m = x[:, 0]

def analyze(t0, t1, label):
    seg = m[int(t0*sr):int(t1*sr)]
    nfft = 4096
    frames = np.lib.stride_tricks.sliding_window_view(seg, nfft)[::nfft//2]
    S = np.abs(fft.rfft(frames*np.hanning(nfft), axis=1))**2
    freqs = np.arange(S.shape[1])*sr/nfft
    avg = S.mean(axis=0)
    print(f"\n--- {label} ({t0}-{t1}s) ---")
    # 找 200-2000Hz 内的谱峰
    band = (freqs > 150) & (freqs < 3000)
    sub = avg[band]; subf = freqs[band]
    # 局部极大
    peaks = []
    for i in range(1, len(sub)-1):
        if sub[i] > sub[i-1] and sub[i] > sub[i+1] and sub[i] > sub.max()/20:
            peaks.append((subf[i], 10*np.log10(sub[i]/avg.max())))
    peaks.sort(key=lambda p: -p[1])
    top = peaks[:12]
    print("  最强谱峰(Hz, rel dB):", ", ".join(f"{f:.0f}({d:.0f})" for f, d in top))
    # 谐波性: 相邻峰间隔的稳定性
    fs = sorted([f for f, _ in top])
    if len(fs) > 3:
        diffs = np.diff(fs)
        print(f"  相邻峰间隔: {np.round(diffs,0)}")
    # 时变性: 各帧谱与平均谱的相关(越高越"稳定持续")
    corr = []
    for i in range(S.shape[0]):
        c = np.corrcoef(S[i, band], avg[band])[0, 1]
        corr.append(c)
    print(f"  帧谱与平均谱相关性: mean={np.mean(corr):.2f}, 稳定帧(>0.8)占比={np.mean(np.array(corr)>0.8)*100:.0f}%")

analyze(615, 625, "正文含可疑谐波带 615-625s")
analyze(600, 610, "正文对照 600-610s")
analyze(1170, 1180, "正文对照 1170-1180s")
analyze(0, 10, "片头 0-10s")

# 60Hz 工频及谐波检查（对 620s 附近）
seg = m[615*sr:625*sr]
nfft = 8192
S = np.abs(fft.rfft(seg[:nfft*20].reshape(-1, nfft)*np.hanning(nfft), axis=1))**2
freqs = np.arange(nfft//2+1)*sr/nfft
avg = S.mean(axis=0)
print("\n工频谐波检查 (615-625s):")
for f0 in [50, 60, 100]:
    lv = []
    for k in range(1, 8):
        i = np.argmin(abs(freqs - f0*k))
        lv.append(10*np.log10(avg[i]/avg.max()))
    print(f"  f0={f0}Hz 谐波相对电平(dB): {np.round(lv,1)}")
