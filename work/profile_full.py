import numpy as np, wave, sys, json

def read_wav(path):
    with wave.open(path, 'rb') as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr

def db(v, eps=1e-10):
    return 20*np.log10(np.maximum(v, eps))

# ---------- whole-file 1s RMS ----------
x, sr = read_wav(r"C:\projects\local\audio_processer\work\full_16k.wav")
m = x[:, 0]
dur = len(m) / sr
step = sr  # 1s
n = len(m) // step
rms = np.array([np.sqrt(np.mean(m[i*step:(i+1)*step]**2)) for i in range(n)])
rdb = db(rms)
print(f"duration={dur:.1f}s ({dur/60:.1f} min), frames={n}")

print("\n[1s-RMS percentiles]")
for p in [1, 2, 5, 10, 20, 30, 50, 70, 90, 95, 99]:
    print(f"  P{p:>2} = {np.percentile(rdb, p):6.2f} dBFS")

# 直方图（粗）
hist, edges = np.histogram(rdb, bins=[-40,-35,-30,-25,-20,-15,-10,-5,0,5])
print("\n[histogram]")
for i, h in enumerate(hist):
    print(f"  {edges[i]:>4.0f}..{edges[i+1]:>4.0f} dB : {h:5d} s ({h/n*100:5.1f}%)")

noise_floor = np.percentile(rdb, 10)
speech_lvl = np.percentile(rdb, 90)
print(f"\nnoise_floor(P10)={noise_floor:.2f} dBFS, speech(P90)={speech_lvl:.2f} dBFS, SNR~{speech_lvl-noise_floor:.1f} dB")

# ---------- adaptive VAD: 能量阈值 + 停顿检测 ----------
# 以 P10 为噪声底，阈值 = max(P10+8dB, P50) 左右
thr = noise_floor + 8.0
active = rdb > thr
print(f"\nenergy VAD threshold={thr:.1f} dBFS -> 有声帧占比 {active.mean()*100:.1f}% ({active.sum()}s)")
# 连续静音段（>=1s）
gaps = []
run = 0
for a in active:
    if not a: run += 1
    else:
        if run >= 1: gaps.append(run)
        run = 0
if run: gaps.append(run)
print(f"静音段(>=1s): {len(gaps)} 段, 总时长 {sum(gaps)/60:.1f} 分钟, 最长 {max(gaps) if gaps else 0}s")

# ---------- 44.1k 频谱：编解码带宽上限 ----------
import numpy.fft as fft
s, sr44 = read_wav(r"C:\projects\local\audio_processer\work\full_44k_stereo.wav")
s = s[:, 0]
print(f"\n[44.1k source] {len(s)/sr44:.0f}s, {sr44} Hz")
for (t0, label) in [(300, '5:00'), (1800, '30:00'), (3600, '60:00')]:
    seg = s[t0*sr44 : (t0+20)*sr44]
    nfft = 8192
    spec = np.zeros(nfft//2+1); cnt = 0
    for i in range(0, len(seg)-nfft, nfft//2):
        spec += np.abs(fft.rfft(seg[i:i+nfft]*np.hanning(nfft)))**2; cnt += 1
    spec = 10*np.log10(spec/cnt+1e-12)
    freqs = np.arange(len(spec))*sr44/nfft
    peak = spec.max()
    # 找最后一个仍高于 peak-70dB 的频点（近似有效带宽）
    valid = np.where(spec > peak-70)[0]
    bw = freqs[valid[-1]] if len(valid) else 0
    print(f"  {label}: peak@{freqs[np.argmax(spec)]:.0f}Hz, 有效带宽(-70dB)≈{bw:.0f} Hz, "
          f"8k={spec[np.argmin(abs(freqs-8000))]-peak:.0f}dB 12k={spec[np.argmin(abs(freqs-12000))]-peak:.0f}dB 16k={spec[np.argmin(abs(freqs-16000))]-peak:.0f}dB")
