import numpy as np, wave

def read_wav(path):
    with wave.open(path, 'rb') as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr

def frame_rms(x, sr, win=0.025, hop=0.010):
    w, h = int(sr*win), int(sr*hop)
    idx = np.arange(0, max(len(x)-w, 0), h)
    return np.array([np.sqrt(np.mean(x[i:i+w]**2)) for i in idx])

def db(v, eps=1e-10):
    return 20*np.log10(np.maximum(v, eps))

print("="*64)
print("STEREO ANALYSIS (60s @600s)")
x, sr = read_wav(r"C:\projects\local\audio_processer\work\samples\stereo_600_60.wav")
L, R = x[:,0], x[:,1]
print(f"sample_rate={sr}, dur={len(L)/sr:.1f}s")
l_rms, r_rms = db(np.sqrt(np.mean(L**2))), db(np.sqrt(np.mean(R**2)))
print(f"L RMS={l_rms:.2f} dBFS   R RMS={r_rms:.2f} dBFS")
print(f"channel correlation = {np.corrcoef(L, R)[0,1]:.4f}")
diff_db = db(np.sqrt(np.mean((L-R)**2)))
print(f"L-R residual RMS = {diff_db:.2f} dBFS (L-R vs L: {diff_db-l_rms:+.1f} dB)")
print("=> " + ("dual-mono，无空间信息可用" if diff_db-l_rms < -25 else "存在立体声差异"))

print()
print("="*64)
print("MONO 16k ANALYSIS")
m, sr2 = read_wav(r"C:\projects\local\audio_processer\work\samples\mono_600_60.wav")
m = m[:,0]
rdb = db(frame_rms(m, sr2))
for p in [1,5,10,20,30,50,70,90,95,99]:
    print(f"  P{p:>2} = {np.percentile(rdb,p):6.2f} dBFS")
noise_floor, speech = np.percentile(rdb,10), np.percentile(rdb,90)
print(f"\n  noise_floor(P10)={noise_floor:.2f}  speech(P90)={speech:.2f}  近似SNR={speech-noise_floor:.1f} dB")
print(f"  > noise+10dB : {np.mean(rdb>noise_floor+10)*100:.1f}% (近似有声占比)")
print(f"  < noise+3dB  : {np.mean(rdb<noise_floor+3)*100:.1f}% (近似纯噪声)")

import numpy.fft as fft
n = 4096; spec = np.zeros(n//2+1); cnt = 0
for i in range(0, len(m)-n, n//2):
    spec += np.abs(fft.rfft(m[i:i+n]*np.hanning(n)))**2; cnt += 1
spec = 10*np.log10(spec/cnt+1e-12); freqs = np.arange(len(spec))*sr2/n; peak = spec.max()
print("\n  长时平均频谱（相对峰值）:")
for f in [100,200,500,1000,2000,4000,6000,7500,8000]:
    print(f"    {f:>5} Hz : {spec[np.argmin(np.abs(freqs-f))]-peak:7.1f} dB")
below = np.where(spec-peak < -60)[0]
cand = below[below > np.argmin(np.abs(freqs-3000))]
if len(cand): print(f"  -60dB 滚降点 ~{freqs[cand[0]]:.0f} Hz（编解码带宽上限/低通）")
