import numpy as np, wave
import numpy.fft as fft

def read_wav(path):
    with wave.open(path, 'rb') as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr

def db(v, eps=1e-12):
    return 10*np.log10(np.maximum(v, eps))

x, sr = read_wav(r"C:\projects\local\audio_processer\work\full_16k.wav")
m = x[:, 0]
win, hop, nfft = int(0.032*sr), int(0.016*sr), 1024
frames = np.lib.stride_tricks.sliding_window_view(m, win)[::hop]
frames = frames * np.hanning(win)
S = np.abs(fft.rfft(frames, nfft))**2
freqs = np.arange(S.shape[1]) * sr / nfft
tot = S.sum(axis=1)
order = np.argsort(tot)
quiet = order[:int(len(order)*0.10)]   # 最安静的 10%
loud  = order[-int(len(order)*0.10):]  # 最响的 10%
noise_spec = db(S[quiet].mean(axis=0) + 1e-20)
speech_spec = db(S[loud].mean(axis=0) + 1e-20)

bands = [(0,80,'0-80Hz 隆隆声'),(80,300,'80-300Hz 基频/房间'),(300,1000,'300-1kHz 语音主体'),
         (1000,4000,'1-4kHz 清晰度'),(4000,8000,'4-8kHz 齿音')]
print(f"{'band':<24}{'noise(dB)':>10}{'speech(dB)':>11}{'diff':>7}")
for lo,hi,name in bands:
    mask = (freqs>=lo)&(freqs<hi)
    nz, sp = db(S[quiet][:,mask].sum(axis=1).mean()), db(S[loud][:,mask].sum(axis=1).mean())
    print(f"{name:<24}{nz:>10.1f}{sp:>11.1f}{sp-nz:>7.1f}")

# 工频/低频峰
print("\n噪声谱峰值检查（最安静10%帧平均）:")
for f0 in [50,100,150,200,250]:
    i = np.argmin(abs(freqs-f0)); print(f"  {f0:>4} Hz: {noise_spec[i]:.1f} dB")
top = np.argsort(noise_spec)[-8:][::-1]
print("噪声谱最强的 8 个频点:", ", ".join(f"{freqs[i]:.0f}Hz({noise_spec[i]:.1f}dB)" for i in top))

# 语音/噪声逐频带信噪比 => 降噪的固有难度
print("\n逐频带 SNR (speech - noise):")
for lo,hi in [(0,80),(80,150),(150,300),(300,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)]:
    mask=(freqs>=lo)&(freqs<hi)
    nz, sp = db(S[quiet][:,mask].sum(axis=1).mean()), db(S[loud][:,mask].sum(axis=1).mean())
    print(f"  {lo:>4}-{hi:<5} Hz : {sp-nz:+.1f} dB")

np.savez(r"C:\projects\local\audio_processer\work\noise_speech_spec.npz",
         freqs=freqs, noise=noise_spec, speech=speech_spec)
print("\nsaved work/noise_speech_spec.npz")
