import wave
import numpy as np
import numpy.fft as fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def read(path):
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32768.0
    return d.reshape(-1, ch)[:, 0], sr

x, sr = read(r"C:\projects\local\audio_processer\work\full_16k.wav")

def env_mod_spectrum(t0, t1, label):
    seg = x[int(t0*sr):int(t1*sr)]
    # 10 ms 帧能量包络
    hop = int(0.01*sr); win = int(0.025*sr)
    idx = np.arange(0, len(seg)-win, hop)
    e = np.array([np.sqrt(np.mean(seg[i:i+win]**2)) for i in idx])
    e = e - e.mean()
    E = np.abs(fft.rfft(e*np.hanning(len(e))))
    f = np.arange(len(E))*1.0/(hop/sr)/len(E)
    band = (f > 0.5) & (f < 20)
    peak = f[band][np.argmax(E[band])]
    print(f"{label:<28} 包络调制峰 = {peak:.2f} Hz   "
          f"(2-8Hz 占比 {(E[(f>=2)&(f<=8)].sum()/max(E[band].sum(),1e-9)*100):.1f}%)")

env_mod_spectrum(0, 300, "片头 0-300s")
env_mod_spectrum(300, 360, "疑似讲课开始 300-360s")
env_mod_spectrum(600, 900, "正文 600-900s")
env_mod_spectrum(2400, 2700, "正文 2400-2700s")
env_mod_spectrum(4560, 4673, "片尾 4560-4673s")

# 谱图：0-120s，动态范围 60dB
def show(ax, t0, t1, title):
    seg = x[int(t0*sr):int(t1*sr)]
    nfft, hop = 1024, 160
    frames = np.lib.stride_tricks.sliding_window_view(seg, nfft)[::hop]
    S = 20*np.log10(np.abs(fft.rfft(frames*np.hanning(nfft), axis=1))+1e-8)
    S -= S.max()
    ax.imshow(S.T, origin="lower", aspect="auto", extent=[0, (t1-t0), 0, sr/2],
              cmap="magma", vmin=-60, vmax=0)
    ax.set_ylim(0, 8000); ax.set_title(title, fontsize=10)
    ax.set_ylabel("Hz")

fig, axes = plt.subplots(3, 1, figsize=(13, 8), dpi=140)
show(axes[0], 0, 60,   "0–60s（无可识别讲课语音，持续能量 −16 dBFS）")
show(axes[1], 60, 120, "60–120s")
show(axes[2], 300, 360, "300–360s（可识别讲课语音）")
for ax in axes: ax.set_xlabel("s")
fig.tight_layout()
fig.savefig(r"C:\projects\local\audio_processer\docs\figures\fig4_head_vs_lecture.png")
print("saved fig4")
