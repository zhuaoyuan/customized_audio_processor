import numpy as np, wave, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for cand in ["Microsoft YaHei", "SimHei", "DengXian"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return d.reshape(-1, ch), sr

import os
OUT = r"C:\projects\local\audio_processer\docs\figures"
os.makedirs(OUT, exist_ok=True)

# ---------- Fig 1: 全片 1s RMS 时间轴 ----------
x, sr = read_wav(r"C:\projects\local\audio_processer\work\full_16k.wav")
m = x[:, 0]
n = len(m) // sr
rms = np.array([np.sqrt(np.mean(m[i*sr:(i+1)*sr]**2)) for i in range(n)])
rdb = 20*np.log10(np.maximum(rms, 1e-10))
t = np.arange(n)/60.0

fig, ax = plt.subplots(figsize=(14, 4.2), dpi=140)
ax.plot(t, rdb, lw=0.5, color="#1f77b4")
p = np.percentile(rdb, [10, 50, 90])
for v, c, lab in [(p[0], "#d62728", f"P10 底噪 {p[0]:.1f} dBFS"),
                  (p[1], "#2ca02c", f"P50 {p[1]:.1f} dBFS"),
                  (p[2], "#ff7f0e", f"P90 {p[2]:.1f} dBFS")]:
    ax.axhline(v, color=c, lw=0.9, ls="--", alpha=0.8, label=lab)
ax.axvspan(0, 0.5, color="#9467bd", alpha=0.18)
ax.axvspan(77.4, 77.9, color="#9467bd", alpha=0.18)
ax.annotate("片头非语音", xy=(0.25, -8), ha="center", fontsize=8, color="#9467bd")
ax.annotate("片尾非语音", xy=(77.65, -8), ha="center", fontsize=8, color="#9467bd")
ax.set_xlabel("时间 (分钟)"); ax.set_ylabel("每 1 秒 RMS (dBFS)")
ax.set_title("全片能量时间轴：77.9 分钟 · 无低于 -35 dBFS 的静音段（持续底噪）")
ax.set_ylim(-36, 2); ax.legend(loc="lower right", fontsize=8, ncol=3)
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_rms_timeline.png")); plt.close(fig)

# ---------- Fig 2: 噪声谱 vs 语音谱 ----------
d = np.load(r"C:\projects\local\audio_processer\work\noise_speech_spec.npz")
f, nz, sp = d["freqs"], d["noise"], d["speech"]
fig, ax = plt.subplots(figsize=(14, 4.2), dpi=140)
ax.plot(f, nz - nz.max(), lw=1.1, color="#d62728", label="最安静 10% 帧（≈纯噪声）")
ax.plot(f, sp - sp.max(), lw=1.1, color="#1f77b4", label="最响 10% 帧（≈语音）")
ax.set_xlim(0, 8000); ax.set_ylim(-80, 3)
ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("相对能量 (dB)")
ax.set_title("噪声与语音频谱：底噪集中在 0–300 Hz（低频隆隆声）+ 3 kHz 以上宽带底噪")
for lo, hi in [(0, 300)]:
    ax.axvspan(lo, hi, color="#d62728", alpha=0.08)
ax.annotate("低频噪声主导", xy=(150, -12), fontsize=9, color="#d62728")
ax.annotate("语音主体 300 Hz–4 kHz", xy=(1100, -30), fontsize=9, color="#1f77b4")
ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_noise_vs_speech.png")); plt.close(fig)

# ---------- Fig 3: 16k 线性频谱（正文片段 vs 片头）----------
def spec_img(ax, sig, sr, title, tmax=8):
    nfft, hop = 1024, 160
    frames = np.lib.stride_tricks.sliding_window_view(sig, nfft)[::hop]
    frames = frames * np.hanning(nfft)
    S = 20*np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-8)
    S = S - S.max()
    ax.imshow(S.T, origin="lower", aspect="auto",
              extent=[0, len(sig)/sr, 0, sr/2], cmap="magma", vmin=-90, vmax=0)
    ax.set_ylim(0, tmax*1000); ax.set_xlabel("时间 (s)"); ax.set_ylabel("频率 (Hz)")
    ax.set_title(title, fontsize=9)

seg_a = m[600*sr:630*sr]
seg_b = m[0:30*sr]
fig, axes = plt.subplots(2, 1, figsize=(13, 6), dpi=140)
spec_img(axes[0], seg_a, sr, "正文片段（10:00–10:30）：水平共振峰 + 能量集中在 4 kHz 以下")
spec_img(axes[1], seg_b, sr, "片头片段（0:00–0:30）：能量平稳、无清晰语音共振峰 → 非讲课内容（碎片化人声/环境声）")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_spectrogram_compare.png")); plt.close(fig)

print("figures written to", OUT)
