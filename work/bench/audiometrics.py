"""共享音频度量工具：P10/P90 帧 RMS、近似 SNR、长时平均频谱、频谱图。"""
import wave
import numpy as np


def read_wav(path):
    with wave.open(path, 'rb') as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr


def frame_rms(x, sr, win=0.025, hop=0.010):
    w, h = int(sr * win), int(sr * hop)
    if len(x) <= w:
        return np.array([np.sqrt(np.mean(x ** 2))])
    idx = np.arange(0, len(x) - w + 1, h)
    return np.array([np.sqrt(np.mean(x[i:i + w] ** 2)) for i in idx])


def db(v, eps=1e-10):
    return 20 * np.log10(np.maximum(v, eps))


def metrics(path, win=0.025, hop=0.010):
    x, sr = read_wav(path)
    m = x[:, 0]
    rdb = db(frame_rms(m, sr, win, hop))
    d = {
        "sr": sr,
        "dur": len(m) / sr,
        "rms_dbfs": float(db(np.sqrt(np.mean(m ** 2)))),
        "peak_dbfs": float(db(np.max(np.abs(m)))),
    }
    for p in (1, 5, 10, 50, 90, 99):
        d[f"p{p}"] = float(np.percentile(rdb, p))
    d["snr_proxy"] = d["p90"] - d["p10"]
    d["above_p10_10db_pct"] = float(np.mean(rdb > d["p10"] + 10) * 100)
    d["below_p10_3db_pct"] = float(np.mean(rdb < d["p10"] + 3) * 100)
    # 底噪电平：取最安静 10% 帧的平均能量（dB），比 P10 更稳
    thr = np.percentile(rdb, 10)
    quiet = rdb[rdb <= thr]
    loud = rdb[rdb >= np.percentile(rdb, 90)]
    d["noise_floor_dbfs"] = float(10 * np.log10(np.mean(10 ** (quiet / 10)) + 1e-20))
    d["speech_level_dbfs"] = float(10 * np.log10(np.mean(10 ** (loud / 10)) + 1e-20))
    d["snr_est_db"] = d["speech_level_dbfs"] - d["noise_floor_dbfs"]
    # 长时平均频谱（相对峰值），用于看高频底噪有无下降
    n = 4096
    spec = np.zeros(n // 2 + 1)
    cnt = 0
    for i in range(0, max(len(m) - n, 1), n // 2):
        spec += np.abs(np.fft.rfft(m[i:i + n] * np.hanning(n))) ** 2
        cnt += 1
    spec = 10 * np.log10(spec / max(cnt, 1) + 1e-12)
    freqs = np.arange(len(spec)) * sr / n
    d["_spec"] = (freqs, spec)
    return d


def band_energy(d):
    freqs, spec = d["_spec"]
    peak = spec.max()
    rel = spec - peak
    out = {}
    for lo, hi in ((0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 8000)):
        sel = (freqs >= lo) & (freqs < hi)
        out[f"{lo}-{hi}Hz"] = float(rel[sel].max()) if sel.any() else float("nan")
    return out


def spectrogram_png(wav_paths, labels, out_png, sr_target=16000, nfft=1024, hop=256):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(len(wav_paths), 1, figsize=(12, 2.6 * len(wav_paths)), sharex=True)
    if len(wav_paths) == 1:
        axes = [axes]
    for ax, p, lab in zip(axes, wav_paths, labels):
        x, sr = read_wav(p)
        m = x[:, 0]
        if sr != sr_target:
            # 简单重采样（线性插值，仅用于出图）
            idx = np.arange(0, len(m), sr / sr_target)
            m = np.interp(idx, np.arange(len(m)), m)
            sr = sr_target
        w = np.hanning(nfft)
        frames = [m[i:i + nfft] * w for i in range(0, max(len(m) - nfft, 1), hop)]
        S = np.abs(np.fft.rfft(np.array(frames), axis=1))
        S = 20 * np.log10(S + 1e-9)
        S = S - S.max()
        t = np.arange(S.shape[0]) * hop / sr
        f = np.arange(S.shape[1]) * sr / nfft
        im = ax.pcolormesh(t, f[:int(8000 * nfft / sr) + 1], S[:, :int(8000 * nfft / sr) + 1].T,
                           vmin=-90, vmax=0, cmap="magma", shading="auto")
        ax.set_ylabel("Hz")
        ax.set_title(lab, fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, pad=0.01, label="dB")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
