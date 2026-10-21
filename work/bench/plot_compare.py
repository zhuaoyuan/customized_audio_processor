"""生成处理前后对比图：频谱图（spectrogram）+ 长时平均频谱曲线。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audiometrics import read_wav, spectrogram_png, metrics, db, frame_rms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = os.path.dirname(os.path.abspath(__file__))
DEN = os.path.join(BENCH, "denoise")


def long_term_spectrum(path):
    x, sr = read_wav(path)
    m = x[:, 0]
    n = 4096
    spec = np.zeros(n // 2 + 1)
    cnt = 0
    for i in range(0, max(len(m) - n, 1), n // 2):
        spec += np.abs(np.fft.rfft(m[i:i + n] * np.hanning(n))) ** 2
        cnt += 1
    spec = 10 * np.log10(spec / max(cnt, 1) + 1e-12)
    f = np.arange(len(spec)) * sr / n
    return f, spec, spec.max()


def main():
    src = os.path.join(BENCH, "..", "samples", "s1_mid.wav")
    src = os.path.abspath(src)
    pngs = {
        "spec": os.path.join(BENCH, "spec_compare_s1_mid.png"),
        "ltas": os.path.join(BENCH, "ltas_compare_s1_mid.png"),
    }

    files, labels = [], []
    variants = [
        ("原始 (baseline)", src),
        ("afftdn nr=12", os.path.join(DEN, "s1_mid_afftdn_nr12.wav")),
        ("afftdn nr=20", os.path.join(DEN, "s1_mid_afftdn_nr20.wav")),
        ("arnndn (rnnoise sh.rnnn)", os.path.join(DEN, "s1_mid_arnndn_sh.wav")),
        ("GTCRN ONNX (sherpa-onnx)", os.path.join(DEN, "s1_mid_gtcrn.wav")),
    ]
    for lab, p in variants:
        if os.path.exists(p):
            files.append(p)
            labels.append(lab)
    print("[files]", files)

    spectrogram_png(files, labels, pngs["spec"])
    print("[png]", pngs["spec"])

    # 长时平均频谱（各自按峰值归一，看形状变化）
    plt.figure(figsize=(11, 5.5))
    for lab, p in zip(labels, files):
        f, s, pk = long_term_spectrum(p)
        plt.plot(f, s - pk, label=f"{lab}", linewidth=0.9)
    plt.xlim(0, 8000)
    plt.ylim(-90, 2)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("relative level (dB, peak-normalised)")
    plt.title("s1_mid.wav long-term average spectrum: before / after denoise")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(pngs["ltas"], dpi=115)
    plt.close()
    print("[png]", pngs["ltas"])

    # 数值表：分频段能量（相对各自峰值）
    print("\n分频段能量（dB，相对各自长时平均谱峰值）:")
    bands = [(0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 5000), (5000, 8000)]
    print("files".ljust(28) + "".join(f"{f'{a}-{b}Hz'.rjust(12)}" for a, b in bands))
    for lab, p in zip(labels, files):
        f, s, pk = long_term_spectrum(p)
        rel = s - pk
        cells = []
        for a, b in bands:
            sel = (f >= a) & (f < b)
            cells.append(f"{rel[sel].max():12.1f}")
        print(lab.ljust(28) + "".join(cells))


if __name__ == "__main__":
    main()
