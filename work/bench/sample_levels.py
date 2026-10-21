import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audiometrics import metrics

for f in [r"work\samples\s0_head.wav", r"work\samples\s1_mid.wav",
          r"work\samples\s2_tail.wav", r"work\samples\mono_600_60.wav"]:
    d = metrics(f)
    print(f"{os.path.basename(f):22s} rms={d['rms_dbfs']:7.2f} p10={d['p10']:7.2f} "
          f"p50={d['p50']:7.2f} p90={d['p90']:7.2f} noise_floor={d['noise_floor_dbfs']:7.2f} "
          f"snr_est={d['snr_est_db']:5.1f} above+10dB={d['above_p10_10db_pct']:5.1f}%")
