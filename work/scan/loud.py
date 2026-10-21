# -*- coding: utf-8 -*-
"""Step 2: EBU R128 响度（ffmpeg ebur128），解析 I / LRA / TruePeak / 阈值。

对 f00 使用修复副本（stco 偏移修正），其余用原始 m4a。
输出 work/scan/loud.json
"""
import concurrent.futures as cf
import json
import os
import re
import subprocess

ROOT = r"C:\projects\local\audio_processer"
SCAN = os.path.join(ROOT, "work", "scan")

SHORTS = ["f00_xifang20140227", "f20_lixiangguo20141219", "f24_xingershangxue",
          "f04_lixiangguo4", "f05_wulixue", "f06_lunyu"]


def ebur128(path):
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
                       capture_output=True)
    txt = r.stderr.decode("utf-8", "replace")
    out = {}
    m = re.search(r"I:\s+(-?[\d.]+)\s+LUFS", txt)
    out["I_lufs"] = float(m.group(1)) if m else None
    m = re.search(r"LRA:\s+(-?[\d.]+)\s+LU", txt)
    out["LRA_lu"] = float(m.group(1)) if m else None
    m = re.search(r"Peak:\s+(-?[\d.]+)\s+dBFS", txt)
    out["true_peak_dbfs"] = float(m.group(1)) if m else None
    m = re.search(r"Threshold:\s+(-?[\d.]+)\s+LUFS", txt)
    out["I_threshold_lufs"] = float(m.group(1)) if m else None
    m = re.search(r"LRA low:\s+(-?[\d.]+)\s+LUFS", txt)
    out["lra_low_lufs"] = float(m.group(1)) if m else None
    m = re.search(r"LRA high:\s+(-?[\d.]+)\s+LUFS", txt)
    out["lra_high_lufs"] = float(m.group(1)) if m else None
    # 若格式不同，回退解析逐帧行最后一条
    if out["I_lufs"] is None:
        for line in reversed(txt.splitlines()):
            m = re.search(r"I:\s+(-?[\d.]+)\s+LUFS", line)
            if m:
                out["I_lufs"] = float(m.group(1))
                break
    out["rc"] = r.returncode
    if out["I_lufs"] is None:
        out["_tail"] = txt.splitlines()[-12:]
    return out


def main():
    probe = json.load(open(os.path.join(SCAN, "probe.json"), encoding="utf-8"))
    files = probe["files"]

    def job(s):
        p = files[s]
        d = ebur128(p.get("decode_src") or p["src"])
        d["src"] = p.get("decode_src") or p["src"]
        print(f"[loud] {s}: I={d['I_lufs']} LUFS LRA={d['LRA_lu']} TP={d['true_peak_dbfs']} dBFS", flush=True)
        return s, d

    res = {}
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for s, d in ex.map(job, SHORTS):
            res[s] = d
    json.dump(res, open(os.path.join(SCAN, "loud.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[loud] done")


if __name__ == "__main__":
    main()
