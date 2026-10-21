# -*- coding: utf-8 -*-
"""Step 1: ffprobe 元数据 + 解码为 16k mono float32 wav。

输入 data/*.m4a，短名映射固定写入 FILES。
输出 work/scan/probe.json 与 work/scan/wav/<short>.wav（16k mono pcm_f32le）。
"""
import json
import os
import subprocess
import sys
import concurrent.futures as cf

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(ROOT, "data")
SCAN = os.path.join(ROOT, "work", "scan")
WAV = os.path.join(SCAN, "wav")

# short -> 原始文件名（不依赖顺序，显式全名）
FILES = {
    "f00_xifang20140227": "0_20140227-丁耘 西方哲学(20140227-丁耘 西方哲学)_normal..m4a",
    "f20_lixiangguo20141219": "20_丁耘理想国.古希腊哲学(20141219-丁耘  古希腊哲学)_normal..m4a",
    "f24_xingershangxue": "24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a",
    "f04_lixiangguo4": "4_丁耘理想国.古希腊哲学(第四卷)_normal..m4a",
    "f05_wulixue": "【复旦哲学系丁耘】 亚里士多德《物理学》(05-20100401)_5_normal..m4a",
    "f06_lunyu": "【复旦哲学系丁耘】孔子《论语》(06)_19_normal..m4a",
}

# f00 的原始 m4a 容器 stco 偏移整体偏小 22118B（moov 内 udta 插入后未更新），
# 直接用会解码失败；work/scan/patch_f00.py 生成修复副本，仅偏移 stco，mdat 数据未改。
DECODE_SRC = {
    "f00_xifang20140227": os.path.join(SCAN, "f00_fixed.m4a"),
}


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-print_format", "json", path],
        capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace"))
    return json.loads(out.stdout.decode("utf-8", "replace"))


def decode(path, wav_out):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path,
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", wav_out]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace"))
    return os.path.getsize(wav_out)


def one(short, name):
    src = os.path.join(DATA, name)
    if not os.path.isfile(src):
        return short, {"error": "missing input", "src": src}
    meta = probe(src)
    fmt = meta.get("format", {})
    ast = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), {})
    dec_src = DECODE_SRC.get(short, src)
    wav_out = os.path.join(WAV, short + ".wav")
    wav_bytes = decode(dec_src, wav_out)
    d = {
        "short": short,
        "file": name,
        "src": src,
        "decode_src": dec_src,
        "repaired": dec_src != src,
        "size_bytes": int(fmt.get("size", os.path.getsize(src))),
        "format_name": fmt.get("format_name"),
        "dur_s": float(fmt.get("duration")),
        "bit_rate_bps": int(fmt.get("bit_rate", 0)),
        "tags": fmt.get("tags", {}),
        "codec": ast.get("codec_name"),
        "profile": ast.get("profile"),
        "sample_rate": int(ast.get("sample_rate", 0)),
        "channels": int(ast.get("channels", 0)),
        "channel_layout": ast.get("channel_layout"),
        "stream_bit_rate_bps": int(ast.get("bit_rate", 0) or 0),
        "wav": wav_out,
        "wav_bytes": wav_bytes,
        "wav_dur_s": round(wav_bytes / 4 / 16000, 3),
    }
    return short, d


def main():
    os.makedirs(WAV, exist_ok=True)
    results = {}
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(one, s, n) for s, n in FILES.items()]
        for f in cf.as_completed(futs):
            s, d = f.result()
            results[s] = d
            print(f"[probe] {s}: dur={d.get('dur_s')} codec={d.get('codec')} "
                  f"sr={d.get('sample_rate')} ch={d.get('channels')} "
                  f"br={d.get('bit_rate_bps')} size={d.get('size_bytes')} wav={d.get('wav_bytes')}",
                  flush=True)
    out = {"root": ROOT, "files": {s: results[s] for s in FILES}}
    with open(os.path.join(SCAN, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    tot = sum(d["dur_s"] for d in results.values())
    print(f"[probe] total dur = {tot/60:.2f} min")


if __name__ == "__main__":
    main()
