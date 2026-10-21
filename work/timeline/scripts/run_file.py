"""端到端跑单个文件：decode -> VAD -> ASR -> diar -> merge -> build。

用法:
  python -X utf8 run_file.py --short 24_形而上学 \
      --src "data/24_....m4a" --outdir work/timeline/24_形而上学 --threads 8

若存在 work/audio-edit/<短名|拼音短名>/{clean.wav,mapping.json} 则自动用 clean 音轨，
并在 transcript 里同时给 orig/clean 双时间；否则用原始 m4a。
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

PY = sys.executable

# data/<name> -> 拼音短名（与 work/audio-edit/wav/filemap.json 一致，便于对接）
ALIAS = {
    "0_": "xfzx", "20_": "lxg1", "24_": "xes", "4_": "lxg4",
    "【复旦哲学系丁耘】 亚里士多德《物理学》": "wlx",
    "【复旦哲学系丁耘】孔子《论语》": "ly",
}


def run(cmd, log_prefix=""):
    print(f"[run] {log_prefix or os.path.basename(cmd[2])}", flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=common.ROOT)
    if p.returncode != 0:
        raise SystemExit(f"step failed ({p.returncode}): {' '.join(cmd)}")
    return time.perf_counter() - t0


def find_clean(short, src_name):
    """在 work/audio-edit 下找 clean.wav + mapping.json（短名或拼音短名都认）。"""
    base = os.path.join(common.ROOT, "work", "audio-edit")
    if not os.path.isdir(base):
        return None
    aliases = {short}
    for k, v in ALIAS.items():
        if src_name.startswith(k):
            aliases.add(v)
    for name in aliases:
        d = os.path.join(base, name)
        clean = os.path.join(d, "clean.wav")
        mapping = os.path.join(d, "mapping.json")
        if os.path.exists(clean):
            return {"clean": clean, "mapping": mapping if os.path.exists(mapping) else None,
                    "dir": d, "alias": name}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True)
    ap.add_argument("--src", required=True, help="data/ 下原始 m4a（或任意音频）")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cache-dir", default=os.path.join(common.ROOT, "work", "timeline", "_cache"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", type=int, default=0, help="1=重跑所有步骤")
    ap.add_argument("--skip-diar", type=int, default=0)
    ap.add_argument("--diar-threshold", type=float, default=0.7)
    ap.add_argument("--merge-threshold", type=float, default=0.7)
    ap.add_argument("--no-clean", type=int, default=0, help="1=忽略 audio-edit 的 clean.wav")
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(common.ROOT, args.src)
    src_name = os.path.basename(src)
    outdir = common.ensure_dir(args.outdir if os.path.isabs(args.outdir)
                               else os.path.join(common.ROOT, args.outdir))
    cache = common.ensure_dir(args.cache_dir)
    meta_path = os.path.join(outdir, "run_meta.json")
    meta = common.read_json(meta_path) if (os.path.exists(meta_path) and not args.force) else {}
    timings = meta.get("timings", {})

    clean = None if args.no_clean else find_clean(args.short, src_name)
    if clean:
        src_audio = clean["clean"]
        mapping = clean["mapping"]
        time_base = "clean"
        print(f"[clean] 使用 audio-edit 产物: {common.rel(src_audio)} "
              f"mapping={'有' if mapping else '无'}", flush=True)
    else:
        src_audio = src
        mapping = None
        time_base = "original"
        print("[clean] work/audio-edit 下无 clean.wav，使用原始音频", flush=True)

    wav = os.path.join(cache, f"{args.short}_16k.wav")
    if args.force or not os.path.exists(wav) or os.path.getmtime(wav) < os.path.getmtime(src_audio):
        t0 = time.perf_counter()
        common.decode_to_wav(src_audio, wav)
        timings["decode_s"] = round(time.perf_counter() - t0, 1)
        print(f"[decode] {common.rel(src_audio)} -> {common.rel(wav)} "
              f"{timings['decode_s']}s", flush=True)

    def maybe(name, fn):
        p = os.path.join(outdir, name)
        if args.force or not os.path.exists(p):
            t = fn()
            timings[name.replace(".json", "_s")] = round(t, 1)
        return p

    vad_p = maybe("vad.json", lambda: run(
        [PY, "-X", "utf8", os.path.join(HERE, "step_vad.py"),
         "--wav", wav, "--out", os.path.join(outdir, "vad.json"),
         "--threads", str(args.threads)], "vad"))
    asr_p = maybe("asr.json", lambda: run(
        [PY, "-X", "utf8", os.path.join(HERE, "step_asr.py"),
         "--wav", wav, "--vad", vad_p, "--out", os.path.join(outdir, "asr.json"),
         "--threads", str(args.threads)], "asr"))
    diar_p = os.path.join(outdir, "diar.json")
    merge_p = os.path.join(outdir, "diar_merged.json")
    if args.skip_diar:
        print("[diar] skipped", flush=True)
    else:
        maybe("diar.json", lambda: run(
            [PY, "-X", "utf8", os.path.join(HERE, "step_diar.py"),
             "--wav", wav, "--out", diar_p,
             "--threshold", str(args.diar_threshold), "--threads", str(args.threads)], "diar"))
        maybe("diar_merged.json", lambda: run(
            [PY, "-X", "utf8", os.path.join(HERE, "step_merge.py"),
             "--wav", wav, "--diar", diar_p, "--out", merge_p,
             "--merge-threshold", str(args.merge_threshold),
             "--threads", str(args.threads)], "merge"))

    build_cmd = [PY, "-X", "utf8", os.path.join(HERE, "step_build.py"),
                 "--short", args.short, "--vad", vad_p, "--asr", asr_p,
                 "--outdir", outdir, "--time-base", time_base,
                 "--diar-merged", merge_p if (not args.skip_diar and os.path.exists(merge_p)) else ""]
    if mapping:
        build_cmd += ["--mapping", mapping]
    if args.force or not os.path.exists(os.path.join(outdir, "transcript.json")):
        t = run(build_cmd, "build")
        timings["build_s"] = round(t, 1)

    meta = {
        "short": args.short, "src": common.rel(src), "audio_used": common.rel(src_audio),
        "time_base": time_base, "mapping": common.rel(mapping) if mapping else None,
        "threads": args.threads, "clean_from_audio_edit": bool(clean),
        "timings": timings,
    }
    common.write_json(meta_path, meta)
    stats = common.read_json(os.path.join(outdir, "stats.json"))
    print(f"[done] {args.short}: segs={stats['n_segments']} chars={stats['n_chars_cn']} "
          f"asr_wall={stats['asr'].get('wall_s')}s rtf={stats['asr'].get('rtf')} "
          f"main={stats['main_ratio_of_speech']*100:.1f}% "
          f"timings={json.dumps(timings, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
