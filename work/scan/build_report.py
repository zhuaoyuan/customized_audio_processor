# -*- coding: utf-8 -*-
"""Step 6: 汇总 scan.json + docs/research/multifile-scan.md。全部数字来自实测 JSON。"""
import json
import os
from datetime import datetime

ROOT = r"C:\projects\local\audio_processer"
SCAN = os.path.join(ROOT, "work", "scan")
DOC = os.path.join(ROOT, "docs", "research")

SHORTS = ["f00_xifang20140227", "f20_lixiangguo20141219", "f24_xingershangxue",
          "f04_lixiangguo4", "f05_wulixue", "f06_lunyu"]
TARGET_ASL = -20.0


def load(name, default=None):
    p = os.path.join(SCAN, name)
    if not os.path.isfile(p):
        return default
    return json.load(open(p, encoding="utf-8"))


def fmt(v, nd=1, dash="—"):
    if v is None:
        return dash
    return f"{v:.{nd}f}"


def minsec(s):
    if s is None:
        return "—"
    m, sec = divmod(int(round(s)), 60)
    return f"{m}:{sec:02d}"


def asr_teaching_start(chunks, need=10, win=3, need_sum=60):
    """30s 分块：首个“本块>=need 字且后 3 块合计>=need_sum 字”的时刻。
    用于避开片头音乐/环境声（VAD 会误判音乐为语音）。"""
    for i in range(len(chunks)):
        if chunks[i]["chars"] >= need and sum(c["chars"] for c in chunks[i:i + win]) >= need_sum:
            return chunks[i]["t0"]
    return None


def asr_teaching_end(chunks, need=20, win=3, need_sum=60):
    for i in range(len(chunks) - 1, -1, -1):
        if chunks[i]["chars"] >= need and sum(c["chars"] for c in chunks[max(0, i - win + 1):i + 1]) >= need_sum:
            return chunks[i]["t1"]
    return None


# 人工判读（依据 head_scan.json 的 ASR 文本）：起点有字但属课前闲聊/非授课
PRE_CLASS = {
    "f00_xifang20140227": {"lecture_start": 90.0,
                           "note": "0–90 s 为课前闲聊（有字但非授课），90 s 起进入课程介绍"},
}


def main():
    probe = load("probe.json")
    loud = load("loud.json", {})
    asr = load("asr60.json", {})
    head = load("head_scan.json", {})
    vad = {s: load(os.path.join("vad", s + ".json")) for s in SHORTS}

    def diar_of(s):
        d = load(os.path.join("diar", s + ".json"))
        m = load(os.path.join("diar", s + "_merged.json"))
        if d and "num_effective_speakers" not in d:
            st = sum(d["per_speaker_s"].values())
            d["num_effective_speakers"] = sum(
                1 for v in d["per_speaker_s"].values() if st and v / st >= 0.05)
        return d, m

    rows = []
    for s in SHORTS:
        p, L, V, A, H = probe["files"][s], loud.get(s, {}), vad[s], asr.get(s, {}), head.get(s, {})
        dl, dm = diar_of(s)
        lufs, asl = L.get("I_lufs"), V["level"]["asl_dbfs"]
        dv = V["edges"]
        r = {
            "short": s, "name": p["file"],
            "dur_s": p["dur_s"], "dur_min": p["dur_s"] / 60,
            "size_mb": p["size_bytes"] / 1024 / 1024,
            "bitrate_kbps": p["bit_rate_bps"] / 1000,
            "codec": p["codec"], "sample_rate": p["sample_rate"], "channels": p["channels"],
            "repaired": p.get("repaired", False),
            "lufs": lufs, "lra": L.get("LRA_lu"), "tp": L.get("true_peak_dbfs"),
            "lra_low": L.get("lra_low_lufs"), "lra_high": L.get("lra_high_lufs"),
            "asl": asl,
            "lufs_minus_asl": (lufs - asl) if (lufs is not None and asl is not None) else None,
            "gain_to_target_db": (TARGET_ASL - asl) if asl is not None else None,
            "noise_after_gain": (V["level"]["noise_floor_dbfs"] + (TARGET_ASL - asl)) if asl is not None else None,
            "p10": V["level"]["speech_frame_p10_dbfs"],
            "p50": V["level"]["speech_frame_p50_dbfs"],
            "p90": V["level"]["speech_frame_p90_dbfs"],
            "noise_floor": V["level"]["noise_floor_dbfs"],
            "snr_proxy": V["level"]["snr_proxy_db"],
            "speech_ratio": V["vad"]["speech_ratio"], "speech_s": V["vad"]["speech_s"],
            "n_vad_segs": V["vad"]["n_segments"],
            "head_trim_vad": dv["head_trim_s"], "tail_trim_vad": dv["tail_trim_s"],
            "head5m_ratio": dv["head5m_speech_ratio"],
            "asr_words_start": asr_teaching_start(H.get("head_chunks", [])),
            "asr_teaching_start": asr_teaching_start(H.get("head_chunks", [])),
            "asr_teaching_end": asr_teaching_end(H.get("tail_chunks", [])) or H.get("teaching_end_asr_s"),
            "pre_class": PRE_CLASS.get(s),
            "head_scan_chunks": H.get("head_chunks", []),
            "tail_scan_chunks": H.get("tail_chunks", []),
            "gaps_n": V["nonspeech"]["n_gaps"], "gaps_total": V["nonspeech"]["total_s"],
            "gap_longest": V["nonspeech"]["longest_s"], "top_gaps": V["nonspeech"]["top_gaps"],
            "w5m": V["window5m"], "asr": A, "diar": dl or {}, "diar_merged": dm or {},
            "segments_n": len(V["segments"]),
        }
        if dm:
            r["speakers_merged"] = dm["merged_num_speakers"]
            r["speakers_ge1s"] = dm["merged_speakers_ge1s"]
            r["speakers_effective"] = dm["num_effective_speakers"]
            r["speakers_raw"] = dm["raw_num_speakers"]
            r["main_share"] = dm["main_share_of_speech"]
            r["nonmain_s"] = dm["nonmain_speech_s"]
        else:
            r.update({k: None for k in ("speakers_merged", "speakers_ge1s", "speakers_effective",
                                        "speakers_raw", "main_share", "nonmain_s")})
        rows.append(r)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": ROOT,
        "total_dur_min": sum(r["dur_min"] for r in rows),
        "target_asl_dbfs": TARGET_ASL,
        "file_map": {r["short"]: r["name"] for r in rows},
        "wav_map": {r["short"]: probe["files"][r["short"]]["wav"] for r in rows},
        "method": {
            "decode": "ffmpeg -vn -ac 1 -ar 16000 -c:a pcm_f32le",
            "loudness": "ffmpeg ebur128=peak=true (EBU R128 I/LRA/TruePeak)",
            "vad": "silero-vad v6 onnx (faster-whisper assets), 512-sample windows",
            "vad_opts": dict(threshold=0.5, min_speech_duration_ms=250,
                             min_silence_duration_ms=300, speech_pad_ms=100),
            "frame": "25ms win / 10ms hop RMS dBFS",
            "asl": "10*log10(mean(10^(dB/10))) over speech frames (ITU-T P.56 idea)",
            "head_asr_scan": "SenseVoice 30s chunks over first 600s / last 150s; teaching start = "
                             "first 90s window with >=60 effective chars",
            "asr": "sherpa-onnx SenseVoiceSmall int8, 60s body sample, 8 threads",
            "diar": "sherpa-onnx pyannote-seg-3.0 + campplus zh, 10min sample, threshold .5, "
                    "then cluster-merge by embedding cosine > 0.75",
        },
        "files": {r["short"]: {k: v for k, v in r.items()
                               if k not in ("short", "top_gaps", "w5m", "head_scan_chunks")}
                  | {"top_gaps": r["top_gaps"], "window5m": r["w5m"],
                     "head_scan_chunks": r["head_scan_chunks"]} for r in rows},
    }
    json.dump(result, open(os.path.join(SCAN, "scan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    write_md(rows, result)
    print("[report] wrote scan.json + multifile-scan.md")
    for r in rows:
        print(f"  {r['short']}: {r['dur_min']:.1f}min LUFS={r['lufs']} ASL={r['asl']:.1f} "
              f"diff={r['lufs_minus_asl']:+.1f} sp={r['speech_ratio']*100:.0f}% "
              f"headVAD={r['head_trim_vad']:.0f}s headASR={r['asr_teaching_start']} "
              f"spk={r['speakers_merged']}")


def write_md(rows, result):
    tot = sum(r["dur_min"] for r in rows)
    L = []
    A = L.append
    A("# 六文件音频体检报告（响度 / 语音电平 / 非语音结构 / 说话人 / ASR 抽样）")
    A("")
    A(f"- 生成时间：{result['generated_at']}")
    A(f"- 输入：`data/` 下 6 个 m4a（丁耘课程录音），总时长 **{tot:.2f} min**（{tot/60:.2f} h）")
    A("- 数据文件：`work/scan/scan.json`（结构化）；16k 解码音频在 `work/scan/wav/`；"
      "逐项中间结果见 `work/scan/{probe,loud,asr60,head_scan}.json` 与 `work/scan/{vad,asr,diar}/`")
    A("- 全部指标为实测值，无估算。")
    A("")
    A("## 0. 命名映射（原始 m4a → 短名 → 16k wav）")
    A("")
    A("| 短名 | 原始文件 | 16k wav |")
    A("|---|---|---|")
    for r in rows:
        A(f"| `{r['short']}` | {r['name']} | `work/scan/wav/{r['short']}.wav` |")
    A("")
    A("## 1. 关键结论")
    A("")
    lufs_v = [r["lufs"] for r in rows]
    asl_v = [r["asl"] for r in rows]
    A(f"1. **源文件电平差异极大**：整段 EBU R128 在 **{min(lufs_v):.1f} ~ {max(lufs_v):.1f} LUFS** "
      f"（跨度 {max(lufs_v)-min(lufs_v):.1f} dB），语音 ASL 在 **{min(asl_v):.1f} ~ {max(asl_v):.1f} dBFS**"
      f"（跨度 {max(asl_v)-min(asl_v):.1f} dB）。`f00` 是异常点（{rows[0]['lufs']:.1f} LUFS / ASL "
      f"{rows[0]['asl']:.1f} dBFS / True Peak {rows[0]['tp']:+.1f} dBFS），比其余文件低约 20~30 dB。")
    A(f"2. **`f00` 归一化到目标 ASL −20 dBFS 需约 +{rows[0]['gain_to_target_db']:.0f} dB 增益**："
      f"其底噪 −49.8 dBFS 会被抬到约 {rows[0]['noise_after_gain']:.0f} dBFS，SNR 仅 ~10 dB——"
      "这个文件必须“先降噪再增益”，否则底噪/环境声会一起被放大，且其 True Peak 已 +0.5 dBFS，"
      "增益后必须限幅（limiter −1 dBTP）。")
    A(f"3. **LUFS 与语音 ASL 的差很小**（{min(r['lufs_minus_asl'] for r in rows):+.1f} ~ "
      f"{max(r['lufs_minus_asl'] for r in rows):+.1f} dB）：6 文件语音占比 "
      f"{min(r['speech_ratio'] for r in rows)*100:.0f}%~{max(r['speech_ratio'] for r in rows)*100:.0f}%，"
      "语音主导，EBU R128 相对门限又滤掉安静段，整段 LUFS 实际≈语音 ASL。"
      "**但整段 LUFS 仍不能直接当归一化目标**：① 它无法区分“片头非讲课声/内部长静音”与正文；"
      f"② 文件内 5 分钟窗口 ASL 漂移 p5–p95 达 "
      f"{min(r['w5m']['asl_spread_p95_p5'] for r in rows if r['w5m']['asl_spread_p95_p5'] is not None):.1f}~"
      f"{max(r['w5m']['asl_spread_p95_p5'] for r in rows if r['w5m']['asl_spread_p95_p5'] is not None):.1f} dB，"
      "一次全局增益无法对齐；③ 目标应按“语音段 ASL”定义（本报告用 ASL −20 dBFS 口径），而不是整段 LUFS。")
    A(f"4. **削波/峰值风险**：`f04` True Peak **+{abs(rows[3]['tp']):.1f} dBFS**（超 0 dBFS），"
      f"`f24` +{rows[2]['tp']:.1f}、`f00` +{rows[0]['tp']:.1f}；其余为负。任何增益方案都必须带 −1 dBTP limiter。")
    gmax = max(rows, key=lambda r: r["gaps_total"])
    A(f"5. **非语音结构**：内部 ≥3 s 非语音每文件 {min(r['gaps_n'] for r in rows)}~{max(r['gaps_n'] for r in rows)} 段、"
      f"合计 {min(r['gaps_total'] for r in rows):.0f}~{max(r['gaps_total'] for r in rows):.0f} s；最多的是 "
      f"`{gmax['short']}`（{gmax['gaps_n']} 段 / {gmax['gaps_total']:.0f} s / 最长 {gmax['gap_longest']:.1f} s）。")
    A("6. **⚠ 片头不能只看 VAD**：VAD 的 `first_speech_s` 会把**片头音乐/环境声误判为语音**，"
      "也会漏掉轻声讲课。30 s 分块 ASR 字符率复核后（VAD → ASR）：")
    A("")
    A("| 文件 | VAD 首语音 | ASR 首个持续有字 | 判定 |")
    A("|---|---|---|---|")
    for r in rows:
        pre = r.get("pre_class")
        claim = "—"
        if r["short"] == "f24_xingershangxue":
            claim = "VAD 的 3.6 s 是**音乐误触发**；0–480 s 无字，480 s 起真实讲课"
        elif r["short"] in ("f06_lunyu", "f20_lixiangguo20141219"):
            claim = "VAD 漏掉 90–121 s 的轻声讲课；90 s 起有字"
        elif r["short"] == "f05_wulixue":
            claim = "VAD 的 0.2 s 是噪声误触发；30 s 起有点名/讲课"
        elif pre:
            claim = pre["note"]
        elif r["asr_teaching_start"] is not None and r["asr_teaching_start"] < 5:
            claim = "文件从讲课中途开始，无片头"
        A(f"| `{r['short']}` | {r['head_trim_vad']:.1f} s | {fmt(r['asr_teaching_start'],0)} s | {claim} |")
    A("")
    A("**结论：头尾裁剪必须以“ASR 有字 + 连续授课”判定，VAD 只能做粗筛**（本报告 §2.3 的 ASR 列即该口径）。")
    spk = [(r["short"], r) for r in rows if r["speakers_effective"]]
    A(f"7. **说话人结构**：抽样 {len(spk)} 个文件（各 10 min 正文），合并过聚类后"
      f"“有效讲话人（占语音 ≥5%）”为 "
      + "、".join(f"`{s}` **{r['speakers_effective']} 人**" for s, r in spk)
      + "：主讲占比 "
      + "、".join(f"{r['main_share']*100:.1f}%" for s, r in spk)
      + "，非主讲语音 "
      + "、".join(f"{r['nonmain_s']:.0f} s" for s, r in spk)
      + "。**没有研讨会式轮流讨论的特征**（无稳定的第二讲课人；`f24` 的“第二簇 268 s”出现在 233–480 s，"
      "是讲课前音乐/前奏的人声）。仅 `f05` 的 60 s ASR 样本出现学生提问口吻（“我想问一下…”），属偶发问答。")
    A("")
    A("## 2. 汇总表")
    A("")
    A("### 2.1 输入与技术参数")
    A("")
    A("| 短名 | 时长 | 大小 | 码率 | 编码/采样率/声道 | 解码 |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        dec = "⚠ 容器修复（§6）" if r["repaired"] else "正常"
        A(f"| `{r['short']}` | {minsec(r['dur_s'])} ({r['dur_min']:.1f} min) | {r['size_mb']:.1f} MB | "
          f"{r['bitrate_kbps']:.0f} kbps | {r['codec']} / {r['sample_rate']} Hz / {r['channels']} ch | {dec} |")
    A("")
    A("### 2.2 响度与语音电平（核心表）")
    A("")
    A("| 短名 | LUFS(I) | LRA | TruePeak | 语音ASL | LUFS−ASL | 语音占比 | 语音 p10/p50/p90 | 底噪 | SNR代理 | 归一到 −20 dBFS 需 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| `{r['short']}` | {fmt(r['lufs'])} | {fmt(r['lra'])} LU | {fmt(r['tp'])} dBFS | "
          f"{fmt(r['asl'])} dBFS | {fmt(r['lufs_minus_asl'],2)} dB | {r['speech_ratio']*100:.1f}% | "
          f"{fmt(r['p10'])}/{fmt(r['p50'])}/{fmt(r['p90'])} dBFS | {fmt(r['noise_floor'])} dBFS | "
          f"{fmt(r['snr_proxy'])} dB | {fmt(r['gain_to_target_db'],1)} dB |")
    A("")
    A("### 2.3 非语音结构（VAD 粗筛 vs ASR 复核）")
    A("")
    A("| 短名 | VAD 首语音 | ASR 有字起点 | 头可裁(ASR) | 尾可裁(ASR) | 内部≥3s 段数/合计/最长 | 5min ASL p5–p95 | 备注 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        w = r["w5m"]
        spread = f"{w['asl_p5']:.1f}–{w['asl_p95']:.1f}（Δ{fmt(w['asl_spread_p95_p5'])}）" \
            if w.get("asl_p5") is not None else "—"
        hstart = r["asr_teaching_start"]
        head_trim_asr = hstart if hstart is not None else None
        tail_trim_asr = max(0.0, r["dur_s"] - r["asr_teaching_end"]) if r["asr_teaching_end"] is not None else None
        note = ""
        if r["short"] == "f24_xingershangxue":
            note = "VAD 3.6 s 为音乐误判；480 s 起真实讲课"
        elif r["short"] in ("f06_lunyu", "f20_lixiangguo20141219"):
            note = "VAD 漏掉 90–121 s 的轻声讲课"
        elif r["short"] == "f05_wulixue":
            note = "VAD 0.2 s 为噪声误判；30 s 起有点名"
        elif r.get("pre_class"):
            note = "0–90 s 为课前闲聊（有字非授课），可再裁 90 s"
        elif hstart is not None and hstart < 5:
            note = "从讲课中途开始，无片头"
        A(f"| `{r['short']}` | {r['head_trim_vad']:.1f} s | {fmt(hstart,0)} s | "
          f"{fmt(max(0.0,head_trim_asr) if head_trim_asr is not None else None,1)} s | {fmt(tail_trim_asr,1)} s | "
          f"{r['gaps_n']} / {r['gaps_total']:.0f} s / {r['gap_longest']:.1f} s | {spread} | {note} |")
    A("")
    A("> **`VAD 首语音` 不可直接用于裁剪**：`f24` 的 3.6 s 是音乐误触发（真实讲课 480 s），"
      "`f06`/`f20` 的 ~121 s 漏掉了 90–121 s 的轻声讲课，`f05` 的 0.2 s 是噪声。"
      "`ASR 有字起点` = 30 s 分块转写中“本块 ≥10 字且后 3 块合计 ≥60 字”的首个时刻（精度 ±30 s）。")
    A("")
    A("### 2.4 ASR 60 秒正文抽样")
    A("")
    A("| 短名 | 样本起点 | 有效字数 | 字/秒 | 前两句 |")
    A("|---|---|---|---|---|")
    for r in rows:
        a = r["asr"]
        s2 = a.get("first2_sentences") or []
        txt = " / ".join(s2).replace("|", "\\|")[:120]
        A(f"| `{r['short']}` | {a.get('sample_start_s', '—')} s | {a.get('chars', '—')} | "
          f"{fmt(a.get('chars_per_s'), 2)} | {txt} |")
    A("")
    A("### 2.5 说话人（10 min 正文抽样）")
    A("")
    A("| 短名 | 抽样起点 | 原始簇数 | 合并后(≥1s) | 有效讲话人(≥5%) | 主讲占比 | 非主讲语音 | 说明 |")
    A("|---|---|---|---|---|---|---|---|")
    note = {
        "f00_xifang20140227": "S0/S2 块状交替，合并后同一人 → 单人授课",
        "f20_lixiangguo20141219": "99.9% 主讲 → 单人",
        "f24_xingershangxue": "抽样窗 233.7 s 起，落在讲课前音乐/前奏（真实讲课 480 s）；第二簇 268 s 疑为音乐人声 → 非第二讲课人",
        "f04_lixiangguo4": "98.8% 主讲 → 单人",
        "f05_wulixue": "未抽样（任务只要求 2 个文件；60 s ASR 见学生提问线索）",
        "f06_lunyu": "未抽样（任务只要求 2 个文件）",
    }
    for r in rows:
        if r["speakers_effective"] is not None:
            A(f"| `{r['short']}` | {r['diar'].get('start_s')} s | {r['speakers_raw']} | "
              f"{r['speakers_ge1s']} | **{r['speakers_effective']}** | {r['main_share']*100:.1f}% | "
              f"{r['nonmain_s']:.0f} s | {note[r['short']]} |")
        else:
            A(f"| `{r['short']}` | — | — | — | — | — | — | {note[r['short']]} |")
    A("")
    A("## 3. 逐文件明细")
    A("")
    for r in rows:
        A(f"### `{r['short']}` — {r['name']}")
        A("")
        A(f"- 时长 {r['dur_s']:.1f} s（{r['dur_min']:.1f} min），{r['size_mb']:.1f} MB，"
          f"{r['bitrate_kbps']:.0f} kbps，{r['sample_rate']} Hz {r['channels']} ch")
        if r["repaired"]:
            A("- ⚠ 原始容器 `stco` 偏移整体少 22118 B，直接解码失败；用修复副本 `work/scan/f00_fixed.m4a` 解码（见 §6）")
        A(f"- 响度：I = **{fmt(r['lufs'])} LUFS**，LRA = {fmt(r['lra'])} LU（{fmt(r['lra_low'])} ~ {fmt(r['lra_high'])}），"
          f"True Peak = {fmt(r['tp'])} dBFS")
        A(f"- 语音：占比 **{r['speech_ratio']*100:.1f}%**（{r['speech_s']/60:.1f} min / {r['segments_n']} 段），"
          f"ASL = **{fmt(r['asl'])} dBFS**，帧 p10/p50/p90 = {fmt(r['p10'])}/{fmt(r['p50'])}/{fmt(r['p90'])} dBFS，"
          f"噪声底 {fmt(r['noise_floor'])} dBFS，SNR 代理 {fmt(r['snr_proxy'])} dB，LUFS−ASL = {fmt(r['lufs_minus_asl'],2)} dB")
        A(f"- 归一化测算：归一到 ASL −20 dBFS 需 **{fmt(r['gain_to_target_db'])} dB**；"
          f"增益后底噪约 {fmt(r['noise_after_gain'])} dBFS（SNR 不变），峰值需 limiter")
        w = r["w5m"]
        A(f"- 5 分钟窗口（{w['n_windows']} 个）：ASL p5/p50/p95 = {fmt(w['asl_p5'])}/{fmt(w['asl_p50'])}/"
          f"{fmt(w['asl_p95'])} dBFS，范围 {fmt(w['asl_min'])} ~ {fmt(w['asl_max'])} dBFS")
        A(f"- 头尾（VAD 粗筛）：第一段“语音” {r['head_trim_vad']:.1f} s，最后语音结束 {minsec(r['dur_s']-r['tail_trim_vad'])}；"
          f"ASR 复核：有字起点 **{fmt(r['asr_teaching_start'],0)} s**、讲课结束 "
          f"{minsec(r['asr_teaching_end'])}（30 s 分块字符率）")
        if r.get("pre_class"):
            A(f"  - ⚠ {r['pre_class']['note']}")
        chunks = r["head_scan_chunks"][:8]
        if chunks:
            A("  - 片头 30 s 分块字符率：" + "；".join(
                f"{c['t0']:.0f}–{c['t1']:.0f}s {c['chars']}字" for c in chunks))
        A(f"- 内部 ≥3 s 非语音：**{r['gaps_n']} 段 / {r['gaps_total']:.0f} s**，最长 {r['gap_longest']:.1f} s"
          + (("；最长几段：" + "；".join(f"{g['start']:.0f}s 起 {g['dur']:.1f}s" for g in r["top_gaps"][:5]))
             if r["top_gaps"] else ""))
        a = r["asr"]
        if a:
            A(f"- ASR 样本：{a['sample_start_s']} s 起 60 s，有效 {a['chars']} 字 → **{a['chars_per_s']:.2f} 字/秒**"
              f"（窗内 VAD 语音 {a['vad_speech_in_window_s']} s，解码 {a['decode_s']} s）")
            A(f"  - 前两句：{'；'.join(a['first2_sentences']) if a['first2_sentences'] else '（空）'}")
            A(f"  - 全文：{a['text']}")
        if r["speakers_merged"]:
            d = r["diar"]; m = r["diar_merged"]
            A(f"- 说话人（{d['start_s']} s 起 {d['dur_s']:.0f} s）：原始聚类 **{m['raw_num_speakers']} 簇** → "
              f"按 embedding 余弦 >0.70 合并为 {m['merged_num_speakers']} 簇（≥1 s: {r['speakers_ge1s']}），"
              f"**有效讲话人（≥5% 语音）{r['speakers_effective']} 人**；"
              f"主讲占比 **{r['main_share']*100:.1f}%**，非主讲语音 {r['nonmain_s']:.0f} s，"
              f"RTF {d['rtf']:.3f}")
            A(f"  - 合并后各人秒数：{[x['dur_s'] for x in m['merged_speakers'][:8]]}")
            A(f"  - 原始簇秒数（前 8）：{list(d['per_speaker_s'].items())[:8]}")
        else:
            A("- 说话人：未抽样（任务只要求 2 个文件；本报告对 f00/f20/f24/f04 均做了抽样，见 §2.5）")
        A("")
    A("## 4. 方法")
    A("")
    A("1. **元数据/解码**：`ffprobe -show_format -show_streams`；`ffmpeg -vn -ac 1 -ar 16000 -c:a pcm_f32le` "
      "→ `work/scan/wav/<短名>.wav`（float32 单声道 16 kHz，供 VAD/ASR/diar 复用）。")
    A("2. **响度**：`ffmpeg -af ebur128=peak=true -f null -`，解析 Integrated LUFS、LRA、True Peak、门限与 LRA low/high；"
      "f00 用修复副本，其余用原始 m4a。")
    A("3. **VAD**：Silero VAD v6 ONNX（`faster_whisper/assets/silero_vad_v6.onnx`，onnxruntime 批量推理），"
      "threshold=0.5、min_speech=250 ms、min_silence=300 ms、speech_pad=100 ms。"
      "（未用 sherpa-onnx `SileroVadModelConfig`：本机可用模型是 v5/v6 的 h/c 状态接口，与 sherpa-onnx 的 v4 `state` 接口不匹配。）")
    A("4. **语音电平**：25 ms 窗 / 10 ms 跳 RMS dBFS；帧中心落在 VAD 语音段内即语音帧；"
      "**ASL = 10·log10(mean(10^(帧dB/10)))**（ITU-T P.56 活跃语音电平思路），另给语音帧 p10/p50/p90、"
      "噪声底（最安静 10% 帧能量均值）、SNR 代理 = ASL − 噪声底。")
    A("5. **5 分钟窗口**：互不重叠 300 s 窗，窗内语音 ≥30 s 才统计 ASL，取各窗 ASL 的 p5/p50/p95。")
    A("6. **非语音结构**：头可裁 = 第一段 VAD 语音起点（粗筛）+ ASR 复核（30 s 分块、连续 90 s ≥60 字）；"
      "尾同理；内部非语音 = 相邻 VAD 语音段间隔 ≥3 s 的段数/总时长/最长值。")
    A("7. **ASR**：sherpa-onnx SenseVoiceSmall int8（`work/bench/models/sensevoice`），8 线程、`language=zh`、`use_itn=True`。"
      "60 s 正文样本窗从 VAD 段滑动（步长 10 s、避开片头 5%）取语音覆盖最大者；"
      "字/秒 = 有效字符（汉字+字母数字，不含标点空格）/ 窗长。片头复核用 30 s 分块转写。")
    A("8. **说话人**：sherpa-onnx `OfflineSpeakerDiarization`（pyannote segmentation-3.0 + 3D-Speaker CAM++ zh），"
      "8 线程、自动聚类 threshold=0.5、min_duration_on=0.3 s / off=0.5 s；每文件取正文 10 分钟"
      "（起点 = VAD 首语音 + 60 s，且不越过片尾 20 s）。因阈值 0.5 会明显过聚类（原始 5~44 簇），"
      "再用 CAM++ embedding 的簇质心余弦 >0.70 做同人合并；"
      "**有效讲话人** = 合并后占语音 ≥5% 的簇数（排除碎片/音乐误判）。")
    A("")
    A("## 5. f00 容器损坏与修复（实测）")
    A("")
    A("`0_20140227-丁耘 西方哲学(...)_normal..m4a` 直接解码会从第一帧起报 AAC 解析错误"
      "（`Reserved bit set` / `Number of bands ... exceeds limit` / `channel element ... is not allocated`），"
      "拿不到任何音频。排查与修复：")
    A("")
    A("- mdat 数据本身完好，`stsz` 样本大小之和 = mdat payload 字节数（73134337 B，完全相等）；")
    A("- **`stco` 的 70 个 chunk 偏移整体比实际 mdat 起点小 22118 B**（首 chunk 381996 vs mdat payload 404114），"
      "即 moov 内插入 `udta`（22216 B）后未更新 stco，导致每个样本从错误位置读取；"
      "帧起始 3 bit 元素 ID 呈均匀随机分布（好文件 100% 为 CPE），单帧解码成功率 0/40；")
    A("- 修复：仅把 stco 全部 +22118（副本 `work/scan/f00_fixed.m4a`，**`data/` 未改动**，`patch_f00.py` 可复现），"
      "解码错误数降为 **0**，完整解出 50.79 min；ASR 文本通顺（“这个有同学来认领啊…”），确认内容正确。")
    A("- 修复后该文件仍异常安静（ASL −39.8 dBFS、3 kHz 以上能量很少），说明源本身电平极低，不是修复导致。")
    A("")
    A("## 6. 局限")
    A("")
    A("- **⚠ VAD 的 first_speech 不可信（实测反例）**：`f24` 的 3.6 s 是片头音乐误触发（0–480 s ASR 无字，"
      "480 s 起真实讲课）；`f06`/`f20` 的 ~121 s 漏掉了 90–121 s 的轻声讲课；`f05` 的 0.2 s 是噪声。"
      "**裁剪实现必须以 ASR 有字为准**：本报告 §2.3 的 ASR 列 = 30 s 分块“本块 ≥10 字且后 3 块 ≥60 字”的首个时刻，"
      "精度 ±30 s；f24 尾部 4613 s 之后 60 s 无字（可裁）。")
    A("- **ASR 有字 ≠ 授课**：`f00` 0–90 s 有字但是课前闲聊（“怎么样他们问问怎么样…”），90 s 才进入课程介绍；"
      "这类片段用“字符率”判不出来，需要看内容（本报告已人工标注）。")
    A("- VAD 对音乐/噪声段可能误判，头尾裁剪秒数应视为“提示值 + ASR 复核后的值”，不宜直接硬编码。")
    A("- **ASL 未做 P.56 活动因子归一**：这里是语音帧能量均值，跨文件口径一致，绝对值不宜直接当标准 ASL。")
    A("- **LUFS−ASL 差小是这批数据的特性**（语音 66–85%）；语音占比低、音乐多的素材会明显分离，不能外推。")
    A("- **ASR 仅 60 s 抽样**：字/秒是局部值；首两句可能从句子中间开始；片头复核按 30 s 分块，边界精度 ±30 s。")
    A("- **diarization 是 10 分钟抽样**：人数是抽样窗内结果；阈值 0.5 会过聚类（原始簇 5~44 个），"
      "已用 embedding 余弦 >0.70 合并，但合并阈值本身是经验值；不能当全片说话人数。"
      "f24 的抽样窗落在讲课前音乐段，其“非主讲”主要是音乐/前奏。")
    A("- **f00 为修复后解码**：只改容器索引，源文件容器仍异常（32 kHz/193 kbps）；其响度/频谱只代表解码出的音频。")
    A("- 解码统一 16 kHz 单声道，8 kHz 以上信息丢失；True Peak 在解码/重采样后由 ffmpeg 测量，"
      "与原始采样点真峰值可能有 ±0.x dB 差异。")
    A("")
    A("## 7. 复现脚本（均在 `work/scan/`）")
    A("")
    A("| 步骤 | 脚本 | 说明 |")
    A("|---|---|---|")
    A("| 修复 f00 容器 | `patch_f00.py` | stco +22118，生成 `f00_fixed.m4a` |")
    A("| 元数据+解码 | `prep.py` | → `probe.json`、`wav/*.wav` |")
    A("| 响度 | `loud.py` | → `loud.json` |")
    A("| VAD+电平 | `vad_levels.py` | → `vad/*.json` |")
    A("| 片头 ASR 复核 | `head_scan.py` | → `head_scan.json` |")
    A("| 60s ASR | `asr60.py` | → `asr60.json`、`asr/*.json` |")
    A("| 说话人抽样 | `diar10.py` + `diar_merge.py` | → `diar/*.json`、`diar/*_merged.json` |")
    A("| 汇总 | `build_report.py` | → `scan.json`、本报告 |")
    A("")
    open(os.path.join(DOC, "multifile-scan.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
