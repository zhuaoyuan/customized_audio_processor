"""时间轴转写 PoC 公共工具：音频解码、Silero VAD(10ms 帧)、分段后处理、JSON IO。"""
import json
import os
import subprocess
import wave

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
VAD_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "venv", "Lib", "site-packages",
                         "faster_whisper", "assets", "silero_vad_v6.onnx")
SENSEVOICE_DIR = os.path.join(ROOT, "work", "bench", "models", "sensevoice")
DIAR_SEG_MODEL = os.path.join(ROOT, "work", "diar", "models",
                              "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
DIAR_EMB_MODEL = os.path.join(ROOT, "work", "diar", "models",
                              "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
WHISPER_ROOT = os.path.join(ROOT, "work", "bench", "models", "whisper")


def rel(p):
    """路径统一成相对 ROOT 的形式（写进 json 便于人读）。"""
    p = os.path.normpath(os.path.abspath(p))
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return p


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p, obj):
    ensure_dir(os.path.dirname(os.path.abspath(p)))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def decode_to_wav(src, dst, sr=16000, channels=1):
    """用 ffmpeg 把任意音频解码成 PCM16 wav（时间轴与原始一致，不做任何裁剪/变速）。"""
    ensure_dir(os.path.dirname(os.path.abspath(dst)))
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
           "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", dst]
    subprocess.run(cmd, check=True)
    return dst


def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ---------------------------------------------------------------------------
# Silero VAD：10 ms 帧级语音概率
# ---------------------------------------------------------------------------
class SileroVAD:
    """faster-whisper 自带的 silero_vad_v6.onnx：原生 32ms 窗流式帧级概率。

    模型输入是 (seq_len, 576) 的**时序序列**（64 样本上下文 + 512 样本窗，16k 下 32ms），
    h/c 是单一 LSTM 状态，按序推进。因此它天然只能给"每 32ms 一个概率"。实测：
      * 按 512 样本 hop 顺序喂（与 faster-whisper 官方实现一致）→ 正确；
      * 按 160 样本 hop 喂（试图把帧率提到 10ms）→ LSTM 状态演化快 3.2 倍，概率崩塌
        （60s 样本上 p>=0.5 占比从 55% 掉到 2%），不可用。
    所以这里用官方 32ms 流式概率 + 10ms 能量谷精修边界（refine_boundaries），
    最终段边界落在 10ms 网格上；不采用"字符比例估时间"。
    """

    WIN = 512
    CTX = 64
    HOP = 512          # 32 ms @16k（silero 原生窗移）
    BATCH = 4096       # 每次喂给 onnxruntime 的窗数（控制内存）

    def __init__(self, model_path=None, threads=1):
        import onnxruntime
        path = model_path or VAD_MODEL
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = threads
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        self.sess = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"],
                                                 sess_options=opts)
        self.path = path

    def probs(self, audio):
        """返回 float32[ceil(len/512)]，第 i 帧对应音频窗 [i*512, i*512+512)（中心 ~i*32ms+16ms）。"""
        x = np.asarray(audio, dtype=np.float32)
        n = max(1, int(np.ceil(len(x) / self.HOP)))
        pad = np.pad(x, (self.CTX, self.WIN + self.HOP), mode="constant")
        h = np.zeros((1, 1, 128), dtype=np.float32)
        c = np.zeros((1, 1, 128), dtype=np.float32)
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, self.BATCH):
            j0 = i
            j1 = min(n, i + self.BATCH)
            rows = np.empty((j1 - j0, self.CTX + self.WIN), dtype=np.float32)
            for k, j in enumerate(range(j0, j1)):
                base = j * self.HOP + self.CTX
                rows[k, :self.CTX] = pad[base - self.CTX:base]
                rows[k, self.CTX:] = pad[base:base + self.WIN]
            p, h, c = self.sess.run(None, {"input": rows, "h": h, "c": c})
            out[j0:j1] = p.reshape(-1)[:j1 - j0]
        return out


def frame_rms(audio, sr, win=0.025, hop=0.010):
    """10ms hop 短时 RMS（win 25ms），用于边界精修。"""
    x = np.asarray(audio, dtype=np.float32)
    w = int(sr * win)
    h = int(sr * hop)
    n = max(1, int(np.ceil(len(x) / h)))
    pad = np.pad(x, (0, w))
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = pad[i * h:i * h + w]
        out[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
    return out


def refine_boundaries(segs, audio, sr=16000, hop=0.010, search=0.18):
    """把每段起止吸附到 ±search 内 10ms RMS 的局部最小值（能量谷）。

    VAD 的 32ms 窗概率天然有 ~32–64ms 的滞后/提前；吸附到能量谷可把边界落到
    10ms 网格上，同时避免切在字中间。返回新段列表（不改变段数）。
    """
    if not segs:
        return segs
    rms = frame_rms(audio, sr, hop=hop)
    nf = len(rms)
    out = []
    prev_e = -1.0
    for i, (s, e) in enumerate(segs):
        # start：在 [s-search, s+0.08] 找最小能量点
        a = max(0, int(np.floor((s - search) / hop)))
        b = min(nf - 1, int(np.ceil((s + 0.08) / hop)))
        if b > a:
            ns = (a + int(np.argmin(rms[a:b + 1]))) * hop
        else:
            ns = s
        # 下限：不比上一段结束更早、不早于 0
        ns = max(ns, prev_e + 0.05, 0.0)
        # end：在 [e-0.08, e+search] 找最小能量点
        a2 = max(0, int(np.floor((e - 0.08) / hop)))
        b2 = min(nf - 1, int(np.ceil((e + search) / hop)))
        if b2 > a2:
            ne = (a2 + int(np.argmin(rms[a2:b2 + 1]))) * hop
        else:
            ne = e
        ne = max(ne, ns + 0.2)
        # 与下一段留 50ms 余量
        if i + 1 < len(segs):
            ne = min(ne, segs[i + 1][0] - 0.03)
            ne = max(ne, ns + 0.2)
        out.append([round(float(ns), 3), round(float(ne), 3)])
        prev_e = ne
    return out


# ---------------------------------------------------------------------------
# 帧级概率 -> 语音段（带回差、最短语音/静音、最长段切分）
# ---------------------------------------------------------------------------
def probs_to_segments(probs, hop=0.01, threshold=0.5, neg_threshold=0.35,
                      min_speech=0.50, min_silence=0.35, max_speech=30.0,
                      pad=0.12, total_dur=None):
    """把 10ms 帧概率转成语音段列表。

    规则（faster-whisper/silero 的硬回差思路，改成 10ms 帧）：
      * prob >= threshold 起段；prob < neg_threshold 记"候选静音"
      * 候选静音持续 >= min_silence 才结束段（短停顿不断句）
      * 语音段 < min_speech 丢弃（之后由 post_merge 并回邻近段）
      * 段 > max_speech 时，在 [max_speech-8s, max_speech] 内找概率最低点强制切开
      * 两端各 pad 秒（补出音头/音尾，ASR 解码用；报告边界仍用回差边界±pad）
    """
    p = np.asarray(probs, dtype=np.float32)
    n = len(p)
    dur = total_dur if total_dur is not None else n * hop
    min_speech_f = max(1, int(round(min_speech / hop)))
    min_silence_f = max(1, int(round(min_silence / hop)))
    max_speech_f = max(1, int(round(max_speech / hop)))

    segs = []
    triggered = False
    start = 0
    sil_start = -1
    for i in range(n):
        v = p[i]
        if not triggered:
            if v >= threshold:
                triggered = True
                start = i
                sil_start = -1
            continue
        if v < neg_threshold:
            if sil_start < 0:
                sil_start = i
            if i - sil_start + 1 >= min_silence_f:
                segs.append((start, sil_start))
                triggered = False
                sil_start = -1
        else:
            sil_start = -1
        # 最长段强制切
        if triggered and (i - start) >= max_speech_f:
            # 近 8s 内最低概率处切（避开把词切半）
            lo = max(start + int(4.0 / hop), i - int(8.0 / hop))
            j = int(np.argmin(p[lo:i + 1])) + lo
            cut = max(lo + 1, j) if j > lo else i
            if cut - start >= min_speech_f:
                segs.append((start, cut))
                start = cut
                sil_start = -1
    if triggered:
        end = sil_start if sil_start > start + min_speech_f else n
        segs.append((start, end))

    # 帧 -> 秒，补 pad
    out = []
    for a, b in segs:
        s = a * hop
        e = min(dur, b * hop)
        s = max(0.0, s - pad)
        e = min(dur, e + pad)
        out.append([round(s, 3), round(e, 3)])
    return out


def post_merge(segs, max_gap=0.30, min_dur=0.50, max_dur=30.0):
    """段后处理：<min_dur 的碎段并入邻近段（gap 内），相邻 gap<max_gap 的段合并。"""
    if not segs:
        return []
    segs = sorted([list(s) for s in segs], key=lambda s: s[0])
    changed = True
    while changed:
        changed = False
        merged = [segs[0]]
        for s in segs[1:]:
            if s[0] - merged[-1][1] <= max_gap:
                merged[-1][1] = max(merged[-1][1], s[1])
                changed = True
            else:
                merged.append(list(s))
        segs = merged
        # 碎段：并入前后 gap 更小的一侧
        if len(segs) <= 1:
            break
        keep = []
        for i, s in enumerate(segs):
            if (s[1] - s[0]) < min_dur:
                changed = True
                if i == 0:
                    segs[1][0] = min(segs[1][0], s[0])
                elif i == len(segs) - 1:
                    segs[i - 1][1] = max(segs[i - 1][1], s[1])
                else:
                    gap_prev = s[0] - segs[i - 1][1]
                    gap_next = segs[i + 1][0] - s[1]
                    if gap_prev <= gap_next:
                        segs[i - 1][1] = max(segs[i - 1][1], s[1])
                    else:
                        segs[i + 1][0] = min(segs[i + 1][0], s[0])
            else:
                keep.append(s)
        segs = keep if keep else segs
    # 超长段再切（post_merge 合并后可能 > max_dur）
    out = []
    for s, e in segs:
        while e - s > max_dur:
            out.append([round(s, 3), round(s + max_dur, 3)])
            s = s + max_dur
        if e - s >= min_dur:
            out.append([round(s, 3), round(e, 3)])
    return out


def fmt_mmss(t, with_hours=False):
    t = max(0.0, float(t))
    if with_hours or t >= 3600:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


def fmt_srt(t):
    t = max(0.0, float(t))
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
