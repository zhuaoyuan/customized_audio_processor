# 本地音频处理基线实测（ASR + 降噪，RTF / 内存 / 前后指标）

> 实测环境：Windows 10，Intel i5-12400F（6C/12T），32GB RAM，RTX 3060 Ti 8GB（驱动 560.94，CUDA Toolkit 12.6 已装），Python 3.10.6，ffmpeg（WinGet 版）。
> 全部安装隔离在 `work/venv`，未装 torch。原始音频：`data/24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a`（44.1k 立体声 AAC 55.8kbps，**4673.3s = 77.9 分钟**）。
> 本文所有数字均为本机实测，脚本在 `work/bench/`，可复现。

## 0. 结论速览（TL;DR）

| 结论 | 实测依据 |
|---|---|
| **不装 torch 完全可行**：SenseVoiceSmall(int8 ONNX, sherpa-onnx) + faster-whisper(CT2) + ONNX 降噪全部跑通 | 见 §2/§3/§4 |
| **SenseVoice 是当前最优性价比**：RTF **0.036–0.046**（4~8 线程），78 分钟整片实测 **2.34 分钟**（30s 分块+8 线程），中文质量明显好于 whisper tiny/base/small | §2/§6 |
| **多进程并行无收益**：4 进程×4 线程并行实测 171.2s，比单进程 8 线程 140.5s **更慢（0.77x）**，6 物理核已被单进程吃满 | §6 |
| **本机有可用的 CUDA**：faster-whisper base CUDA fp16 预热后 RTF **0.0129**，78 分钟 **~1 分钟**；但 SenseVoice 的官方 wheel 是 CPU-only，用不到 GPU | §2.4/§3 |
| **ffmpeg 降噪几乎免费**：afftdn RTF **0.003**（78 分钟 ~14 秒），arnndn RTF 0.011（~52 秒） | §4 |
| **ONNX 降噪（GTCRN）体积 0.51MB、RTF 0.049**，噪声底噪压 38dB 最狠，但会吞字（识别字符 195→161） | §4.1/§4.3 |
| 整条链（解码+降噪+SenseVoice 4 线程）处理 78 分钟：**≈ 3.7 分钟**，纯 CPU | §6 |
| 网络：pip 清华镜像与 hf-mirror 实测 **4.2–5.8 MB/s**，1GB 模型约 3 分钟 | §1 |

## 1. 环境与下载能力实测

### 1.1 机器
| 项 | 值 |
|---|---|
| CPU | 12th Gen Intel Core i5-12400F，6 核 12 线程，2.5GHz 基频 |
| GPU | NVIDIA GeForce RTX 3060 Ti，**8192 MiB**（`nvidia-smi`），驱动 560.94 |
| CUDA | **已安装 CUDA Toolkit 12.6**（`CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`） |
| Python | 3.10.6；`work/venv` 独立环境（pip 26.2.1） |
| ffmpeg | WinGet 版，`afftdn` / `anlmdn` / `arnndn` 滤镜均可用 |

### 1.2 下载速度（清华镜像 / hf-mirror）
| 来源 | 内容 | 体积 | 耗时 | 速率 |
|---|---|---|---|---|
| `pypi.tuna.tsinghua.edu.cn` | scipy-1.15.3 cp310 win_amd64 | 41.3 MB | 9.3s（pip 报传输 5.8 MB/s） | **4.2–5.8 MB/s** |
| `hf-mirror.com` | SenseVoiceSmall `model.int8.onnx` | 228.2 MB | 47.0s | **4.85 MB/s** |
| `raw.githubusercontent.com` | rnnoise `sh.rnnn` | 297 KB | <1s | 直连可用 |
| `github.com/.../releases` | GTCRN `gtcrn_simple.onnx` | 536 KB | <1s | 直连可用 |

**结论**：有效带宽 ~40–50 Mbps，下载不是瓶颈（1GB 约 3–4 分钟）；hf-mirror + 清华 pip 镜像都可用。唯一坑是 huggingface_hub 2.x 默认走 `hf-xet`，从镜像下 faster-whisper 模型会报 `OSError: I/O error: 拒绝访问 (os error 5)`，**必须设 `HF_HUB_DISABLE_XET=1`**（见 §7 坑 2）。

### 1.3 venv 内安装清单
`sherpa-onnx 1.13.8`（+ `sherpa-onnx-core`，内含 onnxruntime CPU provider）、`numpy 2.2.6`、`scipy 1.15.3`、`matplotlib 3.10.9`、`psutil 7.2.2`、`faster-whisper 1.2.1`、`ctranslate2 4.8.3`、`onnxruntime 1.23.2`、`huggingface-hub 2.2.0`。**未安装 torch**。

## 2. ASR 基线 A：sherpa-onnx + SenseVoiceSmall（ONNX int8，CPU）

模型：`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`
| 文件 | 体积 | 说明 |
|---|---|---|
| `model.int8.onnx` | **228.2 MB** | 本方案使用 |
| `model.onnx`（fp32） | 937 MB | 未下载（int8 够用，省 700MB 下载） |
| `tokens.txt` | 316 KB | — |

输入 `work/samples/mono_600_60.wav`（16k 单声道，60.00s 讲座正文）。

| 配置 | 加载耗时 | 解码耗时(60s) | **RTF** | 78 分钟外推 | 峰值 RSS |
|---|---|---|---|---|---|
| 4 线程，整段 60s | 1.10s | 2.77s | **0.0462** | 3.60 min | 483 MB |
| 8 线程，整段 60s | 1.15s | 2.19s | **0.0364** | 2.84 min | 428 MB |
| 4 线程，按 2×30s 切块 | 1.10s | 2.05s | **0.0355** | 2.77 min | 415 MB |

- 模型加载后 RSS ≈ 330 MB（含加载的 228MB 权重），60s 解码峰值 415–483 MB；**78 分钟整片（分块 8 线程）峰值 RSS 985 MB**（长音频分块解码 + 文本累积）。**显存占用 0**（CPU-only）。
- 切块与整段识别结果基本一致（195 vs 199 字），说明 60s 整段喂入没有明显截断/退化。

### 2.1 识别文本（前 2–3 句，实际输出，`work/bench/asr_sv_t4.json`）

> 来来比比那个听政治哲学的人还多，形而上学的应该是没什么人感兴趣的。刘晨光，你还笑，你上上这课，你不是打瞌睡吗？真没什么好看的这样这个今天呢我们整个的这个安排呃我认为大体上把这个就是原来我以上课的这个惯例啊，按照上课的惯例，我把这个每每卷的这个承担者就是主讲的主讲的同学先给给大家分一下。那么然后我我自己再稍微的讲一讲呃形而上学这啊今天这个课的名字叫叫什么原助选读是吧？题是专题研究不管他了。

**质量判断**：内容确实是哲学系课堂开场（点名、"政治哲学"、"形而上学"、"主讲同学"、课程名"原著选读"被打成"原助选读"）、提到课程安排与"专题研究"。**带标点**（`use_itn=1` 生效，逗号/问号/句号齐全），中文为主，少量同音错字。对"成本评估"来说这个质量已经够用。

### 2.2 整片（78 分钟）实测文本样例

对 `work/full_16k.wav`（4673.3s，30s 分块 + 8 线程）**整片真跑**，产出 17040 字（`work/bench/asr_sv_full78min_chunk30_t8.json`）：

- 开头（0–30s，片头音乐/环境声）：
  > 去做。。。。。。。。。..。。。。。起迟到了啊呃，还是这么多人啊小小航，你你那个教研室就肯定是做不下来啊，怎吧。啊，是呃但是这个教室系里现在不大愿意介绍来啊，这个不管了，我们下一次再再说吧。啊教室是这样，就是教室是给本科生上课用的我我我我们这个呃研究生课是神密大师这个研讨班。
- 中段（课程安排）：
  > ……这个可能可会到9点半啊，我不会按这个课程表上说，就是两两两节课啊，我所以喜欢在这不喜欢在教室上课，这是另外一。个理由我我不按他那个打铃的这个这个节奏来的啊，按我们自然的这个节节节奏来……
- 结尾：
  > ……好，这个要不今天今天先先先这样。就我们第二。下星期开始就要就要开始这个啊火炬接力啊，就是李亚国说的火炬接力啊，其他同学到小路这登。

**注意**：头部纯音乐段被识别出 `去做。。。。。。。。`（一串标点），这是 SenseVoice + ITN 在非语音段的典型伪输出。生产链路建议**先过 VAD 切分**（或过滤"标点占比 > 60%"的段），既省算力又避免垃圾文本。

### 2.3 其他样本观察
| 样本 | ASR 输出 | 判断 |
|---|---|---|
| `s1_mid.wav`（30s 讲座正文） | 正常长句 | 适合做降噪评估 |
| `s0_head.wav`（0–30s） | `去做。` | 基本无语音；动态范围仅 ~9dB、持续能量 → 疑似片头音乐/掌声 |
| `s2_tail.wav`（片尾 30s） | `些有几个但以到。` | 同上，疑似片尾音乐/掌声 |

结论：`s0_head`/`s2_tail` 不适合当"语音降噪"样本，用 `s1_mid`（正文）与 `mono_600_60` 才合理。

### 2.4 GPU 结论（SenseVoice 用不到 GPU）
PyPI 的 `sherpa-onnx` wheel 是 **CPU-only** 编译。指定 `provider="cuda"` 时实测报错：

```
session.cc:GetSessionOptionsImpl:355 Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON.
Available providers: CPUExecutionProvider, . Fallback to cpu!
```

→ 自动回退 CPU。**SenseVoice 想上 GPU 必须自行用 `SHERPA_ONNX_ENABLE_GPU=ON` 编译**（需要 VS 编译链），本次不做。好在 CPU 上 78 分钟只要 2.8–3.6 分钟，GPU 对这条链路没有必要性。

## 3. ASR 对照 B：faster-whisper（CTranslate2，不需 torch）

模型来源 `Systran/faster-whisper-*`（经 `HF_ENDPOINT=https://hf-mirror.com` 下载），同一 `mono_600_60.wav`。

| 模型 | 设备/精度 | beam | 模型体积 | 加载 | 峰值 RSS | 解码 60s | **RTF** | 78 分钟 | 峰值显存 |
|---|---|---|---|---|---|---|---|---|---|
| tiny | CPU int8 | 5 | 74.6 MB | 2.62s | 268 MB | 5.83s | **0.0972** | 7.6 min | — |
| tiny | CPU int8 | 1 | 74.6 MB | 2.63s | 269 MB | 2.68s | **0.0446** | 3.5 min | — |
| base | CPU int8 | 5 | 141.0 MB | 2.89s | 313 MB | 2.67s | **0.0445** | 3.5 min | — |
| base | CPU int8 | 1 | 141.0 MB | 2.89s | 311 MB | 1.94s | **0.0324** | 2.5 min | — |
| base | **CUDA fp16**（冷启，含初始化） | 5 | 141.0 MB | 2.75s | 687 MB | 2.49s | 0.0415 | 3.2 min | ~1.9 GB（整卡占用） |
| base | **CUDA fp16**（预热后） | 5 | 141.0 MB | — | 687 MB | **0.77s** | **0.0129** | **1.0 min** | ~1.9 GB（整卡占用） |
| small | **CUDA fp16**（预热后） | 5 | 463.7 MB | 144.2s（含 484MB 下载） | 659 MB | **1.33s** | **0.0221** | **1.73 min** | ~2.4 GB（整卡占用，估） |

> tiny 的 RTF 比 base 还差（0.097 vs 0.0445）不是测错：tiny 在 beam5 下出现重复幻觉（见下），解码 token 数暴涨，导致耗时反而更高。**"小模型=快"不成立，要看是否触发复读**。

### 3.1 文本质量对比（同 60s）

- **base（CPU int8, beam5）**：
  > 来来比听证者的人还多形儿上学的应该是没什么人感兴趣的没有人问你还笑你上这课你不是打个这个听没什么好看大家好 今天我们整个的安排我是为大体上把这个就会原来我一上课的这个管理按照上课的管理我把这个每一圆的这个承担者就是主讲的主讲的同学先给大家分一下那么然后我自己在稍微的讲一讲卿尔上学 今天这个课的名字叫圆株选读是吧专体研究不管他了
  → 同音错字多（形而上学→"形儿上学"/"卿尔上学"，原著选读→"圆株选读"），无标点（ITN 弱），没有明显复读。
- **tiny（CPU int8, beam5）**：
  > 雖然比那个轻重者格局的人还多形怀上形怀应该是没什么人感兴趣的留着我,你还笑着一下像这块,你不是打个事吗你看,你看很清楚,没什么好看叫这个...
  > ...请问上课今天这个课的名字叫研农选读车啊行行行行行行行行行行行行行行行赚笔吧赚笔研就啊赚笔赚笔吧不管他了
  → **繁体/简体混杂 + 尾部整段复读幻觉**，不可用于生产。
- **small（CUDA fp16, beam5）**：
  > 来来来 比那个听政治哲学的人还多行业上学嘛应该是没什么人感兴趣的刘仁光 你还笑着你看 上这一课你不是打个这个吗哈哈哈 看着哈哈哈 这没什么好看的这样 这个 今天呢我们整个的这个安排呃 我说大体上把这个原来我按照上课的惯例我把每一卷的承担者主讲的同学先给大家敦意一下然后我自己再稍微得讲一点行业上学这个课的名字叫什么沿路选读是吧是专体研究 不管它了
  → 比 base 略好但**关键术语仍错**（形而上学→"行业上学"，原著选读→"沿路选读"），还有"哈哈哈"幻觉；模型 464MB 是 SenseVoice 的 2 倍，反而没 SenseVoice 准。
- **CUDA fp16 base**：文本与 CPU int8 base 仅开头 1-2 字差别（"来来比听证者的人还多" → "原来比听证者的人还多"），说明 GPU 精度切换对中文质量影响可忽略。

### 3.2 三个 ASR 方案结论

| 维度 | SenseVoiceSmall ONNX | faster-whisper base | faster-whisper small |
|---|---|---|---|
| 中文质量 | **最好**（错字少、带标点） | 一般（同音错字多、无标点） | 一般（关键术语仍错，有幻觉） |
| CPU RTF | 0.036–0.046 | 0.0445（beam5）/ 0.0324（beam1） | 未测（CPU 上应 ~0.1+） |
| GPU RTF | 官方不支持 | **0.0129（预热）** | 0.0221（预热） |
| 模型体积 | 228 MB | 141 MB | 464 MB |
| 内存 | 415–483 MB | 313 MB | 659 MB |

（faster-whisper tiny 未单列：beam5 下复读幻觉严重、beam1 也出现繁简混杂，已排除。）

**建议**：先用 SenseVoice 走 CPU（够快够准），需要更快/更省 CPU 再考虑 faster-whisper base + CUDA fp16。

## 4. 降噪基线（`work/samples/s1_mid.wav`，30.00s，讲座正文）

度量口径（`work/bench/audiometrics.py`）：25ms 帧 / 10ms 跳，帧 RMS 分位数；`noise_floor` = 最安静 10% 帧的平均能量，`speech_level` = 最响 10% 帧的平均能量，`SNR估计 = speech_level − noise_floor`。**这是无参考的近似指标，不是真实 SNR**（真实 SNR 需要干净语音参考）。原始文件：底噪 −30.9 dBFS、SNR估计 24.1 dB、RMS −14.8 dBFS。

### 4.1 处理前后指标表（全部为 16k/mono/pcm_s16le 输出）

| 方案 | 滤镜/模型 | 底噪(dBFS) | Δ底噪 | P10(dBFS) | P90(dBFS) | SNR估计(dB) | ΔSNR | 耗时(30s) | RTF | 78分钟外推 |
|---|---|---|---|---|---|---|---|---|---|---|
| 原始 | — | −30.87 | — | −30.10 | −9.16 | 24.06 | — | — | — | — |
| **afftdn nr=12** | `afftdn=nr=12:nf=-30:tn=1` | −36.01 | **+5.1** | −34.80 | −9.26 | 29.19 | +5.1 | 0.10s | **0.003** | **14 s** |
| **afftdn nr=20** | `afftdn=nr=20:nf=-25:tn=1` | −44.68 | **+13.8** | −42.55 | −9.34 | 37.81 | +13.7 | 0.10s | **0.003** | 14 s |
| anlmdn s=1e−3 | `anlmdn=s=0.001:p=0.002:r=0.006` | −30.86 | +0.0 | −30.14 | −9.10 | 24.06 | 0.0 | 0.27s | 0.009 | 38 s |
| anlmdn s=1e−2 | `anlmdn=s=0.01:...` | −30.86 | +0.0 | −30.14 | −9.10 | 24.06 | 0.0 | 0.26s | 0.009 | 37 s |
| anlmdn s=5e−2 | `anlmdn=s=0.05:...` | −30.86 | +0.0 | −30.14 | −9.10 | 24.06 | 0.0 | 0.26s | 0.009 | 37 s |
| afftdn+anlmdn | `afftdn=nr=12:...,anlmdn=...` | −36.01 | +5.1 | −34.79 | −9.26 | 29.17 | +5.1 | 0.31s | 0.010 | 45 s |
| highpass80+afftdn | `highpass=f=80,afftdn=nr=12:...` | −36.21 | +5.3 | −34.95 | −9.27 | 29.36 | +5.3 | 0.11s | 0.004 | 16 s |
| **arnndn (rnnoise sh)** | `arnndn=m=sh.rnnn`（297KB） | **−53.47** | **+22.6** | −52.00 | −10.90 | **45.68** | +21.6 | 0.33s | **0.011** | **52 s** |
| highpass80+arnndn | `highpass=f=80,arnndn=...` | −53.38 | +22.5 | −51.95 | −10.90 | 45.58 | +21.5 | 0.30s | 0.010 | 47 s |
| afftdn+arnndn | `afftdn=nr=12:...,arnndn=...` | −52.62 | +21.8 | −51.06 | −15.36 | 41.48 | +17.4 | 0.37s | 0.012 | 56 s |
| **GTCRN ONNX** | sherpa-onnx `OfflineSpeechDenoiser`（0.51MB） | **−69.27** | **+38.4** | −66.81 | −11.08 | **60.66** | +36.6 | 1.48s | **0.0493** | **3.85 min** |

### 4.2 关键读法
- **anlmdn 在本任务里完全无效**（P10/底噪/SNR 与原始一模一样，三档强度都一样）：它是非局部均值平滑，主要去"沙沙"细噪且默认参数偏保守；对本样本底噪（−30.9dBFS 的宽带底噪 + 55.8kbps AAC 压缩噪声）不起作用。**不建议用**。
- **afftdn 是性价比之王**：RTF 0.003，几乎零成本，`nr=12` 温和去掉 5dB 底噪，`nr=20` 去掉 14dB，但 `nr=20` 输出峰值顶到 0.0 dBFS（有削顶风险，落地时要加 limiter/降增益）。
- **arnndn（rnnoise）压制力最强**（+22.6dB 底噪、P10 −52dBFS），但整体电平被压低约 1.6dB（RMS −14.76→−16.31），需要后接增益补偿。
- **GTCRN ONNX** 模型只有 0.51MB、RTF 0.049（CPU 4 线程），底噪压到 −69dBFS（+38.4dB），是"最干净"的，但代价是吞字（见 §4.3）。
- **频谱证据**（见 §5 图）：宽带底噪主要在 1.5–4.5kHz（4.5kHz 以上是 AAC 55.8kbps 的带宽台阶，本身就是空的）；afftdn 把 5–8kHz 段压得最狠（−46→−58/−66dB），而 300–1000Hz（语音主能量）几乎不动（−0.1dB），说明没有伤到语音主体。

### 4.3 降噪对 ASR 的实际影响（客观代理：与原始转写文本的字符相似度）

| 降噪方案 | 识别字符数 | 与原始文本相似度 | 典型差异（人耳/语义） |
|---|---|---|---|
| 原始（无损） | 195 | 1.000 | 把"原著选读"错成"原助选读" |
| afftdn nr=12 | 190 | **0.925** | **"原著选读"识别对了**；"形而上"→"形为上学"、"刘晨光"→"刘春光" |
| arnndn sh.rnnn | 194 | **0.915** | **"原著选读"识别对了**；"形而上"→"形为上学"、"刘晨光"→"刘生光" |
| GTCRN ONNX | 161 | **0.871** | 吞字较多：漏"你不是打瞌睡吗你上上这课"、把"形而上学"打散 |

**结论**：对"人耳可听的降噪"三种方案都不会毁掉 SenseVoice 识别（0.87–0.93 相似度）；afftdn/arnndn 甚至让"原著选读"这个专名从错变对。但 GTCRN 太激进，会漏字，**不要拿它当前置再喂 ASR**。

## 5. 处理前后频谱图（PNG）

![频谱图对比（原始 vs afftdn nr12 vs afftdn nr20 vs arnndn vs GTCRN）](work/bench/spec_compare_s1_mid.png)

![长时平均频谱曲线（各自峰值归一）](work/bench/ltas_compare_s1_mid.png)

分频段能量（dB，相对各自长时平均谱峰值）——底噪主要在 5–8kHz 宽带：

| 方案 | 0–100Hz | 100–300Hz | 300–1000Hz | 1000–3000Hz | 3000–5000Hz | 5000–8000Hz |
|---|---|---|---|---|---|---|
| 原始 | −11.3 | −7.5 | 0.0 | −5.1 | −21.2 | **−46.0** |
| afftdn nr=12 | −12.1 | −7.9 | 0.0 | −5.9 | −23.0 | **−58.4** |
| afftdn nr=20 | −12.6 | −8.1 | 0.0 | −6.0 | −24.2 | **−66.2** |
| arnndn (rnnoise) | **−25.5** | −8.0 | 0.0 | −6.2 | −22.3 | −46.6 |
| GTCRN ONNX | −13.0 | −7.8 | 0.0 | −6.4 | −23.4 | −49.5 |

（0–100Hz 只有 arnndn 明显下降 → 它在去低频隆隆声；5–8kHz 由 afftdn 压得最狠。）

## 6. 78 分钟整片耗时（实测 + 外推，4673.3s）

| 环节 | 实测 RTF | 78 分钟预计耗时 |
|---|---|---|
| **实测整片（30s 分块 + 8 线程，`asr_sv_full78min_chunk30_t8.json`）** | **0.0301** | **2.34 min**（墙钟 140.5s / 纯解码 132.5s） |
| 解码 m4a→16k mono wav（ffmpeg） | 0.0005（2.20s） | **2.2 s** |
| ffmpeg 降噪 afftdn nr=12 | 0.003 | **14 s** |
| ffmpeg 降噪 arnndn | 0.011 | **52 s** |
| GTCRN ONNX 降噪（CPU 4 线程） | 0.0493 | **3.85 min** |
| **SenseVoice 4 线程** | 0.0462 | **3.60 min** |
| **SenseVoice 8 线程** | 0.0364 | **2.84 min** |
| faster-whisper base CPU int8 beam5 | 0.0445 | 3.47 min |
| faster-whisper base CPU int8 beam1 | 0.0324 | 2.53 min |
| **faster-whisper base CUDA fp16（预热）** | 0.0129 | **1.01 min** |
| faster-whisper small CUDA fp16（预热） | 0.0221 | 1.73 min |
| 整链：解码 + afftdn + SenseVoice 4 线程 | — | **≈ 3.7 min**（纯 CPU，无需 GPU） |
| 整链：解码 + GTCRN + SenseVoice 8 线程 | — | **≈ 6.7 min**（纯 CPU） |
| **多进程并行（4 进程 × 4 线程，`parallel_result.json`）** | 0.0366 | **2.85 min（比单进程 8 线程更慢）** |

> **并行实测（负面结果，重要）**：把 78 分钟切成 4×19.5min，用 4 个进程×4 线程并行，总墙钟 **171.2s**，比单进程 8 线程的 **140.5s 慢 22%**（加速比 0.77x）。原因：i5-12400F 只有 6 个物理核，SenseVoice 单进程 8 线程已经把物理核吃满，再多进程只会抢核 + 增加内存带宽争用。**结论：这台机器上"一个进程跑 8 线程"就是最优策略，不要开进程池。**
>
> 外推按"线性 RTF"计算，仅作量级参考；整片实测（2.34 min）比外推（2.84 min）还快约 18%。

## 7. 失败项与坑（都是实测踩出来的）

1. **sherpa-onnx 无法用 GPU**：PyPI wheel 为 CPU-only，`provider="cuda"` 直接回退 CPU，报 `Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON`。要 GPU 只能自己编译（需 VS 工具链，耗时较大，本次未做）。
2. **hf-mirror 下载 faster-whisper 模型报 `OSError: I/O error: 拒绝访问 (os error 5)`**：huggingface_hub 2.2.0 默认走 `hf-xet`，xet 后端在 Windows 上写缓存/日志失败。解决：`HF_HUB_DISABLE_XET=1`、`HF_HOME=<可写目录>`、`HF_ENDPOINT=https://hf-mirror.com`，设完 74.6/141MB 模型正常下载。
3. **ffmpeg `arnndn` 的模型路径不能直接写 Windows 绝对路径**（`arnndn=m=C:/...`）：`C:` 的冒号会破坏 filter 解析，报 `No option name near '/projects/...'`。解决：先 `cd` 到模型目录再用相对路径 `arnndn=m=sh.rnnn`（或在 filter 里转义冒号）。
4. **faster-whisper tiny 在 beam=5 下触发复读幻觉**（"行行行行……赚笔吧"），耗时反而比 base 高（RTF 0.0972 vs 0.0445）。选模型不能只看参数量。
5. **`sherpa_onnx.read_wave` 在 1.13.8 里不存在**（只有 `write_wave`），读 wav 要自己用 `wave`/scipy；`OfflineSpeechDenoiser.run()` 返回 `DenoisedAudio` 对象（`.samples`/`.sample_rate`），不能直接解包。
6. **s0_head/s2_tail 两个样本不是语音**（ASR 只出几个字、动态范围只有 ~9dB），它们更像片头/片尾音乐或掌声。做降噪评估只能用 s1_mid。
7. **非语音段会产生"标点垃圾"**：78 分钟整片开头 30s（片头音乐）被识别成 `去做。。。。。。。。。`。ITN 会把噪声映射成一串标点，必须在 ASR 前加 VAD 或做后处理过滤。
8. **多进程并行反而更慢**（0.77x，见 §6）——6 物理核的机器上不要开进程池。
9. 55.8kbps 的 AAC 源本身高频就到 ~4.5kHz 以上基本是噪声底（§5 频谱可见 4.5kHz 以上台阶），**这是编解码损失，任何降噪都救不回被压掉的语音高频**；若在乎音质，应从更高码率源转。

## 8. 产物清单（`work/bench/`）

| 文件 | 说明 |
|---|---|
| `spec_compare_s1_mid.png` | 处理前后频谱对比图（5 面板） |
| `ltas_compare_s1_mid.png` | 长时平均频谱曲线对比 |
| `denoise/s1_mid_*.wav` | 各降噪方案输出（afftdn/anlmdn/arnndn/combo/gtcrn） |
| `denoise/mono_600_60_*.wav` | 60s 样本的降噪输出（用于 ASR 对比） |
| `asr_sv_t4.json` / `asr_sv_t8.json` / `asr_sv_t4_chunk30.json` | SenseVoice 60s 结果（文本/RTF/内存） |
| `asr_sv_full78min_chunk30_t8.json` | **78 分钟整片实测结果**（17040 字，RTF 0.0301） |
| `asr_whisper_*.json` | faster-whisper 各配置结果（tiny/base/small，CPU int8 / CUDA fp16） |
| `denoise_s1_mid.json` / `denoise_gtcrn.json` | 降噪前后指标原始数据 |
| `asr_after_denoise.json` | 降噪前后识别文本与相似度 |
| `parallel_result.json` + `parallel/` | 4 进程并行吞吐测试（负面结果） |
| `run_asr_sensevoice.py` / `run_asr_whisper.py` / `run_denoise.py` / `run_denoise_onnx.py` / `run_parallel_test.py` / `plot_compare.py` / `audiometrics.py` / `sample_levels.py` | 可复现脚本 |
| `models/` | SenseVoice int8（228MB）、rnnoise sh.rnnn（297KB）、GTCRN（536KB）、whisper tiny/base/small、HF 缓存 |
