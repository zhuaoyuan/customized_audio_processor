# 本地开源方案调研：降噪 / 只保留主讲人 / 中文 ASR

> 目标机器：Windows 10 · RTX 3060 Ti **8 GB** · 32 GB RAM · Python 3.10.6 · ffmpeg 可用
> 素材：`data/24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a`
> 本文件只做**选型调研与事实核实**，未安装任何包、未运行任何模型。模型下载与试跑由后续任务执行。

标注约定：**✅ 已核实**（附来源链接，数字来自官方 README / 论文 / 模型卡）｜**⚠️ 估算**（由已核实 RTF 或参数量推导，需本机实测确认）｜**❓ 未核实**（公开卡片没写，勿当事实引用）。

---

## 0. TL;DR（先看这一段）

1. **"去环境噪声"和"只留主讲人"是两件不同的事，模型不能互换。**
   降噪模型（DNS 系）只学过"噪声"，没学过"另一个人的说话声"；把第二个人当噪声喂给降噪器，结果通常是**人声变形 + 背景更闷**，而不是人声消失。去他人声必须走 **TSE（目标说话人提取）** 或 **diarization + 只保留主讲人片段** 两条路。
2. **本机单声道混录（dualmix）场景下，最稳的"只留主讲人"不是 TSE，而是 diarization 切片。** TSE 需要一段干净的主讲人报名音频（enrollment），而这场讲座里主讲人音量/音色稳定、观众多为短促提问与咳嗽，把"非主讲人片段静音/丢弃"的收益/风险比明显优于把整段送进分离网络。
3. **降噪首选 DeepFilterNet3（pip 一行、MIT/Apache、48 kHz 全带、单线程 CPU RTF 0.19）**；追求 PESQ 再上 MossFormer2_SE_48K 或 FRCRN_SE_16K（ClearerVoice，Apache-2.0 代码）。
4. **中文 ASR 首选 FunASR Paraformer-large（220 M，原生时间戳 + 热词 + ct-punc 标点）**，噪声重/口音重时用 SenseVoice-Small（234 M，CER 更低、CPU 也快），最求鲁棒性用 Qwen3-ASR-1.7B（但需要额外 0.6 B 强制对齐模型才有时间戳）。faster-whisper large-v3 作为兜底，中文标点/长音频幻觉与热词能力弱于 FunASR 系。
5. **效果上限必须说清**：单声道、两人**同时说话**时，任何开源方案都无法"彻底去掉第二个人"；低码率 55 kbps AAC 源的有效带宽只到约 **14 kHz**（本机实测），高频已被编码器砍掉，降噪/分离无法"恢复"，只能"不进一步恶化"。

**推荐组合（详见第 7 节）**

| 组合 | 降噪 | 只留主讲人 | ASR | 依赖体积 | 78 min 预计总耗时 |
|---|---|---|---|---|---|
| **A 最快落地（推荐先跑）** | DeepFilterNet3 | pyannote community-1 切片 | FunASR Paraformer-large + ct-punc + cam++ | ≈ 8–10 GB | **10–20 min** |
| **B 质量优先** | MossFormer2_SE_48K | USEF-TSE（需报名音频）/ MossFormer2_SS_16K + 选路 | Qwen3-ASR-1.7B（+ForcedAligner） | ≈ 14–18 GB | 30–60 min |
| **C CPU 兜底 / 零 CUDA** | GTCRN 或 ffmpeg `arnndn`(RNNoise) | sherpa-onnx diarization 切片 | sherpa-onnx SenseVoice int8 | ≈ 2–3 GB | 30–50 min |

---

## 1. 本机与素材的客观事实（先定约束，再选模型）

### 1.1 容器与编码（ffprobe 实测 ✅）

| 项 | 值 |
|---|---|
| 容器/编码 | MP4/M4A，**AAC-LC** |
| 采样率 / 声道 | 44 100 Hz / 2 ch |
| 音频流码率 | **≈ 55.8 kbps**（整体 ≈ 59.1 kbps） |
| 时长 | **4673.29 s = 77.9 min**（≈ 78 min） |
| 体积 | 34.5 MB |

### 1.2 有效带宽（本机一次性实测 ⚠️ 单点采样）

对工作副本 `work/full_44k_stereo.wav` 取 2000 s 起 60 s，做长时平均功率谱，相对峰值：

| 频率 | 1k | 2k | 4k | 6k | 8k | 10k | 12k | 14k | 15k | 16k+ |
|---|---|---|---|---|---|---|---|---|---|---|
| 相对峰值 dB | −15.5 | −20.2 | −35.9 | −52.5 | −54.0 | −59.4 | −64.2 | −64.7 | **−94.1** | < −100 |

**结论**：15 kHz 处能量骤降 30 dB，−70 dB 滚降点 ≈ **14.25 kHz**。即编码器在约 14–15 kHz 做了低通，**8 kHz 以上相对能量已低 54–65 dB**。这决定了三件事：
- 任何"48 kHz 全带降噪"都不会让声音变亮（没有可恢复的高频），只能避免更闷；
- 高频段以**量化噪声/预回声**为主，降噪器容易把这些当"噪声"一起压掉 → 听感"更干净但更糊"；
- 想提升"清晰度/亮度"，需要的是**带宽扩展（super-resolution）**，不是降噪（见 8.3）。
- 另外该文件是 44.1 kHz 立体声，但若两声道近似相同（dualmix），则**没有空间信息可用于波束成形/空间分离**，只能做单通道算法——这会直接排掉一大类"多通道"方案。

> 上述第 3 点（是否 dual-mono）由音频分析任务确认；若确认 L≈R，则本文件永远只能走**单通道**路线（本文所有推荐都已按单通道给）。

### 1.3 外部下载可达性（安装风险的真实来源）

- **HuggingFace 域名在本机工具侧不可达**（抓取直接返回 "resolves to a non-public IP address"）。这不等于用户网络一定不通，但意味着**必须预设 fallback**：
  - FunASR 模型优先走 **ModelScope**（`iic/...`，README 默认就是 ModelScope ID）；
  - ClearVoice / faster-whisper / pyannote 权重若 HF 不通，用 `HF_ENDPOINT=https://hf-mirror.com` 或 ModelScope 镜像；
  - `pyannote` 的 gated 模型**必须**登录 HF 接受协议，镜像站通常不代理 gated 仓库 → 这是组合 A 唯一的硬依赖（见 5.1）。
- **CUDA 侧**：`faster-whisper`/CTranslate2 需要 **cuBLAS(CUDA 12) + cuDNN 9**；Windows 上官方给的办法是从 Purfview 的 `whisper-standalone-win` 版 libs 解压进 PATH（[faster-whisper README](https://github.com/SYSTRAN/faster-whisper#gpu)）。
- Python 3.10.6 对本文所有方案都满足（FunASR ≥3.8、faster-whisper ≥3.9、pyannote 需较新 torch）。

---

### 1.4 本机现有 Python 环境（只读观察，供排期参考）

`work/venv`（Python 3.10）已有 46 个包，与本议题相关的：
- **faster-whisper + CTranslate2**（CPU 可跑；GPU 需 cuBLAS 12 + cuDNN 9）
- **sherpa-onnx + sherpa-onnx-core + onnxruntime**（含 `sherpa-onnx-cli.exe`，可跑 SenseVoice / GTCRN / diarization）
- numpy / scipy / matplotlib / huggingface_hub / av(PyAV) / tokenizers
- **未安装**：torch / torchaudio / funasr / pyannote.audio / clearvoice / deepfilternet

含义：**组合 C 的底座已经就绪**，只差模型权重下载；**组合 A/B 需要另装 torch 系**（下载 ≈ 2.5 GB，安装后 ≈ 5–7 GB），建议**新建独立 venv**，不要污染这个 CPU 环境（CTranslate2 与 torch 的 CUDA 运行库版本容易互相顶掉）。

---

## 2. 任务边界：降噪 ≠ 去他人声（这是最容易踩的坑）

| | 降噪 (SE / DNS) | 只留主讲人 |
|---|---|---|
| 干扰类型 | 稳态/非稳态**噪声**：空调、风机、底噪、电流声、键盘、纸张、混响 | **另一个人的语音**（以及观众席人声） |
| 训练数据 | 干净语音 + 噪声（DNS / VoiceBank-DEMAND 等） | 干净语音 + **干扰说话人**（WSJ0-2mix / LibriMix / WHAM!），或"目标 + 干扰"成对数据 |
| 输入 | 单路混合 | 混合 + **目标说话人线索**（enrollment 音频 / 视频人脸 / 空间方向） |
| 典型输出副作用 | 音乐噪声、过度抑制、辅音变糊 | 目标说话人音色改变、串音残留（leakage）、分离伪影 |
| 能不能"去掉第二个人" | **不能**。第二个人声是"类语音瞬态"，多数 SE 模型会保留甚至放大 | 能"压掉大部分"，但同时说话时**压不干净**（见第 9 节） |

**推论（本项目的执行顺序）**：
```
源 (55k AAC) → [1] 降噪 DFN3 → [2] diarization（谁在什么时候说）→ [3a] 只保留主讲人片段
                                                          └→ [3b] 若必须"保留重叠段里的主讲人" → TSE（需 enrollment）
→ [4] ASR（建议在 16 kHz 单通道上跑）→ 文本/时间戳
```
顺序注意：**不要**"降噪 → 分离 → 再降噪"叠三层，伪影会累积；TSE/分离之后的音频通常已足够干净，最多再做一次轻量 DFN3（关闭 postfilter）。

---

## 3. A. 降噪候选（单通道、稳态噪声为主）

| 模型 | 采样率 | 参数量 | 速度 RTF | 许可证 | CPU/ONNX | 适配噪声与点评 |
|---|---|---|---|---|---|---|
| **DeepFilterNet3 (DFN3)** | 48 kHz 全带 | ≈ 1.8–2 M 级 ❓（DFN1 官方对比表记 1.80 M ✅） | **0.19**（单线程笔记本 CPU）✅；GPU 更快 ⚠️ | **MIT / Apache-2.0 双许可** ✅ | ✅ pip wheel 自带 Rust `libdf`；也可 `deep-filter` 二进制 | **稳态噪声（空调/底噪/风扇/电流声）首选**。48 kHz 全带、psychoacoustic 损失，对低码率源不会强行"造"高频，听感自然、伪影少。VoiceBank+DEMAND PESQ 3.03（ClearVoice 复测，48 kHz 输入） |
| **RNNoise** | 48 kHz | **0.06 M** ✅ | 极低（CPU 实时绰绰有余）⚠️ | BSD-3（xiph） | ✅ **ffmpeg 内置 `arnndn` 滤镜**，无需 Python | 只适合"轻噪声 + 语音通话"。PESQ 2.29（同表），明显弱于 DFN3/GTCRN；优点是零依赖、可整段 ffmpeg 流水线处理 |
| **GTCRN** | 16 kHz | **48.2 K 参数 / 33 MMACs** ✅ | **0.07**（i5-12400 流式）✅ | 开源（仓库 MIT 类）✅ | ✅ **sherpa-onnx 已支持**，ONNX 极轻 | 超轻量，PESQ 2.87 on VCTK-DEMAND，**优于 RNNoise**，接近 DFN1。**CPU 兜底首选**；16 kHz 限制使其更适合"给 ASR 用"，不适合"保听感" |
| **FRCRN_SE_16K**（ModelScope/ClearerVoice） | 16 kHz | 约 6 M 级 ❓ | GPU 快 ⚠️（<0.05 量级） | 代码 Apache-2.0 ✅；权重以卡片为准 ❓ | 部分 ONNX 导出可用 | PESQ **3.23**（VoiceBank+DEMAND）/ **3.24**（DNS-2020）✅，**降噪强度最高**之一；48 kHz 源需先降到 16 kHz，会二次损失高频。适合"先降噪再 ASR"，不适合"保留音质" |
| **MossFormer2_SE_48K**（ClearerVoice） | 48 kHz | 未公开 ❓（Transformer 量级，数十 M） | GPU 中等 ⚠️（估 RTF 0.05–0.15） | 代码 Apache-2.0 ✅；权重卡片未统一标注 ❓ | ❌ 需 PyTorch GPU 较舒服 | PESQ 3.15 / STOI 0.95（48 kHz）✅，**保真度最好**（SRMR 9.61 最高，说明失真最少）；速度与显存是代价 |
| **MossFormerGAN_SE_16K** | 16 kHz | 未公开 ❓ | 中等 ⚠️ | 同上 | 一般 | PESQ **3.47** / DNS PESQ 3.57，指标最高 ✅，但 GAN 系在**低码率 + 强压缩源上更容易出"金属音/音乐噪声"**，本素材不优先 |
| **Demucs（v4/htdemucs）** | 44.1 kHz 立体声 | 大 | GPU 需 3–7 GB；CPU ≈ 1.5× 音频时长 ✅ | **MIT** ✅ | 可 CPU | **它是"音乐分轨"模型，不是语音降噪**：适合把"人声 vs 背景音乐"分开，不适合空调/底噪。仓库已声明停止维护 ✅。**不推荐**用于本项目 |
| **Resemble Enhance** | 44.1 kHz | 大（denoiser + enhancer 两段） | 慢 ⚠️（扩散/流匹配 enhancer） | **非标准自定义许可（GitHub 标 Other），商用需读条款** ✅ | CPU 很慢 | 卖点是"降噪 + 带宽扩展 + 音质美化"，PESQ 2.84（48 k 评测）✅ 低于 FRCRN/MossFormer2；**对本素材的"变亮"诉求最对症，但会引入"生成感"**，且许可有风险，仅作实验项 |
| **SepFormer / speechbrain 分离模型** | 8/16 kHz | 中 | 慢 ⚠️ | 宽松（MIT/Apache 视仓库） | 可 CPU 但慢 | 本质是**两人分离**，不是降噪；见第 4 节 |

**选型结论（降噪）**
- 主方案：**DFN3**（听感交付 + 质量最好性价比）；
- ASR 前处理分支：**FRCRN_SE_16K 或 GTCRN**（16 kHz 直接对接 ASR，省一次重采样）；
- 想再提升 PESQ：**MossFormer2_SE_48K**（显存/时间换质量）；
- CPU-only 零依赖：**ffmpeg `arnndn`（RNNoise）** 或 **sherpa-onnx GTCRN**。

---

## 4. B. "只留主讲人"：TSE / 分离 / diarization 切片 三条路

| 方案 | 需要 enrollment（主讲人报名音频）？ | 输入 | 同时说话时 | 许可证 | 落地难度 | 点评 |
|---|---|---|---|---|---|---|
| **SpeakerBeam（原始，Delcroix 等）** | **需要**（一段目标说话人语音 → speaker embedding） | 单通道混合 | 能压，但串音明显 | 研究代码为主 | **高**：无官方开箱预训练权重，要自己训练 | 概念鼻祖；工程上不如后面的方案 |
| **TD-SpeakerBeam（Interspeech 2022）** | **需要** | 单通道混合 | 同上，时域实现更轻 | 研究代码（需自训） | **高** | SpeakerBeam 的时域版本，SpeechBrain 生态里可复现；**没有稳定的一键权重**，不建议作为首版 |
| **USEF-TSE**（TASLP 2025） | **需要**（enrollment 语音），但**不需要 speaker-recognition 模型**（embedding-free，用 cross-attention 直接吃 enrollment） | 单通道混合 + 报名音频 | 论文称 WSJ0-2mix / WHAM! / WHAMR! **SI-SDR SOTA** ✅ | **CC BY-NC 4.0 → 非商用** ✅ | 中（有开源权重 `ZBang/USEF-TSE`） | **唯一"音频-only TSE 有现成权重"的候选**，但**许可锁死商用**；对 55 kbps/混响的真实讲座属 OOD，需试听 |
| **MossFormer2_SS_16K + 选路**（ClearerVoice） | **不需要** enrollment；但需要"哪一路是主讲人"的判据（用人耳 10 s 试听 / cam++ 相似度 / diarization 多数票） | 单通道混合 → 输出 2 路 | **盲分离 2 人**，重叠段也会被分到两路，但会串音 | 代码 Apache-2.0 ✅ | **中低**（`pip install clearvoice`） | ClearerVoice 里 TSE 只放了**音视频**版 `AV_MossFormer2_TSE_16K`（要视频），所以**音频-only 只能走"分离 + 选路"**。WSJ0-2mix SI-SNRi 22.0 ✅ |
| **MossFormer2_SS_16K + 先 diarization 定主讲人** | 不需要 | 同上 | 同上 | 同上 | 中 | 最实用的 TSE 替代：**diarization 找出"说话时长最长 + 无重叠"的那一路 embedding**，再用 cam++ 相似度在分离结果里选路 |
| **SepFormer（speechbrain WSJ0-2mix）** | **不需要** | 单通道，2 人 | 串音/换序常见 | 宽松 | 低（`pip install speechbrain`） | "能跑通但不够好"：8 kHz 训练域，对 44.1 k 讲座失真大 |
| **diarization + 只保留主讲人片段**（推荐） | **不需要** | 单通道 | **重叠段会被判给其中一人或直接丢弃** | 见第 5 节 | **最低** | **本项目的正解**：观众短问/咳嗽被整段剔除，主讲人音量稳定时效果最稳；缺点是**重叠段无法"抠出主讲人"**，只能丢弃或保留（有串音） |
| **纯 DSP：帧级能量/聚类门控** | 不需要 | 单通道 | 不支持 | 自研 | 低 | 仅作兜底：当 diarization 装上出问题时，用"主讲人指纹（enrollment）+ 短时 embedding 相似度"做门控，质量一般 |

**关键结论**
1. **TSE 一定要 enrollment**（USEF-TSE 只是"不要 speaker-recognition 模型"，但仍要吃 enrollment 音频）；**分离（SS）不需要**，但需要事后"选哪一路"。
2. ClearerVoice **没有发布音频-only 的 TSE 权重**（只有音视频版），所以"pip 装完就能做 TSE"在本机**不成立**；能开箱的是 USEF-TSE（非商用）与 speechbrain SepFormer。
3. 对"讲座 + 少量观众"这种长时段、主讲人占比 >80% 的音频，**diarization 切片的收益/成本比远高于 TSE**；TSE 留给"必须保留重叠段"的少量片段。

---

## 5. C. 说话人分离（diarization）

| 方案 | 模型/规模 | 显存 | 速度 | 许可证/门槛 | 中文表现 | 点评 |
|---|---|---|---|---|---|---|
| **pyannote.audio `community-1`**（2025-09） | pipeline（segmentation + WeSpeaker 类 embedding） | ⚠️ 2–4 GB（估） | **31 s / 小时音频（H100 80GB）** ✅；3060 Ti 估慢 6–10× → 78 min ≈ **4–7 min** ⚠️ | **代码 MIT；模型 CC-BY-4.0，且 gated：需 HF token + 接受 user conditions** ✅ | AISHELL-4 DER **11.7%**、AliMeeting 20.3%、AMI-SDM 19.9% ✅ | **效果最好的开源一站式流水线**，但引入 HF 账号/协议依赖 + `torchcodec`（Windows 需 ffmpeg DLL）。**组合 A 的核心风险点** |
| **pyannote.audio `3.1`（legacy）** | 同上代 pipeline | 同上 | 略慢 | 同样 gated ✅ | AISHELL-4 **12.2%**、AMI-SDM 22.7% ✅ | 生态资料最多（WhisperX 教程、社区 notebook 都基于它），若 community-1 装不上可退回 |
| **3D-Speaker**（阿里 ModelScope） | CAM++ 7.2 M / ERes2NetV2 17.8 M 等 ✅ | ⚠️ ~2 GB | 快（模块化，CPU 也能跑）⚠️ | **仓库 Apache-2.0** ✅；权重在 ModelScope（许可单独看卡片）❓；**只有开启 overlap detection 才需要 HF token** ✅ | AISHELL-4 **10.30%**、Meeting-CN_ZH-1 18.91%、ZH-2 12.78% ✅ | **对中文会议/讲座最友好的开源 diarization**；中文 DER 指标优于 pyannote（其自评口径）。**推荐替代 pyannote** |
| **FunASR `cam++` + 自动聚类** | CAM++ **7.2 M** ✅ | ⚠️ <2 GB | 很快（与 ASR 同进程） | FunASR 代码 **MIT** ✅；权重许可见模型卡（FunASR Model Open Source License）❓ | 无官方中文 DER 数字 ❓；工程上"VAD 分段 → cam++ 聚类 → 给每句打 spk"足够用 | **最省事**：只需 `AutoModel(..., spk_model="cam++")` 就得到 `sentence_info`（start/end/spk）✅，无需 HF token；缺点是聚类阈值/人数需人工调，重叠段处理弱 |
| **NVIDIA NeMo** | MSDD / Sortformer / TitaNet | ⚠️ 3–6 GB | 中 | NeMo 代码 Apache-2.0；**各模型卡许可不同（CC-BY-4.0 / NVIDIA Open Model License 混用）** ❓ | 英文为主，中文需自训/迁移 ❓ | 功能最全但依赖重（NeMo + megatron 生态），**Windows 安装风险最高**，本项目不推荐 |
| **sherpa-onnx diarization** | segmentation + embedding ONNX | **0**（CPU） | 慢一些 ⚠️ | Apache-2.0 ✅ | 中文可用（官方有中英模型），DER 一般 ❓ | **CPU 兜底组合 C 的 diarization**，零 CUDA |

**选型结论（diarization）**
- 有 HF token、要最好效果：**pyannote community-1**；
- 不想碰 HF gated：**3D-Speaker**（中文 DER 更好、Apache-2.0）或 **FunASR cam++**（与 ASR 同栈，最省集成）；
- 纯 CPU：**sherpa-onnx diarization**。

---

## 6. D. 中文 ASR 候选

| 模型 | 参数量 | 中文 CER/WER（公开口径） | 标点 | 时间戳 | 热词 | 长音频 | RTF / 78 min 预计 | 显存 | 许可 |
|---|---|---|---|---|---|---|---|---|---|
| **FunASR Paraformer-large（paraformer-zh）** | **220 M** ✅ | 历史评测集 CER **10.18%** ✅（该集合上 Whisper-large-v3 为 20.02%） | ✅ 外挂 `ct-punc`（290 M）✅ | ✅ **原生字级时间戳** | ✅ `hotword="关键词 20"` ✅ | ✅ VAD 切分 + `batch_size_s` | H100 **RTF 0.0084（119.6×）** ✅；3060 Ti ⚠️ 估 20–35× → **2–4 min** | ⚠️ 2–3 GB（含 punc/vad/cam++） | 代码 MIT ✅；权重 FunASR Model License ❓ |
| **SenseVoice-Small** | **234 M** ✅ | 同集 CER **7.81%**（本组最低）✅ | ❌ 无独立标点模型（其输出自带标点能力，弱） | ❌ 原生无；**靠 VAD 分段给句级时间戳** ✅ | ❌ | ✅ | H100 **RTF 0.0059（169.6×）**；**CPU 0.058（17.2×）** ✅；3060 Ti 估 40–60× → **~1.5–2.5 min** | ⚠️ 1–2 GB | 同上 |
| **Qwen3-ASR-1.7B / 0.6B** | **1.7 B / 0.6 B** ✅ | 官方对比：中文 Dialog-Mandarin **6.54%**（Whisper-large-v3 14.01%）、ExtremeNoise **16.17%**（Whisper **63.17%**）、22 种方言均值 **15.94%**（Whisper 44.55%）✅ | ✅ 自带（LLM 解码） | ⚠️ 需额外 **Qwen3-ForcedAligner-0.6B** 才有时间戳 ✅ | 弱（无专门热词接口）❓ | ✅ 官方称支持长音频 | ⚠️ 无官方 RTF；估 1.7 B RTF 0.05–0.12 → **4–9 min**（vLLM 后端可更快） | ⚠️ 1.7 B≈4–5 GB（+aligner 1.5 GB，8 GB 需分时加载） | 宽松（Apache-2.0 类）❓ 需看卡片 |
| **faster-whisper large-v3** | **1.55 B**（large-v3）；turbo 809 M ✅ | 中文偏弱：历史集 CER 20.02% ✅；但**对噪声/口音鲁棒性尚可** | ✅ 有标点（中文标点稳定性一般） | ✅ 句级 + `word_timestamps` | ❌ | ✅ VAD 过滤 | **3070 Ti：13 min 音频 fp16/beam5/batch8 = 17 s（RTF 0.022）；非批 63 s（RTF 0.081）** ✅；3060 Ti → 78 min ≈ **3–8 min** | **fp16 ≈ 4.5 GB；int8 ≈ 3 GB** ✅（含 batch=8 时 4.5–6 GB） | **MIT** ✅（模型 MIT） |
| **WhisperX** | 同 Whisper + wav2vec2 align + pyannote | 同上 | ✅ | ✅ **词级**（强制对齐） | ❌ | ✅ VAD 分批 | 官方 **70× realtime（large-v2）** ✅；78 min 含对齐/分离 ≈ **5–10 min** ⚠️ | ≈ 5 GB + align 1 GB + pyannote 3 GB（需分时释放）✅/⚠️ | 宽松；**diarization 需 HF token** ✅ |
| **sherpa-onnx（SenseVoice / Paraformer ONNX）** | 同上游（int8 量化） | 同上游 | ✅ 有标点模型（可选） | ✅ VAD 段级 | ❌ | ✅ | **CPU only**：SenseVoice int8 ⚠️ 估 RTF 0.15–0.3 → **12–25 min** | **0** | Apache-2.0 ✅ |
| **Fun-ASR-Nano-2512**（LLM-based，2025/26） | 800 M ✅ | 历史集 CER 8.06% ✅ | ✅ | ❌ 官方明确"checkpoint 原生时间戳不可靠" ✅ | ✅ 支持热词 ✅ | ✅ | H100 RTF 0.0588（17×）；CPU 0.274（3.6×）✅ | ⚠️ 3–4 GB | 宽松 ❓ |

**选型结论（ASR）**
- **主选 Paraformer-large + ct-punc + fsmn-vad**：中文标点、字级时间戳、热词（本讲座的哲学专名如"亚里士多德/形而上学/四因说"极适合做热词）、长音频、MIT 代码，工程完整度最高；
- **质量/抗噪补充 SenseVoice-Small**：同集 CER 更低、CPU 也快，但**没有字级时间戳与热词**，适合"只要文本"；
- **极致鲁棒 Qwen3-ASR-1.7B**：官方 benchmark 在中文强噪与方言上大幅领先 Whisper，但**要时间戳必须再挂 0.6 B 对齐模型**，8 GB 卡上建议分阶段加载；
- **兜底 faster-whisper large-v3**：成熟、快、VRAM 可控，但中文 CER 与标点/热词能力弱，仅作对照。

---

## 7. E. 三套推荐组合（含体积 / 风险 / 耗时）

> 78 min = 4680 s。耗时是把官方 RTF 按"3060 Ti ≈ 3070 Ti 的 80%、≈ H100 的 1/4–1/6（小模型）"折算的 ⚠️ 估算；CPU 行按 8 线程现代桌面 CPU 折算。**必须本机实测校准**。

### 组合 A：最快落地（GPU，推荐第一版就用这套）

| 环节 | 选型 |
|---|---|
| 解码/重采样 | ffmpeg → 48 kHz 单声道 wav（降噪用）+ 16 kHz 单声道 wav（ASR 用），分块 10 min 以控内存 |
| 降噪 | **DeepFilterNet3**（`pip install deepfilternet`） |
| 只留主讲人 | **pyannote community-1**（或 3D-Speaker / FunASR cam++ 作无 token 备选）→ 保留主讲人片段，非主讲人静音/丢弃 |
| ASR | **FunASR** `paraformer-zh + fsmn-vad + ct-punc + cam++`，热词表填专名 |
| 依赖体积 | torch(CUDA12)+torchaudio ≈ 5–7 GB；DFN3 权重 ~10 MB；pyannote 权重 ~0.5–1 GB；FunASR 权重 ≈ 0.3+0.5+1.2 GB ≈ **合计 8–10 GB** |
| 首次安装风险 | **中**。① pyannote gated 需 HF 登录/协议（本机 HF 不可达时最痛）；② `torchcodec` 在 Windows 要 ffmpeg DLL；③ FunASR 走 ModelScope 下载较稳 |
| 显存峰值 | ⚠️ ≈ 3–4 GB（三段串行，每段结束 `torch.cuda.empty_cache()`） |
| 78 min 预计 | 降噪 **2–5 min** + diarization **4–7 min** + ASR **2–4 min** ≈ **10–20 min** |
| 许可证风险 | **低**（DFN3 MIT/Apache、FunASR MIT、pyannote 模型 CC-BY-4.0 需署名 + 遵守 user conditions） |

### 组合 B：质量优先（要"抠出重叠段主讲人"时用）

| 环节 | 选型 |
|---|---|
| 降噪 | **MossFormer2_SE_48K**（`pip install clearvoice`）或 FRCRN_SE_16K（ASR 分支） |
| 只留主讲人 | ① 从主讲人"干净独白段"（diarization 结果里最长且无重叠的 5–15 s）截 enrollment；② **USEF-TSE** 提取；或 ③ **MossFormer2_SS_16K** 分离 2 路 + cam++/diarization 选路 |
| ASR | **Qwen3-ASR-1.7B**（+ ForcedAligner-0.6B 取时间戳）；或 Paraformer 做交叉校验 |
| 依赖体积 | 上述 + clearvoice/torch 共用 + Qwen3-ASR ≈ 3.5 GB + aligner ≈ 1.2 GB + USEF-TSE 权重 ≈ **14–18 GB** |
| 首次安装风险 | **高**。① clearvoice 权重在 HF（需镜像）；② USEF-TSE 无 pip 包，要 clone 仓库 + 改数据路径；③ Qwen3-ASR 建议 fresh env（py3.12 官方推荐），与本机 3.10 有版本漂移；④ flash-attn 在 Windows 源码编译风险高 → **用 transformers 后端，不装 flash-attn** |
| 显存峰值 | ⚠️ 5–7 GB（USEF-TSE/MossFormer2 与 ASR 不能同时驻留，需换段加载） |
| 78 min 预计 | 降噪 4–12 + 分离/TSE 8–23 + ASR 4–9 ≈ **30–60 min**（+ 调参试听 1–2 h 人工） |
| 许可证风险 | **高**：**USEF-TSE 是 CC BY-NC 4.0 → 禁止商用**；ClearerVoice 权重卡片未统一标注；Qwen3-ASR 需看具体卡片 |

### 组合 C：CPU-only 兜底（不装 CUDA，任何机器可跑）

| 环节 | 选型 |
|---|---|
| 降噪 | 首选 **ffmpeg `arnndn`（RNNoise，BSD-3）**；质量不够换 **sherpa-onnx GTCRN**（48.2 K 参数） |
| 只留主讲人 | **sherpa-onnx speaker-diarization**（Apache-2.0）+ 只保留主讲人片段 |
| ASR | **sherpa-onnx SenseVoice int8**（或 `whisper.cpp`/faster-whisper CPU int8 作对照） |
| 依赖体积 | onnxruntime + sherpa-onnx + ffmpeg ≈ **2–3 GB**（不装 torch） |
| 首次安装风险 | **低**（pip 轮子 + 预编译二进制，无 CUDA/cuDNN/编译） |
| 显存峰值 | **0** |
| 78 min 预计 | RNNoise ≈ 1–3 min（CPU 极快）；GTCRN ≈ 5–8 min；diarization 8–15 min；ASR 12–25 min ≈ **30–50 min** |
| 许可证风险 | **低**（RNNoise BSD-3、sherpa-onnx Apache-2.0、SenseVoice 见卡片） |

**建议执行顺序**：先跑 **C 的"零风险验证"**（1 h 内出第一版文本，验证切分/主讲人判定）→ 再上 **A**（质量与速度兼顾，作为交付主链）→ **B 只在需要重叠段时局部运行**（别整片跑，只对 5–15 min 的重叠密集段跑）。

---

## 8. F. 效果边界与信源上限（必须写进对用户的说明）

### 8.1 "同时说话时无法彻底去掉第二个人"——这是原理性的

- 单通道、无空间信息、无参考信号时，两人**同时发声**是"欠定盲分离"（1 个观测、2 个源）。所有开源方案都在做"统计先验 + 掩码估计"，必然：
  - 有**串音泄漏**（第二人残留 −10 ~ −20 dB 已是好结果）；
  - 对目标说话人有**失真**（音色改变、辅音断裂、"水下感"）；
  - 重叠比例越高越差：WSJ0-2mix 上的 22 dB SI-SDRi **不能**外推到"混响礼堂 + 55 kbps + 观众噪声 + 中文"。
- 现实预期（单通道、本素材）：
  - **非重叠段**（占比 >80%）：diarization 切片可做到"基本只有主讲人"；
  - **重叠段**（观众提问插话/主讲人被打断）：TSE 能把第二人压掉**大部分**，但会留尾音/呼吸/串音；**商用交付建议对这些段做"保留原声 + 标注"或人工复核**，别宣称"已彻底分离"。
- **不要承诺**：把观众笑声/掌声也当"噪声"完全去掉——掌声是宽带瞬态，降噪器会当成噪声压掉，但会留下明显"泵浦/抽吸"感（pumping），听感反而更差。

### 8.2 55 kbps AAC 的上限（本素材实测）

| 环节 | 受影响方式 | 缓解 |
|---|---|---|
| 降噪 | 高频 8 kHz 以上已 −54 dB 以下；降噪器可能把残余高频噪声也压掉 → **更闷**；编码预回声被当成瞬态，可能被"保护"下来 | 用**48 kHz 原生**模型（DFN3/MossFormer2_SE_48K），避免二次重采样；关闭 postfilter 的过抑制 |
| 分离/TSE | 训练域是 8/16 kHz、高保真混音；本素材**立体声近似 dual-mono、有效带宽 14 kHz、含编码伪影** → 域偏移，指标会明显低于论文 | 只在必要时用；enrollment 也要用**同一编码链路**的音频（不要用外部高质量录音做报名，音色会不匹配） |
| ASR | 影响相对最小（语音主能量 0.1–4 kHz），但**专名/低频噪声段 CER 上升**；Whisper 系在长音频+噪声下易**幻觉重复** | Paraformer/SenseVoice 更稳；加 VAD、加专名热词、开 `condition_on_previous_text=False` |
| 音质提升 | 降噪**不能让声音变亮**（没有可恢复的高频） | 若客户要"更清晰/更亮"，需要 **带宽扩展**：`MossFormer2_SR_48K`（ClearerVoice）或 Resemble Enhance 的 enhancer——**属于"生成式美化"，可能改变音色，需客户确认可接受** |

### 8.3 "降噪"和"带宽扩展"别混为一谈

- 降噪 = 去加性噪声；带宽扩展 = 补缺失频带（生成式）。
- 本素材真正的短板是**码率**（55 kbps），不是 SNR。若 SNR 尚可（P10 vs P90 的近似 SNR 由音频分析任务给出），**降噪收益有限，甚至可能伤清晰度**；此时应该只做很轻的 DFN3（或用 GTCRN 低强度）先给 ASR，交付音频则保留原声。

---

## 9. 落地建议（按风险从低到高）

1. **先验证"要不要降噪"**：取 3 段 60 s（头/中/尾），A/B：原始 vs DFN3 vs GTCRN，用 SpeechScore（ClearerVoice 自带，含 PESQ/STOI/DNSMOS/NISQA）打分 + 人耳判断。**若 DNSMOS 提升 <0.1 且主观更闷，就不降噪**（本素材很可能如此）。
2. **diarization 用小样本调参**：先对 5 min 片段跑 pyannote/3D-Speaker，确认 `num_speakers`（建议先不指定、看自动聚类；再固定 2–4 人重跑对比），人工核对主讲人 ID 是否稳定——**ID 抖动是长音频最常见的坑**（同一人被拆成两个 ID）。
3. **"只留主讲人"优先用切片**，不要整片 TSE。切片边界留 150–300 ms padding，避免吃掉字头。
4. **ASR 前统一 16 kHz 单声道**，VAD 用 fsmn-vad/silero，段长 30 s 内；Paraformer 走 `batch_size_s=300` 批处理提速。
5. **热词**：从讲稿/书名/人名生成 30–80 个热词（Paraformer 支持 `hotword="亚里士多德 20"`），能显著降专名 CER。
6. **分块处理**：78 min 不要一次读进内存（44.1 k 立体声 float32 ≈ 824 MB，16 k ≈ 150 MB，尚可；但 48 kHz 模型中间张量会涨）。按 10 min 块 + 10 s overlap，输出后再拼接。
7. **许可证红线**：交付物若涉及商用，**排除 USEF-TSE（CC BY-NC）**；pyannote 模型需署名（CC-BY-4.0）且遵守 gated 用户条款；ClearerVoice/ModelScope 权重逐卡确认。
8. **HF 不可达时的固定写法**：`export HF_ENDPOINT=https://hf-mirror.com`（或 PowerShell `$env:HF_ENDPOINT`）；FunASR 直接用 ModelScope `iic/...` 模型 ID；pyannote gated 仓库镜像不代理 → 必须能登录 HF 才能用组合 A 的 diarization。

---

## 10. 来源清单（核实用）

**降噪**
- DeepFilterNet 仓库（含 DFN3，MIT/Apache 双许可）：<https://github.com/Rikorose/DeepFilterNet>
- DFN3 论文（**单线程 CPU RTF 0.19**）：<https://arxiv.org/abs/2305.08227>
- GTCRN（**48.2 K 参数 / 33 MMACs / 流式 RTF 0.07 / RNNoise 0.06 M / DFN1 1.80 M 对比表**）：<https://github.com/Xiaobin-Rong/gtcrn>
- FRCRN 论文：<https://arxiv.org/abs/2206.07293>
- ClearerVoice-Studio（SE/SS/TSE 模型清单、VoiceBank+DEMAND 与 DNS-2020 全指标表）：<https://github.com/modelscope/ClearerVoice-Studio> 、模型表 <https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md> 、论文 <https://arxiv.org/abs/2506.19398>
- ClearerVoice 代码许可 Apache-2.0：<https://github.com/modelscope/ClearerVoice-Studio/blob/main/LICENSE>
- Demucs（**音乐分轨**，MIT，已停维）：<https://github.com/facebookresearch/demucs>
- Resemble Enhance（denoiser+enhancer，自定义许可）：<https://github.com/resemble-ai/resemble-enhance>
- ffmpeg `arnndn`（RNNoise 滤镜）：<https://ffmpeg.org/ffmpeg-filters.html#arnndn>

**TSE / 分离**
- USEF-TSE 论文（embedding-free，需 enrollment）：<https://arxiv.org/abs/2409.02615> ；代码与权重 <https://github.com/ZBang/USEF-TSE>（**LICENSE: CC BY-NC 4.0**）
- MossFormer2 论文（分离）：<https://arxiv.org/abs/2312.11825>
- SpeakerBeam / TD-SpeakerBeam 系（Interspeech 2022）：<https://www.isca-archive.org/interspeech_2022/zhao22_interspeech.pdf>
- SpeechBrain SepFormer（WSJ0-2mix 预训练）：<https://huggingface.co/speechbrain/sepformer-wsj02mix>

**Diarization**
- pyannote.audio 仓库（community-1 用法、**需 HF token + 接受 user conditions**、DER 对比表、**31 s/小时（H100）**）：<https://github.com/pyannote/pyannote-audio>
- pyannote community-1 模型页（gated）：<https://hf.co/pyannote/speaker-diarization-community-1> ；3.1 legacy：<https://hf.co/pyannote/speaker-diarization-3.1>
- 3D-Speaker（**Apache-2.0**、DER 表、CAM++ 7.2 M、`--include_overlap` 才需 HF token）：<https://github.com/modelscope/3D-Speaker>
- FunASR（cam++ 7.2 M、AutoModel 带 spk 输出、模型 zoo 参数表）：<https://github.com/modelscope/FunASR>

**ASR**
- FunASR 仓库（Paraformer-zh 220 M / SenseVoiceSmall 234 M / ct-punc 290 M / fsmn-vad 0.4 M / cam++ 7.2 M / Fun-ASR-Nano 800 M；MIT 代码）：<https://github.com/modelscope/FunASR>
- FunASR 历史 ASR 评测（**SenseVoice GPU RTF 0.0059、Paraformer 0.0084、Whisper-large-v3 0.0747；CER 7.81% / 10.18% / 20.02%；CPU RTF 0.058 / 0.064**，注意官方声明这是"来源不完整的历史记录"）：<https://www.funasr.com/docs/historical-asr-benchmark.html>
- SenseVoice 论文：<https://arxiv.org/abs/2407.04051>
- Qwen3-ASR 仓库（0.6B/1.7B、52 语种、WER 对比、ForcedAligner-0.6B、2026-01-29 发布）：<https://github.com/QwenLM/Qwen3-ASR> ；论文 <https://arxiv.org/abs/2601.21337>
- faster-whisper（**3070 Ti 上 13 min 音频：fp16/beam5/batch8 = 17 s、VRAM 4.5–6 GB；int8 更低；CPU 对照表**）：<https://github.com/SYSTRAN/faster-whisper>
- WhisperX（**70× realtime large-v2、<8 GB、词级对齐、diarization 需 HF token**）：<https://github.com/m-bain/whisperX>
- sherpa-onnx（ASR/diarization/**GTCRN 语音增强**/标点，ONNX、多平台含 Windows）：<https://github.com/k2-fsa/sherpa-onnx>
- NeMo（说话人相关，代码 Apache-2.0、模型卡许可不一）：<https://github.com/NVIDIA/NeMo>

**本机事实**
- 源文件 ffprobe：AAC 44.1 kHz / 2 ch / 55 825 bps / 4673.29 s（本次实测）
- 有效带宽：−70 dB 滚降 ≈ 14.25 kHz（本次对 `work/full_44k_stereo.wav` 2000 s 处 60 s 长时平均谱实测）

---

### 附：需要后续实测才能落定的数字（不要当成结论写进对外文档）

- DFN3 / FRCRN / MossFormer2_SE_48K / MossFormer2_SS_16K / USEF-TSE 在**本机 3060 Ti** 上的实际 RTF 与显存峰值；
- pyannote community-1 / 3D-Speaker 在**本素材**上的真实 DER 与主讲人 ID 稳定性（长音频 ID 抖动）；
- 本素材的实际信噪比与"是否需要降噪"（由音频分析任务的 P10/P90、频谱滚降结论支撑）；
- 降噪前后 **DNSMOS/PESQ/NISQA**（用 SpeechScore）与 ASR **CER**（需人工转写 3–5 min 金标准）的对照。
