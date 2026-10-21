# audio_processer

把公开平台的课程录音批量处理成「**干净音频 + 可校对的带时间戳转写稿**」的本地工具。
当前处于「需求已对齐、方案已实测验证、等待施工」阶段。

## 现在看什么

| 文件 | 说明 |
|---|---|
| **[docs/开发启动Prompt.md](docs/开发启动Prompt.md)** | **在新 agent 会话里启动开发用的自包含 prompt**（复制代码块即可） |
| **[docs/调研报告_需求对齐版.md](docs/调研报告_需求对齐版.md)** | **主报告（v2）**：5 条需求的实测验证、音量稳定化、批量架构、**开发 token/时间成本**、验收标准 |
| **[docs/实施计划.md](docs/实施计划.md)** | **可直接交给 agent 执行的实施计划**：P0–P8 任务卡、写作用域、验收命令、回归基线 |
| [docs/方案与成本评估.md](docs/方案与成本评估.md) | v1 报告：素材诊断、三条路线（全本地/云/SaaS）、成本模型 |
| [docs/research/multifile-scan.md](docs/research/multifile-scan.md) | 6 文件体检：响度/语音电平/非语音结构/说话人/ASR 抽样 |
| [docs/research/audio-edit-poc.md](docs/research/audio-edit-poc.md) | 降噪 + 裁剪 + 语音电平归一化（V1/V2/V3）PoC |
| [docs/research/timeline-poc.md](docs/research/timeline-poc.md) | VAD 分段 → 句级时间戳 → 发言人的转写 PoC |
| [docs/research/benchmark.md](docs/research/benchmark.md) | ASR/降噪 RTF、内存、前后指标、踩坑记录 |
| [docs/research/local-options.md](docs/research/local-options.md) | 本地模型矩阵（降噪 / TSE / 分离 / 中文 ASR）与许可证 |
| [docs/research/cloud-options.md](docs/research/cloud-options.md) | 云平台核实价格、能力边界、合规 |
| [docs/research/diarization-poc.md](docs/research/diarization-poc.md) | 整片说话人分离（非主讲人仅 1.7%） |
| [out/](out/) | 已产出的实际转写稿与降噪音轨示例 |

## 目录

```
data/          原始 m4a（只读，6 个样例 / 576 min）
out/           可直接消费的产物（转写、干净音轨）
docs/          报告、调研、图
app/           应用代码（待开发，见 docs/实施计划.md）
work/          中间产物、可复现脚本、模型（可随时删）
```

## 一句话结论

- **降噪 + 裁剪 + 音量归一化 + 转写 + LLM 纠错** 全部本机 CPU 可跑（不需要 GPU/torch）；
- **跨文件音量差实测 29.7 dB**，用「语音活跃电平 ASL + 逐文件静态增益 + 受控慢变」归一后 6 文件极差 **0.05 dB**；
- **非讲课内容**（片头音乐/寒暄）每文件 0–480 s，必须用「120 s 窗口 ASR 字符率」裁剪，VAD 不可靠；
- **开发成本**：DSH + deepseek-flash 全期约 **270–400M tokens ≈ ¥16–55**，构建日历 7–12 天（详见报告 §10）。

## 复现关键命令

```powershell
# 环境
work\venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$env:PYTHONIOENCODING='utf-8'
$env:DEEPSEEK_API_KEY='sk-...'   # 不设则跳过 LLM 纠错

# 建库后（见实施计划）：
python -m app.cli scan data/ --out out/
python -m app.cli batch data/ --out out/ --resume
python -m app.cli serve --port 7860
```

## 状态

- ✅ v1 方案探索与成本评估
- ✅ v2 需求对齐：6 文件实测（响度/裁剪/时间戳/纠错）
- ✅ 实施计划（P0–P8 任务卡）已就绪
- ⬜ 施工：P0 基础设施 → P1 解码 → … → P8 回归（见 `docs/实施计划.md`）

## 开始开发

新开一个 agent 会话，把 [`docs/开发启动Prompt.md`](docs/开发启动Prompt.md) 里的代码块整段作为第一条消息发送即可（自带目标、必读文档、环境事实、锁定参数、P0+P1 任务与验收）。

## 仓库说明

- git 仓库只提交**代码/文档/小型中间 JSON**；`data/`、`work/` 下的 wav/模型、`out/*.m4a` 已在 `.gitignore` 中排除（样例数据与模型只在本地保留）。
- 样例音频（6 个 m4a，576 min）与已下载模型不在仓库里；新机器上需重新放回 `data/`、`work/bench/models/`、`work/diar/models/`（或在代码里改为自动下载）。
