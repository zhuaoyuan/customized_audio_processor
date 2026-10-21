# 新 agent 启动 Prompt（复制下面代码块整段使用）

> 使用方式：在项目根目录 `C:\projects\local\audio_processer` 新开一个 agent 会话，把下面代码块里的内容原样作为第一条消息发送。
> 该 prompt 自包含：包含目标、必读文档、环境事实、已锁定参数、本轮任务（P0+P1）、验收命令与已知坑。
> 若只想先做基础设施，把「本轮任务」改成 P0；若想一次做更多，按 `docs/实施计划.md` 的 P0→P8 顺序推进。

```text
你是「audio_processer」项目的开发 agent。工作目录：C:\projects\local\audio_processer（Windows 10，pwsh 可用，当前会话为完全权限）。

## 项目目标
把公开平台的课程录音（m4a）批量处理成「干净音频 + 可校对的带时间戳转写稿」：
1) 降噪；2) 去头尾/长段非人声（裁剪）；3) 语音电平稳定化（跨文件一致）；4) 保守处理无关人声；5) 转写（句级时间戳+发言人）+ LLM 按主题修错字。
应用形态：CLI + 本地 Web UI；需支持 300+ 文件批量、断点续跑、失败重试。

## 开工前必读（按顺序，读了再动手）
1. docs/实施计划.md —— **主计划**：P0–P8 任务卡（依赖/写作用域/实现要点/验收）、数据契约、回归基线、已知坑。严格按它执行。
2. docs/调研报告_需求对齐版.md —— 需求对齐与实测依据（§2 六文件体检、§4 裁剪、§5 音量归一、§7 时间戳、§8 纠错、§9 批量架构、§12 决策记录）。
3. docs/research/audio-edit-poc.md、docs/research/timeline-poc.md、docs/research/multifile-scan.md —— 已跑通的三个 PoC 的方法与实测数字。
4. 参考实现（可直接移植/改造，不要重复造轮子）：
   - work/audio-edit/scripts/run_one.py（降噪→VAD→裁剪→V1/V2 归一化→真峰值回读限幅的完整链路）
   - work/timeline/scripts/（step_vad/step_asr/step_diar/step_build/step_eval + run_file.py：VAD 分段、逐段 SenseVoice、说话人、SRT/MD 生成）
   - work/scan/fix_f00.py 与 work/audio-edit/scripts/fix_stco.py（MP4 stco 偏移修复）
   - work/scan/{vad,asr,loud}*.py、work/bench/run_asr_sensevoice.py、work/diar/run_diar.py（分析/ASR/分离）
   - work/cost/{sum_usage,ts_eval,diff_correct}.py（token 统计、时间戳误差、纠错 diff 校验）

## 环境事实（已经装好，别重复安装）
- Python：work\venv\Scripts\python.exe（3.10）。已装：sherpa-onnx 1.13.8、onnxruntime、numpy、scipy、matplotlib、faster-whisper、ctranslate2。**未装 torch，本方案也不需要**。
- 若缺依赖：work\venv\Scripts\python -m pip install <pkg> -i https://pypi.tuna.tsinghua.edu.cn/simple（fastapi/uvicorn/pyyaml/pytest 可能需要装）。
- ffmpeg/ffprobe 在 PATH（7.1）。
- 模型（已下载，不要重新下）：
  - SenseVoice int8：work/bench/models/sensevoice/{model.int8.onnx, tokens.txt}
  - Silero VAD：在 faster_whisper 包内置 assets（work/venv/Lib/site-packages/faster_whisper/assets/silero_vad_v6.onnx）
  - diarization：work/diar/models/（pyannote-segmentation-3-0/model.onnx + 3dspeaker cam++ onnx）
  - rnnoise：work/bench/models/rnnoise/sh.rnnn
  - faster-whisper：work/bench/models/whisper（HF 缓存，需 HF_HOME 指向这里、HF_HUB_DISABLE_XET=1）
- 样例数据：data/ 下 6 个 m4a（丁耘课程，共 576 min）。其中 0_20140227-丁耘 西方哲学*.m4a 容器 stco 偏移损坏，必须走修复路径。
- 先设：$env:PYTHONIOENCODING='utf-8'；所有脚本读写显式 encoding='utf-8'；Windows 路径一律用 Path/列表参数，不要拼 shell 字符串。

## 已锁定的技术决策（见 docs/实施计划.md §2，不要再讨论/改默认值）
- 目标语音电平 T = −20.0 dBFS ASL（只统计 ≥3 s 连续语音段的语音帧）；跨文件偏差 ≤±0.5 dB。
- 文件内波动：5 min 窗口 ASL 半展宽 (p95−p5)/2 ≤ ±1.5 dB；用「静态增益 + 受控慢变 V2」（窗 20 s/hop 5 s/clamp ±3 dB/速率 ≤1.5 dB/s/20 ms 交叉淡化），源内波动 >6 dB 的文件单独 clamp ±6。
- limiter：真峰值 ≤−1 dBTP；必须用真峰值（4× 过采样测峰→下调 ceiling 重做，ceiling −2.5~−3.0 dBFS），不能用采样峰。
- 头尾边界：**120 s 窗口聚合 ASR 字符率 ≥2.0 连续 2 窗口**判讲课起点（VAD 会被片头音乐骗；f24 真实起点 480 s）。
- 内部长静音：VAD 非语音 ≥3 s 切除，两侧留 0.25 s；碎片语音（<3 s）保留。
- 降噪默认：highpass=f=80, afftdn=nr=12:nf=-30:tn=1（44.1k）；低 SNR 文件可升 nr 或切 arnndn（必须 48k 进出）。
- 无关人声：默认全保留，只删「头尾非讲课 / 非主讲≥5 s 且 ASR 字速率<0.5 / VAD 语音但 ASR 空」，明细写 edit.json。
- 转写：Silero VAD 分段（0.5–30 s，过长 30 s 切）→ 逐段 SenseVoice → 句级 start/end；clean 时间轴 + orig 双时间；默认 speaker="主讲"，抽样 diar 发现稳定第二人才标「其他N」。
- 纠错：deepseek-flash（OpenAI 兼容 https://api.deepseek.com/v1），key 从 DEEPSEEK_API_KEY 读；无 key 自动跳过；分块 2000–2500 字；diff ≤12%、数字/句数校验，超限回退原文并记 rejected。
- 输出：out/<id>/{clean.m4a（44.1k mono AAC 96k）, edit.json, transcript.json/md/srt, corrections.json, report.json, log.txt}；字段名以实施计划 §4 数据契约为准。
- 并发：单进程 + ASR 8 线程（实测进程池更慢）；磁盘水位 <10 GB 停止取新任务。

## 本轮任务（P0 + P1）
P0：配置/logging（app/config.py、app/logging_setup.py、configs/default.yaml）、SQLite 存储（app/storage.py、app/models.py）、依赖骨架（requirements.txt）。
P1：MP4 stco 修复（app/tools/repair_mp4.py）+ 解码与缓存（app/pipeline/decode.py：ffmpeg→16k mono 失败则修复重试，44.1k 按需解码）+ scan 子命令（app/cli.py 的 scan）。
要求：
- 按 docs/实施计划.md 中 P0-T1~T3、P1-T1~T3 的验收命令逐条验证（pytest + 手工命令）；
- 6 个样例都要能 decode 成功，f00 必须 repaired=True；
- 中间 wav 写 work/cache/，阶段结束清理；
- 每完成一个阶段跑一次验收，把结果写进 docs/工程日志.md（新建，按日期追加：做了什么/验收结果/遗留问题）。

## 执行规则
1. 写作用域：只改你当前任务卡声明的文件；不要动 data/、不要动 docs/research/ 下他人报告（只读参考）、不要动 work/ 下 PoC 产物（只读参考）。
2. 不要提交二进制大文件；仓库已有 .gitignore（data/、work 下 wav/onnx/模型、out 的 m4a 都被忽略）。每个阶段完成后本地 commit，消息格式 `feat(P1): decode with stco repair` / `test: ...` / `docs: ...`。
3. 所有参数走 configs/default.yaml，不要硬编码；命令行支持 --preset fast|standard|quality。
4. 中文输出/注释；代码保持小函数、可单测；异常要有明确错误信息与 stage 状态落库。
5. 验收必须真实执行（pytest / 六文件命令），失败就修，不要跳过；遇到计划里没写的阻塞，先写进 docs/工程日志.md 再继续。

## 已知坑（详见实施计划 §7，务必规避）
- stco 偏移损坏：ffmpeg 报 "Reserved bit set / Number of bands exceeds limit"→ 修复 stco 再解码；
- ffmpeg arnndn 不能带 Windows 盘符路径（冒号）→ cd 到模型目录用相对名；rnnoise 只支持 48k，必须 aresample 进出；
- sherpa-onnx PyPI wheel 是 CPU-only，provider=cuda 会静默回退；
- VAD 会把片头音乐/掌声判成语音 → 边界必须用窗口化 ASR 字符率；
- 长音频不要整段喂 ASR（会幻觉），必须 VAD 分段；
- 16k 解码用 float32，别用 int16（源有 >0 dBFS 过冲，先削顶会丢失判断）；
- 单说话人场景 diarization 会裂成几十簇 → 默认不做全量 diar、不做 per-speaker 增益。

## 完成标准（本轮）
- `work\venv\Scripts\python -m app.cli config show`、`scan data/ --out out/`、`status` 可用；
- 6 文件 decode 全成功（f00 repaired），`work/cache/` 只留缓存 wav；
- `pytest tests/` 通过（至少 storage/decode/repair 三个测试文件）；
- docs/工程日志.md 有本轮记录；git commit 完成。

开工。先读 docs/实施计划.md 的 §1–§5 和数据契约，然后从 P0-T1 开始。
```
