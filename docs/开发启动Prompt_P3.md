# P3 启动 Prompt（可直接复制给新 agent 会话）

> 使用方式：在项目根目录 `C:\projects\local\audio_processer` 新开 agent 会话，把下面代码块整段作为第一条消息发送。
> 前置状态：P0（基础设施）、P1（解码/stco 修复）、P2（VAD/电平/边界）已完成并提交；详见 `docs/工程日志.md`。

```text
你是「audio_processer」项目的开发 agent，继续执行 P3（音频渲染：降噪 / 裁剪+映射 / 响度归一化 / 输出 / 六文件回归）。
工作目录：C:\projects\local\audio_processer（Windows 10，pwsh 可用）。上一轮已完成 P0/P1/P2 并提交，本轮只做 P3。

## 开工前必读（按顺序，读了再动手）
1. docs/实施计划.md —— §2 技术决策（锁定值）、§4 数据契约、**§5 的 P3-T1~P3-T5 任务卡**、§7 已知坑。
2. docs/工程日志.md —— 尤其是 P2 一节的「最终裁剪汇总」与「遗留问题」；本轮的输入边界来自这里。
3. docs/调研报告_需求对齐版.md —— §3 降噪、§4 裁剪、§5 音量稳定化（§5.3/§5.4 是 V1/V2/limiter 的实测依据）。
4. docs/research/audio-edit-poc.md —— 降噪+V1/V2/V3+真峰值限幅的完整 PoC 方法与数字。
5. 参考实现（可直接移植，不要重写）：
   - work/audio-edit/scripts/run_one.py（完整链路：降噪→VAD→裁剪→V1/V2→真峰值回读限幅，**本轮最重要**）
   - work/audio-edit/scripts/common.py（frame_power/asl_from_power/window_levels/denoise_file/limiter_file/measure_r128）
   - work/audio-edit/scripts/fix_stco.py（已移植到 app/tools/repair_mp4.py，不必再看）

## 当前仓库状态（不要重复实现）
- 已 ✅ P0：app/config.py（default.yaml+preset+env 合并、校验、config_hash）、app/logging_setup.py、
  app/models.py（Job/StageResult/DecodeResult/AnalyzeResult/EditMap/TranscriptSegment）、
  app/storage.py（jobs/stages、断点续跑、阶段状态机）、configs/default.yaml + presets(fast/standard/quality)。
- 已 ✅ P1：app/pipeline/decode.py（ffmpeg→16k float32/44.1k 按需、发刷 stco 修复重试、work/cache 缓存）、
  app/tools/repair_mp4.py（含 fragmented MP4 识别）、app/cli.py 的 config/scan/status/decode。
- 已 ✅ P2：app/pipeline/analyze.py（Silero VAD、ASL/噪声底/SNR、5min 窗口、内部长静音、120s 窗口+30s 块级精修的
  头尾边界）、app/pipeline/asr.py（SenseVoice 单例封装，P4-T1 已提前落地）、app/cli.py 的 analyze、
  configs/manual_review.yaml（人工抽听修正，已按用户 9 段结论落地）。
- 测试：tests/ 7 个文件、**118 项通过**（`pytest tests/` ≈119 s；`pytest tests/ -m slow` 含 6 文件 ASR 边界 ≈192 s）。
- CLI 现有子命令：config show|check、scan、status、decode、analyze（`python -m app.cli --help`）。
- 数据：`out/jobs.db` 里 6 个 job（stage=analyze、status=pending）；
  `work/cache/` 有 6 个 `<id>.16k.wav` + `<id>.decode.json` + `<id>.analyze.json`（约 2.1 GB，已 gitignore）。

## 6 个样例（data/ 只读）与 P2 已定边界（**直接用，不要重新判**）
| key | job id（前缀） | 文件名关键字 | 时长 min | head s | 尾边界 s（裁） | 内部切除 s | clean min |
|---|---|---|---|---|---|---|---|
| f00 | 621cb91c | `0_20140227...`（stco 损坏） | 50.8 | 0.0 | 3000.5（人工裁 46.7） | 291.4 | 45.2 |
| f04 | 9f3feb49 | `...第四卷...` | 138.2 | 0.0 | 8295.0（不裁） | 181.6 | 135.2 |
| f05 | 0a7fa0ed | `...物理学...` | 105.4 | 0.0 | 6325.7（不裁） | 138.1 | 103.1 |
| f06 | b00e197d | `...论语...` | 101.2 | 119.5 | 6070.6（不裁） | 668.8 | 88.0 |
| f20 | 4db18037 | `20_丁耘...` | 102.5 | 89.5 | 6150.5（裁 0.9） | 209.4 | 97.5 |
| f24 | 7927721f | `24_【复旦...` | 77.9 | 479.5 | 4560.5（裁 112.8） | 447.2 | 60.6 |

P2 实测电平（用于 P3 判断降噪/归一化效果，口径见 analyze.py docstring）：
f00 ASL −39.00 / 噪声底 −49.8 / SNR 10.8；f04 −11.99 / −50.5 / 38.5；f05 −12.76 / −36.5 / 23.7；
f06 −21.85 / −29.7 / 7.8；f20 −24.05 / −49.6 / 25.6；f24 −10.10 / −28.2 / 18.1。
内部切除候选明细在 `work/cache/<id>.analyze.json` 的 `internal_silence.candidates`
（每项含 `cut_start`/`cut_end`，已含 0.25 s 保护带），`trim.head_s/tail_s` 是最终边界（含人工修正）。

## 已锁定技术决策（P3 直接照做，不要再讨论/改默认值）
- 目标语音电平 T = **−20.0 dBFS ASL**；跨文件偏差 ≤±0.5 dB。
- 默认 **V1 静态增益 + V2 受控慢变**：窗 20 s / hop 5 s / clamp ±3 dB（源内波动 >6 dB 的文件放宽到 ±6）/
  速率 ≤1.5 dB/s / 20 ms 交叉淡化；增益 clamp 到 [−24, +24]。
- limiter **必须用真峰值**：4× 过采样测峰 → 不达标就下调 ceiling 重做，ceiling 初始 −2.75 dBFS，
  硬指标 **≤−1.0 dBTP**（**禁止用采样峰限幅**：PoC 实测样本峰限幅只到 +0.80 dBTP）。
- 降噪默认 `highpass=f=80,afftdn=nr=12:nf=-30:tn=1` @44.1k；低 SNR 可升 nr=16；arnndn 必须 48k 进出。
- 裁剪：head/tail 用 P2 结果；内部 VAD 非语音 ≥3 s 切除、两侧各留 0.25 s；碎片语音保留。
- 输出 `clean.m4a`：44.1 kHz mono AAC 96 kbps（源 <44.1k 则升到 44.1k）。
- 所有参数走 `configs/default.yaml`（loudness/denoise/trim/output 段已存在，缺什么补什么，不要硬编码）。

## 本轮任务卡（P3-T1~P3-T5，验收必须真跑）

### P3-T1 降噪 —— `app/pipeline/denoise.py`
- ffmpeg 滤镜封装；**arnndn 不能带 Windows 盘符路径**（冒号会被 filter 解析），要 `cd` 到模型目录用 `m=sh.rnnn`；
  rnnoise 只支持 48k，必须 `aresample` 进出。
- 输入/输出：源 44.1k → 降噪后 44.1k float32 临时 wav（`work/cache/<id>.denoised.wav` 或临时文件，阶段末清理）。
- 验收：f24 前 10 min：语音段 ASL 变化 ≤1 dB、非语音段 RMS 下降 ≥3 dB、耗时 ≤5 s/10 min。
- **用户特别要求**：f20 的伴随杂音需处理 —— 对 f20 单独评估，必要时 `afftdn.nr` 提到 16，并给出前后对比数字。

### P3-T2 裁剪与映射 —— `app/pipeline/trim.py` + `tests/test_trim.py`
- 按 `analyze` 的 `kept` 语义生成区间：head=trim.head_s / tail=trim.tail_s / 内部用 candidates 的 `cut_start/cut_end`；
  输出 clean 44.1k wav（numpy 切片或 ffmpeg 拼接）、`edit.json` 的 `kept`（四元组）与 `removed`（reason 用契约枚举）。
- 提供 `orig_to_clean(t)` / `clean_to_orig(t)`（二分+插值）。
- 验收：映射往返误差 <1 ms；语音段（VAD）保留率 100%；**f24 clean 时长**见下方「开工第一件事」。

### P3-T3 响度归一化 —— `app/pipeline/loudness.py` + `tests/test_loudness.py`
- 在 clean 16k 上按同一口径算 `asl_in`（建议用 `asl_p56_dbfs` 与 V2 窗口口径一致，与 PoC 对齐）；
- V1：`gain = T − asl_in`，clamp [−24,+24]；V2：5 s 控制点（20 s 窗）算窗口 ASL，相对静态增益 clamp，
  限速率，20 ms 交叉淡化；**numpy 向量化**（禁止逐样本 python 循环，目标 RTF <0.02）；
- 44.1k 上施加增益 → 真峰值 limiter（4× 过采样测峰，回读 TP 后下调 ceiling 重做，最多 3 次）→
  重新测 ASL，偏差 >0.03 dB 则迭代校正；
- 验收：`tests/test_loudness.py` 用合成信号断言增益数学/clamp/插值/限幅；真实文件：输出 ASL 与 −20 偏差
  ≤0.5 dB、`ebur128` 真峰值 ≤−1.0 dBTP（硬断言）、削波样本 0、最大增益变化率 ≤1.5 dB/s。

### P3-T4 输出音频 —— `app/pipeline/export.py`
- 44.1k float32 → AAC 96k mono m4a；写 `out/<id>/clean.m4a`、`edit.json`（含 `gain` 段）、
  `report.json`（至少 `decode`/`vad`/`levels`/`trim` 段，`AnalyzeResult.report_*()` 可直接用）。
- 验收：`ffprobe` 验证 44100/mono/96k；体积 ≈43 MB/小时；VLC 可播。

### P3-T5 六文件批量渲染 + 跨文件校验 —— `app/tools/regress.py`
- 对 6 文件跑 P1–P3，输出 `work/regress/audio.csv`：in/out ASL、LUFS、真峰值、裁剪比例、耗时。
- 验收：**6 文件 out ASL 极差 ≤1.0 dB**、**全部 ebur128 真峰值 ≤−1 dBTP**、无削波、单文件总耗时 ≤15 min。
- 允许在 `app/cli.py` 增加最小可跑的 `render <id|--all>` 子命令（与 P1/P2 的 decode/analyze 同风格，便于验收）；
  完整编排器留到 P6（`app/pipeline/orchestrator.py`）。

## 开工第一件事（必须先做，写进工程日志再继续）
**核对 P3-T2 的 clean 时长基准**：计划写「f24 clean ≈3412 s（±3%）」，但按 P2 最终边界实算是
**3633.8 s**（head 479.5 + 尾裁 112.8 + 内部 447.2 → 切 1039.5 s），差 6.5%。
3412 疑似沿用 VAD-only 时代的旧口径（内部切除 763 s + 尾 60 s）。请：
1) 用 f24 与前 10 min 试算实际 kept 区间，确认 3634 s 是正确结果；
2) 若确认，则**按实测重订 P3-T2 验收基准**（例如 clean 3634±2%、语音保留率 100%），
   把决定与数据写进 `docs/工程日志.md`，并在代码注释里注明；
3) 不要为了凑 3412 而多切内容（P2 的抽听已确认边界是用户认可的）。

## 执行规则
1. 写作用域：`app/pipeline/{denoise,trim,loudness,export}.py`、`app/tools/regress.py`、
   `tests/{test_trim,test_loudness,test_export}.py`；补参数只改 `configs/default.yaml`；
   需要 CLI 入口时可改 `app/cli.py`。**不要动** data/、docs/research/、work/ 下的 PoC 产物。
2. 不要提交二进制；`out/*.m4a`、`work/**/*.wav`、`out/review/` 等已在 .gitignore。
3. 每个子阶段完成跑验收并本地 commit，消息形如 `feat(P3): denoise with afftdn` / `test: ...` / `docs: ...`。
4. 代码小函数、可单测；异常带明确错误信息并落 `stages` 状态（app/storage.py 已就绪）。
5. 验收必须真实执行（pytest + 六文件命令），失败就修，不要跳过；计划外的阻塞先写 `docs/工程日志.md` 再继续。
6. **代码/文档一律用文件读写工具改，不要用 PowerShell 的 `Get-Content | Set-Content` 改中文源码**
   （上一轮曾把 analyze.py 写成 UTF-8 二次编码 + 丢换行，无法 git 恢复，已重写）。

## 已知坑（务必规避）
- `ffmpeg arnndn` 不接受 `C:` 绝对路径 → cd 到模型目录用相对路径；rnnoise 只支持 48k，须 `aresample` 进出。
- **真峰值 vs 采样峰**：`alimiter` 限的是采样峰，实测真峰值可高 2.6–2.8 dB；f00 是全库最难案例
  （极安静 + 源含满刻度瞬态），PoC 只做到 −0.90 dBTP，本轮必须做到 ≤−1.0。
- f04 源真峰值 **+5.7 dBTP**（已削顶），limiter 会触发，注意别把削顶瞬态放大 20 dB。
- f00 归一化 **+21.5 dB**，底噪会被抬高约 22 dB，降噪与听感要单独抽检（用户会试听）。
- 长音频不要整段喂 ASR（本轮不需要 ASR，但别把 analyze 缓存弄丢）。
- 16k/44.1k 解码统一 float32，禁止 int16（源有 >0 dBFS 过冲，f04 实测 max|x|=1.77）。
- 文件名含中文/空格/括号：全程 `Path` + 列表参数调 subprocess，不要拼 shell 字符串。
- **pip 安装必须 `$env:PIP_CACHE_DIR="$PWD\work\pip_cache"`**（全局 pip 缓存会让 install 无输出空转）；
  PyPI 直连会 read timeout，只用清华源。依赖已装齐，本轮通常不需要装新包。
- 若 pwsh 全部命令报 `0xC0000142`（DLL 初始化失败）：是会话沙箱问题，请切换到完全权限的文件策略后重试。
- 环境事实：`work\venv\Scripts\python.exe`（3.10，已装 sherpa-onnx 1.13.8 / onnxruntime / numpy / scipy /
  faster-whisper / fastapi / uvicorn / PyYAML / pytest，**无 torch、不需要**）；ffmpeg 7.1 在 PATH；
  模型在 work/bench/models（sensevoice、rnnoise）与 work/diar/models。先设 `$env:PYTHONIOENCODING='utf-8'`。

## 完成标准（本轮）
- `pytest tests/` 全绿（新增 test_trim / test_loudness / test_export，且原有 118 项不回归）；
- `python -m app.cli render --all`（或等价命令）能对 6 文件产出 `out/<id>/clean.m4a + edit.json + report.json`；
- `work/regress/audio.csv` 六文件：out ASL 极差 ≤1.0 dB、真峰值全部 ≤−1 dBTP、削波 0、单文件 ≤15 min；
- f20 的降噪对比数字（用户指定关注点）、f00 的听感抽检片段（处理后 A/B）已给出；
- `docs/工程日志.md` 追加 P3 一节（做了什么/验收数字/偏差/遗留），git commit 完成。
```
