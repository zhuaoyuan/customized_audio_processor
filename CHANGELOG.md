# CHANGELOG

本项目遵循「按阶段里程碑」记录（非严格 SemVer 发布节奏）；当前已发布版本 **v0.2.0** = P0–P10
（v0.1.0 = P0–P8；P9 纠错质量修复、P10 全文成稿并入 v0.2.0）。

## [0.2.0] - 2026-10-24 · P10 全文成稿

用户反馈「转写文本仍有很多问题」的根因：`out/` 现有 6 个 job 的 `correct` 阶段因无 API key
全部 `enabled=false / 0 edits`，看到的是 SenseVoice 原始输出；且原纠错 2,000–2,500 字分块、
只看热词与开头 300 字，无法利用整讲语境，prompt 也明确禁止润色合并。P10 新增独立的
`compose` 能力（不改变现有 9 阶段流水线）：

- **通读整讲 → 课程档案**：全讲逐句转写一次调用，生成标题/主题、8–15 条结构大纲、
  20–50 条术语表（正确写法 + 原文真实出现的错写）、人物/书目；虚构的错写会回查原文丢弃；
- **分块整理**：按段边界切块（默认 1,800 字，块间 2 段重叠），每块带课程档案 + 前文结尾，
  合并碎片句、去语气词、按术语表纠错，输出以 `[mm:ss]` 开头的书面自然段；支持 1–16 并发；
- **校验与回退**：每块校验字数比、时间戳合法与单调为硬条件；增删数字/丢书名号进报告
  （Bekker 页码常被模型规范化，不因数字差异整块回退）；失败重试一次，仍失败只回退该块为
  「机械去语气词」并标 `fallback`；
- **块间去重**：字符级 + 句级去掉重叠造成的重复，合并后强制时间轴单调；
- **时间轴**：成稿段首 `[mm:ss]` 为 **clean 音轨**（处理后的 `clean.m4a`）时间；
  `manuscript.json` 每段另附 `orig_time` / `orig_start_s` 原始音轨映射；
- **产物与缓存**：`manuscript.md` / `manuscript.json` / `compose_card.json` /
  `compose_report.json`；课程档案与分块结果按 transcript hash 缓存，改字后只重跑受影响块；
- **CLI**：`python -m app.cli compose <job|--all> [--dry-run] [--force] [--no-cache]
  [--concurrency N] [--chunk-chars N] [--json]`；不在 `run` 全链路里，显式调用才有成本；
- **真实实测**（1 小时 43 分，1,332 段，串行默认）：31,151 字 → 21,145 字（68%）、
  105 段、18 块 18/18 通过（0 回退）、15/15 大纲节点覆盖，语气词清零；
  **¥0.576**（课程档案 ¥0.073 + 分块 ¥0.502；档案请求 99.4% 命中缓存，属重跑热态，
  新讲冷启动约 **¥0.60**）、生成耗时 14.1 min；
- **测试与交付**：新增 `tests/test_compose.py` 18 项（分块/解析/校验/术语表清洗/去重/缓存/
  回退/跳过/CLI/config/双时间轴映射）；新增 `app/tools/package_release.py`（代码+配置+文档+测试打包，
  排除 data/out/work/模型/密钥）与 2 项打包单测；全量 **475 passed**。

### P10.1 运行复盘：计时/成本口径修复（2026-10-24）

一次真实 A+B+C 运行复盘发现两个工程缺口，已修复（详见 `docs/工程日志.md` 同日节）：

- **去掉重复回读**：`export` 在 `tmp_m4a.replace()` 后又对最终 `clean.m4a` 跑了一次
  `measure_m4a`，且这次不计时（本机外置盘上多花 ~68 s）。现复用循环内最后一次测量；
  `measure_m4a` 的临时 `dec44/dec16.wav` 从输出目录改到 `cache_dir` 的 per-job 子目录（默认 C: 本地盘，避免并发重名）。
  同一源文件实测音频链 `total_s` 从 356.8 s 降到 **216.1 s（−39%）**，`measure_s` 65.8→19.5 s、未计时差额 68.6→0.3 s。
- **correct usage 落盘**：`report.correct` 新增 `usage{prompt_tokens,completion_tokens,
  cache_hit_tokens}`，`jobs.db` 的 correct 阶段 metrics 同步携带；此前只写 `cost_cny`，
  无法与账单对账。
- **可观测性**：report 新增 `measure` 段（ebur128/peak/asl 子耗时）；`timing` 补 `write_s`。
- **测试**：`tests/test_export.py` 新增 render_job「只回读一次 + tmp_dir=cache_dir」集成用例，
  `tests/test_correct.py` 补 3 处 usage 断言；全量 **476 passed**。

### P10.2 推理强度（thinking / reasoning_effort）调研与优化（2026-10-24）

背景：DeepSeek V4.1 的 `deepseek-flash` **默认 thinking=on、effort=high**；此前代码不传参，
所以纠错/成稿的 completion 中 70–90% 是 reasoning token，成本和时延都偏高。

- **调研**：官方支持 `thinking:{"type":"enabled|disabled"}` 与
  `reasoning_effort:"none|low|high|max"`（默认 high）；思考模式下 `temperature` 不生效；
  `usage.completion_tokens_details.reasoning_tokens` 可对账。详见 `docs/推理强度调研与实测.md`。
- **实测**（1 讲、gold 4,200 字 + ASR 原文 2 块 + compose 2 块 + 课程档案）：
  - `correct`：**low 与 high 的金标召回相同（91.94%）**、ASR 术语变体召回相同；
    整讲 11 块实测 low 成本/耗时约为 high 的 81% / 80%（3 样本波动大，以整讲为准）；`off` 金标召回 8.06%（不可用）；
    `max` 触发句数护栏、diff 逼近 12%（不适用）。
  - `compose`：**high** 术语/内容准确度最好（low 把 `physis` 写成 `vis`，max 把「四因说」
    错成「元素」且成本约 2 倍）；课程档案 high 比 low 只贵约 ¥0.03/讲，保留 high。
- **改动**：`correct.py` 新增 `reasoning_payload()` / `chat_completion(reasoning_effort=)`，
  `usage` 增加 `reasoning_tokens`；`compose.py` 分块与档案分别传 effort；配置新增
  `correct.reasoning_effort=low`、`compose.reasoning_effort=high`、
  `compose.analyze_reasoning_effort=high`（`app/config.py` 校验 none/low/high/max）。
- **实测**：整讲 11 块 ¥0.6377→¥0.5159（−19.1%）、150.8→120.9 s（−19.8%）；A+B+C 总成本 ¥1.2438→¥1.1220（−9.8%）。
- **测试**：新增 3 项（reasoning payload、reasoning token 归一、compose 两阶段 effort），
  全量 **479 passed**。

### P9 纠错质量修复（2026-10-22）

真实 DeepSeek API 在 P8 验收中被证明「连通/计费正常，但真实输出 7/7 块被拒、0 edits」。
P9 在不放宽安全阈值的前提下让真实输出可写回：

- **块级/逐段校验分层**：新增 `validate_block`（空输出/Markdown/整块 diff/整块句数），
  数字与书名号从「块级一票否决」下放到逐段护栏 `_seg_guard`（空段/diff/数字/书名号/句数）；
- **逐段抢救**：`_salvage_segments` 依次尝试行 → 句 → **字符级对齐**（difflib opcodes 求段区间），
  选通过段数最多的策略；失败段保留原文 + `rejected_segment` flag，同块其它段照常写回；
  整块全拒才记 `rejected_chunks`；
- **数字比较口径**：全角数字转半角、去数字间千位逗号（仅比较，不改写回文本）；
- **prompt 强化**：数字逐字保留、不改全角/半角、不去千位逗号；
- **新产物字段**：`corrections.json.segment_rejections[]`、`report.correct.rejected_segments`、
  transcript flag `rejected_segment`（`app/models.py`）；
- **真实复测（同一 key / `deepseek-flash` / temperature 0）**：
  - f24 全量 7 块：**202 edits / 164 段被改**（1,175 段的 14.0%）、**0 整块回退**、
    121 段 `diff_too_large` 逐段回退、数字被改 0 段、79.2 s、**¥0.379342**；
    P8 时同文件为 0 edits / 7/7 块回退 / ¥0.434；
  - 金标单块 4,200 字：94 edits、diff 3.31%、0 段回退、**¥0.090809**；
  - 质量抽样均为目标错字（形而上学/亚里士多德/吴寿彭/耶格尔/原著选读/满堂灌/范畴篇…），
    逐段 diff 0.011–0.120 全在 12% 护栏内。
- **测试**：新增 8 项（数字归一 / 块级门 / 对齐映射 / 段护栏 / 部分回退 / 全拒原因 /
  correct_job 逐段抢救 / correct_text 数字回退）；全量 **455 passed**，`-m slow` **4 passed**。
- **成本口径更新**：真实纠错 ≈**¥0.379/文件** → 300 文件 ≈**¥114**（此前 P5 估算 ¥33.6 / P8 实测 ¥130）。

## [0.1.0] - 2026-10-22

首个可用版本：本地 CPU 上把课程录音批量处理成「干净音频 + 可校对的带时间戳转写稿」，
附编排器/CLI/本地 Web UI，并完成 6 文件真实回归、500 h 运行评估与交付文档。

### 关键数字（全部来自 6 文件真实产物）

- 6 文件输入 **34,563.25 s（9.601 h）**，clean 后 **32,082.15 s（8.912 h）**；
- 全链路 **RTF 0.1282**（4,431.48 s / 34,563.25 s）→ **500 h ≈ 64.1 h**（±15%：54.5–73.7 h；冷启动 ≈69 h）；
- 交付磁盘 **≈23.0 GB / 500 h**（clean.m4a ≈21.1–21.8 GiB + 文本 ≈0.26 GiB，≤30 GB 达标）；
- 质量：out ASL **−20.02 ~ −20.07 dBFS**（跨文件极差 0.05 dB）、真峰值 **−1.3 ~ −5.1 dBTP**、削波 **0**、
  时间戳段起点 abs median **0.18 s** / P90 **0.47 s**（阈值 0.5 / 1.5 s）；
- 90 s 切片最小验收：**21.9 s** 跑完全链路，8 件产物齐全。

### P0 基础设施

- Python 3.10 项目骨架、`configs/default.yaml` 配置加载/预设/校验、SQLite `jobs/stages` 状态机、
  日志与路径约定；35 项单测。

### P1 解码与输入修复

- `decode`：源文件 → 16k/44.1k float32 wav，统一 16k 分析 / 44.1k 输出、downmix；
  MP4 `stco` 偏移损坏自动修复重试，修不好进 `out/quarantine/`；累计 81 项单测。

### P2 分析（VAD / 电平 / 边界）

- `analyze`：Silero VAD、P56 语音电平、5 min 波动、内部长静音、头尾边界（120 s 窗口 ASR 字符率复核）、
  人工修正机制（`manual_review.yaml`）；累计 111 项单测（含 P4-T1 提前落地）。

### P3 音频渲染（降噪 / 裁剪 / 响度）

- `denoise`（afftdn 44.1k，按 SNR 自动升 nr）、`trim`（clean 轴 + 双向 TimeMap 映射）、
  `loudness`（V1 静态 + V2 受控慢变 + 4× 真峰值限幅 + ASL 校正迭代）、`encode`（AAC 96k 单声道）
  与 P3-T5 六文件回归脚本；最终验收 6/6 通过（单文件 render 192–735 s，均 ≤15 min）。

### P4 转写 + 说话人 + 时间戳测量

- `asr`：clean.m4a → 16k → P2 段表逐段 SenseVoice（int8，8 线程）→ 双时间轴 `transcript.json`；
  f24 **1,175 段 / 15,524 字**，6 文件 9,180 段 / 140,046 字符，ASR 聚合 RTF 0.056；
- `diarize`：none/sampled/full 三档，默认 none（正文 3×5 min 抽样判 n=1）；
- `measure`：faster-whisper 词级参照，时间戳精度 0.18 / 0.47 s（达标）。

### P5 纠错与导出

- `correct`：deepseek-flash 按主题/热词分块（2,000–2,500 字），diff/数字/书名号/句数校验，
  超限整块回退；无 `DEEPSEEK_API_KEY` 自动降级（`enabled=false`，不阻塞）；
- `export`：`transcript.md / .srt / .txt`（clean 轴、逐段一行、UTF-8 无 BOM）；
  成本口径：无 key ¥0；P5 估算 ¥0.112/文件（mock 口径，P8 真实实测为 **≈¥0.434/文件**，见 P8 段）。

### P6 编排器 + CLI（断点续跑 / 失败隔离）

- `app/pipeline/orchestrator.py`：`run_job / batch_run / artifacts / is_complete / JobLock`；
  以产物为准的 resume、单 job 失败隔离、`work/locks` 进程锁（含 stale 锁清理）、磁盘水位、
  attempts 上限、成功后清理 clean16k；
- CLI 新增 `run / batch / retry / status --watch / serve`；`out/jobs.db` 6 job 全部 done/export，
  每 job 9 行 stages；24+26+5 项新单测。

### P7 本地 Web UI

- FastAPI 后端（13 组端点、SSE 进度、音频 Range、裸流上传、改字重导出）+ 原生 HTML/CSS/JS 前端
  （任务列表/详情/播放/分页/改字/导出/防 XSS），无构建、无 CDN、离线可用；
  `start_web.bat` 一键启动（UTF-8，`127.0.0.1:7860`，无鉴权）；26 项服务端集成测试。

### P8 回归、遗留收尾、运行评估与交付文档

- `app/tools/regress.py` 扩展 + `tests/test_regress.py`（P8-T1）：`--reuse` 读真实产物做结构化断言
  （ASL 极差 ≤1 dB、TP ≤−1 dBTP、削波 0、单文件 render ≤900 s、cut_ratio/ASL/TP 与 P3 基线偏差、
  时间戳基线、ASR 段数/字数偏差），`--rerun` 用独立目录真跑；回归结果落 `work/regress/p8*.json`；
- `app/cli.py` 新增 `regress` 子命令（等价 `python -m app.tools.regress`，默认 `--reuse`）；
- P6/P7 遗留收尾：ASR 等模块阶段运行期间 `jobs.stage` 实时更新（orchestrator 补 2 项单测）；
  改字原子写**沿用原文件换行**（原 CRLF 仍 CRLF、原 LF 仍 LF，server 补 2 项单测）；
- `docs/运行评估.md`（P8-T2）：分阶段 RTF、500 h ≈64.1 h、磁盘 ≈23.0 GB、电费 ≈¥7.7、
  纠错成本、**与 ≤30 h 目标的 2.1 倍差距 + 优化选项 + 重订建议**（如实记录，不凑目标）；
- `docs/使用说明.md`（P8-T3）：10 分钟跑通第一个文件（含 90 s 切片最小验收、Web UI、常见问题）；
- `README.md` 收尾：补「回归与运行评估」入口、更新最终测试数字；
- **真实 DeepSeek API 验证（P8 可选收尾 #3，2026-10-22）**：真实 key 跑通 f24 全量（7 块 ≈101 s，
  **≈¥0.434/文件**，P5 估算的 3.9 倍）；但真实模型输出被现有校验/映射 **7/7 全拒**
  （`unsafe_writeback`，0 改动；单块另见 `number_changed`）——连通/计费已验证，**纠错质量路径待 P9**
  （映射鲁棒性、数字校验口径、提示词/模型、成本重估）；
- 本 `CHANGELOG.md`。

### 测试与校验

| 项 | 结果 |
|---|---|
| `pytest tests/` | **447 passed**（415 基线 + 32 P8 新增：regress 25 + orchestrator 2 + server 2 + cli 3） |
| `pytest tests/ -m slow` | **4 passed**（3 基线 + 1 P8 regress 真实 6 文件 reuse） |
| `pytest tests/test_regress.py -m slow` | **1 passed**（6 文件 `--reuse` 断言全过，秒级） |
| `regress --rerun --limit 2` 抽样真跑 | **PASS**：f00+f20、batch 1,054.9 s（≈17.6 min），2 文件各 8 件产物齐全 |
| 6 文件回归 | out ASL 极差 0.05 dB / 最大 TP −1.30 dBTP / 削波 0 / 单文件最长 render 735.5 s / 时间戳达标 |
| 90 s 切片 | 8 件产物齐全；首次冷启动 CLI wall 21.9 s，温缓存复跑 run_job wall 8.1 s |

### 已知限制 / 未验证

- **500 h 总耗时 64.1 h，未达 DoD ≤30 h**（超 2.1 倍）；详见 `docs/运行评估.md` §7（含优化选项与重订建议）；
- **真实 `DEEPSEEK_API_KEY` 已验证连通/鉴权/计费**（f24 ≈¥0.434/文件）；P8 时真实输出被严格校验
  **7/7 全拒、0 改动**——P9 已修复（见 P9 节：202 edits / 0 整块回退，¥0.379/文件）；
- **`work/cache` 的 16k wav 全量保留会到 ≈107 GiB/500 h**，需按任务清理；「≤30 GB」是导出产物的口径；
- Web UI 未做移动端深度适配、Safari 未验证；人工「播放→改字→保存→导出」全流程建议再点一遍；
- `batch` 的 `--jobs` 只支持 1（进程池并行实测慢 22%）。
