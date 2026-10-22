# P8 启动 Prompt（可直接复制给新 agent 会话）

> 前置状态：P0–P7 已完成并提交（最新提交 `2a54260 docs: P6/P7 工程日志…`）；`pytest tests/` = **415 passed**、`pytest tests/ -m slow` = **3 passed**；`out/jobs.db` 6 个 job 全部 `done/export`，每个 9 行 stages（含 P6 补记的 diarize）；6 个 `out/<id>/` 均 8 件产物；`work/regress/audio.csv` 是 P3 的 6 文件回归基线。
> 使用方式：在项目根目录 `C:\projects\local\audio_processer` 新开 agent 会话，把下面代码块整段作为第一条消息发送。

```text
你是「audio_processer」项目的开发 agent，继续执行 **P8（回归 / 500 h 推算 / 交付文档）**，并顺手收掉 P6/P7 的两个已知遗留。
工作目录：C:\projects\local\audio_processer（Windows 10，pwsh 可用）。P0–P7 已完成并提交，本轮只做 P8；P0–P7 的代码/产物是已验收基线，不重写、不重跑六文件产物（除验收明确允许的独立目录重跑）。

## 0. 本轮目标（一句话）
把「6 文件真实回归 + 断言」固化成可重复执行的 `app/tools/regress.py`（含 pytest 用例）；
用**实测 RTF/磁盘/成本**写出 500 小时推算（`docs/运行评估.md`，数字必须如实，达不到目标就写偏差与优化选项）；
补齐交付文档（`docs/使用说明.md`、`CHANGELOG.md`、README 收尾）并追加 P8 工程日志。

## 1. 开工前必读（按顺序，读了再动手）
1) docs/工程日志.md —— 重点读文件末尾三节：
   - 「P6/P7 开工基准与决定」（10 条锁定决策）；
   - 「P6 编排器/CLI + P7 本地 Web UI」整节（全部真实数字、10 条偏差/遗留）；
   - P4/P5 节（ASR/说话人/measure/纠错/导出的基准与遗留）。
2) docs/实施计划.md —— §5 的 P8-T1~P8-T3、§6 回归基线、§7 已知坑、§10 验收清单。
3) docs/开发启动Prompt_P6P7.md —— 上一轮 prompt（了解 P6/P7 契约与当时状态；其中的 P6/P7 任务已完成，不要重做）。
4) 现有实现（复用，不要重写）：
   - `app/tools/regress.py`：P3 已有 6 文件回归脚本（读 render 结果写 `work/regress/audio.csv`，列含 asl_in/asl_out/lufs/tp/gain/clip/decode_s/denoise_s/trim_s/render_s/encode_s/total_s/ok…）；P8 在它基础上扩展，不要推倒重来；
   - `app/tools/measure.py`：时间戳误差测量（faster-whisper 词级参照，P4 已用；`work/measure_f24.json` 是基线证据）；
   - `app/pipeline/orchestrator.py`：`run_job/batch_run/artifacts/is_complete/JobLock`（P8 若做「stage 实时性」才改）；
   - `app/server.py`：`_atomic_write_json`（P8 若处理 transcript 换行风格才改）；
   - `app/config.py`：`load_config / cfg.get / cfg.path / cfg.replace`。
5) README.md —— 已更新为「已可运行」，P8 只做收尾（补回归/评估链接、最终测试数字）。

## 2. 当前仓库状态（不要重做）
- 最新提交（P6/P7，HEAD=`2a54260`）：
  - bcf1501 feat(P6): orchestrator with resume, failure isolation and job locking
  - 2beea35 feat(P6): run/batch/retry/serve CLI and status --watch
  - ccff53d feat(P7): local FastAPI server (SSE, Range, transcript edit)
  - 5cf6cb2 feat(P7): native web UI, start_web.bat and README
  - 2a54260 docs: P6/P7 工程日志（验收真实数字、偏差与遗留）
- 测试基线：`pytest tests/` = **415 passed**（原 360 + orchestrator 24 + server 26 + CLI 5）；`-m slow` = **3 passed**。不得回归。
- CLI 现有：config show|check、scan、status[--watch]、decode、analyze、render、transcribe、correct、export、run、batch、retry、serve。
- DB：`out/jobs.db` 6 job 全部 `done/export`，0 pending；每个 job **9 行 stages**（decode/analyze/denoise/trim/loudness/asr/diarize/correct/export）。
- 6 个样例（data/ 只读，文件名含中文/空格/括号）：

  | key | job id | clean s | render total_s（P3） | asr_s（P4） |
  |---|---|---|---|---|
  | f00 | 621cb91c_0_20140227_丁耘_西方哲学 | 2745.28 | 192.16 | 107.83 |
  | f04 | 9f3feb49_4_丁耘理想国_古希腊哲学_第四卷 | 8131.83 | 496.26 | 367.73 |
  | f05 | 0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05 | 6201.13 | 668.71 | 592.69* |
  | f06 | b00e197d_复旦哲学系丁耘_孔子_论语_06_19 | 5327.85 | 323.39 | 249.38 |
  | f20 | 4db18037_20_丁耘理想国_古希腊哲学 | 5869.60 | 735.54 | 224.60 |
  | f24 | 7927721f_24_复旦哲学系丁耘_亚里士多德 | 3806.46 | 230.15 | 242.56 |

  > 6 文件输入合计 34,563 s（9.6 h），clean 合计 32,082 s（8.91 h）。
- 每个 `out/<id>/` 已有 8 件文件：clean.m4a、edit.json、report.json、transcript.json/.md/.srt/.txt、corrections.json。
  f24：1175 段 / 15,524 字；f00 998 / f04 2299 / f05 1334 / f06 1679 / f20 1695 段。
- `work/regress/audio.csv`：P3 的 6 文件回归基线（真实数字，**不要覆盖**；P8 输出另存 `work/regress/p8_*.json`）。
  关键值：out ASL −20.02~−20.07、真峰值 −1.3~−5.1 dBTP、削波 0、单文件 render 192–735 s、cut_ratio 0.02–0.099。
- `work/measure_f24.json`：P4-T4 时间戳精度基线（段起点 abs median 0.18 s / P90 0.47 s；段终点 0.26 / 0.59）。
- `work/cache/`：6 文件 `<id>.16k.wav + .decode.json + .analyze.json`（gitignore，**不要删/改**）；`.clean16k.wav` 已由 P6 编排器清理。
- 环境事实：
  - Python 用 `work\venv\Scripts\python`（3.10）；ffmpeg 7.1 在 PATH；Node v22（`node --check`）；Chrome 在 `C:\Program Files\Google\Chrome\Application\chrome.exe`。
  - **没有** python-multipart / httpx / requests；不要新增依赖（P8 不需要）。
  - **没有 DEEPSEEK_API_KEY**（P8 的纠错成本用“无 key”与 P5 估算/金标数据说明，不要伪造真实调用）。
  - 设 `$env:PYTHONIOENCODING='utf-8'`；pip 安装（如确需）必须 `$env:PIP_CACHE_DIR="$PWD\work\pip_cache"`。
  - 源码/文档只用 read/write/edit 工具改，禁止 PowerShell `Get-Content | Set-Content` 写中文文件。
  - 本会话 approval 为 never：不要请求沙箱升级。
  - pwsh 里 `cmd`/`taskkill` 可能不在 PATH：用 `C:\Windows\System32\cmd.exe`、`C:\Windows\System32\taskkill.exe` 全路径；后台进程用 `python -m …` 直接起，不要依赖 cmd。

## 3. 本轮范围与边界
- 做 P8-T1/T2/T3 + 两个 P6/P7 遗留收尾（见 §5「可选收尾」）；不做新功能（不改既有 CLI 语义、不改 P0–P5 流水线算法）。
- **不要覆盖** `out/` 现有 6 个 job 的 8 件产物；**不要重跑** P3/P4/P5 的六文件产物（回归一律用独立目录）。
- 回归/重跑只能用独立 `out_p8/`、`work/cache_p8/`（gitignore 已有 out_p6/out_p7；可 add out_p8），验收后删除。
- `data/`、`docs/research/`、`work/cache` 的 16k/analyze/decode、`work/measure_f24.json`、`work/cost/`、`work/regress/audio.csv` 只读。
- 新文档写在 `docs/`；git 只 add 具体文件，禁止 `add -A`；提交由 Lead 统一做（分提交）。
- 不新增测试依赖；新测试不依赖网络/真实 LLM。

## 4. 开工第一件事（必须先做，写进工程日志再继续）
1) 跑基线：`pytest tests/ -q`（415 passed）、`pytest tests/ -m slow -q`（3 passed）；`python -m app.cli status`（6 done/export）；确认 6 个 `out/<id>/` 8 件齐全、`work/regress/audio.csv` 仍是 6 行。
2) 读 `out/<id>/report.json` 的 timing/levels/asr/correct：记录本轮推算用的**实测分阶段耗时**（见 §2 表），并算出：
   - render（P1–P3）RTF = Σrender_total_s / Σ输入时长 = 2,646.21 / 34,563 = **0.0766**；
   - ASR RTF = Σasr_s / Σclean = 1,784.8 / 32,082 = **0.0556**；
   - 全链路 RTF ≈ (2,646 + 1,785 + 杂项) / 34,563 ≈ **0.129** → **500 h ≈ 64 h**。
   → **P8-T2 的“总耗时 ≤30 h”按实测达不到**，这是本轮必须如实写出的结论（见 §5 P8-T2 与 §9 已知坑）。
3) 在 `docs/工程日志.md` 追加「P8 开工基准与决定」：基线数字、回归模式（`--reuse` 为主 + 独立目录 `--rerun`）、500 h 口径（含 RTF 实测与 30 h 目标差距的处理）、文档清单、写作用域。定完再写代码。

## 5. 任务卡

### P8-T1 6 文件回归 —— `app/tools/regress.py`（扩展）+ `tests/test_regress.py`
对外接口（CLI 与测试共用，固定）：
- `run_regression(data_dir, cfg, *, mode="reuse"|"rerun", out_dir=None, cache_dir=None, limit=None) -> dict`
  返回 `{"ok", "mode", "n", "rows":[...], "summary":{...}}`；`rows` 每项含 job_id/source/clean_s/asl_out_dbfs/asl_dev_db/true_peak_out_dbtp/clipped_samples/cut_ratio/render_total_s/asr_s/total_wall_s/n_segments/chars/ok/error。
- `check_regression(result, *, baseline_csv=None, scan_dir="work/scan") -> dict`
  返回 `{"ok", "failures":[...], "summary":{...}}`；断言：
  1. **跨文件 out ASL 极差 ≤1.0 dB**（实测 0.05，P3 基线）；
  2. **全部真峰值 ≤ −1.0 dBTP**（实测 −1.3~−5.1）；
  3. **削波样本 = 0**；
  4. **单文件 render total_s ≤ 900 s**（15 min，实测最大 735.5 s）；
  5. **cut_ratio 与 P3 基线（work/regress/audio.csv）偏差 ≤5 个百分点**，ASL/TP 与基线偏差（ASL ≤0.5 dB、TP ≤1.0 dB）；
  6. **时间戳**：读 `work/measure_f24.json`（或重新测）断言段起点 abs median ≤0.5 s、P90 ≤1.5 s（P4 实测 0.18/0.47，达标）；
  7. ASR 段数/字数与 P4 基准偏差 ≤10%（f24 1175/15,524 等，见 §2）。
- 两种模式：
  - `--reuse`（默认，快）：只读 `out/<id>/report.json + 8 件产物` 做断言，不跑流水线；这是 `pytest -m slow` 的默认路径；
  - `--rerun`（慢）：用**独立** `--out out_p8 --cache-dir work/cache_p8` 真跑（`orchestrator.batch_run` 或 `render_job+transcribe_job`），再断言；**禁止写 out/**。建议先 `--rerun --limit 2`（f24 + f20）验证链路，`--limit 6` 约 80 min 放后台跑，只在时间允许时做。
- CLI（可选，若加只能由 Lead 改 `app/cli.py`）：`python -m app.cli regress [--reuse|--rerun] [--limit N] [--json]`。
- `tests/test_regress.py`：
  - 快速用例（默认跑）：用 tmp 目录造 6 个假 `out/<id>/report.json` + 8 件文件，覆盖 `check_regression` 各断言通过/失败路径（极端 ASL 极差、TP 超标、cut_ratio 漂移、缺产物、measure 基线缺失）；
  - `@pytest.mark.slow`：`--reuse` 真读 6 个真实 job 断言全过（秒级），并把 `work/regress/p8.json` 落盘；
  - 真实 `--rerun` 不进 pytest（80 min 太慢），作为手册验收写入工程日志。
- 验收：`pytest tests/test_regress.py -q`、`pytest tests/test_regress.py -m slow -q` 全绿；`work/regress/p8.json` 有 6 行 + summary；`--rerun --limit 2` 真跑一张单文件的八件产物齐全（独立目录，验收后删）。

### P8-T2 500 小时推算 —— `docs/运行评估.md`
内容（数字全部来自本轮实测，禁止照抄 PoC/计划的乐观值）：
1. 分阶段 RTF 表：decode/analyze/denoise/trim/loudness/encode/measure/asr/diarize/correct/export_text 的“6 文件合计耗时 / 输入时长”，标明哪些含缓存命中（render 是 P3 真实渲染耗时，asr 是 P4 真实耗时）。
2. 500 h 推算：300 文件 × 100 min，按实测 RTF 0.129 → **约 64 h**（单进程、ASR 8 线程、无并发）；给区间（±15%）。
3. 磁盘：clean.m4a ≈45 MB/h → 500 h ≈ 22.5 GB；transcript/json/md/srt/txt + report ≈1 MB/h → 合计 **≈23–25 GB（≤30 GB 达标）**；cache 峰值估算（16k 1.86 GB/8.9 h…按实测外推）与清理策略（保留 16k/analyze/decode，删 clean16k/中间 wav）。
4. 电费：整机约 200 W × 64 h ≈ 12.8 kWh × 本地电价（写清假设）≈ ¥7–10。
5. 纠错成本：无 key 为 0；有 key 按 P5 金标/f24 mock 实测（7 块 ≈¥0.112/文件）→ 300 文件 ≈¥33.6；给出真实 API 未验证的免责说明。
6. **与 ≤30 h 目标的差距**：如实写“实测 64 h，超目标 2 倍”，给优化选项与预计收益（合并短 ASR 段、减少 measure 迭代/ebur128、`--jobs` 流水线化、换更快 CPU/SSD、跳过边界 ASR 复用缓存等），并给“重订目标为 ≤72 h 或按硬件缩放”的建议。**不要改数字凑目标。**
7. 复现命令与数据来源（`out/*/report.json`、`work/regress/audio.csv`、`work/measure_f24.json`）。

### P8-T3 文档与交付 —— `docs/使用说明.md`、`CHANGELOG.md`、`README.md`
- `docs/使用说明.md`（面向新用户，10 分钟跑通第一个文件）：
  - 环境准备（Python 3.10 / ffmpeg / 模型目录 / `pip install -r requirements.txt`）；
  - 快速路径：`scan → run → status/--watch → serve/Web`，含每个命令的预期输出；
  - 用 90 s 切片做“最小验收”的命令（ffmpeg 切 data/ 里任一 m4a → `out_demo/`），预期 8 件产物与耗时（P7 实测 90 s 样本全链路 ≈10 s）；
  - 常见问题（无 key 纠错降级、中文路径、模型缺失、端口占用、Web 无鉴权）。
  - **命令必须逐条实测可复制执行**（pwsh，UTF-8）。
- `CHANGELOG.md`：v0.1.0（P0–P8 一句话/阶段汇总 + 测试数字 + 关键数字），按阶段分条。
- `README.md`：补「回归与运行评估」小节（指向 `app/tools/regress.py`、`docs/运行评估.md`、`docs/使用说明.md`），更新最终测试数字（415 + 新增 regress 用例数）。

### 可选收尾（P6/P7 遗留；低优先，Lead/S 级）
1. **ASR 期间 stage 实时**：`orchestrator._invoke` 调用模块阶段前 `start_stage(job_id, stage, reset=True)`（已 done 且非 force 时不重置），让 `jobs.stage` 在 ASR/diarize/correct 跑动时显示当前阶段；补 2 项单测；改后跑 `tests/test_orchestrator.py` 与全量。
2. **transcript.json 换行风格**：server 原子写把 CRLF 规范成 LF（P7 验收记录）；二选一：保持现状并在 `docs/使用说明.md` 注明，或让 `_atomic_write_json` 沿用原文件换行（读 bytes 判断 `\r\n`）。改了要补 server 测试。
3. **真实 DEEPSEEK_API_KEY**（若本轮能拿到 key）：用 1 块文本跑一次真实纠错，记录 `rejected_chunks`/diff，补进工程日志；拿不到就维持「未验证」结论。
4. **Web 真人点击**：headless 已覆盖渲染/翻页/seek/错误态；请用户或 Lead 用 Chrome 人工点一次「播放→改字→保存→导出」。

## 6. 执行方式（建议 Agent Teams；单人则按 P8-T1→T2→T3 顺序）
写作用域必须互不重叠；`app/cli.py` 只由 Lead 改；git 由 Lead 统一提交。
- `regress-dev`：`app/tools/regress.py`、`tests/test_regress.py`（P8-T1，关键路径）。
- `docs-dev`：`docs/运行评估.md`、`docs/使用说明.md`、`CHANGELOG.md`、`README.md`（P8-T2/T3，可与回归并行）。
- Lead：基线复跑、真实 `--reuse` 回归与 `--rerun --limit 2/6`、`app/cli.py`（如加 regress 子命令）、可选收尾（orchestrator/server 小改）、`docs/工程日志.md`、最终全量测试、分提交。
依赖：P8-T2 的 RTF 表来自 P8-T1 的 `--reuse` 结果，docs-dev 可先写框架，数字由 Lead 回填；P8-T3 的使用说明建议在 `--rerun --limit 1` 真跑一次后定稿。
并行约定：pytest 各自 `--basetemp=work/pytest_tmp_<名字>`；不要跑全量 pytest（Lead 最后统一跑）；git 只 add 自己文件、禁止 `add -A`；**不要同时启动两个真实 pipeline**（ASR 单例 8 线程会互相拖慢）；临时 out 用 `out_p8/`、`work/cache_p8/`，完成后清理。

## 7. 已锁定技术决策（照做，不改默认值）
- 回归以 \"`--reuse` 读真实产物 + 结构化断言\" 为主、\"独立目录 `--rerun`\" 为真跑证据；**绝不覆盖 out/ 现有 6 job 的 8 件产物**。
- 500 h 推算口径：单进程、ASR 8 线程、本机 i5-12400F；RTF 用 6 文件实测聚合值；超目标如实写并给优化选项/重订建议。
- 不新增依赖（P8 不需要 fastapi/httpx 之外的新包；回归脚本用 stdlib + numpy/pandas 现状）。
- 文档全 UTF-8、pwsh 命令可直接复制；中文文件名用引号/变量处理。
- 提交分三类：代码+测试（P8-T1/收尾）、文档（运行评估/使用说明/CHANGELOG/README）、工程日志。

## 8. 验收（必须真跑，不要只写测试）
1) 测试：`pytest tests/` 全绿（原 415 + 新增 regress 测试）；`pytest tests/ -m slow` 3 passed 不回归 + `tests/test_regress.py -m slow` 通过。
2) `pytest tests/test_regress.py -q` 通过；`--reuse` 真跑 6 文件断言全过，`work/regress/p8.json` 落盘（6 行 + summary：ASL 极差、max TP、max clipped、最大单文件 render、RTF）。
3) `--rerun --limit 2`（独立 `out_p8/ cache_p8`）：2 个文件 8 件产物齐全、report 有 asr/speakers/correct，与 `--reuse` 断言一致；耗时记录。若时间允许 `--limit 6` 全量重跑（≈80 min，后台）。
4) `docs/运行评估.md` 有分阶段 RTF 表、500 h 推算（≈64 h）、磁盘（≈23–25 GB）、电费、纠错成本、**30 h 目标差距与优化/重订建议**。
5) `docs/使用说明.md` 的命令逐条实跑通过：按文档从 0 到「90 s 切片跑出 8 件产物 + Web 打开」；README/CHANGELOG 更新；新用户 10 min 可跑通。
6) 可选收尾若做了：orchestrator/server 测试通过，全量不回归；工程日志记录。
7) `docs/工程日志.md` 追加 P8 一节（做了什么/真实验收数字/偏差与遗留），代码/文档/日志分提交；提交消息风格同 P0–P7。

## 9. 已知坑（务必规避）
- **不要写 out/**：`render_job --force` 会重写 `out/<id>/clean.m4a`；回归/重跑一律独立 `out_p8/`。
- **RTF 不要抄 PoC/计划**：PoC ASR RTF 0.026 是整段喂入口径；本机真实分段 ASR 0.056、render 0.077、合计 0.129 → 500 h ≈64 h，计划 24 h/DoD 30 h 是按旧口径估的；如实写。
- **真实重跑很慢**：`--rerun --limit 6` ≈80 min（render 44 min + ASR 30 min + decode/analyze 6 min）；不要并发跑两个真实 pipeline；`--rerun` 前台会超时，用后台 job + 轮询。
- **时间戳回归**需要 faster-whisper 模型（P4 已在本机缓存）；优先读 `work/measure_f24.json` 断言，不要重复下载/重复测 10 min（一次测量约数分钟）。
- **measure/ebur128 的耗时**在 report.timing 里有（render_total 含 encode/measure）；推算时不要重复计。
- **别覆盖 P3 基线** `work/regress/audio.csv`；P8 结果另存 `work/regress/p8_*.json`。
- **Chrome headless 截图**：不要用 `--virtual-time-budget`（页面有 SSE 长连接会挂起），用 `--headless --disable-gpu --no-sandbox --timeout=15000 --screenshot=… --window-size=1440,900`；截图放 `work/`（gitignore）。
- **pwsh 里 cmd/taskkill 可能不在 PATH**：用全路径；后台进程用 `python -m app.cli …` 直接起。
- **测试污染**：pytest 一律 `--basetemp=work/pytest_tmp_*`；临时 out 用 `out_p8/`、`work/cache_p8/`，完成后删除。
- **中文/UTF-8**：所有文件显式 `encoding='utf-8'`；不要用 PowerShell 管道改源码。
- **提交**：只 `git add` 具体文件；`out/<id>/*.json`、`work/*` 未跟踪文件很多，禁止 `add -A`/`add .`。

## 10. 交接资料
- 提交与状态：HEAD=`2a54260`；`git log --oneline -6`；415 单测 / 3 slow；6 job done、每 job 9 stage 行。
- 数据：6 文件 job id/clean 秒见 §2 表；`work/regress/audio.csv`（P3 回归基线，含全部指标列）；`work/measure_f24.json`（时间戳基线）；`work/transcribe_f24.json`、`work/diar_p4_check.json`、`work/cost/correct_test/`（P5 金标）。
- 文档：`docs/工程日志.md`（P0–P7 全部真实数字与遗留）、`docs/实施计划.md` §5/§6/§10、`docs/开发启动Prompt_P6P7.md`、`docs/调研报告_需求对齐版.md` §9。
- 现有产物样例：`out/<id>/report.json` 字段顺序/内容可直接参考（timing/levels/asr/speakers/correct/output）。
- 关键命令：`python -m app.cli status --json`、`python -m app.cli run --help`、`python -m app.cli batch --help`、`python -m app.tools.regress --help`（如已加 CLI）、`pytest tests/test_regress.py -q`。
- 本轮 scratch 约定：`out_p8/`、`work/cache_p8/`（.gitignore 已有 out_p6/out_p7，可补 out_p8），验收后删除。

## 11. 完成标准（本轮 DoD）
1) pytest 全绿：415 不回归 + 新增 regress 测试；slow 3 不回归；`tests/test_regress.py -m slow` 通过。
2) 6 文件回归：`--reuse` 断言全过（ASL 极差 ≤1.0 dB、TP ≤−1 dBTP、削波 0、cut_ratio ±5pp、单文件 render ≤900 s、时间戳基线达标）；`work/regress/p8.json` 落盘；`--rerun --limit 2` 真跑 2 文件 8 件产物齐全（独立目录）。
3) `docs/运行评估.md`：分阶段 RTF + 500 h 推算（≈64 h，含与 30 h 目标差距、优化选项与重订建议）+ 磁盘 ≤30 GB + 电费 + 纠错成本。
4) `docs/使用说明.md` 实测可跑通「90 s 切片 → 8 件产物 → Web 播放/改字/导出」；`CHANGELOG.md` 完成；README 收尾。
5) `docs/工程日志.md` 追加 P8 真实数字/偏差/遗留；代码/文档/日志分提交（禁止 add -A）。
6) 可选收尾（stage 实时性/换行风格/真实 key）若做则测试通过并记录，未做则在遗留里写清。
```

---

### 交接给下一轮 agent 的提醒（不属于 prompt 正文）
- 本轮最容易翻车的是 **P8-T2 的数字**：计划里的 ≤30 h 是 PoC 旧口径，实测约 64 h；如实写 + 给优化选项，不要为凑目标改 RTF。
- 回归优先用 `--reuse`（秒级、读真实产物）覆盖断言；`--rerun` 只做抽样真跑（`--limit 2`），全量 6 文件约 80 min，放后台且必须独立 `out_p8/`，别碰 `out/`。
- 文档类工作可与回归并行；`docs/使用说明.md` 建议等 `--rerun --limit 1` 真跑通过后再定稿。
- P6/P7 遗留三项（stage 实时性、transcript 换行、真实 key）都是小改，优先保 P8 主线。
