# audio_processer

把公开平台的课程录音批量处理成「**干净音频 + 可校对的带时间戳转写稿**」的本地工具：
解码 → 分析（VAD / 语音电平 / 裁剪边界）→ 降噪 → 裁剪 → 响度归一化 → ASR 转写 →
说话人标注 → LLM 纠错 → 导出 `clean.m4a` + `transcript.md/srt/txt`。
全部在本机 CPU 上跑（不需要 GPU / torch），另带一个只在 `127.0.0.1` 监听的本地 Web UI，
可以完成「扫描/上传 → 跑 → 播放 → 改字 → 导出」全流程。v0.2.0 起另附可选的 `compose`：
把整讲转写通读后整理成完整、通顺的书面讲稿（独立命令，不改变主流水线）。

## 状态

当前版本 **v0.2.0**（P0–P10）。

- ✅ P0 基础设施、P1 解码、P2 分析、P3 渲染（降噪/裁剪/响度）、P4 ASR + 说话人、P5 纠错 + 文本导出
- ✅ P6 编排器与 CLI：断点续跑、失败隔离、磁盘水位、quarantine、`run / batch / retry / status / serve`
- ✅ P7 本地 Web UI：原生 HTML/CSS/JS，无构建、无外部依赖，离线可用
- ✅ P8 回归 / 500 h 推算 / 交付文档：`app/tools/regress.py` 六文件回归、`docs/运行评估.md`、
  `docs/使用说明.md`、`CHANGELOG.md`
- ✅ P9 纠错质量修复：真实 DeepSeek 输出可安全写回（字符对齐映射 + 逐段护栏回退），
  f24 实测 202 edits / 164 段、0 整块回退；真实成本 ≈¥0.379/文件
- ✅ P10 全文成稿（`compose`）：整讲课程档案 + 分块整理为书面稿；1 小时 43 分实测
  31,151→21,145 字、105 段、0 回退，成本 ≈¥0.58–0.61/讲（新讲冷启动约 ¥0.60）、串行 14 min（独立命令，默认不进 `run`）

`data/` 下 6 个真实样例（576 min）已跑完，产物在 `out/<job_id>/`，任务库是 `out/jobs.db`。
实测全链路 **RTF 0.1282 → 500 h（300 文件 × 100 min）≈ 64.1 h**、交付磁盘 **≈23 GB / 500 h**；
与 DoD「≤30 h」的差距、优化选项与重订建议见 [docs/运行评估.md](docs/运行评估.md)。

## 安装

要求 **Windows 10**、**Python 3.10**、**ffmpeg 7.x 在 PATH**。

```powershell
# 1) 建虚拟环境
py -3.10 -m venv work\venv

# 2) 装依赖（本机实测：必须把 pip 缓存指到新目录，否则会无输出空转）
$env:PIP_CACHE_DIR = "$PWD\work\pip_cache"
work\venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 校验环境与配置
work\venv\Scripts\python -m app.cli config check
work\venv\Scripts\python -m app.cli --version
```

模型不在仓库里（体积大、已在 `.gitignore` 排除），需要放到配置约定的位置：

| 用途 | 路径（默认） |
|---|---|
| ASR（SenseVoice int8 + tokens） | `work/bench/models/sensevoice/`（`model.int8.onnx`、`tokens.txt`） |
| 降噪（RNNoise） | `work/bench/models/rnnoise/sh.rnnn` |
| 说话人（可选，默认 `diar.mode=none` 不加载） | `work/diar/models/` |

模型目录可用环境变量 `MODEL_DIR` 覆盖；其余路径见 `configs/default.yaml` 的 `paths` 段。

**LLM 纠错是可选的**：设置 `$env:DEEPSEEK_API_KEY='sk-...'` 才会真正调用；
不设置时 `correct` 阶段自动 `enabled=false` 并写空 `corrections.json`，**不阻塞**整条流水线。

## 快速开始（CLI）

所有命令都在项目根目录 `C:\projects\local\audio_processer` 下执行（PowerShell 建议先
`$env:PYTHONIOENCODING='utf-8'`，或直接用 `start_web.bat` 自带的 UTF-8 控制台）。

```powershell
# 查看合并后的配置 / 校验
work\venv\Scripts\python -m app.cli config show
work\venv\Scripts\python -m app.cli config check

# 扫描目录，把文件登记为 pending 任务
work\venv\Scripts\python -m app.cli scan data/ --out out/

# 单文件全链路：decode → analyze → denoise → trim → loudness → asr → diarize → correct → export
work\venv\Scripts\python -m app.cli run <job_id 或源文件路径>

# 批量跑目录（--resume 跳过已完成；--limit 限制个数；--force 重跑已完成）
work\venv\Scripts\python -m app.cli batch data/ --out out/ --resume

# 重试失败任务（重置失败阶段及其下游后重跑）
work\venv\Scripts\python -m app.cli retry <job_id>

# 状态汇总（--watch 每 2 s 刷新，Ctrl-C 退出）
work\venv\Scripts\python -m app.cli status
work\venv\Scripts\python -m app.cli status --watch

# 全文成稿（可选，独立于 run 全链路；调用 DeepSeek，约 ¥0.5–0.8/讲）
work\venv\Scripts\python -m app.cli compose <job_id>
work\venv\Scripts\python -m app.cli compose <job_id> --dry-run   # 只看分块方案，不调 API

# 启动本地 Web UI（只 bind 127.0.0.1）
work\venv\Scripts\python -m app.cli serve --port 7860
```

每个 job 完成后 `out/<job_id>/` 会有 8 件产物：

```
clean.m4a         降噪 + 裁剪 + 响度归一化后的成品音轨
transcript.json   段表（start/end/orig_start/orig_end/speaker/text_raw/text/flags；
                  flags 可能含 rejected_chunk / rejected_segment）
transcript.md     带时间戳的转写稿
transcript.srt    字幕（clean 轴）
transcript.txt    纯文本
corrections.json  纠错明细（items / rejected_chunks / segment_rejections；无 key 时为空结构）
report.json       全流程指标（decode/vad/levels/trim/denoise/loudness/asr/speakers/correct…）
edit.json         裁剪/增益编辑记录
```

额外跑 `compose` 时，同一目录会多出成稿产物：

```
manuscript.md       全文整理稿（段首 [mm:ss] 为 clean 轴，对齐 clean.m4a；书面语、去语气词、按术语表纠错）
manuscript.json     段落/时间/来源块 + 统计（供程序读取）
compose_card.json   课程档案（结构大纲 + 术语表；transcript 变化后自动失效）
compose_report.json 分块校验、回退、用量与成本
```

## 全文成稿（`compose`，可选）

`correct` 只改同音错字、不润色；`compose` 解决「整段读不通」的问题：先把**整讲**通读成一份
课程档案（结构大纲 + 术语表），再按约 1,800 字分块，把 ASR 口语整理成书面讲稿。块间带重叠、
成稿后自动去重；每块长度/时间戳硬校验、数字/书名号差异进报告，失败只回退该块，不阻塞整篇。

- 实测（1 小时 43 分讲座）：31,151 字 → 21,145 字（68%）、105 段，语气词清零，
  35 条术语全部出现；串行约 14 min，成本约 **¥0.58–0.61/讲**（¥0.576 为档案缓存重跑热态，
  新讲冷启动约 ¥0.60；默认串行；实测 4 路长请求会被网关排队/超时，确有配额再调 `--concurrency`）。
- **时间轴**：段首 `[mm:ss]` 是 **clean 轴**（对齐处理后的 `clean.m4a`）；原始视频时间见
  `manuscript.json` 每段的 `orig_time`（`transcript.md` 括号内的 orig 时间同源）。
- 默认**不在** `run` 全链路里，必须显式调用；`--force` 重做、`--no-cache` 忽略分块缓存、
  `--concurrency N` 调并发；`--dry-run` 不调 API。
- 成稿**未逐字校对**：引文、页码/Bekker 号、专名仍需对照原音复核；无法辨认的乱码会标
  `〔原音不清〕` 或按上下文略去。

## 使用 Web UI

1. 双击根目录的 **`start_web.bat`**（会设置 UTF-8、`cd` 到脚本目录、启动服务并 `pause`；不自动开浏览器）。
2. 浏览器打开 **http://127.0.0.1:7860**。
3. **登记任务**：
   - 顶部输入本地目录（默认 `data/`）→ 点「扫描登记」；
   - 或点「上传音频…」选一个/多个文件 → 服务端保存到 **`out/uploads/`** 并登记（不写 `data/`）。
4. 左侧任务列表点选任务（短 id、状态色、阶段、耗时、错误摘要；顶部汇总 pending/running/done/failed）。
5. 详情页可：
   - 用 `<audio>` 播放 `clean.m4a`（服务端支持 Range，可拖动进度）；
   - 点「开始处理」跑全链路，或「重试」重置失败阶段后重跑；
   - 看流水线阶段、产物列表、关键指标；
   - 在转写表格里翻页（每页 100 行），点时间（如 `[12:34]`）跳转播放；
   - 修改「修正」列的 `text`（「原文」列 `text_raw` 只读），点「保存修改」→ 写回
     `transcript.json` 并重新导出 md/srt/txt；
   - 下载 md / srt / txt 导出文件。
6. 进度通过 SSE（`GET /api/events`）实时刷新；断线时浏览器自动重连，页面另有兜底轮询。

上传上限由服务端控制（单文件 2 GB）；同一任务正在跑时，改字会被拒绝（HTTP 409）。

## 回归与运行评估

6 文件真实回归脚本是 `app/tools/regress.py`（P8-T1 在 P3 六文件回归基础上扩展），
也可用 CLI 入口 `python -m app.cli regress`（与脚本等价，默认 reuse）。两种模式：

- `--reuse`（默认，秒级）：只读 `out/<job_id>/report.json` + 8 件产物做结构化断言，不重跑流水线；
- `--rerun`：用**独立** `--out out_p8 --cache-dir work/cache_p8` 真跑（禁止写 `out/`）。

```powershell
$env:PYTHONIOENCODING='utf-8'

# 读真实产物做断言（ASL 极差 ≤1 dB、TP ≤−1 dBTP、削波 0、单文件 render ≤900 s、
# 与 P3 基线的 cut_ratio/ASL/TP 偏差、时间戳基线、ASR 段数/字数偏差）；结果落 work/regress/p8.json
work\venv\Scripts\python -m app.tools.regress --reuse --json
# 等价的 CLI 入口（默认 reuse）
work\venv\Scripts\python -m app.cli regress --reuse --json

# 独立目录真跑抽样（慢，用后台 job；禁止写 out/）
work\venv\Scripts\python -m app.tools.regress --rerun --limit 2 --out out_p8 --cache-dir work\cache_p8 --json
work\venv\Scripts\python -m app.cli regress --rerun --limit 2 --json
```

本机 `--reuse` 实测：`ok=true`、6/6 文件通过；ASL 极差 0.05 dB、最大真峰值 −1.30 dBTP、
削波 0、最大单文件 render 735.5 s、cut_ratio/ASL/TP 与 P3 基线偏差 0、时间戳 0.18/0.47 s、
ASR 段数/字数偏差 0；汇总见 `work/regress/p8.json`。

`--rerun --limit 2` 抽样真跑（按 data 路径排序取 **f00+f20**，独立 `out_p8/cache_p8`）实测 **PASS**：
batch 1,054.9 s（≈17.6 min），2 文件各 8 件产物齐全、report 含 asr/speakers/correct；
明细见 [docs/运行评估.md](docs/运行评估.md) §8.5。

回归断言口径、6 文件分阶段 RTF 表、500 h 推算（64.1 h）、磁盘（≈23 GB）、电费、
纠错成本、与 DoD ≤30 h 目标的差距和优化/重订建议，见
[**docs/运行评估.md**](docs/运行评估.md)；从零跑通第一个文件的 10 分钟指南见
[**docs/使用说明.md**](docs/使用说明.md)。

## 已知限制

- **性能**：本机实测全链路 RTF 0.1282，500 h 全量约 **64.1 h**，**未达 DoD「≤30 h」**（超约 2.1 倍）；
  差距原因、优化选项与重订建议见 [docs/运行评估.md](docs/运行评估.md) §7。
- **只监听 127.0.0.1，无鉴权**：这是本机工具，不要改 `--host 0.0.0.0` 或转发到公网。
- **单进程 / 单机**：`--jobs` 参数保留但只支持 1；同一 job 不会重复跑（重复提交返回 409），
  但不要并发跑多个大任务（ASR 单例 8 线程，会争 CPU）；Web 与 CLI 也不要同时跑同一个 job。
- **中文文件名 / 路径**：含中文、空格、括号的样例实测可用；请保持控制台 UTF-8
  （`start_web.bat` 已 `chcp 65001`）。超长路径（>260 字符）未验证。
- **首跑模型加载慢**：SenseVoice 首次加载 + 模型目录不在仓库；模型缺失会让 ASR 阶段失败。
- **真实 DeepSeek：P9 后已验证可用**：2026-10-22 真实 key 跑通 f24 全量（7 块 ≈79 s、
  **≈¥0.379/文件**，是 P5 估算 ¥0.112 的 3.4 倍）；P8 时真实输出 7/7 被拒（0 改动），P9 加
  字符对齐映射 + 逐段护栏后，f24 **202 edits / 164 段被改、0 整块回退**，121 段
  `diff_too_large` 保守回退原文（`segment_rejections`）。细节见
  [docs/工程日志.md](docs/工程日志.md) P9 节、[docs/运行评估.md](docs/运行评估.md) §6.4。
  无 key 时纠错自动降级，不影响其余产物。
- **推理强度可配（P10.2）**：`deepseek-flash` 默认 thinking=on、effort=high；默认已改为
  `correct.reasoning_effort=low`（金标召回与 high 相同，整讲实测成本/耗时约降 20%）、
  `compose.reasoning_effort=high`、`compose.analyze_reasoning_effort=high`。可选
  `none/low/high/max`；`temperature` 在思考模式下不生效。实测与依据见
  [docs/推理强度调研与实测.md](docs/推理强度调研与实测.md)。
- **纠错写回保守**：整块 diff 超过 12%、句数变化超 5% 才整块回退；单段超限**只回退该段**
  （同块其它段的修正保留，flag=`rejected_segment`）；极短段里的 1 字修正可能被逐段回退
  （宁少改不串行）；不支持人工在此 UI 里直接编辑 `corrections.json`。
- **改字范围**：只改 `text`，保留 `text_raw` / 时间轴 / 说话人；P8 起原子写沿用原文件换行
  （CRLF→CRLF、LF→LF）；保存后 md/srt/txt 会被覆盖重写。
- **全文成稿是「整理」不是「校对」**：`compose` 会删口头禅、合并碎片句、按术语表纠错，
  也可能略去极短的无效碎片；不保证逐字保真，引用/页码需人工对照原音。它复用 `correct` 的
  API key，但不是 `run` 的必跑阶段（默认不做，避免成本失控）。
- **播放**：2 小时级 `clean.m4a` 首次定位可能需要等浏览器缓冲；Safari 未验证，建议 Chrome/Edge。
- **前端**：未做移动端深度适配；无构建链、无 CDN，离线可用。

## 测试

```powershell
# 全量单测 479 项（455 基线 + P10 成稿 18 + 打包 2 + P10.1 回归 1 + P10.2 推理 3）；必须全绿
work\venv\Scripts\python -m pytest tests/ -q

# 慢测 4 passed（3 基线 + 1 P8 regress 真实 6 文件 reuse，约 7–8 分钟）
work\venv\Scripts\python -m pytest tests/ -m slow -q

# P8 六文件回归用例：25 passed（快速）；1 passed（slow，真实 6 文件 --reuse）
work\venv\Scripts\python -m pytest tests/test_regress.py -q
work\venv\Scripts\python -m pytest tests/test_regress.py -m slow -q
```

## 打包交付

```powershell
work\venv\Scripts\python -m app.tools.package_release
# → dist/audio_processer-v0.2.0-<yyyymmdd>.zip
```

只收 `app/ configs/ docs/ tests/` + 根目录说明/依赖文件，**不含** `data/ out/ work/`、模型、
`.git/`、密钥；实测 89 个文件、压缩后约 **4.5 MiB**。包内附
[docs/交付说明.md](docs/交付说明.md)（功能清单、用法、打包边界、验收清单）。

## 目录结构

```
app/
  cli.py                命令行入口（config/scan/status/decode/…/run/batch/retry/serve）
  server.py             FastAPI 应用（/api/*，本地 Web UI 后端）
  config.py             配置加载 / 预设 / 校验
  storage.py            SQLite jobs/stages 状态机（断点续跑的数据基础）
  models.py             数据模型与常量（阶段顺序、状态、id 生成）
  pipeline/             decode / analyze / denoise / trim / loudness / asr / diarize /
                        correct / compose / export / orchestrator
  web/                  原生前端（index.html / app.js / style.css，无 build）
  tools/                测速、回归、MP4 修复等辅助脚本
configs/                default.yaml、presets/、hotwords/、prompts/
data/                   原始 m4a（只读，不入库）
out/                    产物根目录：<job_id>/ 八件文件、jobs.db、uploads/、logs/
work/                   虚拟环境、缓存（16k/analyze）、模型、证据脚本（可随时删）
docs/                   调研报告、实施计划、工程日志、研究记录
tests/                  pytest（含 -m slow 真实音频回归）
dist/                   打包产物（package_release 生成，已 gitignore）
start_web.bat           双击启动 Web UI（UTF-8，127.0.0.1:7860）
```

## 文档

| 文件 | 说明 |
|---|---|
| [docs/实施计划.md](docs/实施计划.md) | P0–P8 任务卡、写作用域、验收命令、回归基线 |
| [docs/工程日志.md](docs/工程日志.md) | 各阶段的真实施工记录、实测数字、偏差与遗留 |
| [docs/运行评估.md](docs/运行评估.md) | 6 文件分阶段 RTF、500 h ≈64 h 推算、磁盘/电费/纠错成本、与 30 h 目标差距与优化建议 |
| [docs/使用说明.md](docs/使用说明.md) | 新用户 10 分钟跑通第一个文件（环境、快速路径、Web UI、常见问题） |
| [docs/交付说明.md](docs/交付说明.md) | **可打包功能清单、三种典型用法、产物/成本、打包边界与验收清单** |
| [docs/全文成稿_方案与实测.md](docs/全文成稿_方案与实测.md) | P10 全文成稿的方法、实测、成本与边界 |
| [docs/推理强度调研与实测.md](docs/推理强度调研与实测.md) | P10.2 thinking/reasoning_effort 控制方式、小样本实测与推荐配置 |
| [docs/调研报告_需求对齐版.md](docs/调研报告_需求对齐版.md) | 主报告：需求实测、批量架构、开发成本、验收标准 |
| [docs/开发启动Prompt_P6P7.md](docs/开发启动Prompt_P6P7.md) | P6/P7 的启动 prompt（含 API 契约与已知坑） |
| [docs/research/](docs/research/) | 多文件体检、PoC、ASR/降噪基准、说话人、云/本地选型 |
| [CHANGELOG.md](CHANGELOG.md) | v0.2.0（P0–P10）按阶段变更记录与关键数字 |

## 仓库说明

- git 只提交代码 / 文档 / 小型中间 JSON；`data/`、`work/` 下的 wav/模型、`out/` 下的音频与
  db/logs/uploads 已在 `.gitignore` 排除。
- 样例音频（6 个 m4a，576 min）与模型不在仓库里；新机器需重新放回 `data/`、
  `work/bench/models/`、`work/diar/models/`（或设 `MODEL_DIR`）。
- 本机无 `DEEPSEEK_API_KEY` 时各命令照常跑完，只是纠错阶段为 `enabled=false`。
