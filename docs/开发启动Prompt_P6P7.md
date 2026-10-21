# P6+P7 启动 Prompt（可直接复制给新 agent 会话）

> 前置状态：P0–P5 已完成并提交；6 个样例的 `clean.m4a / transcript.json / transcript.md / transcript.srt / transcript.txt / corrections.json / report.json` 已产出；`pytest tests/` 360 passed、`pytest tests/ -m slow` 3 passed。
> 使用方式：在项目根目录 `C:\projects\local\audio_processer` 新开 agent 会话，把下面代码块整段作为第一条消息发送。

```text
你是「audio_processer」项目的开发 agent，继续执行 P6（编排器 / 断点续跑 / CLI）和 P7（本地 Web UI）。
工作目录：C:\projects\local\audio_processer（Windows 10，pwsh 可用）。P0–P5 已完成并提交，本轮只做 P6+P7；P8（回归/500h 推算/交付文档）不在本轮。

## 0. 本轮目标（一句话）
把已有的 decode→analyze→denoise→trim→loudness→asr→diarize→correct→export 串成可断点续跑、失败隔离的编排器与 batch/run/retry/serve CLI；
再做一个本地 FastAPI + 原生 JS 的 Web UI，能完成「扫描/上传 → 跑 → 播放 → 改字 → 导出」全流程。

## 1. 开工前必读（按顺序，读了再动手）
1) docs/实施计划.md —— §2 技术决策、§4.1/§4.4/§4.5 契约、§5 的 P6-T1~P6-T3 与 P7-T1~P7-T3、§7 已知坑、§8 运行手册、§10 DoD。
2) docs/工程日志.md —— 重点读 P3「开工第一件事」、P4/P5 整节（六文件 ASR/diar/measure/纠错/导出的真实数字与遗留）；本轮 P6/P7 的验收基准与已知坑都在这里。
3) docs/调研报告_需求对齐版.md §9（批量架构/并发/失败处理）与 §2.4（说话人）；docs/research/timeline-poc.md 的产物格式。
4) 现有实现（直接复用，不要重写）：
   - app/storage.py：jobs/stages 状态机、start_stage/finish_stage/fail_stage、is_stage_done、next_incomplete_stage、mark_interrupted_pending、begin_attempt、finish_job/fail_job、resume_candidates；
   - app/pipeline/export.py：render_job（P1–P3 全链路 + 阶段回调）、export_text（P5-T4 文本导出）、write_json/_clean_json；
   - app/pipeline/asr.py：transcribe_job（clean16k/分段/双轴/report 读改写）；
   - app/pipeline/diarize.py：diarize_job（默认 none，0.01s；sampled/full 可选）；
   - app/pipeline/correct.py：correct_job（无 key 自动 enabled=false，不阻塞）；
   - app/cli.py：现有 config/scan/status/decode/analyze/render/transcribe/correct/export 子命令的写法；
   - app/config.py：load_config / cfg.get / cfg.section / cfg.path / cfg.replace / validate_config。
5) README.md —— 当前仍是「等待施工」状态；P7-T3 要更新为可运行状态与 Web 使用说明。

## 2. 当前仓库状态（不要重复实现）
- 最新提交（P4/P5）：
  - dc1d586 feat(P4): segment ASR with clean/orig timeline
  - a831b64 feat(P4): diarization (default none) + timestamp measure
  - 60953a7 feat(P5): DeepSeek correction with chunk validation and fallback
  - 80fb3c2 feat(P5): md/srt/txt export + transcribe/correct/export CLI
  - 5eb63dc docs: P4/P5 工程日志
  - cf594a8 chore: ignore pytest basetemp and mock temp dirs
- 测试基线：`pytest tests/` = 360 passed；`pytest tests/ -m slow` = 3 passed（439.52 s）。不得回归。
- CLI 现有：config show|check、scan、status、decode、analyze、render、transcribe、correct、export。
- DB：`out/jobs.db` 6 个 job，全部 status=done、stage=export，0 pending。
- 6 个样例（data/ 只读，文件名含中文/空格/括号）：

  | key | job id | clean s |
  |---|---|---|
  | f00 | 621cb91c_0_20140227_丁耘_西方哲学 | 2745.28 |
  | f04 | 9f3feb49_4_丁耘理想国_古希腊哲学_第四卷 | 8131.83 |
  | f05 | 0a7fa0ed_复旦哲学系丁耘_亚里士多德_物理学_05 | 6201.13 |
  | f06 | b00e197d_复旦哲学系丁耘_孔子_论语_06_19 | 5327.85 |
  | f20 | 4db18037_20_丁耘理想国_古希腊哲学 | 5869.60 |
  | f24 | 7927721f_24_复旦哲学系丁耘_亚里士多德 | 3806.46 |

- 每个 out/<id>/ 已有 8 件文件：clean.m4a、edit.json、report.json、transcript.json、transcript.md、transcript.srt、transcript.txt、corrections.json（五件套 + edit/report/corrections）。
  - f24：1175 段 / 15,524 字符 / n_empty=0；六文件合计 9,180 段 / 140,046 字符。
  - report.json 已有字段：id/source/duration_s/sample_rate/bit_rate/decode/vad/levels/trim/denoise/output/timing/loudness/ok/asr/speakers/correct。
  - transcript.json 顶层是 list（§4.3），每项 start/end/orig_start/orig_end/speaker/text_raw/text/flags。
  - corrections.json 契约：{total, items:[{orig,fixed,reason,chunk}], rejected_chunks:[...]}；无 key 时是空结构。
- work/cache/：6 个 `<id>.16k.wav` + `.decode.json` + `.analyze.json` + 新增的 `.clean16k.wav`（约 2.05 GB，gitignore）。**不要删/改** 16k/analyze/decode 缓存。
- P4/P5 证据（可读，不要改）：work/transcribe_f24.json、work/measure_f24.json、work/diar_p4_check.json、work/regress/audio.csv、work/timeline/*、work/cost/correct_test/*。
- P6/P7 缺口：app/pipeline/orchestrator.py、app/server.py、app/web/、start_web.bat、tests/test_orchestrator.py、tests/test_server.py 均不存在；README 未更新。
- 环境事实：
  - Python 用 work\venv\Scripts\python（3.10）；ffmpeg 7.1 在 PATH；Node v22（可 `node --check`）；Chrome 在 `C:\Program Files\Google\Chrome\Application\chrome.exe`（可 headless 截图）。
  - requirements.txt 已装 fastapi==0.156.4、uvicorn==0.54.0；**没有** python-multipart、httpx、aiofiles、sse-starlette。**本轮优先不新增依赖**：上传走 `application/octet-stream` 裸流 + `?filename=`，API 测试用标准库 urllib 对真实 uvicorn 实例；若坚持用 multipart/TestClient，才安装 python-multipart/httpx 并固定版本、写进 requirements.txt，且说明理由。
  - 不要用 requests/httpx 改 P5 的 LLM 客户端（P5 固定 urllib）。
  - 设 `$env:PYTHONIOENCODING='utf-8'`；pip 安装必须 `$env:PIP_CACHE_DIR="$PWD\work\pip_cache"`；pwsh 若全命令 0xC0000142，切完全权限文件策略后重试。
  - 源码/文档只用 read/write/edit 工具改，禁止 PowerShell `Get-Content | Set-Content` 改中文源码。

## 3. 本轮范围与边界
- 做 P6-T1/T2/T3、P7-T1/T2/T3；不做 P8。
- **不要重跑 P3/P4/P5 的六文件产物**；不要改 clean.m4a / 既有 report 字段（只允许读-改-写追加 timing/stage 类字段）；不要改 data/、docs/research/、work/ 下 P1–P5 证据。
- 新人声/上传文件写到 `out/uploads/`（新建，加入 .gitignore），**不要写 data/**；P7 的 run 测试也优先用独立 `out_p6/`、`work/cache_p6/`，验收后删除，不污染现有 out/。
- P6 默认单进程；ASR 仍单例 8 线程；`--jobs` 只保留参数，不做进程池。
- 安全：API 只 bind 127.0.0.1，无鉴权（本地工具）；所有 job id 必须来自 DB；产物/导出只允许已知文件名，禁止路径穿越；上传文件名去目录、按 `make_job_id` 生成 id。

## 4. 开工第一件事（必须先做，写进工程日志再继续）
1) 跑基线：`pytest tests/ -q`（360 passed）、`pytest tests/ -m slow`（3 passed）；`python -m app.cli status`（6 done/export）；确认六份 out/<id>/ 八件文件齐全。
2) 读 `out/jobs.db` 的 stages：注意 **P3 的 audio export 与 P5-T4 的 text export 共用 stage 名 `export`**，P5 已覆盖 P3 的 export metrics；P6 必须明确「export = 最终全部产物完成」的语义，不能只看最后一条 metrics。
3) 在 `docs/工程日志.md` 追加「P6/P7 开工基准与决定」：记录基线数字、export 阶段语义、上传目录、缓存清理策略、是否需要新依赖；定完再写代码。
4) 先决定并写进 docstring：
   - **stage 顺序**：decode → analyze → denoise → trim → loudness → asr → diarize → correct → export；export 在 audio+text+transcript+corrections 全部落盘后才算 done。
   - **resume 判定以产物为准**：八件文件齐全 + report.asr/correct 存在 → 直接 skipped；缺哪段补哪段，不能因为 stages 缺 diarize 就把 done 任务整条重跑。
   - **清理策略**：默认保留 `<id>.16k.wav`、`.analyze.json`、`.decode.json`（否则 resume 要重跑分析）；只删 `.clean16k.wav`、44.1k/denoise/normalized 等中间 wav；不得调用 decode.cleanup_cache 的默认清理。
   - **锁**：`work/locks/<job_id>.lock`，Windows 用 msvcrt.locking 或 O_EXCL+PID 兜底；同一 job 不能被 CLI 与 Web 同时跑。

## 5. 任务卡

### P6-T1 编排器 —— app/pipeline/orchestrator.py + tests/test_orchestrator.py
对外接口固定（CLI 与 server 都会调用）：
- `run_job(job_id, source, cfg, *, force=False, out_dir=None, cache_dir=None, on_event=None) -> dict`
- `batch_run(inputs, cfg, *, resume=True, limit=None, force=False, on_event=None) -> dict`
- 失败不抛，返回 `{ok, job_id, stage, error, skipped, timings, report}`；`on_event(stage, status, metrics)` 风格同 render_job。

实现要点：
1. 启动时 `store.mark_interrupted_pending()`；`begin_attempt`；磁盘水位检查（`shutil.disk_usage(out_dir).free < batch.disk_free_min_gb*1e9` → 停止取新任务并返回 `stopped_reason="disk_low"`）。
2. 复用 render_job 跑 P1–P3（它会 decode→analyze→denoise→trim→loudness→audio export，并回调阶段）。编排器包一层 on_stage：
   - decode/analyze/denoise/trim/loudness 直接落 stages；
   - render_job 的 `export` 回调先存到 `audio_export_metrics`，**不单独落库**；
   - 全部完成后统一落一条 `export`，metrics 合并 audio（tp/asl/size）+ text（md/srt/txt）+ transcript/corrections 计数。
3. 之后依次：`transcribe_job`（已有 transcript 会 skip）→ `diarize_job`（默认 none，0.01s；mode!=none 才 full）→ `correct_job`（无 key enabled=false 也要落 stages）→ `export_text`（force=False 可 skip，但缺文件要补）。
4. 完成判定：out/<id>/ 八件文件齐全且 report 有 asr/correct/speakers；否则 fail_job + 明确错误。成功 finish_job(done, stage="export")。
5. 失败隔离：单文件异常→fail_stage/fail_job，继续下一个；batch 结束返回 failures 列表。decode 修不好→复制到 `out/quarantine/<id>.<ext>`（不删 data/），job 记 failed。
6. 已有 6 个 done 任务：`batch data/ --out out --resume` 必须全部 skipped（不能重新跑 ASR/render），并打印 skipped 标记。
7. 单测用 monkeypatch 假 stage 覆盖：顺序、resume skip、失败继续、磁盘水位、锁、清理不删 analyze、quarantine；不要用真实 100 min 音频做单测。

### P6-T2 CLI —— app/cli.py（Lead 写；teammates 提需求）
新增（风格同现有子命令，argparse + CJK 表格 + `--json`）：
- `python -m app.cli run <id|path> [--out] [--cache-dir] [--force] [--resume] [--json]`
- `python -m app.cli batch <dir> --out out/ --resume [--limit N] [--force] [--jobs 1] [--json]`
- `python -m app.cli retry <id> [--json]`（reset failed/downstream stages 后重跑）
- `python -m app.cli status [--watch]`（--watch 每 2 s 刷新，Ctrl-C 退出）
- `python -m app.cli serve --port 7860`（uvicorn.run(app.server.app, host="127.0.0.1")）
- `batch` 内部先 discover/upsert（复用 scan 的 discover_files），再走 batch_run；`--resume` 跳过 done，`--force` 重跑 done；失败退出码 1。
- `run` 对未登记的 path 先 upsert job；`--resume` 从最后未完成 stage 继续。
- `serve` 启动后打印 `http://127.0.0.1:7860`（stdout/stderr 均可，UTF-8）。

### P6-T3 失败处理与清理
- 磁盘水位、quarantine、单文件失败隔离、attempts 上限（config batch.max_attempts）。
- 清理：成功后删 `.clean16k.wav` 与 render 残留（`.clean44k/.normalized44k/.denoised44k/.fixed.m4a`）；保留 16k/analyze/decode；`--keep-intermediates` 可保留。
- 日志轮转已由 logging_setup 实现，不要重复造。
- 验收：磁盘水位单测；quarantine 单测；真实 kill/resume（见 §8）。

### P7-T1 API —— app/server.py + tests/test_server.py
FastAPI，`app = FastAPI()`；所有路由前缀 `/api`；静态资源 `/` `/static`；响应 JSON 用 ensure_ascii=False。
端点固定：
- `GET /api/health` → `{"ok": true, "version": ...}`
- `GET /api/jobs` → `{"summary": {...}, "jobs": [...]}`（复用 Storage.summary/jobs_snapshot）
- `POST /api/jobs`：
  - JSON `{"path": "data/"}` → 扫描目录登记，返回 `{created, existing, jobs:[...]}`；路径必须存在。
  - `application/octet-stream` + `?filename=xxx.m4a` → 裸流上传到 `out/uploads/<sanitized>`，登记 job，返回 `{job_id, source}`；限制合理大小（例如 2 GB），文件名取 basename、去危险字符。
- `POST /api/jobs/{id}/run` → 启动后台线程跑 `run_job`（单 worker + per-job 锁；已在跑返回 409）；返回 202。
- `POST /api/jobs/{id}/retry` → reset failed/downstream 后启动 run；返回 202。
- `GET /api/jobs/{id}` → `{job, report, stages, artifacts, transcript, corrections}`（transcript 最多几千条，直接返回；大文件不要内嵌）。
- `GET /api/jobs/{id}/artifacts` → `[{name, size, mtime}]`（只列已知 7 件 + uploads 不列）。
- `GET /api/jobs/{id}/audio` → clean.m4a，支持 HTTP Range（Starlette FileResponse 原生；用 `Range: bytes=0-1023` 验证 206）。
- `GET /api/events` → SSE：`text/event-stream`，每 1–2 s 轮询 DB 变化推送 `event: snapshot` + JSON，15 s 心跳 `: ping`；不要阻塞事件循环（DB 查询用 `asyncio.to_thread`）。
- `POST /api/jobs/{id}/transcript` → body `{"segments":[{"index":0,"text":"..."}]}` 或完整 list；只改 text，保留 text_raw/start/end/orig/speaker/flags；原子写 transcript.json，然后调 `export_text(..., force=True)` 重生成 md/srt/txt；返回更新后的产物列表。跑任务中返回 409。
- `GET /api/jobs/{id}/export?format=md|srt|txt` → FileResponse，UTF-8；未知格式 400。
- 静态：`GET /` → app/web/index.html；`/static/app.js`、`/static/style.css`；不要引入前端构建链。
- 后台线程用独立 Storage 连接（SQLite WAL），不要跨线程复用同一个连接；per-job 锁防重复。

### P7-T2 前端 —— app/web/index.html / app.js / style.css
原生 HTML/CSS/JS，无框架、无 build。功能：
- 左侧任务列表：id 短名、状态色、stage、耗时、错误摘要；顶部汇总（pending/running/done/failed）；点击选中。
- 顶部：选择/输入本地目录扫描（POST JSON path）、上传文件（fetch 裸流）、刷新。
- 详情：音频播放器 `<audio controls src="/api/jobs/<id>/audio">`；转写表格（时间/发言人/原文/修正），2299 行也要能打开（分页或虚拟滚动）；点击时间跳转播放。
- 改字：每行 text 可编辑，保存按钮 POST transcript；成功后刷新 md/srt 状态；text_raw 只读展示。
- 操作：开始/重试、导出 md/srt（下载）、SSE 实时进度（EventSource /api/events）。
- 空态/错误态/加载态；中文界面；无 JS 报错；DOM 用 textContent 防 XSS，不要 innerHTML 拼用户文本。

### P7-T3 启动脚本与 README
- `start_web.bat`（UTF-8）：cd 到脚本目录、设 PYTHONIOENCODING=utf-8、调 `work\venv\Scripts\python -m app.cli serve --port 7860`，打印 URL，pause。
- README 更新为「已可运行」：安装、scan/batch/run/serve 命令、Web 使用、上传目录、无 key 纠错降级、已知限制、测试命令。
- 不要求自动打开浏览器；双击 bat 能看到 URL 即可。

## 6. 执行方式（建议 Agent Teams；单人则按 P6→P7 顺序）
写作用域必须互不重叠；app/cli.py 与 configs/default.yaml 只由 Lead 改；app/storage.py 需要小 helper 也由 Lead 改。
- `orchestrator-dev`：app/pipeline/orchestrator.py、tests/test_orchestrator.py（P6-T1/T3，关键路径）。
- `server-dev`：app/server.py、tests/test_server.py（P7-T1，按固定 API 契约）。
- `web-dev`：app/web/**、start_web.bat、README.md（P7-T2/T3，按固定 API 契约并行）。
- Lead：app/cli.py、configs/default.yaml、app/storage.py（如需）、集成/真实 kill-resume/六文件 skip、docs/工程日志.md、最终全量测试、分提交。
依赖：orchestrator 的 `run_job/batch_run` 接口先定；server 可先按接口写，缺实现时在路由内 import；web 只依赖 API 契约，可与 server 并行。
并行约定：pytest 各自 `--basetemp=work/pytest_tmp_<名字>`；不要跑全量 pytest（Lead 最后统一跑）；git 只 add 自己文件、禁止 add -A；不要两个 agent 同时跑同一个 job；不要同时启动两个真实 pipeline。

## 7. 已锁定技术决策（照做，不改默认值）
- 编排：单进程；stage 顺序 decode→analyze→denoise→trim→loudness→asr→diarize→correct→export；export 最后落库。
- 断点续跑：产物优先；done 任务 batch --resume 全 skip；失败/中断从最后未完成 stage 继续；单文件失败不阻塞其它文件。
- 清理：保留 16k/analyze/decode；默认删除 clean16k 与 44.1k/denoise/normalized 中间 wav；不删 analyze 缓存。
- Web：FastAPI + uvicorn + 原生 JS；127.0.0.1:7860；无鉴权；裸流上传；SSE 轮询 DB；不引入前端构建/额外 HTTP 客户端。
- 人工改字：只改 text，保留 text_raw/时间/说话人；保存后强制重导出 md/srt/txt。
- 报告：report.json 只读-改-写，保留 P3/P4/P5 既有字段；追加 stage 相关 timing 可以，但不得覆盖 asr/speakers/correct 语义。

## 8. 验收（必须真跑，不要只写测试）
1) 测试：`pytest tests/` 全绿（原 360 项不回归 + 新增 orchestrator/server 测试）；`pytest tests/ -m slow` 3 passed。
2) 六文件 skip：`python -m app.cli batch data/ --out out --resume --limit 6` → 6/6 skipped，0 次 ASR/render 重跑（对照 stages/日志）。
3) fresh 全链路：把 f24 跑到独立目录：
   `python -m app.cli run data\24_...m4a --out out_p6 --cache-dir work\cache_p6 --json`
   → 八件文件齐全；report 有 asr/speakers/correct；总耗时记录。
4) 断点续跑：fresh run 中途 kill（在 denoise/asr 阶段），再 `run <id> --out out_p6 --cache-dir work\cache_p6 --resume`；已完成 stage 不重跑，最终产物齐全。若真实 kill 太慢，至少用假 stage 单测覆盖 + 真跑记录说明。
5) 磁盘水位：cfg.replace({"batch.disk_free_min_gb": 1e9}) 调 batch_run → stopped_reason=disk_low，不启动新 job（单测）。
6) 失败隔离/quarantine：构造坏 .m4a → 单 job failed、其它继续、quarantine 有副本；单测。
7) Web API：`python -m app.cli serve --port 7860` 启动后，用标准库 urllib 对真实实例测试：
   - GET /api/health、/api/jobs、/api/jobs/<f24>、/artifacts 200；
   - GET /audio with `Range: bytes=0-1023` → 206；
   - GET /api/events 能读到首个 snapshot 事件（读几行后关闭）；
   - POST /api/jobs/<f24>/transcript 改一段 text → 200，transcript.json/md/srt 已更新；
   - GET /export?format=md → 200 且内容含转写；未知格式 400；
   - 上传一个 60–120 s 的 f24 切片（ffmpeg 生成到 work/，不写 data/）→ POST /api/jobs 裸流 → run → 轮询到 done → 八件文件齐全。
8) 前端：`node --check app/web/app.js`；`GET /`、`/static/app.js` 200；Chrome headless 截图并 read_image 检查：
   `"C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --disable-gpu --screenshot=work/web_home.png --window-size=1440,900 http://127.0.0.1:7860`
   截图应能看到任务列表/详情/播放器，无空白页。若无法交互测试，明确写「截图 + API 已验，人工点击待用户」。
9) `start_web.bat` 内容正确，能启动并打印 URL（可用 `cmd /c` + 超时/手动确认）。
10) docs/工程日志.md 追加 P6/P7 一节（做了什么/验收真实数字/偏差/遗留）；代码/测试/日志分提交，消息风格同 P0–P5。

## 9. 已知坑（务必规避）
- **export 阶段名复用**：P3 audio export 与 P5 text export 都叫 export，P5 已覆盖 metrics；P6 必须自己合并/最后落库，不要依赖 stages.export 的旧 metrics 判断。
- **render_job 的 skip**：它按 clean.m4a+report.json 判定 skip，会回调所有 stage；orchestrator 要识别 skipped 并继续 ASR/correct/text export，不要误判整个 job 完成。
- **render_job 会设置 job? 不，它只回调 on_stage**；job.status/stage 由 CLI/orchestrator 负责。不要重复 finish_job。
- **不要删 analyze 缓存**：删除 16k 会让 analyze 缓存失效（缓存校验 wav bytes），resume 变慢；clean16k 可删。
- **SQLite 跨线程**：每个线程/请求新建 Storage，WAL；不要共享连接。
- **SSE 阻塞**：DB 查询用 to_thread，`await asyncio.sleep`；加 heartbeat，设 Cache-Control: no-cache。
- **上传**：不写 data/；文件名 basename + make_job_id；限制大小；裸流读取用 `request.stream()` 落盘，不要一次性读 2 GB 进内存。
- **Range**：用 FileResponse 并实际 curl/urllib 验 206；不要手写只支持 200 的实现。
- **前端**：2299 行表格要分页/虚拟滚动；用 textContent 防 XSS；不要 innerHTML 拼用户文本。
- **Windows 路径/中文**：全程 Path + 列表参数；UTF-8；不要 PowerShell 改源码。
- **依赖**：没有 python-multipart/httpx；优先裸流 + urllib 测试，避免新增。若必须装：`$env:PIP_CACHE_DIR="$PWD\work\pip_cache"` + 清华源，并把版本 pin 进 requirements.txt。
- **测试污染**：pytest 一律 --basetemp=work/pytest_tmp_*；临时 out 用 out_p6/out_p7，不要写 out/ 现有 6 个 job；完成后清理。
- **不要重跑六文件**：P3 的 f24 clean 是 3806.46 s；P4 段表 1175；P5 无 key 已写空 corrections；这些是已验证产物。
- **Chrome 截图**：headless 需要页面无 JS 报错；截图路径放 work/，不要提交。
- **pwsh 0xC0000142**：切完全权限文件策略；pytest 带独立 basetemp；git 只 add 具体文件。

## 10. 交接资料
- 六文件 job id/clean 秒：见 §2 表；`work/regress/audio.csv` 是 P3 回归。
- P4：work/transcribe_f24.json、`app/pipeline/asr.py`；P4 时间戳测量 `work/measure_f24.json`。
- P4-T3：work/diar_p4_check.json（正文抽样 f24/f00 均 n=1）。
- P5：work/cost/correct_test/（金标）、`app/pipeline/correct.py`、`configs/prompts/correct_system.txt`、`configs/hotwords/`。
- 现有产物样例：out/<id>/report.json 的字段顺序/内容可直接参考；f24 transcript 1175 条。
- 关键命令：`python -m app.cli --help`、`python -m app.cli status --json`、`python -m app.cli render --help`、`python -m app.cli transcribe --help`、`python -m app.cli correct --help`、`python -m app.cli export --help`。

## 11. 完成标准（本轮 DoD）
1) pytest 全绿：原 360 + 新增 orchestrator/server 测试；slow 3 不回归。
2) `batch data/ --out out --resume` → 6/6 skipped，0 重跑；f24 fresh run 到独立 out 目录 → 八件文件齐全。
3) 真实 kill 后 `--resume` 不重复已完成 stage；磁盘水位/quarantine/失败隔离有单测或真跑证据。
4) Web API 全端点通过（health/jobs/detail/artifacts/audio Range/SSE/transcript edit/export/upload+run）；上传 60–120 s 样本能跑完并播放。
5) Chrome headless 截图可见任务列表/详情/播放器；`node --check` 通过；无 JS 报错。
6) `start_web.bat` 双击可启动并打印 `http://127.0.0.1:7860`；README 已更新。
7) docs/工程日志.md 追加 P6/P7 真实数字/偏差/遗留；代码/测试/日志分开 commit。
```

---

### 交接给下一轮 agent 的提醒（不属于 prompt 正文）
- 本轮 P6/P7 最容易翻车的是「export 阶段名复用」和「render_job 的 skip 回调」；先读 `app/cli.py` 的 `_stage_recorder` 与 `app/pipeline/export.py:render_job` 再动手。
- 上传/API 测试优先用真实 uvicorn + 标准库 urllib，避免为了 TestClient 引入 httpx；裸流上传可完全绕开 python-multipart。
- 如果时间紧，优先保证 P6 的 resume/failure isolation + P7 的 API/播放/改字/导出；前端视觉可先用最简 CSS，再补截图验收。
