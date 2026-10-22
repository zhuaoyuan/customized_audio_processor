# CHANGELOG

本项目遵循「按阶段里程碑」记录（非严格 SemVer 发布节奏）；当前唯一版本 **v0.1.0** = P0–P8 全部完成。

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
  成本口径：无 key ¥0，有 key 估算 f24 ≈¥0.112/文件 → 300 文件 ≈¥33.6（真实 API 未验证）。

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
- **真实 `DEEPSEEK_API_KEY` 路径未验证**：仅 mock 金标 + mock E2E；真实计费以账单为准；
- **`work/cache` 的 16k wav 全量保留会到 ≈107 GiB/500 h**，需按任务清理；「≤30 GB」是导出产物的口径；
- Web UI 未做移动端深度适配、Safari 未验证；人工「播放→改字→保存→导出」全流程建议再点一遍；
- `batch` 的 `--jobs` 只支持 1（进程池并行实测慢 22%）。
