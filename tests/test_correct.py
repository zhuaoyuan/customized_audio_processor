# -*- coding: utf-8 -*-
"""P5-T1/T2/T3 纠错阶段单测：HTTP mock、分块、校验回退、金标验收。

原则
----
- **绝不访问真实 API**：所有 HTTP 用例都打到本进程用 ``http.server`` 起的 mock OpenAI
  兼容服务（``POST /v1/chat/completions``），服务用完即关；
- mock 返回内容视为外部不可信数据，只走文本校验路径；
- 不依赖 P4 真实 transcript：全部用合成 transcript fixture 写进 ``tmp_path``。
"""

from __future__ import annotations

import difflib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from app.models import FLAG_REJECTED_CHUNK, FLAG_REJECTED_SEGMENT
from app.pipeline.correct import (
    Chunk,
    CorrectError,
    _normalize_usage,
    _num_runs,
    _salvage_segments,
    _seg_guard,
    chat_completion,
    compute_cost,
    correct_job,
    correct_text,
    diff_edits,
    load_hotwords,
    load_system_prompt,
    map_fixed_to_segments,
    plan_chunks,
    split_sentences,
    validate_block,
    validate_correction,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "work" / "cost" / "correct_test"


# ================================================================ mock server
def chat_response(
    content: str,
    *,
    prompt_tokens: int = 4000,
    completion_tokens: int = 3000,
    cache_hit_tokens: int = 0,
    model: str = "deepseek-flash",
) -> Dict[str, Any]:
    return {
        "id": "mock-completion",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_cache_hit_tokens": cache_hit_tokens,
        },
    }


def ok(content: str, **kwargs: Any) -> Tuple[int, Dict[str, Any]]:
    return 200, chat_response(content, **kwargs)


class _MockState:
    def __init__(self, responses: Sequence[Any]) -> None:
        self.responses = list(responses)
        self.requests: List[Dict[str, Any]] = []
        self.lock = threading.Lock()


def _make_handler(state: _MockState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self) -> None:  # noqa: N802 - http.server 协议
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except ValueError:
                body = {}
            with state.lock:
                state.requests.append(
                    {"path": self.path, "body": body, "headers": dict(self.headers)}
                )
                idx = min(len(state.requests) - 1, max(0, len(state.responses) - 1))
                response = state.responses[idx] if state.responses else (500, {"error": "no mock"})
            if callable(response):
                response = response(body)
            status, payload = response
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8") \
                if isinstance(payload, (dict, list)) else str(payload).encode("utf-8")
            try:
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):  # 客户端超时先断开
                pass

        def log_message(self, fmt: str, *args: Any) -> None:  # 静音
            return

    return Handler


class MockServer:
    """本地 OpenAI 兼容 mock；作为上下文管理器使用，退出时 shutdown + close。"""

    def __init__(self, responses: Sequence[Any]) -> None:
        self.state = _MockState(responses)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self.state))
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    def __enter__(self) -> "MockServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> List[Dict[str, Any]]:
        return self.state.requests


@pytest.fixture()
def make_cfg(cfg):
    """在 conftest 的临时路径 cfg 上叠加 mock 配置。"""

    def _make(server: MockServer, **overrides: Any):
        values: Dict[str, Any] = {
            "correct.api_key": "mock-test-key",
            "correct.base_url": server.base_url,
            "correct.retry_backoff_s": 0.01,
            "correct.concurrency": 2,
        }
        values.update(overrides)
        return cfg.replace(values)

    return _make


# ================================================================ transcript fixture
def seg(
    text: str,
    *,
    start: float = 0.0,
    end: float = 1.0,
    speaker: str = "主讲",
    text_raw: Optional[str] = None,
    flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "orig_start": start,
        "orig_end": end,
        "speaker": speaker,
        "text_raw": text_raw if text_raw is not None else text,
        "text": text,
        "flags": list(flags or []),
    }


def write_job(tmp_out: Path, job_id: str, segments: Sequence[Dict[str, Any]],
              report: Optional[Dict[str, Any]] = None) -> Path:
    job_out = tmp_out / job_id
    job_out.mkdir(parents=True, exist_ok=True)
    (job_out / "transcript.json").write_text(
        json.dumps(list(segments), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report is not None:
        (job_out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return job_out


def _sized_segment(i: int, n: int = 100) -> Dict[str, Any]:
    """长度恰为 n、以句号结尾的确定性段文本。"""
    prefix = f"第{i:02d}段"
    text = prefix + "字" * max(0, n - len(prefix) - 1) + "。"
    return seg(text, start=float(i), end=float(i) + 1.0)


def _chunkable_segment(i: int, n: int = 100) -> Dict[str, Any]:
    """长度恰为 n、含一个刻意错字「做」的段文本（多块并发用例）。

    正文用按位置变化的汉字（不是同一个字重复），避免给 SequenceMatcher 造出病态输入。
    """
    prefix = f"第{i:02d}段"
    body_len = max(0, n - len(prefix) - 2)
    body = "".join(
        chr(0x4E00 + (k * 37 + 11 + i * 97) % 2000) for k in range(body_len)
    ).replace("做", "学")  # 保证「做」只出现在段末刻意错字处
    return seg(prefix + body + "做" + "。", start=float(i), end=float(i) + 1.0)


def _out_dir(cfg: Any) -> Path:
    return Path(cfg.path("out_dir"))


# ================================================================ 文本工具 / 分块
def test_split_sentences():
    assert split_sentences("甲。乙！丙？丁；戊") == ["甲。", "乙！", "丙？", "丁；", "戊"]
    assert split_sentences("他说：“好。”然后走了。") == ["他说：“好。”", "然后走了。"]
    assert split_sentences("……") == ["……"]
    assert split_sentences("") == []
    assert split_sentences("没有句号") == ["没有句号"]


def test_plan_chunks_deterministic_and_segment_aligned():
    segments = [_sized_segment(i) for i in range(40)]  # 40 × 100 = 4000 字符
    chunks_a = plan_chunks(segments, cfg=None)
    chunks_b = plan_chunks(segments, cfg=None)
    assert [c.text for c in chunks_a] == [c.text for c in chunks_b]
    assert [c.content_hash for c in chunks_a] == [c.content_hash for c in chunks_b]
    assert len(chunks_a) == 2
    # 每块 2000–2500，两块的段不重叠、覆盖全部段、顺序一致
    covered: List[int] = []
    for chunk in chunks_a:
        assert 2000 <= chunk.char_count <= 2500
        assert chunk.text == "\n".join(chunk.segment_texts)
        assert chunk.sentence_counts == [1] * len(chunk.segment_texts)
        covered.extend(chunk.segment_indices)
    assert covered == list(range(40))
    assert [c.index for c in chunks_a] == [1, 2]


def test_plan_chunks_never_splits_a_segment():
    long_text = "长" * 2999 + "。"
    chunks = plan_chunks([seg(long_text)], cfg=None)
    assert len(chunks) == 1
    assert chunks[0].segment_indices == [0]
    assert chunks[0].segment_texts == [long_text]
    assert chunks[0].char_count > 2500  # 单段异常长：允许超上限，但不切段


def test_plan_chunks_tail_rebalance():
    segments = [_sized_segment(i) for i in range(40)]
    chunks = plan_chunks(segments, cfg=None)
    assert [c.char_count for c in chunks] == [2000, 2000]
    # 尾部回填后仍保持段完整：第一块末尾段与第二块首段是相邻的不同段
    assert chunks[0].segment_indices[-1] + 1 == chunks[1].segment_indices[0]


def test_plan_chunks_empty():
    assert plan_chunks([seg(""), seg("  ")], cfg=None) == []


# ================================================================ 校验
def test_validate_normal_and_gold_ratio():
    orig = "这是测试文本。数字2020保持不变，《书名》也在，共3页。"
    fixed = "这是测试文本。数字2020保持不变，《书名》也在，共3页。"
    detail = validate_correction(orig, fixed, cfg=None)
    assert detail["ok"] is True
    assert detail["diff_ratio"] == 0.0

    # 小型同音修正（只改 1 个字）通过
    detail = validate_correction("这个教研室就肯定是做不下来。",
                                 "这个教研室就肯定是坐不下来。", cfg=None)
    assert detail["ok"] is True
    assert 0 < detail["diff_ratio"] <= 0.12


def test_validate_rejects_oversized_diff():
    orig = "这是测试文本。数字2020保持不变，《书名》也在。"
    fixed = "这是一段完全不相干的话，和原文没有任何关系。"
    detail = validate_correction(orig, fixed, cfg=None)
    assert detail["ok"] is False
    assert detail["reason"] == "diff_too_large"


def test_validate_rejects_number_changes():
    orig = "版本2020年的第3页。《书名》不变。"
    detail = validate_correction(orig, "版本202年的第3页。《书名》不变。", cfg=None)
    assert detail["ok"] is False and detail["reason"] == "number_changed"
    # 数字个数不变但数值被改，也要拒绝（防"页码被改写"）
    detail = validate_correction(orig, "版本2021年的第3页。《书名》不变。", cfg=None)
    assert detail["ok"] is False and detail["reason"] == "number_changed"


def test_validate_rejects_book_title_changes():
    # 用较长文本把 diff 率压在 12% 以内，确保命中"书名号数量变化"而不是 diff 超限
    orig = "参考《形而上学》与《工具论》两本书中的相关论述，并对照吴寿彭与李真的两个中译本仔细核对。"
    fixed = orig.replace("《", "").replace("》", "")
    detail = validate_correction(orig, fixed, cfg=None)
    assert detail["ok"] is False and detail["reason"] == "book_title_changed"


def test_validate_rejects_sentence_count_changes():
    orig = "".join("这是第%d句。" % i for i in range(20))
    fixed = "".join("这是第%d句，" % i for i in range(5)) + \
        "".join("这是第%d句。" % i for i in range(5, 20))
    detail = validate_correction(orig, fixed, cfg=None)
    assert detail["ok"] is False
    assert detail["reason"] == "sentence_count_changed"


def test_validate_rejects_markdown_wrapper():
    detail = validate_correction("这是测试文本。", "```\n这是测试文本。\n```", cfg=None)
    assert detail["ok"] is False and detail["reason"] == "markdown_wrapper"


def test_validate_rejects_empty_output():
    detail = validate_correction("这是测试文本。", "   ", cfg=None)
    assert detail["ok"] is False and detail["reason"] == "empty_output"


# ================================================================ 写回映射
def test_map_fixed_to_segments_line_mode():
    chunk = Chunk(
        index=1, text="甲。乙。\n丙。丁。",
        segment_indices=[0, 1], segment_texts=["甲。乙。", "丙。丁。"],
        sentence_counts=[2, 2], char_count=8, content_hash="x",
    )
    assert map_fixed_to_segments(chunk, "甲。乙。\n丙。丁。", cfg=None) == ["甲。乙。", "丙。丁。"]


def test_map_fixed_to_segments_sentence_fallback():
    chunk = Chunk(
        index=1, text="甲。乙。\n丙。丁。",
        segment_indices=[0, 1], segment_texts=["甲。乙。", "丙。丁。"],
        sentence_counts=[2, 2], char_count=8, content_hash="x",
    )
    # 模型丢了换行：按句数（2+2）猜回段
    assert map_fixed_to_segments(chunk, "甲。乙。丙。丁。", cfg=None) == ["甲。乙。", "丙。丁。"]


def test_map_fixed_to_segments_unsafe_returns_none():
    chunk = Chunk(
        index=1, text="甲。乙。\n丙。丁。",
        segment_indices=[0, 1], segment_texts=["甲。乙。", "丙。丁。"],
        sentence_counts=[2, 2], char_count=8, content_hash="x",
    )
    # 句数对不上 → 无法安全一对一回填
    assert map_fixed_to_segments(chunk, "甲。乙。丙。", cfg=None) is None
    # 段内容被整体搬串（每段 diff 超限）→ 拒绝
    assert map_fixed_to_segments(chunk, "丙。丁。\n甲。乙。", cfg=None) is None


# ================================================================ P9：真实模型输出的鲁棒写回
def test_num_runs_normalizes_fullwidth_and_thousands():
    """数字校验口径（P9）：全角→半角、去千位逗号；数值本身变化仍不等。"""
    assert _num_runs("２０２０年的第３页") == _num_runs("2020年的第3页")
    assert _num_runs("共1,234页内容") == _num_runs("共1234页内容")
    assert _num_runs("第2卷") != _num_runs("第3卷")


def test_validate_block_defers_numbers_and_books():
    """块级门不再因个别数字/书名号整块否决（下放到逐段护栏），但大改仍拒。"""
    orig = "版本2020年的第3页。《书名》不变。"
    changed_numbers = "版本2021年的第3页。《书名》不变。"
    block = validate_block(orig, changed_numbers, cfg=None)
    assert block["ok"] is True and block["diff_ratio"] <= 0.12

    wholesale = validate_block(orig, "这是一段完全不相干的话，和原文没有任何关系。", cfg=None)
    assert wholesale["ok"] is False and wholesale["reason"] == "diff_too_large"
    assert validate_block("这是测试文本。", "```\n这是测试文本。\n```", cfg=None)["reason"] \
        == "markdown_wrapper"


def test_map_fixed_to_segments_alignment_fallback():
    """行数、句数都对不上时，用字符级对齐兜底（P9）。"""
    chunk = Chunk(
        index=1,
        text="这是第一段文本内容。\n这是第二段文本内容。",
        segment_indices=[0, 1],
        segment_texts=["这是第一段文本内容。", "这是第二段文本内容。"],
        sentence_counts=[1, 1], char_count=22, content_hash="x",
    )
    # 模型把第一段句号改成逗号、整体并成一行：按行(1≠2)、按句(1≠2)都不行，对齐能恢复
    fixed = "这是第一段文本内容，这是第二段文本内容。"
    assert map_fixed_to_segments(chunk, fixed, cfg=None) == [
        "这是第一段文本内容，", "这是第二段文本内容。"
    ]


def test_seg_guard_reasons():
    assert _seg_guard("这个教研室就肯定是做不下来。", "这个教研室就肯定是坐不下来。",
                      cfg=None) == ""
    assert _seg_guard("这个教研室就肯定是做不下来。", "完全不同的文本内容，和原文无关。",
                      cfg=None) == "diff_too_large"
    assert _seg_guard("版本2020年不变。", "版本2021年不变。", cfg=None) == "number_changed"
    assert _seg_guard("参阅《工具论》相关论述再仔细核对。",
                      "参阅工具论相关论述再仔细核对。", cfg=None) == "book_title_changed"
    assert _seg_guard("这是第一句测试文本。这是第二句测试文本。",
                      "这是第一句测试文本，这是第二句测试文本。",
                      cfg=None) == "sentence_count_changed"


def test_salvage_reverts_only_offending_segment():
    """逐段抢救：只有改坏数字的那段回退，其它段的修正保留。"""
    texts = ["这个教研室就肯定是做不下来。", "版本2020年不变。", "第三段内容不变。"]
    chunk = Chunk(
        index=1, text="\n".join(texts), segment_indices=[0, 1, 2],
        segment_texts=list(texts), sentence_counts=[1, 1, 1],
        char_count=sum(len(t) for t in texts), content_hash="x",
    )
    fixed = "这个教研室就肯定是坐不下来。\n版本2021年不变。\n第三段内容不变。"
    salvaged, rejections, strategy = _salvage_segments(chunk, fixed, cfg=None)
    assert strategy == "lines"
    assert salvaged == ["这个教研室就肯定是坐不下来。", "版本2020年不变。", "第三段内容不变。"]
    assert rejections == [{"segment": 1, "reason": "number_changed"}]


def test_salvage_all_rejected_reports_common_reason():
    texts = ["版本2020年保持不变的内容。", "共1234页内容说明。"]
    chunk = Chunk(
        index=1, text="\n".join(texts), segment_indices=[0, 1],
        segment_texts=list(texts), sentence_counts=[1, 1],
        char_count=sum(len(t) for t in texts), content_hash="x",
    )
    fixed = "版本2021年保持不变的内容。\n共1235页内容说明。"
    salvaged, rejections, _strategy = _salvage_segments(chunk, fixed, cfg=None)
    assert salvaged == texts
    assert all(r["reason"] == "number_changed" for r in rejections)
    assert len(rejections) == 2


# ================================================================ 热词 / prompt
def test_load_hotwords_lookup_order_and_tolerance(tmp_path: Path, cfg):
    hw = tmp_path / "hotwords"
    hw.mkdir()
    (hw / "default.txt").write_text("通用词\n", encoding="utf-8")
    (hw / "f24.txt").write_text("亚里士多德\n形而上学\n", encoding="utf-8")
    (hw / "job123.txt").write_text("精确词,另一个词\n", encoding="utf-8")
    local_cfg = cfg.replace({"correct.hotwords_dir": str(hw)})

    assert load_hotwords(local_cfg, "job123", "data/x.m4a") == ["精确词", "另一个词"]
    assert load_hotwords(local_cfg, "f24_xxx", "data/24_x.m4a") == ["亚里士多德", "形而上学"]
    assert load_hotwords(local_cfg, "none", "data/x.m4a") == ["通用词"]
    # 目录不存在 → 容错返回 []
    missing = cfg.replace({"correct.hotwords_dir": str(tmp_path / "nope")})
    assert load_hotwords(missing, "job", "data/x.m4a") == []
    # 空文件不报错
    (hw / "empty.txt").write_text("   \n", encoding="utf-8")


def test_real_f24_hotwords(cfg):
    words = load_hotwords(
        cfg,
        job_id="7927721f_24_复旦哲学系丁耘_亚里士多德",
        source="data/24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a",
    )
    for expected in ("亚里士多德", "形而上学", "吴寿彭", "耶格尔", "原著选读", "潜能与现实"):
        assert expected in words


def test_system_prompt_fixed(cfg):
    words = load_hotwords(
        cfg,
        job_id="7927721f_24_复旦哲学系丁耘_亚里士多德",
        source="data/24_x.m4a",
    )
    text = load_system_prompt(cfg)
    assert "只输出修正后的正文" in text
    assert "不要用先验知识大段改写" in text  # 来自 configs/prompts/correct_system.txt
    assert "热词" not in text  # 热词只放 user 段
    assert load_system_prompt(cfg) == text  # 固定文本 → 前缀缓存
    assert words


# ================================================================ HTTP 客户端（mock）
def test_chat_completion_reasoning_payload(make_cfg):
    """reasoning_effort 落进请求体：low/high/max 开思考，none 关思考。"""
    with MockServer([ok("好")]) as server:
        cfg = make_cfg(server)
        chat_completion(cfg, [{"role": "user", "content": "hi"}], reasoning_effort="max")
        body = server.requests[0]["body"]
        assert body["thinking"] == {"type": "enabled"}
        assert body["reasoning_effort"] == "max"
    with MockServer([ok("好")]) as server:
        cfg = make_cfg(server)
        chat_completion(cfg, [{"role": "user", "content": "hi"}], reasoning_effort="none")
        body = server.requests[0]["body"]
        assert body["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in body


def test_normalize_usage_reads_reasoning_tokens():
    usage = _normalize_usage({
        "prompt_tokens": 100, "completion_tokens": 50,
        "prompt_cache_hit_tokens": 10,
        "completion_tokens_details": {"reasoning_tokens": 40},
    })
    assert usage == {"prompt_tokens": 100, "completion_tokens": 50,
                     "cache_hit_tokens": 10, "reasoning_tokens": 40}
    # 幂等：correct_job 会对已归一化的 usage 再归一一次，不能把 reasoning 抹成 0
    again = _normalize_usage(usage)
    assert again["reasoning_tokens"] == 40


def test_chat_completion_single_roundtrip(make_cfg):
    with MockServer([ok("修正后的正文。")]) as server:
        cfg = make_cfg(server)
        result = chat_completion(cfg, [{"role": "system", "content": "sys"},
                                       {"role": "user", "content": "usr"}])
        assert result["text"] == "修正后的正文。"
        assert result["usage"]["prompt_tokens"] == 4000
        assert len(server.requests) == 1
        request = server.requests[0]
        assert request["path"] == "/v1/chat/completions"
        assert request["headers"].get("Authorization") == "Bearer mock-test-key"
        assert request["body"]["model"] == "deepseek-flash"
        assert request["body"]["messages"][0]["role"] == "system"


def test_chat_completion_retry_429_then_ok(make_cfg):
    with MockServer([(429, {"error": {"message": "rate limit"}}), ok("成功。")]) as server:
        cfg = make_cfg(server, **{"correct.max_retries": 2})
        result = chat_completion(cfg, [{"role": "user", "content": "x"}])
        assert result["text"] == "成功。"
        assert len(server.requests) == 2


def test_chat_completion_5xx_clear_error(make_cfg):
    with MockServer([(500, {"error": {"message": "server boom"}})]) as server:
        cfg = make_cfg(server, **{"correct.max_retries": 1})
        with pytest.raises(CorrectError) as excinfo:
            chat_completion(cfg, [{"role": "user", "content": "x"}])
        assert "500" in str(excinfo.value)
        assert len(server.requests) == 2  # 初次 + 1 次重试


def test_chat_completion_timeout(make_cfg):
    def slow(_body: Any):
        time.sleep(0.6)
        return ok("迟到的正文。")

    with MockServer([slow]) as server:
        cfg = make_cfg(server, **{"correct.timeout_s": 0.15, "correct.max_retries": 0})
        with pytest.raises(CorrectError) as excinfo:
            chat_completion(cfg, [{"role": "user", "content": "x"}])
        assert "超时" in str(excinfo.value)
        assert len(server.requests) == 1


def test_compute_cost_default_prices(cfg):
    usage = {"prompt_tokens": 4000, "completion_tokens": 3000, "cache_hit_tokens": 1000}
    # 未命中 3000×¥1/M + 命中 1000×¥0.02/M + 输出 3000×¥4/M = ¥0.01502
    assert compute_cost(usage, cfg) == pytest.approx(0.01502, rel=1e-9)
    priced = cfg.replace({
        "correct.price_input_cny_per_m": 2.0,
        "correct.price_output_cny_per_m": 8.0,
        "correct.price_cache_hit_cny_per_m": 0.04,
    })
    assert compute_cost(usage, priced) == pytest.approx(0.03004, rel=1e-9)


# ================================================================ correct_job 端到端（mock）
def test_correct_job_no_key_writes_empty_and_keeps_transcript(cfg, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    job_id = "job_nokey"
    segments = [seg("这是第一段。"), seg("这是第二段。")]
    report = {
        "id": job_id, "source": "data/x.m4a", "ok": True,
        "decode": {"repaired": False, "ok": True},
        "vad": {"engine": "silero-v6"}, "trim": {"clean_dur_s": 1.0},
        "timing": {"decode_s": 2.2},
    }
    job_out = write_job(_out_dir(cfg), job_id, segments, report)
    before = (job_out / "transcript.json").read_bytes()
    local_cfg = cfg.replace({"correct.api_key": "", "correct.base_url": "http://127.0.0.1:9/v1"})

    events: List[Tuple[str, str, Dict[str, Any]]] = []
    result = correct_job(job_id, local_cfg, on_stage=lambda s, st, m: events.append((s, st, m)))

    assert result["ok"] is True
    assert result["enabled"] is False and result["skipped"] is True
    assert (job_out / "corrections.json").read_text(encoding="utf-8").strip() != ""
    corrections = json.loads((job_out / "corrections.json").read_text(encoding="utf-8"))
    assert corrections == {"total": 0, "items": [], "rejected_chunks": [],
                           "segment_rejections": []}
    assert (job_out / "transcript.json").read_bytes() == before
    updated_report = json.loads((job_out / "report.json").read_text(encoding="utf-8"))
    assert updated_report["decode"] == {"repaired": False, "ok": True}
    assert updated_report["trim"] == {"clean_dur_s": 1.0}
    assert updated_report["timing"]["decode_s"] == 2.2
    assert updated_report["correct"]["enabled"] is False
    assert updated_report["correct"]["chunks"] == 0
    assert updated_report["correct"]["usage"] == {
        "prompt_tokens": 0, "completion_tokens": 0,
        "cache_hit_tokens": 0, "reasoning_tokens": 0,
    }
    assert "correct_s" in updated_report["timing"]
    assert events and events[-1][0] == "correct" and events[-1][1] == "done"


def test_correct_job_empty_transcript_writes_empty_structure(cfg, make_cfg):
    job_id = "job_empty"
    job_out = write_job(_out_dir(cfg), job_id, [],
                        {"id": job_id, "source": "data/x.m4a", "ok": True,
                         "timing": {"decode_s": 1.0}})
    with MockServer([]) as server:
        result = correct_job(job_id, make_cfg(server))
        assert server.requests == []  # 没有文本不发请求
    assert result["ok"] is True and result["chunks"] == 0
    corrections = json.loads((job_out / "corrections.json").read_text(encoding="utf-8"))
    assert corrections == {"total": 0, "items": [], "rejected_chunks": [],
                           "segment_rejections": []}
    report = json.loads((job_out / "report.json").read_text(encoding="utf-8"))
    assert report["correct"]["enabled"] is True and report["correct"]["chunks"] == 0


def test_correct_text_without_key_is_noop(cfg, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    local_cfg = cfg.replace({"correct.api_key": ""})
    result = correct_text("这是测试文本。", local_cfg)
    assert result["ok"] is True and result["enabled"] is False
    assert result["fixed"] == "这是测试文本。"
    assert result["reason"] == "disabled_no_api_key"


def test_correct_job_multi_chunk_concurrency(make_cfg, cfg):
    """2 块并发：每块各自的 response 都能正确映射回段（按请求内容回话，不依赖到达顺序）。"""
    job_id = "job_multi"
    segments = [_chunkable_segment(i) for i in range(40)]  # 40×100 = 4000 → 2 块
    job_out = write_job(_out_dir(cfg), job_id, segments,
                        {"id": job_id, "source": "data/x.m4a", "ok": True})
    before_texts = [item["text"] for item in segments]

    def reply(body: Any):
        user = body["messages"][1]["content"]
        chunk_text = user.split("【待改文本】\n", 1)[1]
        return ok(chunk_text.replace("做", "坐"))

    with MockServer([reply]) as server:
        local_cfg = make_cfg(server, **{"correct.concurrency": 2})
        result = correct_job(job_id, local_cfg)
        assert len(server.requests) == 2

    assert result["ok"] is True
    assert result["chunks"] == 2
    assert result["rejected"] == 0
    assert result["edits"] == 40  # 每段 1 处 做→坐
    assert result["usage"]["prompt_tokens"] == 8000  # 2 块 × 4000
    assert result["usage"]["completion_tokens"] == 6000  # 2 块 × 3000
    assert result["cost_cny"] == pytest.approx(0.032, rel=1e-6)
    # usage 必须随 report 落盘，才能与 cost/账单对账（P10 后补）
    updated_report = json.loads((job_out / "report.json").read_text(encoding="utf-8"))
    assert updated_report["correct"]["usage"] == {
        "prompt_tokens": 8000, "completion_tokens": 6000,
        "cache_hit_tokens": 0, "reasoning_tokens": 0,
    }

    transcript = json.loads((job_out / "transcript.json").read_text(encoding="utf-8"))
    assert [item["text"] for item in transcript] == [t.replace("做", "坐") for t in before_texts]
    assert all(item["flags"] == [] for item in transcript)
    corrections = json.loads((job_out / "corrections.json").read_text(encoding="utf-8"))
    assert corrections["total"] == 40
    assert {item["chunk"] for item in corrections["items"]} == {1, 2}


def test_correct_job_single_chunk_roundtrip(make_cfg, cfg):
    job_id = "job_ok"
    original_1 = "这个教研室就肯定是做不下来。"
    original_2 = "这是第二段，包含内容。"
    corrected_1 = "这个教研室就肯定是坐不下来。"
    corrected_2 = "这是第二段，包含内容。"
    segments = [
        seg(original_1, start=0.0, end=3.0, text_raw=original_1),
        seg(original_2, start=3.0, end=6.0, text_raw=original_2),
    ]
    report = {
        "id": job_id, "source": "data/24_x.m4a", "ok": True,
        "decode": {"repaired": False, "ok": True},
        "timing": {"decode_s": 1.0, "total_s": 100.0},
    }
    job_out = write_job(_out_dir(cfg), job_id, segments, report)
    content = corrected_1 + "\n" + corrected_2

    events: List[Tuple[str, str, Dict[str, Any]]] = []
    with MockServer([ok(content, cache_hit_tokens=1000)]) as server:
        local_cfg = make_cfg(server)
        result = correct_job(job_id, local_cfg,
                             on_stage=lambda s, st, m: events.append((s, st, m)))
        assert len(server.requests) == 1
        # 单块往返：user prompt 含主题与热词，system 固定
        body = server.requests[0]["body"]
        assert body["messages"][0]["role"] == "system"
        user_content = body["messages"][1]["content"]
        assert "【课程主题】" in user_content
        assert "24_x.m4a" in user_content
        assert "【热词表】" in user_content
        assert "【待改文本】" in user_content
        assert body.get("thinking") == {"type": "enabled"}
        assert body.get("reasoning_effort") == "low"  # 默认纠错强度 */

        assert "亚里士多德" in user_content  # 默认热词表命中
        assert original_1.replace("，", "").replace("。", "")[:6] in user_content

    assert result["ok"] is True
    assert result["enabled"] is True
    assert result["chunks"] == 1
    assert result["edits"] >= 1
    assert result["rejected"] == 0
    assert result["cost_cny"] == pytest.approx(0.01502, rel=1e-9)

    transcript = json.loads((job_out / "transcript.json").read_text(encoding="utf-8"))
    assert transcript[0]["text"] == corrected_1
    assert transcript[0]["text_raw"] == original_1  # text_raw 保留
    assert transcript[0]["flags"] == []
    assert transcript[1]["text"] == corrected_2
    assert transcript[1]["start"] == 3.0  # 时间/说话人不动

    corrections = json.loads((job_out / "corrections.json").read_text(encoding="utf-8"))
    assert set(corrections) == {"total", "items", "rejected_chunks", "segment_rejections"}
    assert corrections["segment_rejections"] == []
    assert corrections["total"] == len(corrections["items"]) == result["edits"]
    assert corrections["rejected_chunks"] == []
    assert any(item["orig"] == "做" and item["fixed"] == "坐" for item in corrections["items"])
    for item in corrections["items"]:
        assert set(item) == {"orig", "fixed", "reason", "chunk"}
        assert item["chunk"] == 1
        assert item["reason"]

    updated_report = json.loads((job_out / "report.json").read_text(encoding="utf-8"))
    assert updated_report["decode"] == {"repaired": False, "ok": True}
    assert updated_report["correct"]["enabled"] is True
    assert updated_report["correct"]["chunks"] == 1
    assert updated_report["correct"]["edits"] == corrections["total"]
    assert updated_report["correct"]["rejected"] == 0
    assert updated_report["correct"]["cost_cny"] == pytest.approx(0.01502, rel=1e-9)
    # usage 与 cost 同源落盘：prompt 4000 / completion 3000 / cache 1000
    assert updated_report["correct"]["usage"] == {
        "prompt_tokens": 4000, "completion_tokens": 3000,
        "cache_hit_tokens": 1000, "reasoning_tokens": 0,
    }
    assert updated_report["timing"]["decode_s"] == 1.0
    assert updated_report["timing"]["correct_s"] >= 0.0
    assert events and events[-1][1] == "done"


def test_correct_job_rejects_chunk_and_rolls_back(make_cfg, cfg):
    job_id = "job_reject"
    segments = [seg("第一段原文内容，数字2020。"), seg("第二段原文内容，《书名》。")]
    report = {"id": job_id, "source": "data/x.m4a", "ok": True,
              "decode": {"ok": True, "repaired": False}, "timing": {"decode_s": 1.0}}
    job_out = write_job(_out_dir(cfg), job_id, segments, report)
    bad = "完全不同的第一段。\n完全不同的第二段。"

    with MockServer([ok(bad)]) as server:
        result = correct_job(job_id, make_cfg(server))

    assert result["ok"] is True
    assert result["chunks"] == 1
    assert result["edits"] == 0
    assert result["rejected"] == 1
    corrections = json.loads((job_out / "corrections.json").read_text(encoding="utf-8"))
    assert corrections["total"] == 0 and corrections["items"] == []
    assert corrections["rejected_chunks"][0]["chunk"] == 1
    assert corrections["rejected_chunks"][0]["reason"] == "diff_too_large"
    assert 0 < corrections["rejected_chunks"][0]["diff_ratio"] <= 2.0
    # 整块回退：transcript 的 text 不动（只写 flags），flags 标记 rejected_chunk
    transcript = json.loads((job_out / "transcript.json").read_text(encoding="utf-8"))
    assert [item["text"] for item in transcript] == [item["text"] for item in segments]
    assert all(FLAG_REJECTED_CHUNK in item["flags"] for item in transcript)
    updated_report = json.loads((job_out / "report.json").read_text(encoding="utf-8"))
    assert updated_report["correct"]["rejected"] == 1


def test_correct_job_salvages_per_segment(make_cfg, cfg):
    """P9：同块内只有一段改坏数字 → 该段回退、其它段的修正保留，整块不回退。"""
    job_id = "job_salvage"
    s1 = "这个教研室就肯定是做不下来。"
    s2 = "版本2020年不变。"
    s3 = "第三段内容不变。"
    segments = [seg(s1), seg(s2), seg(s3)]
    report = {"id": job_id, "source": "data/x.m4a", "ok": True,
              "decode": {"ok": True, "repaired": False}, "timing": {"decode_s": 1.0}}
    job_out = write_job(_out_dir(cfg), job_id, segments, report)
    content = "这个教研室就肯定是坐不下来。\n版本2021年不变。\n第三段内容不变。"

    with MockServer([ok(content)]) as server:
        result = correct_job(job_id, make_cfg(server))

    assert result["ok"] is True
    assert result["rejected"] == 0            # 没有整块回退
    assert result["rejected_segments"] == 1
    assert result["edits"] >= 1

    transcript = json.loads((job_out / "transcript.json").read_text(encoding="utf-8"))
    assert transcript[0]["text"] == "这个教研室就肯定是坐不下来。"
    assert transcript[1]["text"] == s2        # 数字被改的段回退原文
    assert FLAG_REJECTED_SEGMENT in transcript[1]["flags"]
    assert FLAG_REJECTED_CHUNK not in transcript[1]["flags"]
    assert transcript[2]["text"] == s3

    corrections = json.loads((job_out / "corrections.json").read_text(encoding="utf-8"))
    assert corrections["rejected_chunks"] == []
    assert corrections["segment_rejections"] == [{
        "chunk": 1, "segment": 1, "segment_in_chunk": 1, "reason": "number_changed",
    }]
    assert any(item["orig"] == "做" and item["fixed"] == "坐"
               for item in corrections["items"])

    updated_report = json.loads((job_out / "report.json").read_text(encoding="utf-8"))
    assert updated_report["correct"]["rejected"] == 0
    assert updated_report["correct"]["rejected_segments"] == 1


def test_correct_text_number_change_reverts(make_cfg):
    """单块金标工具：数字被改 → reason=number_changed、fixed 保持原文。"""
    with MockServer([ok("版本2021年不变。")]) as server:
        result = correct_text("版本2020年不变。", make_cfg(server))
    assert result["ok"] is False
    assert result["reason"] == "number_changed"
    assert result["fixed"] == "版本2020年不变。"


def test_correct_job_request_failure_returns_error(make_cfg, cfg):
    job_id = "job_500"
    segments = [seg("这是第一段内容，数字2020。"), seg("这是第二段内容。")]
    job_out = write_job(_out_dir(cfg), job_id, segments)
    before = (job_out / "transcript.json").read_bytes()

    with MockServer([(500, {"error": {"message": "boom"}})]) as server:
        local_cfg = make_cfg(server, **{"correct.max_retries": 1})
        events: List[Tuple[str, str, Dict[str, Any]]] = []
        result = correct_job(job_id, local_cfg,
                             on_stage=lambda s, st, m: events.append((s, st, m)))

    assert result["ok"] is False
    assert result["stage"] == "correct"
    assert "500" in result["error"]
    assert not (job_out / "corrections.json").exists()
    assert (job_out / "transcript.json").read_bytes() == before
    assert events and events[-1][1] == "failed"


def test_correct_job_skips_existing_and_force_reruns(make_cfg, cfg):
    job_id = "job_skip"
    segments = [seg("这是第一段内容，数字2020。")]
    job_out = write_job(_out_dir(cfg), job_id, segments)
    content = "这是第一段内容，数字2020。"

    with MockServer([ok(content), ok(content)]) as server:
        local_cfg = make_cfg(server)
        first = correct_job(job_id, local_cfg)
        second = correct_job(job_id, local_cfg)
        assert first["ok"] and not first["skipped"]
        assert second["ok"] and second["skipped"] is True
        assert len(server.requests) == 1
        third = correct_job(job_id, local_cfg, force=True)
        assert third["ok"] and not third["skipped"]
        assert len(server.requests) == 2
    assert (job_out / "corrections.json").is_file()


# ================================================================ 金标验收
def _gold_path(name: str) -> Path:
    return GOLD_DIR / name


def _gold_coverage(orig_text: str, fixed_text: str) -> Tuple[int, int, List[str]]:
    """金标修正点覆盖率：gold 的 orig 变体在原文出现处，逐字对齐片段 == corrected，
    且该区间内所有差异 op 都由生成的 items（diff opcode）覆盖。"""
    gold = json.loads(_gold_path("corrections_chunk1.json").read_text(encoding="utf-8"))
    gold_items = gold.get("items") or []
    o = re.sub(r"\s+", "", orig_text)
    f = re.sub(r"\s+", "", fixed_text)
    ops = difflib.SequenceMatcher(None, o, f, autojunk=False).get_opcodes()
    edits = diff_edits(o, f)

    def b_start(i: int) -> int:
        for tag, a1, a2, b1, b2 in ops:
            if a1 <= i < a2:
                return b1 + (i - a1) if tag == "equal" else b1
        return len(f)

    def b_end(i: int) -> int:
        for tag, a1, a2, b1, b2 in ops:
            if a1 < i <= a2:
                return b1 + (i - a1) if tag == "equal" else b2
        return len(f)

    def variants(value: str) -> List[str]:
        return [part for part in (value or "").split("/") if part]

    def covered_once(variant: str, gold_fixed: str) -> bool:
        start = 0
        while True:
            pos = o.find(variant, start)
            if pos < 0:
                return False
            start = pos + 1
            i1, i2 = pos, pos + len(variant)
            j1, j2 = b_start(i1), b_end(i2)
            aligned = f[j1:j2]
            if aligned != gold_fixed and aligned not in variants(gold_fixed):
                continue
            needed = [op for op in ops if op[0] != "equal" and op[1] >= i1 and op[2] <= i2]
            if needed and not all(
                any(edit.i1 == op[1] and edit.i2 == op[2] and
                    edit.j1 == op[3] and edit.j2 == op[4] for edit in edits)
                for op in needed
            ):
                continue
            return True

    covered = 0
    missed: List[str] = []
    for item in gold_items:
        hit = any(covered_once(v, item.get("fixed") or "")
                  for v in variants(item.get("orig") or ""))
        if hit:
            covered += 1
        else:
            missed.append(item.get("orig"))
    return covered, len(gold_items), missed


@pytest.mark.skipif(
    not (_gold_path("input_chunk1.txt").is_file()
         and _gold_path("corrected_chunk1.txt").is_file()
         and _gold_path("corrections_chunk1.json").is_file()),
    reason="缺少金标文件 work/cost/correct_test/",
)
def test_gold_acceptance(make_cfg, cfg):
    orig_text = _gold_path("input_chunk1.txt").read_text(encoding="utf-8").strip()
    corrected_text = _gold_path("corrected_chunk1.txt").read_text(encoding="utf-8").strip()
    gold = json.loads(_gold_path("corrections_chunk1.json").read_text(encoding="utf-8"))

    with MockServer([ok(corrected_text, prompt_tokens=4200, completion_tokens=4209,
                        cache_hit_tokens=0)]) as server:
        local_cfg = make_cfg(server, **{"correct.hotwords_dir": str(ROOT / "configs" / "hotwords")})
        result = correct_text(
            orig_text, local_cfg,
            job_id="7927721f_24_复旦哲学系丁耘_亚里士多德",
            source="data/24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a",
        )

    assert result["enabled"] is True
    assert result["ok"] is True, result
    assert result["reason"] == ""
    assert result["fixed"] == corrected_text
    # 校验：diff 2.2% ≤12%；句数基本不变（本实现把"。。"/"……"连续标点算 1 句，
    # 与 PoC 脚本的 naive re.split 计数差 1，故只断言相对变化 ≤5%）
    detail = result["validation"]
    assert detail["ok"] is True
    assert detail["diff_ratio"] <= 0.12
    assert 110 <= detail["n_sentences_orig"] <= 125
    assert abs(detail["sentence_change_ratio"]) <= 0.05
    assert result["diff_ratio"] == pytest.approx(91 / 4200, abs=1e-6)
    # items：模型输出与金标的对应差异都进 items（数量级正确）
    assert len(result["items"]) >= 50
    assert result["cost_cny"] == pytest.approx((4200 * 1 + 4209 * 4) / 1_000_000, rel=1e-9)

    # 金标覆盖率 ≥70%（实际 34/35≈97%，未覆盖的「形而上学的里论」是金标 orig
    # 变体本身在原文中不存在——原文写作「形象上学的里论」——数据口径问题）
    covered, total, missed = _gold_coverage(orig_text, corrected_text)
    coverage = covered / total
    assert total == len(gold["items"])
    assert coverage >= 0.70, f"金标覆盖率 {covered}/{total}={coverage:.3f}，未覆盖：{missed}"
