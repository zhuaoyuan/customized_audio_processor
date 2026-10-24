# -*- coding: utf-8 -*-
"""P5 真实 transcript + 本地 mock OpenAI 兼容服务 的端到端纠错验收。

不访问真实 API；把 f24 的 transcript.json/report.json 复制到临时 out 目录，启动
stdlib http.server 模拟 /v1/chat/completions，验证：
1) 正常路径：分块、写回、corrections.json、report.correct；
2) 拒绝路径：某块返回大幅改写 → 校验拒绝、整块回退、flags += rejected_chunk。
"""
import json
import shutil
import sys
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config
from app.pipeline.correct import correct_job, plan_chunks

JOB = "7927721f_24_复旦哲学系丁耘_亚里士多德"
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "out" / JOB


class MockHandler(BaseHTTPRequestHandler):
    mode = "fix"          # fix | reject
    n_requests = 0
    lock = threading.Lock()

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        user = next((m.get("content", "") for m in payload.get("messages", [])
                     if m.get("role") == "user"), "")
        marker = "【待改文本】\n"
        body = user.split(marker, 1)[1] if marker in user else user
        with MockHandler.lock:
            MockHandler.n_requests += 1
            n = MockHandler.n_requests
        if MockHandler.mode == "reject" and n == 2:
            # 第 2 块：大幅改写（删句 + 加字），应被 diff/句数校验拒绝
            body = "这是一段完全不同的文本，用来验证拒绝回退。"
        else:
            # 单字同音修正（1/15 段内差异，低于单段 12% 上限，可安全写回）
            body = body.replace("愿意接", "愿意借")
        content = json.dumps({
            "id": "mock", "object": "chat.completion", "model": payload.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": body},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(user), "completion_tokens": len(body),
                      "total_tokens": len(user) + len(body)},
        }, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def make_cfg(base_url, out_dir):
    return load_config(overrides={
        "paths.out_dir": str(out_dir),
        "correct.base_url": base_url,
        "correct.api_key": "sk-mock-not-real",
        "correct.enabled": True,
        "correct.timeout_s": 30,
        "correct.retry_backoff_s": 0.01,
    })


def run_case(server, mode, tag):
    MockHandler.mode = mode
    MockHandler.n_requests = 0
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    tmp = Path(tempfile.mkdtemp(prefix=f"p5mock_{tag}_", dir=str(ROOT / "work")))
    job_out = tmp / JOB
    job_out.mkdir(parents=True)
    shutil.copy2(SRC / "transcript.json", job_out / "transcript.json")
    shutil.copy2(SRC / "report.json", job_out / "report.json")
    cfg = make_cfg(base, tmp)
    result = correct_job(JOB, cfg, force=True)
    report = json.loads((job_out / "report.json").read_text(encoding="utf-8"))
    corrections = json.loads((job_out / "corrections.json").read_text(encoding="utf-8"))
    transcript = json.loads((job_out / "transcript.json").read_text(encoding="utf-8"))
    texts_changed = sum(1 for a, b in zip(json.loads((SRC / "transcript.json").read_text(encoding="utf-8")), transcript)
                        if a["text"] != b["text"])
    flagged = sum(1 for seg in transcript if "rejected_chunk" in (seg.get("flags") or []))
    print(f"\n=== {tag} (mock mode={mode}) ===")
    print("ok", result.get("ok"), "enabled", result.get("enabled"), "chunks", result.get("chunks"),
          "edits", result.get("edits"), "rejected", result.get("rejected"),
          "cost", result.get("cost_cny"), "elapsed", result.get("elapsed_s"))
    print("report.correct", json.dumps(report.get("correct"), ensure_ascii=False))
    print("corrections total", corrections.get("total"),
          "rejected_chunks", corrections.get("rejected_chunks"),
          "first items", corrections.get("items")[:3])
    print("changed segments", texts_changed, "flagged", flagged)
    print("mock requests", MockHandler.n_requests)
    return result, report, corrections, transcript


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # 先确认 plan_chunks 的分块规模
        cfg0 = load_config()
        tr = json.loads((SRC / "transcript.json").read_text(encoding="utf-8"))
        chunks = plan_chunks(tr, cfg0)
        print("plan_chunks:", len(chunks), [c.char_count for c in chunks])
        ok_case = run_case(server, "fix", "fix")
        reject_case = run_case(server, "reject", "reject")
        assert ok_case[0]["ok"] and ok_case[0]["edits"] >= 1
        assert ok_case[0]["rejected"] == 0
        assert ok_case[2]["total"] == ok_case[0]["edits"]
        assert reject_case[0]["ok"] and reject_case[0]["rejected"] >= 1
        assert reject_case[1]["correct"]["enabled"] is True
        assert reject_case[1]["correct"]["rejected"] >= 1
        print("\nMOCK_E2E_PASS")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
