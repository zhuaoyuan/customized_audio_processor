"""把 docs/research/timeline-poc.md 的样例块（md 前 30 行 / srt 前 20 条）同步为真实产物。"""
import io
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DOC = os.path.join(ROOT, "docs", "research", "timeline-poc.md")
OUTDIR = os.path.join(ROOT, "work", "timeline", "24_形而上学")


def md_block():
    lines = io.open(os.path.join(OUTDIR, "transcript.md"), encoding="utf-8").read().splitlines()
    # 跳过标题/说明（前 5 行），取正文前 30 行
    body = [ln for ln in lines if ln.startswith("[")]
    return "\n".join(body[:30])


def srt_block():
    raw = io.open(os.path.join(OUTDIR, "transcript.srt"), encoding="utf-8").read()
    cues = [c for c in raw.strip().split("\n\n") if c.strip()]
    out = []
    for c in cues[:20]:
        lines = c.splitlines()
        out.append(lines[0])
        out.append(lines[1])
        out.append("\n".join(lines[2:]))
        out.append("")
    return "\n".join(out).rstrip()


def main():
    doc = io.open(DOC, encoding="utf-8").read()
    doc = doc.replace("<!--SAMPLES_MD-->", "```\n" + md_block() + "\n```")
    doc = doc.replace("<!--SAMPLES_SRT-->", "```\n" + srt_block() + "\n```")
    io.open(DOC, "w", encoding="utf-8").write(doc)
    print("[doc] snippets synced")


if __name__ == "__main__":
    main()
