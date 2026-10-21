import json, os, re, datetime

src = r"C:\projects\local\audio_processer\work\bench\asr_sv_full78min_chunk30_t8.json"
outdir = r"C:\projects\local\audio_processer\out"
os.makedirs(outdir, exist_ok=True)
d = json.load(open(src, encoding="utf-8"))
text = d["text"]

# 去掉片头 0-480s 的误触发碎片（例如 "去做。。。。。。。。。"），
# 正文实测从 ≈480s（8:00）开始，到 ≈4610s 结束
LEAD = text.find("起迟到了")
if LEAD > 0:
    text = text[LEAD:]
T_START, T_END = 480.0, 4610.0

text = re.sub(r"[。．\.]{3,}", "。", text)
text = text.lstrip("。 ,，")

# 分段
raw = re.split(r"(?<=[。？！])", text)
segments, buf = [], ""
for s in raw:
    s = s.strip()
    if not s:
        continue
    buf += s
    if len(buf) >= 80:
        segments.append(buf); buf = ""
if buf:
    segments.append(buf)

total = sum(len(s) for s in segments) or 1
lines, acc = [], 0
for s in segments:
    t = T_START + acc / total * (T_END - T_START)
    acc += len(s)
    mm, ss = int(t // 60), int(t % 60)
    lines.append(f"**[{mm:02d}:{ss:02d}]** {s}\n")

header = f"""# 亚里士多德《形而上学》(01) · 机器转写草稿

> 来源：`data/24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a`（77.9 分钟）
> 引擎：SenseVoiceSmall int8（sherpa-onnx，CPU 8 线程），整片实测 RTF 0.0301，输出 {d['text'].__len__()} 字
> 有效正文：约 **08:00–76:50**（片头 0–480 s、片尾约 4600 s 后为非讲课内容，已剔除）
> 生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}

**状态：未校对的机器转写草稿。**
- 同音专名错误：原著选读→原助选读、形而上学→行式上学/学而上/形儿上学、亚里士多德→亚里士德、丁耘未出现；
- 建议用热词表 + 人工校对后再作为正式稿；
- **时间戳是按字符数比例估算的近似值**（SenseVoice 不输出原生时间戳），偏差可达数十秒，仅用于定位。
"""

with open(os.path.join(outdir, "24_形而上学01_机器转写草稿.md"), "w", encoding="utf-8") as f:
    f.write(header + "\n---\n\n" + "\n".join(lines))
with open(os.path.join(outdir, "24_形而上学01_机器转写草稿.txt"), "w", encoding="utf-8") as f:
    f.write(text)

print("chars:", len(text), "segments:", len(segments))
print("first 80:", text[:80])
