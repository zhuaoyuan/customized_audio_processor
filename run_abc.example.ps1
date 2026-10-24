# =====================================================================
#  对目标目录下所有文件跑 A+B+C：转写+纠错+导出 → 全文成稿
#  用法：pwsh -NoProfile -ExecutionPolicy Bypass -File .\run_abc.ps1
#        或双击同目录的 run_abc.cmd
#  只需改下面 5 行；本文件含 API key，已被 .gitignore 忽略，不要提交。
# =====================================================================

$root = '\PATH\TO\audio_processer'
$py   = "$root\work\venv\Scripts\python.exe"
$src  = ''                    # 目标目录（也可以是单个文件）
$out  = ''                        # 输出目录
$env:DEEPSEEK_API_KEY = 'sk-...'

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH       = $root

New-Item -ItemType Directory -Force -Path $out | Out-Null

& $py -m app.cli batch   $src --out $out    # A+B：登记 + 全链路（含纠错、导出）
& $py -m app.cli compose --all --out $out   # C：全文成稿

# 想「A+B 有失败就别跑 C」，把上面最后一行换成：
# if ($LASTEXITCODE -eq 0) { & $py -m app.cli compose --all --out $out }
