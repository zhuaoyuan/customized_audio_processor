@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo 启动 Web UI: http://127.0.0.1:7860
echo 按 Ctrl+C 停止服务；关闭本窗口即退出
work\venv\Scripts\python -m app.cli serve --port 7860
pause
