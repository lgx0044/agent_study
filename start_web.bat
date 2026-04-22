@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo   紫微斗数 Agent Web 服务
echo   访问: http://127.0.0.1:8000
echo   按 Ctrl+C 停止
echo ╚══════════════════════════════════════════════════════════════╝
echo.
cd /d "%~dp0"
uvicorn web_app:app --host 127.0.0.1 --port 8000 --reload
