@echo off
title Qwen 2.5 7B API Server
echo 正在启动本地大模型 API 服务...

D:\software\llama-cpp-bin\llama-server.exe -m D:\models\qwen2.5-7b\qwen2.5-7b-instruct-q4_k_m.gguf -c 4096 -ngl 99 --port 8080 --host 127.0.0.1

pause