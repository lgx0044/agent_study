#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 LLM GPU 调用脚本
支持 Qwen 2.5 7B 和 Hermes 3 8B，通过 llama.cpp 调用 RTX 5060 加速
"""

import subprocess
import os
import sys
import tempfile
import argparse
import re
import time

# ==================== 模型配置 ====================
MODELS = {
    "qwen": {
        "path":  r"D:\models\qwen2.5-7b\qwen2.5-7b-instruct-q4_k_m.gguf",
        "name":  "Qwen 2.5 7B Q4_K_M",
        "format": "chatml",
        "system": "你是一个有帮助的AI助手，请用中文回答。",
    },
    "hermes": {
        "path":  r"D:\models\hermes3-8b\Hermes-3-Llama-3.1-8B.Q4_K_M.gguf",
        "name":  "Hermes 3 Llama 3.1 8B Q4_K_M",
        "format": "chatml",
        "system": "You are a helpful assistant. Respond in Chinese when asked in Chinese.",
    },
    "qwopus": {
        "path":  r"D:\models\qwopus3.5-9b\Qwen3.5-9B.Q4_K_M.gguf",
        "name":  "Qwopus 3.5 9B v3 Q4_K_M (带思考)",
        "format": "chatml",
        "system": "你是一个有帮助的AI助手，请用中文回答。",
        "thinking": True,  # 支持深度思考模式
    },
}

LLAMA_CLI    = r"D:\software\llama-cpp-b8741\llama-cli.exe"
GPU_LAYERS   = 99
CPU_THREADS  = 6
CONTEXT_SIZE = 4096

DEFAULT_TEMP    = 0.7
DEFAULT_TOP_P  = 0.9
DEFAULT_MAX    = 5120

# ==================== 格式构建 ====================

def build_prompt_chatml(messages: list[dict], system: str = "") -> str:
    """ChatML 格式（Hermes / Llama 3 通用）"""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def build_prompt_qwen(messages: list[dict], system: str = "") -> str:
    """Qwen ChatML 格式"""
    return build_prompt_chatml(messages, system)


# ==================== 核心调用 ====================

def chat(
    prompt: str,
    model_key: str = "qwen",
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_MAX,
    history: list[dict] | None = None,
) -> str:
    """
    与本地模型对话

    Args:
        prompt:     用户输入
        model_key:  模型 key ("qwen" 或 "hermes")
        temperature: 温度
        max_tokens:  最大生成 token 数
        history:    对话历史
    """
    cfg = MODELS.get(model_key, MODELS["qwen"])
    messages = list(history) if history else []
    messages.append({"role": "user", "content": prompt})

    formatted = build_prompt_chatml(messages, cfg["system"])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(formatted)
        prompt_file = f.name

    try:
        cmd = [
            LLAMA_CLI,
            "-m", cfg["path"],
            "-f", prompt_file,
            "-n", str(max_tokens),
            "-ngl", str(GPU_LAYERS),
            "-t", str(CPU_THREADS),
            "--temp", str(temperature),
            "--top-p", str(DEFAULT_TOP_P),
            "-c", str(CONTEXT_SIZE),
            "--no-display-prompt",
            "--single-turn",      # 单轮生成后退出
            "--log-disable",      # 禁用日志输出
        ]

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        t0 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=env,
            timeout=120,
        )
        elapsed = time.time() - t0

        text = _extract(result.stdout + result.stderr)
        stats = _extract_perf(result.stdout + result.stderr)
        stats["elapsed"] = round(elapsed, 2)

        if stats.get("eval_speed"):
            print(f"  ⚡ {stats['eval_speed']} tok/s | {elapsed:.1f}s", file=sys.stderr)

        return text
    finally:
        os.unlink(prompt_file)


# ==================== 辅助函数 ====================

_LOG_KEYWORDS = [
    "llama_model_loader:", "llama_model_load:", "llm_load_",
    "llama_perf_", "ggml_cuda_init:", "GGML_CUDA",
    "common_init_from_params:", "system_info:",
    "sampler chain:", "sampler seed:", "sampler params:",
    "generate:", "repeat_last_n", "dry_multiplier", "dry_penalty",
    "top_k:", "top_p:", "min_p:", "typical_p:", "mirostat",
    "penalty_prompt:", "penalty_freq:", "penalty_present:",
    "VMM:", "build      :", "model      :", "modalities :",
    "available commands:", "/exit", "/regen", "/clear", "/read", "/glob",
    "Loading model", "Exiting...", "load_backend:",
    "offloaded", "llama_context:", "▄", "██",
    "kv self size", "graph nodes", "graph splits",
    "Prompt:", "Generation:",
]


def _is_log(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    # 进度点行 / banner 图案
    if len(stripped) < 3 and set(stripped.replace(" ", "")) <= {".▄▀█"}:
        return True
    # llama banner 图案行
    if set(stripped.replace(" ", "")) <= {"▄", "▀", "█"}:
        return True
    return any(kw in line for kw in _LOG_KEYWORDS)


def _extract(output: str) -> str:
    lines = output.split("\n")
    parts = [l for l in lines if not _is_log(l)]
    text = "\n".join(parts).strip()
    # 过滤 prompt 显示行
    text = re.sub(r"^>\s*<\|im_start\|>.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(user|assistant|system)\s*\n", "", text, flags=re.I)
    # 只保留 assistant 之后的回答（去掉回显的 system/user prompt）
    m = re.search(r"<\|im_start\|>assistant\s*\n", text)
    if m:
        text = text[m.end():]
    text = text.replace("[end of text]", "").strip()
    # 提取思考内容
    thinking = ""
    m = re.search(r"\[Start thinking\](.*?)\[End thinking\]", text, re.DOTALL)
    if m:
        thinking = m.group(1).strip()
    # 提取最终回答（思考部分之后）
    if "[End thinking]" in text:
        answer = text.split("[End thinking]", 1)[-1].strip()
    else:
        answer = text
    # 如果有思考内容，格式化输出
    if thinking:
        return f"💭 思考:\n{thinking}\n\n📝 回答:\n{answer}"
    return answer


def _extract_perf(output: str) -> dict:
    stats = {}
    # 旧格式: eval time = xxx ms / N runs (xx ms per token, yy tokens per second)
    m = re.search(r"eval time\s+=\s+[\d.]+ ms\s*/\s+\d+ runs\s+\(\s*[\d.]+ ms per token,\s*([\d.]+)\s*tokens per s", output)
    if m:
        stats["eval_speed"] = round(float(m.group(1)), 1)
    # 新格式: [ Prompt: xxx t/s | Generation: yyy t/s ]
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", output)
    if m:
        stats["eval_speed"] = round(float(m.group(1)), 1)
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s", output)
    if m:
        stats["prompt_speed"] = round(float(m.group(1)), 1)
    m = re.search(r"offloaded\s+(\d+)/(\d+) layers to GPU", output)
    if m:
        stats["gpu_layers"] = f"{m.group(1)}/{m.group(2)}"
    return stats


# ==================== 交互模式 ====================

def interactive(model_key: str = "qwen"):
    cfg = MODELS.get(model_key, MODELS["qwen"])
    print("=" * 50)
    print(f"  {cfg['name']} · GPU 模式 · RTX 5060")
    print("  输入 'quit' 退出, 'clear' 清空历史")
    print("=" * 50)
    print()

    history = []
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            history = []
            print("✅ 历史已清空\n")
            continue

        print("助手: ", end="", flush=True)
        t0 = time.time()
        reply = chat(user_input, model_key=model_key, history=history)
        print(reply)
        print(f"  ({time.time() - t0:.1f}s)\n")

        history += [{"role": "user", "content": user_input},
                    {"role": "assistant", "content": reply}]
        if len(history) > 20:
            history = history[-16:]


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="本地 LLM GPU 调用工具 (Qwen / Hermes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python local_llm.py -m qwen    -p "你好"
  python local_llm.py -m hermes  -p "你好"
  python local_llm.py -m qwopus  -p "1+1等于几？"
  python local_llm.py -m qwen    -i
  python local_llm.py -m qwopus  -i
  python local_llm.py -m qwen    -p "写个快排" --code
        """,
    )
    parser.add_argument("-m", "--model", choices=["qwen", "hermes", "qwopus"], default="qwen",
                        help="选择模型 (默认 qwen)")
    parser.add_argument("-p", "--prompt", type=str, help="单次提问")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    parser.add_argument("-n", "--max-tokens", type=int, default=DEFAULT_MAX, help="最大 token 数")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP, help="温度")
    parser.add_argument("--code", action="store_true", help="编程模式 (低温)")

    args = parser.parse_args()

    cfg = MODELS[args.model]
    if not os.path.exists(cfg["path"]):
        print(f"❌ 模型文件不存在: {cfg['path']}")
        sys.exit(1)

    temp = 0.2 if args.code else args.temp

    if args.interactive:
        interactive(args.model)
    elif args.prompt:
        print(f"模型: {cfg['name']}")
        print(f"温度: {temp}")
        print()
        reply = chat(args.prompt, model_key=args.model, temperature=temp,
                     max_tokens=args.max_tokens)
        print(reply)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
