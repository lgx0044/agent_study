# 用本地 7B 模型搭一个紫微斗数 AI Agent

> 全本地部署 · ReAct 自愈推理 · 三路混合检索 · 三层记忆 · Web 界面
>
> GitHub: [lgx0044/agent_study](https://github.com/lgx0044/agent_study)

---

## 这个项目做了什么

一个完全在本地运行的紫微斗数（中国传统命理学）智能问答系统。没有调用任何云端 API，所有推理、检索、对话都在你自己的机器上完成。

核心能力：

- **ReAct 自愈推理** — Agent 会先思考（Thought）→ 决定要不要查资料（Action）→ 获取结果（Observation）→ 给出最终答案。如果中间推理出错，它会自动纠正
- **三路混合检索** — 不是简单的向量搜索，而是向量检索 + BM25 关键词检索 + Cross-Encoder 重排序，三路召回再精排
- **三层记忆架构** — 短期记忆（当前对话上下文）、长期记忆（跨会话 ChromaDB）、动态系统提示（实时注入记忆和时间信息）
- **Web 界面** — FastAPI 后端 + SSE 流式输出，前端实时展示 Agent 的思考过程、工具调用、检索结果

一句话总结：**用 7B 本地模型做到了接近云端 API 的问答体验，同时保证数据不出本机。**

---

## 技术架构

```
┌─────────────────────────────────────────────────┐
│                   用户浏览器                       │
│              index.html (原生 JS)                 │
└────────────────────┬────────────────────────────┘
                     │ SSE (text/event-stream)
┌────────────────────▼────────────────────────────┐
│              FastAPI Web 服务                     │
│              web_app.py                           │
│                                                   │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ ReAct    │  │ 三层记忆管理  │  │ 会话持久化  │ │
│  │ 推理引擎  │  │ 短/长/动态   │  │ JSON 文件   │ │
│  └────┬─────┘  └──────────────┘  └────────────┘ │
│       │                                            │
│       ▼                                            │
│  ┌─────────────────────────────────────────────┐  │
│  │          混合检索 Pipeline                    │  │
│  │                                              │  │
│  │  向量检索 ──┐                                 │  │
│  │  (BGE-small) ├── 去重合并 ── CrossEncoder    │  │
│  │  BM25 ──────┘              (BGE-reranker)     │  │
│  │                                              │  │
│  └─────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────┘
                     │ OpenAI 兼容 API
┌────────────────────▼────────────────────────────┐
│         llama.cpp Server (本地推理)               │
│         Qwen2.5-7B-Instruct (Q4_K_M 量化)       │
└─────────────────────────────────────────────────┘
```

### 关键组件说明

| 组件 | 技术 | 作用 |
|------|------|------|
| LLM | Qwen2.5-7B-Instruct, Q4_K_M 量化, llama.cpp | 核心推理引擎，本地运行 |
| Embedding | BGE-small-zh-v1.5 (90MB) | 中文文本向量化 |
| Reranker | BGE-reranker-base (1GB) | 检索结果精排 |
| 向量数据库 | ChromaDB | 存储知识库和长期记忆 |
| 关键词检索 | BM25Okapi + jieba 分词 | 中文关键词匹配 |
| 后端 | FastAPI + SSE | 异步服务 + 流式输出 |
| 前端 | 原生 HTML/CSS/JS | 无框架依赖，暗色主题 |

---

## 项目结构

```
agent_study/
├── web_app.py              # FastAPI 主服务（19.5KB，核心文件）
├── index.html              # Web 前端界面（31.5KB）
├── advanced_rag.py         # 混合检索 Pipeline 类
├── agent_react.py          # ReAct Agent 原型实现
├── langgraph_agent.py      # LangGraph 状态机版本
├── memory.py               # 记忆管理模块
├── llm_engine.py           # LLM API 封装
├── clean_data.py           # 知识库数据清洗
├── start_web.bat           # Windows 一键启动脚本
├── llama.bat               # llama-server 启动脚本
│
├── local_models/           # 本地模型（需自行下载）
│   ├── bge-small-zh-v1.5/          # Embedding 模型
│   └── bge-reranker-base/          # Reranker 模型
│
├── chroma_db_hybrid/       # 紫微知识库（873 条，已建好）
├── long_term_memory_db/    # 长期记忆存储
└── data/sessions/          # 会话持久化文件
```

---

## ReAct 推理流程详解

ReAct（Reasoning + Acting）是这个项目的核心推理框架。每一轮对话，Agent 会执行以下循环（最多 3 轮）：

```
用户: "天机星入命宫是什么意思？"
        │
        ▼
   ┌─ Thought ─────────────────────────────┐
   │ "用户问的是天机星的命宫含义，需要查资料" │
   └──────────────┬────────────────────────┘
                  │
                  ▼
   ┌─ Action: search_ziwei ────────────────┐
   │ {"query": "天机星 命宫 特点"}          │
   └──────────────┬────────────────────────┘
                  │
                  ▼
   ┌─ Observation ─────────────────────────┐
   │ "天机星入命宫，主智慧、善谋略..."       │
   │ (从知识库检索到的 3 条相关文档)         │
   └──────────────┬────────────────────────┘
                  │
                  ▼
   ┌─ Final Answer ────────────────────────┐
   │ "天机星入命宫，其人聪明伶俐..."         │
   └───────────────────────────────────────┘
```

如果第一轮检索到的资料不够，Agent 会自动发起新的检索请求（换关键词），最多迭代 3 次。

在 Web 界面上，整个思考过程以可折叠卡片的形式实时展示——用户能看到 Agent 在"想什么"、"查了什么"、"找到了什么"。

---

## 三路混合检索

只用向量检索有个问题：命理学有很多专业术语（如"化忌""入庙""旺相"），向量模型对术语的精确匹配能力不如关键词检索。

所以做了三路召回：

1. **向量检索**（BGE-small-zh-v1.5）— 处理同义词、语义相近的查询
2. **BM25 关键词检索**（jieba 分词）— 精确匹配专业术语
3. **Cross-Encoder 重排序**（BGE-reranker-base）— 对两路结果合并去重后，用更强的语义模型重新打分排序

实际效果：873 条知识库文档中，先用向量取 top-10 + BM25 取 top-10 → 合并去重 → Reranker 精排 → 返回最相关的 3 条给 Agent。

---

## 三层记忆系统

| 层级 | 存储 | 生命周期 | 作用 |
|------|------|---------|------|
| 短期记忆 | Python 列表（内存） | 当前会话 | 保留最近 3 轮对话上下文 |
| 长期记忆 | ChromaDB（磁盘） | 永久 | 跨会话的知识，向量检索召回相关记忆 |
| 动态提示 | 每次请求动态生成 | 单次请求 | 注入当前时间 + 相关长期记忆 + 近期摘要 |

每条助手回复都会写入长期记忆。下次用户提问时，系统会自动检索与当前问题最相关的历史对话片段，注入到系统提示中。

这样 Agent 能"记住"之前聊过什么，即使重启服务也不会丢失。

---

## 环境要求

- **操作系统**：Windows 10/11（Linux/macOS 也行，启动脚本改一下）
- **GPU**：NVIDIA 显卡，显存 ≥ 6GB（RTX 3060 及以上推荐）
- **Python**：3.10

### 硬件参考

| 配置 | 说明 |
|------|------|
| GPU | NVIDIA RTX 5060 (8GB VRAM) |
| 模型 | Qwen2.5-7B Q4_K_M 量化（约 4.7GB） |
| 推理速度 | 约 60-70 tokens/s |
| 显存占用 | 模型 4.7GB + Embedding 0.5GB + Reranker 1GB ≈ 6.2GB |

---

## 快速开始

### 第一步：准备 Conda 环境

```bash
conda create -n agent python=3.10 -y
conda activate agent
```

### 第二步：安装依赖

```bash
pip install fastapi uvicorn openai langchain-community chromadb
pip install sentence-transformers rank-bm25 jieba numpy torch
```

### 第三步：下载模型

你需要 3 个模型文件：

| 模型 | 用途 | 大小 | 下载地址 |
|------|------|------|---------|
| Qwen2.5-7B-Instruct-Q4_K_M.gguf | LLM 推理 | ~4.7GB | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) |
| bge-small-zh-v1.5 | Embedding | ~90MB | [HuggingFace](https://huggingface.co/BAAI/bge-small-zh-v1.5) |
| bge-reranker-base | Reranker | ~1GB | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-base) |

把 BGE 模型放到 `local_models/` 目录下：

```
local_models/
├── bge-small-zh-v1.5/
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── config.json
└── bge-reranker-base/
    ├── model.safetensors
    ├── tokenizer.json
    └── config.json
```

### 第四步：启动 LLM 推理服务

```bash
# 启动 llama.cpp server（需要提前下载 llama-server.exe）
llama-server.exe -m path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 99 -c 8192 --port 8080
```

参数说明：
- `-ngl 99`：把所有层加载到 GPU
- `-c 8192`：上下文窗口 8192 tokens
- `--port 8080`：监听端口

看到 `HTTP server is listening` 就说明启动成功了。

### 第五步：启动 Web 服务

```bash
cd agent_study
uvicorn web_app:app --host 127.0.0.1 --port 8000
```

或者 Windows 下双击 `start_web.bat`。

首次启动会加载 Embedding 模型和 Reranker 模型，大约需要 10-20 秒。看到以下输出就说明就绪了：

```
[*] 紫微知识库加载成功（873 条 | 向量+BM25+Reranker）
[*] 长期记忆库加载成功
=============================================================
  Ziwei Agent v2.1  started
  URL: http://127.0.0.1:8000
=============================================================
```

### 第六步：打开浏览器

访问 http://127.0.0.1:8000 即可开始对话。

---

## 试一试

一些有趣的问题可以问：

- "天机星入命宫是什么意思？"
- "紫微斗数和八字有什么区别？"
- "命宫有天同星和太阴星，性格怎么样？"
- "化忌入财帛宫意味着什么？"
- "帮我分析一下事业宫有武曲星的影响"

你会看到 Agent 先思考、再搜索知识库、最后给出专业回答，整个推理过程在界面上实时展示。

---

## 学到了什么

这个项目是跟着 [Datawhale Hello-Agents](https://github.com/datawhalechina/hello-agents) 课程学习的实践成果，三天完成了从入门到完整项目。核心收获：

1. **ReAct 比普通 Prompt 好在哪** — 让 LLM 显式输出推理步骤，错误可观测、可纠正
2. **混合检索不是噱头** — 向量检索解决语义问题，BM25 解决精确匹配问题，Reranker 解决排序质量问题
3. **本地 LLM 没那么难** — llama.cpp + 量化模型 + OpenAI 兼容协议，7B 模型就能做很多事
4. **记忆系统是 Agent 的灵魂** — 没有记忆的 Agent 每次都是"失忆"的对话，有了三层记忆才能真正"认识"用户

---

## 后续优化方向

- [ ] 流式 token 输出（逐字显示而非等完整响应）
- [ ] 对话导出为 Markdown/PDF
- [ ] 支持多用户并发
- [ ] 知识库增量更新接口
- [ ] 接入更大参数的本地模型（Qwen2.5-14B / 32B）

---

*如果觉得有用，欢迎 star*
