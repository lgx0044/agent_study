# ==============================================================================
# 紫微斗数 Agent - FastAPI 服务层（完整优化版）
# 运行: uvicorn web_app:app --reload --port 8000
# ==============================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import re
import time
import asyncio
from typing import AsyncGenerator, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch
from datetime import datetime
import uuid

# 线程池
executor = ThreadPoolExecutor(max_workers=4)

# ==========================================
# 配置
# ==========================================
LLM_BASE_URL = os.getenv("LLM_URL", "http://127.0.0.1:8080/v1")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_PATH", "./local_models/bge-small-zh-v1.5")
ZIWEI_DB_DIR = "./chroma_db_hybrid"
LONG_TERM_DB_DIR = "./long_term_memory_db"
DATA_DIR = Path("./data/sessions")
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="紫微斗数 Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 初始化
# ==========================================
print("[*] 正在初始化紫微斗数 Agent v2.0...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] 使用设备: {device}")

llm_client = OpenAI(base_url=LLM_BASE_URL, api_key="sk-no-key")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={'device': device}
)

ziwei_db = None
ziwei_retriever = None
try:
    ziwei_db = Chroma(persist_directory=ZIWEI_DB_DIR, embedding_function=embeddings)
    ziwei_retriever = ziwei_db.as_retriever(search_kwargs={"k": 3})
    print("[*] 紫微知识库加载成功")
except Exception as e:
    print(f"[!] 知识库加载失败: {e}")

long_term_db = None
try:
    long_term_db = Chroma(
        collection_name="user_memory",
        persist_directory=LONG_TERM_DB_DIR,
        embedding_function=embeddings
    )
    print("[*] 长期记忆库加载成功")
except Exception as e:
    print(f"[!] 长期记忆库加载失败: {e}")

# ==========================================
# 数据模型
# ==========================================
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class Message(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    thought: Optional[str] = None
    tool: Optional[str] = None
    observation: Optional[str] = None

class Session(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list = []

# ==========================================
# 会话管理（持久化版）
# ==========================================
def get_session_path(session_id: str) -> Path:
    return DATA_DIR / f"{session_id}.json"

def load_session(session_id: str) -> Session:
    path = get_session_path(session_id)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Session(**data)
        except:
            pass
    return Session(
        id=session_id,
        title="新会话",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        messages=[]
    )

def save_session(session: Session):
    session.updated_at = datetime.now().isoformat()
    path = get_session_path(session.id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(session.dict(), f, ensure_ascii=False, indent=2)

def get_all_sessions() -> list:
    sessions = []
    for path in DATA_DIR.glob("*.json"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sessions.append({
                    "id": data["id"],
                    "title": data["title"],
                    "updated_at": data["updated_at"]
                })
        except:
            pass
    return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

# 内存缓存
session_cache = {}

def get_session(session_id: str) -> Session:
    if session_id not in session_cache:
        session_cache[session_id] = load_session(session_id)
    return session_cache[session_id]

# ==========================================
# Agent 核心逻辑
# ==========================================
SYSTEM_PROMPT = """你是一个专业的紫微斗数命理大师。你拥有一个外部知识库工具：
- search_ziwei: 必须用于查询命理、星曜、宫位相关的专业知识。参数格式：{"query": "搜索词"}

你必须严格按照以下格式思考和输出：

Thought: [你的思考过程]
Action: [search_ziwei 或者 None]
Action Input: [纯 JSON 格式的搜索参数]

如果不需要调用工具，请输出：
Thought: [思考过程]
Final Answer: [最终给用户的自然语言回答]

注意：
1. 调用工具时，Action Input 必须是合法的 JSON
2. 回答要专业、有深度，结合紫微斗数理论"""

def query_knowledge(query: str) -> str:
    """查询紫微知识库"""
    if ziwei_retriever is None:
        return "知识库未加载"
    docs = ziwei_retriever.invoke(query)
    return "\n".join([d.page_content for d in docs[:3]])

def get_dynamic_system_prompt(session: Session, current_query: str) -> str:
    """动态组装系统提示"""
    today = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    
    long_term_str = "无"
    if long_term_db:
        try:
            related = long_term_db.similarity_search(current_query, k=1)
            long_term_str = related[0].page_content if related else "无"
        except:
            pass
    
    # 从消息历史中提取摘要
    recent_summary = ""
    if session.messages:
        last_msgs = session.messages[-4:]
        recent_summary = " | ".join([f"{m.role}: {m.content[:50]}..." for m in last_msgs if len(m.content) > 50])

    return f"""{SYSTEM_PROMPT}

【当前时间】：{today}

【记忆中枢】：
- 长期记忆：{long_term_str[:200]}
- 近期摘要：{recent_summary or "暂无"}

请结合记忆和专业知识回答用户。"""

async def stream_llm(messages, temperature=0.1, max_tokens=512):
    """流式调用 LLM，逐 token 输出"""
    loop = asyncio.get_event_loop()
    
    def _stream():
        try:
            response = llm_client.chat.completions.create(
                model="qwen2.5-7b",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            full_content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_content += token
                    yield token
        except Exception as e:
            yield f"[ERROR: {str(e)}]"
    
    # 在线程池中运行
    for token in await loop.run_in_executor(executor, lambda: list(_stream())):
        yield token

async def call_llm_sync(messages, temperature=0.1, stop=None, max_tokens=512):
    """同步调用 LLM"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: llm_client.chat.completions.create(
            model="qwen2.5-7b",
            messages=messages,
            temperature=temperature,
            stop=stop,
            max_tokens=max_tokens
        ).choices[0].message.content
    )

async def run_agent_stream(session: Session, user_input: str) -> AsyncGenerator[str, None]:
    """流式执行 Agent"""
    loop = asyncio.get_event_loop()
    
    # 1. 组装上下文
    dynamic_system = get_dynamic_system_prompt(session, user_input)
    messages = [{"role": "system", "content": dynamic_system}]
    
    # 添加历史消息（最近6条）
    for msg in session.messages[-6:]:
        messages.append({"role": msg.role, "content": msg.content})
    
    messages.append({"role": "user", "content": user_input})
    
    # 创建消息ID
    msg_id = str(uuid.uuid4())[:8]
    
    # 发送开始事件
    yield f"data: {json.dumps({'type': 'start', 'id': msg_id, 'timestamp': datetime.now().isoformat()}, ensure_ascii=False)}\n\n"
    
    # 2. ReAct 循环
    max_iterations = 3
    full_assistant_msg = ""
    thought_content = ""
    tool_name = ""
    observation_content = ""
    final_answer = ""
    
    for step in range(max_iterations):
        # 调用 LLM
        try:
            assistant_msg = await call_llm_sync(messages, temperature=0.1, stop=["Observation:", "观察结果:"])
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'LLM 调用失败: {str(e)}'}, ensure_ascii=False)}\n\n"
            return
        
        full_assistant_msg = assistant_msg
        
        # 解析 Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", assistant_msg, re.DOTALL)
        if thought_match:
            thought_content = thought_match.group(1).strip()
            yield f"data: {json.dumps({'type': 'thought', 'content': thought_content}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)  # 小延迟让前端有时间处理
        
        messages.append({"role": "assistant", "content": assistant_msg})
        
        # 解析 Action
        action_match = re.search(r"Action:\s*(.*?)\n", assistant_msg)
        action = action_match.group(1).strip() if action_match else "None"
        
        # 退出条件
        if action == "None" or "Final Answer" in assistant_msg:
            # 提取最终回答
            final_match = re.search(r"Final Answer:\s*(.*)", assistant_msg, re.DOTALL)
            final_answer = final_match.group(1).strip() if final_match else assistant_msg
            
            # 流式输出最终答案
            yield f"data: {json.dumps({'type': 'answer_start'}, ensure_ascii=False)}\n\n"
            
            # 逐字输出（打字机效果）
            for char in final_answer:
                yield f"data: {json.dumps({'type': 'answer_token', 'content': char}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.02)
            
            yield f"data: {json.dumps({'type': 'answer_end', 'content': final_answer}, ensure_ascii=False)}\n\n"
            break
        
        # 解析 Action Input
        input_match = re.search(r"Action Input:\s*(\{.*?\})", assistant_msg, re.DOTALL)
        action_input_str = input_match.group(1).strip() if input_match else "{}"
        
        # 执行工具
        if "search_ziwei" in action:
            tool_name = "search_ziwei"
            yield f"data: {json.dumps({'type': 'tool', 'name': tool_name, 'query': action_input_str}, ensure_ascii=False)}\n\n"
            
            try:
                args = json.loads(action_input_str)
                query = args.get("query", user_input)
                observation_content = await loop.run_in_executor(executor, lambda: query_knowledge(query))
                
                # 显示检索摘要
                obs_summary = observation_content[:150] + "..." if len(observation_content) > 150 else observation_content
                yield f"data: {json.dumps({'type': 'observation', 'content': obs_summary, 'full_length': len(observation_content)}, ensure_ascii=False)}\n\n"
            except json.JSONDecodeError as e:
                observation_content = f"JSON 解析失败: {str(e)}"
                yield f"data: {json.dumps({'type': 'error', 'content': 'JSON 格式错误'}, ensure_ascii=False)}\n\n"
            
            messages.append({
                "role": "user",
                "content": f"Observation: {observation_content}\n请基于以上资料生成 Final Answer。"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Observation: 工具 {action} 不存在。"
            })
    else:
        # 超时
        final_answer = "推演超时，请重新提问。"
        yield f"data: {json.dumps({'type': 'error', 'content': '推演超时'}, ensure_ascii=False)}\n\n"
    
    # 3. 保存消息
    msg = Message(
        id=msg_id,
        role="assistant",
        content=final_answer,
        timestamp=datetime.now().isoformat(),
        thought=thought_content if thought_content else None,
        tool=tool_name if tool_name else None,
        observation=observation_content[:200] if observation_content else None
    )
    session.messages.append(msg)
    
    # 用户消息
    user_msg = Message(
        id=str(uuid.uuid4())[:8],
        role="user",
        content=user_input,
        timestamp=datetime.now().isoformat()
    )
    session.messages.append(user_msg)
    
    # 更新标题（如果是第一条消息）
    if len(session.messages) == 2:
        session.title = user_input[:20] + ("..." if len(user_input) > 20 else "")
    
    save_session(session)
    
    # 更新长期记忆（如果会话很长）
    if len(session.messages) > 10 and long_term_db:
        try:
            text = f"User: {user_input}\nAssistant: {final_answer}"
            await loop.run_in_executor(executor, lambda: long_term_db.add_documents([Document(page_content=text)]))
        except:
            pass
    
    yield "data: [DONE]\n\n"

# ==========================================
# API 路由
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/sessions")
async def list_sessions():
    """列出所有会话"""
    return {"sessions": get_all_sessions()}

@app.post("/api/sessions")
async def create_session():
    """创建新会话"""
    session_id = str(uuid.uuid4())[:8]
    session = Session(
        id=session_id,
        title="新会话",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        messages=[]
    )
    save_session(session)
    session_cache[session_id] = session
    return {"session_id": session_id}

@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """获取会话详情"""
    session = get_session(session_id)
    return session.dict()

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    path = get_session_path(session_id)
    if path.exists():
        path.unlink()
    if session_id in session_cache:
        del session_cache[session_id]
    return {"status": "ok"}

@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "md"):
    """导出会话"""
    session = get_session(session_id)
    
    if format == "json":
        return JSONResponse(content=session.dict())
    
    # Markdown 格式
    lines = [f"# {session.title}\n", f"创建时间: {session.created_at}\n\n---\n\n"]
    
    for msg in session.messages:
        time = msg.timestamp.split("T")[1][:8] if "T" in msg.timestamp else msg.timestamp
        if msg.role == "user":
            lines.append(f"**用户** ({time}):\n{msg.content}\n\n")
        else:
            lines.append(f"**助手** ({time}):\n")
            if msg.thought:
                lines.append(f"> 💭 {msg.thought}\n\n")
            if msg.tool:
                lines.append(f"> 🔧 工具: {msg.tool}\n\n")
            lines.append(f"{msg.content}\n\n---\n\n")
    
    return JSONResponse(
        content={"content": "".join(lines), "filename": f"{session.title}.md"}
    )

@app.get("/api/chat/stream")
async def chat_stream(message: str, session_id: str = "default"):
    """流式对话接口"""
    session = get_session(session_id)
    
    return StreamingResponse(
        run_agent_stream(session, message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """非流式对话接口"""
    session = get_session(request.session_id)
    
    dynamic_system = get_dynamic_system_prompt(session, request.message)
    messages = [{"role": "system", "content": dynamic_system}]
    for msg in session.messages[-6:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})
    
    try:
        answer = await call_llm_sync(messages, temperature=0.7)
    except Exception as e:
        return {"error": str(e)}
    
    # 保存
    session.messages.append(Message(
        id=str(uuid.uuid4())[:8],
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat()
    ))
    session.messages.append(Message(
        id=str(uuid.uuid4())[:8],
        role="assistant",
        content=answer,
        timestamp=datetime.now().isoformat()
    ))
    save_session(session)
    
    return {"response": answer}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm": LLM_BASE_URL,
        "knowledge_db": ziwei_db is not None,
        "memory_db": long_term_db is not None,
        "version": "2.0.0"
    }

# ==========================================
# 启动
# ==========================================
print("""
╔══════════════════════════════════════════════════════════════╗
║  🔮 紫微斗数 Agent v2.0 已启动                                 ║
║  访问: http://127.0.0.1:8000                                 ║
║  新特性: 流式输出 | 会话持久化 | 打字机效果 | 多会话管理        ║
╚══════════════════════════════════════════════════════════════╝
""")
