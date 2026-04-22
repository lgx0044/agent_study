# ==============================================================================
# 终极版：紫微斗数 AI Agent (带三层记忆、RAG 检索、自愈 ReAct)
# 依赖环境: pip install openai langchain langchain-community sentence-transformers chromadb
# ==============================================================================

import os
# 解决OpenMP库重复初始化的问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import json
import time
import torch
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def get_device():
    """根据实际硬件情况选择设备"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# [配置区] 请确保这些路径与你本地的环境一致
# ==========================================
LLM_BASE_URL = "http://127.0.0.1:8080/v1"  # 你的 llama-server 地址
LLM_MODEL_NAME = "qwen2.5-7b"              # 模型名称
EMBEDDING_MODEL_PATH = "./local_models/bge-small-zh-v1.5" # 你的 embedding 模型路径
LONG_TERM_DB_DIR = "./long_term_memory_db" # 长期记忆数据库持久化路径
ZIWEI_DB_DIR = "./chroma_db_hybrid"        # 你的紫微斗数 RAG 数据库路径

# 初始化全局 LLM 客户端
llm_client = OpenAI(base_url=LLM_BASE_URL, api_key="sk-no-key-required")

def call_llm(messages, temperature=0.7, stop=None):
    """封装大模型调用，统一处理流式和非流式"""
    response = llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=messages,
        temperature=temperature,
        stop=stop
    )
    return response.choices[0].message.content

# ==========================================
# 1. 知识库 RAG 引擎 (真实业务查询)
# ==========================================
class ZiweiKnowledgeBase:
    def __init__(self):
        print("[*] 正在加载紫微斗数 RAG 知识库...")
        device = get_device()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH, 
            model_kwargs={'device': device}
        )
        self.vector_db = Chroma(
            persist_directory=ZIWEI_DB_DIR, 
            embedding_function=self.embeddings
        )
        # 这里为了精简主流程，使用基础向量检索。
        # 你可以随时把它替换为你 Day 5 写的 EnsembleRetriever + Rerank 混合检索代码
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

    def query(self, search_text):
        docs = self.retriever.invoke(search_text)
        return [doc.page_content for doc in docs]

# ==========================================
# 2. 三层记忆中枢 (Memory Hub)
# ==========================================
class AgentMemoryHub:
    def __init__(self, short_term_max=3, summary_trigger=5):
        print("[*] 正在挂载 Agent 三层记忆中枢...")
        self.short_term_buffer = [] 
        self.max_rounds = short_term_max
        self.summary_context = "暂无历史摘要。"
        self.summary_trigger_count = summary_trigger
        self.total_turns = 0
        
        # 挂载长期记忆向量库
        device = get_device()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH, 
            model_kwargs={'device': device}
        )
        self.long_term_db = Chroma(
            collection_name="user_memory", 
            persist_directory=LONG_TERM_DB_DIR, 
            embedding_function=self.embeddings
        )

    def save_chat(self, user_input, final_answer):
        """保存提问与回答，隔离内部演算"""
        self.total_turns += 1
        self.short_term_buffer.append({"role": "user", "content": user_input})
        self.short_term_buffer.append({"role": "assistant", "content": final_answer})
        
        # 滑动窗口：溢出部分存入长期记忆
        if len(self.short_term_buffer) > self.max_rounds * 2:
            old_user = self.short_term_buffer.pop(0)
            old_assistant = self.short_term_buffer.pop(0)
            
            text_to_save = f"User: {old_user['content']}\nAssistant: {old_assistant['content']}"
            self.long_term_db.add_documents([Document(page_content=text_to_save)])
            
            # 定期触发摘要压缩
            if self.total_turns % self.summary_trigger_count == 0:
                print("\n[🧠 记忆系统] 正在进行记忆碎片整理与摘要压缩...")
                prompt = f"请简要总结以下对话的核心事实（如用户属性、命理特征）：\n原摘要：{self.summary_context}\n新记录：{str(self.short_term_buffer)}"
                self.summary_context = call_llm([{"role": "user", "content": prompt}], temperature=0.3)

    def get_dynamic_system_prompt(self, base_prompt, current_query):
        """动态组装：系统指令 + 长期记忆 + 摘要"""
        try:
            related_history = self.long_term_db.similarity_search(current_query, k=1)
            long_term_str = related_history[0].page_content if related_history else "无"
        except Exception:
            long_term_str = "无"
            
        return f"""{base_prompt}

【记忆中枢】(若与当前问题冲突，以用户最新说法为准)：
- 长期深层记忆关联：{long_term_str}
- 近期历史摘要：{self.summary_context}

请结合上述记忆和你的专业知识回答用户。
"""

    def get_short_term_messages(self):
        return list(self.short_term_buffer)

# ==========================================
# 3. 核心 ReAct 引擎 (带自愈机制)
# ==========================================
def run_react_engine(initial_messages, rag_system, max_iterations=4):
    """执行 Thought -> Action -> Observation 循环"""
    working_messages = list(initial_messages) 
    
    for step in range(max_iterations):
        # 呼叫大模型，设置 stop 词为 Observation: 防止大模型自导自演
        response = call_llm(
            working_messages, 
            temperature=0.1, 
            stop=["Observation:", "观察结果:"]
        )
        
        print(f"🤖 [内部推演 {step+1}]:\n{response}\n")
        working_messages.append({"role": "assistant", "content": response})
        
        # 提取 Action
        action_match = re.search(r"Action:\s*(.*?)\n", response)
        action = action_match.group(1).strip() if action_match else "None"
        
        # 退出条件：不需要工具，或者已经给出 Final Answer
        if action == "None" or "Final Answer" in response:
            return response
            
        # 提取 Action Input (JSON)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)
        action_input_str = input_match.group(1).strip() if input_match else "{}"
        
        # 路由到真实工具
        if "search_ziwei" in action:
            try:
                args = json.loads(action_input_str)
                query = args.get("query", "")
                print(f"🛠️ [检索调用] 正在数据库中查询: {query}")
                
                # 调用真实的 RAG 知识库
                docs = rag_system.query(query)
                observation = "\n".join(docs) if docs else "未找到相关资料。"
                
                working_messages.append({
                    "role": "user", 
                    "content": f"Observation: 检索到以下资料：\n{observation}\n请基于以上资料生成 Final Answer。"
                })
                
            except json.JSONDecodeError as e:
                # JSON 自愈逻辑
                print(f"⚠️ [触发自愈] 检测到模型 JSON 格式错误，正在引导修复...")
                working_messages.append({
                    "role": "user", 
                    "content": f"Observation: Action Input JSON 解析失败 ({str(e)})。请检查是否遗漏逗号或括号，并重新输出格式正确的 Thought 和 Action。"
                })
                continue
        else:
            # 幻觉工具自愈
            working_messages.append({"role": "user", "content": f"Observation: 工具 {action} 不存在，只能使用 search_ziwei。"})
            continue

    return "Final Answer: 抱歉，问题过于复杂，推演超时，请换个方式提问。"

# ==========================================
# 4. 终端会话 Orchestrator (入口)
# ==========================================
def main():
    print("\n" + "="*70)
    print("🔮 紫微斗数 Agent 终极版已启动 (内核: Qwen-7B + ChromaDB)")
    print("="*70)
    
    # 实例化真实组件
    rag_system = ZiweiKnowledgeBase()
    memory_hub = AgentMemoryHub(short_term_max=3, summary_trigger=5)
    
    BASE_SYSTEM_PROMPT = """你是一个专业的紫微斗数命理大师。你拥有一个外部工具：
- search_ziwei: 必须用于查询命理、星曜、宫位相关的专业知识。参数格式：{"query": "具体的搜索词"}

你必须严格按照以下格式思考和输出：
Thought: [你的思考过程]
Action: [search_ziwei 或者 None]
Action Input: [纯 JSON 格式的搜索参数]

如果不需要调用工具，或者已经有了答案，请输出：
Thought: [思考过程]
Final Answer: [最终给用户的自然语言回答]"""

    while True:
        user_input = input("\n🧑‍💻 访客: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("大师：有缘再见！")
            break
        if not user_input: continue
            
        # 1. 向记忆中枢请求动态 System Prompt
        dynamic_system = memory_hub.get_dynamic_system_prompt(BASE_SYSTEM_PROMPT, user_input)
        
        # 2. 组装对话上下文
        messages_to_agent = [{"role": "system", "content": dynamic_system}]
        messages_to_agent.extend(memory_hub.get_short_term_messages())
        messages_to_agent.append({"role": "user", "content": user_input})
        
        print("\n⏳ 大师正在推演命盘...")
        
        # 3. 启动 ReAct 引擎进行深度思考与查库
        start_time = time.time()
        final_raw_answer = run_react_engine(messages_to_agent, rag_system)
        
        # 4. 提取出干净的最终回答给用户看
        display_answer = re.sub(r"Thought:.*?Final Answer:\s*", "", final_raw_answer, flags=re.DOTALL).strip()
        
        print(f"\n✨ 大师回答 ({time.time() - start_time:.1f}s): {display_answer}")
        
        # 5. 将对话落库，进行数据隔离
        memory_hub.save_chat(user_input, display_answer)

if __name__ == "__main__":
    main()