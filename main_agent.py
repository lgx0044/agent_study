import os
# 解决OpenMP库重复初始化的问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import json
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 导入并实例化 llm_engine 和 rag_system
from llm_engine import LocalLLMEngine
from advanced_rag import Advanced_RAG_Pipeline

# 初始化引擎和数据库
llm = LocalLLMEngine()
rag_system = Advanced_RAG_Pipeline(
    vector_db_path="./chroma_db_hybrid",
    embedding_model_path="./local_models/bge-small-zh-v1.5",
    reranker_model_path="./local_models/bge-reranker-base"
)

# ==========================================
# 1. 记忆中枢 (Memory Hub)
# ==========================================
class AgentMemoryHub:
    def __init__(self, short_term_max=3, summary_trigger=5):
        self.short_term_buffer = [] 
        self.max_rounds = short_term_max
        self.summary_context = "暂无历史摘要。"
        self.summary_trigger_count = summary_trigger
        self.total_turns = 0
        
        print("[*] 正在挂载长期记忆向量库...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(model_name="./local_models/bge-small-zh-v1.5", model_kwargs={'device': device})
        self.vector_db = Chroma(collection_name="ziwei_long_term", persist_directory="./long_term_db", embedding_function=self.embeddings)

    def save_chat(self, user_input, final_answer, llm_engine):
        """💥 数据隔离：只接收纯净的最终对话，拒绝 ReAct 中间态"""
        self.total_turns += 1
        
        self.short_term_buffer.append({"role": "user", "content": user_input})
        self.short_term_buffer.append({"role": "assistant", "content": final_answer})
        
        # 触发滑动窗口：挤出最老的数据到长期记忆
        if len(self.short_term_buffer) > self.max_rounds * 2:
            old_user = self.short_term_buffer.pop(0)
            old_assistant = self.short_term_buffer.pop(0)
            
            # 落盘到长期记忆
            text_to_save = f"User: {old_user['content']}\nAssistant: {old_assistant['content']}"
            self.vector_db.add_documents([Document(page_content=text_to_save, metadata={"type": "history"})])
            
            # 触发摘要压缩
            if self.total_turns % self.summary_trigger_count == 0:
                print("\n[🧠 记忆系统] 正在进行记忆碎片整理与摘要生成...")
                prompt = f"请简要总结以下对话的核心事实（如用户属性、命理特征），忽略寒暄：\n原摘要：{self.summary_context}\n新记录：{str(self.short_term_buffer)}"
                # 调用大模型生成摘要 (不流式输出)
                self.summary_context = llm_engine.chat(prompt, system_msg="你是一个记忆总结员。", stream=False)

    def get_dynamic_system_prompt(self, base_prompt, current_query):
        """🧠 根据当前问题，动态组装三层记忆上下文"""
        # 从长期记忆中捞取相关历史
        related_history = self.vector_db.similarity_search(current_query, k=1)
        long_term_str = related_history[0].page_content if related_history else "无"
        
        dynamic_prompt = f"""{base_prompt}

【记忆中枢】(若与当前问题冲突，以用户最新说法为准)：
- 长期深层记忆关联：{long_term_str}
- 近期历史摘要：{self.summary_context}

请结合上述记忆和你的专业知识回答用户。
"""
        return dynamic_prompt

    def get_short_term_messages(self):
        """返回短期记忆的副本，防止被 ReAct 污染"""
        return list(self.short_term_buffer)


# ==========================================
# 2. 带有自愈机制的 ReAct 引擎 (独立运行，不污染记忆)
# ==========================================
def run_react_engine(initial_messages, llm_client, rag_system, max_iterations=4):
    """
    initial_messages: 包含 Dynamic System Prompt + Short-term Memory + Current Query
    """
    # 复制一份用于内部演算
    working_messages = list(initial_messages) 
    
    for step in range(max_iterations):
        # 呼叫大模型 (使用 stop 词打断)
        response = llm_client.chat.completions.create(
            model="qwen2.5-7b",
            messages=working_messages,
            temperature=0.1,
            stop=["Observation:", "观察结果:"] 
        ).choices[0].message.content
        
        print(f"🤖 [内部推演]:\n{response}")
        working_messages.append({"role": "assistant", "content": response})
        
        action_match = re.search(r"Action:\s*(.*?)\n", response)
        action = action_match.group(1).strip() if action_match else "None"
        
        # 退出条件
        if action == "None" or "Final Answer" in response:
            return response
            
        # 解析与自愈
        input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)
        action_input_str = input_match.group(1).strip() if input_match else "{}"
        
        if "search_ziwei" in action:
            try:
                args = json.loads(action_input_str)
                query = args.get("query", "")
                print(f"🛠️ [检索调用]: {query}")
                
                _, docs = rag_system.query(query)
                observation = "\n".join([d.page_content for d in docs])
                working_messages.append({
                    "role": "user", 
                    "content": f"Observation: {observation}\n请基于以上资料生成 Final Answer。"
                })
            except json.JSONDecodeError as e:
                print(f"⚠️ [触发自愈]: 修复损坏的 JSON格式...")
                working_messages.append({"role": "user", "content": f"Observation: JSON 解析失败 {str(e)}。请修正语法。"})
                continue
        else:
            working_messages.append({"role": "user", "content": f"Observation: 工具 {action} 不存在。"})
            continue

    return "抱歉，系统推演超时，请换个方式提问。"


# ==========================================
# 3. 终极主循环 (Orchestrator)
# ==========================================
def main():
    print("="*60)
    print("🔮 紫微斗数 Agent 已启动 (三层记忆引擎 + 自愈机制)")
    print("="*60)
    
    memory_hub = AgentMemoryHub(short_term_max=3, summary_trigger=5)
    
    BASE_SYSTEM_PROMPT = """你是一个紫微斗数大师。你拥有一个外部工具：
- search_ziwei: 必须用于查询生僻的命理概念。参数格式：{"query": "搜索词"}
你必须按照 Thought -> Action -> Action Input 的格式输出。如果不需要工具，输出 Action: None。"""

    while True:
        user_input = input("\n🧑‍💻 访客: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input: continue
            
        # 1. 向记忆中枢请求：带有长期与摘要记忆的 System Prompt
        dynamic_system = memory_hub.get_dynamic_system_prompt(BASE_SYSTEM_PROMPT, user_input)
        
        # 2. 组装发给大模型的上下文：System + 短期记录 + 此次提问
        messages_to_agent = [{"role": "system", "content": dynamic_system}]
        messages_to_agent.extend(memory_hub.get_short_term_messages())
        messages_to_agent.append({"role": "user", "content": user_input})
        
        # 3. 将干净的上下文送入 ReAct 引擎进行推演
        print("\n⏳ 正在深度推演...")
        final_answer = run_react_engine(messages_to_agent, llm, rag_system)
        
        # 清理最终答案前的废话 (把 Thought 抹掉，只给用户看答案)
        display_answer = re.sub(r"Thought:.*?Final Answer:\s*", "", final_answer, flags=re.DOTALL).strip()
        print(f"\n🤖 大师: {display_answer}")
        
        # 4. 💥 将纯净的提问和回答存入记忆中枢
        memory_hub.save_chat(user_input, display_answer, llm)

if __name__ == "__main__":
    main()