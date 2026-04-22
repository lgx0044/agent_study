# %% [markdown]
# # Day 6 真实演练：基于 Qwen-7B 的 ReAct Agent
# **依赖库**: re, json

import re
import json

# 假设你已经导入了你前两天写的类
from llm_engine import LocalLLMEngine
from advanced_rag import Advanced_RAG_Pipeline 

# 1. 初始化引擎和数据库
llm = LocalLLMEngine()
rag_system = Advanced_RAG_Pipeline(
    vector_db_path="./chroma_db_hybrid",
    embedding_model_path="./local_models/bge-small-zh-v1.5",
    reranker_model_path="./local_models/bge-reranker-base"
)

# 2. 定义工具说明书 (给 Qwen 看的)
TOOLS_DESC = """
可用工具列表:
[
  {
    "name": "search_ziwei",
    "description": "当用户询问紫微斗数、星曜、宫位、四化等命理知识时，必须使用此工具获取专业资料。",
    "parameters": {
      "query": "用于检索的关键词，例如 '贪狼星在命宫'"
    }
  }
]
"""

# 3. 构造极其严格的 ReAct System Prompt
REACT_SYSTEM_PROMPT = f"""你是一个紫微斗数专家。你拥有一个外部知识库工具。
面对用户的问题，你可以选择直接回答，或者使用工具。

{TOOLS_DESC}

请你务必严格按照以下格式思考和输出（不要输出其他无关废话）：

Thought: [思考你现在需要做什么]
Action: [只能是 search_ziwei 或者 None]
Action Input: [如果是 search_ziwei，请输出纯 JSON 格式的参数，如 {{"query": "..."}}]

(如果 Action 不是 None，你只需输出到 Action Input 即可，暂停输出等待观察结果)
"""

# 4. 正则解析器 (工业级容错，处理 Qwen 偶尔多加的 Markdown 符号)
def extract_action(text):
    # 提取 Action
    action_match = re.search(r"Action:\s*(.*?)\n", text)
    action = action_match.group(1).strip() if action_match else "None"
    
    # 提取 JSON (处理可能带有 ```json 的情况)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if not input_match:
        # 尝试匹配 markdown 块
        input_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        
    action_input = input_match.group(1).strip() if input_match else "{}"
    return action, action_input

# 5. Agent 主循环 (State Machine)
def run_react_agent(question, max_iterations=3):
    print(f"\n👨‍💼 用户: {question}")
    
    # 初始对话历史
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    for step in range(max_iterations):
        print(f"\n--- [第 {step+1} 轮] ---")
        # 1. 呼叫大模型
        response = llm.client.chat.completions.create(
            model="qwen2.5-7b",
            messages=messages,
            temperature=0.1 # 工具调用阶段，温度调低保证 JSON 稳定
        ).choices[0].message.content
        
        # 2. 把模型的思考存入历史
        messages.append({"role": "assistant", "content": response})
        
        # 3. 解析模型意图
        action, action_input = extract_action(response)
        
        # 4. 路由逻辑 (工具执行)
        if "search_ziwei" in action:
            try:
                args = json.loads(action_input)
                query = args.get("query", question)
                print(f"� 检索中: {query}")
                
                # 调用你的 RAG 获取资料
                _, context_docs = rag_system.query(query)
                observation = "\n".join([doc.page_content for doc in context_docs])
                
                print(f"✅ 检索完成: 找到 {len(context_docs)} 条相关资料")
                
                # 喂给大模型
                messages.append({
                    "role": "user", 
                    "content": f"Observation: {observation}\n请根据以上资料给出 Final Answer (最终回答)。"
                })
                
            except Exception as e:
                print(f"❌ 检索失败: {e}")
                messages.append({"role": "user", "content": "Observation: 工具调用失败，请直接根据你的常识回答。"})
                
        else:
            print("\n🏁 任务结束")
            # 打印最终回答
            final_response = llm.client.chat.completions.create(
                model="qwen2.5-7b",
                messages=messages,
                temperature=0.7
            ).choices[0].message.content
            print(f"\n📝 最终回答:\n{final_response}")
            break

# ================= 运行测试 =================
if __name__ == "__main__":
    run_react_agent("帮我查一下紫微星在财帛宫是什么意思？")
# %%
