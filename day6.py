from langchain_community.llms import CTransformers

from llm_engine import LocalLLMEngine

# 使用 LangChain 的 CTransformers 包装器
print("正在初始化 Agent 大脑...")
agent_llm = CTransformers(
    model=r"D:\models\qwen2.5-7b\qwen2.5-7b-instruct-q3_k_m.gguf",
    model_type="qwen",
    config={
        'max_new_tokens': 256,
        'temperature': 0.1,  # [核心重点] Agent 必须设为低温度，防止它乱编 JSON 格式
        'context_length': 2048,
        'gpu_layers': 35
    }
)

import json

# 模拟 Agent 的 System Prompt 说明书
agent_prompt = """你是一个紫微斗数专家系统的大脑。你拥有一个名为 `search_ziwei` 的工具，用于查询命理知识。

请分析用户的提问。如果需要查询知识库，请务必且**仅仅**输出以下 JSON 格式，不要输出任何额外的解释文本：
{
    "action": "search_ziwei",
    "query": "提取出的核心搜索词"
}

如果不需要查询，请直接回答用户。

用户提问：贪狼星在夫妻宫是什么意思？
你的输出："""

print("正在测试 Agent 的 JSON 输出能力...")
response = agent_llm(agent_prompt)
print(f"\n模型原始输出:\n{response}")

# 测试代码能否成功解析它
try:
    # 清洗可能存在的 markdown 代码块符号 (大模型常犯的错)
    clean_response = response.strip().replace("```json", "").replace("```", "")
    action_dict = json.loads(clean_response)
    print("\n[成功] 模型成功输出了标准的 JSON！")
    print(f"解析结果 -> 动作: {action_dict['action']}, 搜索词: {action_dict['query']}")
except json.JSONDecodeError:
    print("\n[失败] 模型输出的不是标准 JSON 格式，需要优化 Prompt 或使用正则提取。")