import os
import torch
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage

# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. 定义工具 (Tools) 
# ==========================================
@tool
def search_ziwei(query: str) -> str:
    """当用户询问命理、紫微斗数、星曜、宫位时，使用此工具检索知识库。"""
    print(f"\n[🛠️ 节点执行: 工具调用] 正在检索 -> {query}")
    return f"检索结果：{query} 是一颗重要的星曜，具体表现因宫位而异。"

tools = [search_ziwei]
tool_node = ToolNode(tools)

# ==========================================
# 2. 初始化环境 (大模型 & 长期记忆)
# ==========================================
print("[*] 正在加载 Qwen 大模型与 Chroma 长期记忆引擎...")
llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-none",
    model="qwen2.5-7b",
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)

# 初始化专门用于做摘要的模型实例（不带工具，防止它乱调）
llm_for_summary = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1", api_key="sk-none", model="qwen2.5-7b", temperature=0.3
)

EMBEDDING_MODEL_PATH = "./local_models/bge-small-zh-v1.5"
LONG_TERM_DB_DIR = "./long_term_memory_db"

try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, model_kwargs={'device': get_device()})
    long_term_db = Chroma(
        collection_name="user_memory", 
        persist_directory=LONG_TERM_DB_DIR, 
        embedding_function=embeddings
    )
    print("[OK] 环境初始化成功！")
except Exception as e:
    print(f"[ERROR] 长期记忆初始化失败: {e}")
    long_term_db = None

# ==========================================
# 3. 定义全局状态 (AgentState) - 三层记忆的载体
# ==========================================
class AgentState(TypedDict):
    # 短期记忆：自动追加新消息
    messages: Annotated[list, add_messages]
    # 摘要记忆：存储过往对话的压缩总结
    summary: str
    # 长期记忆：当前回合检索到的相关历史
    long_term_context: str

# ==========================================
# 4. 定义节点 (Nodes)
# ==========================================
def memory_retrieval_node(state: AgentState):
    """【节点 1】: 提取长期记忆"""
    print("\n[🧠 节点执行: 记忆检索] 正在潜入潜意识...")
    long_term_context = ""
    latest_user_msg = state["messages"][-1] 
    
    if long_term_db and isinstance(latest_user_msg, HumanMessage):
        try:
            related_memories = long_term_db.similarity_search(latest_user_msg.content, k=1)
            if related_memories:
                long_term_context = related_memories[0].page_content
                print(f"[OK] 唤醒相关长期记忆: {long_term_context[:20]}...")
        except Exception:
            pass
            
    return {"long_term_context": long_term_context}

def oracle_node(state: AgentState):
    """【节点 2】: 大脑推演"""
    print("\n[🧠 节点执行: 大脑思考]...")
    
    # 动态组装三层记忆作为 System Prompt
    sys_prompt = "你是一个专业的紫微斗数大师。请结合以下用户的历史记忆进行对话：\n"
    if state.get("summary"):
        sys_prompt += f"\n【背景摘要】: {state['summary']}"
    if state.get("long_term_context"):
        sys_prompt += f"\n【深层记忆】: {state['long_term_context']}"
        
    messages_for_llm = [{"role": "system", "content": sys_prompt}] + state["messages"]
    
    response = llm_with_tools.invoke(messages_for_llm)
    return {"messages": [response]}

def memory_management_node(state: AgentState):
    """【节点 3】: 记忆落盘与碎片整理 (极其关键的节点)"""
    print("\n[🧠 节点执行: 记忆整理] 落盘与摘要压缩...")
    messages = state["messages"]
    updates = {}
    
    # 1. 保存到长期记忆 (落盘)
    if len(messages) >= 2 and long_term_db:
        # 取倒数第二条(Human)和最后一条(AI)
        if isinstance(messages[-2], HumanMessage) and isinstance(messages[-1], AIMessage):
            if not messages[-1].tool_calls: # 过滤掉中间调用工具的消息
                try:
                    memory_str = f"User: {messages[-2].content}\nAI: {messages[-1].content}"
                    long_term_db.add_documents([Document(page_content=memory_str)])
                    print("[OK] 当前对话已刻入长期记忆库(ChromaDB)。")
                except Exception as e:
                    print(f"落盘失败: {e}")

    # 2. 触发摘要记忆与短期截断 (滑动窗口)
    # 假设我们最多只保留 6 条消息（3轮对话），超过则进行压缩
    if len(messages) > 6:
        # 提取最老的两条消息（一问一答）准备压缩
        msgs_to_summarize = messages[0:2]
        old_dialogue = f"User: {msgs_to_summarize[0].content}\nAI: {msgs_to_summarize[1].content}"
        
        print("[⚡ 触发机制] 短期记忆溢出，正在生成摘要...")
        current_summary = state.get("summary", "暂无摘要")
        summary_prompt = f"请简要总结以下对话的核心事实。若已有摘要，请融合更新。\n已有摘要：{current_summary}\n新增对话：{old_dialogue}"
        
        # 呼叫无工具的大模型生成新摘要
        new_summary_response = llm_for_summary.invoke(summary_prompt)
        updates["summary"] = new_summary_response.content
        print(f"[✨ 摘要更新]: {updates['summary']}")
        
        # 💥 核心魔法：使用 RemoveMessage 从状态图中删掉最老的两条消息，释放显存！
        updates["messages"] = [RemoveMessage(id=m.id) for m in msgs_to_summarize]

    return updates

# ==========================================
# 5. 路由逻辑 (Routers)
# ==========================================
def route_after_oracle(state: AgentState):
    """判断大脑思考后的流向"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return "memory_management" # 生成自然语言后，去整理记忆

# ==========================================
# 6. 编排终极图网络
# ==========================================
print("[*] 正在装配三层记忆图网络...")
graph_builder = StateGraph(AgentState)

graph_builder.add_node("memory_retrieval", memory_retrieval_node)
graph_builder.add_node("oracle", oracle_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("memory_management", memory_management_node)

graph_builder.add_edge(START, "memory_retrieval")
graph_builder.add_edge("memory_retrieval", "oracle")

graph_builder.add_conditional_edges(
    "oracle", 
    route_after_oracle,
    {"tools": "tools", "memory_management": "memory_management"}
)

graph_builder.add_edge("tools", "oracle")
graph_builder.add_edge("memory_management", END)

app = graph_builder.compile()

# ==========================================
# 7. 运行！
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("🔮 紫微斗数 Agent (短期滑动 + 摘要压缩 + 长期向量)")
    print("="*60)
    
    # 初始化一个空状态，图会帮我们一直维护它
    current_state = {"messages": [], "summary": "", "long_term_context": ""}
    
    while True:
        user_input = input("\n🧑‍💻 你: ")
        if user_input.lower() in ['quit', 'q']: break
            
        # 注意：不再在外部维护列表，直接把新消息喂给图
        events = app.stream(
            {"messages": [("user", user_input)]}, 
            config={"configurable": {"thread_id": "1"}}, # 必须加一个 thread_id，LangGraph 需要它来追踪状态
            stream_mode="values"
        )
        
        for event in events:
            latest_msg = event["messages"][-1]
            if isinstance(latest_msg, AIMessage) and not latest_msg.tool_calls:
                print(f"\n✨ 大师最终回答: {latest_msg.content}")