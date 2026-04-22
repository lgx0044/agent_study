from openai import OpenAI
import time

class LocalLLMEngine:
    def __init__(self, base_url="http://127.0.0.1:8080/v1"):
        """
        初始化本地 API 客户端
        base_url: 指向你刚才启动的 llama-server
        """
        print(f"[*] 正在连接本地大模型服务: {base_url}")
        self.client = OpenAI(
            base_url=base_url,
            api_key="sk-no-key-required" # 本地模型不需要真实秘钥
        )
        
    def chat(self, prompt, system_msg="你是一个专业的AI助手。", temperature=0.7, stream=True):
        """
        发起对话 (支持流式输出，打字机效果)
        """
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        # 记录耗时
        t0 = time.time()
        
        # 发起 API 请求
        response = self.client.chat.completions.create(
            model="qwen2.5-7b", # 这里的名字可以随便写，因为本地只有一个模型
            messages=messages,
            temperature=temperature,
            stream=stream # 开启流式输出
        )
        
        print("\n🤖 Qwen: ", end="", flush=True)
        full_reply = ""
        
        if stream:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_reply += content
            print(f"\n\n[⏱️ 耗时: {time.time() - t0:.2f} 秒]")
            return full_reply
        else:
            ans = response.choices[0].message.content
            print(ans)
            print(f"\n[⏱️ 耗时: {time.time() - t0:.2f} 秒]")
            return ans

# 测试一下瞬间响应的快感：
if __name__ == "__main__":
    llm = LocalLLMEngine()
    llm.chat("你好！请用一句话介绍一下紫微斗数。")