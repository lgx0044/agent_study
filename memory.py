class SlidingWindowMemory:
    def __init__(self, max_rounds=3):
        """
        初始化滑动窗口记忆模块。
        max_rounds: 最多记住几轮“一问一答”（3轮 = 6条消息）
        """
        self.history = []
        self.max_rounds = max_rounds

    def save_context(self, user_query, final_answer):
        """
        保存一轮完整的对话（注意：绝不保存中间的 JSON 和报错！）
        """
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": final_answer})
        
        # 💥 核心机制：滑动窗口截断（防止 Token 撑爆显存）
        # 如果当前长度超过了允许的最大条数，就把最老的两句话（一问一答）切掉
        if len(self.history) > self.max_rounds * 2:
            self.history = self.history[-self.max_rounds * 2:]

    def load_memory_variables(self, system_prompt):
        """
        组装发给大模型的完整上下文：System Prompt + 历史记忆
        """
        # 永远把 System Prompt 放在最前面
        messages = [{"role": "system", "content": system_prompt}]
        # 拼上历史聊天记录
        messages.extend(self.history)
        return messages