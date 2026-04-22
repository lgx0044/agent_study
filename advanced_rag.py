import os
# 解决OpenMP库重复初始化的问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import jieba
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
# from llm_engine import LocalLLMEngine

class Advanced_RAG_Pipeline:
    def __init__(self, vector_db_path, embedding_model_path, reranker_model_path):
        """
        初始化Advanced_RAG_Pipeline类
        
        Args:
            vector_db_path: 向量数据库路径
            embedding_model_path: Embedding模型路径
            reranker_model_path: 重排序模型路径
        """
        # 1. 加载Embedding模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # 2. 加载向量数据库
        self.vector_db = Chroma(
            collection_name="ziwei_hybrid",
            persist_directory=vector_db_path,
            embedding_function=self.embeddings
        )
        
        # 3. 准备BM25语料库
        docs = self.vector_db.get()
        self.corpus = docs['documents']
        self.corpus_docs = [Document(page_content=doc) for doc in self.corpus]
        
        # 4. 分词并初始化BM25
        def chinese_tokenizer(text):
            return list(jieba.cut(text))
        
        tokenized_corpus = [chinese_tokenizer(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 5. 加载重排序模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reranker = CrossEncoder(reranker_model_path, device=device)
    
    def retrieve(self, query, top_k=10):
        """
        混合检索（向量检索 + BM25检索）
        
        Args:
            query: 查询语句
            top_k: 返回结果数量
            
        Returns:
            混合检索结果（去重后的文档列表）
        """
        # 1. 向量检索
        v_results = self.vector_db.similarity_search(query, k=top_k)
        
        # 2. BM25检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 归一化BM25分数
        if np.max(bm25_scores) != 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
        
        # 找到BM25得分最高的前k个索引
        top_n_idx = np.argsort(bm25_scores)[-top_k:][::-1]
        b_results = [self.corpus_docs[idx] for idx in top_n_idx if bm25_scores[idx] > 0]
        
        # 3. 合并去重
        # 创建一个字典来存储唯一的文档内容
        unique_docs = {}
        for doc in v_results + b_results:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        
        # 转换回列表
        all_candidates = list(unique_docs.values())
        return all_candidates
    
    def post_process(self, query, candidates, top_n=3):
        """
        重排序检索结果
        
        Args:
            query: 查询语句
            candidates: 检索结果列表
            top_n: 返回前n个结果
            
        Returns:
            重排序后的结果列表
        """
        # 构建模型输入的[问题, 文档]对
        pairs = [[query, doc.page_content] for doc in candidates]
        
        # 计算相关性分数
        scores = self.reranker.predict(pairs)
        
        # 将分数与文档组合并排序
        scored_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        # 返回前top_n个结果
        final_results = [item[0] for item in scored_results[:top_n]]
        
        return final_results
    
    def query(self, question, llm_engine=None):
        """
        全链路问答
        
        Args:
            question: 用户问题
            
        Returns:
            最终生成的Prompt和检索到的文档
        """
        # 1. 混合检索
        candidates = self.retrieve(question)
        
        # 2. 重排序
        refined_context = self.post_process(question, candidates)
        
        # 3. 构造Prompt
        context_str = "\n".join([f"资料{i+1}: {c.page_content}" for i, c in enumerate(refined_context)])
        prompt = f"""你是一个基于私有知识库的助手。请严格根据以下参考资料回答问题。
如果资料中没有相关信息，请直接回答"资料中未提及"，不要胡乱猜测。

参考资料：
{context_str}

问题：{question}
回答："""

        # 4. 调用LLM
        if llm_engine:
            response = llm_engine.chat(prompt)
        
        return prompt, refined_context

# 调用例子
if __name__ == "__main__":
    # 模型和数据库路径
    vector_db_path = "./chroma_db_hybrid"
    embedding_model_path = "./local_models/bge-small-zh-v1.5"
    reranker_model_path = "./local_models/bge-reranker-base"
    
    # 初始化RAG pipeline
    rag_system = Advanced_RAG_Pipeline(
        vector_db_path=vector_db_path,
        embedding_model_path=embedding_model_path,
        reranker_model_path=reranker_model_path
    )
    
    # 测试查询
    test_questions = [
        "贪狼星的特点是什么？",
        "紫微星在命宫有什么影响？",
        "天机星在事业宫代表什么？",
        "今天吃的是什么？"
    ]

    # 2. 实例化你刚才写的常驻 API 客户端
    # my_llm = LocalLLMEngine()
    
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"测试问题: {question}")
        print(f"{'='*80}")
        # prompt, context = rag_system.query(question, llm_engine=my_llm)
        rag_system.query(question)

        print(f"\n{'='*80}")
