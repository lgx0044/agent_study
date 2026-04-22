import numpy as np
from rank_bm25 import BM25Okapi
import jieba  # 中文分词工具

# 1. 准备 BM25 语料库
# 我们直接使用昨天切好的 docs 里的文本
print("正在初始化 BM25 检索器...")
corpus = [doc.page_content for doc in docs]

# BM25 需要先分词。对于中文，我们使用 jieba 分词
def chinese_tokenizer(text):
    return list(jieba.cut(text))

tokenized_corpus = [chinese_tokenizer(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 2. 定义双路检索函数
def hybrid_search(query, k=3, alpha=0.5):
    """
    alpha: 混合权重。1.0 为纯向量检索，0.0 为纯 BM25。
    """
    # --- 第一路：向量检索 ---
    # 我们使用昨天的 vector_db
    vector_results = vector_db.similarity_search_with_score(query, k=k)
    
    # --- 第二路：BM25 检索 ---
    tokenized_query = chinese_tokenizer(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 归一化 BM25 分数（让它跟向量分数在一个量级，这里简化处理）
    if np.max(bm25_scores) != 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)
    
    # 找到 BM25 得分最高的前 k 个索引
    top_n_idx = np.argsort(bm25_scores)[-k:][::-1]
    
    print(f"查询问题: {query}")
    print("\n--- [路 1] 向量检索召回 ---")
    for doc, score in vector_results:
        print(f"内容: {doc.page_content[:50]}... (L2距离: {score:.4f})")
        
    print("\n--- [路 2] BM25 关键词召回 ---")
    for idx in top_n_idx:
        if bm25_scores[idx] > 0:
            print(f"内容: {corpus[idx][:50]}... (BM25得分: {bm25_scores[idx]:.4f})")

# 3. 测试混合检索的效果
# 找一个具体的关键词试试，比如你日记里提到的某个具体的人名或物品
test_query = "2024年3月5日" 
hybrid_search(test_query)