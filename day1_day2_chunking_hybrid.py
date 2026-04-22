# -*- coding: utf-8 -*-
"""
混合策略文本分割：
1. 先按章节标题（星曜+宫位）分割
2. 对大块内容（超过500字）做语义分割
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import re
import os

# 模型本地存储路径
model_path = "./local_models/bge-small-zh-v1.5"

# 1. 加载 Embedding 模型
print("正在加载 Embedding 模型...")
if os.path.exists(model_path):
    print("使用本地模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'}
    )
else:
    print("首次运行，正在下载模型到本地...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    model.save(model_path)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'}
    )

# 2. 读取文件
print("正在读取文件...")
with open('data2_cleaned.txt', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 3. 第一阶段：按章节标题分割
print("第一阶段：按章节标题分割...")

def split_by_headings(text):
    """按星曜和宫位标题分割文本"""
    # 匹配模式：一、紫微星 或 （1）命宫 或 (1)命宫
    pattern = r'([一二三四五六七八九十]+、[^\n]+|\([\d]+\)[^\n]+|（[\d]+）[^\n]+)'
    
    # 找到所有标题位置
    matches = list(re.finditer(pattern, text))
    
    chunks = []
    
    # 处理前言部分（第一个标题之前）
    if matches and matches[0].start() > 0:
        preface = text[:matches[0].start()].strip()
        if preface:
            chunks.append({
                'title': '前言/概述',
                'content': preface
            })
    
    # 处理每个章节
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_content = text[start:end].strip()
        
        chunks.append({
            'title': title,
            'content': section_content
        })
    
    return chunks

# 执行标题分割
heading_chunks = split_by_headings(content)
print(f"标题分割完成，共 {len(heading_chunks)} 个章节")

# 4. 第二阶段：对大块内容做语义分割
print("\n第二阶段：对大块内容进行语义分割...")

# 初始化语义分割器（使用较宽松的阈值）
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70  # 比90宽松，比50严格
)

final_docs = []
MIN_CHUNK_SIZE = 300  # 小于300字不分割
MAX_CHUNK_SIZE = 800  # 大于800字强制分割

for chunk in heading_chunks:
    title = chunk['title']
    text = chunk['content']
    text_length = len(text)
    
    # 小段落直接保留
    if text_length < MIN_CHUNK_SIZE:
        final_docs.append(Document(
            page_content=text,
            metadata={'source': title, 'split_method': 'heading_only'}
        ))
        print(f"  [{title}] 长度{text_length}，直接保留")
    
    # 中等段落尝试语义分割
    elif text_length <= MAX_CHUNK_SIZE:
        try:
            sub_docs = semantic_splitter.create_documents([text])
            if len(sub_docs) > 1:
                for j, doc in enumerate(sub_docs):
                    doc.metadata = {
                        'source': f"{title}_part{j+1}",
                        'split_method': 'semantic'
                    }
                    final_docs.append(doc)
                print(f"  [{title}] 长度{text_length}，语义分割为 {len(sub_docs)} 段")
            else:
                final_docs.append(Document(
                    page_content=text,
                    metadata={'source': title, 'split_method': 'heading_only'}
                ))
                print(f"  [{title}] 长度{text_length}，语义分割后仍为1段")
        except Exception as e:
            # 语义分割失败，直接保留
            final_docs.append(Document(
                page_content=text,
                metadata={'source': title, 'split_method': 'heading_only', 'error': str(e)}
            ))
            print(f"  [{title}] 语义分割失败，直接保留: {e}")
    
    # 大段落强制按大小分割
    else:
        # 先尝试语义分割
        try:
            sub_docs = semantic_splitter.create_documents([text])
            for j, doc in enumerate(sub_docs):
                doc.metadata = {
                    'source': f"{title}_part{j+1}",
                    'split_method': 'semantic_large'
                }
                final_docs.append(doc)
            print(f"  [{title}] 长度{text_length}，语义分割为 {len(sub_docs)} 段")
        except Exception as e:
            # 失败则按字符数简单分割
            print(f"  [{title}] 语义分割失败，使用字符分割: {e}")
            # 简单按段落分割
            paragraphs = text.split('\n\n')
            current_chunk = ""
            chunk_num = 1
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < MAX_CHUNK_SIZE:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        final_docs.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={'source': f"{title}_part{chunk_num}", 'split_method': 'char_fallback'}
                        ))
                        chunk_num += 1
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                final_docs.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={'source': f"{title}_part{chunk_num}", 'split_method': 'char_fallback'}
                ))

# 5. 输出结果
print(f"\n{'='*50}")
print(f"分割完成！共生成 {len(final_docs)} 个文档块")
print(f"{'='*50}")

# 统计信息
semantic_count = sum(1 for d in final_docs if 'semantic' in d.metadata.get('split_method', ''))
heading_count = sum(1 for d in final_docs if d.metadata.get('split_method') == 'heading_only')
fallback_count = sum(1 for d in final_docs if 'fallback' in d.metadata.get('split_method', ''))

print(f"\n统计：")
print(f"  - 仅标题分割: {heading_count} 个")
print(f"  - 语义分割: {semantic_count} 个")
print(f"  - 字符分割(备选): {fallback_count} 个")

# 显示前5个块的预览
print(f"\n前5个文档块预览：")
for i, doc in enumerate(final_docs[:5]):
    preview = doc.page_content[:100].replace('\n', ' ')
    # 过滤掉可能导致编码问题的字符
    preview = preview.encode('gbk', errors='ignore').decode('gbk')
    source = doc.metadata['source'].encode('gbk', errors='ignore').decode('gbk')
    print(f"\n[{i+1}] {source}")
    print(f"    长度: {len(doc.page_content)} 字")
    print(f"    内容: {preview}...")

# 6. 保存结果（可选）
save_to_chroma = True
if save_to_chroma:
    from langchain_community.vectorstores import Chroma
    
    db_path = "./chroma_db_hybrid"
    print(f"\n正在创建向量数据库...")
    
    vector_db = Chroma.from_documents(
        documents=final_docs,
        embedding=embeddings,
        collection_name="ziwei_hybrid",
        persist_directory=db_path
    )
    vector_db.persist()
    print(f"数据库已保存至: {db_path}")
    
    # 测试检索
    print("\n测试检索：查询'贪狼'")
    results = vector_db.similarity_search("贪狼", k=3)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r.metadata['source']}")
        print(f"    {r.page_content[:150]}...")
    
# 使用 similarity_search_with_score 可以看到相似度分值
    # 使用 similarity_search_with_score 可以看到相似度分值
# 分数通常是 L2 距离，数值越小越相似（注意：不同算法定义不同）
    results_with_scores = vector_db.similarity_search_with_score("贪狼", k=3)
    for doc, score in results_with_scores:
        print(f"得分: {score:.4f} | 内容: {doc.page_content[:50]}...")

