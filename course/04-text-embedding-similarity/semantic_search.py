"""基于长文本的语义检索系统

本节目的：
    在第3节学习了如何对短文本进行向量化和相似度计算后，本节将这些技术
    应用到实际场景：从长文本（如小说）中检索与用户查询最相关的段落。

核心流程：
    1. 读取长文本文件
    2. 将长文本按固定长度切分成多个文本块
    3. 对每个文本块进行向量化（embedding）
    4. 用户输入查询问题
    5. 对查询进行向量化
    6. 计算查询向量与所有文本块向量的相似度
    7. 返回相似度最高的文本块

数学视角（简要）：
    - Embedding 可视为高维向量，文本块之间的相似性等价于向量间的“方向接近程度”
    - 余弦相似度是归一化内积：cos_sim(a,b)=dot(a,b)/(||a||*||b||)
      这在统计上接近“相关性”的直觉：方向一致则相似度高

应用场景：
    - 从小说中找到包含特定内容的段落
    - 从文档中检索相关信息
    - 问答系统的基础技术

依赖安装：
    pip install openai numpy python-dotenv

最小调用方式：
    python semantic_search.py
"""

from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv 

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")


def main():

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    
    print("=" * 70)
    print("步骤1：读取长文本文件")
    print("=" * 70)
    
    # 读取魔戒节选文本
    # 可以换成其他文本进行尝试
    text_file_path = "course/04-text-embedding-similarity/data/魔戒节选.txt" # 注意修改为文件的真实位置，Windows 与 Linux 不同
    with open(text_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    print(f"\n文本文件路径: {text_file_path}")
    print(f"文本总字符数: {len(full_text)}")
    print(f"文本前100个字符: {full_text[:100]}...")
    
    print("\n" + "=" * 70)
    print("步骤2：按固定长度切分文本")
    print("=" * 70)
    
    # 固定长度切分（每个块1000字符，重叠50字符）
    chunk_size = 1000  # 每个文本块的大小
    overlap = 50      # 块之间的重叠部分，避免切断语义
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)  # 移动位置，考虑重叠
    
    print(f"\n切分参数:")
    print(f"  - 每块大小: {chunk_size} 字符")
    print(f"  - 重叠大小: {overlap} 字符")
    print(f"  - 切分结果: {len(chunks)} 个文本块")
    print(f"\n前3个文本块示例:")
    for i in range(min(3, len(chunks))):
        print(f"\n文本块 {i+1} (长度 {len(chunks[i])} 字符):")
        print(f"  {chunks[i][:80]}...")
    
    print("\n" + "=" * 70)
    print("步骤3：对所有文本块进行向量化")
    print("=" * 70)
    print(f"\n正在向量化 {len(chunks)} 个文本块，请稍候...")
    
    # 存储所有文本块的向量
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        # 对每个文本块调用 embedding API
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=chunk
        )
        embedding = np.array(response.data[0].embedding)
        embeddings.append(embedding)
        
        # 每处理10个块打印一次进度
        if (i + 1) % 10 == 0:
            print(f"  已处理: {i+1}/{len(chunks)} 个文本块")
    
    print(f"\n向量化完成！")
    print(f"每个向量的维度: {embeddings[0].shape}")
    print(f"总共 {len(embeddings)} 个向量")
    
    print("\n" + "=" * 70)
    print("步骤4：输入查询并进行语义检索")
    print("=" * 70)
    
    # 用户查询
    query = "托尔金的生平"
    print(f"\n查询问题: {query}")
    
    # 对查询进行向量化
    print(f"正在向量化查询...")
    query_response = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=query
    )
    query_embedding = np.array(query_response.data[0].embedding)
    print(f"查询向量维度: {query_embedding.shape}")
    
    print("\n" + "=" * 70)
    print("步骤5：计算查询与所有文本块的相似度")
    print("=" * 70)
    
    # 计算查询向量与所有文本块向量的余弦相似度
    similarities = []
    
    query_norm = np.linalg.norm(query_embedding)  # 查询向量的模
    
    for i, embedding in enumerate(embeddings):
        # 计算余弦相似度
        dot_product = np.dot(query_embedding, embedding)
        embedding_norm = np.linalg.norm(embedding)
        similarity = dot_product / (query_norm * embedding_norm)
        similarities.append(similarity)
    
    print(f"\n已计算 {len(similarities)} 个相似度值")
    print(f"相似度范围: [{min(similarities):.4f}, {max(similarities):.4f}]")
    
    print("\n" + "=" * 70)
    print("步骤6：返回最相关的文本块")
    print("=" * 70)
    
    # 找到相似度最高的前3个文本块
    top_k = 3
    
    # 获取相似度最高的索引（使用 numpy 的 argsort，降序排列）
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n最相关的 {top_k} 个文本块:\n")
    
    for rank, idx in enumerate(top_indices, 1):
        print(f"排名 {rank}:")
        print(f"  文本块索引: {idx}")
        print(f"  相似度: {similarities[idx]:.4f}")
        print(f"  文本块长度: {len(chunks[idx])} 字符")
        # 只显示前150字符，避免输出过长
        preview = chunks[idx][:150] + "..." if len(chunks[idx]) > 150 else chunks[idx]
        print(f"  内容预览:\n{preview}")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"""
    本次检索统计:
    - 文本总字符数: {len(full_text)}
    - 切分后文本块数: {len(chunks)}
    - 查询问题: {query}
    - 返回结果数: {top_k}
    - 最高相似度: {similarities[top_indices[0]]:.4f}
    
    从结果可以看出，语义检索能够找到与查询最相关的段落，
    即使查询词和文本中的词不完全匹配，也能通过语义理解找到相关内容。
    """)


if __name__ == "__main__":  
    main()


"""
===============================================================================
扩展思考与优化方向
===============================================================================

1. 文本切分策略
-------------------------------------------------------------------------------

当前使用的是固定长度切分，这种方法简单但有局限：
- 可能在句子中间切断，破坏语义完整性
- 不同段落的信息密度不同，固定长度可能不合适



思考：为什么要设置重叠（overlap）？有没有其他更好的切分策略？


2. 检索性能优化
-------------------------------------------------------------------------------

当前的检索方法是"暴力搜索"（遍历所有文本块计算相似度）：
- 优点：简单、精确
- 缺点：当文本块数量很大时，速度慢

优化方向：
- 使用向量数据库（如 ChromaDB、Faiss）：第5节将学习
- 建立索引：加速检索
- 批量计算：利用矩阵运算的并行性

下一节预告
-------------------------------------------------------------------------------

第5节将学习 ChromaDB，一个专门的向量数据库：
- 自动管理向量的存储和检索
- 提供高效的相似度搜索
- 支持元数据过滤
- 提供持久化存储

3. 线性代数与概率统计视角
-------------------------------------------------------------------------------

线性代数：
- 余弦相似度本质是归一化内积，衡量向量夹角；归一化后可去除“尺度”影响
- 批量相似度计算可用矩阵乘法加速（把所有 embedding 组成矩阵）

概率统计：
- 可将 embedding 的每一维视为随机变量的观测值；归一化相当于降低幅度噪声的影响
- 若对向量进行中心化/标准化，可在一定程度上减少“高频维度”对相似度的偏置

===============================================================================
"""
