"""使用OpenAI官方SDK进行文本向量化（Embedding）

为什么要做向量化？
    计算机只能处理数字，无法直接理解文本。无论是 LLM 还是其他 NLP 模型，
    本质上都是对向量数据进行复杂的数学运算来计算概率分布。
    因此，第一步就是将文本转换为数字向量（embedding）。

文本向量化是将文本映射到高维向量空间的过程，这些向量能够捕捉文本的语义信息。
向量化后，可以通过计算向量之间的距离来衡量文本之间的语义差异程度。

本节目的：
    通过手写代码，直观感受向量的生成过程和向量距离的计算方法。
    理解从文本到向量、从向量到距离的完整流程。

核心概念：
- Embedding：将文本映射到高维向量空间（通常是几百到几千维）
- 向量距离：通过数学方法衡量两个向量在空间中的"远近"

依赖安装：
    pip install openai numpy
    
关于 numpy：
    numpy 是 Python 中用于科学计算的基础库，提供了高效的多维数组对象和数学运算函数。
    在本例中，我们使用 numpy 进行向量的点积运算和模长计算。

最小调用方式：
    python text_embedding.py
"""

from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

def load_root_env():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        env_file = os.path.join(current_dir, ".env")
        if os.path.exists(env_file):
            load_dotenv(dotenv_path=env_file)
            return
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("未在仓库根目录找到.env文件，请检查是否放在根目录！")
        current_dir = parent_dir

load_root_env()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
def get_text_embedding(text: str, client) -> np.ndarray:

    print(f"\n正在向量化文本: {text}")
    response = client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=text
        )
    # print(f"\n文本的原始向量表示:\n {response}") #原始向量表示是一个长列表
    embedding = np.array(response.data[0].embedding)  # 转换为numpy数组
    print(f"向量维度: {embedding.shape}")
    print(f"向量前10个值: {embedding[:10]}")
    return embedding    
def caculate_embedding_distance(embeddingA: np.ndarray, embeddingB: np.ndarray, text_a: str, text_b: str) -> float:
    dot_product_a_b = np.dot(embeddingA, embeddingB)  # 向量点积：A·B
    norm_A = np.linalg.norm(embeddingA)  # 向量1的模：||A||
    norm_B = np.linalg.norm(embeddingB)  # 向量2的模：||B||
    similarity_a_b = dot_product_a_b / (norm_A * norm_B)  # 余弦相似度：(A·B)/(||A||*||B||)
    print(f"'{text_a}' vs '{text_b}': {similarity_a_b:.4f}")

def main():
    # 创建OpenAI客户端实例
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    
    # 准备测试文本
    text1 = "天津，这座海滨直辖市在中国北方，以中西文化交汇的魅力、独特的历史景观及美食文化著称。"
    text2 = "天津是一座融合中西文化的中国北方沿海直辖市，以独特的历史风貌、滨海开放气质和美食文化闻名。"
    text3 = "从京城胡同里的老字号到天南海北的特色菜馆，隋坡用镜头记录美食"
    
    print("=" * 60)
    print("步骤1：获取文本的向量表示")
    print("=" * 60)
    embedding1 = get_text_embedding(text1, client)
    embedding2 = get_text_embedding(text2, client)
    embedding3 = get_text_embedding(text3, client)
    

    print("\n" + "=" * 60)
    print("步骤2：计算向量之间的距离")
    print("=" * 60)
    print("\n使用余弦相似度来衡量距离")
    print("余弦相似度公式: (向量A · 向量B) / (||向量A|| * ||向量B||)")
    print("值在[-1, 1]之间，越接近1表示距离越近（越相似）")
    print("\n这里使用线性代数的思维，用 numpy 提供的向量运算来手动计算距离。")
    print("通过这种方式，可以直观感受向量距离计算的每一步。\n")
    
    # 计算文本1和文本2的余弦相似度
    # 余弦相似度即两个向量之间的夹角余弦值，值在[-1, 1]之间，越接近1表示距离越近（越相似）
    # 使用 numpy 的基础向量运算：点积（np.dot）和模长（np.linalg.norm）
    caculate_embedding_distance(embedding1, embedding2, text1, text2)
    caculate_embedding_distance(embedding1, embedding3, text1, text3)
    caculate_embedding_distance(embedding2, embedding3, text2, text3)
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print("\n观察结果可以发现：")
    print("- 文本1和文本2都在描述天津这座城市，虽然用词不同，但语义相似度很高")
    print("- 文本3虽然也提到了美食，但主题是摄影记录，与前两个文本的相似度较低")
    print("\n这说明 embedding 成功捕捉了文本的语义信息，而不仅仅是关键词匹配。")


if __name__ == "__main__":  
    main()


"""
===============================================================================
拓展阅读：NLP 领域的发展与 Embedding 技术
===============================================================================

从符号到向量：NLP 的演进之路
-------------------------------------------------------------------------------

在机器学习出现之前，计算机处理文本主要依靠规则和符号匹配。比如搜索"天津"，
只能找到包含"天津"这两个字的文档，无法理解"海滨直辖市"也在描述天津。

机器学习的引入改变了这一切。机器学习是让计算机从数据中自动学习规律的技术。
在 NLP 领域，机器学习的发展经历了几个重要阶段：

1. 词袋模型（Bag of Words, 2000年代早期）
   - 最简单的向量化方法：统计每个词出现的次数
   - 问题：丢失了词序信息，"狗咬人"和"人咬狗"向量相同
   - 问题：无法理解语义，"天津"和"直辖市"被当作完全无关的词

2. TF-IDF（Term Frequency-Inverse Document Frequency, 2000年代）
   - 改进：降低常见词（如"的"、"是"）的权重
   - 问题：仍然无法理解语义关系

3. Word2Vec（2013年，Google）
   - 突破性进展：用神经网络学习词的向量表示
   - 核心思想：语义相近的词在向量空间中距离也近
   - 效果："国王" - "男人" + "女人" ≈ "女王"（向量运算捕捉语义关系）
   - 问题：一个词只有一个向量，无法处理多义词（"苹果"可能是水果或公司）

4. 上下文相关的 Embedding（2018年后，BERT/GPT时代）
   - 同一个词在不同句子中有不同的向量表示
   - 向量由整个句子的上下文决定
   - OpenAI 的 text-embedding 模型就属于这一代技术

Embedding 与 Encoder/Decoder
-------------------------------------------------------------------------------

在现代 NLP 模型中，经常会听到 Encoder（编码器）和 Decoder（解码器）这两个概念：

- Encoder（编码器）：将输入文本转换为向量表示
  * 输入：文本（如"天津是直辖市"）
  * 输出：向量（如 [0.23, -0.45, 0.67, ...]）
  * 作用：理解和压缩信息
  * 例子：BERT、OpenAI 的 text-embedding 模型

- Decoder（解码器）：将向量转换回文本
  * 输入：向量
  * 输出：文本
  * 作用：生成和表达信息
  * 例子：GPT 模型的生成部分

Encoder 指的是将原始输入数据编码成一种内部的表示（往往是向量、张量等），Embedding 是向量表示的结果。
本节使用的 NNIT-Ada-3-large 模型是一个文本到向量的 Encoder，只负责将文本转换为向量。还有一些可以将图片、音频等转换为向量的 Encoder。

如何表征"语义"
-------------------------------------------------------------------------------

语义是文本的含义。Embedding 通过以下方式捕捉语义：

1. 分布式假设（Distributional Hypothesis）
   - 核心思想：一个词的意思由它周围的词决定
   - "经常出现在相似上下文中的词，语义相近"
   - 例如："天津"和"北京"经常出现在"直辖市"、"城市"等相同的上下文中

2. 神经网络的学习过程
   - 模型在海量文本上训练（如维基百科、新闻、书籍等）
   - 学习预测：给定上下文，预测下一个词
   - 在这个过程中，模型自动学会了将语义相近的文本映射到相近的向量

3. 高维空间的几何结构
   - Embedding 通常有几百到几千个维度
   - 每个维度可能隐含地表示某种语义特征（如"地理位置"、"时间"、"情感"等）
   - 语义相近的文本在这个高维空间中聚集在一起

思考题
-------------------------------------------------------------------------------

1. 为什么"天津是直辖市"和"天津是直辖市天津是直辖市"（重复两遍）的语义应该相同？
   如果用欧氏距离（坐标差值的模）来衡量，会得到什么结果？

2. 计算向量距离有多种方法，常见的包括：
   - 欧氏距离（Euclidean Distance）：sqrt(sum((A[i] - B[i])^2))
   - 余弦相似度（Cosine Similarity）：(A · B) / (||A|| * ||B||)
   - 曼哈顿距离（Manhattan Distance）：各维度差值的绝对值之和
   - 杰卡德相似度（Jaccard Similarity）：集合的交集与并集之比
   
   为什么本例使用余弦相似度？它与欧氏距离的本质区别是什么？

3. 如果要比较两篇文档的"主题相似度"，应该用余弦相似度还是欧氏距离？
   哪个更符合"语义比较"的直觉？

4. Word2Vec 时代的 embedding 是固定的（一个词一个向量），而现代 embedding
   是上下文相关的。这对"苹果公司"和"吃苹果"这两个句子的向量化有什么影响？

===============================================================================
"""
