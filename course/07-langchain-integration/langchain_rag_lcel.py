"""第7节: 使用 LangChain 1.x 简化 RAG 开发流程 (LCEL 风格)

本节目标
-----------------------------------------------------------------------------
第6节里, 我们手写了一个最小 RAG:
- splitter 切块
- OpenAI embedding 向量化
- ChromaDB 持久化 + top3 召回
- 把 top3 作为上下文, 让 LLM 生成答案

这一节做同一件事, 但把"胶水代码"交给 LangChain 1.x:
- 用 ChatOpenAI / OpenAIEmbeddings 简化 OpenAI 启动
- 用 Chroma 向量库封装持久化与检索
- 用 langchain-core 的 LCEL (Runnable) 手动拼出 RAG 链路 (不用 langchain_classic)

LangChain 是什么
-----------------------------------------------------------------------------
LangChain 是一个用于搭建 LLM 应用的 Python 框架. 它把常见的组件抽象出来, 例如:
- LLM 模型调用 (例如 OpenAI)
- Embedding 向量化
- 向量库/检索器 (Retriever)
- Prompt 模板
- 把这些组件串起来的链式调用 (LCEL / Runnable)

LCEL 是什么
-----------------------------------------------------------------------------
LCEL 的全称是 LangChain Expression Language. 你可以把它理解为"管道式拼装":

- 每个组件都像一个小函数: 输入 -> 输出.
- 组件之间用 `|` 连接, 表示"把前一个输出, 交给后一个作为输入".
- 组件也可以并行组装成一个 dict, 例如:
  {"context": retriever | format_docs, "input": question}
  表示同一个 question 既拿去检索生成 context, 也作为 input 传给 prompt.

这样做的好处是: 数据流清晰, 链路可复用, 你可以把整条 RAG 当成一个对象去 invoke().

为什么用 LangChain
-----------------------------------------------------------------------------
好处:
- 少写胶水代码: 组件之间的连接方式比较固定, LangChain 能帮你省掉很多重复代码
- 更容易替换组件: 例如把向量库从 Chroma 换成其他实现, 通常只需要改少量初始化代码
- 更容易组合流程: 用 Runnable 把 "检索 -> 组 Prompt -> 调 LLM" 组合成一个可复用的对象

代价与风险:
- 抽象层增加: 排错时要理解 LangChain 的数据流, 不如手写代码直观
- API 变化快: 版本升级可能导致 import 路径或用法变化, 需要同步调整代码
- 过度封装的风险: 小项目如果一开始就堆太多链, 可能比手写更难读

依赖安装 (建议)
-----------------------------------------------------------------------------
pip install langchain-core langchain-openai langchain-chroma langchain-text-splitters python-dotenv

环境变量 (.env 放在项目根目录)
-----------------------------------------------------------------------------

运行
-----------------------------------------------------------------------------
python course/07-langchain-integration/langchain_rag_lcel.py

"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma


class EnvConfig:
    def __init__(self):
        load_dotenv()

        self.api_key = self.require_env("API_KEY", os.getenv("API_KEY"))
        self.base_url = os.getenv("BASE_URL")
        self.chat_model_name = self.require_env("MODEL_NAME", os.getenv("MODEL_NAME"))
        self.embedding_model_name = self.require_env("EMBEDDING_MODEL_NAME", os.getenv("EMBEDDING_MODEL_NAME"))

    def require_env(self, name: str, value: Optional[str]) -> str:
        if value is None:
            raise RuntimeError("缺少环境变量 " + name + ". 请在 .env 中设置 " + name + "=... 后再运行.")
        if value.strip() == "":
            raise RuntimeError("缺少环境变量 " + name + ". 请在 .env 中设置 " + name + "=... 后再运行.")
        return value

    def apply_to_langchain_openai(self):
        # LangChain OpenAI 组件默认读取 OPENAI_API_KEY / OPENAI_BASE_URL.
        os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url is not None and self.base_url.strip() != "":
            os.environ["OPENAI_BASE_URL"] = self.base_url


class TextCorpus:
    def __init__(self, text_file_path: Path, source_name: str):
        self.text_file_path = text_file_path
        self.source_name = source_name

    def load(self) -> str:
        if not self.text_file_path.exists():
            raise FileNotFoundError("找不到示例文本文件: " + str(self.text_file_path))
        with open(self.text_file_path, "r", encoding="utf-8") as file:
            return file.read()

    def to_documents(self, text: str) -> List[Document]:
        documents: List[Document] = []
        documents.append(Document(page_content=text, metadata={"source": self.source_name}))
        return documents


class LangChainRAGPipeline:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        chunk_size: int,
        overlap: int,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.vector_store = None

    def split(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
        )
        chunked_documents = splitter.split_documents(documents)
        return chunked_documents

    def build_vector_store(self, embedding_model_name: str) -> Chroma:
        embeddings = OpenAIEmbeddings(model=embedding_model_name)

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
        )
        return self.vector_store

    def ensure_index(self, documents: List[Document]):
        if self.vector_store is None:
            raise RuntimeError("vector_store 未初始化. 请先调用 build_vector_store().")

        # 为了让脚本可重复运行: 如果库为空就写入, 否则复用.
        # 这里读取底层 collection 的 count, 属于教学用法.
        existing_count = self.vector_store._collection.count()
        if existing_count == 0:
            print("检测到向量库为空, 开始写入文本块: " + str(len(documents)))
            self.vector_store.add_documents(documents)
            print("写入完成.")
        else:
            print("检测到向量库已有数据 (count=" + str(existing_count) + "), 跳过写入.")

    def build_chain(self, chat_model_name: str):
        if self.vector_store is None:
            raise RuntimeError("vector_store 未初始化. 请先调用 build_vector_store().")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        system_prompt = (
            "你是一个严谨的助教. 你只能使用给定的 Context 回答问题."
            "如果 Context 不足以回答, 直接说'材料不足', 并指出缺少什么信息."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "Question:\n{input}\n\nContext:\n{context}"),
            ]
        )

        llm = ChatOpenAI(model=chat_model_name)
        format_docs_runnable = RunnableLambda(self.format_docs)
        output_parser = StrOutputParser()

        rag_chain = (
            {"context": retriever | format_docs_runnable, "input": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
        )
        return rag_chain

    def format_docs(self, docs: List[Document]) -> str:
        # 把多个 Document 拼成一个 Context 字符串.
        # 这里保持简单: 每段之间用分隔线隔开.
        parts: List[str] = []
        index = 0
        while index < len(docs):
            parts.append(docs[index].page_content)
            index = index + 1
        return "\n\n---\n\n".join(parts)

    def retrieve_with_score(self, question: str, top_k: int) -> List[tuple]:
        # 直接从向量库取 top_k, 并拿到 score.
        # 对 Chroma 来说, 这里的 score 通常是距离值: 越小越相近.
        if self.vector_store is None:
            raise RuntimeError("vector_store 未初始化. 请先调用 build_vector_store().")

        results = self.vector_store.similarity_search_with_score(question, k=top_k)
        return results


def main():
    config = EnvConfig()
    config.apply_to_langchain_openai()

    corpus = TextCorpus(
        text_file_path=Path("course/04-text-embedding-similarity/data/魔戒节选.txt"),
        source_name="魔戒节选",
    )

    pipeline = LangChainRAGPipeline(
        persist_directory=Path("course/07-langchain-integration/chroma_db"),
        collection_name="lotr_langchain_rag",
        chunk_size=1000,
        overlap=80,
    )

    print("=" * 70)
    print("步骤1: 读取示例长文本")
    print("=" * 70)

    full_text = corpus.load()
    print("文本路径: " + str(corpus.text_file_path))
    print("文本字符数: " + str(len(full_text)))
    print("文本预览: " + full_text[:120] + "...")

    print("\n" + "=" * 70)
    print("步骤2: splitter 切分文本")
    print("=" * 70)

    raw_documents = corpus.to_documents(text=full_text)
    chunked_documents = pipeline.split(documents=raw_documents)

    print("切分参数: chunk_size=" + str(pipeline.chunk_size) + ", overlap=" + str(pipeline.overlap))
    print("切分结果: " + str(len(chunked_documents)) + " 个文本块")

    preview_count = 2
    if len(chunked_documents) < preview_count:
        preview_count = len(chunked_documents)

    index = 0
    while index < preview_count:
        preview_text = chunked_documents[index].page_content[:80] + "..."
        print("- chunk[" + str(index) + "] preview=" + preview_text)
        index = index + 1

    print("\n" + "=" * 70)
    print("步骤3: 初始化 embedding 与向量库 (Chroma, 持久化)")
    print("=" * 70)

    pipeline.build_vector_store(embedding_model_name=config.embedding_model_name)
    pipeline.ensure_index(documents=chunked_documents)

    print("\n" + "=" * 70)
    print("步骤4: 构建 retriever (top3 召回)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("步骤5: 用 LCEL 组装 RAG Chain (retrieve -> prompt -> LLM)")
    print("=" * 70)
    rag_chain = pipeline.build_chain(chat_model_name=config.chat_model_name)

    print("\n" + "=" * 70)
    print("步骤6: 运行 RAG (top3 召回 + 生成答案)")
    print("=" * 70)

    question = "托尔金的生平是什么?"

    print("\n召回结果 (带 score, 越小越相近):")
    top_k = 3
    retrieved_with_score = pipeline.retrieve_with_score(question=question, top_k=top_k)
    index = 0
    while index < len(retrieved_with_score):
        doc = retrieved_with_score[index][0]
        score = retrieved_with_score[index][1]
        preview = doc.page_content
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print("- rank=" + str(index + 1) + " score=" + str(score) + " preview=" + preview)
        index = index + 1

    answer = rag_chain.invoke(question)

    print("Question: " + question)
    print("\nAnswer:\n" + answer)


if __name__ == "__main__":
    main()
