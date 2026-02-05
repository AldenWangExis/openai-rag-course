"""第7节: LangChain 1.x 的“过度封装”

目标:
- 尽量少写“功能代码”(自己不造轮子)
- 尽量多用 LangChain 组件(代价是更难摸清内部细节)

提醒:
LangChain 确实能显著减少胶水代码、提高组件复用与替换效率，但抽象层次越高，
对系统真实行为的可见性就越低：例如数据是怎么进库的、检索与重排的细节、
失败时在哪里重试、性能瓶颈在哪里等，都会变得更难定位。

因此要权衡：
- 什么时候适合手写实现：当你需要极强的可控性/可解释性、要做深度定制、或需要精准排错与性能优化时。
- 什么时候适合用第三方包（如 LangChain/Pysantic AI）：当流程足够标准化、你更关心快速搭建与迭代、以及可替换的组件化能力时。

运行:
  python course/07-langchain-integration/langchain_rag_lcel_overengineered.py

环境变量(.env 放在项目根目录):

"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _require_env(name: str, value: Optional[str]) -> str:
    if value is None or value.strip() == "":
        raise RuntimeError(f"缺少环境变量 {name}. 请在 .env 中设置 {name}=... 后再运行.")
    return value


def main() -> None:
    load_dotenv()

    api_key = _require_env("API_KEY", os.getenv("API_KEY"))
    base_url = os.getenv("BASE_URL")
    chat_model_name = _require_env("MODEL_NAME", os.getenv("MODEL_NAME"))
    embedding_model_name = _require_env("EMBEDDING_MODEL_NAME", os.getenv("EMBEDDING_MODEL_NAME"))

    os.environ["OPENAI_API_KEY"] = api_key
    if base_url is not None and base_url.strip() != "":
        os.environ["OPENAI_BASE_URL"] = base_url

    text_path = Path("course/04-text-embedding-similarity/data/魔戒节选.txt")
    text = text_path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    documents = splitter.create_documents([text], metadatas=[{"source": "魔戒节选"}])

    persist_directory = Path("course/07-langchain-integration/chroma_db")

    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    ids = [
        hashlib.sha256((doc.metadata.get("source", "") + "\n" + doc.page_content).encode("utf-8")).hexdigest()
        for doc in documents
    ]
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name="lotr_langchain_rag",
        persist_directory=str(persist_directory),
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严谨的助教. 你只能使用给定的 Context 回答问题."
                "如果 Context 不足以回答, 直接说'材料不足', 并指出缺少什么信息.",
            ),
            ("user", "Question:\n{input}\n\nContext:\n{context}"),
        ]
    )
    llm = ChatOpenAI(model=chat_model_name)
    format_docs = RunnableLambda(lambda docs: "\n\n---\n\n".join(d.page_content for d in docs))

    rag = RunnablePassthrough.assign(context=retriever | format_docs) | prompt | llm | StrOutputParser()

    question = "托尔金的生平是什么?"
    answer = rag.invoke(question)
    print(answer)


if __name__ == "__main__":
    main()
