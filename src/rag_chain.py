from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .config import settings


# ========= Utils =========
def extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    out = []
    for d in docs or []:
        md = d.metadata or {}
        out.append(
            {
                "source": md.get("source", "unknown"),
                "page": md.get("page", None),
                "chunk_id": md.get("chunk_id", None),
            }
        )
    return out


def _format_docs_as_context(docs: List[Document]) -> str:
    """Format docs into a single context string with source/page/chunk for citation."""
    parts: List[str] = []
    for d in docs or []:
        md = d.metadata or {}
        src = md.get("source_basename") or md.get("source", "unknown")
        page = md.get("page", None)
        chunk_id = md.get("chunk_id", None)

        header = f"Source: {src}"
        if page is not None:
            header += f" | page={page}"
        if chunk_id is not None:
            header += f" | chunk={chunk_id}"

        parts.append(header + "\n" + (d.page_content or ""))

    return "\n\n---\n\n".join(parts).strip()


# ========= LLM / Embeddings / VectorStore caches =========
@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """DeepSeek (OpenAI-compatible) via ChatOpenAI + base_url."""
    if not settings.deepseek_api_key:
        raise RuntimeError("请在 .env 中设置 DEEPSEEK_API_KEY")

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=getattr(settings, "temperature", 0.2),
        max_tokens=getattr(settings, "max_tokens", None),
        timeout=getattr(settings, "request_timeout", None),
        max_retries=getattr(settings, "max_retries", 2),
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Load persistent Chroma store."""
    return Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
        collection_name=settings.collection_name,
    )


def build_retriever():
    """VectorStore -> Retriever with compatible kwargs."""
    vs = get_vectorstore()

    search_type = settings.search_type  # "similarity" or "mmr"
    k = settings.k
    fetch_k = settings.fetch_k

    if search_type == "mmr":
        kwargs = {"k": k, "fetch_k": fetch_k}
    else:
        kwargs = {"k": k}

    return vs.as_retriever(search_type=search_type, search_kwargs=kwargs)


# ========= RAG Prompt =========
RAG_SYSTEM = """You are a Building Technology assistant.

You MUST answer strictly based on the provided Context and cite evidence (source/page/chunk).
If the Context is insufficient to support a conclusion, clearly state:
INSUFFICIENT_CONTEXT: <reason>
and suggest better retrieval keywords.

Output format (keep the section titles exactly as written below):

1) Conclusion (1–2 sentences)
2) Explanation (bullet points)
3) Evidence (source/page/chunk + short summary)
4) Assumptions & Scope
5) Suggested Next Retrieval

Language rule:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM),
        ("human", "question：{input}\n\nContext:\n{context}"),
    ]
)


def build_rag_chain():
    """
    Stable RAG runnable without langchain.chains.* imports.

    Input:  {"input": question}
    Output: {"answer": str, "context": List[Document]}
    """
    llm = get_llm()
    retriever = build_retriever()
    parser = StrOutputParser()

    def _retrieve(x: Dict[str, Any]) -> Dict[str, Any]:
        q = str(x.get("input", "")).strip()
        docs = retriever.invoke(q)
        return {"input": q, "context_docs": docs, "context": _format_docs_as_context(docs)}

    def _generate(x: Dict[str, Any]) -> Dict[str, Any]:
        answer = (RAG_PROMPT | llm | parser).invoke({"input": x["input"], "context": x["context"]})
        return {"answer": answer, "context": x["context_docs"]}

    return RunnableLambda(_retrieve) | RunnableLambda(_generate)