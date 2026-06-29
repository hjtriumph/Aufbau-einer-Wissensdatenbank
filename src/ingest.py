import os
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings


def load_documents(docs_dir: str) -> list[Document]:
    """加载 docs_dir 下的 PDF / txt / md 文档为 LangChain Document 列表"""
    docs: list[Document] = []

    # -------- 1) 加载 PDF（逐页成为 Document）--------
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext == ".pdf":
                loader = PyPDFLoader(path)
                pdf_docs = loader.load()
                # 给每页写入 metadata，便于后续引用来源
                for d in pdf_docs:
                    d.metadata.update({
                        "source": path,
                        "source_type": "pdf",
                    })
                docs.extend(pdf_docs)

    # -------- 2) 加载 txt / md（DirectoryLoader 批量加载）--------
    for glob in ["**/*.txt", "**/*.md"]:
        loader = DirectoryLoader(
            docs_dir,
            glob=glob,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )
        text_docs = loader.load()
        for d in text_docs:
            # DirectoryLoader 通常会有 source 字段，这里补充 source_type
            d.metadata.update({"source_type": "text"})
        docs.extend(text_docs)

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    将长文档切分成 chunk，提升向量检索效果
    - chunk_size/chunk_overlap 是论文里常写的关键超参数
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],  # 让切分尽量贴近自然段落
    )
    chunks = splitter.split_documents(docs)

    # 给 chunk 增加 chunk_id，便于 debug/引用
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def build_vectorstore(chunks: list[Document]) -> None:
    """使用本地 HuggingFace Embedding + Chroma 构建向量库并持久化"""
    print(f"[ingest] 加载 embedding 模型: {settings.embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    os.makedirs(settings.chroma_dir, exist_ok=True)

    print(f"[ingest] 写入 Chroma 向量库: {settings.chroma_dir}")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name,
    )
    
    print("[ingest] 向量库构建完成（已持久化）")


def ingest():
    """总入口：加载→切分→向量化→存储"""
    if not os.path.isdir(settings.docs_dir):
        raise FileNotFoundError(f"找不到文档目录: {settings.docs_dir}")

    print(f"[ingest] 从目录加载文档: {settings.docs_dir}")
    docs = load_documents(settings.docs_dir)
    if not docs:
        raise RuntimeError("没有找到任何文档！请把 PDF/MD/TXT 放到 ./data/docs")

    print(f"[ingest] 原始 Document 数量（PDF为页数）: {len(docs)}")

    chunks = split_documents(docs)
    print(f"[ingest] 切分后 chunk 数量: {len(chunks)}")

    build_vectorstore(chunks)


if __name__ == "__main__":
    ingest()
