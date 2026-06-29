import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document

from .rag_chain import build_retriever
from .config import settings


def extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        md = d.metadata or {}
        out.append(
            {
                "source": md.get("source_basename") or md.get("source"),
                "page": md.get("page"),
                "chunk_id": md.get("chunk_id"),
            }
        )
    return out


def run_single_query(query: str) -> Dict[str, Any]:
    retriever = build_retriever()

    start = time.time()
    docs = retriever.invoke(query)
    latency = time.time() - start

    return {
        "query": query,
        "k": settings.k,
        "search_type": settings.search_type,
        "latency_sec": round(latency, 3),
        "num_docs": len(docs),
        "sources": extract_sources(docs),
        "preview": [
            d.page_content[:300].replace("\n", " ")
            for d in docs
        ],
    }


def run_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    results = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            query = obj["question"]
            result = run_single_query(query)
            results.append(result)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=str, help="单个查询")
    ap.add_argument("--dataset", type=str, help="jsonl 格式，每行包含 {'question': ...}")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=str, help="保存为 jsonl 文件")
    args = ap.parse_args()

    if args.q:
        result = run_single_query(args.q)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"\nQuery: {result['query']}")
            print(f"Latency: {result['latency_sec']} sec")
            print("\nTop results:")
            for i, src in enumerate(result["sources"], 1):
                print(f"{i}. {src}")
    elif args.dataset:
        results = run_dataset(args.dataset)
        if args.out:
            out_path = Path(args.out)
            with out_path.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print("请使用 --q 或 --dataset")


if __name__ == "__main__":
    main()