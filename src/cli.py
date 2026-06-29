import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .rag_chain import build_rag_chain, get_llm
from .few_shot_chain import build_few_shot_rag_chain


DIRECT_LLM_SYSTEM = """You are a Building Technology assistant.

Answer the user's question directly using your own knowledge.
Do not use any external context or retrieval.
If you are uncertain, say so clearly.

Output format (keep the section titles exactly as written below):

1) Conclusion
2) Explanation
3) Confidence / Limits

Language rule:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
"""


def _extract_answer(out: Any) -> str:
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for k in ("answer", "result", "output", "output_text", "text"):
            if k in out and isinstance(out[k], str):
                return out[k]
        return json.dumps(out, ensure_ascii=False, indent=2)
    return str(out)


def run_llm(question: str) -> Dict[str, Any]:
    llm = get_llm()
    prompt = [
        ("system", DIRECT_LLM_SYSTEM),
        ("human", f"Question: {question}"),
    ]
    out = llm.invoke(prompt)

    answer = getattr(out, "content", None)
    if not isinstance(answer, str):
        answer = str(out)

    return {
        "mode": "llm",
        "question": question,
        "answer": answer,
        "sources": [],
    }


def run_rag(question: str) -> Dict[str, Any]:
    rag = build_rag_chain()
    out = rag.invoke({"input": question})
    return {
        "mode": "rag",
        "question": question,
        "answer": _extract_answer(out),
        "context": out.get("context") if isinstance(out, dict) else None,
    }


def run_fewshot(question: str, num_examples: int = 3) -> Dict[str, Any]:
    chain = build_few_shot_rag_chain(num_examples=num_examples)
    out = chain.invoke({"input": question})

    return {
        "mode": f"fewshot_{num_examples}",
        "question": question,
        "answer": _extract_answer(out),
        "context": out.get("context") if isinstance(out, dict) else None,
        "sources": out.get("sources", []) if isinstance(out, dict) else [],
        "num_examples": num_examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["llm", "rag", "fewshot"],
        default="fewshot",
        help="Select run mode",
    )
    ap.add_argument("--q", required=True, help="Input question (natural language)")
    ap.add_argument(
        "--fewshot-n",
        type=int,
        default=3,
        help="Number of few-shot examples to use when mode=fewshot",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (useful for evaluation/logging)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Append results to a jsonl file",
    )
    ap.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieval sources (for debugging)",
    )

    args = ap.parse_args()

    if args.mode == "llm":
        result = run_llm(args.q)
    elif args.mode == "rag":
        result = run_rag(args.q)
    else:
        result = run_fewshot(args.q, num_examples=args.fewshot_n)

    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    if not args.show_sources:
        if "sources" in result:
            result.pop("sources", None)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()