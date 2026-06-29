from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .benchmark_cases import TEST_CASES, Case
from .cli import run_llm, run_fewshot
from .rag_chain import get_llm


METHODS: Tuple[str, ...] = (
    "llm",
    "fewshot_0",
    "fewshot_1",
    "fewshot_3",
    "fewshot_5",
    "fewshot_7",
    "fewshot_10",
)

FEWSHOT_METHOD_TO_N: Dict[str, int] = {
    "fewshot_0": 0,
    "fewshot_1": 1,
    "fewshot_3": 3,
    "fewshot_5": 5,
    "fewshot_7": 7,
    "fewshot_10": 10,
}


# ============================================================
# Basic text utilities
# ============================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def normalize_for_match(text: Any) -> str:
    """
    Normalize text for simple key-point matching.
    """
    text = normalize_text(text).lower()
    text = text.replace("％", "%")
    text = re.sub(r"\s+", " ", text)
    return text


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.mean(values), 4)


# ============================================================
# Metric 1: Key point coverage
# ============================================================

def compute_key_point_coverage(answer: str, key_points: List[str]) -> float:
    """
    Checks how many manually defined key points appear in the answer.

    Example:
    key_points = ["4.91", "4.90", "occupancy"]
    score = matched / total
    """
    if not key_points:
        return 0.0

    answer_norm = normalize_for_match(answer)

    matched = 0
    for kp in key_points:
        kp_norm = normalize_for_match(kp)
        if kp_norm and kp_norm in answer_norm:
            matched += 1

    return round(matched / len(key_points), 4)


def get_matched_key_points(answer: str, key_points: List[str]) -> List[str]:
    answer_norm = normalize_for_match(answer)
    matched = []

    for kp in key_points:
        kp_norm = normalize_for_match(kp)
        if kp_norm and kp_norm in answer_norm:
            matched.append(kp)

    return matched


# ============================================================
# Metric 2 + 3: LLM judge
# completeness + answer relevance
# ============================================================

JUDGE_SYSTEM = """You are an impartial evaluator for a Building Technology QA benchmark.

You will evaluate a model answer based on:
1. completeness
2. answer_relevance

Definitions:

completeness:
- Measures whether the answer covers the important information from the reference answer and key points.
- Score range: 0.0 to 1.0
- 1.0 means all important points are covered.
- 0.5 means partially covered.
- 0.0 means mostly missing or incorrect.

answer_relevance:
- Measures whether the answer directly addresses the user's question.
- Score range: 0.0 to 1.0
- 1.0 means fully relevant and focused.
- 0.5 means partly relevant.
- 0.0 means irrelevant.

Important rules:
- Be strict but fair.
- Do not reward unsupported extra information.
- Ignore minor wording differences.
- Return JSON only.
"""

JUDGE_USER_TEMPLATE = """Question:
{question}

Reference answer:
{gold_answer}

Required key points:
{key_points}

Model answer:
{answer}

Return JSON only in this format:
{{
  "completeness": 0.0,
  "answer_relevance": 0.0,
  "reason": "short explanation"
}}
"""


def extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from an LLM response.
    """
    if not text:
        return {}

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    return {}


def judge_answer(case: Case, answer: str) -> Dict[str, Any]:
    """
    Uses the project LLM as judge.
    """
    llm = get_llm()

    prompt = [
        ("system", JUDGE_SYSTEM),
        (
            "human",
            JUDGE_USER_TEMPLATE.format(
                question=case.question,
                gold_answer=case.gold_answer,
                key_points=json.dumps(case.key_points, ensure_ascii=False),
                answer=answer,
            ),
        ),
    ]

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        data = extract_json(content)

        completeness = safe_float(data.get("completeness"), 0.0)
        answer_relevance = safe_float(data.get("answer_relevance"), 0.0)

        completeness = max(0.0, min(1.0, completeness))
        answer_relevance = max(0.0, min(1.0, answer_relevance))

        return {
            "completeness": round(completeness, 4),
            "answer_relevance": round(answer_relevance, 4),
            "judge_reason": normalize_text(data.get("reason", "")),
        }

    except Exception as e:
        return {
            "completeness": 0.0,
            "answer_relevance": 0.0,
            "judge_reason": f"Judge failed: {e}",
        }


# ============================================================
# Run methods
# ============================================================

def run_one_method(method: str, question: str) -> Dict[str, Any]:
    start = time.perf_counter()

    if method == "llm":
        result = run_llm(question)
        num_examples = None

    elif method in FEWSHOT_METHOD_TO_N:
        num_examples = FEWSHOT_METHOD_TO_N[method]
        result = run_fewshot(question, num_examples=num_examples)

    else:
        raise ValueError(f"Unsupported method: {method}")

    latency = round(time.perf_counter() - start, 4)

    if not isinstance(result, dict):
        result = {"answer": str(result)}

    answer = normalize_text(result.get("answer", ""))

    return {
        "method": method,
        "question": question,
        "answer": answer,
        "num_examples": num_examples,
        "latency_sec": latency,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_row(case: Case, method: str, result: Dict[str, Any]) -> Dict[str, Any]:
    answer = result.get("answer", "")

    key_point_coverage = compute_key_point_coverage(
        answer=answer,
        key_points=case.key_points,
    )

    matched_key_points = get_matched_key_points(
        answer=answer,
        key_points=case.key_points,
    )

    judge_scores = judge_answer(case, answer)

    return {
        "case_id": case.id,
        "question": case.question,
        "question_type": case.question_type,
        "difficulty": case.difficulty,
        "method": method,
        "num_examples": result.get("num_examples"),
        "gold_answer": case.gold_answer,
        "answer": answer,

        # Main metrics
        "completeness": judge_scores["completeness"],
        "key_point_coverage": key_point_coverage,
        "answer_relevance": judge_scores["answer_relevance"],

        # Extra info
        "latency_sec": result.get("latency_sec", 0.0),
        "matched_key_points": matched_key_points,
        "key_points": case.key_points,
        "judge_reason": judge_scores["judge_reason"],
        "notes": case.notes,
        "expected_evidence": [asdict(x) for x in case.expected_evidence],
    }


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    summary: List[Dict[str, Any]] = []

    for method, items in grouped.items():
        summary.append(
            {
                "method": method,
                "n_cases": len(items),
                "avg_completeness": mean([safe_float(x.get("completeness")) for x in items]),
                "avg_key_point_coverage": mean([safe_float(x.get("key_point_coverage")) for x in items]),
                "avg_answer_relevance": mean([safe_float(x.get("answer_relevance")) for x in items]),
                "avg_latency_sec": mean([safe_float(x.get("latency_sec")) for x in items]),
            }
        )

    summary.sort(
        key=lambda x: METHODS.index(x["method"]) if x["method"] in METHODS else 999
    )

    return summary


def build_radar_summary(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Radar-ready CSV.
    Only keeps the three requested metrics.
    """
    radar_rows: List[Dict[str, Any]] = []

    for row in summary_rows:
        radar_rows.append(
            {
                "method": row["method"],
                "completeness": row["avg_completeness"],
                "key_point_coverage": row["avg_key_point_coverage"],
                "answer_relevance": row["avg_answer_relevance"],
                "n_cases": row["n_cases"],
                "avg_latency_sec": row["avg_latency_sec"],
            }
        )

    return radar_rows


# ============================================================
# Save helpers
# ============================================================

def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_csv_safe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe_rows: List[Dict[str, Any]] = []

    for row in rows:
        x = dict(row)

        for key in (
            "matched_key_points",
            "key_points",
            "expected_evidence",
        ):
            if key in x:
                x[key] = json.dumps(x[key], ensure_ascii=False)

        safe_rows.append(x)

    return safe_rows


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Print
# ============================================================

def print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    print("\n=== QA Benchmark Summary ===")
    print("Metrics: completeness | key_point_coverage | answer_relevance")

    for row in summary_rows:
        print(
            f"{row['method']:12s} | "
            f"comp={row['avg_completeness']:.4f} | "
            f"key={row['avg_key_point_coverage']:.4f} | "
            f"rel={row['avg_answer_relevance']:.4f} | "
            f"lat={row['avg_latency_sec']:.4f}s"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="QA benchmark: LLM vs different few-shot RAG settings"
    )

    ap.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
        help="Methods to benchmark",
    )

    ap.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="How many test cases to run. Use 0 for all cases.",
    )

    ap.add_argument(
        "--out-dir",
        type=str,
        default="benchmark_outputs_qa",
        help="Output directory",
    )

    args = ap.parse_args()

    if args.num_cases < 0:
        raise ValueError("--num-cases must be >= 0")

    cases = TEST_CASES if args.num_cases == 0 else TEST_CASES[: args.num_cases]

    print(f"Running QA benchmark on {len(cases)} cases")
    print(f"Methods: {args.methods}")
    print("Metrics: completeness, key_point_coverage, answer_relevance")

    all_rows: List[Dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        print(f"\n[{i}/{len(cases)}] {case.id} | {case.question}")

        for method in args.methods:
            print(f"  -> {method}")

            result = run_one_method(method, case.question)
            row = evaluate_row(case, method, result)

            all_rows.append(row)

            print(
                f"     comp={row['completeness']} | "
                f"key={row['key_point_coverage']} | "
                f"rel={row['answer_relevance']} | "
                f"lat={row['latency_sec']}s"
            )

    summary = aggregate_rows(all_rows)
    radar_summary = build_radar_summary(summary)

    print_summary(summary)

    out_dir = Path(args.out_dir)

    save_jsonl(out_dir / "benchmark_raw.jsonl", all_rows)
    save_csv(out_dir / "benchmark_raw.csv", make_csv_safe_rows(all_rows))

    save_csv(out_dir / "benchmark_summary_overall.csv", summary)
    save_json(out_dir / "benchmark_summary_overall.json", summary)

    save_csv(out_dir / "benchmark_summary_radar.csv", radar_summary)
    save_json(out_dir / "benchmark_summary_radar.json", radar_summary)

    print("\nSaved files:")
    print(f"- {out_dir / 'benchmark_raw.jsonl'}")
    print(f"- {out_dir / 'benchmark_raw.csv'}")
    print(f"- {out_dir / 'benchmark_summary_overall.csv'}")
    print(f"- {out_dir / 'benchmark_summary_overall.json'}")
    print(f"- {out_dir / 'benchmark_summary_radar.csv'}")
    print(f"- {out_dir / 'benchmark_summary_radar.json'}")


if __name__ == "__main__":
    main()