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
from .few_shot_chain import build_few_shot_rag_chain as build_cot_chain
from .few_shot_no_cot import build_few_shot_rag_chain as build_no_cot_chain
from .rag_chain import get_llm


METHODS: Tuple[str, ...] = (
    "fewshot_cot",
    "fewshot_no_cot",
)


# ============================================================
# Basic utilities
# ============================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        value = float(x)
        return max(0.0, min(1.0, value))
    except Exception:
        return default


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.mean(values), 4)


def extract_json(text: str) -> Dict[str, Any]:
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


def make_csv_safe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe_rows: List[Dict[str, Any]] = []

    for row in rows:
        x = dict(row)
        for key in (
            "key_points",
            "expected_evidence",
            "retrieved_contexts",
        ):
            if key in x:
                x[key] = json.dumps(x[key], ensure_ascii=False)
        safe_rows.append(x)

    return safe_rows


# ============================================================
# Run methods
# ============================================================

def run_one_method(method: str, question: str, num_examples: int) -> Dict[str, Any]:
    start = time.perf_counter()

    if method == "fewshot_cot":
        chain = build_cot_chain(num_examples=num_examples)
    elif method == "fewshot_no_cot":
        chain = build_no_cot_chain(num_examples=num_examples)
    else:
        raise ValueError(f"Unsupported method: {method}")

    out = chain.invoke({"input": question})
    latency = round(time.perf_counter() - start, 4)

    if not isinstance(out, dict):
        out = {"answer": str(out)}

    answer = normalize_text(out.get("answer", ""))

    return {
        "method": method,
        "question": question,
        "answer": answer,
        "context": out.get("context", ""),
        "sources": out.get("sources", []),
        "num_examples": num_examples,
        "latency_sec": latency,
    }


# ============================================================
# LLM judge:
# reasoning quality + completeness
# ============================================================

JUDGE_SYSTEM = """You are an impartial evaluator for a Building Technology QA benchmark.

You evaluate the visible final answer only.

Metrics:

1. reasoning_quality
- Measures whether the answer gives a clear, logical, well-structured justification.
- Good reasoning quality means:
  - the conclusion is supported by explanation,
  - important numbers or comparisons are interpreted correctly,
  - the answer separates conclusion, explanation, evidence, assumptions, or limitations when useful,
  - the answer does not just state a short claim without justification.
- Score range: 0.0 to 1.0.
- This is the most important metric.

2. completeness
- Measures whether the answer covers the reference answer and required key points.
- Score range: 0.0 to 1.0.
- 1.0 means all important points are covered.
- 0.5 means partially covered.
- 0.0 means mostly missing or incorrect.

Important rules:
- Be strict but fair.
- Do not reward irrelevant extra text.
- Do not require hidden chain-of-thought.
- Judge only the visible explanation quality in the final answer.
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
  "reasoning_quality": 0.0,
  "completeness": 0.0,
  "reason": "short explanation"
}}
"""


def judge_answer(case: Case, answer: str) -> Dict[str, Any]:
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

        reasoning_quality = safe_float(data.get("reasoning_quality"), 0.0)
        completeness = safe_float(data.get("completeness"), 0.0)

        return {
            "reasoning_quality": round(reasoning_quality, 4),
            "completeness": round(completeness, 4),
            "judge_reason": normalize_text(data.get("reason", "")),
        }

    except Exception as e:
        return {
            "reasoning_quality": 0.0,
            "completeness": 0.0,
            "judge_reason": f"Judge failed: {e}",
        }


# ============================================================
# Evaluation
# ============================================================

def evaluate_row(case: Case, method: str, result: Dict[str, Any]) -> Dict[str, Any]:
    answer = result.get("answer", "")
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
        "reasoning_quality": judge_scores["reasoning_quality"],
        "completeness": judge_scores["completeness"],

        # Latency raw value
        "latency_sec": result.get("latency_sec", 0.0),

        # Extra info
        "judge_reason": judge_scores["judge_reason"],
        "key_points": case.key_points,
        "expected_evidence": [asdict(x) for x in case.expected_evidence],
        "notes": case.notes,
        "sources": result.get("sources", []),
    }


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    summary: List[Dict[str, Any]] = []

    for method, items in grouped.items():
        avg_latency = mean([float(x.get("latency_sec", 0.0)) for x in items])

        summary.append(
            {
                "method": method,
                "n_cases": len(items),
                "avg_reasoning_quality": mean(
                    [safe_float(x.get("reasoning_quality")) for x in items]
                ),
                "avg_completeness": mean(
                    [safe_float(x.get("completeness")) for x in items]
                ),
                "avg_latency_sec": avg_latency,
            }
        )

    summary.sort(
        key=lambda x: METHODS.index(x["method"]) if x["method"] in METHODS else 999
    )

    return summary


def add_latency_score(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert latency into a radar-friendly score.
    Higher is better.

    Formula:
        latency_score = fastest_avg_latency / current_avg_latency

    Fastest method gets 1.0.
    Slower methods get lower scores.
    """
    latencies = [
        float(row["avg_latency_sec"])
        for row in summary_rows
        if float(row["avg_latency_sec"]) > 0
    ]

    if not latencies:
        for row in summary_rows:
            row["latency_score"] = 0.0
        return summary_rows

    fastest = min(latencies)

    for row in summary_rows:
        latency = float(row["avg_latency_sec"])
        if latency <= 0:
            score = 0.0
        else:
            score = fastest / latency

        row["latency_score"] = round(max(0.0, min(1.0, score)), 4)

    return summary_rows


def build_radar_summary(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    radar_rows: List[Dict[str, Any]] = []

    for row in summary_rows:
        radar_rows.append(
            {
                "method": row["method"],
                "reasoning_quality": row["avg_reasoning_quality"],
                "completeness": row["avg_completeness"],
                "latency_score": row["latency_score"],
                "avg_latency_sec": row["avg_latency_sec"],
                "n_cases": row["n_cases"],
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
    print("\n=== Few-shot CoT vs No-CoT Benchmark Summary ===")
    print("Metrics: reasoning_quality ↑ | completeness ↑ | latency_score ↑")

    for row in summary_rows:
        print(
            f"{row['method']:16s} | "
            f"reason={row['avg_reasoning_quality']:.4f} | "
            f"comp={row['avg_completeness']:.4f} | "
            f"lat_score={row['latency_score']:.4f} | "
            f"lat={row['avg_latency_sec']:.4f}s"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark: few-shot CoT vs few-shot no-CoT"
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
        "--num-examples",
        type=int,
        default=7,
        help="Number of few-shot examples for both CoT and no-CoT.",
    )

    ap.add_argument(
        "--out-dir",
        type=str,
        default="benchmark_outputs_fewshot_cot_compare",
        help="Output directory",
    )

    args = ap.parse_args()

    if args.num_cases < 0:
        raise ValueError("--num-cases must be >= 0")

    if args.num_examples < 0:
        raise ValueError("--num-examples must be >= 0")

    cases = TEST_CASES if args.num_cases == 0 else TEST_CASES[: args.num_cases]

    print(f"Running Few-shot CoT comparison on {len(cases)} cases")
    print(f"Methods: {args.methods}")
    print(f"Few-shot examples: {args.num_examples}")
    print("Metrics: reasoning_quality, completeness, latency_score")

    all_rows: List[Dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        print(f"\n[{i}/{len(cases)}] {case.id} | {case.question}")

        for method in args.methods:
            print(f"  -> {method}")

            result = run_one_method(
                method=method,
                question=case.question,
                num_examples=args.num_examples,
            )

            row = evaluate_row(case, method, result)
            all_rows.append(row)

            print(
                f"     reason={row['reasoning_quality']} | "
                f"comp={row['completeness']} | "
                f"lat={row['latency_sec']}s"
            )

    summary = aggregate_rows(all_rows)
    summary = add_latency_score(summary)
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