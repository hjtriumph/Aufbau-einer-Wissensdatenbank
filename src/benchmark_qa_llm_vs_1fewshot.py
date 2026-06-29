from __future__ import annotations

from .benchmark_qa import (
    main as _unused_main,
    run_one_method,
    evaluate_row,
    aggregate_rows,
    build_radar_summary,
    print_summary,
    save_jsonl,
    save_csv,
    save_json,
    make_csv_safe_rows,
)
from .benchmark_cases import TEST_CASES
from pathlib import Path
import argparse
from typing import Any, Dict, List


METHODS = ("llm", "fewshot_1")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="QA benchmark: LLM vs 1 few-shot RAG"
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
        default="benchmark_outputs_qa_llm_vs_1fewshot",
        help="Output directory",
    )

    args = ap.parse_args()

    if args.num_cases < 0:
        raise ValueError("--num-cases must be >= 0")

    cases = TEST_CASES if args.num_cases == 0 else TEST_CASES[: args.num_cases]

    print(f"Running QA benchmark on {len(cases)} cases")
    print(f"Methods: {METHODS}")
    print("Metrics: completeness, key_point_coverage, answer_relevance")

    all_rows: List[Dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        print(f"\n[{i}/{len(cases)}] {case.id} | {case.question}")

        for method in METHODS:
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