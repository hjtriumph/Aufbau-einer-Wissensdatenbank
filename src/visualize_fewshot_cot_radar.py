from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = "benchmark_outputs_fewshot_cot_compare/benchmark_summary_radar.csv"
DEFAULT_OUTPUT = "benchmark_outputs_fewshot_cot_compare/radar_fewshot_cot_compare.png"

DEFAULT_METRICS = [
    "reasoning_quality",
    "completeness",
    "latency_score",
]

DEFAULT_METHOD_ORDER = [
    "fewshot_cot",
    "fewshot_no_cot",
]

DISPLAY_NAME_MAP = {
    "reasoning_quality": "reasoning\nquality",
    "completeness": "completeness",
    "latency_score": "latency\nscore",
}


def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    if "method" not in df.columns:
        raise ValueError("CSV must contain a 'method' column.")

    return df


def validate_metrics(df: pd.DataFrame, metrics: List[str]) -> None:
    missing = [m for m in metrics if m not in df.columns]

    if missing:
        raise ValueError(
            f"Missing metric columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def order_methods(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    existing = df["method"].astype(str).tolist()
    ordered = [m for m in DEFAULT_METHOD_ORDER if m in existing]
    remaining = [m for m in existing if m not in ordered]

    final_order = ordered + remaining

    df["__order"] = df["method"].astype(str).apply(
        lambda x: final_order.index(x)
    )

    return df.sort_values("__order").drop(columns="__order")


def make_angles(n_axes: int) -> List[float]:
    angles = [i / float(n_axes) * 2 * math.pi for i in range(n_axes)]
    return angles + angles[:1]


def close_values(values: List[float]) -> List[float]:
    return values + values[:1]


def metric_labels(metrics: List[str]) -> List[str]:
    return [DISPLAY_NAME_MAP.get(m, m) for m in metrics]

def adjust_axis_label_positions(ax, angles: List[float]) -> None:
    for label, angle in zip(ax.get_xticklabels(), angles):
        display_angle = math.pi / 2 - angle
        x_direction = math.cos(display_angle)
        y_direction = math.sin(display_angle)

        if x_direction > 0.25:
            label.set_horizontalalignment("left")
        elif x_direction < -0.25:
            label.set_horizontalalignment("right")
        else:
            label.set_horizontalalignment("center")

        if y_direction > 0.25:
            label.set_verticalalignment("bottom")
        elif y_direction < -0.25:
            label.set_verticalalignment("top")
        else:
            label.set_verticalalignment("center")


def plot_radar(
    df: pd.DataFrame,
    metrics: List[str],
    output_path: str,
    title: str,
    show: bool = False,
) -> None:
    if len(metrics) < 3:
        raise ValueError("Radar chart needs at least 3 metrics.")

    angles = make_angles(len(metrics))

    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(
        angles[:-1],
        metric_labels(metrics),
        fontsize=16,
    )

    adjust_axis_label_positions(ax, angles[:-1])

    ax.set_rlabel_position(0)
    plt.yticks(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        fontsize=13,
    )
    plt.ylim(0, 1)

    for _, row in df.iterrows():
        method = str(row["method"])

        values = []
        for metric in metrics:
            value = row[metric]
            if pd.isna(value):
                value = 0.0
            values.append(float(value))

        values = close_values(values)

        ax.plot(
            angles,
            values,
            linewidth=2,
            label=method,
        )
        ax.fill(
            angles,
            values,
            alpha=0.10,
        )
    ax.tick_params(axis="x", pad=38)
    ax.tick_params(axis="y", pad=8)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.84, bottom=0.28)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(3, len(df)),
        frameon=False,
        fontsize=13,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.45)

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Radar chart for Few-shot CoT vs No-CoT benchmark"
    )

    ap.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to benchmark_summary_radar.csv",
    )

    ap.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output image path",
    )

    ap.add_argument(
        "--title",
        type=str,
        default="Few-shot CoT vs No-CoT Radar Chart",
        help="Radar chart title",
    )

    ap.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric columns to plot",
    )

    ap.add_argument(
        "--show",
        action="store_true",
        help="Show plot window",
    )

    args = ap.parse_args()

    df = load_data(args.input)
    validate_metrics(df, args.metrics)
    df = order_methods(df)

    plot_radar(
        df=df,
        metrics=args.metrics,
        output_path=args.output,
        title=args.title,
        show=args.show,
    )

    print(f"Radar chart saved to: {args.output}")


if __name__ == "__main__":
    main()

