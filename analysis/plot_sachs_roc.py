#!/usr/bin/env python3
"""Create count-scale and normalized ROC plots for the Sachs experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_sachs_roc as runner
from analysis.aggregate_sachs_roc import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DIRECTED_NEGATIVES,
    DIRECTED_POSITIVES,
    SKELETON_NEGATIVES,
    SKELETON_POSITIVES,
    ValidationError,
    aggregate_results,
)
from src.sachs_data import DEFAULT_DATA_ROOT


METHOD_STYLES: Mapping[str, Tuple[str, str]] = {
    "ps_mip_unknown": ("#1f77b4", "o"),
    "utigsp_unknown": ("#d62728", "D"),
    "utigsp_intended": ("#9467bd", "P"),
    "gnies_unknown": ("#ff7f0e", "s"),
    "igsp_oracle": ("#8c564b", "<"),
    "gies_oracle": ("#2ca02c", "v"),
    "ps_mip_oracle": ("#17becf", "^"),
}
FALLBACK_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
FALLBACK_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "<", ">", "h")


def _normalise_methods(methods: Optional[Sequence[str]]) -> Optional[List[str]]:
    if methods is None:
        return None
    values: List[str] = []
    for value in methods:
        values.extend(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(values) - set(runner.METHODS))
    if unknown:
        raise ValueError(f"unknown method(s): {unknown}")
    if not values:
        raise ValueError("at least one method must be selected")
    requested = set(values)
    return [method for method in runner.METHODS if method in requested]


def _method_style(method: str, rank: int) -> Tuple[str, str]:
    return METHOD_STYLES.get(
        method,
        (
            FALLBACK_COLORS[rank % len(FALLBACK_COLORS)],
            FALLBACK_MARKERS[rank % len(FALLBACK_MARKERS)],
        ),
    )


def _atomic_savefig(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    metadata: Dict[str, Any] = {"Creator": "micodag Sachs ROC analysis"}
    if path.suffix.lower() == ".pdf":
        metadata.update({"CreationDate": None, "ModDate": None})
    try:
        fig.savefig(
            temporary_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _validate_points(points: pd.DataFrame) -> None:
    required = {
        "method",
        "method_label",
        "setting_index",
        "tuning_parameter",
        "tuning_value",
        "paper_directed_tp",
        "paper_directed_fp",
        "paper_skeleton_tp",
        "paper_skeleton_fp",
        "directed_tpr",
        "directed_fpr",
        "skeleton_tpr",
        "skeleton_fpr",
    }
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"roc_points.csv is missing columns: {missing}")
    if points.empty:
        raise ValueError("roc_points.csv contains no validated operating points")
    numeric = sorted(required - {"method", "method_label", "tuning_parameter", "tuning_value"})
    for column in numeric:
        points[column] = pd.to_numeric(points[column], errors="raise")


def _legend_handles(
    methods: Sequence[str], show_nonoptimal: bool
) -> List[Line2D]:
    handles: List[Line2D] = []
    for rank, method in enumerate(methods):
        color, marker = _method_style(method, rank)
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=marker,
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                color=color,
                label=str(runner.METHOD_LABELS.get(method, method)),
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            color="0.45",
            linestyle=(0, (3, 3)),
            linewidth=1.2,
            label="Random reference",
        )
    )
    if show_nonoptimal:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="x",
                markersize=7,
                markeredgewidth=1.2,
                color="black",
                label="Feasible, nonoptimal PS-MIP",
            )
        )
    return handles


def _scatter_methods(
    ax: Any,
    points: pd.DataFrame,
    methods: Sequence[str],
    x_column: str,
    y_column: str,
    annotate: bool,
) -> None:
    for rank, method in enumerate(methods):
        subset = points.loc[points["method"] == method].sort_values(
            "setting_index", kind="stable"
        )
        if subset.empty:
            continue
        color, marker = _method_style(method, rank)
        ax.scatter(
            subset[x_column],
            subset[y_column],
            s=46,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            alpha=0.92,
            zorder=3,
        )
        nonoptimal = subset.loc[subset["is_nonoptimal"]]
        if not nonoptimal.empty:
            ax.scatter(
                nonoptimal[x_column],
                nonoptimal[y_column],
                s=64,
                marker="x",
                color="black",
                linewidths=1.2,
                zorder=4,
            )
        if annotate:
            for _, row in subset.iterrows():
                label = f"{row['tuning_value']}"
                ax.annotate(
                    label,
                    (row[x_column], row[y_column]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=color,
                )


def _finish_figure(
    fig: Any, methods: Sequence[str], title: str, show_nonoptimal: bool
) -> None:
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.legend(
        handles=_legend_handles(methods, show_nonoptimal),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(4, len(methods) + 1 + int(show_nonoptimal)),
        frameon=False,
        fontsize=8.5,
        handletextpad=0.45,
        columnspacing=1.1,
    )
    legend_items = len(methods) + 1 + int(show_nonoptimal)
    legend_rows = (legend_items + 3) // 4
    axes_top = 0.78 - max(0, legend_rows - 2) * 0.065
    fig.subplots_adjust(
        top=axes_top, bottom=0.15, left=0.08, right=0.98, wspace=0.25
    )


def plot_roc_points(
    points: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT,
    methods: Optional[Sequence[str]] = None,
    annotate: bool = False,
    dpi: int = 220,
    validation_status: str = "complete",
) -> List[Path]:
    """Write count-scale and normalized two-panel ROC figures."""

    points = points.copy()
    _validate_points(points)
    selected_methods = _normalise_methods(methods)
    if selected_methods is None:
        observed = set(points["method"])
        selected_methods = [method for method in runner.METHODS if method in observed]
    points = points.loc[points["method"].isin(selected_methods)].copy()
    if points.empty:
        raise ValueError("none of the selected methods has a validated ROC point")
    observed = set(points["method"])
    selected_methods = [method for method in selected_methods if method in observed]
    if "is_nonoptimal" in points:
        points["is_nonoptimal"] = points["is_nonoptimal"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
    elif "status" in points:
        points["is_nonoptimal"] = points["status"].eq("ok_nonoptimal")
    else:
        points["is_nonoptimal"] = False
    show_nonoptimal = bool(points["is_nonoptimal"].any())
    title_suffix = (
        "" if validation_status == "complete" else " [PARTIAL - incomplete validation]"
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.color": "0.88",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    count_figure, count_axes = plt.subplots(1, 2, figsize=(9.2, 4.15))
    count_panels = (
        (
            count_axes[0],
            "paper_directed_fp",
            "paper_directed_tp",
            DIRECTED_NEGATIVES,
            DIRECTED_POSITIVES,
            "Directed edges",
        ),
        (
            count_axes[1],
            "paper_skeleton_fp",
            "paper_skeleton_tp",
            SKELETON_NEGATIVES,
            SKELETON_POSITIVES,
            "Skeleton edges",
        ),
    )
    for ax, x_column, y_column, negatives, positives, title in count_panels:
        ax.plot(
            [0, negatives],
            [0, positives],
            color="0.45",
            linestyle=(0, (3, 3)),
            linewidth=1.2,
            zorder=1,
        )
        _scatter_methods(
            ax, points, selected_methods, x_column, y_column, annotate
        )
        ax.set_title(title)
        ax.set_xlabel(f"False positives (of {negatives})")
        ax.set_ylabel(f"True positives (of {positives})")
        x_max = 45 if title == "Directed edges" else negatives
        ax.set_xlim(-0.025 * x_max, 1.025 * x_max)
        ax.set_ylim(-0.04 * positives, 1.04 * positives)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    _finish_figure(
        count_figure,
        selected_methods,
        "Sachs et al. (2005): TP-FP operating points" + title_suffix,
        show_nonoptimal,
    )

    rate_figure, rate_axes = plt.subplots(1, 2, figsize=(9.2, 4.15))
    rate_panels = (
        (rate_axes[0], "directed_fpr", "directed_tpr", "Directed edges"),
        (rate_axes[1], "skeleton_fpr", "skeleton_tpr", "Skeleton edges"),
    )
    for ax, x_column, y_column, title in rate_panels:
        ax.plot(
            [0, 1],
            [0, 1],
            color="0.45",
            linestyle=(0, (3, 3)),
            linewidth=1.2,
            zorder=1,
        )
        _scatter_methods(
            ax, points, selected_methods, x_column, y_column, annotate
        )
        ax.set_title(title)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.025, 1.025)
        ax.set_aspect("equal", adjustable="box")
    _finish_figure(
        rate_figure,
        selected_methods,
        "Sachs et al. (2005): normalized ROC operating points" + title_suffix,
        show_nonoptimal,
    )

    output_dir = Path(output_dir).expanduser().resolve()
    outputs = [
        output_dir / "sachs_roc_counts.png",
        output_dir / "sachs_roc_counts.pdf",
        output_dir / "sachs_roc_rates.png",
        output_dir / "sachs_roc_rates.pdf",
    ]
    try:
        for path in outputs[:2]:
            _atomic_savefig(count_figure, path, dpi)
        for path in outputs[2:]:
            _atomic_savefig(rate_figure, path, dpi)
    finally:
        plt.close(count_figure)
        plt.close(rate_figure)
    return outputs


def _read_validated_summary(
    summary_dir: Path, allow_incomplete: bool
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report_path = summary_dir / "validation_report.json"
    points_path = summary_dir / "roc_points.csv"
    if not report_path.is_file() or not points_path.is_file():
        raise ValueError(
            "validated summary is missing; run without --skip-aggregation first"
        )
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    status = report.get("status")
    if status != "complete" and not allow_incomplete:
        raise ValueError(
            f"validation status is {status!r}; pass --allow-incomplete only for "
            "an explicitly partial diagnostic plot"
        )
    return pd.read_csv(points_path), report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Sachs fragments and create TP-FP and normalized ROC "
            "scatter plots."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=f"CSV files or directories (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--methods",
        nargs="+",
        help="subset of methods; comma-separated values are also accepted",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="permit a clearly marked partial diagnostic plot",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="plot an existing validated roc_points.csv and validation report",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="label each point with its tuning value",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    selected_methods = _normalise_methods(args.methods)
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        if args.skip_aggregation:
            points, report = _read_validated_summary(
                output_dir, args.allow_incomplete
            )
        else:
            _combined, points, report = aggregate_results(
                inputs=args.inputs or None,
                output_dir=output_dir,
                data_root=args.data_root,
                methods=selected_methods,
                allow_incomplete=args.allow_incomplete,
            )
        outputs = plot_roc_points(
            points,
            output_dir=output_dir,
            methods=selected_methods,
            annotate=args.annotate,
            dpi=args.dpi,
            validation_status=str(report.get("status", "unknown")),
        )
    except (ValidationError, ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Sachs ROC plots ({report.get('status', 'unknown')} validation):"
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
