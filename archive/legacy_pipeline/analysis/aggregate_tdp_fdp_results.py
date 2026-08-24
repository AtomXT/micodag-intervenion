#!/usr/bin/env python3
"""Combine trial fragments and create three-method comparison outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_tdp_fdp_experiments import (
    GNIES_GRIDS,
    MIP_GRIDS,
    UTIGSP_GRID,
)


DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / "experiment_results" / "tdp_fdp",
    PROJECT_ROOT / "experiment_results" / "tdp_fdp_utigsp" / "parts",
]
BEST_DCPDAG_TABLE = (
    PROJECT_ROOT / "experiment_results" / "tdp_fdp" / "best_dcpdag_summary.csv"
)
BEST_DCPDAG_LATEX_TABLE = (
    PROJECT_ROOT / "experiment_results" / "tdp_fdp" / "best_dcpdag_table.tex"
)
METHOD_LABELS = {
    "mip_profiled": "PS-MIP",
    "gnies": "GnIES",
    "utigsp": "UT-IGSP",
}
METHOD_ORDER = ["PS-MIP", "GnIES", "UT-IGSP"]
NUMERIC_COLUMNS = [
    "graph", "trial", "penalty", "target_penalty", "dcdi_hidden_dim",
    "dcdi_mu_init", "dcdi_max_iterations", "fdp", "tdp", "d_cpdag",
    "union_target_error", "fit_seconds", "metric_seconds", "mip_gap",
]


def _trial_paths(input_dirs):
    paths = {
        path.resolve()
        for input_dir in input_dirs
        if input_dir.exists()
        for path in input_dir.rglob("trial_*.csv")
    }
    return sorted(paths)


def _read_results(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "target_penalty" not in frame:
            frame["target_penalty"] = np.nan
        if "fit_seconds" not in frame and "fit_wall_seconds" in frame:
            frame["fit_seconds"] = frame["fit_wall_seconds"]
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True, sort=False)
    for column in NUMERIC_COLUMNS:
        if column not in results:
            results[column] = np.nan
        results[column] = pd.to_numeric(results[column], errors="coerce")
    return results


def _expected_results():
    rows = []
    for trial in range(1, 11):
        for graph in (1, 2, 3):
            for method, grid in (
                ("mip_profiled", MIP_GRIDS[graph]),
                ("gnies", GNIES_GRIDS[graph]),
                ("utigsp", UTIGSP_GRID),
            ):
                rows.extend(
                    dict(
                        method=method,
                        graph=graph,
                        trial=trial,
                        penalty=penalty,
                        target_penalty=np.nan,
                    )
                    for penalty in grid
                )
    return pd.DataFrame(rows)


def _with_comparison_keys(frame):
    keyed = frame.copy()
    for column in (
        "target_penalty",
        "dcdi_hidden_dim",
        "dcdi_mu_init",
        "dcdi_max_iterations",
    ):
        if column not in keyed:
            keyed[column] = np.nan
    keyed["_target_key"] = keyed["target_penalty"].fillna(-1.0)
    keyed["_hidden_key"] = keyed["dcdi_hidden_dim"].fillna(-1.0)
    keyed["_mu_key"] = keyed["dcdi_mu_init"].fillna(-1.0)
    keyed["_iterations_key"] = keyed["dcdi_max_iterations"].fillna(-1.0)
    return keyed


def _curve(method_results):
    method_results = method_results.copy()
    method_results["degenerate"] = (
        method_results[["fdp", "tdp"]].abs().le(1e-12).all(axis=1)
    )
    summary = (
        method_results.groupby("penalty", as_index=False)
        .agg(
            fdp=("fdp", "mean"),
            tdp=("tdp", "mean"),
            degenerate_rate=("degenerate", "mean"),
        )
        .sort_values("penalty")
    )
    return summary[summary["degenerate_rate"] < 0.5]


def _plot_path(axis, method_results, *, color, marker, label, linestyle="-"):
    summary = _curve(method_results)
    if summary.empty:
        return
    axis.plot(
        summary["fdp"],
        summary["tdp"],
        color=color,
        marker=marker,
        linestyle=linestyle,
        label=label,
    )


def _plot(successful, figure_dir):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    styles = {
        "mip_profiled": ("tab:blue", "o", "PS-MIP"),
        "gnies": ("tab:orange", "s", "GnIES"),
        "utigsp": ("tab:red", "D", "UT-IGSP"),
    }
    for graph, axis in zip((1, 2, 3), axes):
        subset = successful[successful["graph"] == graph]
        for method, (color, marker, label) in styles.items():
            method_results = subset[subset["method"] == method].copy()
            if method_results.empty:
                continue
            _plot_path(
                axis,
                method_results,
                color=color,
                marker=marker,
                label=label,
            )
        axis.set_title(f"Graph {graph}")
        axis.set_xlabel("FDP")
        axis.grid(alpha=0.25)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("TDP")
    handles, _ = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend()
    figure.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / "tdp_fdp.pdf")
    figure.savefig(figure_dir / "tdp_fdp.png", dpi=200)
    plt.close(figure)


def _best_dcpdag_summary(successful):
    """Apply the manuscript's trial-wise best-penalty selection rule."""
    selected = (
        successful.sort_values(
            ["method", "graph", "trial", "d_cpdag", "fit_seconds", "penalty"]
        )
        .drop_duplicates(["method", "graph", "trial"], keep="first")
        .copy()
    )
    trial_counts = selected.groupby(["method", "graph"])["trial"].nunique()
    incomplete = trial_counts[trial_counts != 10]
    if not incomplete.empty:
        raise RuntimeError(
            "cannot generate the best-d_cpdag table without ten trials for "
            f"every method and graph:\n{incomplete.to_string()}"
        )

    selected["rgap_percent"] = 100.0 * selected["mip_gap"]
    table = (
        selected.groupby(["graph", "method"], as_index=False)
        .agg(
            time_seconds=("fit_seconds", "mean"),
            tdp=("tdp", "mean"),
            fdp=("fdp", "mean"),
            d_cpdag=("d_cpdag", "mean"),
            rgap_percent=("rgap_percent", "mean"),
        )
    )
    table["method"] = table["method"].map(METHOD_LABELS)
    table["_method_order"] = table["method"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )
    table = table.sort_values(["graph", "_method_order"]).drop(
        columns="_method_order"
    )
    table["time_seconds"] = table["time_seconds"].round(2)
    table[["tdp", "fdp"]] = table[["tdp", "fdp"]].round(3)
    table["d_cpdag"] = table["d_cpdag"].round(1)
    table["rgap_percent"] = table["rgap_percent"].round(4)
    return table


def _write_best_dcpdag_summary(table, path):
    """Write the table with the same precision used in the manuscript."""
    formatted = table.copy()
    formatted["time_seconds"] = formatted["time_seconds"].map(lambda x: f"{x:.2f}")
    formatted["tdp"] = formatted["tdp"].map(lambda x: f"{x:.3f}")
    formatted["fdp"] = formatted["fdp"].map(lambda x: f"{x:.3f}")
    formatted["d_cpdag"] = formatted["d_cpdag"].map(lambda x: f"{x:.1f}")
    formatted["rgap_percent"] = formatted["rgap_percent"].map(
        lambda x: "" if pd.isna(x) else f"{x:.4f}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(path, index=False)


def _write_best_dcpdag_latex_table(table, path):
    """Write the complete tabular block consumed directly by the manuscript."""
    lines = [
        r"\begin{tabular}{clrrrrr}",
        r"    \hline",
        r"    Graph & Method & Time (s) & TDP & FDP &",
        r"    \(d_{\mathrm{CPDAG}}\) & RGAP (\%) \\",
        r"    \hline",
    ]
    for result in table.itertuples(index=False):
        rgap = "--" if pd.isna(result.rgap_percent) else f"{result.rgap_percent:.0f}"
        lines.append(
            f"    {result.graph} & {result.method} & {result.time_seconds:.2f} & "
            f"{result.tdp:.3f} & {result.fdp:.3f} & {result.d_cpdag:.1f} & "
            f"{rgap} \\\\"
        )
    lines.extend([r"    \hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, nargs="+", default=DEFAULT_INPUT_DIRS
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment_results" / "tdp_fdp_comparison",
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--table-output", type=Path, default=BEST_DCPDAG_TABLE
    )
    parser.add_argument(
        "--latex-table-output", type=Path, default=BEST_DCPDAG_LATEX_TABLE
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    paths = _trial_paths(args.input_dir)
    if not paths:
        joined = ", ".join(str(path) for path in args.input_dir)
        raise FileNotFoundError(f"no trial CSV files below: {joined}")

    results = _read_results(paths)
    expected = _expected_results()
    keyed_results = _with_comparison_keys(results)
    keyed_expected = _with_comparison_keys(expected)
    helper_keys = ["_target_key", "_hidden_key", "_mu_key", "_iterations_key"]
    merge_keys = ["method", "graph", "trial", "penalty", *helper_keys]

    result_key_index = pd.MultiIndex.from_frame(keyed_results[merge_keys])
    expected_key_index = pd.MultiIndex.from_frame(keyed_expected[merge_keys])
    expected_mask = result_key_index.isin(expected_key_index)

    comparison_keyed = keyed_results.loc[expected_mask].copy()
    unexpected_keyed = keyed_results.loc[~expected_mask].copy()
    observed_keys = comparison_keyed[merge_keys].drop_duplicates()
    missing = (
        keyed_expected.merge(observed_keys, on=merge_keys, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns=["_merge", *helper_keys])
    )
    duplicate_mask = comparison_keyed.duplicated(merge_keys, keep=False)
    duplicates = comparison_keyed.loc[duplicate_mask].drop(columns=helper_keys)
    duplicate_key_count = len(
        comparison_keyed.loc[duplicate_mask, merge_keys].drop_duplicates()
    )
    comparison = comparison_keyed.drop(columns=helper_keys)
    unexpected = unexpected_keyed.drop(columns=helper_keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing.to_csv(args.output_dir / "missing_combinations.csv", index=False)
    unexpected.to_csv(args.output_dir / "unexpected_combinations.csv", index=False)
    duplicates.to_csv(
        args.output_dir / "duplicate_expected_combinations.csv", index=False
    )
    results.to_csv(args.output_dir / "raw_results.csv", index=False)
    comparison.to_csv(args.output_dir / "combined_results.csv", index=False)
    successful = comparison[
        comparison["status"].isin(["ok", "ok_with_gap"])
    ].copy()

    status_counts = comparison.groupby(["method", "graph", "status"]).size()
    lines = [
        f"input directories: {len(args.input_dir)}",
        f"trial fragments: {len(paths)}",
        f"expected comparison rows: {len(expected)}",
        f"raw observed rows: {len(results)}",
        f"comparison rows: {len(comparison)}",
        f"missing combinations: {len(missing)}",
        f"unexpected rows: {len(unexpected)}",
        f"successful comparison rows: {len(successful)}",
        f"duplicate expected keys: {duplicate_key_count}",
        f"rows belonging to duplicate expected keys: {len(duplicates)}",
        "",
        "comparison status counts:",
        status_counts.to_string() if not status_counts.empty else "(none)",
    ]
    report = "\n".join(lines)
    (args.output_dir / "validation_report.txt").write_text(report + "\n")
    if duplicate_key_count:
        print(report)
        raise RuntimeError(
            f"found {duplicate_key_count} duplicate expected comparison keys; "
            f"see {args.output_dir / 'duplicate_expected_combinations.csv'}"
        )

    summary = (
        successful.groupby(
            ["method", "graph", "penalty", "target_penalty"],
            as_index=False,
            dropna=False,
        )
        .agg(
            trials=("trial", "nunique"),
            fdp_mean=("fdp", "mean"),
            fdp_std=("fdp", "std"),
            tdp_mean=("tdp", "mean"),
            tdp_std=("tdp", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_median=("fit_seconds", "median"),
            metric_seconds_mean=("metric_seconds", "mean"),
            mip_gap_max=("mip_gap", "max"),
        )
    )
    summary.to_csv(args.output_dir / "curve_summary.csv", index=False)
    best_dcpdag = _best_dcpdag_summary(successful)
    _write_best_dcpdag_summary(best_dcpdag, args.table_output)
    _write_best_dcpdag_latex_table(best_dcpdag, args.latex_table_output)
    _plot(successful, args.figure_dir)
    print(report)


if __name__ == "__main__":
    main()
