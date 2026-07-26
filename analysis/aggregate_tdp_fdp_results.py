#!/usr/bin/env python3
"""Combine trial results and create FDP/TDP plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_tdp_fdp_experiments import GNIES_GRIDS, MIP_GRIDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment_results" / "tdp_fdp",
    )
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("trial_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no trial CSV files in {args.input_dir}")

    results = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    keys = ["method", "graph", "trial", "penalty"]
    duplicate_count = int(results.duplicated(keys).sum())
    expected_rows = [
        {"method": method, "graph": graph, "trial": trial, "penalty": penalty}
        for trial in range(1, 11)
        for graph in (1, 2, 3)
        for method, grid in (
            ("mip_profiled", MIP_GRIDS[graph]),
            ("gnies", GNIES_GRIDS[graph]),
        )
        for penalty in grid
    ]
    expected_frame = pd.DataFrame(expected_rows)
    observed_keys = results[keys].drop_duplicates()
    missing = (
        expected_frame.merge(observed_keys, on=keys, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
    )
    missing.to_csv(args.input_dir / "missing_combinations.csv", index=False)
    results.to_csv(args.input_dir / "combined_results.csv", index=False)
    successful = results[results["status"].isin(["ok", "ok_with_gap"])].copy()
    summary = (
        successful.groupby(["method", "graph", "penalty"], as_index=False)
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
    summary.to_csv(args.input_dir / "curve_summary.csv", index=False)

    lines = [
        f"trial files: {len(paths)}",
        f"expected rows: {len(expected_frame)}",
        f"observed rows: {len(results)}",
        f"missing combinations: {len(missing)}",
        f"successful rows: {len(successful)}",
        f"duplicate rows: {duplicate_count}",
        "",
        "status counts:",
        results.groupby(["method", "graph", "status"]).size().to_string(),
    ]
    (args.input_dir / "validation_report.txt").write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    styles = {
        "mip_profiled": ("tab:blue", "o", "MIP-profiled"),
        "gnies": ("tab:orange", "s", "GnIES"),
    }
    for graph, axis in zip((1, 2, 3), axes):
        subset = successful[successful["graph"] == graph]
        for method, (color, marker, label) in styles.items():
            method_results = subset[subset["method"] == method].copy()
            if graph == 1:
                endpoint = {"mip_profiled": 0.64, "gnies": 100}[method]
                method_results = method_results[method_results["penalty"] != endpoint]
            method_results["degenerate"] = (
                method_results[["fdp", "tdp"]].abs().le(1e-12).all(axis=1)
            )
            method_summary = (
                method_results
                .groupby("penalty", as_index=False)
                .agg(
                    fdp=("fdp", "mean"),
                    tdp=("tdp", "mean"),
                    degenerate_rate=("degenerate", "mean"),
                )
                .sort_values("penalty")
            )
            method_summary = method_summary[method_summary["degenerate_rate"] < 0.5]
            axis.plot(
                method_summary["fdp"],
                method_summary["tdp"],
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
    axes[-1].legend()
    figure.tight_layout()
    plot_dir = Path(__file__).resolve().parent
    figure.savefig(plot_dir / "tdp_fdp.pdf")
    figure.savefig(plot_dir / "tdp_fdp.png", dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
