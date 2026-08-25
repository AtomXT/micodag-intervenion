#!/usr/bin/env python3
"""Aggregate main-experiment fragments, plot TDP--FDP paths, and build tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_main_experiment import (
    DEFAULT_NUM_REPLICATES,
    EDGE_MULTIPLIERS,
    FIELDS as RESULT_FIELDS,
    MAIN_EXPERIMENT_VERSION,
    METHODS,
    P_VALUES,
    method_setting_ids,
)
from src.main_experiment_data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MASTER_SEED,
    MAX_MASTER_SEED,
    load_main_experiment_manifest,
)


DEFAULT_INPUT = PROJECT_ROOT / "experiment_results" / "main_experiment" / "parts"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiment_results" / "main_experiment" / "summary"
GENERATED_OUTPUTS = (
    "combined_results.csv",
    "duplicate_rows.csv",
    "missing_rows.csv",
    "unexpected_rows.csv",
    "missing_shared_metadata.csv",
    "invalid_shared_data.csv",
    "manifest_mismatches.csv",
    "missing_method_config.csv",
    "invalid_method_config.csv",
    "unsuccessful_rows.csv",
    "best_selection_coverage.csv",
    "validation_report.txt",
    "curve_summary.csv",
    "main_tdp_fdp.pdf",
    "main_tdp_fdp.png",
    "best_dcpdag_selected_rows.csv",
    "best_dcpdag_summary.csv",
    "best_dcpdag_summary.tex",
)
KEY = ["p", "e", "replicate", "method", "setting_id"]
SUCCESS_STATUSES = {"ok", "ok_nonoptimal"}
NUMERIC_COLUMNS = [
    "experiment_version", "p", "e", "replicate", "seed",
    "num_environments", "num_interventional_environments",
    "n_observational", "n_interventional", "total_samples",
    "target_edge_count", "edge_probability", "realized_edges",
    "screen_constant", "screen_alpha", "screen_edges", "screen_parent_sets",
    "graph_penalty", "target_penalty", "d_cpdag", "fdp", "tdp",
    "target_hamming", "union_target_error", "fit_seconds", "metric_seconds",
    "solve_seconds", "objective_value", "objective_bound", "mip_gap",
    "solver_status_code", "solution_count", "node_count",
]
METHOD_LABELS = {
    "ps_mip_unknown": "PS-MIP",
    "dcdi_g_unknown": "DCDI-G",
    "utigsp_unknown": "UT-IGSP",
    "gnies_unknown": "GnIES",
    "ps_mip_oracle": "PS-MIP (oracle)",
    "dcdi_g_oracle": "DCDI-G (oracle)",
    "igsp_oracle": "IGSP (oracle)",
    "gies_oracle": "GIES (oracle)",
}
STYLES = {
    "ps_mip_unknown": ("tab:blue", "o", "-"),
    "dcdi_g_unknown": ("tab:purple", "P", "-"),
    "utigsp_unknown": ("tab:red", "D", "-"),
    "gnies_unknown": ("tab:orange", "s", "-"),
    "ps_mip_oracle": ("tab:cyan", "^", "--"),
    "dcdi_g_oracle": ("tab:purple", "X", "--"),
    "igsp_oracle": ("tab:brown", "<", "--"),
    "gies_oracle": ("tab:green", "v", "--"),
}


def _result_paths(inputs):
    paths = set()
    for value in inputs:
        path = value.expanduser().resolve()
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            # DCDI artifacts contain staged CSVs below nested directories; only
            # direct fragments and one method-directory level are valid.
            paths.update(candidate.resolve() for candidate in path.glob("*.csv"))
            paths.update(candidate.resolve() for candidate in path.glob("*/*.csv"))
    return sorted(paths)


def _read_results(paths):
    frames = []
    # Fail at ingestion with a useful error instead of allowing a partial row
    # schema to crash later while grouping or producing the audit tables.
    required = set(RESULT_FIELDS)
    for path in paths:
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["_source"] = str(path)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True, sort=False)
    for column in NUMERIC_COLUMNS:
        if column not in results:
            results[column] = np.nan
        results[column] = pd.to_numeric(results[column], errors="coerce")
    versions = results["experiment_version"].to_numpy(dtype=float)
    if not (
        np.isfinite(versions) & (versions == MAIN_EXPERIMENT_VERSION)
    ).all():
        raise ValueError(
            "every result row must have the current integral experiment version "
            f"{MAIN_EXPERIMENT_VERSION}"
        )
    results = results.copy()
    if results.empty:
        raise ValueError(
            f"no experiment-version {MAIN_EXPERIMENT_VERSION} rows were found"
        )
    return results


def _expected_results(num_replicates):
    return pd.DataFrame(
        [
            {
                "p": p,
                "e": edge_multiplier,
                "replicate": replicate,
                "method": method,
                "setting_id": setting_id,
            }
            for p in P_VALUES
            for edge_multiplier in EDGE_MULTIPLIERS
            for replicate in range(1, num_replicates + 1)
            for method in METHODS
            for setting_id in method_setting_ids(method)
        ]
    )


def _setting_rank(method, setting_id):
    try:
        setting_ids = method_setting_ids(method)
    except ValueError:
        return len(METHODS) * 1000
    try:
        return setting_ids.index(setting_id)
    except ValueError:
        return len(setting_ids)


def _validate_manifest_binding(validated_results, manifest, output_dir):
    """Require every result fragment to belong to the selected data suite."""
    expected_instances = pd.DataFrame(
        [
            {
                "p": p,
                "e": edge_multiplier,
                "replicate": replicate,
                "expected_seed": entry["seed"],
                "expected_data_sha256": entry["data_sha256"],
                "expected_instance_sha256": entry["instance_sha256"],
            }
            for (p, edge_multiplier, replicate), entry in manifest["_index"].items()
        ]
    )
    compared = validated_results.merge(
        expected_instances,
        on=["p", "e", "replicate"],
        how="left",
        validate="many_to_one",
    )
    mismatch_mask = (
        compared["expected_seed"].isna()
        | compared["expected_data_sha256"].isna()
        | compared["expected_instance_sha256"].isna()
        | compared["seed"].ne(compared["expected_seed"])
        | compared["data_sha256"].astype(str).ne(
            compared["expected_data_sha256"].astype(str)
        )
        | compared["instance_sha256"].astype(str).ne(
            compared["expected_instance_sha256"].astype(str)
        )
    )
    columns = KEY + [
        "seed",
        "expected_seed",
        "data_sha256",
        "expected_data_sha256",
        "instance_sha256",
        "expected_instance_sha256",
    ]
    mismatches = compared.loc[mismatch_mask, columns]
    mismatches.to_csv(output_dir / "manifest_mismatches.csv", index=False)
    if not mismatches.empty:
        raise RuntimeError(
            "result rows do not belong to the selected persisted data suite; see "
            f"{output_dir / 'manifest_mismatches.csv'}"
        )


def _validate(results, expected, output_dir, manifest):
    duplicate_mask = results.duplicated(KEY, keep=False)
    duplicates = results.loc[duplicate_mask].sort_values(KEY)
    duplicates.to_csv(output_dir / "duplicate_rows.csv", index=False)
    if not duplicates.empty:
        raise RuntimeError(
            f"found duplicate result keys; see {output_dir / 'duplicate_rows.csv'}"
        )

    observed = results[KEY].drop_duplicates()
    missing = expected.merge(observed, on=KEY, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    unexpected = observed.merge(expected, on=KEY, how="left", indicator=True)
    unexpected = unexpected[unexpected["_merge"] == "left_only"].drop(
        columns="_merge"
    )
    missing.to_csv(output_dir / "missing_rows.csv", index=False)
    unexpected.to_csv(output_dir / "unexpected_rows.csv", index=False)

    validated_results = results.merge(
        expected, on=KEY, how="inner", validate="one_to_one"
    )

    instance_keys = ["p", "e", "replicate"]
    shared_columns = [
        "seed", "data_sha256", "instance_sha256", "true_targets", "num_environments",
        "num_interventional_environments", "n_observational",
        "n_interventional", "total_samples", "target_edge_count",
        "realized_edges", "edge_probability",
    ]
    for column in shared_columns:
        if column not in validated_results:
            validated_results[column] = np.nan
    missing_shared_mask = validated_results[shared_columns].isna().any(axis=1)
    for column in ("data_sha256", "instance_sha256", "true_targets"):
        missing_shared_mask |= validated_results[column].astype(str).str.strip().eq("")
    missing_shared = validated_results.loc[
        missing_shared_mask, KEY + shared_columns
    ]
    missing_shared.to_csv(output_dir / "missing_shared_metadata.csv", index=False)
    if not missing_shared.empty:
        raise RuntimeError(
            "result rows have missing shared-data metadata; see "
            f"{output_dir / 'missing_shared_metadata.csv'}"
        )
    shared = (
        validated_results.groupby(instance_keys, as_index=False)
        .agg(**{f"{column}_count": (column, "nunique") for column in shared_columns})
    )
    count_columns = [f"{column}_count" for column in shared_columns]
    invalid_shared = shared[(shared[count_columns] != 1).any(axis=1)]
    invalid_shared.to_csv(output_dir / "invalid_shared_data.csv", index=False)
    if not invalid_shared.empty:
        raise RuntimeError("methods within an instance do not share one seed/data hash")

    _validate_manifest_binding(validated_results, manifest, output_dir)

    config_keys = ["p", "e", "method", "setting_id"]
    missing_config_mask = validated_results["method_config"].isna()
    missing_config_mask |= validated_results["method_config"].astype(str).str.strip().eq("")
    missing_config = validated_results.loc[
        missing_config_mask, KEY + ["method_config"]
    ]
    missing_config.to_csv(output_dir / "missing_method_config.csv", index=False)
    if not missing_config.empty:
        raise RuntimeError("result rows have missing declared method configuration")
    config_counts = (
        validated_results.groupby(config_keys, as_index=False)
        .agg(method_config_count=("method_config", "nunique"))
    )
    invalid_config = config_counts[config_counts["method_config_count"] != 1]
    invalid_config.to_csv(output_dir / "invalid_method_config.csv", index=False)
    if not invalid_config.empty:
        raise RuntimeError(
            "replicates mix method configurations; see invalid_method_config.csv"
        )

    return missing, unexpected


def _curve_summary(successful):
    successful = successful.copy()
    successful["_optimal_numeric"] = successful["optimal"].map(
        {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
    )
    group = [
        "p", "e", "method", "target_setting", "setting_id",
        "tuning_parameter", "tuning_value", "graph_penalty", "target_penalty",
    ]
    summary = (
        successful.groupby(group, as_index=False, dropna=False)
        .agg(
            replicates=("replicate", "nunique"),
            fdp_mean=("fdp", "mean"),
            fdp_std=("fdp", "std"),
            tdp_mean=("tdp", "mean"),
            tdp_std=("tdp", "std"),
            d_cpdag_mean=("d_cpdag", "mean"),
            d_cpdag_std=("d_cpdag", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
            screen_edges_mean=("screen_edges", "mean"),
            screen_parent_sets_mean=("screen_parent_sets", "mean"),
            mip_gap_mean=("mip_gap", "mean"),
            optimal_rate=("_optimal_numeric", "mean"),
        )
    )
    summary["_setting_order"] = [
        _setting_rank(method, setting)
        for method, setting in zip(summary["method"], summary["setting_id"])
    ]
    return summary.sort_values(
        ["e", "p", "method", "_setting_order"]
    ).drop(columns="_setting_order")


def _ordered_curve(frame, method):
    order = {setting: index for index, setting in enumerate(method_setting_ids(method))}
    ordered = frame.copy()
    ordered["_order"] = ordered["setting_id"].map(order)
    return ordered.sort_values("_order")


def _plot_dcdi(axis, frame, color, marker, label):
    axis.scatter(frame["fdp_mean"], frame["tdp_mean"], color=color, marker=marker,
                 s=42, label=label, zorder=3)
    branches = [
        (["g0p01_t0p001", "g0p1_t0p001", "g1_t0p001"], "-"),
        (["g0p1_t0p0001", "g0p1_t0p001", "g0p1_t0p01"], ":"),
    ]
    for setting_ids, linestyle in branches:
        branch = frame.set_index("setting_id").reindex(setting_ids).dropna(
            subset=["fdp_mean", "tdp_mean"]
        )
        if len(branch) > 1:
            axis.plot(
                branch["fdp_mean"], branch["tdp_mean"], color=color,
                linewidth=1.4, linestyle=linestyle, alpha=0.8,
            )


def _plot(curves, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for row, edge_multiplier in enumerate(EDGE_MULTIPLIERS):
        for column, p in enumerate(P_VALUES):
            axis = axes[row, column]
            panel = curves[(curves["p"] == p) & (curves["e"] == edge_multiplier)]
            for method in METHODS:
                method_curve = panel[panel["method"] == method]
                if method_curve.empty:
                    continue
                method_curve = _ordered_curve(method_curve, method)
                color, marker, linestyle = STYLES[method]
                if method == "dcdi_g_unknown":
                    _plot_dcdi(axis, method_curve, color, marker, METHOD_LABELS[method])
                else:
                    axis.plot(
                        method_curve["fdp_mean"], method_curve["tdp_mean"],
                        color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.5, markersize=5, label=METHOD_LABELS[method],
                        markerfacecolor=("none" if method.endswith("oracle") else color),
                        markeredgecolor=color,
                    )
            axis.set_title(f"p={p}, e={edge_multiplier}")
            axis.grid(alpha=0.25)
            axis.set_xlim(-0.02, 1.02)
            axis.set_ylim(-0.02, 1.02)
            if row == len(EDGE_MULTIPLIERS) - 1:
                axis.set_xlabel("FDP")
            if column == 0:
                axis.set_ylabel("TDP")

    handles, labels = [], []
    for axis in axes.flat:
        panel_handles, panel_labels = axis.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
            ncol=3, frameon=False,
        )
    figure.suptitle("Main experiment TDP–FDP paths", y=0.995)
    figure.text(
        0.5,
        0.012,
        "Metric-complete PS-MIP incumbents are included; see curve_summary.csv "
        "for MIP gaps and optimality rates.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.88))
    figure.savefig(output_dir / "main_tdp_fdp.pdf")
    figure.savefig(output_dir / "main_tdp_fdp.png", dpi=200)
    plt.close(figure)


def _complete_grid_groups(successful, expected):
    group_columns = ["p", "e", "replicate", "method"]
    columns = group_columns + [
        "successful_settings", "expected_settings", "missing_settings",
        "complete_grid",
    ]
    observed_by_group = {
        key: set(group["setting_id"])
        for key, group in successful.groupby(group_columns)
    }
    rows = []
    for key, group in expected.groupby(group_columns):
        p, edge_multiplier, replicate, method = key
        observed = observed_by_group.get(key, set())
        expected_settings = set(group["setting_id"])
        rows.append(
            {
                "p": p,
                "e": edge_multiplier,
                "replicate": replicate,
                "method": method,
                "successful_settings": len(observed),
                "expected_settings": len(expected_settings),
                "missing_settings": json.dumps(
                    sorted(expected_settings - observed)
                ),
                "complete_grid": observed == expected_settings,
            }
        )
    coverage = pd.DataFrame(rows, columns=columns)
    # Preserve boolean row-selection semantics even when there are no
    # successful method/replicate groups.  An empty object-typed Series is
    # otherwise interpreted by pandas as a column selector.
    coverage["complete_grid"] = coverage["complete_grid"].astype(bool)
    return coverage


def _mean_sd(mean, standard_deviation):
    if pd.isna(mean):
        return ""
    if pd.isna(standard_deviation):
        return f"{mean:.3f}"
    return f"{mean:.3f} ({standard_deviation:.3f})"


def _write_latex_summary(summary, path):
    """Write a compact, escaped display table; keep rich audit fields in CSV."""
    display = pd.DataFrame(
        {
            "p": summary["p"].astype(int),
            "e": summary["e"].astype(int),
            "method": summary["method_label"],
            "dCPDAG": [
                _mean_sd(mean, standard_deviation)
                for mean, standard_deviation in zip(
                    summary["d_cpdag_mean"], summary["d_cpdag_std"]
                )
            ],
            "FDP": [
                _mean_sd(mean, standard_deviation)
                for mean, standard_deviation in zip(
                    summary["fdp_mean"], summary["fdp_std"]
                )
            ],
            "TDP": [
                _mean_sd(mean, standard_deviation)
                for mean, standard_deviation in zip(
                    summary["tdp_mean"], summary["tdp_std"]
                )
            ],
            "fit s": [
                "" if pd.isna(value) else f"{value:.2f}"
                for value in summary["fit_seconds_mean"]
            ],
            "grid s": [
                "" if pd.isna(value) else f"{value:.2f}"
                for value in summary["grid_fit_seconds_mean"]
            ],
            "MIP gap / opt.": [
                (
                    ""
                    if pd.isna(gap) and pd.isna(optimal_rate)
                    else (
                        f"{gap:.4f} / {optimal_rate:.3f}"
                        if not pd.isna(gap) and not pd.isna(optimal_rate)
                        else f"{gap:.4f}" if not pd.isna(gap)
                        else f"{optimal_rate:.3f}"
                    )
                )
                for gap, optimal_rate in zip(
                    summary["mip_gap_mean"], summary["optimal_rate"]
                )
            ],
        }
    )
    tabular = display.to_latex(
        None, index=False, escape=True, column_format="@{}rrlrrrrrr@{}"
    )
    path.write_text(
        "% Trial-wise minimum true d_cpdag; post-hoc oracle potential.\n"
        "\\begingroup\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
        + tabular
        + "\\endgroup\n"
    )


def _best_dcpdag(successful, expected, output_dir):
    coverage = _complete_grid_groups(successful, expected)
    coverage.to_csv(output_dir / "best_selection_coverage.csv", index=False)
    complete = coverage.loc[coverage["complete_grid"]]
    group_columns = ["p", "e", "replicate", "method"]
    if complete.empty:
        selected = successful.iloc[0:0].copy()
        selected["selection_rule"] = "trialwise_min_true_d_cpdag"
        selected["tuning_interpretation"] = "posthoc_oracle_potential"
        selected.to_csv(output_dir / "best_dcpdag_selected_rows.csv", index=False)
        return pd.DataFrame()
    eligible = successful.merge(complete[group_columns], on=group_columns, how="inner")
    eligible["_setting_order"] = [
        _setting_rank(method, setting)
        for method, setting in zip(eligible["method"], eligible["setting_id"])
    ]
    selected = (
        eligible.sort_values(group_columns + ["d_cpdag", "_setting_order"])
        .drop_duplicates(group_columns, keep="first")
        .drop(columns="_setting_order")
    )
    selected["selection_rule"] = "trialwise_min_true_d_cpdag"
    selected["tuning_interpretation"] = "posthoc_oracle_potential"
    selected.to_csv(output_dir / "best_dcpdag_selected_rows.csv", index=False)

    if selected.empty:
        return pd.DataFrame()
    selected["optimal_numeric"] = selected["optimal"].map(
        {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
    )
    group = ["p", "e", "method", "target_setting"]
    summary = (
        selected.groupby(group, as_index=False)
        .agg(
            replicates=("replicate", "nunique"),
            d_cpdag_mean=("d_cpdag", "mean"),
            d_cpdag_std=("d_cpdag", "std"),
            fdp_mean=("fdp", "mean"),
            fdp_std=("fdp", "std"),
            tdp_mean=("tdp", "mean"),
            tdp_std=("tdp", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_std=("fit_seconds", "std"),
            mip_gap_mean=("mip_gap", "mean"),
            optimal_rate=("optimal_numeric", "mean"),
        )
    )
    count_rows = []
    setting_counts = (
        selected.groupby(group + ["setting_id"]).size().rename("count").reset_index()
    )
    for key, frame in setting_counts.groupby(group):
        count_rows.append(
            dict(
                zip(group, key),
                selected_setting_counts=json.dumps(
                    dict(zip(frame["setting_id"], frame["count"].astype(int))),
                    sort_keys=True,
                ),
            )
        )
    counts = pd.DataFrame(count_rows)
    summary = summary.merge(counts, on=group, how="left")
    per_replicate_grid_time = (
        eligible.groupby(group + ["replicate"], as_index=False)
        .agg(grid_fit_seconds=("fit_seconds", "sum"))
    )
    grid_time = (
        per_replicate_grid_time.groupby(group, as_index=False)
        .agg(
            grid_fit_seconds_mean=("grid_fit_seconds", "mean"),
            grid_fit_seconds_std=("grid_fit_seconds", "std"),
        )
    )
    summary = summary.merge(grid_time, on=group, how="left")
    summary.insert(4, "method_label", summary["method"].map(METHOD_LABELS))
    summary["_method_order"] = summary["method"].map(
        {method: index for index, method in enumerate(METHODS)}
    )
    summary = summary.sort_values(["p", "e", "_method_order"]).drop(
        columns="_method_order"
    )
    summary["selection_rule"] = "trialwise_min_true_d_cpdag"
    summary["tuning_interpretation"] = "posthoc_oracle_potential"
    summary.to_csv(output_dir / "best_dcpdag_summary.csv", index=False)
    _write_latex_summary(summary, output_dir / "best_dcpdag_summary.tex")
    return summary


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", default=[DEFAULT_INPUT])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="persisted dataset suite that produced the result fragments",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless all expected rows succeeded (use for final tables)",
    )
    args = parser.parse_args()
    if args.num_replicates < 1:
        parser.error("num-replicates must be positive")
    if args.num_replicates > DEFAULT_NUM_REPLICATES:
        parser.error(
            f"num-replicates cannot exceed {DEFAULT_NUM_REPLICATES}"
        )
    if not 0 <= args.seed <= MAX_MASTER_SEED:
        parser.error(f"seed must be between 0 and {MAX_MASTER_SEED}")
    return args


def main():
    args = _parse_args()
    paths = _result_paths(args.input)
    if not paths:
        raise FileNotFoundError("no result CSV files found in --input")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Do not leave prior summaries or diagnostics looking current if this
    # aggregation fails validation.  These files are fully regenerated below.
    input_paths = set(paths)
    for filename in GENERATED_OUTPUTS:
        generated_path = output_dir / filename
        if generated_path not in input_paths:
            generated_path.unlink(missing_ok=True)

    results = _read_results(paths)
    manifest = load_main_experiment_manifest(
        args.data_root, master_seed=args.seed
    )
    expected = _expected_results(args.num_replicates)
    missing, unexpected = _validate(
        results, expected, output_dir, manifest
    )
    results.to_csv(output_dir / "combined_results.csv", index=False)

    # Unexpected fragments are diagnostic input only.  Never let a stale
    # replicate or setting change a curve or a truth-selected table.
    analysis_results = results.merge(
        expected, on=KEY, how="inner", validate="one_to_one"
    )
    finite_metrics = np.isfinite(
        analysis_results[["d_cpdag", "fdp", "tdp"]].to_numpy(dtype=float)
    ).all(axis=1)
    successful = analysis_results[
        analysis_results["status"].isin(SUCCESS_STATUSES) & finite_metrics
    ].copy()
    unsuccessful = analysis_results.loc[
        ~analysis_results.index.isin(successful.index)
    ].copy()
    unsuccessful.to_csv(output_dir / "unsuccessful_rows.csv", index=False)

    coverage = _complete_grid_groups(successful, expected)
    coverage.to_csv(output_dir / "best_selection_coverage.csv", index=False)
    incomplete_groups = coverage.loc[~coverage["complete_grid"]]
    status_counts = analysis_results.groupby(["method", "status"]).size()
    report = "\n".join(
        [
            f"input files: {len(paths)}",
            f"expected rows: {len(expected)}",
            f"observed version-{MAIN_EXPERIMENT_VERSION} rows: {len(results)}",
            f"expected-key rows: {len(analysis_results)}",
            f"successful metric rows: {len(successful)}",
            f"missing rows: {len(missing)}",
            f"unexpected rows: {len(unexpected)}",
            f"unsuccessful rows: {len(unsuccessful)}",
            f"incomplete method/replicate grids: {len(incomplete_groups)}",
            "",
            "status counts:",
            status_counts.to_string() if not status_counts.empty else "(none)",
        ]
    )
    (output_dir / "validation_report.txt").write_text(report + "\n")
    print(report)
    if args.require_complete and (
        len(missing) or len(unexpected) or len(unsuccessful) or len(incomplete_groups)
    ):
        raise RuntimeError("main experiment is incomplete; see validation outputs")

    curves = _curve_summary(successful)
    curves.to_csv(output_dir / "curve_summary.csv", index=False)
    _plot(curves, output_dir)
    _best_dcpdag(successful, expected, output_dir)


if __name__ == "__main__":
    main()
