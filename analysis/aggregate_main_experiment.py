#!/usr/bin/env python3
"""Aggregate main-experiment fragments, plot TDP--FDP paths, and build tables."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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
    _screen_alpha,
    _screen_statistics,
    method_setting_ids,
)
from data.DataGeneration import moralize
from MIP import estimate_moral_graph
from src.main_experiment_data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MASTER_SEED,
    MAX_MASTER_SEED,
    load_main_experiment_instance,
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
    "completion_runtime_summary.csv",
    "completion_runtime_summary.tex",
    "target_recovery_summary.csv",
    "target_tpr_fpr_summary.csv",
    "target_tpr_fpr.pdf",
    "target_tpr_fpr.png",
    "curve_summary.csv",
    "main_tdp_fdp.pdf",
    "main_tdp_fdp.png",
    "ps_mip_screen_diagnostics.csv",
    "ps_mip_screen_summary.csv",
    "ps_mip_screen_summary.tex",
    "best_dcpdag_selected_rows.csv",
    "best_dcpdag_summary.csv",
    "best_dcpdag_summary.tex",
    "best_dcpdag_paper.tex",
)
KEY = ["p", "e", "replicate", "method", "setting_id"]
SUCCESS_STATUSES = {"ok", "ok_nonoptimal"}
DCDI_METHODS = {"dcdi_g_unknown", "dcdi_g_oracle"}
STATISTICAL_METHODS = tuple(
    method for method in METHODS if method not in DCDI_METHODS
)
TARGET_MATRIX_METHODS = ("ps_mip_unknown", "utigsp_unknown")
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


def _curve_summary(successful, planned_replicates=DEFAULT_NUM_REPLICATES):
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
    summary["replicates_planned"] = int(planned_replicates)
    summary["complete_replicates"] = (
        summary["replicates"] == int(planned_replicates)
    )
    summary["contains_nonoptimal_incumbents"] = (
        summary["method"].str.startswith("ps_mip")
        & summary["optimal_rate"].lt(1.0)
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


def _plot(curves, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    has_nonoptimal_incumbents = False
    for row, edge_multiplier in enumerate(EDGE_MULTIPLIERS):
        for column, p in enumerate(P_VALUES):
            axis = axes[row, column]
            panel = curves[(curves["p"] == p) & (curves["e"] == edge_multiplier)]
            for method in STATISTICAL_METHODS:
                method_curve = panel[panel["method"] == method]
                if method_curve.empty:
                    continue
                method_curve = _ordered_curve(method_curve, method)
                color, marker, linestyle = STYLES[method]
                axis.plot(
                    method_curve["fdp_mean"], method_curve["tdp_mean"],
                    color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.5, markersize=5, label=METHOD_LABELS[method],
                    markerfacecolor=("none" if method.endswith("oracle") else color),
                    markeredgecolor=color,
                )
                nonoptimal = method_curve[
                    method_curve["contains_nonoptimal_incumbents"].fillna(False)
                ]
                if not nonoptimal.empty:
                    has_nonoptimal_incumbents = True
                    axis.scatter(
                        nonoptimal["fdp_mean"], nonoptimal["tdp_mean"],
                        color="black", marker="x", s=31, linewidths=1.0,
                        zorder=7,
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
    if has_nonoptimal_incumbents:
        handles.append(
            Line2D(
                [], [], linestyle="none", marker="x", markersize=6,
                color="black",
            )
        )
        labels.append("PS-MIP includes nonoptimal incumbents")
    if handles:
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
            ncol=4, frameon=False,
        )
    figure.suptitle("Main experiment TDP–FDP paths", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(output_dir / "main_tdp_fdp.pdf")
    figure.savefig(output_dir / "main_tdp_fdp.png", dpi=200)
    plt.close(figure)


def _completion_runtime_summary(analysis_results, expected, successful):
    """Summarize computational outcomes without assigning metrics to failures."""
    observed_columns = KEY + ["status", "error", "fit_seconds"]
    accounting = expected.merge(
        analysis_results[observed_columns],
        on=KEY,
        how="left",
        validate="one_to_one",
        indicator="_observation_merge",
    )
    accounting["_observed"] = accounting["_observation_merge"].eq("both")
    accounting = accounting.merge(
        successful[KEY].assign(_metric_complete=True),
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    accounting["_metric_complete"] = accounting["_metric_complete"].eq(True)
    status = accounting["status"].fillna("").astype(str)
    error = accounting["error"].fillna("").astype(str)
    accounting["_preflight_timeout"] = (
        status.eq("fit_error")
        & error.str.contains("runtime check exceeded", case=False, regex=False)
    )
    accounting["_fit_timeout"] = (
        status.eq("fit_error")
        & ~accounting["_preflight_timeout"]
        & error.str.contains(
            r"fit exceeded|subprocess limit", case=False, regex=True
        )
    )
    accounting["_metric_timeout"] = (
        status.eq("metric_error")
        & error.str.contains("exceeded", case=False, regex=False)
    )
    recognized = (
        accounting["_metric_complete"]
        | accounting["_preflight_timeout"]
        | accounting["_fit_timeout"]
        | accounting["_metric_timeout"]
    )
    accounting["_other_failure"] = accounting["_observed"] & ~recognized

    rows = []
    for method in METHODS:
        frame = accounting[accounting["method"] == method]
        completed = frame[frame["_metric_complete"]]
        completed_times = pd.to_numeric(
            completed["fit_seconds"], errors="coerce"
        ).dropna()
        observed_times = pd.to_numeric(
            frame.loc[frame["_observed"], "fit_seconds"], errors="coerce"
        ).dropna()
        planned = len(frame)
        successful_fits = int(frame["_metric_complete"].sum())
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "planned_fits": planned,
                "observed_rows": int(frame["_observed"].sum()),
                "successful_metric_fits": successful_fits,
                "completion_rate": successful_fits / planned if planned else np.nan,
                "ok_rows": int(
                    (frame["_metric_complete"] & frame["status"].eq("ok")).sum()
                ),
                "nonoptimal_incumbent_rows": int(
                    (
                        frame["_metric_complete"]
                        & frame["status"].eq("ok_nonoptimal")
                    ).sum()
                ),
                "fit_timeout_rows": int(frame["_fit_timeout"].sum()),
                "preflight_timeout_rows": int(frame["_preflight_timeout"].sum()),
                "metric_timeout_rows": int(frame["_metric_timeout"].sum()),
                "other_failure_rows": int(frame["_other_failure"].sum()),
                "missing_rows": int((~frame["_observed"]).sum()),
                "completed_fit_seconds_mean": (
                    completed_times.mean() if len(completed_times) else np.nan
                ),
                "completed_fit_seconds_median": (
                    completed_times.median() if len(completed_times) else np.nan
                ),
                "completed_fit_seconds_q25": (
                    completed_times.quantile(0.25) if len(completed_times) else np.nan
                ),
                "completed_fit_seconds_q75": (
                    completed_times.quantile(0.75) if len(completed_times) else np.nan
                ),
                "observed_fit_seconds_median": (
                    observed_times.median() if len(observed_times) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_completion_latex(summary, path):
    display = pd.DataFrame(
        {
            "Method": summary["method_label"],
            "Completed": [
                f"{int(done)}/{int(planned)}"
                for done, planned in zip(
                    summary["successful_metric_fits"], summary["planned_fits"]
                )
            ],
            "Nonoptimal": summary["nonoptimal_incumbent_rows"].astype(int),
            "Fit timeout": summary["fit_timeout_rows"].astype(int),
            "Preflight timeout": summary["preflight_timeout_rows"].astype(int),
            "Missing": summary["missing_rows"].astype(int),
            "Median completed fit (s)": [
                "" if pd.isna(value) else f"{value:.2f}"
                for value in summary["completed_fit_seconds_median"]
            ],
        }
    )
    tabular = display.to_latex(
        None, index=False, escape=True, column_format="@{}lrrrrrr@{}"
    )
    tabular = (
        tabular.replace("\\toprule", "\\hline")
        .replace("\\midrule", "\\hline")
        .replace("\\bottomrule", "\\hline")
    )
    path.write_text(
        "% Computational completion under the recorded resource budget.\n"
        "\\begingroup\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        + tabular
        + "\\par\\smallskip\\footnotesize Completed means a row with finite "
        "accuracy metrics. Failed and missing fits are not assigned accuracy "
        "values. Median fit time uses completed rows only; fit and preflight "
        "timeout columns are disjoint. DCDI is excluded from statistical "
        "accuracy summaries and retained here as a computational outcome.\n"
        "\\endgroup\n"
    )


def _target_recovery_summary(
    successful,
    planned_replicates=DEFAULT_NUM_REPLICATES,
):
    """Summarize available unknown-target recovery without oracle rows."""
    unknown_methods = {
        method for method in METHODS if not method.endswith("_oracle")
    }
    frame = successful[successful["method"].isin(unknown_methods)].copy()
    if frame.empty:
        return pd.DataFrame()
    finite_targets = np.isfinite(
        frame[["target_hamming", "union_target_error"]].to_numpy(dtype=float)
    ).any(axis=1)
    frame = frame.loc[finite_targets].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["target_hamming_decisions"] = (
        frame["num_interventional_environments"] * frame["p"]
    )
    frame["target_hamming_rate"] = (
        frame["target_hamming"] / frame["target_hamming_decisions"]
    )
    frame["union_target_decisions"] = frame["p"]
    frame["union_target_error_rate"] = (
        frame["union_target_error"] / frame["union_target_decisions"]
    )
    group = [
        "p", "e", "method", "target_setting", "setting_id",
        "tuning_parameter", "tuning_value", "graph_penalty", "target_penalty",
    ]
    summary = (
        frame.groupby(group, as_index=False, dropna=False)
        .agg(
            replicates=("replicate", "nunique"),
            target_hamming_replicates=("target_hamming", "count"),
            target_hamming_decisions=("target_hamming_decisions", "first"),
            target_hamming_mean=("target_hamming", "mean"),
            target_hamming_std=("target_hamming", "std"),
            target_hamming_rate_mean=("target_hamming_rate", "mean"),
            target_hamming_rate_std=("target_hamming_rate", "std"),
            union_target_decisions=("union_target_decisions", "first"),
            union_target_error_replicates=("union_target_error", "count"),
            union_target_error_mean=("union_target_error", "mean"),
            union_target_error_std=("union_target_error", "std"),
            union_target_error_rate_mean=("union_target_error_rate", "mean"),
            union_target_error_rate_std=("union_target_error_rate", "std"),
        )
    )
    summary["replicates_planned"] = int(planned_replicates)
    summary["complete_replicates"] = (
        summary["replicates"] == int(planned_replicates)
    )
    summary["_setting_order"] = [
        _setting_rank(method, setting)
        for method, setting in zip(summary["method"], summary["setting_id"])
    ]
    return summary.sort_values(
        ["e", "p", "method", "_setting_order"]
    ).drop(columns="_setting_order")


def _binary_target_matrix(value, expected_shape, field_name):
    """Decode and validate one persisted environment-by-node target matrix."""
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        matrix = np.asarray(json.loads(text))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {field_name}: {error}") from error
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{field_name} has shape {matrix.shape}; expected {expected_shape}"
        )
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError(f"{field_name} must be binary")
    return matrix.astype(int)


def _target_classification_rows(results):
    """Compute replicate-level target confusion counts, TPR, and FPR."""
    frame = results[
        results["method"].isin(TARGET_MATRIX_METHODS)
        & results["status"].isin(SUCCESS_STATUSES)
    ].copy()
    columns = [
        "p", "e", "replicate", "method", "target_setting", "setting_id",
        "tuning_parameter", "tuning_value", "graph_penalty", "target_penalty",
        "optimal", "target_true_positives", "target_false_positives",
        "target_true_negatives", "target_false_negatives", "target_tpr",
        "target_fpr",
    ]
    rows = []
    for _, row in frame.iterrows():
        expected_shape = (
            int(row["num_interventional_environments"]),
            int(row["p"]),
        )
        estimated = _binary_target_matrix(
            row.get("estimated_targets"), expected_shape, "estimated_targets"
        )
        truth = _binary_target_matrix(
            row.get("true_targets"), expected_shape, "true_targets"
        )
        if estimated is None or truth is None:
            continue

        true_positives = int(((estimated == 1) & (truth == 1)).sum())
        false_positives = int(((estimated == 1) & (truth == 0)).sum())
        true_negatives = int(((estimated == 0) & (truth == 0)).sum())
        false_negatives = int(((estimated == 0) & (truth == 1)).sum())
        positive_total = true_positives + false_negatives
        negative_total = false_positives + true_negatives
        if not positive_total or not negative_total:
            raise ValueError(
                "target TPR/FPR requires at least one positive and one negative "
                f"decision for p={int(row['p'])}, e={int(row['e'])}, "
                f"replicate={int(row['replicate'])}"
            )
        recorded_hamming = pd.to_numeric(
            pd.Series([row.get("target_hamming")]), errors="coerce"
        ).iloc[0]
        if np.isfinite(recorded_hamming) and int(recorded_hamming) != (
            false_positives + false_negatives
        ):
            raise ValueError(
                "persisted target_hamming disagrees with the saved target matrices"
            )

        rows.append(
            {
                "p": int(row["p"]),
                "e": int(row["e"]),
                "replicate": int(row["replicate"]),
                "method": row["method"],
                "target_setting": row["target_setting"],
                "setting_id": row["setting_id"],
                "tuning_parameter": row["tuning_parameter"],
                "tuning_value": row["tuning_value"],
                "graph_penalty": row["graph_penalty"],
                "target_penalty": row["target_penalty"],
                "optimal": row["optimal"],
                "target_true_positives": true_positives,
                "target_false_positives": false_positives,
                "target_true_negatives": true_negatives,
                "target_false_negatives": false_negatives,
                "target_tpr": true_positives / positive_total,
                "target_fpr": false_positives / negative_total,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _common_target_rows(target_rows):
    """Keep replicates containing every target setting for both methods."""
    expected_pairs = {
        (method, setting_id)
        for method in TARGET_MATRIX_METHODS
        for setting_id in method_setting_ids(method)
    }
    kept = []
    common_replicates = {}
    for (p, edge_multiplier), cell in target_rows.groupby(["p", "e"]):
        available = {
            int(replicate): set(zip(group["method"], group["setting_id"]))
            for replicate, group in cell.groupby("replicate")
        }
        replicate_ids = tuple(
            sorted(
                replicate
                for replicate, pairs in available.items()
                if pairs == expected_pairs
            )
        )
        if not replicate_ids:
            continue
        kept.append(cell[cell["replicate"].isin(replicate_ids)])
        common_replicates[(int(p), int(edge_multiplier))] = replicate_ids
    if not kept:
        return target_rows.copy(), common_replicates
    return pd.concat(kept, ignore_index=True), common_replicates


def _target_tpr_fpr_summary(
    target_rows,
    planned_replicates=DEFAULT_NUM_REPLICATES,
):
    """Summarize target TPR/FPR paths on common complete replicates."""
    common, replicate_ids = _common_target_rows(target_rows)
    if common.empty:
        return pd.DataFrame()
    common = common.copy()
    common["_optimal_numeric"] = common["optimal"].map(
        {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
    )
    group = [
        "p", "e", "method", "target_setting", "setting_id",
        "tuning_parameter", "tuning_value", "graph_penalty", "target_penalty",
    ]
    summary = (
        common.groupby(group, as_index=False, dropna=False)
        .agg(
            replicates=("replicate", "nunique"),
            target_tpr_mean=("target_tpr", "mean"),
            target_tpr_std=("target_tpr", "std"),
            target_fpr_mean=("target_fpr", "mean"),
            target_fpr_std=("target_fpr", "std"),
            target_true_positives_mean=("target_true_positives", "mean"),
            target_false_positives_mean=("target_false_positives", "mean"),
            target_true_negatives_mean=("target_true_negatives", "mean"),
            target_false_negatives_mean=("target_false_negatives", "mean"),
            optimal_rate=("_optimal_numeric", "mean"),
        )
    )

    summary["replicates_planned"] = int(planned_replicates)
    summary["complete_replicates"] = (
        summary["replicates"] == int(planned_replicates)
    )
    summary["common_replicate_ids"] = [
        json.dumps(replicate_ids[(int(p), int(edge_multiplier))])
        for p, edge_multiplier in zip(summary["p"], summary["e"])
    ]
    summary["contains_nonoptimal_incumbents"] = (
        summary["method"].eq("ps_mip_unknown")
        & summary["optimal_rate"].lt(1.0)
    )
    summary["comparison_population"] = (
        "common_replicates_across_methods_and_all_target_settings"
    )
    summary["_setting_order"] = [
        _setting_rank(method, setting)
        for method, setting in zip(summary["method"], summary["setting_id"])
    ]
    return summary.sort_values(
        ["e", "p", "method", "_setting_order"]
    ).drop(columns="_setting_order")


def _plot_target_tpr_fpr(curves, output_dir):
    """Plot environment-specific target-recovery tuning paths."""
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    has_nonoptimal_incumbents = False
    max_fpr = float(curves["target_fpr_mean"].max())
    min_tpr = float(curves["target_tpr_mean"].min())
    x_upper = min(1.0, max(0.1, np.ceil((max_fpr + 0.02) * 20) / 20))
    y_lower = max(0.0, np.floor((min_tpr - 0.03) * 20) / 20)

    for row, edge_multiplier in enumerate(EDGE_MULTIPLIERS):
        for column, p in enumerate(P_VALUES):
            axis = axes[row, column]
            panel = curves[
                (curves["p"] == p) & (curves["e"] == edge_multiplier)
            ]
            for method in TARGET_MATRIX_METHODS:
                method_curve = panel[panel["method"] == method]
                if method_curve.empty:
                    continue
                method_curve = _ordered_curve(method_curve, method)
                color, marker, linestyle = STYLES[method]
                axis.plot(
                    method_curve["target_fpr_mean"],
                    method_curve["target_tpr_mean"],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.6,
                    markersize=5.5,
                    label=METHOD_LABELS[method],
                )
                nonoptimal = method_curve[
                    method_curve["contains_nonoptimal_incumbents"].fillna(False)
                ]
                if not nonoptimal.empty:
                    has_nonoptimal_incumbents = True
                    axis.scatter(
                        nonoptimal["target_fpr_mean"],
                        nonoptimal["target_tpr_mean"],
                        color="black",
                        marker="x",
                        s=31,
                        linewidths=1.0,
                        zorder=7,
                    )
            if not panel.empty:
                axis.text(
                    0.97,
                    0.05,
                    f"n={int(panel['replicates'].iloc[0])}",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    color="0.35",
                )
            axis.set_title(f"p={p}, e={edge_multiplier}")
            axis.set_xlim(0, x_upper)
            axis.set_ylim(y_lower, 1.005)
            axis.grid(alpha=0.25)
            if row == 1:
                axis.set_xlabel("Target FPR")
            if column == 0:
                axis.set_ylabel("Target TPR")

    handles, labels = [], []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if has_nonoptimal_incumbents:
        handles.append(
            Line2D(
                [], [], linestyle="none", marker="x", markersize=6,
                color="black",
            )
        )
        labels.append("PS-MIP includes nonoptimal incumbents")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle("Intervention-target TPR–FPR paths", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_dir / "target_tpr_fpr.pdf")
    figure.savefig(output_dir / "target_tpr_fpr.png", dpi=200)
    plt.close(figure)


def _screen_diagnostics(
    data_root,
    manifest,
    master_seed,
    num_replicates=DEFAULT_NUM_REPLICATES,
):
    """Recompute the exact PS-MIP screen and its truth-retention diagnostics."""
    rows = []
    for p in P_VALUES:
        for edge_multiplier in EDGE_MULTIPLIERS:
            for replicate in range(1, num_replicates + 1):
                data, true_dag, _, _ = load_main_experiment_instance(
                    data_root,
                    p,
                    edge_multiplier,
                    replicate,
                    manifest=manifest,
                    master_seed=master_seed,
                )
                alpha = _screen_alpha(data)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    screen = np.asarray(
                        estimate_moral_graph(data[0], alpha), dtype=bool
                    )
                true_dag = np.asarray(true_dag, dtype=bool)
                true_moral = np.asarray(moralize(true_dag), dtype=bool)
                directed_edges = int(true_dag.sum())
                directed_retained = int((true_dag & screen).sum())
                moral_edges = int(np.triu(true_moral, k=1).sum())
                moral_retained = int(
                    np.triu(true_moral & screen, k=1).sum()
                )
                screen_edges, screen_parent_sets = _screen_statistics(screen)
                warning_messages = sorted({str(item.message) for item in caught})
                rows.append(
                    {
                        "p": p,
                        "e": edge_multiplier,
                        "replicate": replicate,
                        "n_screen": len(data[0]),
                        "screen_alpha": alpha,
                        "true_directed_edges": directed_edges,
                        "true_directed_edges_retained": directed_retained,
                        "true_directed_edge_recall": (
                            directed_retained / directed_edges
                            if directed_edges else 1.0
                        ),
                        "true_moral_edges": moral_edges,
                        "true_moral_edges_retained": moral_retained,
                        "true_moral_edge_recall": (
                            moral_retained / moral_edges if moral_edges else 1.0
                        ),
                        "screen_edges": screen_edges,
                        "screen_parent_sets": screen_parent_sets,
                        "screen_warning_count": len(caught),
                        "screen_warning_messages": " | ".join(warning_messages),
                    }
                )
    diagnostics = pd.DataFrame(rows)
    group = diagnostics.groupby(["p", "e"], as_index=False)
    summary = group.agg(
        replicates=("replicate", "nunique"),
        true_directed_edges=("true_directed_edges", "sum"),
        true_directed_edges_retained=("true_directed_edges_retained", "sum"),
        true_directed_edge_recall_mean=("true_directed_edge_recall", "mean"),
        true_directed_edge_recall_std=("true_directed_edge_recall", "std"),
        true_moral_edges=("true_moral_edges", "sum"),
        true_moral_edges_retained=("true_moral_edges_retained", "sum"),
        true_moral_edge_recall_mean=("true_moral_edge_recall", "mean"),
        true_moral_edge_recall_std=("true_moral_edge_recall", "std"),
        screen_edges_mean=("screen_edges", "mean"),
        screen_edges_std=("screen_edges", "std"),
        screen_parent_sets_mean=("screen_parent_sets", "mean"),
        screen_parent_sets_std=("screen_parent_sets", "std"),
        warning_instances=("screen_warning_count", lambda values: int((values > 0).sum())),
    )
    summary["true_directed_edge_recall_pooled"] = (
        summary["true_directed_edges_retained"]
        / summary["true_directed_edges"]
    )
    summary["true_moral_edge_recall_pooled"] = (
        summary["true_moral_edges_retained"] / summary["true_moral_edges"]
    )
    return diagnostics, summary


def _write_screen_latex(summary, path):
    lines = [
        "% Generated PS-MIP screening diagnostics; do not hand-edit.",
        "    \\begingroup",
        "    \\small",
        "    \\setlength{\\tabcolsep}{5pt}",
        "    \\begin{tabular}{rrrrr}",
        "        \\hline",
        "        $p$ & $e$ & $n$ & True-edge recall & "
        "Mean screen edges / parent sets \\\\",
        "        \\hline",
    ]
    for _, row in summary.sort_values(["p", "e"]).iterrows():
        lines.append(
            "        "
            f"{int(row['p'])} & {int(row['e'])} & {int(row['replicates'])} & "
            f"{row['true_directed_edge_recall_pooled']:.3f} & "
            f"{row['screen_edges_mean']:.1f} / "
            f"{row['screen_parent_sets_mean']:.1f} \\\\"
        )
    lines.extend(
        [
            "        \\hline",
            "    \\end{tabular}",
            "    \\par\\smallskip\\footnotesize The screen is recomputed from "
            "the persisted observational sample using "
            "$\\alpha=\\sqrt{\\log(p)/1000}$ and the experiment's exact "
            "GraphicalLasso support rule.",
            "    \\endgroup",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


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


def _write_latex_summary(
    summary,
    path,
    planned_replicates=DEFAULT_NUM_REPLICATES,
):
    """Write a compact, escaped display table; keep rich audit fields in CSV."""
    display = pd.DataFrame(
        {
            "p": summary["p"].astype(int),
            "e": summary["e"].astype(int),
            "method": summary["method_label"],
            "n": [
                f"{int(value)}/{int(planned_replicates)}"
                for value in summary["replicates"]
            ],
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
        None, index=False, escape=True, column_format="@{}rrlrrrrrrr@{}"
    )
    tabular = (
        tabular.replace("\\toprule", "\\hline")
        .replace("\\midrule", "\\hline")
        .replace("\\bottomrule", "\\hline")
    )
    path.write_text(
        "% Trial-wise minimum true d_cpdag; post-hoc oracle potential only.\n"
        "\\begingroup\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
        + tabular
        + "\\par\\smallskip\\footnotesize Values are mean (SD) over the "
        f"displayed number of complete replicate grids out of {planned_replicates}. "
        "The tuning setting is chosen separately in each replicate using true "
        "$d_{\\mathrm{CPDAG}}$; this is post-hoc oracle potential, not a "
        "deployable tuning rule. Methods without completed accuracy estimates "
        "are omitted and reported in the separate completion/runtime summary.\n"
        + "\\endgroup\n"
    )


def _write_paper_potential_table(
    summary,
    path,
    planned_replicates=DEFAULT_NUM_REPLICATES,
):
    """Write a compact two-panel table intended for direct manuscript input."""
    panels = (
        (
            "Unknown intervention targets",
            ("ps_mip_unknown", "utigsp_unknown", "gnies_unknown"),
        ),
        (
            "Known-target oracle references",
            ("ps_mip_oracle", "igsp_oracle", "gies_oracle"),
        ),
    )
    lines = [
        "% Generated from best_dcpdag_summary.csv; do not hand-edit.",
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Trial-wise post-hoc oracle potential. Within each "
        "eligible replicate, the setting with the smallest true "
        "$d_{\\mathrm{CPDAG}}$ is selected. Values are replicate means; "
        "$n$ gives the number of complete predeclared tuning grids out of "
        f"{planned_replicates}. This truth-based selection is descriptive, "
        "not a deployable tuning rule.}",
        "    \\label{tab:synthetic-best-dcpdag}",
        "    \\begingroup",
        "    \\scriptsize",
        "    \\setlength{\\tabcolsep}{2.3pt}",
        "    \\renewcommand{\\arraystretch}{1.03}",
    ]
    for panel_index, (title, methods) in enumerate(panels):
        if panel_index:
            lines.append("    \\hfill")
        lines.extend(
            [
                "    \\begin{minipage}[t]{0.49\\textwidth}",
                "    \\centering",
                f"    \\textbf{{{title}}}\\\\[2pt]",
                "    \\begin{tabular}{rrlrrrr}",
                "        \\hline",
                "        $p$ & $e$ & Method & $n$ & "
                "$d_{\\mathrm{CPDAG}}$ & FDP & TDP \\\\",
                "        \\hline",
            ]
        )
        for p in P_VALUES:
            for edge_multiplier in EDGE_MULTIPLIERS:
                for method in methods:
                    match = summary[
                        (summary["p"] == p)
                        & (summary["e"] == edge_multiplier)
                        & (summary["method"] == method)
                    ]
                    label = METHOD_LABELS[method].replace(" (oracle)", "")
                    if match.empty:
                        n_value = f"0/{planned_replicates}"
                        d_cpdag = fdp = tdp = "--"
                    else:
                        row = match.iloc[0]
                        n_value = (
                            f"{int(row['replicates'])}/{planned_replicates}"
                        )
                        d_cpdag = f"{row['d_cpdag_mean']:.3f}"
                        fdp = f"{row['fdp_mean']:.3f}"
                        tdp = f"{row['tdp_mean']:.3f}"
                    lines.append(
                        "        "
                        f"{p} & {edge_multiplier} & {label} & {n_value} & "
                        f"{d_cpdag} & {fdp} & {tdp} \\\\"
                    )
        lines.extend(
            [
                "        \\hline",
                "    \\end{tabular}",
                "    \\end{minipage}",
            ]
        )
    lines.extend(
        [
            "    \\par\\smallskip\\footnotesize Failed or missing fits are "
            "not imputed. Methods without completed accuracy estimates are "
            "reported only in the generated completion/runtime table. Standard "
            "deviations, selected-setting counts, MIP gaps, and optimality "
            "rates are in the companion CSV.",
            "    \\endgroup",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _best_dcpdag(
    successful,
    expected,
    output_dir,
    planned_replicates=DEFAULT_NUM_REPLICATES,
):
    successful = successful[~successful["method"].isin(DCDI_METHODS)].copy()
    expected = expected[~expected["method"].isin(DCDI_METHODS)].copy()
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
    summary.insert(6, "replicates_planned", int(planned_replicates))
    summary.insert(
        7,
        "complete_replicates",
        summary["replicates"] == int(planned_replicates),
    )
    summary["_method_order"] = summary["method"].map(
        {method: index for index, method in enumerate(METHODS)}
    )
    summary = summary.sort_values(["p", "e", "_method_order"]).drop(
        columns="_method_order"
    )
    summary["selection_rule"] = "trialwise_min_true_d_cpdag"
    summary["tuning_interpretation"] = "posthoc_oracle_potential"
    summary.to_csv(output_dir / "best_dcpdag_summary.csv", index=False)
    _write_latex_summary(
        summary,
        output_dir / "best_dcpdag_summary.tex",
        planned_replicates,
    )
    _write_paper_potential_table(
        summary,
        output_dir / "best_dcpdag_paper.tex",
        planned_replicates,
    )
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

    completion = _completion_runtime_summary(
        analysis_results, expected, successful
    )
    completion.to_csv(
        output_dir / "completion_runtime_summary.csv", index=False
    )
    _write_completion_latex(
        completion, output_dir / "completion_runtime_summary.tex"
    )
    target_recovery = _target_recovery_summary(
        successful, args.num_replicates
    )
    target_recovery.to_csv(
        output_dir / "target_recovery_summary.csv", index=False
    )
    target_rows = _target_classification_rows(analysis_results)
    target_curves = _target_tpr_fpr_summary(
        target_rows,
        args.num_replicates,
    )
    target_curves.to_csv(
        output_dir / "target_tpr_fpr_summary.csv", index=False
    )
    if not target_curves.empty:
        _plot_target_tpr_fpr(target_curves, output_dir)
    screen_diagnostics, screen_summary = _screen_diagnostics(
        args.data_root,
        manifest,
        args.seed,
        args.num_replicates,
    )
    screen_diagnostics.to_csv(
        output_dir / "ps_mip_screen_diagnostics.csv", index=False
    )
    screen_summary.to_csv(
        output_dir / "ps_mip_screen_summary.csv", index=False
    )
    _write_screen_latex(
        screen_summary, output_dir / "ps_mip_screen_summary.tex"
    )

    if args.require_complete and (
        len(missing) or len(unexpected) or len(unsuccessful) or len(incomplete_groups)
    ):
        raise RuntimeError("main experiment is incomplete; see validation outputs")

    statistical_successful = successful[
        successful["method"].isin(STATISTICAL_METHODS)
    ].copy()
    curves = _curve_summary(
        statistical_successful,
        args.num_replicates,
    )
    curves.to_csv(output_dir / "curve_summary.csv", index=False)
    _plot(curves, output_dir)
    _best_dcpdag(
        statistical_successful,
        expected,
        output_dir,
        args.num_replicates,
    )


if __name__ == "__main__":
    main()
