"""Internal plot/export helpers for run_chamber_experiment.py --plot."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from src import chamber_data as data
from src import chamber_experiment as study

COLORS = ("#0072B2", "#D55E00", "#8B5AA4", "#E69F00", "#009E73", "#333333", "#CC79A7")
MARKERS = ("o", "s", "D", "^", "v", "P", "X")
STYLE = dict(zip(study.METHODS, zip(COLORS, MARKERS)))


def configurations(design):
    """Use the frozen manifest, never the universe of available datasets."""
    return tuple(design.get("configurations", {})) or tuple(dict.fromkeys(j["configuration"] for j in design["jobs"]))


def failure_statuses(design):
    return tuple(design.get("failure_statuses", study.FAILURE))


def visible(rows, panel, view):
    return [r for r in rows if r["status"] in study.SUCCESS and
            (view != "sparse" or r["metrics"][panel + "_fp"] <= 25)]


def point_order(row):
    m = row["metrics"]
    return (-m["directed_tp"], m["directed_fp"], -m["skeleton_tp"], m["skeleton_fp"], row["job"]["setting"]["tuning_value"])


def summarize(design, rows):
    output = {"planned": len(design["jobs"]), "accounted": len(rows),
              "all_planned_accounted": len(rows) == len(design["jobs"]),
              "all_planned_successful": len(rows) == len(design["jobs"]) and all(r["status"] in study.SUCCESS for r in rows),
              "design_sha256": data.digest(design), "configurations": {}}
    for cfg in configurations(design):
        methods = {}
        for method in study.METHODS:
            path = [r for r in rows if r["job"]["configuration"] == cfg and r["job"]["method"] == method]
            valid = [r for r in path if r["status"] in study.SUCCESS]
            best = {}
            for budget in (3, 5):
                eligible = [r for r in valid if r["metrics"]["directed_fp"] <= budget]
                row = min(eligible, key=point_order) if eligible else None
                best[str(budget)] = None if row is None else {
                    "job_index": row["job"]["index"], "setting": row["job"]["setting"],
                    "metrics": row["metrics"], "status": row["status"]}
            methods[method] = {"planned": sum(j["configuration"] == cfg and j["method"] == method for j in design["jobs"]),
                "accounted": len(path), "successful": len(valid),
                "status_counts": {status: sum(r["status"] == status for r in path) for status in (*study.SUCCESS, *failure_statuses(design))},
                "outside_25_fp": {panel: sum(r["metrics"][panel + "_fp"] > 25 for r in valid) for panel in ("directed", "skeleton")},
                "minimum_directed_fp": min((r["metrics"]["directed_fp"] for r in valid), default=None),
                "best_at_directed_fp_budget": best,
                "elapsed_seconds": sum(r["elapsed_seconds"] for r in path)}
        output["configurations"][cfg] = methods
    return output


def plot_panels(axes, rows, methods, view):
    for panel, ax in zip(("directed", "skeleton"), axes):
        xkey, ykey = (panel + "_fpr", panel + "_tpr") if view == "rates" else (panel + "_fp", panel + "_tp")
        for index, method in enumerate(methods):
            selected = [r for r in visible(rows, panel, view) if r["job"]["method"] == method]
            if not selected:
                continue
            color, marker = STYLE[method]
            # No displacement or averaging: repeated operating points stay exact.
            ax.scatter([r["metrics"][xkey] for r in selected], [r["metrics"][ykey] for r in selected],
                       s=88 - index*5, marker=marker, facecolors="none", edgecolors=color,
                       linewidths=1.35, alpha=.85, clip_on=False, zorder=3+index)
            nonoptimal = [r for r in selected if r["status"] == "ok_nonoptimal"]
            if nonoptimal:
                ax.scatter([r["metrics"][xkey] for r in nonoptimal], [r["metrics"][ykey] for r in nonoptimal],
                           marker="x", color=color, s=25, linewidths=1, clip_on=False, zorder=12)
        if view == "rates":
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
        else:
            ax.set(xlim=(0, 25 if view == "sparse" else 45 if panel == "directed" else 22), ylim=(0, 23),
                   xlabel="False-positive edges", ylabel="True-positive edges")
            ax.set_yticks([0, 5, 10, 15, 20, 23])
            ax.set_xticks([0, 5, 10, 15, 20, 25] if view == "sparse" else
                          list(range(0, 46, 5)) if panel == "directed" else [0, 5, 10, 15, 20, 22])
        ax.set_title("Directed edges" if panel == "directed" else "Skeleton edges", fontsize=12)
        ax.grid(color="#e8e8e8", lw=.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)


def legend(fig, methods, y=.035):
    handles = [Line2D([], [], linestyle="none", marker=STYLE[m][1], markerfacecolor="none",
                      markeredgecolor=STYLE[m][0], markersize=7, label=study.LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=2,
               frameon=False, fontsize=9, handletextpad=.5, columnspacing=2.2)


def path_figure(cfg, rows, methods, view, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2))
    fig.subplots_adjust(left=.08, right=.965, bottom=.33, top=.81, wspace=.28)
    plot_panels(axes, rows, methods, view)
    configuration = "Actuator chain" if cfg == "scm_4" else "Actuator collider"
    fig.suptitle(f"Causal Chambers | {configuration} ({cfg})", fontsize=16, y=.97)
    count = len([r for r in rows if r["status"] in study.SUCCESS and r["job"]["method"] in methods])
    planned = sum(summary["configurations"][cfg][m]["planned"] for m in methods)
    fig.text(.5, .91, f"40,000 observations | 10 variables | 23 reference edges | {count}/{planned} successful fits",
             ha="center", fontsize=10)
    detail = "Full-range normalized paths" if view == "rates" else "0-25 FP viewing window; other points remain in full results" if view == "sparse" else "Full-range paths; every successful operating point retained"
    fig.text(.5, .865, detail, ha="center", fontsize=9, color="#555555")
    legend(fig, methods, y=.048)
    fig.text(.5, .025, "Unconnected settings; overlaps retained. Cross inside a marker = feasible but nonoptimal PS-MIP.",
             fontsize=8, ha="center", color="#555555")
    return fig


def overview(rows, summary):
    configs = tuple(summary["configurations"])
    single = len(configs) == 1
    fig, axes = plt.subplots(len(configs), 2, figsize=(10.5, 6.2 if single else 9.3), squeeze=False)
    fig.subplots_adjust(left=.08, right=.965, top=.76 if single else .83,
                        bottom=.33 if single else .225, hspace=.65, wspace=.27)
    for cfg, row_axes in zip(configs, axes):
        selected = [r for r in rows if r["job"]["configuration"] == cfg]
        plot_panels(row_axes, selected, study.METHODS, "full")
        fig.text(.08, row_axes[0].get_position().y1 + .065,
                 "Actuator chain (scm_4)" if cfg == "scm_4" else "Actuator collider (scm_5)",
                 fontsize=12, weight="bold")
    fig.suptitle("Causal Chambers: seven-condition pilot", fontsize=17, y=.98)
    good = sum(r["status"] in study.SUCCESS for r in rows)
    fig.text(.5, .92 if single else .94, f"{good}/{summary['planned']} successful fits | 40,000 observations per configuration | no rank transformation",
             ha="center", fontsize=10)
    legend(fig, study.METHODS, y=.04)
    fig.text(.5, .015, "Reference: 21 physical + 2 programmed links. Full paths; no lines, envelope, or AUC.", ha="center", fontsize=9)
    return fig


def diagnostic_figure(cfg, report):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.3))
    fig.subplots_adjust(left=.09, right=.88, bottom=.13, top=.87, wspace=.35, hspace=.48)
    for ax, environment in zip(axes.flat, report["environments"]):
        corr = np.asarray(environment["residual_correlation"])
        display = corr.copy()
        np.fill_diagonal(display, np.nan)
        im = ax.imshow(display, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(10), data.NODES, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(10), data.NODES, fontsize=7)
        ax.set_title(environment["environment"], fontsize=12)
    cb = fig.add_axes([.91, .27, .02, .46])
    fig.colorbar(im, cax=cb, label="Residual correlation")
    fig.suptitle(f"Model check: {cfg}", fontsize=17, y=.97)
    fig.text(.5, .925, "Residuals after regression on documented parents, fitted separately within each environment",
             ha="center", fontsize=9)
    fig.text(.5, .035, "Off-diagonal dependence conflicts with independent SEM errors. No rows or variables were removed.\n"
             "These diagnostics do not identify a unique cause or establish that omitted physical effects are absent.",
             ha="center", fontsize=9, linespacing=1.7)
    return fig


def export_csv(path, rows):
    records = []
    for row in rows:
        job = row["job"]
        record = {"configuration": job["configuration"], "method": job["method"], "job_index": job["index"],
                  "setting_index": job["setting_index"], "setting_id": job["setting"]["setting_id"],
                  "tuning_value": job["setting"]["tuning_value"], "status": row["status"],
                  "elapsed_seconds": row["elapsed_seconds"], **row.get("metrics", {})}
        result = row.get("result", {})
        if row["status"] in study.SUCCESS:
            record["estimated_target_count"] = int(np.asarray(result["estimated_targets"]).sum())
        record.update({key: result.get(key) for key in ("objective_value", "objective_bound", "mip_gap", "solver_status", "optimal")})
        records.append(record)
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_report(path, design, rows, summary, diagnostics):
    good = sum(r["status"] in study.SUCCESS for r in rows)
    ps = [r for r in rows if r["job"]["method"].startswith("ps_mip")]
    optimal = sum(r["status"] == "ok" for r in ps)
    lines = ["# Causal Chambers pilot results", "",
        f"Accounted for {len(rows)}/{summary['planned']} planned fits; {good} successful. "
        f"PS-MIP: {optimal}/{len(ps)} accounted fits certified optimal within the frozen 1e-4 MIP-gap tolerance.", "",
        "## Protocol", "",
        "Each included configuration retains all 40,000 observations and the same ten variables. "
        "One reference-derived affine transformation is shared across environments; there is no rank transformation, row filtering, or subsampling. "
        "IGSP and UT-IGSP use Gaussian invariance tests, unlike the Sachs kernel-test protocol. "
        "These are controlled physical/hybrid experiments, not biological validation. Each reference combines 21 documented physical sensor links with two programmed actuator links.", "",
        "The CSVs and generator contain 10,000 observations per environment; the README's inherited 1,000-observation description is not used.", "",
        "## Sparse comparison (descriptive, not penalty selection)", "",
        "Entries are directed TP/FP, followed by skeleton TP/FP. An unattained budget is reported as such. "
        "Ties use directed FP, skeleton TP, skeleton FP, then the smaller tuning value. Full paths remain in the bundle.", ""]
    for cfg in configurations(design):
        lines += [f"### {cfg}", "", "| Method | FP <= 3 | FP <= 5 | Minimum directed FP | Successful/planned |",
                  "|---|---|---|---:|---:|"]
        for method in study.METHODS:
            item = summary["configurations"][cfg][method]
            cells = []
            for budget in ("3", "5"):
                point = item["best_at_directed_fp_budget"][budget]
                if point is None:
                    cells.append("Unattained")
                else:
                    m = point["metrics"]
                    cells.append(f"{m['directed_tp']}/{m['directed_fp']}; {m['skeleton_tp']}/{m['skeleton_fp']} "
                                 f"(setting {point['setting']['tuning_value']:.5g})")
            lines.append(f"| {study.LABELS[method]} | {' | '.join(cells)} | {item['minimum_directed_fp']} | {item['successful']}/{item['planned']} |")
        lines += ["", "Physical versus programmed recovery at the descriptive FP <= 5 points:", "",
                  "| Method | Physical directed TP / 21 | Programmed directed TP / 2 | Directed TP with ambiguous orientation |",
                  "|---|---:|---:|---:|"]
        for method in study.METHODS:
            point = summary["configurations"][cfg][method]["best_at_directed_fp_budget"]["5"]
            if point is not None:
                m = point["metrics"]
                lines.append(f"| {study.LABELS[method]} | {m['physical_directed_tp']} | {m['programmed_directed_tp']} | {m['directed_tp_ambiguous_in_returned_graph']} |")
        lines += [""]
    lines += ["## Model suitability", "",
        "The intervention protocol is clearer than Sachs: incoming actuator links are explicitly removed at intervention targets. "
        "However, real sensor noise is not the same as the programmed actuator noise. The following pre-fit diagnostics were recorded before recovery evaluation:", "",
        "| Configuration | Largest absolute residual correlation | Non-target residual variance ratio range | Largest held-out quadratic improvement |",
        "|---|---:|---:|---:|"]
    for cfg in configurations(design):
        reports = diagnostics[cfg]["environments"]
        correlations = [np.max(np.abs(np.asarray(e["residual_correlation"]) - np.eye(10))) for e in reports]
        ratios = [v["residual_variance_ratio"] for e in reports[1:] for v in e["non_target_changes_from_reference"].values()]
        gains = [v for e in reports for v in e["quadratic_heldout_mse_fraction_improvement"].values() if v is not None]
        lines.append(f"| {cfg} | {max(correlations):.3f} | {min(ratios):.3f}-{max(ratios):.3f} | {100*max(gains):.3f}% |")
    lines += ["", "Small quadratic improvements support approximate linearity of the measured relationships on this protocol; "
        "they do not establish an exactly linear-Gaussian model. Residual correlation conflicts with independent errors, and changes in non-target residual variance conflict with shared non-target noise. "
        "Correlated sensor noise, unrepresented effects, and changing noise levels are plausible explanations, not uniquely identified causes.", "",
        "Some interventions on actuator roots principally change means. PS-MIP's centered score does not exploit pure mean shifts; "
        "the hidden-target diagnostic can therefore miss documented root interventions even when the graph is recovered well. "
        "This is different from pretending the experimenter lacks target labels: the complete and known-present cases retain them.", "",
        "GnIES permits changes in target noise without requiring all incoming coefficients to be removed, whereas PS-MIP uses hard interventions and shared non-target variances. "
        "Performance differences must be interpreted together with these model and information differences, not as a controlled comparison of optimization alone.", "",
        "## Numerical bounds and completion", ""]
    for cfg in configurations(design):
        pre = design["configurations"][cfg]["preflight"]
        lines.append(f"- {cfg}: {pre['local_optima_checked']:,} local optima checked without graph truth; "
                     f"maximum required diagonal {pre['max_diagonal']:.4g}, maximum coefficient magnitude {pre['max_abs_coefficient']:.4g}. "
                     f"Frozen bounds: {pre['bounds']}.")
    lines += ["", "The original upper bounds of 10 already cover these preflight extrema; the tenfold-margin bounds do not explain any recovery gain. "
        "The optimizer and synthetic defaults were not changed. Successful PS-MIP artifacts include the fitted precision factor, incumbent objective, objective bound, solver status, gap, and inactive-bound checks.", "",
        "## Points outside the 0-25 FP viewing window", "",
        "The display limit is applied independently per panel; full-range figures and raw artifacts preserve all settings.", "",
        "| Configuration | Method | Directed outside | Skeleton outside |",
        "|---|---|---:|---:|"]
    for cfg in configurations(design):
        for method in study.METHODS:
            outside = summary["configurations"][cfg][method]["outside_25_fp"]
            lines.append(f"| {cfg} | {study.LABELS[method]} | {outside['directed']} | {outside['skeleton']} |")
    failures = [r for r in rows if r["status"] not in study.SUCCESS]
    lines += ["", "## Failures and nonoptimal solutions", ""]
    lines += ([f"- Job {r['job']['index']}: {r['job']['configuration']} / {r['job']['method']} / {r['job']['setting']['setting_id']}: {r['status']}" for r in failures] or ["No failed fits among the accounted results."])
    lines += [f"Nonoptimal feasible PS-MIP fits: {sum(r['status'] == 'ok_nonoptimal' for r in rows)}.", "",
        "## Runtime and solver diagnostics", "",
        "All fits start fresh. Times below include accounted failures; missing fits are not counted as zero-runtime successes. "
        "PS-MIP optimality is certification within the numerical tolerance, not a guarantee of correct causal recovery.", "",
        "| Configuration | Method | Successful / planned | Certified PS-MIP | Nonoptimal | Failed / timed out | Worker minutes | Median / maximum PS-MIP gap (%) |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for cfg in configurations(design):
        for method in study.METHODS:
            selected = [r for r in rows if r["job"]["configuration"] == cfg and r["job"]["method"] == method]
            valid = [r for r in selected if r["status"] in study.SUCCESS]
            gaps = [100*r["result"]["mip_gap"] for r in valid if "mip_gap" in r.get("result", {})]
            gap_text = f"{np.median(gaps):.4g} / {max(gaps):.4g}" if gaps else "not applicable / unavailable"
            certified = str(sum(r["status"] == "ok" for r in selected)) if method.startswith("ps_mip") else "not applicable"
            item = summary["configurations"][cfg][method]
            lines.append(f"| {cfg} | {study.LABELS[method]} | {len(valid)}/{item['planned']} | {certified} | "
                         f"{sum(r['status']=='ok_nonoptimal' for r in selected)} | {len(selected)-len(valid)} | "
                         f"{sum(r['elapsed_seconds'] for r in selected)/60:.3f} | {gap_text} |")
    if "timing" in design:
        lines += ["", f"Frozen timing policy: {design['timing']}. "
                  "Raw rows report algorithm/solver overruns, and attempts.jsonl reports external watchdog use and total worker time.", ""]
    lines += ["",
        "## Interpretation and limitations", "",
        "This pilot assesses model suitability and recovery, not whether one method can be made to win. "
        "Directed counts use saved DAG representatives and can depend on unresolved orientations; graph-class ambiguity counts are retained. "
        "Missing edges in the physical reference are not proven biological or physical non-effects. "
        "No AUC, artificial envelope, bootstrap confirmation, target validation, or false-discovery guarantee is claimed. "
        "Raw TP counts are not compared directly with Sachs's 18-edge reference.", "",
        "## Sources", "",
        "- [Dataset and protocols](https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_camera_v1)",
        "- [Official camera reference adjacency](https://github.com/juangamella/causal-chamber-package/tree/main/causalchamber/ground_truth/adjacencies)",
        "- [Causal Chambers paper](https://www.nature.com/articles/s42256-024-00964-x)", ""]
    path.write_text("\n".join(lines))


def export(root, data_root, design, rows, allow_partial=False):
    """Caller holds the run lock and has already validated every supplied row."""
    summary = summarize(design, rows)
    destination = root / ("summary_preview" if allow_partial else "summary")
    destination.mkdir(parents=True, exist_ok=True)
    data.write_json(destination/"validated_summary.json", summary)
    export_csv(destination/"all_operating_points.csv", rows)
    configs = configurations(design)
    diagnostics = {cfg: data.read_json(data_root/cfg/"diagnostics.json") for cfg in configs}
    for cfg in configs:
        data.write_json(destination/f"{cfg}_diagnostics.json", diagnostics[cfg])
    write_report(destination/"RESULTS.md", design, rows, summary, diagnostics)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    with PdfPages(destination/"causal_chambers.pdf") as pdf:
        def save(fig, name):
            if allow_partial:
                fig.text(.5, .995, "INCOMPLETE PREVIEW", ha="center", va="top", fontsize=8, color="red")
            fig.savefig(destination/(name + ".png"), dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
        save(overview(rows, summary), "causal_chambers_overview")
        for cfg in configs:
            selected = [r for r in rows if r["job"]["configuration"] == cfg]
            for group, methods in (("seven_condition", study.METHODS), ("ps_mip_only", study.METHODS[:3])):
                for view in ("full", "sparse", "rates"):
                    save(path_figure(cfg, selected, methods, view, summary), f"{cfg}_{group}_{view}")
            save(diagnostic_figure(cfg, diagnostics[cfg]), f"{cfg}_residual_diagnostics")
    data.write_json(destination/"export_identity.json", {"design_sha256": data.digest(design),
        "plot_code_sha256": data.file_hash(__file__), "result_files": {r["job"]["output"]: data.file_hash(root/r["job"]["output"]) for r in rows},
        "export_files": {p.name: data.file_hash(p) for p in destination.iterdir() if p.is_file() and p.name != "export_identity.json"}})
    print(f"Exported {len(rows)}/{len(design['jobs'])} accounted fits to {destination}")
    return summary
