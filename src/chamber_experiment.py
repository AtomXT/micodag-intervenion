"""Shared method adapters, frozen design, and validation for the chamber experiment."""
from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import platform
import time
import warnings
from pathlib import Path

import numpy as np

from src import chamber_data as data

LEGACY_METHODS = ("ps_mip_complete", "ps_mip_present", "ps_mip_unknown", "utigsp_present",
                  "gnies_present", "gies_complete", "igsp_complete")
METHODS = ("ps_mip_unknown", "gies_oracle", "utigsp_unknown", "igsp_oracle", "gnies_unknown")
ALL_METHODS = tuple(dict.fromkeys((*LEGACY_METHODS, *METHODS)))
LABELS = {
    "ps_mip_complete": "PS-MIP: complete targets",
    "ps_mip_present": "PS-MIP: known-present targets",
    "ps_mip_unknown": "PS-MIP: unknown targets",
    "utigsp_present": "UT-IGSP: targets supplied",
    "gnies_present": "GnIES: target union supplied",
    "gies_complete": "GIES: complete targets",
    "igsp_complete": "IGSP: complete targets",
    "gies_oracle": "GIES (oracle)",
    "utigsp_unknown": "UT-IGSP: unknown targets",
    "igsp_oracle": "IGSP (oracle)",
    "gnies_unknown": "GnIES: unknown targets",
}
TARGET_INFORMATION = {
    "ps_mip_complete": "documented targets fixed present; other indicators fixed absent",
    "ps_mip_present": "documented targets fixed present; other indicators learnable",
    "ps_mip_unknown": "intervention labels hidden as diagnostic; all indicators learnable; reference remains observational",
    "utigsp_present": "documented environment-specific targets supplied; additional targets learnable",
    "gnies_present": "documented target union supplied; additional union members learnable",
    "gies_complete": "documented environment-specific target lists treated as complete",
    "igsp_complete": "documented environment-specific target lists treated as complete",
    "gies_oracle": "oracle documented environment-specific target lists treated as complete; no graph supplied",
    "utigsp_unknown": "no intervention targets supplied; environment-specific targets learnable; reference remains observational",
    "igsp_oracle": "oracle documented environment-specific target lists treated as complete; no graph supplied",
    "gnies_unknown": "no intervention targets or initial union supplied; target union learnable",
}
GRAPH_MULTIPLIERS = tuple(2. ** (k / 2) for k in range(-8, 9))
NATIVE_MULTIPLIERS = tuple(2. ** k for k in range(-4, 11))
CI_ALPHAS = (.001, .01, .05, .1, .2, .3, .4, .5)
SEED = 20260901
NATIVE_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SUCCESS = ("ok", "ok_nonoptimal")
FAILURE = ("failed", "timeout", "watchdog_timeout", "process_failed")
LEGACY_PROFILE = "scm4_unscreened_1h_v1"
PROFILE = "scm4_unknown_oracle_1h_v1"
LEGACY_JOBS = Path("archive/causal_chambers/2026-09-06/known_target_quest_jobs")
REUSABLE_METHODS = {"ps_mip_unknown": "ps_mip_unknown", "gies_oracle": "gies_complete", "igsp_oracle": "igsp_complete"}
CORE_FILES = ("src/chamber_data.py", "src/chamber_experiment.py", "experiments/run_chamber_experiment.py",
              "MIP_profiled.py", "MIP.py", "UTIGSP.py", "IGSP.py", "GIES.py",
              "src/utils.py", "src/main_experiment_cli.py", "src/main_experiment_data.py",
              "data/DataGeneration.py", "requirements-chambers.txt",
              *(f"experiments/quest_jobs/chamber_experiment_{m}.sh" for m in METHODS))

# Reviewed predecessors for the historical seven-condition profile only. Its
# adapter behavior, settings and metrics are preserved by the new method branches.
# These revisions cannot be used to certify new unknown-target baseline fits.
POSTPROCESS_COMPATIBLE_SOURCES = {
    "src/chamber_experiment.py": {"84f4f4633ee0160c8f8de1d05ff14f22e5cc25931a33cb6a5954dd7df6d5071b"},
    "experiments/run_chamber_experiment.py": {"b711742f68e69657df6d29cf406b15ec1f953391f905550347bd0a4d646bc1ff"},
}


def runtime_identity():
    distributions = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])
    packages = {}
    for name in ("gnies", "gies", "causaldag", "conditional_independence", "graphical_models"):
        module = importlib.import_module(name)
        base = Path(module.__file__).parent
        packages[name] = data.digest({str(p.relative_to(base)): data.file_hash(p) for p in sorted(base.rglob("*.py"))})
    return {"python": platform.python_version(), "distribution_identity": data.digest(distributions),
            "versions": dict(distributions), "package_source_hashes": packages}


def methods_for_profile(profile):
    if profile == PROFILE: return METHODS
    if profile == LEGACY_PROFILE: return LEGACY_METHODS
    raise ValueError("unknown chamber profile")


def implementation_identity(profile=PROFILE):
    files = [name for name in CORE_FILES if not name.startswith("experiments/quest_jobs/")]
    output = {name: data.file_hash(data.ROOT / name) for name in files}
    for method in methods_for_profile(profile):
        name = f"experiments/quest_jobs/chamber_experiment_{method}.sh"
        path = LEGACY_JOBS / Path(name).name if profile == LEGACY_PROFILE else Path(name)
        output[name] = data.file_hash(data.ROOT / path)
    return output


def check_apis():
    import gnies
    from GIES import validate_runtime
    from IGSP import fit as igsp
    from UTIGSP import fit as utigsp
    from MIP_profiled import optimization
    for name in ("center", "known_targets"):
        if name not in inspect.signature(gnies.fit).parameters:
            raise ValueError(f"GnIES does not support required {name}")
    for fit in (igsp, utigsp):
        if "invariance_test" not in inspect.signature(fit).parameters:
            raise ValueError("IGSP adapter lacks explicit invariance test")
    for name in ("target_status", "gamma_lower", "gamma_upper", "coefficient_bound"):
        if name not in inspect.signature(optimization).parameters:
            raise ValueError(f"PS-MIP lacks {name}")
    return {"gnies": importlib.metadata.version("gnies"), "gies": validate_runtime(),
            "igsp_invariance": "gaussian", "utigsp_invariance": "gaussian"}


def settings(method, sample_sizes):
    if method not in ALL_METHODS: raise ValueError("unknown chamber method")
    n = sum(sample_sizes)
    base = float(np.log(n) / n)
    if method.startswith("ps_mip"):
        return [{"setting_id": f"graph_{i:02d}_target_fixed16", "tuning_value": c,
                 "graph_penalty": c * base, "target_penalty": [16 * base] * (len(sample_sizes) - 1),
                 "target_penalty_formula": "16*log(N)/N", "environment_weights": (np.asarray(sample_sizes) / n).tolist()}
                for i, c in enumerate(GRAPH_MULTIPLIERS)]
    if method in ("gnies_present", "gies_complete", "gnies_unknown", "gies_oracle"):
        return [{"setting_id": f"native_bic_{i:02d}", "tuning_value": c,
                 "lambda": c * .5 * float(np.log(n))} for i, c in enumerate(NATIVE_MULTIPLIERS)]
    return [{"setting_id": f"alpha_{i:02d}", "tuning_value": alpha, "alpha": alpha,
             "alpha_inv": 1e-5, "invariance_test": "gaussian", "depth": 4, "nruns": 10}
            for i, alpha in enumerate(CI_ALPHAS)]


def target_status(method, documented):
    if method == "ps_mip_unknown": return None
    if method == "ps_mip_complete": return np.asarray(documented, dtype=int).copy()
    if method == "ps_mip_present": return np.where(np.asarray(documented) == 1, 1, -1)
    raise ValueError("not a PS-MIP condition")


def adjacency(value, p):
    a = np.asarray(value)
    if a.shape != (p, p) or not np.isfinite(a).all() or not np.isin(a, (0, 1)).all() or np.any(np.diag(a)):
        raise ValueError("invalid adjacency shape, values, or diagonal")
    return a.astype(int)


def validate_dag(value, p):
    a = adjacency(value, p)
    degree = a.sum(axis=0)
    stack = list(np.flatnonzero(degree == 0))
    visited = 0
    while stack:
        j = stack.pop()
        visited += 1
        for child in np.flatnonzero(a[j]):
            degree[child] -= 1
            if degree[child] == 0: stack.append(child)
    if visited != p: raise ValueError("estimated DAG is cyclic")
    return a


def metrics(dag, graph, metadata):
    p = len(metadata["nodes"])
    dag = validate_dag(dag, p).astype(bool)
    graph = adjacency(graph, p).astype(bool)
    truth = validate_dag(metadata["reference_dag"], p).astype(bool)
    out = {}
    for panel, estimate, reference, mask in (
        ("directed", dag, truth, ~np.eye(p, dtype=bool)),
        ("skeleton", dag | dag.T, truth | truth.T, np.triu(np.ones((p, p), dtype=bool), 1))):
        counts = {"tp": np.sum(estimate & reference & mask), "fp": np.sum(estimate & ~reference & mask),
                  "fn": np.sum(~estimate & reference & mask), "tn": np.sum(~estimate & ~reference & mask)}
        out.update({f"{panel}_{key}": int(value) for key, value in counts.items()})
        out[f"{panel}_tpr"] = float(counts["tp"] / (counts["tp"] + counts["fn"]))
        out[f"{panel}_fpr"] = float(counts["fp"] / (counts["fp"] + counts["tn"]))
    ambiguous = graph & graph.T
    out["ambiguous_undirected_edges"] = int(np.triu(ambiguous, 1).sum())
    out["directed_tp_ambiguous_in_returned_graph"] = int((dag & truth & ambiguous).sum())
    out["directed_fp_ambiguous_in_returned_graph"] = int((dag & ~truth & ambiguous).sum())
    for kind in ("physical", "programmed"):
        ref = np.asarray(metadata[kind + "_edges"], dtype=bool)
        out[kind + "_directed_tp"] = int((dag & ref).sum())
        out[kind + "_skeleton_tp"] = int(((dag | dag.T) & ref).sum())
        out[kind + "_reversed"] = int((dag.T & ref).sum())
    return out


def fit_method(job, arrays, documented, bounds, seed=SEED, threads=8):
    """Never receives graph truth; unknown-target adapters receive no labels."""
    from src.utils import interventional_cpdag
    method, setting = job["method"], job["setting"]
    p = arrays[0].shape[1]
    before = data.array_hash(arrays)
    started = time.monotonic()
    extra = {}
    if method.startswith("ps_mip"):
        import gurobipy as gp
        from MIP_profiled import optimization
        gp.setParam("OutputFlag", 1)
        gp.setParam("Threads", threads)
        gp.setParam("Seed", seed)
        centered = data.ps_arrays(arrays)
        superstructure = ~np.eye(p, dtype=bool)
        gamma, estimated_targets, gap, score, solve_seconds, solver = optimization(
            centered, superstructure, setting["graph_penalty"], l_delta=setting["target_penalty"],
            target_status=target_status(method, documented), max_parents=None, configuration_limit=20_000,
            time_limit=job["time_limit"], return_metadata=True, **bounds["bounds"])
        dag = (np.abs(gamma) > 1e-6).astype(int)
        np.fill_diagonal(dag, 0)
        estimated_targets = np.asarray(estimated_targets, dtype=int)
        graph = interventional_cpdag(dag, estimated_targets)
        b = bounds["bounds"]
        diagonals = list(np.diag(gamma))
        for e, a in enumerate(centered[1:]):
            diagonals.extend((1 / a.std(axis=0))[estimated_targets[e] == 1])
        off = gamma.copy()
        np.fill_diagonal(off, 0)
        extrema = {"min_diagonal": float(min(diagonals)), "max_diagonal": float(max(diagonals)),
                   "max_abs_coefficient": float(np.abs(off).max())}
        inactive = (extrema["min_diagonal"] > b["gamma_lower"] and extrema["max_diagonal"] < b["gamma_upper"]
                    and extrema["max_abs_coefficient"] < b["coefficient_bound"])
        if not inactive: raise ValueError("active numerical bound in selected PS-MIP solution")
        extra = dict(solver, gamma=gamma, objective_value=score, mip_gap=gap, solve_seconds=solve_seconds,
                     screen_edges=p * (p - 1) // 2, screen_parent_sets=p * 2 ** (p - 1),
                     numerical_bounds=b, selected_bound_extrema=extrema, bounds_inactive=inactive)
        representation = "dag_and_i_cpdag"
        preprocessing = "within_environment_centering_and_reference_sd_scaling"
    elif method in ("utigsp_present", "igsp_complete", "utigsp_unknown", "igsp_oracle"):
        arguments = {k: setting[k] for k in ("alpha", "alpha_inv", "invariance_test", "depth", "nruns")}
        if method in ("utigsp_present", "utigsp_unknown"):
            from UTIGSP import fit
            known = documented if method == "utigsp_present" else None
            dag, estimated_targets = fit(arrays, known_interventions=known, seed=seed, **arguments)
        else:
            from IGSP import fit
            dag = fit(arrays, documented, seed=seed, **arguments)
            estimated_targets = documented.copy()
        graph = interventional_cpdag(dag, estimated_targets)
        representation, preprocessing = "dag_and_i_cpdag", "common_affine_only_means_retained"
    elif method in ("gnies_present", "gnies_unknown"):
        import gnies
        from gnies.utils import pdag_to_dag
        known = set(np.flatnonzero(documented.any(axis=0))) if method == "gnies_present" else set()
        score, graph, union = gnies.fit(arrays, approach="greedy", lmbda=setting["lambda"],
                                       center=True, known_targets=known)
        if not known <= set(union): raise ValueError("GnIES dropped a documented target")
        estimated_targets = np.zeros(p, dtype=int)
        estimated_targets[list(union)] = 1
        dag = pdag_to_dag(graph)
        extra = {"objective_value": score}
        representation, preprocessing = "i_cpdag_and_deterministic_consistent_extension", "gnies_center_true"
    elif method in ("gies_complete", "gies_oracle"):
        from GIES import fit
        from gies.utils import pdag_to_dag
        graph, score = fit(arrays, documented, lmbda=setting["lambda"], timeout=job["time_limit"])
        dag, estimated_targets = pdag_to_dag(graph), documented.copy()
        extra = {"objective_value": score}
        representation, preprocessing = "i_cpdag_and_deterministic_consistent_extension", "gies_package_environment_centering"
    else:
        raise ValueError("unknown method")
    after = data.array_hash(arrays)
    if before != after: raise ValueError("method mutated the common input arrays")
    return dict(extra, estimated_dag=validate_dag(dag, p), estimated_graph=adjacency(graph, p),
                estimated_targets=estimated_targets, graph_representation=representation,
                preprocessing=preprocessing, fit_seconds=time.monotonic() - started,
                common_input_sha256_before=before, common_input_sha256_after=after)


def validate_result(row, job, design, allow_failure=False):
    if row["job"] != job or row["design_sha256"] != data.digest(design):
        raise ValueError("mixed or stale result identity")
    if "reuse" in row:
        validate_reused_result(row, job, design, allow_failure)
    if row["status"] not in SUCCESS:
        if allow_failure and row["status"] in FAILURE:
            if row["status"] == "timeout" and (job["method"].startswith("ps_mip")
                    or row.get("failure_kind") != "algorithm_timeout" or not row.get("traceback")):
                raise ValueError("invalid algorithm timeout")
            return
        raise ValueError("unsuccessful fit cannot be presented as successful")
    result = row["result"]
    meta = design["configurations"][job["configuration"]]["data"]
    p, method = len(meta["nodes"]), job["method"]
    dag = validate_dag(result["estimated_dag"], p)
    graph = adjacency(result["estimated_graph"], p)
    estimate = np.asarray(result["estimated_targets"])
    documented = np.asarray(meta["documented_targets"])
    if not np.isin(estimate, (0, 1)).all(): raise ValueError("nonbinary targets")
    if method in ("gnies_present", "gnies_unknown"):
        if estimate.shape != (p,) or (method == "gnies_present" and np.any(estimate[documented.any(axis=0)] != 1)):
            raise ValueError("invalid target union or documented target disappeared")
    else:
        if estimate.shape != documented.shape: raise ValueError("wrong environment target shape")
        if method in ("ps_mip_complete", "gies_complete", "igsp_complete", "gies_oracle", "igsp_oracle") and not np.array_equal(estimate, documented):
            raise ValueError("complete target lists changed")
        if method in ("ps_mip_present", "utigsp_present") and np.any(estimate[documented == 1] != 1):
            raise ValueError("documented target disappeared")
    if result["common_input_sha256_before"] != job["data_sha256"] or result["common_input_sha256_after"] != job["data_sha256"]:
        raise ValueError("common input identity mismatch")
    from gies.utils import is_consistent_extension
    if not is_consistent_extension(dag, graph): raise ValueError("inconsistent DAG representative")
    if method in ("gnies_present", "gnies_unknown"):
        from gnies.utils import dag_to_icpdag
        recomputed = dag_to_icpdag(dag, set(np.flatnonzero(estimate)))
    else:
        from src.utils import interventional_cpdag
        recomputed = interventional_cpdag(dag, estimate)
    if not np.array_equal(graph, recomputed): raise ValueError("saved graph class disagrees with DAG/targets")
    if row["metrics"] != metrics(dag, graph, meta): raise ValueError("metrics disagree with saved graph")
    timing = row["timing"]
    measured = result["solve_seconds"] if method.startswith("ps_mip") else result["fit_seconds"]
    if (timing["fit_limit_seconds"] != job["time_limit"] or not np.isfinite(measured) or measured < 0
            or abs(timing["budget_overrun_seconds"] - max(0., measured - job["time_limit"])) > 1e-6):
        raise ValueError("incorrect saved fit timing")
    if method.startswith("ps_mip"):
        if result["screen_edges"] != p * (p - 1) // 2 or result["screen_parent_sets"] != p * 2 ** (p - 1):
            raise ValueError("unexpected PS-MIP superstructure")
        if result["solution_count"] < 1: raise ValueError("no feasible incumbent")
        if result["numerical_bounds"] != design["configurations"][job["configuration"]]["preflight"]["bounds"] or not result["bounds_inactive"]:
            raise ValueError("invalid numerical-bound identity")
        gamma = np.asarray(result["gamma"])
        if gamma.shape != (p, p) or not np.isfinite(gamma).all(): raise ValueError("invalid fitted Gamma")
        support = (np.abs(gamma) > 1e-6).astype(int)
        np.fill_diagonal(support, 0)
        if not np.array_equal(support, dag): raise ValueError("Gamma and saved DAG disagree")
        b = result["numerical_bounds"]
        if not (np.all(np.diag(gamma) > b["gamma_lower"]) and np.all(np.diag(gamma) < b["gamma_upper"])):
            raise ValueError("active diagonal bound")
        off = gamma.copy()
        np.fill_diagonal(off, 0)
        if np.any(np.abs(off) >= b["coefficient_bound"]): raise ValueError("active coefficient bound")
        if bool(result["optimal"]) != (row["status"] == "ok"):
            raise ValueError("incorrect optimality label")
        if row["status"] == "ok" and (result["solver_status_code"] != 2 or result["mip_gap"] > 1e-4 + 1e-12
              or abs(result["objective_value"] - result["objective_bound"]) > 1e-4 * abs(result["objective_value"]) + 1e-8):
            raise ValueError("uncertified optimality claim")


def validate_reused_result(row, job, design, allow_failure=False):
    """A reuse wrapper retains the complete original row and fitting identity."""
    reuse = row["reuse"]
    source, original = reuse["source_design"], reuse["source_row"]
    if (design["profile"] != PROFILE or source["profile"] != LEGACY_PROFILE
            or job["method"] not in REUSABLE_METHODS or "reuse" in original
            or reuse["source_row_sha256"] != data.digest(original)):
        raise ValueError("invalid reuse provenance or non-reusable method")
    current_code = implementation_identity(LEGACY_PROFILE)
    if set(source["implementation"]) != set(current_code) or any(
            saved != current_code[name] and saved not in POSTPROCESS_COMPATIBLE_SOURCES.get(name, set())
            for name, saved in source["implementation"].items()):
        raise ValueError("unreviewed implementation in reused fit")
    original_job = original["job"]
    if original_job["method"] != REUSABLE_METHODS[job["method"]] or original_job not in source["jobs"]:
        raise ValueError("incompatible reused method")
    for key in ("configurations", "seed", "threads", "native_threads", "timing", "screening", "evaluation"):
        if design[key] != source[key]: raise ValueError(f"reused fit has different {key}")
    for key in ("configuration", "setting_index", "setting", "time_limit", "data_sha256"):
        if job[key] != original_job[key]: raise ValueError(f"reused fit has different {key}")
    omit = {"job", "design_sha256", "reuse"}
    if {k: v for k, v in row.items() if k not in omit} != {k: v for k, v in original.items() if k not in omit}:
        raise ValueError("reused fit payload was altered")
    validate_result(original, original_job, source, allow_failure=allow_failure)


def make_design(data_root=data.DATA_ROOT, threads=8, time_limit=3600, profile=PROFILE):
    time_limit = float(time_limit)
    if threads < 1 or not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("positive threads and finite positive time limit required")
    configurations, jobs = {}, []
    for cfg in ("scm_4",):
        arrays, metadata, bounds = data.load(cfg, data_root)
        configurations[cfg] = {"data": metadata, "preflight": bounds}
        for method in methods_for_profile(profile):
            for index, setting in enumerate(settings(method, [len(a) for a in arrays])):
                jobs.append({"index": len(jobs), "configuration": cfg, "method": method,
                    "setting_index": index, "setting": setting, "target_information": TARGET_INFORMATION[method],
                    "time_limit": time_limit,
                    "data_sha256": metadata["data_sha256"],
                    "output": f"{cfg}/parts/{method}/setting_{index + 1:02d}.json"})
    count = 63 if profile == PROFILE else 97
    if len(jobs) != count or len({j["output"] for j in jobs}) != count:
        raise ValueError(f"expected {count} unique scm_4 settings")
    return {"version": 3 if profile == PROFILE else 2, "profile": profile,
            "role": "controlled_physical_hybrid_pilot_not_biological_validation",
            "configurations": configurations, "jobs": jobs, "seed": SEED, "threads": threads,
            "native_threads": 1, "implementation": implementation_identity(profile), "runtime": runtime_identity(),
            "timing": {"fit_limit_seconds": time_limit, "worker_watchdog_seconds": time_limit + 120,
                       "ps_mip_limit_scope": "native solver; startup, precomputation and saving are separate",
                       "baseline_limit_scope": "entire method adapter with interrupt timer"},
            "screening": {"enabled": False, "candidate_pairs": 45, "parent_sets": 5120},
            "failure_statuses": list(FAILURE),
            "initialization": ("fresh worker per fit; no warm starts; explicit provenance-preserving reuse allowed for unchanged PS-MIP/GIES/IGSP"
                               if profile == PROFILE else "fresh worker per fit; no imported fits or warm starts"),
            "selection_policy": "report all settings; FP budgets 3 and 5 descriptive only; no path promotion",
            "evaluation": "reference DAG counts, no I-CPDAG scoring; deterministic reference-independent extensions",
            "diagnostic_note": "all ten variables and all rows retained regardless of model diagnostics"}


def validate_identity(design, data_root=data.DATA_ROOT, *, postprocess=False):
    """Fitting is strict; saved-result analysis may use a different runtime.

    Never rewrite the saved manifest: each result must still match its original
    design hash. The caller must also validate every saved graph and metric.
    """
    current = make_design(data_root, design["threads"], design["timing"]["fit_limit_seconds"], design["profile"])
    provenance = None
    if postprocess:
        provenance = {"policy": "saved_results_runtime_portable_v1", "design_sha256": data.digest(design),
                      "runtime_matches_fit": design["runtime"] == current["runtime"],
                      "runtime": current["runtime"], "implementation": current["implementation"].copy(),
                      "accepted_validation_code_revisions": {}}
        # Runtime/package identities describe how the fits were produced; they
        # need not describe the machine that later counts edges and draws plots.
        current["runtime"] = design["runtime"]
        accepted_sources = POSTPROCESS_COMPATIBLE_SOURCES if design["profile"] == LEGACY_PROFILE else {}
        for name, accepted in accepted_sources.items():
            saved = design["implementation"].get(name)
            if saved != current["implementation"][name] and saved in accepted:
                provenance["accepted_validation_code_revisions"][name] = {
                    "saved": saved, "current": current["implementation"][name]}
                current["implementation"][name] = saved
    if design != current:
        raise ValueError("stale/incompatible frozen design, code, data, or runtime")
    if postprocess and not provenance["runtime_matches_fit"]:
        warnings.warn("Postprocessing runtime differs from the fitting runtime; continuing with saved-graph "
                      "validation. The original fitting identity is preserved. Fitting/resumption remain strict.",
                      UserWarning, stacklevel=2)
    return provenance
