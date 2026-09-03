#!/usr/bin/env python3
"""Fit one predeclared method/setting on the Sachs biological benchmark.

Each invocation owns one small, atomic CSV fragment.  The fitting jobs save
the returned graph itself (not only performance numbers), so the paper-style
TP/FP counts can be audited and recomputed after the Quest results are copied
back.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
from math import log
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sachs_data import (
    DATA_FORMAT_VERSION,
    DEFAULT_DATA_ROOT,
    SACHS_NODE_NAMES,
    SOURCE_COMMIT,
    load_sachs_data,
)

NODE_NAMES = SACHS_NODE_NAMES


EXPERIMENT_VERSION = 2
PROTOCOL_REVISION = 2
DEFAULT_METHOD_SEED = 20260901
DEFAULT_TIME_LIMIT = 4 * 60 * 60
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "experiment_results" / "sachs" / "parts"

METHODS = (
    "ps_mip_unknown",
    "utigsp_unknown",
    "gnies_unknown",
    "utigsp_intended",
    "ps_mip_oracle",
    "igsp_oracle",
    "gies_oracle",
)
METHOD_LABELS = {
    "ps_mip_unknown": "PS-MIP (unknown targets)",
    "utigsp_unknown": "UT-IGSP* (unknown targets)",
    "gnies_unknown": "GnIES (unknown targets)",
    "utigsp_intended": "UT-IGSP (intended targets supplied)",
    "ps_mip_oracle": "PS-MIP (intended-target oracle)",
    "igsp_oracle": "IGSP (intended-target oracle)",
    "gies_oracle": "GIES (intended-target oracle)",
}
ORACLE_METHODS = {"ps_mip_oracle", "igsp_oracle", "gies_oracle"}
METHOD_PREPROCESSING = {
    "ps_mip_unknown": (
        "within_environment_centering_then_observational_population_sd_scaling"
    ),
    "ps_mip_oracle": (
        "within_environment_centering_then_observational_population_sd_scaling"
    ),
    "utigsp_unknown": "author_log_values; tests handle centering internally",
    "utigsp_intended": "author_log_values; tests handle centering internally",
    "igsp_oracle": "author_log_values; tests handle centering internally",
    "gnies_unknown": "author_log_values; gnies center=True",
    "gies_oracle": "author_log_values; GIES score centers internally",
}
METHOD_IMPLEMENTATION_FILES = {
    "ps_mip_unknown": ("MIP.py", "MIP_profiled.py", "src/utils.py"),
    "ps_mip_oracle": ("MIP.py", "MIP_profiled.py", "src/utils.py"),
    "utigsp_unknown": ("UTIGSP.py", "src/utils.py"),
    "utigsp_intended": ("UTIGSP.py", "src/utils.py"),
    "igsp_oracle": ("IGSP.py", "UTIGSP.py", "src/utils.py"),
    "gnies_unknown": (),
    "gies_oracle": ("GIES.py", "src/main_experiment_cli.py"),
}

# The PS-MIP graph path is deliberately wider than the three-point synthetic
# path: this is one fixed dataset and the goal is an ROC-style regularization
# path.  The unknown-target path holds its target penalty fixed so graph
# sparsity is the only quantity swept along the curve.
PS_BIC_MULTIPLIERS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16)
PS_TARGET_BIC_MULTIPLIER = 16

# This is the eight-point Sachs path in the UT-IGSP authors' released code.
UTIGSP_CI_ALPHAS = (0.1, 0.01, 0.001, 0.2, 0.3, 0.4, 0.5, 0.05)
UTIGSP_INVARIANCE_ALPHA = 1e-20
UTIGSP_DEPTH = 4
UTIGSP_NRUNS = 10

# Exact multipliers used by the GnIES paper's Sachs experiment.  Its runner
# converts each command-line value to multiplier*log(N) before calling GnIES.
GNIES_LAMBDAS = (0.01, 0.25, 0.5, 0.75, 1.0, 2.0)

# Public UT-IGSP history contains GIES multipliers 2/200 and 600--900, but no
# generated files.  Retaining their union makes that provenance explicit and
# spans the very different density regimes of this strongly correlated data.
GIES_BIC_MULTIPLIERS = (2, 200, 600, 700, 800, 900)

COMPLETE_SCREEN_EDGES = len(NODE_NAMES) * (len(NODE_NAMES) - 1) // 2
COMPLETE_PARENT_SETS = len(NODE_NAMES) * 2 ** (len(NODE_NAMES) - 1)
CONFIGURATION_LIMIT = 20_000

METHOD_SOURCES = {
    "utigsp_unknown": {
        "name": "UT-IGSP",
        "url": "https://github.com/uhlerlab/causaldag",
        "distribution": "causaldag",
    },
    "utigsp_intended": {
        "name": "UT-IGSP",
        "url": "https://github.com/uhlerlab/causaldag",
        "distribution": "causaldag",
    },
    "igsp_oracle": {
        "name": "IGSP",
        "url": "https://github.com/uhlerlab/causaldag",
        "distribution": "causaldag",
    },
    "gnies_unknown": {
        "name": "GnIES",
        "url": "https://github.com/juangamella/gnies",
        "distribution": "gnies",
    },
    "gies_oracle": {
        "name": "GIES",
        "url": "https://github.com/juangamella/gies",
        "distribution": "gies",
    },
}

RESULT_FIELDS = [
    "experiment_version",
    "data_format_version",
    "method",
    "method_label",
    "setting_index",
    "setting_id",
    "tuning_parameter",
    "tuning_value",
    "target_knowledge",
    "seed",
    "data_sha256",
    "source_commit",
    "node_names",
    "sample_sizes",
    "n_observational",
    "n_interventional",
    "total_samples",
    "method_config",
    "graph_representation",
    "estimated_dag",
    "estimated_graph",
    "estimated_targets",
    "directed_tp",
    "directed_fp",
    "directed_fn",
    "directed_tn",
    "directed_tpr",
    "directed_fpr",
    "skeleton_tp",
    "skeleton_fp",
    "skeleton_fn",
    "skeleton_tn",
    "skeleton_tpr",
    "skeleton_fpr",
    "screen_type",
    "screen_edges",
    "screen_parent_sets",
    "fit_seconds",
    "solve_seconds",
    "objective_value",
    "objective_bound",
    "mip_gap",
    "solver_status",
    "solver_status_code",
    "optimal",
    "solution_count",
    "node_count",
    "status",
    "error",
]


def _number_id(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _total_samples(data: Iterable[np.ndarray]) -> int:
    return int(sum(len(values) for values in data))


def method_settings(method: str, data: Iterable[np.ndarray]) -> list[dict[str, Any]]:
    """Return the ordered, predeclared path for one Sachs method."""
    if method not in METHODS:
        raise ValueError(f"unknown Sachs method {method!r}")
    total = _total_samples(data)
    if method == "ps_mip_unknown":
        base = log(total) / total
        return [
            {
                "setting_id": (
                    f"graph_x{_number_id(graph_multiplier)}"
                    f"_target_x{_number_id(PS_TARGET_BIC_MULTIPLIER)}"
                ),
                "tuning_parameter": "graph_bic_multiplier",
                "tuning_value": graph_multiplier,
                "graph_penalty": graph_multiplier * base,
                "target_penalty": PS_TARGET_BIC_MULTIPLIER * base,
                "bic_multiplier": graph_multiplier,
                "target_bic_multiplier": PS_TARGET_BIC_MULTIPLIER,
            }
            for graph_multiplier in PS_BIC_MULTIPLIERS
        ]
    if method == "ps_mip_oracle":
        base = log(total) / total
        return [
            {
                "setting_id": f"bic_x{_number_id(multiplier)}",
                "tuning_parameter": "graph_bic_multiplier",
                "tuning_value": multiplier,
                "graph_penalty": multiplier * base,
                # This term is constant when targets are fixed, but saving the
                # common base value keeps the score contract explicit.
                "target_penalty": base,
                "bic_multiplier": multiplier,
            }
            for multiplier in PS_BIC_MULTIPLIERS
        ]
    if method in {"utigsp_unknown", "utigsp_intended", "igsp_oracle"}:
        return [
            {
                "setting_id": f"alpha_{_number_id(alpha)}",
                "tuning_parameter": "conditional_independence_alpha",
                "tuning_value": alpha,
                "alpha": alpha,
                "alpha_inv": UTIGSP_INVARIANCE_ALPHA,
            }
            for alpha in UTIGSP_CI_ALPHAS
        ]
    if method == "gnies_unknown":
        return [
            {
                "setting_id": f"author_logN_x{_number_id(multiplier)}",
                "tuning_parameter": "author_multiplier_times_log_n",
                "tuning_value": multiplier,
                "lambda_multiplier": multiplier,
                "lambda": multiplier * log(total),
            }
            for multiplier in GNIES_LAMBDAS
        ]
    if method == "gies_oracle":
        base = 0.5 * log(total)
        return [
            {
                "setting_id": f"author_bic_x{_number_id(multiplier)}",
                "tuning_parameter": "author_multiplier_times_half_log_n",
                "tuning_value": multiplier,
                "bic_multiplier": multiplier,
                "lambda": multiplier * base,
            }
            for multiplier in GIES_BIC_MULTIPLIERS
        ]
    raise AssertionError("unreachable method branch")


def method_setting_ids(method: str, data: Iterable[np.ndarray]) -> list[str]:
    return [setting["setting_id"] for setting in method_settings(method, data)]


def _json(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _binary_adjacency(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    p = len(NODE_NAMES)
    if array.shape != (p, p) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {p}-by-{p} matrix")
    if not np.all(np.isclose(array, 0) | np.isclose(array, 1)):
        raise ValueError(f"{name} must contain only zero/one entries")
    result = np.isclose(array, 1).astype(np.uint8)
    if np.any(np.diag(result)):
        raise ValueError(f"{name} must have a zero diagonal")
    return result


def _is_acyclic(adjacency: np.ndarray) -> bool:
    indegree = adjacency.sum(axis=0).astype(int)
    ready = [int(node) for node in np.flatnonzero(indegree == 0)]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    return visited == len(adjacency)


def _validate_dag(values: Any) -> np.ndarray:
    dag = _binary_adjacency(values, name="estimated DAG")
    if not _is_acyclic(dag):
        raise ValueError("estimated DAG is cyclic")
    return dag


def roc_counts(estimated_dag: Any, true_dag: Any) -> dict[str, float | int]:
    """Compute both paper-style counts and conventional ROC rates."""
    estimated = _validate_dag(estimated_dag).astype(bool)
    truth = _validate_dag(true_dag).astype(bool)
    off_diagonal = ~np.eye(len(truth), dtype=bool)

    directed_tp = int(np.sum(estimated & truth))
    directed_fp = int(np.sum(estimated & ~truth & off_diagonal))
    directed_fn = int(np.sum(~estimated & truth))
    directed_tn = int(np.sum(~estimated & ~truth & off_diagonal))

    estimated_skeleton = estimated | estimated.T
    true_skeleton = truth | truth.T
    upper = np.triu(np.ones_like(truth, dtype=bool), k=1)
    skeleton_tp = int(np.sum(estimated_skeleton & true_skeleton & upper))
    skeleton_fp = int(np.sum(estimated_skeleton & ~true_skeleton & upper))
    skeleton_fn = int(np.sum(~estimated_skeleton & true_skeleton & upper))
    skeleton_tn = int(np.sum(~estimated_skeleton & ~true_skeleton & upper))

    return {
        "directed_tp": directed_tp,
        "directed_fp": directed_fp,
        "directed_fn": directed_fn,
        "directed_tn": directed_tn,
        "directed_tpr": directed_tp / (directed_tp + directed_fn),
        "directed_fpr": directed_fp / (directed_fp + directed_tn),
        "skeleton_tp": skeleton_tp,
        "skeleton_fp": skeleton_fp,
        "skeleton_fn": skeleton_fn,
        "skeleton_tn": skeleton_tn,
        "skeleton_tpr": skeleton_tp / (skeleton_tp + skeleton_fn),
        "skeleton_fpr": skeleton_fp / (skeleton_fp + skeleton_tn),
    }


@contextmanager
def _wall_time_limit(seconds: float, label: str):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("time limit must be positive and finite")
    previous = signal.getsignal(signal.SIGALRM)

    def timeout(_signum, _frame):
        raise TimeoutError(f"{label} exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _implementation_fingerprint(method: str) -> dict[str, Any]:
    """Fingerprint the local estimator code that is part of run identity."""
    relative_paths = (
        "experiments/run_sachs_roc.py",
        "src/sachs_data.py",
        *METHOD_IMPLEMENTATION_FILES[method],
    )
    file_digests: dict[str, str] = {}
    combined = hashlib.sha256()
    combined.update(f"sachs-protocol-{PROTOCOL_REVISION}\0".encode("ascii"))
    for relative_path in relative_paths:
        path = PROJECT_ROOT / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        file_digests[relative_path] = digest
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\0")
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "files": file_digests,
        "sha256": combined.hexdigest(),
    }


def _seed_usage(method: str) -> str:
    if method.startswith("ps_mip"):
        return "gurobi_Seed"
    if method in {"utigsp_unknown", "utigsp_intended", "igsp_oracle"}:
        return "python_and_numpy_search_initialization"
    return "not_applicable_deterministic_package"


def _runtime_environment_fingerprint() -> dict[str, Any]:
    """Record enough of the resolved Python environment to detect path drift."""
    distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            canonical = re.sub(r"[-_.]+", "-", name).lower()
            distributions[canonical] = distribution.version
    distributions = dict(sorted(distributions.items()))
    payload = {
        "python_version": sys.version.split()[0],
        "distributions": distributions,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    key_names = (
        "numpy",
        "scipy",
        "scikit-learn",
        "causaldag",
        "gnies",
        "gies",
        "gurobipy",
    )
    return {
        "python_version": payload["python_version"],
        "distributions_sha256": digest,
        "key_distributions": {
            name: distributions[name] for name in key_names if name in distributions
        },
    }


def _runtime_source(method: str) -> dict[str, Any]:
    if method.startswith("ps_mip"):
        return {
            "name": "PS-MIP",
            "implementation": "MIP_profiled.optimization",
            "gurobipy_version": _distribution_version("gurobipy"),
        }
    source = dict(METHOD_SOURCES[method])
    source["version"] = _distribution_version(source.pop("distribution"))
    return source


def _ps_data(data: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Center each context and scale columns by observational variability."""
    arrays = [np.asarray(values, dtype=float) for values in data]
    scale = arrays[0].std(axis=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("observational columns must have positive finite variance")
    return [(values - values.mean(axis=0)) / scale for values in arrays]


def _complete_superstructure(p: int) -> np.ndarray:
    moral = np.ones((p, p), dtype=bool)
    np.fill_diagonal(moral, False)
    return moral


def _run_ps_mip(
    data: list[np.ndarray],
    intended_targets: np.ndarray | None,
    setting: dict[str, Any],
    *,
    oracle: bool,
    seed: int,
    threads: int,
    time_limit: float,
) -> dict[str, Any]:
    try:
        import gurobipy as gp
    except ImportError as error:
        raise ImportError("PS-MIP requires gurobipy and a usable Gurobi license") from error
    from MIP_profiled import optimization
    from src.utils import interventional_cpdag

    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", threads)
    gp.setParam("Seed", seed)
    transformed = _ps_data(data)
    moral = _complete_superstructure(transformed[0].shape[1])
    started = time.perf_counter()
    if oracle and intended_targets is None:
        raise ValueError("oracle PS-MIP requires intended targets")
    if not oracle and intended_targets is not None:
        raise ValueError("unknown-target PS-MIP must not receive intended targets")
    result = optimization(
        transformed,
        moral,
        setting["graph_penalty"],
        l_delta=setting["target_penalty"],
        target_status=intended_targets,
        max_parents=None,
        configuration_limit=CONFIGURATION_LIMIT,
        time_limit=time_limit,
        return_metadata=True,
    )
    gamma, targets, gap, objective, solve_seconds, metadata = result
    targets = np.asarray(targets, dtype=np.uint8)
    dag_support = np.asarray(np.abs(gamma) > 1e-6, dtype=np.uint8)
    # The fitted precision-factor representation carries positive scale
    # parameters on its diagonal; they are not graph self-loops.
    np.fill_diagonal(dag_support, 0)
    dag = _validate_dag(dag_support)
    graph = _binary_adjacency(
        interventional_cpdag(dag, targets), name="estimated I-CPDAG"
    )
    return {
        "estimated_dag": dag,
        "estimated_graph": graph,
        "estimated_targets": targets,
        "graph_representation": "dag_and_i_cpdag",
        "fit_seconds": time.perf_counter() - started,
        "solve_seconds": solve_seconds,
        "objective_value": objective,
        "objective_bound": metadata["objective_bound"],
        "mip_gap": gap,
        "solver_status": metadata["solver_status"],
        "solver_status_code": metadata["solver_status_code"],
        "optimal": metadata["optimal"],
        "solution_count": metadata["solution_count"],
        "node_count": metadata["node_count"],
        "screen_type": "complete_undirected_superstructure",
        "screen_edges": COMPLETE_SCREEN_EDGES,
        "screen_parent_sets": COMPLETE_PARENT_SETS,
        "preprocessing": (
            "within_environment_centering_then_observational_population_sd_scaling"
        ),
    }


def _run_utigsp(
    data: list[np.ndarray],
    setting: dict[str, Any],
    *,
    known_interventions: np.ndarray | None,
    seed: int,
    time_limit: float,
) -> dict[str, Any]:
    from UTIGSP import fit
    from src.utils import interventional_cpdag

    started = time.perf_counter()
    with _wall_time_limit(time_limit, "UT-IGSP fit"):
        dag, targets = fit(
            data,
            known_interventions=known_interventions,
            alpha=setting["alpha"],
            alpha_inv=setting["alpha_inv"],
            depth=UTIGSP_DEPTH,
            nruns=UTIGSP_NRUNS,
            seed=seed,
        )
    dag = _validate_dag(dag)
    targets = np.asarray(targets, dtype=np.uint8)
    graph = _binary_adjacency(
        interventional_cpdag(dag, targets), name="estimated I-CPDAG"
    )
    return {
        "estimated_dag": dag,
        "estimated_graph": graph,
        "estimated_targets": targets,
        "graph_representation": "dag_and_i_cpdag",
        "fit_seconds": time.perf_counter() - started,
        "preprocessing": "author_log_values; tests handle centering internally",
    }


def _run_igsp(
    data: list[np.ndarray],
    intended_targets: np.ndarray,
    setting: dict[str, Any],
    *,
    seed: int,
    time_limit: float,
) -> dict[str, Any]:
    from IGSP import fit
    from src.utils import interventional_cpdag

    started = time.perf_counter()
    with _wall_time_limit(time_limit, "IGSP fit"):
        dag = fit(
            data,
            intended_targets,
            alpha=setting["alpha"],
            alpha_inv=setting["alpha_inv"],
            depth=UTIGSP_DEPTH,
            nruns=UTIGSP_NRUNS,
            seed=seed,
        )
    dag = _validate_dag(dag)
    graph = _binary_adjacency(
        interventional_cpdag(dag, intended_targets), name="estimated I-CPDAG"
    )
    return {
        "estimated_dag": dag,
        "estimated_graph": graph,
        "estimated_targets": intended_targets,
        "graph_representation": "dag_and_i_cpdag",
        "fit_seconds": time.perf_counter() - started,
        "preprocessing": "author_log_values; tests handle centering internally",
    }


def _pdag_extension(graph: Any, package: str) -> np.ndarray:
    if package == "gnies":
        from gnies.utils import pdag_to_dag
    elif package == "gies":
        from gies.utils import pdag_to_dag
    else:
        raise ValueError(f"unknown extension package {package}")
    return _validate_dag(pdag_to_dag(np.asarray(graph, dtype=int)))


def _run_gnies(
    data: list[np.ndarray], setting: dict[str, Any], *, time_limit: float
) -> dict[str, Any]:
    try:
        import gnies
    except ImportError as error:
        raise ImportError("GnIES requires the gnies package") from error

    started = time.perf_counter()
    with _wall_time_limit(time_limit, "GnIES fit"):
        score, graph, target_union = gnies.fit(
            data,
            approach="greedy",
            lmbda=setting["lambda"],
            center=True,
        )
    graph = _binary_adjacency(graph, name="estimated GnIES I-CPDAG")
    dag = _pdag_extension(graph, "gnies")
    target_vector = np.zeros(len(NODE_NAMES), dtype=np.uint8)
    target_vector[list(target_union)] = 1
    return {
        "estimated_dag": dag,
        "estimated_graph": graph,
        "estimated_targets": target_vector,
        "graph_representation": "i_cpdag_and_deterministic_consistent_extension",
        "fit_seconds": time.perf_counter() - started,
        "objective_value": score,
        "preprocessing": "author_log_values; gnies center=True",
    }


def _run_gies(
    data: list[np.ndarray],
    intended_targets: np.ndarray,
    setting: dict[str, Any],
    *,
    time_limit: float,
) -> dict[str, Any]:
    from GIES import fit

    started = time.perf_counter()
    graph, score = fit(
        data,
        intended_targets,
        lmbda=setting["lambda"],
        timeout=time_limit,
    )
    graph = _binary_adjacency(graph, name="estimated GIES I-CPDAG")
    dag = _pdag_extension(graph, "gies")
    return {
        "estimated_dag": dag,
        "estimated_graph": graph,
        "estimated_targets": intended_targets,
        "graph_representation": "i_cpdag_and_deterministic_consistent_extension",
        "fit_seconds": time.perf_counter() - started,
        "objective_value": score,
        "preprocessing": "author_log_values; GIES score centers internally",
    }


def _run_method(
    method: str,
    data: list[np.ndarray],
    intended_targets: np.ndarray,
    setting: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Unknown-target branches receive no accepted graph or intended-target
    # matrix.  Truth crosses the method boundary only in named oracle methods.
    if method == "ps_mip_unknown":
        return _run_ps_mip(
            data,
            None,
            setting,
            oracle=False,
            seed=args.seed,
            threads=args.threads,
            time_limit=args.time_limit,
        )
    if method == "utigsp_unknown":
        return _run_utigsp(
            data,
            setting,
            known_interventions=None,
            seed=args.seed,
            time_limit=args.time_limit,
        )
    if method == "gnies_unknown":
        return _run_gnies(data, setting, time_limit=args.time_limit)
    if method == "utigsp_intended":
        return _run_utigsp(
            data,
            setting,
            known_interventions=intended_targets,
            seed=args.seed,
            time_limit=args.time_limit,
        )
    if method == "ps_mip_oracle":
        return _run_ps_mip(
            data,
            intended_targets,
            setting,
            oracle=True,
            seed=args.seed,
            threads=args.threads,
            time_limit=args.time_limit,
        )
    if method == "igsp_oracle":
        return _run_igsp(
            data,
            intended_targets,
            setting,
            seed=args.seed,
            time_limit=args.time_limit,
        )
    if method == "gies_oracle":
        return _run_gies(
            data, intended_targets, setting, time_limit=args.time_limit
        )
    raise ValueError(f"unknown method {method}")


def _base_row(
    args: argparse.Namespace,
    data: list[np.ndarray],
    info: dict[str, Any],
    setting: dict[str, Any],
) -> dict[str, Any]:
    sizes = [len(values) for values in data]
    method_config = {
        "source": _runtime_source(args.method),
        "implementation": _implementation_fingerprint(args.method),
        "runtime_environment": _runtime_environment_fingerprint(),
        "setting": setting,
        "fit_time_limit_seconds": args.time_limit,
        "threads": args.threads,
        "seed_usage": _seed_usage(args.method),
        "data_preprocessing": METHOD_PREPROCESSING[args.method],
        "roc_basis": "returned_or_deterministic_representative_dag",
        "utigsp_protocol_note": (
            "modern Gaussian-invariance adapter with alpha_inv=1e-20; the public "
            "paper-era code/result provenance is internally inconsistent"
        )
        if args.method in {"utigsp_unknown", "utigsp_intended", "igsp_oracle"}
        else None,
    }
    return {field: "" for field in RESULT_FIELDS} | {
        "experiment_version": EXPERIMENT_VERSION,
        "data_format_version": DATA_FORMAT_VERSION,
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "setting_index": args.setting_index,
        "setting_id": setting["setting_id"],
        "tuning_parameter": setting["tuning_parameter"],
        "tuning_value": setting["tuning_value"],
        "target_knowledge": (
            "intended_targets_oracle"
            if args.method in ORACLE_METHODS
            else "intended_targets_known_present"
            if args.method == "utigsp_intended"
            else "unknown"
        ),
        "seed": args.seed,
        "data_sha256": info["data_sha256"],
        "source_commit": SOURCE_COMMIT,
        "node_names": _json(list(NODE_NAMES)),
        "sample_sizes": _json(sizes),
        "n_observational": sizes[0],
        "n_interventional": sum(sizes[1:]),
        "total_samples": sum(sizes),
        "method_config": _json(method_config),
        "status": "started",
    }


def _add_result(
    row: dict[str, Any], result: dict[str, Any], true_dag: np.ndarray
) -> None:
    estimated_dag = _validate_dag(result["estimated_dag"])
    row.update(roc_counts(estimated_dag, true_dag))
    row["estimated_dag"] = _json(estimated_dag)
    row["estimated_graph"] = _json(result.get("estimated_graph"))
    row["estimated_targets"] = _json(result.get("estimated_targets"))
    row["graph_representation"] = result["graph_representation"]
    for field in (
        "screen_type",
        "screen_edges",
        "screen_parent_sets",
        "fit_seconds",
        "solve_seconds",
        "objective_value",
        "objective_bound",
        "mip_gap",
        "solver_status",
        "solver_status_code",
        "optimal",
        "solution_count",
        "node_count",
    ):
        if field in result:
            row[field] = result[field]
    config = json.loads(row["method_config"])
    if result["preprocessing"] != config["data_preprocessing"]:
        raise ValueError(
            "method result preprocessing does not match the declared run identity"
        )
    row["status"] = (
        "ok_nonoptimal" if result.get("optimal") is False else "ok"
    )


def _read_one_row(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != RESULT_FIELDS or len(rows) != 1:
        raise RuntimeError(f"{path} is not a one-row Sachs result fragment")
    return rows[0]


def _matches_identity(row: dict[str, str], expected: dict[str, Any]) -> bool:
    keys = (
        "experiment_version",
        "data_format_version",
        "method",
        "setting_index",
        "setting_id",
        "tuning_parameter",
        "tuning_value",
        "target_knowledge",
        "seed",
        "data_sha256",
        "source_commit",
        "method_config",
    )
    return all(str(row[key]) == str(expected[key]) for key in keys)


def _write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _output_lock(path: Path):
    """Prevent overlapping submissions from fitting the same fragment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    with lock_path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another process is already working on result {path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _default_output(method: str, setting_index: int) -> Path:
    return (
        DEFAULT_RESULTS_ROOT
        / method
        / f"setting_{setting_index + 1:02d}.csv"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one method/setting for the Sachs ROC benchmark."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--setting-index",
        type=int,
        required=True,
        help="zero-based index in the selected method's predeclared path",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_METHOD_SEED)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="rerun a matching checkpoint whose status is not successful",
    )
    parser.add_argument(
        "--retry-nonoptimal",
        action="store_true",
        help=(
            "rerun a matching PS-MIP checkpoint that contains a feasible "
            "but nonoptimal solution"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing fragment even when its identity differs",
    )
    args = parser.parse_args(argv)
    if args.setting_index < 0:
        parser.error("--setting-index must be nonnegative")
    if args.seed < 0 or args.seed > 2**32 - 1:
        parser.error("--seed must be between 0 and 2^32-1")
    if args.method.startswith("ps_mip") and args.seed > 2_000_000_000:
        parser.error("PS-MIP --seed must not exceed Gurobi's limit of 2000000000")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not np.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit must be positive and finite")
    args.data_root = args.data_root.expanduser().resolve()
    args.output = (
        _default_output(args.method, args.setting_index)
        if args.output is None
        else args.output.expanduser().resolve()
    )
    return args


def _run_locked(args: argparse.Namespace) -> dict[str, Any]:
    data, true_dag, intended_targets, info = load_sachs_data(args.data_root)
    settings = method_settings(args.method, data)
    if args.setting_index >= len(settings):
        raise ValueError(
            f"setting index {args.setting_index} is out of range for "
            f"{args.method}; expected 0..{len(settings) - 1}"
        )
    setting = settings[args.setting_index]
    row = _base_row(args, data, info, setting)

    # --overwrite is also the recovery path for a truncated or obsolete
    # fragment, so do not require the old file to be parseable in that mode.
    existing = None if args.overwrite else _read_one_row(args.output)
    if existing is not None:
        if not _matches_identity(existing, row):
            raise RuntimeError(
                f"existing result identity does not match requested run: {args.output}"
            )
        existing_status = existing["status"]
        should_retry = (
            existing_status == "ok_nonoptimal" and args.retry_nonoptimal
        ) or (
            existing_status not in {"ok", "ok_nonoptimal"}
            and args.retry_failures
        )
        if not should_retry:
            print(f"checkpointed ({existing['status']}): {args.output}", flush=True)
            if existing_status not in {"ok", "ok_nonoptimal"}:
                raise RuntimeError(
                    "matching checkpoint is not successful; inspect it and rerun "
                    "with --retry-failures, or use --overwrite after changing the "
                    "code or environment"
                )
            return existing

    started = time.perf_counter()
    try:
        result = _run_method(
            args.method, data, intended_targets, setting, args
        )
        _add_result(row, result, true_dag)
    except Exception as error:
        row["fit_seconds"] = time.perf_counter() - started
        row["status"] = "fit_error"
        row["error"] = f"{type(error).__name__}: {error}"
        _write_row(args.output, row)
        print(f"failed result saved: {args.output}", flush=True)
        raise

    _write_row(args.output, row)
    print(
        f"saved {args.method} {setting['setting_id']} ({row['status']}): "
        f"{args.output}",
        flush=True,
    )
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    with _output_lock(args.output):
        return _run_locked(args)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
