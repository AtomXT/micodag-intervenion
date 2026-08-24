#!/usr/bin/env python3
"""Run the main synthetic comparison on shared generated instances."""

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
import subprocess
import sys
import time
from contextlib import contextmanager
from math import comb
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPENDENCY_LOCK_PATH = PROJECT_ROOT / "requirements-dcdi.txt"
DEPENDENCY_LOCK_SHA256 = hashlib.sha256(DEPENDENCY_LOCK_PATH.read_bytes()).hexdigest()

from src.main_experiment_data import (
    DATA_FORMAT_VERSION,
    DEFAULT_DATA_ROOT,
    DEFAULT_MASTER_SEED,
    DEFAULT_NUM_REPLICATES,
    EDGE_MULTIPLIERS,
    MainExperimentDataError,
    MAX_MASTER_SEED,
    N_INTERVENTIONAL,
    N_OBSERVATIONAL,
    NUM_ENVIRONMENTS,
    NUM_INTERVENTIONAL_ENVIRONMENTS,
    P_VALUES,
    data_digest,
    load_main_experiment_instance,
    load_main_experiment_manifest,
)


MAIN_EXPERIMENT_VERSION = 6
COMPLETED_STATUSES = {"ok", "ok_nonoptimal"}
FAILURE_STATUSES = {"fit_error", "metric_error"}

METHODS = (
    "ps_mip_unknown",
    "dcdi_g_unknown",
    "utigsp_unknown",
    "gnies_unknown",
    "ps_mip_oracle",
    "gies_oracle",
)
ORACLE_METHODS = {"ps_mip_oracle", "gies_oracle"}
PRIMARY_SETTING_IDS = {
    "ps_mip_unknown": "bic_x1",
    "dcdi_g_unknown": "g0p1_t0p001",
    "utigsp_unknown": "alpha_1em05",
    "gnies_unknown": "bic_x1",
    "ps_mip_oracle": "g1",
    "gies_oracle": "bic_x1",
}

# Main tuning grids. BIC multipliers are applied to each method's native BIC
# scale. PS-MIP follows a tied one-dimensional path; DCDI uses a small
# literature-derived two-penalty cross rather than a large Cartesian search.
PS_UNKNOWN_BIC_MULTIPLIERS = (0.5, 1.0, 2.0)
PS_ORACLE_GRAPH_MULTIPLIERS = (0.5, 1.0, 2.0)
DCDI_GRID = (
    ("g0p01_t0p001", 0.01, 0.001),
    ("g0p1_t0p0001", 0.1, 0.0001),
    ("g0p1_t0p001", 0.1, 0.001),
    ("g0p1_t0p01", 0.1, 0.01),
    ("g1_t0p001", 1.0, 0.001),
)
SCORE_BIC_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
UTIGSP_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 5e-2)

# PS-MIP screens only the observational sample. Its scale follows the requested
# high-dimensional order rather than a fixed graphical-lasso regularizer.
SCREEN_CONSTANT = 5.0
MAX_PARENTS = None
CONFIGURATION_LIMIT = 1_100_000
UTIGSP_DEPTH = 4
UTIGSP_NRUNS = 10
DCDI_HIDDEN_DIM = 8
DCDI_MU_INIT = 1e-2
DCDI_MAX_ITERATIONS = 1_000_000

OFFICIAL_IMPLEMENTATIONS = {
    "dcdi_g_unknown": {
        "name": "DCDI",
        "source_url": "https://github.com/slachapelle/dcdi",
        "source_version": "594d328eae7795785e0d1a1138945e28a4fec037",
        "entrypoint": "main.py",
    },
    "utigsp_unknown": {
        "name": "UT-IGSP",
        "source_url": "https://github.com/uhlerlab/causaldag",
        "tested_version": "causaldag==0.1a163",
        "source_commit": "c664971bbffa3a42097564d58a92b763fe09c4f5",
        "tested_algorithm_package": "graphical-model-learning==0.1a8",
        "algorithm_source_url": "https://github.com/uhlerlab/graphical_model_learning",
    },
    "gnies_unknown": {
        "name": "GnIES",
        "source_url": "https://github.com/juangamella/gnies",
        "tested_version": "gnies==0.3.3",
        "source_commit": "eac150951469d0f059978ae319cabfadf102b02b",
        "tested_algorithm_dependency": "ges==1.1.1",
        "algorithm_dependency_source": "c7eaad4d4150f146ab349bb95a5b7043cb3dc90d",
    },
    "gies_oracle": {
        "name": "GIES",
        "source_url": "https://github.com/juangamella/gies",
        "tested_version": "gies==0.0.3",
        "implementation_authors": "Olga Kolotuhina and Juan L. Gamella",
    },
}

TESTED_PYTHON_REQUIREMENTS = {
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "pandas": "2.2.3",
    "matplotlib": "3.9.4",
    "seaborn": "0.11.2",
    "networkx": "3.2.1",
    "scikit-learn": "1.6.1",
    "torch": "2.2.2",
    "cdt": "0.6.0",
    "urllib3": "1.26.20",
    "causaldag": "0.1a163",
    "graphical-model-learning": "0.1a8",
    "conditional-independence": "0.1a6",
    "graphical-models": "0.1a21",
    "pgmpy": "0.1.25",
    "gnies": "0.3.3",
    "ges": "1.1.1",
    "gies": "0.0.3",
    "gurobipy": "12.0.3",
}


class OfficialRuntimeError(RuntimeError):
    """Raised when a selected official baseline is unavailable or unusable."""

SMOKE_TIME_LIMIT = 60.0
DEFAULT_TIME_LIMIT = 3600.0
RUNTIME_PROBE_TIMEOUT = 60.0

FIELDS = [
    "experiment_version", "p", "e", "replicate", "seed", "method",
    "setting_id", "tuning_parameter", "tuning_value", "target_setting",
    "num_environments", "num_interventional_environments",
    "n_observational", "n_interventional", "total_samples",
    "target_edge_count", "edge_probability", "realized_edges", "data_sha256",
    "instance_sha256",
    "screen_constant", "screen_alpha", "screen_edges", "screen_parent_sets",
    "penalty_rule", "graph_penalty", "target_penalty", "method_config",
    "d_cpdag", "fdp", "tdp", "target_hamming", "union_target_error",
    "estimated_targets", "estimated_target_union", "true_targets",
    "fit_seconds", "metric_seconds", "solve_seconds", "objective_value",
    "objective_bound", "mip_gap", "solver_status", "solver_status_code",
    "optimal", "solution_count", "node_count", "artifact_dir", "cached",
    "status", "error",
]


def _json_array(values):
    if values is None:
        return ""
    return json.dumps(np.asarray(values, dtype=int).tolist(), separators=(",", ":"))


def _identify_python_distributions(distributions):
    """Return installed versions, rejecting only missing distributions."""
    installed = {}
    errors = []
    for distribution in distributions:
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{distribution} is not installed")
            continue
        installed[distribution] = actual
    if errors:
        raise OfficialRuntimeError(
            "official baseline dependency check failed: " + "; ".join(errors)
        )
    return installed


def _resolved_python_environment():
    """Record all resolved distributions so transitive drift changes identity."""
    distributions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            canonical = re.sub(r"[-_.]+", "-", name).lower()
            distributions[canonical] = distribution.version
    distributions = dict(sorted(distributions.items()))
    payload = json.dumps(
        distributions, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "distributions": distributions,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _probe_selected_python_apis(methods):
    """Import the exact official entry APIs before any expensive fit begins."""
    selected = set(methods)
    statements = ["import numpy, scipy"]
    if selected & {"ps_mip_unknown", "ps_mip_oracle"}:
        statements.append("import gurobipy, sklearn")
    if "utigsp_unknown" in selected:
        statements.append(
            "from causaldag import (MemoizedCI_Tester, "
            "MemoizedInvarianceTester, gauss_invariance_suffstat, "
            "gauss_invariance_test, partial_correlation_suffstat, "
            "partial_correlation_test, unknown_target_igsp)"
        )
    if "gnies_unknown" in selected:
        statements.extend(
            ["import ges, gnies", "assert callable(gnies.fit)"]
        )
    if "gies_oracle" in selected:
        statements.extend(
            [
                "import gies",
                "from gies.scores import GaussIntL0Pen",
                "assert callable(gies.fit) and callable(gies.fit_bic)",
            ]
        )
    try:
        subprocess.run(
            [sys.executable, "-c", ";".join(statements)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=RUNTIME_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as error:
        raise OfficialRuntimeError(
            f"official Python API preflight exceeded {RUNTIME_PROBE_TIMEOUT:g} seconds"
        ) from error
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or ""
        raise OfficialRuntimeError(
            "official Python API preflight failed"
            f"{': ' + detail.strip()[-2000:] if detail.strip() else ''}"
        ) from error


def _validate_competitor_runtime(args):
    """Bind every selected competitor row to the implementation actually used."""
    installed_distributions = _identify_python_distributions(
        TESTED_PYTHON_REQUIREMENTS.keys()
    )
    _probe_selected_python_apis(args.methods)
    identified_python = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "official_api_preflight": "passed",
        "installed_distributions": installed_distributions,
        "resolved_environment": _resolved_python_environment(),
    }
    runtime = {"shared_python": identified_python}
    selected = set(args.methods)
    if "utigsp_unknown" in selected:
        runtime["utigsp_unknown"] = identified_python
    if "gnies_unknown" in selected:
        runtime["gnies_unknown"] = identified_python
    if "gies_oracle" in selected:
        from GIES import validate_runtime

        try:
            runtime["gies_oracle"] = validate_runtime()
        except Exception as error:
            raise OfficialRuntimeError(
                f"requested Python GIES runtime check failed: {error}"
            ) from error
    if "dcdi_g_unknown" in selected:
        from DCDI import validate_official_install

        try:
            dcdi_runtime = validate_official_install(
                args.dcdi_root, rscript=args.rscript
            )
        except Exception as error:
            raise OfficialRuntimeError(
                f"official DCDI runtime check failed: {error}"
            ) from error
        runtime["dcdi_g_unknown"] = dcdi_runtime
    return runtime


def _key(row):
    return (
        int(row["p"]), int(row["e"]), int(row["replicate"]), row["method"],
        row["setting_id"],
    )


def _checkpoint_matches(
    previous, row, data_sha256, instance_sha256, seed, *, retry_failures=False
):
    status = previous.get("status", "")
    reusable_status = status in COMPLETED_STATUSES or (
        status in FAILURE_STATUSES and not retry_failures
    )
    return (
        reusable_status
        and previous.get("data_sha256") == data_sha256
        and previous.get("instance_sha256") == instance_sha256
        and previous.get("seed") == str(seed)
        and previous.get("method_config") == row["method_config"]
    )


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {
            "experiment_version", "setting_id"
        }.issubset(reader.fieldnames):
            raise RuntimeError(
                f"{path} uses the pre-grid result schema; choose a new output path"
            )
        rows = [
            {field: row.get(field, "") for field in FIELDS}
            for row in reader
        ]
    stale = [
        row for row in rows
        if row["experiment_version"] != str(MAIN_EXPERIMENT_VERSION)
    ]
    if stale:
        raise RuntimeError(
            f"{path} contains results from a different experiment version"
        )
    return rows


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=_key))
    os.replace(temporary, path)


@contextmanager
def _result_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another main experiment is using {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _time_limit(seconds, label):
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


def _bic_penalty(data):
    total_samples = sum(len(values) for values in data)
    return float(np.log(total_samples) / total_samples)


def _score_bic_penalty(data):
    return float(0.5 * np.log(sum(len(values) for values in data)))


def _screen_alpha(data):
    p = data[0].shape[1]
    n_screen = len(data[0])
    return float(SCREEN_CONSTANT * np.sqrt(np.log(p) / n_screen))


def _screen_statistics(moral):
    moral = np.asarray(moral, dtype=bool)
    degrees = moral.sum(axis=0).astype(int)
    screen_edges = int(np.triu(moral, k=1).sum())
    screen_parent_sets = int(
        sum(
            comb(int(degree), size)
            for degree in degrees
            for size in range(
                min(
                    int(degree),
                    MAX_PARENTS if MAX_PARENTS is not None else int(degree),
                )
                + 1
            )
        )
    )
    return screen_edges, screen_parent_sets


def _number_id(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def method_setting_ids(method):
    """Return setting IDs in the predeclared plotting/tie-break order."""
    if method == "ps_mip_unknown":
        return [f"bic_x{_number_id(x)}" for x in PS_UNKNOWN_BIC_MULTIPLIERS]
    if method == "ps_mip_oracle":
        return [f"g{_number_id(x)}" for x in PS_ORACLE_GRAPH_MULTIPLIERS]
    if method == "dcdi_g_unknown":
        return [setting_id for setting_id, _, _ in DCDI_GRID]
    if method == "utigsp_unknown":
        return [f"alpha_{_number_id(x)}" for x in UTIGSP_ALPHAS]
    if method in {"gnies_unknown", "gies_oracle"}:
        return [f"bic_x{_number_id(x)}" for x in SCORE_BIC_MULTIPLIERS]
    raise ValueError(f"unknown method {method}")


def _method_settings(method, data):
    """Return the explicit tuning settings for one method and instance."""
    if method == "ps_mip_unknown":
        base = _bic_penalty(data)
        return [
            {
                "setting_id": f"bic_x{_number_id(multiplier)}",
                "tuning_parameter": "tied_bic_multiplier",
                "tuning_value": multiplier,
                "graph_penalty": multiplier * base,
                "target_penalty": multiplier * base,
                "graph_bic_multiplier": multiplier,
                "target_bic_multiplier": multiplier,
            }
            for multiplier in PS_UNKNOWN_BIC_MULTIPLIERS
        ]
    if method == "ps_mip_oracle":
        base = _bic_penalty(data)
        return [
            {
                "setting_id": f"g{_number_id(multiplier)}",
                "tuning_parameter": "graph_bic_multiplier",
                "tuning_value": multiplier,
                "graph_penalty": multiplier * base,
                # Fixed targets make this an additive constant, so it is not swept.
                "target_penalty": base,
                "graph_bic_multiplier": multiplier,
                "target_bic_multiplier": 1.0,
            }
            for multiplier in PS_ORACLE_GRAPH_MULTIPLIERS
        ]
    if method == "dcdi_g_unknown":
        return [
            {
                "setting_id": setting_id,
                "tuning_parameter": "graph_penalty,target_penalty",
                "tuning_value": f"{graph_penalty:g},{target_penalty:g}",
                "graph_penalty": graph_penalty,
                "target_penalty": target_penalty,
            }
            for setting_id, graph_penalty, target_penalty in DCDI_GRID
        ]
    if method == "utigsp_unknown":
        return [
            {
                "setting_id": f"alpha_{_number_id(alpha)}",
                "tuning_parameter": "alpha=alpha_inv",
                "tuning_value": alpha,
                "alpha": alpha,
            }
            for alpha in UTIGSP_ALPHAS
        ]
    if method in {"gnies_unknown", "gies_oracle"}:
        base = _score_bic_penalty(data)
        return [
            {
                "setting_id": f"bic_x{_number_id(multiplier)}",
                "tuning_parameter": "bic_multiplier",
                "tuning_value": multiplier,
                "graph_penalty": multiplier * base,
                "bic_multiplier": multiplier,
            }
            for multiplier in SCORE_BIC_MULTIPLIERS
        ]
    raise ValueError(f"unknown method {method}")


def _declared_settings(method, data, setting, args):
    """Return the fixed settings saved before a method is attempted."""
    official_runtime = getattr(args, "official_runtime", {}).get(method, {})
    common = {
        "main_experiment_version": MAIN_EXPERIMENT_VERSION,
        "data_format_version": DATA_FORMAT_VERSION,
        "generator_numpy_version": args.generator_numpy_version,
        "data_source": "persisted_main_experiment_suite",
        "data_selection": "full_instance",
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "validated_python_runtime": getattr(args, "official_runtime", {}).get(
            "shared_python", {}
        ),
        "setting_id": setting["setting_id"],
        "tuning_parameter": setting["tuning_parameter"],
        "tuning_value": setting["tuning_value"],
        # These execution settings are part of checkpoint identity.  A
        # time-limited incumbent is reusable under the same budget, while a
        # larger requested budget intentionally triggers a fresh fit.
        "fit_time_limit": float(args.time_limit),
        "metric_time_limit": float(args.metric_time_limit),
        "threads": int(args.threads),
    }
    if method in {"ps_mip_unknown", "ps_mip_oracle"}:
        config = {
            **common,
            "screen_constant": SCREEN_CONSTANT,
            "screen_alpha": _screen_alpha(data),
            "n_screen": len(data[0]),
            "max_parents": MAX_PARENTS,
            "configuration_limit": CONFIGURATION_LIMIT,
            "graph_penalty": setting["graph_penalty"],
            "target_penalty": setting["target_penalty"],
            "graph_bic_multiplier": setting["graph_bic_multiplier"],
            "target_bic_multiplier": setting["target_bic_multiplier"],
        }
        return (
            "bic_multiplier_logN_over_N",
            setting["graph_penalty"],
            setting["target_penalty"],
            config,
        )
    if method == "dcdi_g_unknown":
        config = {
            **common,
            **OFFICIAL_IMPLEMENTATIONS[method],
            "validated_runtime": official_runtime,
            "implementation": "hash_verified_vendored_author_snapshot",
            "execution": "direct_author_entrypoint_plot_keyword_compatibility",
            "model": "DCDI-G",
            "intervention_type": "perfect",
            "intervention_knowledge": "unknown",
            "graph_penalty": setting["graph_penalty"],
            "target_penalty": setting["target_penalty"],
            "hidden_dim": DCDI_HIDDEN_DIM,
            "mu_init": DCDI_MU_INIT,
            "max_iterations": DCDI_MAX_ITERATIONS,
            "normalize_data": True,
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
            "device": "cpu",
            "precision": "float64",
            "python_runtime": "same_active_interpreter",
            "rscript": str(args.rscript),
        }
        return (
            "compact_reference_penalty_grid",
            setting["graph_penalty"],
            setting["target_penalty"],
            config,
        )
    if method == "utigsp_unknown":
        config = {
            **common,
            **OFFICIAL_IMPLEMENTATIONS[method],
            "validated_runtime": official_runtime,
            "implementation": "official_author_package_api",
            "alpha": setting["alpha"],
            "alpha_inv": setting["alpha"],
            "depth": UTIGSP_DEPTH,
            "nruns": UTIGSP_NRUNS,
        }
        return "alpha_grid", "", "", config
    if method == "gnies_unknown":
        return (
            "bic_multiplier_half_logN", setting["graph_penalty"], "",
            {
                **common,
                **OFFICIAL_IMPLEMENTATIONS[method],
                "validated_runtime": official_runtime,
                "implementation": "official_author_package_api",
                "approach": "greedy",
                "lambda": setting["graph_penalty"],
                "bic_multiplier": setting["bic_multiplier"],
            },
        )
    if method == "gies_oracle":
        return (
            "bic_multiplier_half_logN", setting["graph_penalty"], "",
            {
                **common,
                **OFFICIAL_IMPLEMENTATIONS[method],
                "validated_runtime": official_runtime,
                "implementation": "requested_juangamella_gies_package_api",
                "known_targets": True,
                "lambda": setting["graph_penalty"],
                "bic_multiplier": setting["bic_multiplier"],
            },
        )
    raise ValueError(f"unknown method {method}")


def run_our_method(data, args, setting, known_targets=None):
    """Run PS-MIP; supplying known_targets fixes the existing target indicators."""
    import gurobipy as gp

    from MIP import estimate_moral_graph
    from MIP_profiled import optimization
    from src.utils import interventional_cpdag

    if known_targets is not None:
        known_targets = np.asarray(known_targets)
        expected = (len(data) - 1, data[0].shape[1])
        if known_targets.shape != expected or not np.isin(known_targets, (0, 1)).all():
            raise ValueError(
                "known_targets must be binary with one row per interventional "
                "environment; do not include an observational row"
            )
        known_targets = known_targets.astype(int)

    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", args.threads)
    started = time.perf_counter()
    moral = estimate_moral_graph(data[0], _screen_alpha(data))
    screen_edges, screen_parent_sets = _screen_statistics(moral)
    gamma, fitted_targets, gap, objective, solve_seconds, metadata = optimization(
        data,
        moral,
        setting["graph_penalty"],
        l_delta=setting["target_penalty"],
        target_status=known_targets,
        max_parents=MAX_PARENTS,
        configuration_limit=CONFIGURATION_LIMIT,
        time_limit=args.time_limit,
        return_metadata=True,
    )
    fitted_targets = np.asarray(fitted_targets, dtype=int)
    estimated_dag = (np.abs(gamma) > 1e-6).astype(int)
    np.fill_diagonal(estimated_dag, 0)
    estimated_icpdag = interventional_cpdag(estimated_dag, fitted_targets)
    return {
        "estimated_icpdag": estimated_icpdag,
        "estimated_targets": None if known_targets is not None else fitted_targets,
        "screen_edges": screen_edges,
        "screen_parent_sets": screen_parent_sets,
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
    }


def _run_dcdi(data, args, setting, p, edge_multiplier, replicate, seed):
    from DCDI import optimization
    from src.utils import interventional_cpdag

    if getattr(args, "transient_artifacts", False):
        artifact_dir = None
    else:
        runtime_identity = hashlib.sha256(
            json.dumps(
                args.official_runtime["dcdi_g_unknown"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]
        dcdi_configuration = (
            "official_direct_594d328_"
            f"lock_{DEPENDENCY_LOCK_SHA256[:12]}_runtime_{runtime_identity}_"
            f"lambda_{setting['graph_penalty']:g}_"
            f"lambdaR_{setting['target_penalty']:g}_"
            f"hid_{DCDI_HIDDEN_DIM}_mu_{DCDI_MU_INIT:g}_"
            f"iter_{DCDI_MAX_ITERATIONS}"
        )
        artifact_dir = (
            args.output.parent
            / f"{args.output.stem}_dcdi_artifacts"
            / f"p_{p}_e_{edge_multiplier}_replicate_{replicate:03d}_seed_{seed}"
            / (
                f"nobs_{len(data[0])}_nint_{len(data[1])}_"
                f"data_{data_digest(data)[:12]}"
            )
            / dcdi_configuration
        )
    dag, targets, upstream_seconds, metadata = optimization(
        data,
        graph_penalty=setting["graph_penalty"],
        target_penalty=setting["target_penalty"],
        dcdi_path=args.dcdi_root,
        random_seed=seed,
        hidden_dim=DCDI_HIDDEN_DIM,
        mu_init=DCDI_MU_INIT,
        num_train_iter=DCDI_MAX_ITERATIONS,
        timeout=args.time_limit,
        artifact_dir=artifact_dir,
        rscript=args.rscript,
        dependency_lock_sha256=DEPENDENCY_LOCK_SHA256,
    )
    targets = np.asarray(targets, dtype=int)
    return {
        "estimated_icpdag": interventional_cpdag(dag, targets),
        "estimated_targets": targets,
        # The adapter preserves upstream's original training time when a
        # completed artifact is reused; wall time here would then be near zero.
        "fit_seconds": upstream_seconds,
        "solve_seconds": upstream_seconds,
        "artifact_dir": metadata["artifact_dir"],
        "cached": metadata["cached"],
    }


def _run_utigsp(data, args, setting, seed):
    from UTIGSP import fit
    from src.utils import interventional_cpdag

    started = time.perf_counter()
    with _time_limit(args.time_limit, "UT-IGSP fit"):
        dag, targets = fit(
            data,
            alpha=setting["alpha"],
            alpha_inv=setting["alpha"],
            depth=UTIGSP_DEPTH,
            nruns=UTIGSP_NRUNS,
            seed=seed,
        )
    targets = np.asarray(targets, dtype=int)
    return {
        "estimated_icpdag": interventional_cpdag(dag, targets),
        "estimated_targets": targets,
        "fit_seconds": time.perf_counter() - started,
    }


def _run_gnies(data, args, setting):
    try:
        import gnies
    except ImportError as error:
        raise ImportError("GnIES requires the `gnies` package") from error

    started = time.perf_counter()
    with _time_limit(args.time_limit, "GnIES fit"):
        score, estimated_icpdag, targets = gnies.fit(
            data, approach="greedy", lmbda=setting["graph_penalty"]
        )
    target_union = np.zeros(data[0].shape[1], dtype=int)
    target_union[list(targets)] = 1
    return {
        "estimated_icpdag": (np.abs(estimated_icpdag) > 1e-6).astype(int),
        "estimated_target_union": target_union,
        "fit_seconds": time.perf_counter() - started,
        "objective_value": score,
    }


def _run_gies(data, true_targets, args, setting):
    from GIES import fit

    started = time.perf_counter()
    estimated_icpdag, score = fit(
        data,
        known_targets=true_targets,
        lmbda=setting["graph_penalty"],
        timeout=args.time_limit,
    )
    return {
        "estimated_icpdag": estimated_icpdag,
        "fit_seconds": time.perf_counter() - started,
        "objective_value": score,
    }


def _run_method(
    method, data, true_targets, args, setting, p, edge_multiplier, replicate, seed
):
    # Unknown-target calls receive data only. Truth crosses the method boundary
    # only in the two explicitly named oracle branches.
    if method == "ps_mip_unknown":
        return run_our_method(data, args, setting, known_targets=None)
    if method == "dcdi_g_unknown":
        return _run_dcdi(
            data, args, setting, p, edge_multiplier, replicate, seed
        )
    if method == "utigsp_unknown":
        return _run_utigsp(data, args, setting, seed)
    if method == "gnies_unknown":
        return _run_gnies(data, args, setting)
    if method == "ps_mip_oracle":
        return run_our_method(data, args, setting, known_targets=true_targets)
    if method == "gies_oracle":
        return _run_gies(data, true_targets, args, setting)
    raise ValueError(f"unknown method {method}")


def _base_row(
    p, edge_multiplier, replicate, seed, method, setting, data, dag,
    true_targets, args, data_sha256, instance_sha256,
):
    from data.DataGeneration import edge_probability_for_multiplier

    row = {field: "" for field in FIELDS}
    penalty_rule, graph_penalty, target_penalty, method_config = (
        _declared_settings(method, data, setting, args)
    )
    row.update(
        experiment_version=MAIN_EXPERIMENT_VERSION,
        p=p,
        e=edge_multiplier,
        replicate=replicate,
        seed=seed,
        method=method,
        setting_id=setting["setting_id"],
        tuning_parameter=setting["tuning_parameter"],
        tuning_value=setting["tuning_value"],
        target_setting="known_oracle" if method in ORACLE_METHODS else "unknown",
        num_environments=len(data),
        num_interventional_environments=len(data) - 1,
        n_observational=len(data[0]),
        n_interventional=len(data[1]),
        total_samples=sum(len(values) for values in data),
        target_edge_count=edge_multiplier * p,
        edge_probability=edge_probability_for_multiplier(p, edge_multiplier),
        realized_edges=int(np.asarray(dag).sum()),
        data_sha256=data_sha256,
        instance_sha256=instance_sha256,
        screen_constant=SCREEN_CONSTANT if method.startswith("ps_mip") else "",
        screen_alpha=_screen_alpha(data) if method.startswith("ps_mip") else "",
        penalty_rule=penalty_rule,
        graph_penalty=graph_penalty,
        target_penalty=target_penalty,
        method_config=json.dumps(
            method_config, sort_keys=True, separators=(",", ":")
        ),
        true_targets=_json_array(true_targets),
    )
    return row


def _add_fit_result(row, result, true_targets):
    for field in (
        "screen_edges", "screen_parent_sets",
        "fit_seconds", "solve_seconds", "objective_value", "objective_bound",
        "mip_gap", "solver_status", "solver_status_code", "optimal",
        "solution_count", "node_count", "artifact_dir", "cached",
    ):
        if field in result:
            row[field] = result[field]
    estimated_targets = result.get("estimated_targets")
    estimated_union = result.get("estimated_target_union")
    if estimated_targets is not None:
        estimated_targets = np.asarray(estimated_targets, dtype=int)
        estimated_union = estimated_targets.any(axis=0).astype(int)
        row["estimated_targets"] = _json_array(estimated_targets)
    if estimated_union is not None:
        estimated_union = np.asarray(estimated_union, dtype=int)
        row["estimated_target_union"] = _json_array(estimated_union)

    if row["method"] not in ORACLE_METHODS:
        true_union = np.asarray(true_targets, dtype=int).any(axis=0).astype(int)
        if estimated_targets is not None:
            row["target_hamming"] = int(
                np.abs(estimated_targets - np.asarray(true_targets, dtype=int)).sum()
            )
        if estimated_union is not None:
            row["union_target_error"] = int(np.abs(estimated_union - true_union).sum())


def _add_metrics(row, estimated, truth, metric_time_limit):
    from src.utils import cpdag_distance, equivalence_class_fdp_tdp

    started = time.perf_counter()
    row["d_cpdag"] = cpdag_distance(estimated, truth)
    with _time_limit(metric_time_limit, "equivalence-class metric evaluation"):
        fdp, tdp = equivalence_class_fdp_tdp(estimated, truth)
    row.update(
        fdp=fdp,
        tdp=tdp,
        metric_seconds=time.perf_counter() - started,
    )


def _run(args):
    from src.utils import interventional_cpdag

    manifest = load_main_experiment_manifest(
        args.data_root, master_seed=args.seed
    )
    args.generator_numpy_version = manifest["provenance"]["numpy_version"]
    args.official_runtime = _validate_competitor_runtime(args)
    rows = {_key(row): row for row in _read_rows(args.output)}
    had_errors = False
    for p in args.p_values:
        for edge_multiplier in args.edge_multipliers:
            replicates = (
                [args.replicate]
                if args.replicate is not None
                else range(1, args.num_replicates + 1)
            )
            for replicate in replicates:
                data, true_dag, true_targets, instance_info = (
                    load_main_experiment_instance(
                        args.data_root,
                        p,
                        edge_multiplier,
                        replicate,
                        manifest=manifest,
                        master_seed=args.seed,
                    )
                )
                seed = instance_info["seed"]
                data_sha256 = instance_info["data_sha256"]
                instance_sha256 = instance_info["instance_sha256"]
                if len(data[0]) != args.n_observational or any(
                    len(values) != args.n_interventional for values in data[1:]
                ):
                    raise RuntimeError(
                        "persisted sample sizes do not match the main-experiment design"
                    )
                truth = interventional_cpdag(true_dag, true_targets)
                print(
                    f"instance p={p} e={edge_multiplier} replicate={replicate} "
                    f"seed={seed} edges={int(true_dag.sum())}",
                    flush=True,
                )
                for method in args.methods:
                    settings = _method_settings(method, data)
                    if args.setting_ids is not None:
                        settings = [
                            setting
                            for setting in settings
                            if setting["setting_id"] in args.setting_ids
                        ]
                    if not settings:
                        raise ValueError(
                            f"none of --setting-ids {args.setting_ids} apply to {method}"
                        )
                    for setting in settings:
                        row = _base_row(
                            p, edge_multiplier, replicate, seed, method, setting,
                            data, true_dag, true_targets, args,
                            data_sha256, instance_sha256,
                        )
                        key = _key(row)
                        previous = rows.get(key)
                        if previous is not None and _checkpoint_matches(
                            previous,
                            row,
                            data_sha256,
                            instance_sha256,
                            seed,
                            retry_failures=args.retry_failures,
                        ):
                            if previous.get("status") in FAILURE_STATUSES:
                                had_errors = True
                            print("skip", key, flush=True)
                            continue

                        print("run", key, flush=True)
                        started = time.perf_counter()
                        try:
                            result = _run_method(
                                method, data, true_targets, args, setting, p,
                                edge_multiplier, replicate, seed,
                            )
                            _add_fit_result(row, result, true_targets)
                        except Exception as error:
                            had_errors = True
                            row.update(
                                fit_seconds=time.perf_counter() - started,
                                status="fit_error",
                                error=f"{type(error).__name__}: {error}",
                            )
                        else:
                            metric_started = time.perf_counter()
                            try:
                                _add_metrics(
                                    row,
                                    result["estimated_icpdag"],
                                    truth,
                                    args.metric_time_limit,
                                )
                            except Exception as error:
                                had_errors = True
                                row.update(
                                    metric_seconds=(
                                        time.perf_counter() - metric_started
                                    ),
                                    status="metric_error",
                                    error=f"{type(error).__name__}: {error}",
                                )
                            else:
                                row["status"] = (
                                    "ok_nonoptimal"
                                    if row["optimal"] is False
                                    else "ok"
                                )
                        rows[key] = row
                        _write_rows(args.output, rows.values())
                        print(row["status"], key, flush=True)

    print(f"results: {args.output}", flush=True)
    return 1 if had_errors else 0


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the main unknown-target and oracle synthetic comparison."
    )
    parser.add_argument("--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES)
    parser.add_argument(
        "--replicate",
        type=int,
        help="run one 1-based replicate (useful for an existing job array)",
    )
    parser.add_argument("--p-values", type=int, nargs="+", default=P_VALUES)
    parser.add_argument(
        "--edge-multipliers", type=int, nargs="+", default=EDGE_MULTIPLIERS
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--setting-ids",
        nargs="+",
        help="run only matching setting IDs (normally used with one method)",
    )
    parser.add_argument(
        "--all-settings",
        action="store_true",
        help="in smoke mode, run complete tuning grids instead of one primary setting",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help=(
            "retry matching fit_error/metric_error rows; by default they remain "
            "checkpointed so later settings can progress on resubmission"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--time-limit",
        type=float,
        help="fit limit in seconds (default: 60 for smoke, 3600 otherwise)",
    )
    parser.add_argument(
        "--metric-time-limit",
        type=float,
        help="metric limit in seconds (default: 60 for smoke, 3600 otherwise)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="result CSV (defaults to a separate smoke or full result path)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "persisted main-experiment dataset root "
            f"(default: {DEFAULT_DATA_ROOT})"
        ),
    )
    parser.add_argument(
        "--dcdi-root", type=Path, default=PROJECT_ROOT / "external" / "dcdi"
    )
    parser.add_argument(
        "--rscript",
        default="Rscript",
        help="Rscript executable used only by unmodified DCDI reporting",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run primary settings on persisted p=10, e=1, replicate 1 data",
    )
    args = parser.parse_args()
    if args.num_replicates < 1 or args.threads < 1:
        parser.error("num-replicates and threads must be positive")
    if args.num_replicates > DEFAULT_NUM_REPLICATES:
        parser.error(
            f"--num-replicates cannot exceed the persisted design's "
            f"{DEFAULT_NUM_REPLICATES} replicates"
        )
    if args.replicate is not None and args.replicate < 1:
        parser.error("replicate must be positive")
    if args.replicate is not None and args.replicate > DEFAULT_NUM_REPLICATES:
        parser.error(
            f"--replicate cannot exceed {DEFAULT_NUM_REPLICATES}"
        )
    if not 0 <= args.seed <= MAX_MASTER_SEED or any(
        p < 2 for p in args.p_values
    ):
        parser.error(
            f"seed must be between 0 and {MAX_MASTER_SEED}, and p-values "
            "must be at least 2"
        )
    if any(e < 0 for e in args.edge_multipliers):
        parser.error("edge multipliers must be nonnegative")
    if any(p not in P_VALUES for p in args.p_values):
        parser.error(f"p-values must come from the persisted grid {P_VALUES}")
    if any(e not in EDGE_MULTIPLIERS for e in args.edge_multipliers):
        parser.error(
            f"edge multipliers must come from the persisted grid {EDGE_MULTIPLIERS}"
        )
    if any(
        value is not None and value <= 0
        for value in (args.time_limit, args.metric_time_limit)
    ):
        parser.error("time limits must be positive")
    if args.all_settings and args.setting_ids is not None:
        parser.error("--all-settings and --setting-ids cannot be combined")

    if args.output is None:
        if args.smoke_test:
            filename = "smoke_grid.csv"
        elif args.replicate is not None:
            parts = [f"replicate_{args.replicate:03d}"]
            if len(args.p_values) == 1:
                parts.append(f"p_{args.p_values[0]}")
            if len(args.edge_multipliers) == 1:
                parts.append(f"e_{args.edge_multipliers[0]}")
            if len(args.methods) == 1:
                parts.append(args.methods[0])
            if args.setting_ids is not None and len(args.setting_ids) == 1:
                parts.append(args.setting_ids[0])
            filename = "_".join(parts) + ".csv"
        else:
            filename = "results_grid.csv"
        args.output = PROJECT_ROOT / "experiment_results" / "main_experiment" / filename
    args.output = args.output.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.dcdi_root = args.dcdi_root.expanduser().resolve()
    args.n_observational = N_OBSERVATIONAL
    args.n_interventional = N_INTERVENTIONAL
    if args.smoke_test:
        args.p_values = [10]
        args.edge_multipliers = [1]
        args.num_replicates = 1
        args.replicate = 1
        if args.setting_ids is None and not args.all_settings:
            args.setting_ids = sorted(
                {PRIMARY_SETTING_IDS[method] for method in args.methods}
            )
        if args.time_limit is None:
            args.time_limit = SMOKE_TIME_LIMIT
        if args.metric_time_limit is None:
            args.metric_time_limit = SMOKE_TIME_LIMIT
    else:
        if args.time_limit is None:
            args.time_limit = DEFAULT_TIME_LIMIT
        if args.metric_time_limit is None:
            args.metric_time_limit = DEFAULT_TIME_LIMIT
    return args


def main():
    args = _parse_args()
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    try:
        with _result_lock(lock_path):
            return _run(args)
    except MainExperimentDataError as error:
        print(f"main experiment data error: {error}", file=sys.stderr)
        return 2
    except OfficialRuntimeError as error:
        print(f"official baseline setup error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
