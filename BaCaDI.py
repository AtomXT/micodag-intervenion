#!/usr/bin/env python3
"""Adapter for the official BaCaDI unknown-intervention implementation.

The author package is vendored unchanged under ``external/bacadi`` at the
pinned AISTATS 2023 source commit.  This adapter supplies compatibility aliases
for the tested modern JAX runtime, standardizes variables using observational
data only, and converts BaCaDI's particle posterior to one coherent point
estimate by selecting the highest-weight acyclic joint particle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

from src.main_experiment_cli import (
    add_main_data_arguments,
    instance_summary,
    load_cli_instance,
    selected_method_seed,
    wall_time_limit,
)
from src.main_experiment_data import MainExperimentDataError


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BACADI_PATH = PROJECT_ROOT / "external" / "bacadi"
BACADI_SOURCE_URL = "https://github.com/haeggee/bacadi"
BACADI_COMMIT = "b2c27d5fc560f7e0ec2aa282cd2e0a0e6e8637ea"
BACADI_PYTHON_TREE_SHA256 = (
    "6d54e25cf1667375011b02c0d345278154a6d81eb312a74c20287a4dbf409855"
)
BACADI_VENDOR_MANIFEST = "VENDORED_SOURCE.json"
JAX_COMPATIBILITY_VERSION = "jax-0.4-author-api-aliases-v1"

DEFAULT_N_STEPS = 3_000
DEFAULT_N_PARTICLES = 20
DEFAULT_GRAPH_PRIOR_EDGES_PER_NODE = 2
DEFAULT_N_GRAD_MC_SAMPLES = 128
DEFAULT_N_ACYCLICITY_MC_SAMPLES = 32
DEFAULT_TARGET_REGULARIZATION = 1.0
DEFAULT_OBSERVATION_NOISE = 1.0
DEFAULT_INTERVENTION_NOISE = 1.0
DEFAULT_OPTIMIZER_STEP_SIZE = 0.005
DEFAULT_KERNEL = "frob-joint-interv-add"
DEFAULT_GRAPH_PRIOR = "er"
DEFAULT_MODEL_PRIOR = "lingauss"
DEFAULT_ALPHA_LINEAR = 0.01
DEFAULT_BETA_LINEAR = 1.0
DEFAULT_TAU = 1.0
DEFAULT_H_LATENT = 5.0
DEFAULT_H_THETA = 500.0
DEFAULT_H_INTERVENTION = 5.0
DEFAULT_INTERVENTIONS_PER_ENVIRONMENT = 0
DEFAULT_GRADIENT_ESTIMATOR = "reparam"
DEFAULT_SCORE_FUNCTION_BASELINE = 0.0
DEFAULT_EDGE_MEAN = 0.0
DEFAULT_EDGE_SD = 1.0
DEFAULT_INITIAL_EDGE_SD = 0.3
DEFAULT_INTERVENTION_MEAN = 0.0
DEFAULT_INTERVENTION_PRIOR_MEAN = 0.0
DEFAULT_INTERVENTION_PRIOR_SD = 10.0


class BaCaDIError(RuntimeError):
    """Raised when the pinned BaCaDI implementation cannot return an estimate."""


def _python_tree_digest(checkout: Path) -> tuple[str, int]:
    """Hash the path and bytes of every vendored upstream Python file."""
    files = sorted(
        (path for path in (checkout / "bacadi").rglob("*.py")),
        key=lambda path: path.relative_to(checkout).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(checkout).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest(), len(files)


def _validate_checkout(path=DEFAULT_BACADI_PATH) -> Path:
    """Validate the minimal author snapshot and its pinned provenance."""
    checkout = Path(path).expanduser().resolve()
    manifest_path = checkout / BACADI_VENDOR_MANIFEST
    required = [
        manifest_path,
        checkout / "LICENSE",
        checkout / "bacadi" / "inference" / "bacadi_joint.py",
        checkout / "bacadi" / "models" / "linearGaussian.py",
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            "vendored BaCaDI source is incomplete; restore external/bacadi "
            f"from Git (missing {missing[:3]})"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaCaDIError("cannot read the vendored BaCaDI provenance manifest") from error
    if (
        manifest.get("format_version") != 1
        or manifest.get("source_url") != f"{BACADI_SOURCE_URL}.git"
        or manifest.get("source_commit") != BACADI_COMMIT
        or manifest.get("python_tree_sha256") != BACADI_PYTHON_TREE_SHA256
    ):
        raise BaCaDIError("vendored BaCaDI provenance does not match the pinned source")
    observed_digest, observed_count = _python_tree_digest(checkout)
    if observed_digest != BACADI_PYTHON_TREE_SHA256 or observed_count != int(
        manifest.get("python_file_count", -1)
    ):
        raise BaCaDIError(
            "vendored BaCaDI Python source differs from commit " + BACADI_COMMIT
        )
    return checkout


class _JaxIndex:
    """Legacy ``jax.ops.index`` object backed by normal indexing syntax."""

    def __getitem__(self, item):
        return item


def _install_jax_compatibility():
    """Install only the aliases removed since the authors' JAX release."""
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    try:
        import jax
        import jax.experimental
        import jax.ops
        import jax.tree_util
        from jax.example_libraries import optimizers, stax
    except Exception as error:
        raise ImportError(
            "BaCaDI requires JAX, jaxlib, python-igraph, and tqdm; install "
            "requirements-bacadi.txt in the selected Quest environment"
        ) from error

    jax.config.update("jax_enable_x64", True)
    if not hasattr(jax.tree_util, "tree_multimap"):
        jax.tree_util.tree_multimap = jax.tree_util.tree_map

    sys.modules.setdefault("jax.experimental.optimizers", optimizers)
    sys.modules.setdefault("jax.experimental.stax", stax)
    if not hasattr(jax.experimental, "optimizers"):
        jax.experimental.optimizers = optimizers
    if not hasattr(jax.experimental, "stax"):
        jax.experimental.stax = stax

    if not hasattr(jax.ops, "index"):
        jax.ops.index = _JaxIndex()
    if not hasattr(jax.ops, "index_update"):
        jax.ops.index_update = lambda array, index, values: array.at[index].set(values)
    if not hasattr(jax.ops, "index_mul"):
        jax.ops.index_mul = lambda array, index, values: array.at[index].multiply(values)

    if "jax.interpreters.masking" not in sys.modules:
        masking = types.ModuleType("jax.interpreters.masking")
        masking.ShapeError = type("ShapeError", (ValueError,), {})
        sys.modules["jax.interpreters.masking"] = masking

    # The direct adapter does not use the optional SERGIO generator.  Its
    # import is nevertheless unconditional in the author factory module.
    if "sergio.sergio_sampler" not in sys.modules:
        sergio_package = types.ModuleType("sergio")
        sergio_module = types.ModuleType("sergio.sergio_sampler")

        def _unavailable_sergio(*_args, **_kwargs):
            raise BaCaDIError("SERGIO is not included in the BaCaDI point adapter")

        sergio_module.sergio_clean = _unavailable_sergio
        sergio_package.sergio_sampler = sergio_module
        sys.modules.setdefault("sergio", sergio_package)
        sys.modules["sergio.sergio_sampler"] = sergio_module
    return jax


def _load_bacadi_joint(path=DEFAULT_BACADI_PATH):
    """Load the pinned author class after installing runtime-only aliases."""
    checkout = _validate_checkout(path)
    jax = _install_jax_compatibility()
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    loaded = sys.modules.get("bacadi")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if checkout not in loaded_path.parents:
            raise BaCaDIError(
                f"a different bacadi package is already imported from {loaded_path}"
            )
    previous_cwd = Path.cwd()
    try:
        # Upstream's factory import creates a relative `store` directory. Keep
        # that import-only side effect outside the repository.
        with tempfile.TemporaryDirectory(prefix="bacadi-import-") as directory:
            os.chdir(directory)
            from bacadi.inference.bacadi_joint import BaCaDIJoint
    finally:
        os.chdir(previous_cwd)
    return jax, BaCaDIJoint


def validate_runtime(path=DEFAULT_BACADI_PATH) -> dict:
    """Import the exact author API and record the runtime used for a fit."""
    checkout = _validate_checkout(path)
    jax, model_class = _load_bacadi_joint(checkout)
    distributions = {}
    for name in ("jax", "jaxlib", "python-igraph", "tqdm"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise BaCaDIError(f"required BaCaDI distribution is missing: {name}") from error
    return {
        "source_url": BACADI_SOURCE_URL,
        "source_commit": BACADI_COMMIT,
        "source_tree_sha256": BACADI_PYTHON_TREE_SHA256,
        "source_path": str(checkout),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "jax_backend": jax.default_backend(),
        "distributions": distributions,
        "compatibility": JAX_COMPATIBILITY_VERSION,
        "entrypoint": f"{model_class.__module__}.{model_class.__name__}",
    }


def _as_arrays(data):
    """Validate observational-first environment matrices."""
    arrays = []
    try:
        datasets = list(data)
    except TypeError as error:
        raise ValueError("data must be an iterable of environment matrices") from error
    if len(datasets) < 2:
        raise ValueError("data must include observational and interventional samples")
    for environment, values in enumerate(datasets):
        values = values.values if hasattr(values, "values") else values
        try:
            array = np.ascontiguousarray(np.asarray(values, dtype=float))
        except (TypeError, ValueError) as error:
            raise ValueError(f"environment {environment} must be numeric") from error
        if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
            raise ValueError(f"environment {environment} must be a nonempty matrix")
        if not np.isfinite(array).all():
            raise ValueError(f"environment {environment} contains nonfinite values")
        arrays.append(array)
    p = arrays[0].shape[1]
    if any(array.shape[1] != p for array in arrays):
        raise ValueError("all environments must use the same ordered variables")
    return arrays


def _standardized_stack(arrays):
    """Use observational moments only and return stacked data and environment IDs."""
    center = arrays[0].mean(axis=0)
    scale = arrays[0].std(axis=0, ddof=0)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("observational variables must have finite positive variance")
    standardized = [np.ascontiguousarray((array - center) / scale) for array in arrays]
    stacked = np.vstack(standardized)
    environments = np.concatenate(
        [np.full(len(array), environment, dtype=np.int32) for environment, array in enumerate(arrays)]
    )
    return stacked, environments, center, scale


def _is_dag(adjacency) -> bool:
    """Return whether a binary source-by-target adjacency is acyclic."""
    graph = np.asarray(adjacency, dtype=int)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        return False
    indegree = graph.sum(axis=0).astype(int)
    stack = list(np.flatnonzero(indegree == 0))
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        for child in np.flatnonzero(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                stack.append(int(child))
    return visited == len(graph)


def _decode_graph_ids(graph_ids, p):
    """Decode upstream's little-endian packbits adjacency representation."""
    packed = np.asarray(graph_ids, dtype=np.uint8)
    if packed.ndim != 2:
        raise BaCaDIError("BaCaDI returned malformed graph particle IDs")
    bits = np.unpackbits(packed, axis=1, bitorder="little")
    if bits.shape[1] < p * p:
        raise BaCaDIError("BaCaDI graph particle IDs are too short")
    graphs = bits[:, : p * p].reshape((-1, p, p)).astype(int)
    if np.any(np.diagonal(graphs, axis1=1, axis2=2)):
        raise BaCaDIError("BaCaDI returned a graph particle with a self-loop")
    return graphs


def _select_highest_weight_acyclic(distribution, p, environments):
    """Select one joint graph/target particle without using simulation truth."""
    if not isinstance(distribution, tuple) or len(distribution) < 3:
        raise BaCaDIError("BaCaDI returned an unexpected particle distribution")
    graphs = _decode_graph_ids(distribution[0], p)
    raw_targets = np.asarray(distribution[-2])
    log_weights = np.asarray(distribution[-1], dtype=float)
    expected_targets = (len(graphs), environments, p)
    if (
        raw_targets.shape != expected_targets
        or not np.isfinite(raw_targets).all()
        or not np.isin(raw_targets, (0, 1)).all()
    ):
        raise BaCaDIError(
            "BaCaDI target particles must be finite and binary with shape "
            f"{expected_targets}; observed {raw_targets.shape}"
        )
    targets = raw_targets.astype(int)
    if log_weights.shape != (len(graphs),) or np.isnan(log_weights).any():
        raise BaCaDIError("BaCaDI returned malformed particle weights")
    acyclic = np.asarray([_is_dag(graph) for graph in graphs], dtype=bool)
    candidates = np.flatnonzero(acyclic & np.isfinite(log_weights))
    if not len(candidates):
        raise BaCaDIError("BaCaDI returned no finite-weight acyclic particle")
    selected = int(candidates[np.argmax(log_weights[candidates])])
    return graphs[selected], targets[selected], {
        "posterior_particles": int(len(graphs)),
        "acyclic_particles": int(acyclic.sum()),
        "selected_particle": selected,
        "selected_log_weight": float(log_weights[selected]),
        "point_estimate": "highest_weight_acyclic_joint_particle",
    }


def fit(
    data,
    *,
    seed=0,
    source_path=DEFAULT_BACADI_PATH,
    n_steps=DEFAULT_N_STEPS,
    n_particles=DEFAULT_N_PARTICLES,
    graph_prior_edges_per_node=DEFAULT_GRAPH_PRIOR_EDGES_PER_NODE,
    n_grad_mc_samples=DEFAULT_N_GRAD_MC_SAMPLES,
    n_acyclicity_mc_samples=DEFAULT_N_ACYCLICITY_MC_SAMPLES,
    target_regularization=DEFAULT_TARGET_REGULARIZATION,
):
    """Fit main-paper joint linear-Gaussian BaCaDI with unknown targets."""
    arrays = _as_arrays(data)
    integers = {
        "seed": seed,
        "n_steps": n_steps,
        "n_particles": n_particles,
        "graph_prior_edges_per_node": graph_prior_edges_per_node,
        "n_grad_mc_samples": n_grad_mc_samples,
        "n_acyclicity_mc_samples": n_acyclicity_mc_samples,
    }
    for name, value in integers.items():
        if (
            isinstance(value, (bool, np.bool_))
            or int(value) != value
            or int(value) < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer")
        integers[name] = int(value)
    for name in (
        "n_steps",
        "n_particles",
        "n_grad_mc_samples",
        "n_acyclicity_mc_samples",
    ):
        if integers[name] == 0:
            raise ValueError(f"{name} must be positive")
    if integers["graph_prior_edges_per_node"] <= 0:
        raise ValueError("graph_prior_edges_per_node must be positive")
    target_regularization = float(target_regularization)
    if not np.isfinite(target_regularization) or target_regularization < 0:
        raise ValueError("target_regularization must be finite and nonnegative")

    stacked, environment_ids, center, scale = _standardized_stack(arrays)
    jax, model_class = _load_bacadi_joint(source_path)
    model = model_class(
        random_state=jax.random.PRNGKey(integers["seed"]),
        model_param={
            "obs_noise": DEFAULT_OBSERVATION_NOISE,
            "mean_edge": DEFAULT_EDGE_MEAN,
            "sig_edge": DEFAULT_EDGE_SD,
            "init_sig_edge": DEFAULT_INITIAL_EDGE_SD,
            "interv_mean": DEFAULT_INTERVENTION_MEAN,
            "interv_noise": DEFAULT_INTERVENTION_NOISE,
            "interv_prior_mean": DEFAULT_INTERVENTION_PRIOR_MEAN,
            "interv_prior_std": DEFAULT_INTERVENTION_PRIOR_SD,
        },
        kernel=DEFAULT_KERNEL,
        graph_prior=DEFAULT_GRAPH_PRIOR,
        edges_per_node=integers["graph_prior_edges_per_node"],
        model_prior=DEFAULT_MODEL_PRIOR,
        alpha_linear=DEFAULT_ALPHA_LINEAR,
        beta_linear=DEFAULT_BETA_LINEAR,
        tau=DEFAULT_TAU,
        h_latent=DEFAULT_H_LATENT,
        h_theta=DEFAULT_H_THETA,
        h_interv=DEFAULT_H_INTERVENTION,
        interv_per_env=DEFAULT_INTERVENTIONS_PER_ENVIRONMENT,
        lambda_regul=target_regularization,
        optimizer={"name": "rmsprop", "stepsize": DEFAULT_OPTIMIZER_STEP_SIZE},
        n_grad_mc_samples=integers["n_grad_mc_samples"],
        n_acyclicity_mc_samples=integers["n_acyclicity_mc_samples"],
        grad_estimator_z=DEFAULT_GRADIENT_ESTIMATOR,
        score_function_baseline=DEFAULT_SCORE_FUNCTION_BASELINE,
        n_steps=integers["n_steps"],
        n_particles=integers["n_particles"],
        callback_every=0,
        callback=None,
        verbose=False,
    )
    started = time.perf_counter()
    model.fit(jax.numpy.asarray(stacked), jax.numpy.asarray(environment_ids))
    distribution = model.particle_mixture()
    dag, targets, metadata = _select_highest_weight_acyclic(
        distribution, stacked.shape[1], len(arrays) - 1
    )
    metadata.update(
        fit_seconds=time.perf_counter() - started,
        standardized_with="observational_mean_and_population_sd",
        observational_center=center.tolist(),
        observational_scale=scale.tolist(),
    )
    return dag, targets, metadata


def main():
    """Run BaCaDI on one persisted main-experiment instance."""
    parser = argparse.ArgumentParser(
        description="Run joint linear-Gaussian BaCaDI with unknown targets."
    )
    add_main_data_arguments(parser, method_seed=True)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_BACADI_PATH)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--n-particles", type=int, default=DEFAULT_N_PARTICLES)
    parser.add_argument("--time-limit", type=float, default=172800)
    parser.add_argument("--metric-time-limit", type=float, default=3600)
    args = parser.parse_args()
    if not np.isfinite([args.time_limit, args.metric_time_limit]).all() or min(
        args.time_limit, args.metric_time_limit
    ) <= 0:
        parser.error("time limits must be positive and finite")
    try:
        data, true_dag, true_targets, info = load_cli_instance(args)
    except MainExperimentDataError as error:
        parser.error(str(error))
    method_seed = selected_method_seed(args, info)
    from src.utils import compute_errors

    try:
        with wall_time_limit(args.time_limit, "BaCaDI fit"):
            dag, targets, metadata = fit(
                data,
                seed=method_seed,
                source_path=args.source_path,
                n_steps=args.n_steps,
                n_particles=args.n_particles,
            )
        with wall_time_limit(args.metric_time_limit, "metric evaluation"):
            d_cpdag, target_error, fdp, tdp = compute_errors(
                dag, targets, true_dag, true_targets
            )
    except (BaCaDIError, ImportError, ValueError, TimeoutError) as error:
        parser.error(str(error))
    print("instance:", instance_summary(args, info))
    print("source:", f"{BACADI_SOURCE_URL}@{BACADI_COMMIT}")
    print("estimated DAG:\n", dag)
    print("estimated targets:\n", targets)
    print("selection:", metadata)
    print(
        "metrics:",
        {"d_cpdag": d_cpdag, "target_hamming": target_error, "FDP": fdp, "TDP": tdp},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACADI_COMMIT",
    "BACADI_SOURCE_URL",
    "BaCaDIError",
    "DEFAULT_BACADI_PATH",
    "fit",
    "validate_runtime",
]
