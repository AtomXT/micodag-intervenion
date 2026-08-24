#!/usr/bin/env python3
"""Thin adapter for the requested ``juangamella/gies`` Python package.

This module contains no graph search, score, or essential-graph algorithm. It
validates the project's arrays, translates the known intervention targets to
the package convention, calls the package, and validates the returned graph.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.main_experiment_cli import (
    add_main_data_arguments,
    instance_summary,
    load_cli_instance,
    primary_score_penalty,
    wall_time_limit,
)
from src.main_experiment_data import MainExperimentDataError


GIES_SOURCE_URL = "https://github.com/juangamella/gies"
GIES_REFERENCE_VERSION = "0.0.3"


class GIESError(RuntimeError):
    """Raised when the requested Python GIES package cannot return a result."""


def _load_gies_api():
    """Import the package lazily so data-only project utilities remain usable."""
    try:
        import gies
        from gies.scores import GaussIntL0Pen
    except ImportError as error:
        raise GIESError(
            "GIES requires the Python package from "
            f"{GIES_SOURCE_URL}; install it with `python3 -m pip install gies`"
        ) from error
    if not callable(getattr(gies, "fit", None)) or not callable(
        getattr(gies, "fit_bic", None)
    ):
        raise GIESError("the installed gies package does not expose fit and fit_bic")
    return gies, GaussIntL0Pen


def validate_runtime() -> dict[str, Any]:
    """Validate the requested package API and record, but do not enforce, its version."""
    gies, score_class = _load_gies_api()
    try:
        version = importlib.metadata.version("gies")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "package": "gies",
        "version": version,
        "module_path": str(Path(gies.__file__).resolve()),
        "fit_api": f"{gies.fit.__module__}.{gies.fit.__name__}",
        "score_api": f"{score_class.__module__}.{score_class.__name__}",
        "source_url": GIES_SOURCE_URL,
    }


def _arrays_and_targets(
    data: Sequence[Any], known_targets: Any
) -> tuple[list[np.ndarray], np.ndarray]:
    arrays = [
        np.ascontiguousarray(
            np.asarray(x.values if hasattr(x, "values") else x, dtype=float)
        )
        for x in data
    ]
    if len(arrays) < 2 or any(x.ndim != 2 or len(x) == 0 for x in arrays):
        raise ValueError(
            "data must contain observational data followed by interventional samples"
        )
    p = arrays[0].shape[1]
    if p == 0 or any(x.shape[1] != p for x in arrays):
        raise ValueError("all environments must contain the same ordered variables")
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("data must contain only finite values")

    targets = np.asarray(known_targets)
    expected = (len(arrays) - 1, p)
    if targets.shape != expected or not np.isin(targets, (0, 1)).all():
        raise ValueError(
            "known_targets must be binary with shape (len(data) - 1, p); "
            "do not include an observational row"
        )
    return arrays, targets.astype(int)


def _intervention_family(targets: np.ndarray) -> list[list[int]]:
    """Return the package's zero-based target list, including the obs environment."""
    return [[]] + [np.flatnonzero(row).tolist() for row in targets]


def fit(
    data: Sequence[Any],
    known_targets: Any,
    *,
    lmbda: float | None = None,
    timeout: float | None = None,
) -> tuple[np.ndarray, float]:
    """Run the requested Python GIES implementation with known targets.

    ``data[0]`` is observational and ``known_targets`` omits that observational
    row. ``lmbda=None`` calls ``gies.fit_bic`` directly. An explicit penalty is
    passed to the package's ``GaussIntL0Pen`` score before calling ``gies.fit``.
    """
    arrays, targets = _arrays_and_targets(data, known_targets)
    if lmbda is not None and (not np.isfinite(lmbda) or lmbda <= 0):
        raise ValueError("lmbda must be positive and finite")
    if timeout is not None and (not np.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive when supplied")

    gies, score_class = _load_gies_api()
    interventions = _intervention_family(targets)

    def run_package():
        if lmbda is None:
            return gies.fit_bic(arrays, interventions)
        score = score_class(arrays, interventions, lmbda=float(lmbda))
        return gies.fit(score)

    try:
        if timeout is None:
            estimated, score = run_package()
        else:
            with wall_time_limit(timeout, "GIES fit"):
                estimated, score = run_package()
    except TimeoutError:
        raise
    except Exception as error:
        raise GIESError(f"the requested Python GIES fit failed: {error}") from error

    estimated = np.asarray(estimated, dtype=float)
    p = arrays[0].shape[1]
    if estimated.shape != (p, p) or not np.isfinite(estimated).all():
        raise GIESError(
            f"gies returned an invalid I-CPDAG shape {estimated.shape}"
        )
    if not np.all(np.isclose(estimated, 0.0) | np.isclose(estimated, 1.0)):
        raise GIESError("gies returned a nonbinary I-CPDAG")
    if np.any(np.diag(np.abs(estimated) > 1e-8)):
        raise GIESError("gies returned an I-CPDAG with self-loops")
    try:
        score = float(score)
    except (TypeError, ValueError) as error:
        raise GIESError("gies returned a nonnumeric score") from error
    if not np.isfinite(score):
        raise GIESError("gies returned a nonfinite score")
    return (np.abs(estimated) > 1e-8).astype(int), score


optimization = fit


def main() -> int:
    """Run the requested Python GIES oracle on one persisted main instance."""
    parser = argparse.ArgumentParser(
        description="Run the juangamella/gies oracle with known targets."
    )
    add_main_data_arguments(parser)
    parser.add_argument("--bic-multiplier", type=float, default=1.0)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--metric-time-limit", type=float, default=3600)
    args = parser.parse_args()
    if (
        not np.isfinite(
            [args.bic_multiplier, args.time_limit, args.metric_time_limit]
        ).all()
        or args.bic_multiplier <= 0
        or args.time_limit <= 0
        or args.metric_time_limit <= 0
    ):
        parser.error("BIC multiplier and time limits must be positive and finite")
    try:
        data, true_dag, true_targets, info = load_cli_instance(args)
    except MainExperimentDataError as error:
        parser.error(str(error))

    from src.utils import (
        cpdag_distance,
        equivalence_class_fdp_tdp,
        interventional_cpdag,
    )

    penalty = args.bic_multiplier * primary_score_penalty(data)
    started = time.perf_counter()
    estimated, score = fit(
        data,
        true_targets,
        lmbda=penalty,
        timeout=args.time_limit,
    )
    runtime = time.perf_counter() - started
    with wall_time_limit(args.metric_time_limit, "metric evaluation"):
        truth = interventional_cpdag(true_dag, true_targets)
        d_cpdag = cpdag_distance(estimated, truth)
        fdp, tdp = equivalence_class_fdp_tdp(estimated, truth)

    print("Instance:", instance_summary(args, data, true_dag, true_targets, info))
    print(
        "Configuration:",
        {"target_setting": "known_oracle", "bic_multiplier": args.bic_multiplier,
         "lambda": penalty, "fit_time_limit": args.time_limit,
         "metric_time_limit": args.metric_time_limit},
    )
    print("I-CPDAG:\n", estimated)
    print("Known targets:\n", true_targets)
    print("Score:", score)
    print("Runtime:", runtime)
    print("d_cpdag, FDP, TDP:", (d_cpdag, fdp, tdp))
    return 0


__all__ = [
    "GIESError",
    "GIES_REFERENCE_VERSION",
    "GIES_SOURCE_URL",
    "fit",
    "main",
    "optimization",
    "validate_runtime",
]


if __name__ == "__main__":
    raise SystemExit(main())
