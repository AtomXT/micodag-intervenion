#!/usr/bin/env python3
"""Validate Sachs data and method APIs without fitting or saving results."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import numpy as np


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_sachs_roc as experiment
from src.sachs_data import DEFAULT_DATA_ROOT, SACHS_NODE_NAMES, load_sachs_data

NODE_NAMES = SACHS_NODE_NAMES


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required package {distribution!r} is not installed") from error


def _check_method_api(method: str) -> str:
    if method.startswith("ps_mip"):
        import gurobipy
        from MIP_profiled import optimization

        if not callable(optimization):
            raise RuntimeError("MIP_profiled.optimization is not callable")
        return f"gurobipy {_version('gurobipy')}"
    if method in {"utigsp_unknown", "utigsp_intended", "igsp_oracle"}:
        import causaldag
        from IGSP import fit as igsp_fit
        from UTIGSP import fit as utigsp_fit

        required = (
            "MemoizedCI_Tester",
            "MemoizedInvarianceTester",
            "gauss_invariance_suffstat",
            "gauss_invariance_test",
            "partial_correlation_suffstat",
            "partial_correlation_test",
        )
        missing = [name for name in required if not hasattr(causaldag, name)]
        if missing or not callable(utigsp_fit) or not callable(igsp_fit):
            raise RuntimeError(f"incompatible causaldag API; missing {missing}")
        return f"causaldag {_version('causaldag')}"
    if method == "gnies_unknown":
        import gnies
        from gnies.utils import pdag_to_dag

        if not callable(gnies.fit) or not callable(pdag_to_dag):
            raise RuntimeError("incompatible gnies API")
        return f"gnies {_version('gnies')}"
    if method == "gies_oracle":
        import gies
        from gies.scores import GaussIntL0Pen
        from gies.utils import pdag_to_dag

        if not all(callable(value) for value in (gies.fit, GaussIntL0Pen, pdag_to_dag)):
            raise RuntimeError("incompatible gies API")
        return f"gies {_version('gies')}"
    raise RuntimeError(f"unknown method {method}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the frozen Sachs benchmark and selected method APIs."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--methods", nargs="+", choices=experiment.METHODS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    methods = list(experiment.METHODS if args.methods is None else args.methods)
    try:
        data, true_dag, intended_targets, info = load_sachs_data(
            args.data_root.expanduser().resolve()
        )
        if sum(len(values) for values in data) != 5846:
            raise RuntimeError("validated archive does not total 5,846 samples")
        if int(np.asarray(true_dag).sum()) != 18:
            raise RuntimeError("validated accepted DAG does not contain 18 arcs")
        if np.asarray(intended_targets).shape != (5, len(NODE_NAMES)):
            raise RuntimeError("validated intended-target matrix has the wrong shape")
        if experiment.COMPLETE_PARENT_SETS != 11264:
            raise RuntimeError("complete-superstructure parent-set count drifted")

        print("Sachs data: OK")
        print("  source commit:", info["source_commit"])
        print("  data SHA-256:", info["data_sha256"])
        print("  node order:", ", ".join(NODE_NAMES))
        print("  sample sizes:", [len(values) for values in data])
        print("  accepted arcs:", int(np.asarray(true_dag).sum()))
        print("  complete PS-MIP screen: 55 pairs / 11,264 parent sets")

        checked = {}
        for method in methods:
            runtime = _check_method_api(method)
            checked.setdefault(runtime, []).append(method)
            settings = experiment.method_settings(method, data)
            print(
                f"{method}: OK — {len(settings)} settings "
                f"({settings[0]['setting_id']} … {settings[-1]['setting_id']})"
            )
        print("No fits were run and no result files were written.")
    except Exception as error:
        print(f"setup check failed: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            "If the data are missing, run `python3 experiments/prepare_sachs_data.py` "
            "from a login node first.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
