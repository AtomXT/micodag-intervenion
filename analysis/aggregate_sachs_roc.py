#!/usr/bin/env python3
"""Validate Sachs experiment fragments and compute ROC operating points.

The aggregation deliberately reports the operating points produced by the
pre-registered tuning grids.  It does not add endpoints, take a monotone
envelope, or calculate an AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_sachs_roc as runner
from src.sachs_data import DEFAULT_DATA_ROOT, SACHS_NODE_NAMES, load_sachs_data


DEFAULT_INPUT = PROJECT_ROOT / "experiment_results" / "sachs" / "parts"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiment_results" / "sachs" / "summary"

REQUIRED_FIELDS = tuple(runner.RESULT_FIELDS)

KEY_FIELDS = ("method", "setting_index")
N_NODES = 11
DIRECTED_POSITIVES = 18
DIRECTED_NEGATIVES = 92
SKELETON_POSITIVES = 18
SKELETON_NEGATIVES = 37
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CONTEXT_TARGET_METHODS = {
    "ps_mip_unknown",
    "utigsp_unknown",
    "utigsp_intended",
    "ps_mip_oracle",
    "igsp_oracle",
    "gies_oracle",
}
ORACLE_TARGET_METHODS = {"ps_mip_oracle", "igsp_oracle", "gies_oracle"}
GRAPH_REPRESENTATIONS = {
    "ps_mip_unknown": "dag_and_i_cpdag",
    "utigsp_unknown": "dag_and_i_cpdag",
    "utigsp_intended": "dag_and_i_cpdag",
    "ps_mip_oracle": "dag_and_i_cpdag",
    "igsp_oracle": "dag_and_i_cpdag",
    "gnies_unknown": "i_cpdag_and_deterministic_consistent_extension",
    "gies_oracle": "i_cpdag_and_deterministic_consistent_extension",
}


class ValidationError(RuntimeError):
    """Raised after a machine-readable validation report has been written."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _normalise_methods(methods: Optional[Sequence[str]]) -> List[str]:
    available = list(runner.METHODS)
    if methods is None:
        return available
    requested: List[str] = []
    for value in methods:
        requested.extend(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            f"unknown method(s): {unknown}; expected a subset of {available}"
        )
    if not requested:
        raise ValueError("at least one method must be selected")
    requested_set = set(requested)
    return [method for method in available if method in requested_set]


def _result_paths(inputs: Optional[Sequence[Path]]) -> List[Path]:
    roots = [DEFAULT_INPUT] if not inputs else list(inputs)
    paths = set()
    for root in roots:
        path = Path(root).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() == ".csv":
                paths.add(path)
        elif path.is_dir():
            paths.update(candidate.resolve() for candidate in path.rglob("*.csv"))
    return sorted(paths)


def _read_results(paths: Sequence[Path]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    frames: List[pd.DataFrame] = []
    file_errors: List[Dict[str, Any]] = []
    for path in paths:
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as exc:  # pandas' parser exceptions vary by version
            file_errors.append({"file": str(path), "error": str(exc)})
            continue
        missing = sorted(set(REQUIRED_FIELDS) - set(frame.columns))
        if missing:
            file_errors.append(
                {"file": str(path), "error": "missing required columns", "missing": missing}
            )
            continue
        frame = frame.copy()
        frame["aggregation_source_file"] = str(path)
        frame["aggregation_source_row"] = np.arange(2, len(frame) + 2)
        frames.append(frame)
    if not frames:
        columns = list(REQUIRED_FIELDS) + [
            "aggregation_source_file",
            "aggregation_source_row",
        ]
        return pd.DataFrame(columns=columns), file_errors
    return pd.concat(frames, ignore_index=True, sort=False), file_errors


def _lookup(value: Any, names: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _expected_settings(
    methods: Sequence[str], data: Sequence[np.ndarray]
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for method in methods:
        settings_value = runner.method_settings(method, data)
        if isinstance(settings_value, Mapping):
            settings: Iterable[Any] = [
                dict(setting, setting_id=setting_id)
                if isinstance(setting, Mapping)
                else {"setting_id": setting_id, "value": setting}
                for setting_id, setting in settings_value.items()
            ]
        else:
            settings = settings_value
        for fallback_index, setting in enumerate(settings):
            explicit_index = _lookup(setting, ("setting_index", "index"))
            setting_index = (
                fallback_index if explicit_index is None else int(explicit_index)
            )
            setting_id = _lookup(setting, ("setting_id", "id"))
            if setting_id is None and isinstance(setting, str):
                setting_id = setting
            tuning_parameter = _lookup(
                setting, ("tuning_parameter", "parameter", "parameter_name")
            )
            tuning_value = _lookup(setting, ("tuning_value", "value"))
            record = {
                "method": method,
                "setting_index": setting_index,
                "setting_id": None if setting_id is None else str(setting_id),
                "tuning_parameter": (
                    None if tuning_parameter is None else str(tuning_parameter)
                ),
                "tuning_value": tuning_value,
                "full_setting": setting,
            }
            key = (method, setting_index)
            if key in index:
                raise ValueError(f"method_settings returned duplicate key {key}")
            index[key] = record
            rows.append(record)
    return pd.DataFrame(rows), index


def _parse_integer(value: Any, field: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return int(text)


def _parse_nonnegative_float(value: Any, field: str, allow_blank: bool = False) -> Optional[float]:
    text = str(value).strip()
    if allow_blank and (not text or text.lower() in {"nan", "none"}):
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not np.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and nonnegative, got {value!r}")
    return parsed


def _parse_boolean(value: Any, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} must be true or false, got {value!r}")


def _is_dag(adjacency: np.ndarray) -> bool:
    indegrees = adjacency.sum(axis=0).astype(int)
    ready = [int(node) for node in np.flatnonzero(indegrees == 0)]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegrees[child] -= 1
            if indegrees[child] == 0:
                ready.append(int(child))
    return visited == adjacency.shape[0]


def _parse_adjacency(value: Any, field: str, require_dag: bool) -> Tuple[np.ndarray, str]:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not valid JSON: {exc.msg}") from exc
    array = np.asarray(payload)
    if array.shape != (N_NODES, N_NODES):
        raise ValueError(
            f"{field} must have shape {(N_NODES, N_NODES)}, got {array.shape}"
        )
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only binary numbers") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{field} must contain only 0 and 1")
    adjacency = numeric.astype(np.uint8)
    if np.any(np.diag(adjacency)):
        raise ValueError(f"{field} must have a zero diagonal")
    if require_dag and not _is_dag(adjacency):
        raise ValueError(f"{field} must be acyclic")
    canonical = json.dumps(adjacency.tolist(), separators=(",", ":"))
    return adjacency, canonical


def _binary_array(values: Any, field: str, shape: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {array.shape}")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only binary numbers") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{field} must contain only 0 and 1")
    return numeric.astype(np.uint8)


def _parse_required_json(value: Any, field: str) -> Any:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not valid JSON: {exc.msg}") from exc


def _parse_estimated_targets(
    value: Any, method: str, intended_targets: np.ndarray
) -> Tuple[np.ndarray, str]:
    payload = _parse_required_json(value, "estimated_targets")
    if method == "gnies_unknown":
        targets = _binary_array(payload, "estimated_targets", (N_NODES,))
    elif method in CONTEXT_TARGET_METHODS:
        targets = _binary_array(
            payload,
            "estimated_targets",
            intended_targets.shape,
        )
    else:
        raise ValueError(f"no estimated-target contract is declared for {method!r}")

    if method in ORACLE_TARGET_METHODS and not np.array_equal(
        targets, intended_targets
    ):
        raise ValueError(
            f"{method} estimated_targets must equal the intended-target oracle"
        )
    if method == "utigsp_intended" and np.any(
        np.logical_and(intended_targets == 1, targets != 1)
    ):
        raise ValueError(
            "utigsp_intended estimated_targets must contain every supplied "
            "known-present target"
        )
    canonical = json.dumps(targets.tolist(), separators=(",", ":"))
    return targets, canonical


def _validate_graph_contract(
    method: str,
    graph_representation: str,
    estimated_dag: np.ndarray,
    estimated_graph: np.ndarray,
    estimated_targets: np.ndarray,
    recompute_icpdag: bool,
) -> None:
    expected_representation = GRAPH_REPRESENTATIONS.get(method)
    if graph_representation != expected_representation:
        raise ValueError(
            f"graph_representation {graph_representation!r} does not match "
            f"{expected_representation!r} for {method}"
        )

    dag_skeleton = np.logical_or(estimated_dag, estimated_dag.T)
    graph_skeleton = np.logical_or(estimated_graph, estimated_graph.T)
    if not np.array_equal(dag_skeleton, graph_skeleton):
        raise ValueError(
            "estimated_graph and estimated_dag must have identical skeletons"
        )
    graph_directed = np.logical_and(estimated_graph, ~estimated_graph.T.astype(bool))
    if np.any(np.logical_and(graph_directed, estimated_dag != 1)):
        raise ValueError(
            "every directed estimated_graph edge must have the same orientation "
            "in estimated_dag"
        )
    for child in range(N_NODES):
        parents = [int(node) for node in np.flatnonzero(estimated_dag[:, child])]
        for left, right in combinations(parents, 2):
            if dag_skeleton[left, right]:
                continue
            if not (
                graph_directed[left, child]
                and graph_directed[right, child]
            ):
                raise ValueError(
                    "estimated_dag introduces an unshielded collider that is "
                    "not directed in estimated_graph, so it is not a "
                    "consistent extension"
                )

    if expected_representation == "dag_and_i_cpdag" and recompute_icpdag:
        from src.utils import interventional_cpdag

        try:
            recomputed_values = interventional_cpdag(
                estimated_dag, estimated_targets
            )
        except Exception as exc:
            raise ValueError(f"could not recompute the saved I-CPDAG: {exc}") from exc
        recomputed = _binary_array(
            recomputed_values, "recomputed I-CPDAG", (N_NODES, N_NODES)
        )
        if not np.array_equal(recomputed, estimated_graph):
            raise ValueError(
                "estimated_graph does not equal the I-CPDAG recomputed from "
                "estimated_dag and estimated_targets"
            )


def _validate_saved_metrics(
    row: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> List[str]:
    reasons: List[str] = []
    count_fields = (
        "directed_tp",
        "directed_fp",
        "directed_fn",
        "directed_tn",
        "skeleton_tp",
        "skeleton_fp",
        "skeleton_fn",
        "skeleton_tn",
    )
    rate_fields = (
        "directed_tpr",
        "directed_fpr",
        "skeleton_tpr",
        "skeleton_fpr",
    )
    for field in count_fields:
        try:
            observed = _parse_integer(row[field], field)
        except ValueError as exc:
            reasons.append(str(exc))
            continue
        if observed != recomputed[field]:
            reasons.append(
                f"saved {field}={observed} does not match recomputed "
                f"{recomputed[field]}"
            )
    for field in rate_fields:
        try:
            observed = float(str(row[field]).strip())
        except ValueError:
            reasons.append(f"{field} must be numeric, got {row[field]!r}")
            continue
        if not np.isfinite(observed) or not np.isclose(
            observed, recomputed[field], rtol=0.0, atol=1e-12
        ):
            reasons.append(
                f"saved {field}={observed!r} does not match recomputed "
                f"{recomputed[field]!r}"
            )
    return reasons


def _parse_optional_json(value: Any, field: str) -> Optional[str]:
    text = str(value).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not valid JSON: {exc.msg}") from exc
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )


def _parse_method_config_identity(
    value: Any, method: str, expected_setting: Any
) -> Tuple[Dict[str, Any], str]:
    payload = _parse_required_json(value, "method_config")
    if not isinstance(payload, dict):
        raise ValueError("method_config must be a JSON object")
    if "setting" not in payload:
        raise ValueError("method_config must contain its setting object")
    if _canonical_json(payload["setting"]) != _canonical_json(expected_setting):
        raise ValueError(
            "method_config.setting does not match the predeclared method setting"
        )

    implementation = payload.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("method_config.implementation must be an object")
    if implementation.get("protocol_revision") != getattr(
        runner, "PROTOCOL_REVISION", None
    ):
        raise ValueError(
            "method_config implementation protocol_revision does not match the runner"
        )
    if not SHA256_RE.fullmatch(str(implementation.get("sha256", ""))):
        raise ValueError(
            "method_config implementation sha256 must be a SHA-256 digest"
        )
    implementation_files = implementation.get("files")
    if not isinstance(implementation_files, dict) or not implementation_files:
        raise ValueError("method_config implementation files must be nonempty")
    if any(
        not SHA256_RE.fullmatch(str(digest))
        for digest in implementation_files.values()
    ):
        raise ValueError(
            "every method_config implementation file digest must be SHA-256"
        )

    runtime_environment = payload.get("runtime_environment")
    if not isinstance(runtime_environment, dict):
        raise ValueError("method_config.runtime_environment must be an object")
    if not str(runtime_environment.get("python_version", "")).strip():
        raise ValueError(
            "method_config.runtime_environment.python_version is empty"
        )
    if not SHA256_RE.fullmatch(
        str(runtime_environment.get("distributions_sha256", ""))
    ):
        raise ValueError(
            "method_config runtime distributions_sha256 must be a SHA-256 digest"
        )
    if not isinstance(runtime_environment.get("key_distributions"), dict):
        raise ValueError(
            "method_config.runtime_environment.key_distributions must be an object"
        )

    expected_seed_usage = runner._seed_usage(method)
    if payload.get("seed_usage") != expected_seed_usage:
        raise ValueError(
            f"method_config.seed_usage must equal {expected_seed_usage!r}"
        )

    identity = dict(payload)
    identity.pop("setting")
    return payload, _canonical_json(identity)


def _same_setting_value(observed: Any, expected: Any) -> bool:
    if expected is None:
        return True
    observed_text = str(observed).strip()
    try:
        return float(observed_text) == float(expected)
    except (TypeError, ValueError):
        return observed_text == str(expected)


def _source(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "file": str(row["aggregation_source_file"]),
        "row": int(row["aggregation_source_row"]),
    }


def _issue_row(row: Mapping[str, Any], reasons: Sequence[str]) -> Dict[str, Any]:
    issue = _source(row)
    issue.update(
        {
            "method": str(row.get("method", "")),
            "setting_index": str(row.get("setting_index", "")),
            "setting_id": str(row.get("setting_id", "")),
            "reasons": list(reasons),
        }
    )
    return issue


def _runner_constant(names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(runner, name):
            return getattr(runner, name)
    return None


def _info_value(info: Any, names: Sequence[str]) -> Any:
    return _lookup(info, names)


def _truth(true_dag: Any) -> np.ndarray:
    array = np.asarray(true_dag)
    if array.shape != (N_NODES, N_NODES):
        raise ValueError(
            f"Sachs truth must have shape {(N_NODES, N_NODES)}, got {array.shape}"
        )
    numeric = array.astype(float)
    if not np.isin(numeric, (0.0, 1.0)).all() or np.any(np.diag(numeric)):
        raise ValueError("Sachs truth must be a binary, zero-diagonal adjacency")
    truth = numeric.astype(np.uint8)
    if not _is_dag(truth):
        raise ValueError("Sachs truth adjacency is cyclic")
    directed_positive_count = int(truth.sum())
    directed_negative_count = int(N_NODES * (N_NODES - 1) - truth.sum())
    skeleton = np.logical_or(truth, truth.T)
    upper = np.triu(np.ones_like(skeleton, dtype=bool), k=1)
    skeleton_positive_count = int(np.logical_and(skeleton, upper).sum())
    skeleton_negative_count = int(np.logical_and(~skeleton, upper).sum())
    observed = (
        directed_positive_count,
        directed_negative_count,
        skeleton_positive_count,
        skeleton_negative_count,
    )
    expected = (
        DIRECTED_POSITIVES,
        DIRECTED_NEGATIVES,
        SKELETON_POSITIVES,
        SKELETON_NEGATIVES,
    )
    if observed != expected:
        raise ValueError(
            "unexpected Sachs truth universe: "
            f"observed {observed}, expected {expected}"
        )
    return truth


def _metrics(estimated_dag: np.ndarray, true_dag: np.ndarray) -> Dict[str, Any]:
    off_diagonal = ~np.eye(N_NODES, dtype=bool)
    estimated = estimated_dag.astype(bool)
    truth = true_dag.astype(bool)

    directed_tp = int(np.logical_and.reduce((estimated, truth, off_diagonal)).sum())
    directed_fp = int(
        np.logical_and.reduce((estimated, ~truth, off_diagonal)).sum()
    )
    directed_fn = int(
        np.logical_and.reduce((~estimated, truth, off_diagonal)).sum()
    )
    directed_tn = int(
        np.logical_and.reduce((~estimated, ~truth, off_diagonal)).sum()
    )

    estimated_skeleton = np.logical_or(estimated, estimated.T)
    true_skeleton = np.logical_or(truth, truth.T)
    upper = np.triu(np.ones((N_NODES, N_NODES), dtype=bool), k=1)
    skeleton_tp = int(
        np.logical_and.reduce((estimated_skeleton, true_skeleton, upper)).sum()
    )
    skeleton_fp = int(
        np.logical_and.reduce((estimated_skeleton, ~true_skeleton, upper)).sum()
    )
    skeleton_fn = int(
        np.logical_and.reduce((~estimated_skeleton, true_skeleton, upper)).sum()
    )
    skeleton_tn = int(
        np.logical_and.reduce((~estimated_skeleton, ~true_skeleton, upper)).sum()
    )

    return {
        # These four counts are the coordinates used in the UT-IGSP Sachs plot.
        "paper_directed_tp": directed_tp,
        "paper_directed_fp": directed_fp,
        "paper_skeleton_tp": skeleton_tp,
        "paper_skeleton_fp": skeleton_fp,
        "directed_tp": directed_tp,
        "directed_fp": directed_fp,
        "directed_tn": directed_tn,
        "directed_fn": directed_fn,
        "directed_tpr": directed_tp / DIRECTED_POSITIVES,
        "directed_fpr": directed_fp / DIRECTED_NEGATIVES,
        "directed_positives": DIRECTED_POSITIVES,
        "directed_negatives": DIRECTED_NEGATIVES,
        "skeleton_tp": skeleton_tp,
        "skeleton_fp": skeleton_fp,
        "skeleton_tn": skeleton_tn,
        "skeleton_fn": skeleton_fn,
        "skeleton_tpr": skeleton_tp / SKELETON_POSITIVES,
        "skeleton_fpr": skeleton_fp / SKELETON_NEGATIVES,
        "skeleton_positives": SKELETON_POSITIVES,
        "skeleton_negatives": SKELETON_NEGATIVES,
    }


def _row_signature(row: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(str(row[field]).strip() for field in REQUIRED_FIELDS)


def _empty_outputs(raw_columns: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined_columns = list(
        dict.fromkeys(
            list(raw_columns)
            + [
                "estimated_dag_canonical",
                "estimated_graph_canonical",
                "estimated_targets_canonical",
                "is_nonoptimal",
            ]
        )
    )
    metric_columns = [
        "paper_directed_tp",
        "paper_directed_fp",
        "paper_skeleton_tp",
        "paper_skeleton_fp",
        "directed_tp",
        "directed_fp",
        "directed_tn",
        "directed_fn",
        "directed_tpr",
        "directed_fpr",
        "directed_positives",
        "directed_negatives",
        "skeleton_tp",
        "skeleton_fp",
        "skeleton_tn",
        "skeleton_fn",
        "skeleton_tpr",
        "skeleton_fpr",
        "skeleton_positives",
        "skeleton_negatives",
    ]
    combined = pd.DataFrame(columns=combined_columns)
    roc_columns = list(dict.fromkeys(combined_columns + metric_columns))
    return combined, pd.DataFrame(columns=roc_columns)


def aggregate_results(
    inputs: Optional[Sequence[Path]] = None,
    output_dir: Path = DEFAULT_OUTPUT,
    data_root: Path = DEFAULT_DATA_ROOT,
    methods: Optional[Sequence[str]] = None,
    allow_incomplete: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Aggregate result fragments, validate coverage, and recompute ROC metrics."""

    output_dir = Path(output_dir).expanduser().resolve()
    selected_methods = _normalise_methods(methods)
    data, true_dag_value, intended_targets_value, info = load_sachs_data(
        Path(data_root).expanduser().resolve()
    )
    true_dag = _truth(true_dag_value)
    intended_targets = _binary_array(
        intended_targets_value,
        "loaded intended_targets",
        (5, N_NODES),
    )
    sample_sizes = [int(len(values)) for values in data]
    expected_node_names = _info_value(info, ("node_names",))
    if expected_node_names is None:
        expected_node_names = list(SACHS_NODE_NAMES)
    try:
        __import__("causaldag")
        recompute_icpdag = True
        exact_icpdag_note = (
            "enforced for dag_and_i_cpdag rows using src.utils.interventional_cpdag"
        )
    except ImportError as exc:
        recompute_icpdag = False
        exact_icpdag_note = (
            "skipped because causaldag is unavailable in the aggregation "
            f"environment: {exc}"
        )
    expected_frame, expected_index = _expected_settings(selected_methods, data)
    expected_keys = set(expected_index)

    paths = _result_paths(inputs)
    raw, file_errors = _read_results(paths)
    known_methods = set(runner.METHODS)
    selected_set = set(selected_methods)
    ignored_unselected = 0
    unexpected_rows: List[Dict[str, Any]] = []
    malformed_rows: List[Dict[str, Any]] = []
    non_ok_rows: List[Dict[str, Any]] = []
    conflict_rows: List[Dict[str, Any]] = []
    nonoptimal_rows: List[Dict[str, Any]] = []
    method_config_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    method_seeds: Dict[str, set] = defaultdict(set)
    identical_duplicate_rows = 0
    global_errors: List[str] = []

    expected_experiment_version = _runner_constant(
        ("SACHS_EXPERIMENT_VERSION", "EXPERIMENT_VERSION")
    )
    expected_data_format_version = _info_value(
        info, ("data_format_version", "format_version")
    )
    if expected_data_format_version is None:
        expected_data_format_version = _runner_constant(("DATA_FORMAT_VERSION",))
    expected_data_sha256 = _info_value(
        info, ("data_sha256", "processed_data_sha256", "dataset_sha256")
    )
    expected_source_commit = _info_value(
        info, ("source_commit", "upstream_commit")
    )

    keyed_rows: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for index, row in raw.iterrows():
        method = str(row["method"]).strip()
        if method in known_methods and method not in selected_set:
            ignored_unselected += 1
            continue
        if method not in known_methods:
            unexpected_rows.append(_issue_row(row, [f"unknown method {method!r}"]))
            continue
        try:
            setting_index = _parse_integer(row["setting_index"], "setting_index")
        except ValueError as exc:
            malformed_rows.append(_issue_row(row, [str(exc)]))
            continue
        key = (method, setting_index)
        if key not in expected_keys:
            unexpected_rows.append(
                _issue_row(row, [f"unexpected method/setting_index key {key}"])
            )
            continue
        keyed_rows[key].append(index)

    keep_indices = set()
    for key, indices in keyed_rows.items():
        signatures = {_row_signature(raw.loc[index]) for index in indices}
        if len(signatures) > 1:
            conflict_rows.append(
                {
                    "method": key[0],
                    "setting_index": key[1],
                    "rows": [_source(raw.loc[index]) for index in indices],
                }
            )
            continue
        keep_indices.add(indices[0])
        identical_duplicate_rows += len(indices) - 1

    accepted: List[
        Tuple[pd.Series, np.ndarray, str, Optional[str], Optional[str], bool]
    ] = []
    accepted_versions: Dict[str, set] = {
        "experiment_version": set(),
        "data_format_version": set(),
        "data_sha256": set(),
        "source_commit": set(),
    }
    for index in sorted(keep_indices):
        row = raw.loc[index]
        method = str(row["method"]).strip()
        setting_index = _parse_integer(row["setting_index"], "setting_index")
        key = (method, setting_index)
        expected = expected_index[key]
        reasons: List[str] = []

        expected_setting_id = expected["setting_id"]
        observed_setting_id = str(row["setting_id"]).strip()
        if not observed_setting_id:
            reasons.append("setting_id is empty")
        elif expected_setting_id is not None and observed_setting_id != expected_setting_id:
            reasons.append(
                f"setting_id {observed_setting_id!r} does not match expected "
                f"{expected_setting_id!r}"
            )

        expected_tuning_parameter = expected["tuning_parameter"]
        observed_tuning_parameter = str(row["tuning_parameter"]).strip()
        if (
            expected_tuning_parameter is not None
            and observed_tuning_parameter != expected_tuning_parameter
        ):
            reasons.append(
                f"tuning_parameter {observed_tuning_parameter!r} does not match "
                f"{expected_tuning_parameter!r}"
            )
        if not _same_setting_value(row["tuning_value"], expected["tuning_value"]):
            reasons.append(
                f"tuning_value {str(row['tuning_value']).strip()!r} does not "
                f"match {expected['tuning_value']!r}"
            )

        observed_label = str(row["method_label"]).strip()
        expected_label = str(runner.METHOD_LABELS[method])
        if observed_label != expected_label:
            reasons.append(
                f"method_label {observed_label!r} does not match {expected_label!r}"
            )
        for field in (
            "tuning_parameter",
            "tuning_value",
            "target_knowledge",
            "graph_representation",
        ):
            if not str(row[field]).strip():
                reasons.append(f"{field} is empty")

        try:
            observed_node_names = _parse_required_json(row["node_names"], "node_names")
            if observed_node_names != list(expected_node_names):
                reasons.append("node_names do not match the loaded Sachs data")
        except ValueError as exc:
            reasons.append(str(exc))
        try:
            observed_sample_sizes = _parse_required_json(
                row["sample_sizes"], "sample_sizes"
            )
            if observed_sample_sizes != sample_sizes:
                reasons.append("sample_sizes do not match the loaded Sachs data")
        except ValueError as exc:
            reasons.append(str(exc))
        for field, expected_value in (
            ("n_observational", sample_sizes[0]),
            ("n_interventional", sum(sample_sizes[1:])),
            ("total_samples", sum(sample_sizes)),
        ):
            try:
                observed_value = _parse_integer(row[field], field)
                if observed_value != expected_value:
                    reasons.append(
                        f"{field}={observed_value} does not match {expected_value}"
                    )
            except ValueError as exc:
                reasons.append(str(exc))

        seed: Optional[int] = None
        try:
            seed = _parse_integer(row["seed"], "seed")
            if seed < 0:
                reasons.append("seed must be nonnegative")
        except ValueError as exc:
            reasons.append(str(exc))

        oracle_methods = set(
            getattr(
                runner,
                "ORACLE_METHODS",
                {name for name in runner.METHODS if name.endswith("_oracle")},
            )
        )
        expected_target_knowledge = (
            "intended_targets_oracle"
            if method in oracle_methods
            else "intended_targets_known_present"
            if method == "utigsp_intended"
            else "unknown"
        )
        observed_target_knowledge = str(row["target_knowledge"]).strip()
        if observed_target_knowledge != expected_target_knowledge:
            reasons.append(
                f"target_knowledge {observed_target_knowledge!r} does not match "
                f"{expected_target_knowledge!r}"
            )
        try:
            _parse_nonnegative_float(row["fit_seconds"], "fit_seconds")
            _parse_nonnegative_float(
                row["solve_seconds"], "solve_seconds", allow_blank=True
            )
        except ValueError as exc:
            reasons.append(str(exc))

        data_sha256 = str(row["data_sha256"]).strip()
        if not SHA256_RE.fullmatch(data_sha256):
            reasons.append("data_sha256 must be a 64-character hexadecimal digest")
        if not str(row["source_commit"]).strip():
            reasons.append("source_commit is empty")
        if (
            expected_experiment_version is not None
            and str(row["experiment_version"]).strip()
            != str(expected_experiment_version)
        ):
            reasons.append(
                f"experiment_version does not match {expected_experiment_version!r}"
            )
        if (
            expected_data_format_version is not None
            and str(row["data_format_version"]).strip()
            != str(expected_data_format_version)
        ):
            reasons.append(
                f"data_format_version does not match {expected_data_format_version!r}"
            )
        if (
            expected_data_sha256 is not None
            and data_sha256 != str(expected_data_sha256)
        ):
            reasons.append("data_sha256 does not match the loaded Sachs data")
        if (
            expected_source_commit is not None
            and str(row["source_commit"]).strip() != str(expected_source_commit)
        ):
            reasons.append("source_commit does not match the loaded Sachs data")

        status = str(row["status"]).strip().lower()
        is_nonoptimal = status == "ok_nonoptimal"
        if is_nonoptimal:
            nonoptimal_reasons: List[str] = []
            if not method.startswith("ps_mip"):
                nonoptimal_reasons.append(
                    "ok_nonoptimal is valid only for a PS-MIP method"
                )
            try:
                if _parse_boolean(row["optimal"], "optimal"):
                    nonoptimal_reasons.append(
                        "ok_nonoptimal must be accompanied by optimal=false"
                    )
            except ValueError as exc:
                nonoptimal_reasons.append(str(exc))
            if nonoptimal_reasons:
                non_ok_rows.append(_issue_row(row, nonoptimal_reasons))
                continue
        elif status != "ok":
            non_ok_rows.append(
                _issue_row(
                    row,
                    [
                        f"status is {str(row['status']).strip()!r}, expected "
                        "'ok' (or PS-MIP 'ok_nonoptimal' with optimal=false)"
                    ],
                )
            )
            continue
        if str(row["error"]).strip():
            reasons.append("an ok row must have an empty error field")

        method_config_identity: Optional[str] = None
        try:
            _method_config, method_config_identity = _parse_method_config_identity(
                row["method_config"], method, expected["full_setting"]
            )
        except ValueError as exc:
            reasons.append(str(exc))
        try:
            estimated_dag, estimated_dag_canonical = _parse_adjacency(
                row["estimated_dag"], "estimated_dag", require_dag=True
            )
        except ValueError as exc:
            reasons.append(str(exc))
            estimated_dag = np.empty((0, 0), dtype=np.uint8)
            estimated_dag_canonical = ""
        try:
            estimated_graph, estimated_graph_canonical = _parse_adjacency(
                row["estimated_graph"], "estimated_graph", require_dag=False
            )
        except ValueError as exc:
            reasons.append(str(exc))
            estimated_graph = np.empty((0, 0), dtype=np.uint8)
            estimated_graph_canonical = None
        try:
            estimated_targets, estimated_targets_canonical = _parse_estimated_targets(
                row["estimated_targets"], method, intended_targets
            )
        except ValueError as exc:
            reasons.append(str(exc))
            estimated_targets = np.empty((0,), dtype=np.uint8)
            estimated_targets_canonical = None

        if (
            estimated_dag.shape == (N_NODES, N_NODES)
            and estimated_graph.shape == (N_NODES, N_NODES)
            and estimated_targets.size
        ):
            try:
                _validate_graph_contract(
                    method,
                    str(row["graph_representation"]).strip(),
                    estimated_dag,
                    estimated_graph,
                    estimated_targets,
                    recompute_icpdag,
                )
            except (ImportError, ValueError) as exc:
                reasons.append(str(exc))
            reasons.extend(_validate_saved_metrics(row, _metrics(estimated_dag, true_dag)))

        if reasons:
            malformed_rows.append(_issue_row(row, reasons))
            continue

        assert seed is not None
        assert method_config_identity is not None
        method_config_groups[method][method_config_identity].append(_source(row))
        method_seeds[method].add(seed)

        for field in accepted_versions:
            accepted_versions[field].add(str(row[field]).strip())
        accepted.append(
            (
                row,
                estimated_dag,
                estimated_dag_canonical,
                estimated_graph_canonical,
                estimated_targets_canonical,
                is_nonoptimal,
            )
        )
        if is_nonoptimal:
            nonoptimal_record = _source(row)
            nonoptimal_record.update(
                {
                    "method": method,
                    "setting_index": setting_index,
                    "setting_id": observed_setting_id,
                    "solver_status": str(row["solver_status"]).strip(),
                    "mip_gap": str(row["mip_gap"]).strip(),
                }
            )
            nonoptimal_rows.append(nonoptimal_record)

    for field, values in accepted_versions.items():
        if len(values) > 1:
            global_errors.append(
                f"accepted rows disagree on {field}: {sorted(values)}"
            )

    method_config_drift: List[Dict[str, Any]] = []
    for method in selected_methods:
        groups = method_config_groups.get(method, {})
        if len(groups) > 1:
            group_records = [
                {
                    "identity_sha256": hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest(),
                    "rows": sources,
                }
                for identity, sources in sorted(groups.items())
            ]
            method_config_drift.append(
                {"method": method, "identity_groups": group_records}
            )
            global_errors.append(
                f"{method} rows have heterogeneous method_config identities"
            )

    seed_drift: List[Dict[str, Any]] = []
    for method in selected_methods:
        seeds = sorted(method_seeds.get(method, set()))
        if len(seeds) > 1:
            seed_drift.append({"method": method, "seeds": seeds})
            global_errors.append(
                f"{method} rows use multiple top-level seeds: {seeds}"
            )

    accepted_keys = {
        (str(row["method"]).strip(), int(str(row["setting_index"]).strip()))
        for row, _dag, _dag_json, _graph_json, _targets_json, _nonoptimal in accepted
    }
    missing_keys = sorted(
        expected_keys - accepted_keys,
        key=lambda value: (list(runner.METHODS).index(value[0]), value[1]),
    )
    missing_rows = [expected_index[key] for key in missing_keys]

    raw_columns = list(raw.columns)
    combined_records: List[Dict[str, Any]] = []
    roc_records: List[Dict[str, Any]] = []
    for (
        row,
        estimated_dag,
        dag_json,
        graph_json,
        targets_json,
        is_nonoptimal,
    ) in accepted:
        record = row.to_dict()
        record["setting_index"] = int(str(record["setting_index"]).strip())
        record["seed"] = int(str(record["seed"]).strip())
        record["fit_seconds"] = float(str(record["fit_seconds"]).strip())
        record["estimated_dag_canonical"] = dag_json
        record["estimated_graph_canonical"] = graph_json
        record["estimated_targets_canonical"] = targets_json
        record["is_nonoptimal"] = bool(is_nonoptimal)
        combined_records.append(record)
        metric_record = dict(record)
        metric_record.update(_metrics(estimated_dag, true_dag))
        roc_records.append(metric_record)

    if combined_records:
        combined = pd.DataFrame(combined_records)
        method_rank = {method: rank for rank, method in enumerate(runner.METHODS)}
        combined["_method_rank"] = combined["method"].map(method_rank)
        combined = combined.sort_values(
            ["_method_rank", "setting_index", "setting_id"], kind="stable"
        ).drop(columns="_method_rank").reset_index(drop=True)
        roc_points = pd.DataFrame(roc_records)
        roc_points["_method_rank"] = roc_points["method"].map(method_rank)
        roc_points = roc_points.sort_values(
            ["_method_rank", "setting_index", "setting_id"], kind="stable"
        ).drop(columns="_method_rank").reset_index(drop=True)
    else:
        combined, roc_points = _empty_outputs(raw_columns)

    blockers = bool(
        file_errors
        or unexpected_rows
        or malformed_rows
        or non_ok_rows
        or conflict_rows
        or missing_rows
        or global_errors
    )
    report: Dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "status": "incomplete" if blockers else "complete",
        "allow_incomplete": bool(allow_incomplete),
        "input_files": [str(path) for path in paths],
        "selected_methods": selected_methods,
        "schema_validation": {
            "contract": "experiments.run_sachs_roc.RESULT_FIELDS",
            "required_fields": list(REQUIRED_FIELDS),
        },
        "counts": {
            "input_files": len(paths),
            "raw_rows": len(raw),
            "ignored_unselected_method_rows": ignored_unselected,
            "expected_rows": len(expected_frame),
            "accepted_rows": len(combined),
            "accepted_nonoptimal_rows": len(nonoptimal_rows),
            "identical_duplicate_rows_dropped": identical_duplicate_rows,
            "missing_rows": len(missing_rows),
            "unexpected_rows": len(unexpected_rows),
            "malformed_rows": len(malformed_rows),
            "non_ok_rows": len(non_ok_rows),
            "conflicting_duplicate_keys": len(conflict_rows),
        },
        "truth_universe": {
            "nodes": N_NODES,
            "directed_positives": DIRECTED_POSITIVES,
            "directed_negatives": DIRECTED_NEGATIVES,
            "skeleton_positives": SKELETON_POSITIVES,
            "skeleton_negatives": SKELETON_NEGATIVES,
        },
        "graph_validation": {
            "representation_by_method": GRAPH_REPRESENTATIONS,
            "consistent_extension_check": "enforced for every accepted row",
            "exact_i_cpdag_recomputation": exact_icpdag_note,
            "pdag_extension_note": (
                "GnIES/GIES DAGs are required to be consistent extensions of "
                "their saved I-CPDAGs; exact equality to one package-chosen "
                "extension is not required because valid extensions are not unique."
            ),
        },
        "target_validation": {
            "context_specific_shape": [5, N_NODES],
            "context_specific_methods": sorted(CONTEXT_TARGET_METHODS),
            "oracle_methods_must_equal_intended_targets": sorted(
                ORACLE_TARGET_METHODS
            ),
            "utigsp_intended_semantics": (
                "every supplied intended target must remain present; learned "
                "off-targets are allowed"
            ),
            "gnies_unknown_shape": [N_NODES],
            "gnies_unknown_semantics": (
                "native learned target union; context assignments are not "
                "fabricated because GnIES does not return them"
            ),
        },
        "expected_settings": expected_frame.where(
            pd.notnull(expected_frame), None
        ).to_dict(orient="records"),
        "accepted_nonoptimal_rows": nonoptimal_rows,
        "method_identity_validation": {
            "rule": (
                "within each method, method_config must be identical after "
                "removing only the top-level setting object"
            ),
            "method_config_drift": method_config_drift,
            "seed_rule": "one top-level seed per selected method path",
            "seed_drift": seed_drift,
            "observed_seeds": {
                method: sorted(method_seeds.get(method, set()))
                for method in selected_methods
            },
        },
        "issues": {
            "file_errors": file_errors,
            "global_errors": global_errors,
            "missing_rows": missing_rows,
            "unexpected_rows": unexpected_rows,
            "malformed_rows": malformed_rows,
            "non_ok_rows": non_ok_rows,
            "conflicting_duplicates": conflict_rows,
        },
        "outputs_written": False,
    }

    report_path = output_dir / "validation_report.json"
    if blockers and not allow_incomplete:
        _atomic_json(report, report_path)
        raise ValidationError(
            "Sachs result validation failed; inspect " f"{report_path}"
        )

    _atomic_csv(combined, output_dir / "combined_results.csv")
    _atomic_csv(roc_points, output_dir / "roc_points.csv")
    report["outputs_written"] = True
    _atomic_json(report, report_path)
    return combined, roc_points, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Sachs result fragments and recompute directed/skeleton "
            "ROC operating points."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=f"CSV files or directories (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--methods",
        nargs="+",
        help="subset of runner methods; comma-separated values are also accepted",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write partial summaries while recording every validation issue",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        combined, _roc_points, report = aggregate_results(
            inputs=args.inputs or None,
            output_dir=args.output_dir,
            data_root=args.data_root,
            methods=args.methods,
            allow_incomplete=args.allow_incomplete,
        )
    except (ValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Sachs aggregation {report['status']}: {len(combined)}/"
        f"{report['counts']['expected_rows']} operating points"
    )
    print(f"ROC points: {Path(args.output_dir).resolve() / 'roc_points.csv'}")
    print(
        "Validation report: "
        f"{Path(args.output_dir).resolve() / 'validation_report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
