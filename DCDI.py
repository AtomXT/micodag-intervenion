#!/usr/bin/env python3
"""Adapter for the official DCDI-G implementation with unknown targets.

The upstream source is intentionally kept outside this repository's Python
package.  By default, this module expects the official repository at
``external/dcdi`` checked out at :data:`DCDI_COMMIT`.

Only data values and environment labels are passed to DCDI.  The upstream
loader also requires ground-truth DAG and intervention files, so this adapter
stages an all-zero DAG and blank target rows rather than exposing simulation
truth to the estimator.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DCDI_PATH = PROJECT_ROOT / "external" / "dcdi"
DCDI_COMMIT = "594d328eae7795785e0d1a1138945e28a4fec037"
COMPLETION_FILENAME = "adapter_complete.json"
DEFAULT_HIDDEN_DIM = 8
DEFAULT_MU_INIT = 1e-2

# DCDI's final reporting calls R-backed CDT metrics after saving the learned
# graph.  The project evaluates results itself, so the subprocess replaces only
# those reporting functions; the model, objective, and optimization are
# untouched.
_UPSTREAM_LAUNCHER = """\
import runpy
import sys

checkout, entrypoint = sys.argv[1:3]
arguments = sys.argv[3:]
sys.path.insert(0, checkout)
import cdt.metrics as metrics
metrics.SID = lambda *args, **kwargs: 0.0
metrics.SHD_CPDAG = lambda *args, **kwargs: 0.0
from matplotlib.figure import Figure
_savefig = Figure.savefig
def _savefig_without_removed_padding(self, *args, **kwargs):
    kwargs.pop("padding", None)
    return _savefig(self, *args, **kwargs)
Figure.savefig = _savefig_without_removed_padding
sys.argv = [entrypoint] + arguments
runpy.run_path(entrypoint, run_name="__main__")
"""


class DCDIError(RuntimeError):
    """Raised when the external DCDI fit cannot produce valid artifacts."""


def _as_arrays(data: Sequence[Any]) -> Sequence[np.ndarray]:
    """Validate observational-first environment data and return float arrays."""
    arrays = [
        np.ascontiguousarray(
            np.asarray(x.values if hasattr(x, "values") else x, dtype=float)
        )
        for x in data
    ]
    if len(arrays) < 2:
        raise ValueError(
            "data must contain the observational sample followed by at least "
            "one interventional environment"
        )
    if any(x.ndim != 2 or x.shape[0] == 0 for x in arrays):
        raise ValueError("every environment must be a nonempty two-dimensional array")
    p = arrays[0].shape[1]
    if p == 0 or any(x.shape[1] != p for x in arrays):
        raise ValueError("all environments must contain the same ordered variables")
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("data must contain only finite values")
    return arrays


def _validate_options(
    graph_penalty: float,
    target_penalty: float,
    random_seed: int,
    hidden_dim: int,
    mu_init: float,
    num_train_iter: int,
    timeout: Optional[float],
) -> None:
    """Validate scalar options before creating files or starting a process."""
    if not np.isfinite(graph_penalty) or graph_penalty < 0:
        raise ValueError("graph_penalty must be finite and nonnegative")
    if not np.isfinite(target_penalty) or target_penalty < 0:
        raise ValueError("target_penalty must be finite and nonnegative")
    if not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    if not isinstance(hidden_dim, (int, np.integer)) or hidden_dim <= 0:
        raise ValueError("hidden_dim must be a positive integer")
    if not np.isfinite(mu_init) or mu_init <= 0:
        raise ValueError("mu_init must be finite and positive")
    if not isinstance(num_train_iter, (int, np.integer)) or num_train_iter <= 0:
        raise ValueError("num_train_iter must be a positive integer")
    if timeout is not None and (not np.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive when supplied")


def _validate_checkout(path: Union[str, os.PathLike[str]]) -> Tuple[Path, bool]:
    """Return a clean official checkout at the pinned commit."""
    checkout = Path(path).expanduser().resolve()
    required = [
        checkout / "main.py",
        checkout / "dcdi" / "data.py",
        checkout / "dcdi" / "train.py",
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"DCDI checkout is missing required files at {checkout}. "
            f"Expected the official repository at commit {DCDI_COMMIT}."
        )

    try:
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise DCDIError(
            f"cannot verify the DCDI checkout at {checkout}; it must retain its git metadata"
        ) from error
    if head != DCDI_COMMIT:
        raise DCDIError(
            f"DCDI checkout is at {head}, expected pinned commit {DCDI_COMMIT}"
        )

    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = bool(status.strip())
    if dirty:
        raise DCDIError(
            "DCDI checkout has tracked local modifications; use a clean pinned checkout"
        )
    return checkout, dirty


def _resolve_python(path: Union[str, os.PathLike[str]]) -> Path:
    """Resolve an interpreter before changing the subprocess working directory."""
    value = os.fspath(path)
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = Path(os.path.abspath(candidate))
    else:
        located = shutil.which(value)
        if located is None:
            raise FileNotFoundError(f"Python interpreter not found: {value}")
        resolved = Path(os.path.abspath(located))
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"Python interpreter is not executable: {resolved}")
    return resolved


def _stage_inputs(arrays: Sequence[np.ndarray], data_path: Path) -> None:
    """Write DCDI's required files without staging any ground truth."""
    data_path.mkdir(parents=True, exist_ok=True)
    p = arrays[0].shape[1]
    pooled = np.concatenate(arrays, axis=0)
    regimes = np.concatenate(
        [np.full(len(values), regime, dtype=int) for regime, values in enumerate(arrays)]
    )

    np.save(data_path / "data_interv1.npy", pooled)
    np.save(data_path / "DAG1.npy", np.zeros((p, p), dtype=int))
    np.savetxt(data_path / "regime1.csv", regimes, fmt="%d", delimiter=",")

    # The official unknown-target path ignores these masks during fitting, but
    # DataManagerFile still insists on one CSV row per sample.
    with (data_path / "intervention1.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([] for _ in range(len(pooled)))


def _data_digest(arrays: Sequence[np.ndarray]) -> str:
    """Hash array shapes and values to protect persistent runs from stale reuse."""
    digest = hashlib.sha256()
    for values in arrays:
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(values, dtype="<f8", order="C").tobytes())
    return digest.hexdigest()


def _configuration(
    arrays: Sequence[np.ndarray],
    graph_penalty: float,
    target_penalty: float,
    normalize_data: bool,
    random_seed: int,
    gpu: bool,
    float_precision: bool,
    hidden_dim: int,
    mu_init: float,
    num_train_iter: int,
) -> Dict[str, Any]:
    """Build the reproducibility manifest for a persistent fit directory."""
    return {
        "adapter": "DCDI-G-perfect-unknown",
        "dcdi_commit": DCDI_COMMIT,
        "data_sha256": _data_digest(arrays),
        "environment_shapes": [list(values.shape) for values in arrays],
        "graph_penalty": float(graph_penalty),
        "target_penalty": float(target_penalty),
        "normalize_data": bool(normalize_data),
        "random_seed": int(random_seed),
        "gpu": bool(gpu),
        "float_precision": bool(float_precision),
        "hidden_dim": int(hidden_dim),
        "mu_init": float(mu_init),
        "num_train_iter": int(num_train_iter),
    }


def _write_or_check_manifest(run_path: Path, configuration: Dict[str, Any]) -> bool:
    """Write/check a run manifest and report whether a completion marker exists."""
    run_path.mkdir(parents=True, exist_ok=True)
    manifest_path = run_path / "adapter_config.json"
    dag_path = run_path / "experiment" / "train" / "DAG.npy"
    target_path = run_path / "experiment" / "train" / "gumbel_interv.npy"
    completion_path = run_path / COMPLETION_FILENAME

    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DCDIError(f"cannot read existing run manifest {manifest_path}") from error
        if previous != configuration:
            raise DCDIError(
                f"persistent run directory {run_path} contains a different configuration"
            )
    elif dag_path.exists() or target_path.exists() or completion_path.exists():
        raise DCDIError(
            f"persistent run directory {run_path} has untracked DCDI artifacts"
        )
    else:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(configuration, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, manifest_path)

    return completion_path.exists()


def _output_paths(run_path: Path) -> Tuple[Path, Path, Path]:
    train_path = run_path / "experiment" / "train"
    return (
        train_path / "DAG.npy",
        train_path / "gumbel_interv.npy",
        run_path / COMPLETION_FILENAME,
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completion_payload(run_path: Path) -> Dict[str, Any]:
    dag_path, target_path, _ = _output_paths(run_path)
    dag_stat = dag_path.stat()
    target_stat = target_path.stat()
    if target_stat.st_mtime_ns < dag_stat.st_mtime_ns:
        raise DCDIError(
            "DCDI intervention targets predate the final DAG; the run is incomplete"
        )
    return {
        "version": 1,
        "dag_sha256": _file_digest(dag_path),
        "target_sha256": _file_digest(target_path),
        "dag_mtime_ns": dag_stat.st_mtime_ns,
        "target_mtime_ns": target_stat.st_mtime_ns,
    }


def _write_completion_marker(run_path: Path) -> None:
    """Atomically mark a pair of validated final upstream outputs complete."""
    _, _, completion_path = _output_paths(run_path)
    payload = _completion_payload(run_path)
    temporary = completion_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, completion_path)


def _validate_completion_marker(run_path: Path) -> None:
    """Require a marker whose hashes match the completed output pair."""
    dag_path, target_path, completion_path = _output_paths(run_path)
    if not dag_path.is_file() or not target_path.is_file():
        raise DCDIError("completed DCDI cache is missing an output artifact")
    try:
        marker = json.loads(completion_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DCDIError(f"cannot read DCDI completion marker {completion_path}") from error
    if marker.get("version") != 1:
        raise DCDIError("unsupported DCDI completion marker version")
    if marker.get("target_mtime_ns", -1) < marker.get("dag_mtime_ns", 0):
        raise DCDIError("DCDI completion marker records stale intervention targets")
    if marker.get("dag_sha256") != _file_digest(dag_path):
        raise DCDIError("cached DCDI DAG does not match its completion marker")
    if marker.get("target_sha256") != _file_digest(target_path):
        raise DCDIError("cached DCDI targets do not match their completion marker")


def _quarantine_outputs(run_path: Path) -> Sequence[str]:
    """Move incomplete/corrupt adapter outputs aside so upstream can retry."""
    existing = [path for path in _output_paths(run_path) if path.exists()]
    if not existing:
        return []
    quarantine_path = run_path / "incomplete" / str(time.time_ns())
    quarantine_path.mkdir(parents=True, exist_ok=False)
    moved = []
    for source in existing:
        destination = quarantine_path / source.name
        os.replace(source, destination)
        moved.append(str(destination))
    return moved


def _diagnostic_suffix(log_path: Path, persistent: bool) -> str:
    """Keep persistent log paths useful and preserve tails from temporary runs."""
    if persistent:
        return f"; see {log_path}"
    try:
        lines = log_path.read_text(errors="replace").splitlines()[-30:]
    except OSError:
        return "; temporary DCDI log was unavailable"
    excerpt = "\n".join(lines)[-4000:]
    if not excerpt:
        return "; temporary DCDI log was empty"
    return f"; temporary DCDI log tail:\n{excerpt}"


def _is_acyclic(dag: np.ndarray) -> bool:
    """Check acyclicity without adding another runtime dependency."""
    indegree = dag.sum(axis=0).astype(int)
    ready = list(np.flatnonzero(indegree == 0))
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in np.flatnonzero(dag[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    return visited == len(dag)


def _read_outputs(
    experiment_path: Path, p: int, num_interventional: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate upstream artifacts in this repository's orientation."""
    train_path = experiment_path / "train"
    dag_path = train_path / "DAG.npy"
    target_path = train_path / "gumbel_interv.npy"
    if not dag_path.is_file() or not target_path.is_file():
        raise DCDIError(
            f"DCDI did not produce both {dag_path.name} and {target_path.name} in {train_path}"
        )

    raw_dag = np.asarray(np.load(dag_path), dtype=float)
    if raw_dag.shape != (p, p) or not np.isfinite(raw_dag).all():
        raise DCDIError(f"invalid DCDI DAG shape or values: {raw_dag.shape}")
    if not np.all(np.isclose(raw_dag, 0.0) | np.isclose(raw_dag, 1.0)):
        raise DCDIError("DCDI DAG.npy is not binary")
    dag = (raw_dag > 0.5).astype(int)
    if np.any(np.diag(dag)) or not _is_acyclic(dag):
        raise DCDIError("DCDI DAG.npy is not a directed acyclic graph")

    probabilities = np.asarray(np.load(target_path), dtype=float)
    expected_shape = (p, num_interventional + 1)
    if probabilities.shape != expected_shape or not np.isfinite(probabilities).all():
        raise DCDIError(
            f"invalid gumbel_interv.npy shape or values: expected {expected_shape}, "
            f"received {probabilities.shape}"
        )
    if np.any(probabilities < -1e-8) or np.any(probabilities > 1 + 1e-8):
        raise DCDIError("gumbel_interv.npy contains values outside [0, 1]")

    # Upstream probabilities encode retaining the observational mechanism, so
    # a probability below 0.5 means that variable is an intervention target.
    targets = (probabilities[:, 1:] < 0.5).T.astype(int)
    return dag, targets


def _read_fit_seconds(experiment_path: Path, fallback: float) -> Tuple[float, str]:
    """Read upstream training time, falling back to subprocess wall time."""
    timing_path = experiment_path / "train" / "timing.pkl"
    if timing_path.is_file():
        try:
            with timing_path.open("rb") as handle:
                value = float(pickle.load(handle))
            if np.isfinite(value) and value >= 0:
                return value, "upstream_timing"
        except (OSError, TypeError, ValueError, pickle.PickleError):
            pass
    return float(fallback), "subprocess_wall"


def optimization(
    data: Sequence[Any],
    graph_penalty: float,
    target_penalty: float,
    *,
    dcdi_path: Union[str, os.PathLike[str]] = DEFAULT_DCDI_PATH,
    normalize_data: bool = True,
    random_seed: int = 42,
    gpu: bool = False,
    float_precision: bool = False,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    mu_init: float = DEFAULT_MU_INIT,
    num_train_iter: int = 1_000_000,
    timeout: Optional[float] = None,
    artifact_dir: Optional[Union[str, os.PathLike[str]]] = None,
    python_executable: Union[str, os.PathLike[str]] = sys.executable,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Fit official DCDI-G in the perfect, unknown-target setting.

    Parameters
    ----------
    data:
        Observational sample followed by each interventional environment.  All
        arrays must use the same variable order.
    graph_penalty, target_penalty:
        DCDI's graph sparsity coefficient ``lambda`` and intervention-family
        sparsity coefficient ``lambda_R``.
    dcdi_path:
        Clean checkout of the official DCDI repository at :data:`DCDI_COMMIT`.
    normalize_data:
        Ask DCDI to standardize variables using its training split.
    random_seed:
        Shared NumPy/PyTorch and data-split seed used by upstream DCDI.
    gpu, float_precision:
        Pass through upstream ``--gpu`` and ``--float`` flags.  DCDI otherwise
        runs on CPU in double precision.
    hidden_dim, mu_init:
        Neural-network width and initial augmented-Lagrangian penalty. Their
        defaults match the faster scalability configuration in the DCDI
        supplement while retaining unknown intervention targets.
    num_train_iter:
        Maximum optimization steps; upstream early stopping can finish sooner.
    timeout:
        Optional subprocess wall-time limit in seconds.
    artifact_dir:
        Optional dedicated persistent run directory.  Matching completed runs
        are reused after their manifest and output shapes are validated.
    python_executable:
        Python interpreter containing DCDI's dependencies, normally the active
        interpreter or the project's ``.venv-dcdi`` interpreter.

    Returns
    -------
    dag, targets, fit_seconds, metadata
        ``dag`` is a binary ``p x p`` adjacency with rows as parents and columns
        as children.  ``targets`` is binary ``K x p`` with one row per
        non-observational environment.
    """
    arrays = _as_arrays(data)
    _validate_options(
        graph_penalty,
        target_penalty,
        random_seed,
        hidden_dim,
        mu_init,
        num_train_iter,
        timeout,
    )
    if normalize_data:
        pooled = np.concatenate(arrays, axis=0)
        if np.any(pooled.std(axis=0) == 0):
            raise ValueError("DCDI normalization requires every variable to vary")

    checkout, dirty = _validate_checkout(dcdi_path)
    interpreter = _resolve_python(python_executable)
    configuration = _configuration(
        arrays,
        graph_penalty,
        target_penalty,
        normalize_data,
        random_seed,
        gpu,
        float_precision,
        hidden_dim,
        mu_init,
        num_train_iter,
    )

    temporary: Optional[tempfile.TemporaryDirectory[str]] = None
    if artifact_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="dcdi_adapter_")
        run_path = Path(temporary.name)
    else:
        run_path = Path(artifact_dir).expanduser().resolve()

    run_path.mkdir(parents=True, exist_ok=True)
    lock_path = run_path / ".adapter.lock"
    lock_handle = lock_path.open("a")
    lock_acquired = False
    try:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
        except BlockingIOError as error:
            raise DCDIError(
                f"another DCDI process is already using {run_path}"
            ) from error
        marker_exists = _write_or_check_manifest(run_path, configuration)
        data_path = run_path / "data"
        experiment_path = run_path / "experiment"
        experiment_path.mkdir(parents=True, exist_ok=True)
        quarantined_outputs = []
        cached = False
        dag = None
        targets = None
        if marker_exists:
            try:
                _validate_completion_marker(run_path)
                dag, targets = _read_outputs(
                    experiment_path, arrays[0].shape[1], len(arrays) - 1
                )
                cached = True
            except (DCDIError, OSError, ValueError, EOFError):
                quarantined_outputs = list(_quarantine_outputs(run_path))
        elif any(path.exists() for path in _output_paths(run_path)):
            # Upstream periodically writes targets and writes the final DAG
            # before refreshing those targets.  Without our marker, neither a
            # single artifact nor an apparent pair is safe to reuse.
            quarantined_outputs = list(_quarantine_outputs(run_path))
        _stage_inputs(arrays, data_path)

        arguments = [
            "--train",
            "--data-path",
            str(data_path),
            "--num-vars",
            str(arrays[0].shape[1]),
            "--i-dataset",
            "1",
            "--exp-path",
            str(experiment_path),
            "--model",
            "DCDI-G",
            "--intervention",
            "--intervention-type",
            "perfect",
            "--intervention-knowledge",
            "unknown",
            "--reg-coeff",
            repr(float(graph_penalty)),
            "--coeff-interv-sparsity",
            repr(float(target_penalty)),
            "--random-seed",
            str(int(random_seed)),
            "--hid-dim",
            str(int(hidden_dim)),
            "--mu-init",
            repr(float(mu_init)),
            "--num-train-iter",
            str(int(num_train_iter)),
            "--no-w-adjs-log",
        ]
        if normalize_data:
            arguments.append("--normalize-data")
        if gpu:
            arguments.append("--gpu")
        if float_precision:
            arguments.append("--float")

        command = [
            str(interpreter),
            "-c",
            _UPSTREAM_LAUNCHER,
            str(checkout),
            str(checkout / "main.py"),
            *arguments,
        ]
        started = time.perf_counter()
        returncode = 0
        log_path = run_path / "dcdi.log"
        if not cached:
            try:
                with log_path.open("a") as log_handle:
                    completed = subprocess.run(
                        command,
                        cwd=checkout,
                        timeout=timeout,
                        check=False,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(
                    f"DCDI exceeded its {timeout:g}-second subprocess limit"
                    f"{_diagnostic_suffix(log_path, artifact_dir is not None)}"
                ) from error
            except OSError as error:
                raise DCDIError(
                    f"cannot launch DCDI with Python interpreter {python_executable}"
                ) from error
            returncode = int(completed.returncode)
        wall_seconds = time.perf_counter() - started

        if not cached:
            if returncode:
                raise DCDIError(
                    f"DCDI subprocess exited with status {returncode}"
                    f"{_diagnostic_suffix(log_path, artifact_dir is not None)}"
                )
            try:
                dag, targets = _read_outputs(
                    experiment_path, arrays[0].shape[1], len(arrays) - 1
                )
                _write_completion_marker(run_path)
            except (DCDIError, OSError, ValueError, EOFError) as error:
                raise DCDIError(
                    "DCDI exited without producing a valid completed result"
                    f"{_diagnostic_suffix(log_path, artifact_dir is not None)}"
                ) from error
        if dag is None or targets is None:
            raise DCDIError("DCDI completed without returning validated outputs")
        fit_seconds, timing_source = _read_fit_seconds(experiment_path, wall_seconds)

        metadata: Dict[str, Any] = {
            **configuration,
            "model": "DCDI-G",
            "intervention_type": "perfect",
            "intervention_knowledge": "unknown",
            "num_variables": int(arrays[0].shape[1]),
            "num_interventional_environments": len(arrays) - 1,
            "sample_sizes": [len(values) for values in arrays],
            "cached": cached,
            "fit_seconds_source": timing_source,
            "subprocess_wall_seconds": wall_seconds,
            "upstream_returncode": returncode,
            "log_path": str(log_path) if artifact_dir is not None else None,
            "quarantined_incomplete_outputs": quarantined_outputs,
            "upstream_reporting_metrics_disabled": True,
            "upstream_plot_padding_compatibility": True,
            "checkout_path": str(checkout),
            "checkout_dirty": dirty,
            "python_executable": str(interpreter),
            "artifact_dir": None if artifact_dir is None else str(run_path),
        }
        return dag, targets, fit_seconds, metadata
    finally:
        if lock_acquired:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
        if temporary is not None:
            temporary.cleanup()


fit = optimization


def _cli_path_component(value: float) -> str:
    return (
        f"{float(value):.12g}"
        .replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
    )


def main() -> int:
    """Run DCDI-G on one repository synthetic dataset and print diagnostics."""
    parser = argparse.ArgumentParser(
        description="Run official DCDI-G with perfect, unknown interventions."
    )
    parser.add_argument("--graph", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=6)
    parser.add_argument("--graph-penalty", type=float, default=0.1)
    parser.add_argument("--target-penalty", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--float", dest="float_precision", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--mu-init", type=float, default=DEFAULT_MU_INIT)
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "SyntheticData",
    )
    parser.add_argument("--dcdi-root", type=Path, default=DEFAULT_DCDI_PATH)
    parser.add_argument(
        "--dcdi-python",
        type=Path,
        default=PROJECT_ROOT / ".venv-dcdi" / "bin" / "python",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="persistent fit directory; defaults below experiment_results/dcdi_manual",
    )
    args = parser.parse_args()

    from MIP import _load
    from src.utils import compute_errors, interventional_cpdag

    artifact_dir = args.artifact_dir
    if artifact_dir is None:
        precision = "float32" if args.float_precision else "float64"
        device = "gpu" if args.gpu else "cpu"
        normalization = "normalized" if args.normalize else "raw"
        run_name = (
            f"lambda_{_cli_path_component(args.graph_penalty)}_"
            f"lambdaR_{_cli_path_component(args.target_penalty)}_"
            f"hid_{args.hidden_dim}_mu_{_cli_path_component(args.mu_init)}_"
            f"seed_{args.seed}_{normalization}_{device}_{precision}_"
            f"iter_{args.max_iterations}"
        )
        artifact_dir = (
            PROJECT_ROOT
            / "experiment_results"
            / "dcdi_manual"
            / f"graph_{args.graph}"
            / f"trial_{args.iteration:02d}"
            / run_name
        )

    data, _, true_dag, true_targets = _load(
        args.data_root, args.graph, args.iteration
    )
    dag, targets, fit_seconds, metadata = optimization(
        data,
        graph_penalty=args.graph_penalty,
        target_penalty=args.target_penalty,
        dcdi_path=args.dcdi_root,
        normalize_data=args.normalize,
        random_seed=args.seed,
        gpu=args.gpu,
        float_precision=args.float_precision,
        hidden_dim=args.hidden_dim,
        mu_init=args.mu_init,
        num_train_iter=args.max_iterations,
        timeout=args.time_limit,
        artifact_dir=artifact_dir,
        python_executable=args.dcdi_python,
    )
    estimated_icpdag = interventional_cpdag(dag, targets)
    errors = compute_errors(dag, targets, true_dag, true_targets)

    print("DAG:\n", dag)
    print("I-CPDAG:\n", estimated_icpdag)
    print("Targets:\n", targets)
    print("Runtime:", fit_seconds)
    print("Cached:", metadata["cached"])
    print("Artifacts:", metadata["artifact_dir"])
    print("d_cpdag, environment target error, FDP, TDP:", errors)
    return 0

__all__ = [
    "DCDI_COMMIT",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_DCDI_PATH",
    "DEFAULT_MU_INIT",
    "DCDIError",
    "fit",
    "main",
    "optimization",
]


if __name__ == "__main__":
    raise SystemExit(main())
