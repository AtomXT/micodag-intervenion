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

from src.main_experiment_cli import (
    add_main_data_arguments,
    instance_summary,
    load_cli_instance,
    selected_method_seed,
    wall_time_limit,
)
from src.main_experiment_data import MainExperimentDataError


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DCDI_PATH = PROJECT_ROOT / "external" / "dcdi"
DCDI_SOURCE_URL = "https://github.com/slachapelle/dcdi"
DCDI_COMMIT = "594d328eae7795785e0d1a1138945e28a4fec037"
COMPLETION_FILENAME = "adapter_complete.json"
DEFAULT_HIDDEN_DIM = 8
DEFAULT_MU_INIT = 1e-2
RUNTIME_PROBE_TIMEOUT = 60.0
DCDI_PLOT_COMPATIBILITY_VERSION = "matplotlib-padding-to-pad_inches-v1"
_DCDI_PLOT_COMPATIBILITY = '''"""Plot-only compatibility for the unmodified DCDI author source."""
from matplotlib.figure import Figure

_original_savefig = Figure.savefig

def _savefig_with_legacy_padding(self, *args, **kwargs):
    if "padding" in kwargs:
        kwargs.setdefault("pad_inches", kwargs.pop("padding"))
    return _original_savefig(self, *args, **kwargs)

Figure.savefig = _savefig_with_legacy_padding
'''


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
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = bool(status.strip())
    if dirty:
        raise DCDIError(
            "DCDI checkout has local modifications or untracked files; use the clean "
            "pinned author checkout"
        )
    compiled_overrides = [
        path
        for suffix in ("*.so", "*.pyd", "*.dylib")
        for path in checkout.rglob(suffix)
    ]
    if compiled_overrides:
        raise DCDIError(
            "DCDI checkout contains an ignored compiled module that could override "
            f"the author source: {compiled_overrides[0]}"
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


def _validate_official_runtime(
    interpreter: Path, rscript_executable: Union[str, os.PathLike[str]]
) -> Dict[str, Any]:
    """Check dependencies required by the authors' unmodified entry point.

    This checks that the upstream imports and R packages are available and
    records their actual versions. It deliberately does not enforce package
    version equality. The final upstream report calls ``pcalg`` and ``SID``.
    """
    rscript_value = os.fspath(rscript_executable)
    rscript_candidate = Path(rscript_value).expanduser()
    if rscript_candidate.is_absolute() or rscript_candidate.parent != Path("."):
        rscript = str(Path(os.path.abspath(rscript_candidate)))
    else:
        rscript = shutil.which(rscript_value) or ""
    if not rscript or not Path(rscript).is_file() or not os.access(rscript, os.X_OK):
        raise DCDIError(
            f"Rscript executable not found: {rscript_value}; the unmodified DCDI "
            "reporting step requires R with the pcalg and SID packages"
        )
    if Path(rscript).name != "Rscript":
        raise DCDIError(
            "the selected R executable must be named Rscript because upstream CDT "
            "invokes that program name directly"
        )
    probe_environment = os.environ.copy()
    probe_environment["PATH"] = os.pathsep.join(
        [str(Path(rscript).parent), probe_environment.get("PATH", "")]
    )

    runtime_probe = (
        "import hashlib,importlib,importlib.metadata as m,json,platform,re,sys;"
        "modules={'numpy':'numpy','scipy':'scipy','pandas':'pandas',"
        "'matplotlib':'matplotlib','seaborn':'seaborn','networkx':'networkx',"
        "'scikit-learn':'sklearn','torch':'torch','cdt':'cdt'};"
        "_=[importlib.import_module(v) for v in modules.values()];"
        "env={re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version "
        "for d in m.distributions() if d.metadata.get('Name')};"
        "env=dict(sorted(env.items()));"
        "frozen=json.dumps(env,sort_keys=True,separators=(',',':'));"
        "print(json.dumps({'python':platform.python_version(),"
        "'executable':sys.executable,'packages':{n:m.version(n) for n in modules},"
        "'resolved_environment':{'distributions':env,"
        "'sha256':hashlib.sha256(frozen.encode()).hexdigest()}},"
        "sort_keys=True))"
    )
    try:
        probe_output = subprocess.run(
            [str(interpreter), "-c", runtime_probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=RUNTIME_PROBE_TIMEOUT,
            env=probe_environment,
        )
        python_runtime = json.loads(probe_output.stdout.splitlines()[-1])
    except subprocess.TimeoutExpired as error:
        raise DCDIError(
            f"DCDI Python runtime check exceeded {RUNTIME_PROBE_TIMEOUT:g} seconds"
        ) from error
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or ""
        raise DCDIError(
            "the DCDI interpreter is missing a required official-runtime package; "
            "install requirements-dcdi.txt in the active environment"
            f"{': ' + detail.strip()[-2000:] if detail.strip() else ''}"
        ) from error
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as error:
        raise DCDIError("could not identify the active DCDI Python runtime") from error
    r_probe = (
        'pkgs <- c("pcalg", "SID"); '
        'missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; '
        'if (length(missing)) { '
        'write(paste("missing R packages:", paste(missing, collapse=", ")), stderr()); '
        'quit(status=2L) }; '
        'ip <- installed.packages(); '
        'entries <- sort(paste(ip[,"Package"], ip[,"Version"], sep="==")); '
        'cat(as.character(packageVersion("pcalg")), "\\n", '
        'as.character(packageVersion("SID")), "\\n", '
        'R.version.string, "\\n", '
        'paste(entries, collapse="\\n"), sep="")'
    )
    try:
        versions = subprocess.run(
            [rscript, "--vanilla", "-e", r_probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=RUNTIME_PROBE_TIMEOUT,
        ).stdout.splitlines()
    except subprocess.TimeoutExpired as error:
        raise DCDIError(
            f"DCDI R runtime check exceeded {RUNTIME_PROBE_TIMEOUT:g} seconds"
        ) from error
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or ""
        raise DCDIError(
            "the unmodified DCDI reporting step requires the R packages pcalg "
            f"and SID{': ' + detail.strip() if detail.strip() else ''}"
        ) from error
    if len(versions) < 3:
        raise DCDIError("could not determine the installed pcalg and SID versions")
    r_distributions = {}
    for entry in versions[3:]:
        if "==" in entry:
            name, version = entry.split("==", 1)
            r_distributions[name] = version
    r_distributions = dict(sorted(r_distributions.items()))
    r_payload = json.dumps(
        r_distributions, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "python_version": python_runtime["python"],
        "python_executable": python_runtime["executable"],
        "python_packages": python_runtime["packages"],
        "resolved_environment": python_runtime["resolved_environment"],
        # Keep the selected symlink path. Upstream CDT launches the literal
        # command name ``Rscript``, so PATH must contain the selected path's
        # directory rather than only the resolved target's directory.
        "rscript": str(Path(os.path.abspath(rscript))),
        "rscript_realpath": str(Path(rscript).resolve()),
        "pcalg_version": versions[0],
        "sid_version": versions[1],
        "r_version": versions[2],
        "resolved_r_environment": {
            "distributions": r_distributions,
            "sha256": hashlib.sha256(r_payload).hexdigest(),
        },
    }


def validate_official_install(
    dcdi_path: Union[str, os.PathLike[str]] = DEFAULT_DCDI_PATH,
    *,
    rscript: Union[str, os.PathLike[str]] = "Rscript",
) -> Dict[str, Any]:
    """Validate and describe the exact author checkout and active runtime."""
    checkout, _ = _validate_checkout(dcdi_path)
    interpreter = _resolve_python(sys.executable)
    runtime = _validate_official_runtime(interpreter, rscript)
    return {
        "source_url": DCDI_SOURCE_URL,
        "source_commit": DCDI_COMMIT,
        "checkout": str(checkout),
        **runtime,
    }


def _official_command(
    interpreter: Path, checkout: Path, arguments: Sequence[str]
) -> list[str]:
    """Build a direct call to the unmodified author entry point."""
    return [str(interpreter), str(checkout / "main.py"), *arguments]


def _prepare_runtime_compatibility(run_path: Path) -> Path:
    """Stage a child-only Matplotlib keyword adapter outside author source."""
    compatibility_path = run_path / "runtime_compatibility"
    compatibility_path.mkdir(parents=True, exist_ok=True)
    sitecustomize = compatibility_path / "sitecustomize.py"
    expected = _DCDI_PLOT_COMPATIBILITY
    if sitecustomize.is_file() and sitecustomize.read_text() == expected:
        return compatibility_path
    temporary = sitecustomize.with_suffix(".py.tmp")
    temporary.write_text(expected)
    os.replace(temporary, sitecustomize)
    return compatibility_path


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
    official_runtime: Dict[str, Any],
    dependency_lock_sha256: Optional[str],
) -> Dict[str, Any]:
    """Build the reproducibility manifest for a persistent fit directory."""
    return {
        "adapter": "DCDI-G-perfect-unknown",
        "official_source_url": DCDI_SOURCE_URL,
        "dcdi_commit": DCDI_COMMIT,
        "official_entrypoint": "main.py",
        "official_execution": "direct_author_entrypoint_plot_compatibility_v2",
        "runtime_compatibility": {
            "version": DCDI_PLOT_COMPATIBILITY_VERSION,
            "scope": "Matplotlib savefig padding keyword only",
            "sha256": hashlib.sha256(
                _DCDI_PLOT_COMPATIBILITY.encode()
            ).hexdigest(),
        },
        "official_runtime": dict(official_runtime),
        "dependency_lock_sha256": dependency_lock_sha256,
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
    rscript: Union[str, os.PathLike[str]] = "Rscript",
    dependency_lock_sha256: Optional[str] = None,
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
    rscript:
        Rscript executable used by the unmodified upstream reporting step.
    dependency_lock_sha256:
        Optional hash of the caller's dependency lock. It becomes part of a
        persistent artifact's identity, preventing reuse under another lock.
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
    interpreter = _resolve_python(sys.executable)
    official_runtime = _validate_official_runtime(interpreter, rscript)
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
        official_runtime,
        dependency_lock_sha256,
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
        compatibility_path = _prepare_runtime_compatibility(run_path)

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

        command = _official_command(interpreter, checkout, arguments)
        upstream_environment = os.environ.copy()
        rscript_directory = str(Path(official_runtime["rscript"]).parent)
        upstream_environment["PATH"] = os.pathsep.join(
            [rscript_directory, upstream_environment.get("PATH", "")]
        )
        # Ignore any local __pycache__ left by prior inspection and compile the
        # clean tracked source into this run's isolated cache directory.
        upstream_environment["PYTHONPYCACHEPREFIX"] = str(
            run_path / "python_cache"
        )
        # Python imports this child-only sitecustomize before the author entry
        # point. It only translates the removed Matplotlib plotting keyword;
        # the DCDI checkout and all estimation code remain untouched.
        upstream_environment["PYTHONPATH"] = str(compatibility_path)
        upstream_environment["MPLCONFIGDIR"] = str(run_path / "matplotlib_cache")
        started = time.perf_counter()
        returncode = 0
        log_path = run_path / "dcdi.log"
        if not cached:
            try:
                with log_path.open("a") as log_handle:
                    completed = subprocess.run(
                        command,
                        cwd=checkout,
                        env=upstream_environment,
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
                    f"cannot launch DCDI with the active Python interpreter {interpreter}"
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
            "official_source_modified": False,
            "local_algorithm_implementation": False,
            "official_execution": "direct_author_entrypoint_plot_compatibility_v2",
            "upstream_argv": command,
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


def main() -> int:
    """Run DCDI-G on one persisted main-experiment instance."""
    parser = argparse.ArgumentParser(
        description="Run official DCDI-G with perfect, unknown interventions."
    )
    add_main_data_arguments(parser, method_seed=True)
    parser.add_argument("--graph-penalty", type=float, default=0.1)
    parser.add_argument("--target-penalty", type=float, default=1e-3)
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--float", dest="float_precision", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--mu-init", type=float, default=DEFAULT_MU_INIT)
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--metric-time-limit", type=float, default=3600)
    parser.add_argument("--dcdi-root", type=Path, default=DEFAULT_DCDI_PATH)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="optional persistent fit directory; omitted runs in cleaned temporary storage",
    )
    args = parser.parse_args()
    if args.metric_time_limit <= 0 or not np.isfinite(args.metric_time_limit):
        parser.error("metric time limit must be positive and finite")
    from src.utils import compute_errors, interventional_cpdag
    try:
        data, true_dag, true_targets, info = load_cli_instance(args)
    except MainExperimentDataError as error:
        parser.error(str(error))
    method_seed = selected_method_seed(args, info)
    artifact_dir = (
        None if args.artifact_dir is None else args.artifact_dir.expanduser().resolve()
    )
    dag, targets, fit_seconds, metadata = optimization(
        data,
        graph_penalty=args.graph_penalty,
        target_penalty=args.target_penalty,
        dcdi_path=args.dcdi_root,
        normalize_data=args.normalize,
        random_seed=method_seed,
        gpu=args.gpu,
        float_precision=args.float_precision,
        hidden_dim=args.hidden_dim,
        mu_init=args.mu_init,
        num_train_iter=args.max_iterations,
        timeout=args.time_limit,
        artifact_dir=artifact_dir,
        rscript=args.rscript,
    )
    with wall_time_limit(args.metric_time_limit, "metric evaluation"):
        estimated_icpdag = interventional_cpdag(dag, targets)
        errors = compute_errors(dag, targets, true_dag, true_targets)

    print("Instance:", instance_summary(args, data, true_dag, true_targets, info))
    print(
        "Configuration:",
        {"graph_penalty": args.graph_penalty,
         "target_penalty": args.target_penalty, "method_seed": method_seed,
         "normalize": args.normalize, "fit_time_limit": args.time_limit,
         "metric_time_limit": args.metric_time_limit,
         "artifact_dir": metadata["artifact_dir"]},
    )
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
    "DCDI_SOURCE_URL",
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
