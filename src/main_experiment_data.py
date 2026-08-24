"""Reusable on-disk data contract for the main synthetic experiment.

The experiment-specific generator and every model-fitting job import this
module so the design, seed derivation, serialization, and validation rules
cannot drift apart.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shlex
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.DataGeneration import generate_experiment_instance


DATA_FORMAT_VERSION = 2
EXPERIMENT_NAME = "main_synthetic_experiment"
P_VALUES = (10, 20, 50)
EDGE_MULTIPLIERS = (1, 4)
NUM_INTERVENTIONAL_ENVIRONMENTS = 5
NUM_ENVIRONMENTS = 1 + NUM_INTERVENTIONAL_ENVIRONMENTS
N_OBSERVATIONAL = 1000
N_INTERVENTIONAL = 200
DEFAULT_NUM_REPLICATES = 10
DEFAULT_MASTER_SEED = 20260823
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "main_experiment"
MANIFEST_NAME = "manifest.json"
MAX_MASTER_SEED = int(np.iinfo(np.int64).max)

ARCHIVE_KEYS = {
    "format_version",
    "master_seed",
    "p",
    "edge_multiplier",
    "replicate",
    "seed",
    "data_sha256",
    "instance_sha256",
    "data_0",
    "data_1",
    "data_2",
    "data_3",
    "data_4",
    "data_5",
    "true_dag",
    "true_targets",
}


class MainExperimentDataError(RuntimeError):
    """Raised when persisted main-experiment data are missing or invalid."""


def _validate_master_seed(master_seed: int) -> int:
    if not isinstance(master_seed, (int, np.integer)):
        raise MainExperimentDataError("master seed must be an integer")
    master_seed = int(master_seed)
    if not 0 <= master_seed <= MAX_MASTER_SEED:
        raise MainExperimentDataError(
            f"master seed must be between 0 and {MAX_MASTER_SEED}"
        )
    return master_seed


def experiment_design(master_seed: int = DEFAULT_MASTER_SEED) -> dict:
    """Return the canonical, JSON-serializable main-experiment design."""
    master_seed = _validate_master_seed(master_seed)
    return {
        "p_values": list(P_VALUES),
        "edge_multipliers": list(EDGE_MULTIPLIERS),
        "num_replicates": DEFAULT_NUM_REPLICATES,
        "num_interventional_environments": NUM_INTERVENTIONAL_ENVIRONMENTS,
        "n_observational": N_OBSERVATIONAL,
        "n_interventional": N_INTERVENTIONAL,
        "master_seed": int(master_seed),
    }


def instance_seed(master_seed: int, p: int, edge_multiplier: int, replicate: int) -> int:
    """Derive the exact seed formerly used by the in-memory runner."""
    master_seed = _validate_master_seed(master_seed)
    sequence = np.random.SeedSequence(
        [int(master_seed), int(p), int(edge_multiplier), int(replicate)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def data_digest(data) -> str:
    """Hash observational-first sample arrays using the result-schema contract."""
    digest = hashlib.sha256()
    for values in data:
        array = np.asarray(values, dtype="<f8", order="C")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def instance_digest(
    data,
    true_dag,
    true_targets,
    *,
    master_seed: int,
    p: int,
    edge_multiplier: int,
    replicate: int,
    seed: int,
) -> str:
    """Hash samples, truth, and identifying metadata canonically."""
    digest = hashlib.sha256()
    metadata = {
        "data_format_version": DATA_FORMAT_VERSION,
        "edge_multiplier": int(edge_multiplier),
        "master_seed": int(master_seed),
        "p": int(p),
        "replicate": int(replicate),
        "seed": int(seed),
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for label, values, dtype in (
        *[(f"data_{index}", values, "<f8") for index, values in enumerate(data)],
        ("true_dag", true_dag, "u1"),
        ("true_targets", true_targets, "u1"),
    ):
        array = np.asarray(values, dtype=dtype, order="C")
        digest.update(label.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of one archive file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instance_relative_path(p: int, edge_multiplier: int, replicate: int) -> Path:
    return (
        Path(f"p_{int(p)}")
        / f"e_{int(edge_multiplier)}"
        / f"replicate_{int(replicate):03d}.npz"
    )


def expected_instance_keys():
    return [
        (p, edge_multiplier, replicate)
        for p in P_VALUES
        for edge_multiplier in EDGE_MULTIPLIERS
        for replicate in range(1, DEFAULT_NUM_REPLICATES + 1)
    ]


def generation_command(
    data_root: Path, master_seed: int = DEFAULT_MASTER_SEED
) -> str:
    return (
        "python3 experiments/generate_main_experiment_data.py "
        f"--output-dir {shlex.quote(str(data_root))} --seed {int(master_seed)}"
    )


def _validate_generation_target(data_root: Path, *, overwrite: bool) -> None:
    """Reject broad or unrelated overwrite targets before creating anything."""
    data_root = data_root.resolve()
    broad_targets = {
        Path(data_root.anchor),
        Path.home().resolve(),
        PROJECT_ROOT,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "experiments",
        Path.cwd().resolve(),
    }
    if data_root in broad_targets or data_root in PROJECT_ROOT.parents:
        raise MainExperimentDataError(
            f"refusing to use broad data-generation target {data_root}"
        )
    if not overwrite or not data_root.exists() or data_root == DEFAULT_DATA_ROOT:
        return

    manifest_path = data_root / MANIFEST_NAME
    try:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise MainExperimentDataError(
            f"refusing to overwrite unrelated directory {data_root}; choose the "
            "dedicated default path or a directory with a main-experiment manifest"
        ) from error
    if not isinstance(existing_manifest, dict):
        raise MainExperimentDataError(
            f"refusing to overwrite unrelated directory {data_root}"
        )
    try:
        existing_seed = _validate_master_seed(
            existing_manifest["design"]["master_seed"]
        )
        existing_design = existing_manifest["design"]
        expected_design = experiment_design(existing_seed)
        has_suite_signature = (
            existing_manifest.get("experiment") == EXPERIMENT_NAME
            and isinstance(existing_manifest.get("data_format_version"), int)
            and all(
                existing_design.get(field) == value
                for field, value in expected_design.items()
            )
            and isinstance(existing_manifest.get("instances"), list)
            and len(existing_manifest["instances"]) == len(expected_instance_keys())
            and isinstance(existing_manifest.get("provenance"), dict)
            and existing_manifest["provenance"].get("generator")
            == "experiments/generate_main_experiment_data.py"
        )
    except (KeyError, TypeError, MainExperimentDataError):
        has_suite_signature = False
    if not has_suite_signature:
        raise MainExperimentDataError(
            f"refusing to overwrite unrelated directory {data_root}"
        )


def _scalar(archive, name: str):
    value = np.asarray(archive[name])
    if value.shape != ():
        raise MainExperimentDataError(f"archive field {name!r} must be scalar")
    return value.item()


def _validate_binary(name: str, values: np.ndarray) -> None:
    if not np.isin(values, (0, 1)).all():
        raise MainExperimentDataError(f"{name} must contain only 0 and 1")


def _read_archive(path: Path, expected: dict | None = None):
    if not path.is_file():
        raise MainExperimentDataError(f"missing main-experiment archive: {path}")
    observed_file_digest = file_digest(path)
    if expected is not None and observed_file_digest != expected["file_sha256"]:
        raise MainExperimentDataError(f"file checksum mismatch for {path}")

    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != ARCHIVE_KEYS:
                missing = sorted(ARCHIVE_KEYS - keys)
                extra = sorted(keys - ARCHIVE_KEYS)
                raise MainExperimentDataError(
                    f"invalid keys in {path}; missing={missing}, extra={extra}"
                )
            metadata = {
                "format_version": int(_scalar(archive, "format_version")),
                "master_seed": int(_scalar(archive, "master_seed")),
                "p": int(_scalar(archive, "p")),
                "e": int(_scalar(archive, "edge_multiplier")),
                "replicate": int(_scalar(archive, "replicate")),
                "seed": int(_scalar(archive, "seed")),
                "stored_data_sha256": str(_scalar(archive, "data_sha256")),
                "stored_instance_sha256": str(
                    _scalar(archive, "instance_sha256")
                ),
            }
            data = [np.array(archive[f"data_{index}"], copy=True) for index in range(6)]
            true_dag = np.array(archive["true_dag"], copy=True)
            true_targets = np.array(archive["true_targets"], copy=True)
    except MainExperimentDataError:
        raise
    except Exception as error:
        raise MainExperimentDataError(f"could not read {path}: {error}") from error

    if metadata["format_version"] != DATA_FORMAT_VERSION:
        raise MainExperimentDataError(
            f"unsupported data format {metadata['format_version']} in {path}"
        )
    p = metadata["p"]
    expected_shapes = [(N_OBSERVATIONAL, p)] + [
        (N_INTERVENTIONAL, p)
    ] * NUM_INTERVENTIONAL_ENVIRONMENTS
    for index, (values, shape) in enumerate(zip(data, expected_shapes)):
        if values.shape != shape:
            raise MainExperimentDataError(
                f"data_{index} in {path} has shape {values.shape}; expected {shape}"
            )
        if values.dtype != np.dtype("float64") or not np.isfinite(values).all():
            raise MainExperimentDataError(
                f"data_{index} in {path} must be finite float64"
            )
    if true_dag.shape != (p, p):
        raise MainExperimentDataError(
            f"true_dag in {path} has shape {true_dag.shape}; expected {(p, p)}"
        )
    if true_dag.dtype != np.dtype("uint8"):
        raise MainExperimentDataError(f"true_dag in {path} must use uint8 storage")
    _validate_binary("true_dag", true_dag)
    if np.any(np.tril(true_dag, k=0)):
        raise MainExperimentDataError(f"true_dag in {path} is not strictly upper triangular")
    expected_target_shape = (NUM_INTERVENTIONAL_ENVIRONMENTS, p)
    if true_targets.shape != expected_target_shape:
        raise MainExperimentDataError(
            f"true_targets in {path} has shape {true_targets.shape}; "
            f"expected {expected_target_shape}"
        )
    if true_targets.dtype != np.dtype("uint8"):
        raise MainExperimentDataError(
            f"true_targets in {path} must use uint8 storage"
        )
    _validate_binary("true_targets", true_targets)
    target_counts = true_targets.sum(axis=1)
    if np.any(target_counts < 1) or np.any(target_counts > p // 2):
        raise MainExperimentDataError(
            f"true_targets in {path} violate the 1..floor(p/2) target rule"
        )

    derived_seed = instance_seed(
        metadata["master_seed"], p, metadata["e"], metadata["replicate"]
    )
    if metadata["seed"] != derived_seed:
        raise MainExperimentDataError(f"derived seed mismatch in {path}")
    observed_data_digest = data_digest(data)
    observed_instance_digest = instance_digest(
        data,
        true_dag,
        true_targets,
        master_seed=metadata["master_seed"],
        p=p,
        edge_multiplier=metadata["e"],
        replicate=metadata["replicate"],
        seed=metadata["seed"],
    )
    if observed_data_digest != metadata["stored_data_sha256"]:
        raise MainExperimentDataError(f"sample checksum mismatch in {path}")
    if observed_instance_digest != metadata["stored_instance_sha256"]:
        raise MainExperimentDataError(f"instance checksum mismatch in {path}")

    if expected is not None:
        comparisons = {
            "master_seed": metadata["master_seed"],
            "p": p,
            "e": metadata["e"],
            "replicate": metadata["replicate"],
            "seed": metadata["seed"],
            "data_sha256": observed_data_digest,
            "instance_sha256": observed_instance_digest,
            "realized_edges": int(true_dag.sum()),
        }
        for field, observed in comparisons.items():
            if observed != expected[field]:
                raise MainExperimentDataError(
                    f"manifest field {field!r} does not match {path}"
                )
        if [int(value) for value in target_counts] != expected["target_counts"]:
            raise MainExperimentDataError(
                f"manifest target counts do not match {path}"
            )

    info = {
        "p": p,
        "e": metadata["e"],
        "replicate": metadata["replicate"],
        "seed": metadata["seed"],
        "data_sha256": observed_data_digest,
        "instance_sha256": observed_instance_digest,
        "file_sha256": observed_file_digest,
        "realized_edges": int(true_dag.sum()),
        "target_counts": [int(value) for value in target_counts],
    }
    return data, true_dag.astype(int), true_targets.astype(int), info


def _manifest_index(manifest: dict) -> dict:
    index = {}
    for entry in manifest["instances"]:
        try:
            key = (int(entry["p"]), int(entry["e"]), int(entry["replicate"]))
        except (KeyError, TypeError, ValueError) as error:
            raise MainExperimentDataError("manifest contains an invalid instance key") from error
        if key in index:
            raise MainExperimentDataError(f"manifest contains duplicate instance {key}")
        expected_path = instance_relative_path(*key).as_posix()
        if entry.get("path") != expected_path:
            raise MainExperimentDataError(
                f"manifest path for {key} must be {expected_path!r}"
            )
        index[key] = entry
    return index


def load_main_experiment_manifest(
    data_root: Path,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> dict:
    """Load and validate the complete 60-instance manifest without reading arrays."""
    data_root = Path(data_root).expanduser().resolve()
    manifest_path = data_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise MainExperimentDataError(
            f"missing main-experiment manifest: {manifest_path}\n"
            f"Generate it first with: {generation_command(data_root, master_seed)}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise MainExperimentDataError(
            f"could not read main-experiment manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise MainExperimentDataError(
            f"main-experiment manifest must be a JSON object: {manifest_path}"
        )
    if manifest.get("data_format_version") != DATA_FORMAT_VERSION:
        raise MainExperimentDataError(
            f"{manifest_path} does not use data format {DATA_FORMAT_VERSION}"
        )
    if manifest.get("experiment") != EXPERIMENT_NAME:
        raise MainExperimentDataError(f"wrong experiment type in {manifest_path}")
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("numpy_version"), str)
        or not provenance["numpy_version"].strip()
    ):
        raise MainExperimentDataError(
            f"{manifest_path} does not record its generator NumPy version"
        )
    expected_design = experiment_design(master_seed)
    if manifest.get("design") != expected_design:
        raise MainExperimentDataError(
            f"{manifest_path} does not match the requested main-experiment design "
            f"or master seed {master_seed}"
        )
    if not isinstance(manifest.get("instances"), list):
        raise MainExperimentDataError(f"instances must be a list in {manifest_path}")
    index = _manifest_index(manifest)
    expected_keys = set(expected_instance_keys())
    if set(index) != expected_keys:
        missing = sorted(expected_keys - set(index))
        extra = sorted(set(index) - expected_keys)
        raise MainExperimentDataError(
            f"manifest instance grid is incomplete; missing={missing}, extra={extra}"
        )
    for key, entry in index.items():
        required = {
            "master_seed", "p", "e", "replicate", "seed", "path", "data_sha256",
            "instance_sha256", "file_sha256", "realized_edges", "target_counts",
        }
        if not required.issubset(entry):
            raise MainExperimentDataError(
                f"manifest entry {key} is missing {sorted(required - set(entry))}"
            )
        expected_seed = instance_seed(master_seed, *key)
        if int(entry["master_seed"]) != master_seed:
            raise MainExperimentDataError(f"manifest master seed mismatch for {key}")
        if int(entry["seed"]) != expected_seed:
            raise MainExperimentDataError(f"manifest seed mismatch for {key}")
        if not (data_root / entry["path"]).is_file():
            raise MainExperimentDataError(
                f"manifest archive is missing for {key}: {data_root / entry['path']}"
            )
    manifest["_index"] = index
    manifest["_data_root"] = data_root
    return manifest


def load_main_experiment_instance(
    data_root: Path,
    p: int,
    edge_multiplier: int,
    replicate: int,
    *,
    manifest: dict | None = None,
    master_seed: int = DEFAULT_MASTER_SEED,
):
    """Load and fully validate one persisted experimental instance."""
    if manifest is None:
        manifest = load_main_experiment_manifest(data_root, master_seed=master_seed)
    else:
        resolved_root = Path(data_root).expanduser().resolve()
        if manifest.get("_data_root") != resolved_root:
            raise MainExperimentDataError(
                "supplied manifest belongs to a different data root"
            )
        if manifest.get("design", {}).get("master_seed") != int(master_seed):
            raise MainExperimentDataError(
                "supplied manifest belongs to a different master seed"
            )
    key = (int(p), int(edge_multiplier), int(replicate))
    entry = manifest["_index"].get(key)
    if entry is None:
        raise MainExperimentDataError(f"main-experiment instance {key} is absent")
    path = manifest["_data_root"] / entry["path"]
    return _read_archive(path, expected=entry)


def validate_main_experiment_data(
    data_root: Path,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> dict:
    """Read and validate the manifest and all 60 archives."""
    manifest = load_main_experiment_manifest(data_root, master_seed=master_seed)
    for key in expected_instance_keys():
        load_main_experiment_instance(
            data_root, *key, manifest=manifest, master_seed=master_seed
        )
    return manifest


def _write_archive(
    path: Path,
    data,
    true_dag,
    true_targets,
    *,
    master_seed: int,
    p: int,
    edge_multiplier: int,
    replicate: int,
    seed: int,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_sha256 = data_digest(data)
    instance_sha256 = instance_digest(
        data,
        true_dag,
        true_targets,
        master_seed=master_seed,
        p=p,
        edge_multiplier=edge_multiplier,
        replicate=replicate,
        seed=seed,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                format_version=np.asarray(DATA_FORMAT_VERSION, dtype="<i8"),
                master_seed=np.asarray(master_seed, dtype="<i8"),
                p=np.asarray(p, dtype="<i8"),
                edge_multiplier=np.asarray(edge_multiplier, dtype="<i8"),
                replicate=np.asarray(replicate, dtype="<i8"),
                seed=np.asarray(seed, dtype="<u8"),
                data_sha256=np.asarray(data_sha256),
                instance_sha256=np.asarray(instance_sha256),
                **{
                    f"data_{index}": np.asarray(values, dtype="<f8", order="C")
                    for index, values in enumerate(data)
                },
                true_dag=np.asarray(true_dag, dtype="u1", order="C"),
                true_targets=np.asarray(true_targets, dtype="u1", order="C"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _, _, _, info = _read_archive(path)
    return {
        "master_seed": master_seed,
        "p": p,
        "e": edge_multiplier,
        "replicate": replicate,
        "seed": seed,
        "path": instance_relative_path(p, edge_multiplier, replicate).as_posix(),
        **{
            field: info[field]
            for field in (
                "data_sha256", "instance_sha256", "file_sha256",
                "realized_edges", "target_counts",
            )
        },
    }


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _generation_lock(data_root: Path):
    data_root = Path(data_root).expanduser().resolve()
    data_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = data_root.parent / f".{data_root.name}.generation.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MainExperimentDataError(
                f"another process is generating {data_root}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _recover_interrupted_backup(data_root: Path) -> None:
    """Restore the sole validated backup left by an interrupted directory swap."""
    if data_root.exists():
        return
    backups = sorted(data_root.parent.glob(f".{data_root.name}.backup-*"))
    if not backups:
        return
    if len(backups) != 1 or not backups[0].is_dir():
        raise MainExperimentDataError(
            f"cannot choose safely among interrupted backups for {data_root}: "
            f"{backups}"
        )
    backup = backups[0]
    try:
        raw_manifest = json.loads(
            (backup / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        backup_seed = int(raw_manifest["design"]["master_seed"])
        validate_main_experiment_data(backup, master_seed=backup_seed)
    except Exception as error:
        raise MainExperimentDataError(
            f"interrupted backup is not recoverable: {backup}"
        ) from error
    os.replace(backup, data_root)


def generate_main_experiment_data(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    overwrite: bool = False,
) -> dict:
    """Generate, verify, and install the complete main-experiment dataset suite."""
    data_root = Path(data_root).expanduser().resolve()
    _validate_generation_target(data_root, overwrite=overwrite)
    with _generation_lock(data_root):
        _recover_interrupted_backup(data_root)
        if data_root.exists() and not overwrite:
            try:
                return validate_main_experiment_data(
                    data_root, master_seed=master_seed
                )
            except MainExperimentDataError as error:
                raise MainExperimentDataError(
                    f"existing data root {data_root} is not a valid matching suite; "
                    "use --overwrite to replace it"
                ) from error

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{data_root.name}.staging-", dir=data_root.parent
            )
        )
        try:
            entries = []
            for p, edge_multiplier, replicate in expected_instance_keys():
                seed = instance_seed(master_seed, p, edge_multiplier, replicate)
                data, true_dag, true_targets = generate_experiment_instance(
                    p,
                    edge_multiplier,
                    num_interventional_environments=NUM_INTERVENTIONAL_ENVIRONMENTS,
                    observational_samples=N_OBSERVATIONAL,
                    interventional_samples=N_INTERVENTIONAL,
                    seed=seed,
                )
                path = staging / instance_relative_path(
                    p, edge_multiplier, replicate
                )
                entries.append(
                    _write_archive(
                        path,
                        data,
                        true_dag,
                        true_targets,
                        master_seed=master_seed,
                        p=p,
                        edge_multiplier=edge_multiplier,
                        replicate=replicate,
                        seed=seed,
                    )
                )
            manifest = {
                "data_format_version": DATA_FORMAT_VERSION,
                "experiment": EXPERIMENT_NAME,
                "design": experiment_design(master_seed),
                "provenance": {
                    "numpy_version": np.__version__,
                    "python_version": platform.python_version(),
                    "generator": "experiments/generate_main_experiment_data.py",
                },
                "instances": entries,
            }
            _write_manifest(staging / MANIFEST_NAME, manifest)
            validate_main_experiment_data(staging, master_seed=master_seed)

            backup = None
            if data_root.exists():
                backup = Path(
                    tempfile.mkdtemp(
                        prefix=f".{data_root.name}.backup-", dir=data_root.parent
                    )
                )
                backup.rmdir()
                os.replace(data_root, backup)
            try:
                os.replace(staging, data_root)
            except Exception:
                if backup is not None and backup.exists() and not data_root.exists():
                    os.replace(backup, data_root)
                raise
            if backup is not None:
                shutil.rmtree(backup)
            return validate_main_experiment_data(
                data_root, master_seed=master_seed
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
