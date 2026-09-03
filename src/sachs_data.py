"""Pinned data contract for the Sachs et al. (2005) UT-IGSP experiment.

The six source files are taken directly from the UT-IGSP authors' repository
at one immutable commit.  They already contain natural-log-transformed
measurements.  This module deliberately performs no centering, scaling,
standardization, or other numerical transformation.

``load_sachs_data`` returns six observational-first sample arrays, the
11-by-11 consensus DAG, a 5-by-11 intended-target matrix corresponding to the
five interventional arrays, and a provenance/validation information mapping.
Archives are always opened with ``allow_pickle=False``.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import ssl
import shutil
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "sachs"
MANIFEST_NAME = "manifest.json"
ARCHIVE_NAME = "sachs.npz"
RAW_DIRECTORY_NAME = "raw"

DATASET_ID = "sachs-utigsp-uai2020-real-data"
DATA_FORMAT_VERSION = 1
GENERATOR = "experiments/prepare_sachs_data.py"
SOURCE_REPOSITORY = "https://github.com/csquires/utigsp"
SOURCE_COMMIT = "dda019b4bd2708fce4d6e383f2c3d37297759bd3"
SOURCE_DIRECTORY = "real_data_analysis/sachs/data"
SOURCE_COMMIT_URL = SOURCE_REPOSITORY + "/tree/" + SOURCE_COMMIT

SACHS_NODE_NAMES = (
    "Raf",
    "Mek",
    "PLCg",
    "PIP2",
    "PIP3",
    "Erk",
    "Akt",
    "PKA",
    "PKC",
    "p38",
    "JNK",
)

# Observational first, matching the environment order used by the UT-IGSP
# paper's Sachs analysis.  Target indices refer to SACHS_NODE_NAMES.
SACHS_ENVIRONMENTS = (
    {
        "name": "observational",
        "filename": "iv=.txt",
        "encoded_filename": "iv%3D.txt",
        "rows": 1755,
        "target_index": None,
        "target_name": None,
        "sha256": "5b1fcdd33e3e3307af483b3781ab20599984ddc7c834fc3ff52131b5041635c3",
    },
    {
        "name": "Akt",
        "filename": "iv=6.txt",
        "encoded_filename": "iv%3D6.txt",
        "rows": 911,
        "target_index": 6,
        "target_name": "Akt",
        "sha256": "7e28807243a93c629c4918b3b5afb40bc0a2ef2cb7ed0e7533dbaa8feb752ae0",
    },
    {
        "name": "PKC",
        "filename": "iv=8.txt",
        "encoded_filename": "iv%3D8.txt",
        "rows": 723,
        "target_index": 8,
        "target_name": "PKC",
        "sha256": "e06533efe08799611f86125a4a89aa3af1d6d22ee31c11500f2950ea6d91cdd1",
    },
    {
        "name": "PIP2",
        "filename": "iv=3.txt",
        "encoded_filename": "iv%3D3.txt",
        "rows": 810,
        "target_index": 3,
        "target_name": "PIP2",
        "sha256": "ec11214d72fb20b6d6532916c6b727e3f77e59b785f3fce31ab503f3003fcdd4",
    },
    {
        "name": "Mek",
        "filename": "iv=1.txt",
        "encoded_filename": "iv%3D1.txt",
        "rows": 799,
        "target_index": 1,
        "target_name": "Mek",
        "sha256": "3a10450da527a7398f9d6a760afb7ab12f72ed87be8beadbbbf727c3281c83ac",
    },
    {
        "name": "PIP3",
        "filename": "iv=4.txt",
        "encoded_filename": "iv%3D4.txt",
        "rows": 848,
        "target_index": 4,
        "target_name": "PIP3",
        "sha256": "843bbab99d212dc7d48334cf9c502fe38814c4a4a001d576494e1b2674ced126",
    },
)

SACHS_SAMPLE_COUNTS = tuple(int(env["rows"]) for env in SACHS_ENVIRONMENTS)
SACHS_TOTAL_SAMPLES = sum(SACHS_SAMPLE_COUNTS)

# The exact consensus DAG in real_data_analysis/sachs/sachs_meta.py at the
# pinned source commit.  Each pair is (parent index, child index).
SACHS_CONSENSUS_ARCS = (
    (0, 1),
    (1, 5),
    (2, 3),
    (2, 8),
    (3, 8),
    (4, 2),
    (4, 3),
    (4, 6),
    (7, 0),
    (7, 1),
    (7, 5),
    (7, 6),
    (7, 9),
    (7, 10),
    (8, 0),
    (8, 1),
    (8, 9),
    (8, 10),
)

ARCHIVE_KEYS = {
    "data_format_version",
    "dataset_id",
    "samples",
    "offsets",
    "true_dag",
    "intended_targets",
    "node_names",
    "environment_names",
    "raw_filenames",
    "data_sha256",
    "raw_bundle_sha256",
}


class SachsDataError(RuntimeError):
    """Raised when the pinned Sachs package is absent, unsafe, or invalid."""


def source_url(environment: Mapping[str, object]) -> str:
    """Return the immutable raw URL for one source environment.

    ``encoded_filename`` is intentionally stored explicitly so every equals
    sign remains percent-encoded in generated URLs.
    """

    return (
        "https://raw.githubusercontent.com/csquires/utigsp/"
        + SOURCE_COMMIT
        + "/"
        + SOURCE_DIRECTORY
        + "/"
        + str(environment["encoded_filename"])
    )


def file_sha256(path: Union[Path, str]) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_sha256(data: Sequence[np.ndarray]) -> str:
    """Hash observational-first samples using a platform-stable encoding."""

    digest = hashlib.sha256()
    digest.update(b"sachs-utigsp-samples-v1\0")
    for index, values in enumerate(data):
        array = np.asarray(values, dtype="<f8", order="C")
        digest.update(("data_%d" % index).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def raw_bundle_sha256(raw_directory: Union[Path, str]) -> str:
    """Hash the six named raw files in canonical environment order."""

    raw_directory = Path(raw_directory)
    digest = hashlib.sha256()
    digest.update(b"sachs-utigsp-raw-bundle-v1\0")
    for environment in SACHS_ENVIRONMENTS:
        filename = str(environment["filename"])
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        path = raw_directory / filename
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def consensus_dag() -> np.ndarray:
    """Return a fresh uint8 adjacency matrix for the 18-edge consensus DAG."""

    dag = np.zeros((len(SACHS_NODE_NAMES), len(SACHS_NODE_NAMES)), dtype=np.uint8)
    for parent, child in SACHS_CONSENSUS_ARCS:
        dag[parent, child] = 1
    return dag


def intended_target_matrix() -> np.ndarray:
    """Return intended targets for the five interventional environments."""

    targets = np.zeros(
        (len(SACHS_ENVIRONMENTS) - 1, len(SACHS_NODE_NAMES)), dtype=np.uint8
    )
    for row, environment in enumerate(SACHS_ENVIRONMENTS[1:]):
        targets[row, int(environment["target_index"])] = 1
    return targets


def _read_raw_file(path: Path, environment: Mapping[str, object]) -> np.ndarray:
    if not path.is_file():
        raise SachsDataError("missing Sachs source file: %s" % path)
    observed_sha256 = file_sha256(path)
    expected_sha256 = str(environment["sha256"])
    if observed_sha256 != expected_sha256:
        raise SachsDataError(
            "source checksum mismatch for %s: expected %s, observed %s"
            % (path, expected_sha256, observed_sha256)
        )

    rows = []  # type: List[List[float]]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as error:
                raise SachsDataError("empty Sachs source file: %s" % path) from error
            if tuple(header) != SACHS_NODE_NAMES:
                raise SachsDataError(
                    "wrong header in %s: expected %r, observed %r"
                    % (path, SACHS_NODE_NAMES, tuple(header))
                )
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(SACHS_NODE_NAMES):
                    raise SachsDataError(
                        "%s:%d has %d columns; expected %d"
                        % (path, line_number, len(row), len(SACHS_NODE_NAMES))
                    )
                try:
                    rows.append([float(value) for value in row])
                except ValueError as error:
                    raise SachsDataError(
                        "%s:%d contains a non-numeric value" % (path, line_number)
                    ) from error
    except (OSError, UnicodeError) as error:
        raise SachsDataError("could not read Sachs source file %s: %s" % (path, error)) from error

    values = np.asarray(rows, dtype=np.float64, order="C")
    expected_shape = (int(environment["rows"]), len(SACHS_NODE_NAMES))
    if values.shape != expected_shape:
        raise SachsDataError(
            "%s has shape %r; expected %r" % (path, values.shape, expected_shape)
        )
    if not np.isfinite(values).all():
        raise SachsDataError("%s contains NaN or infinite values" % path)
    return values


def _read_raw_directory(raw_directory: Path) -> List[np.ndarray]:
    return [
        _read_raw_file(raw_directory / str(environment["filename"]), environment)
        for environment in SACHS_ENVIRONMENTS
    ]


def _validate_binary(name: str, values: np.ndarray) -> None:
    if not np.isin(values, (0, 1)).all():
        raise SachsDataError("%s must contain only zero and one" % name)


def _is_acyclic(adjacency: np.ndarray) -> bool:
    indegrees = adjacency.sum(axis=0).astype(int)
    ready = [int(index) for index in np.flatnonzero(indegrees == 0)]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            child_index = int(child)
            indegrees[child_index] -= 1
            if indegrees[child_index] == 0:
                ready.append(child_index)
    return visited == adjacency.shape[0]


def _source_manifest_entries() -> List[Dict[str, object]]:
    entries = []  # type: List[Dict[str, object]]
    for environment in SACHS_ENVIRONMENTS:
        entries.append(
            {
                "environment": str(environment["name"]),
                "filename": str(environment["filename"]),
                "raw_path": RAW_DIRECTORY_NAME + "/" + str(environment["filename"]),
                "url": source_url(environment),
                "sha256": str(environment["sha256"]),
                "rows": int(environment["rows"]),
                "intended_target_index": environment["target_index"],
                "intended_target": environment["target_name"],
            }
        )
    return entries


def _environment_manifest_entries() -> List[Dict[str, object]]:
    entries = []  # type: List[Dict[str, object]]
    for index, environment in enumerate(SACHS_ENVIRONMENTS):
        entries.append(
            {
                "index": index,
                "name": str(environment["name"]),
                "rows": int(environment["rows"]),
                "source_file": str(environment["filename"]),
                "intended_target_index": environment["target_index"],
                "intended_target": environment["target_name"],
            }
        )
    return entries


def _consensus_manifest() -> Dict[str, object]:
    return {
        "edge_count": len(SACHS_CONSENSUS_ARCS),
        "arcs_by_index": [list(arc) for arc in SACHS_CONSENSUS_ARCS],
        "arcs_by_name": [
            [SACHS_NODE_NAMES[parent], SACHS_NODE_NAMES[child]]
            for parent, child in SACHS_CONSENSUS_ARCS
        ],
        "upstream_definition": (
            SOURCE_COMMIT_URL + "/real_data_analysis/sachs/sachs_meta.py"
        ),
    }


def _build_manifest(
    *,
    archive_sha256: str,
    samples_sha256: str,
    raw_sha256: str,
    retrieval_mode: str,
) -> Dict[str, object]:
    return {
        "data_format_version": DATA_FORMAT_VERSION,
        "dataset": DATASET_ID,
        "generator": GENERATOR,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "commit_url": SOURCE_COMMIT_URL,
            "directory": SOURCE_DIRECTORY,
            "retrieval_mode": retrieval_mode,
            "files": _source_manifest_entries(),
        },
        "citations": [
            {
                "work": "Sachs et al. (2005), Science 308(5721), 523-529",
                "doi": "10.1126/science.1105809",
                "url": "https://doi.org/10.1126/science.1105809",
            },
            {
                "work": "Squires, Wang, and Uhler (2020), UAI / PMLR 124",
                "url": "https://proceedings.mlr.press/v124/squires20a.html",
            },
            {
                "work": "UT-IGSP authors' Sachs reproduction files",
                "url": SOURCE_COMMIT_URL + "/real_data_analysis/sachs",
            },
        ],
        "transformation": {
            "source_state": "Values in the pinned files are already natural-log transformed.",
            "preparation": "Parsed as float64 and concatenated without changing values.",
            "standardized": False,
            "centered": False,
            "scaled": False,
        },
        "node_names": list(SACHS_NODE_NAMES),
        "environments": _environment_manifest_entries(),
        "sample_counts": list(SACHS_SAMPLE_COUNTS),
        "total_samples": SACHS_TOTAL_SAMPLES,
        "num_variables": len(SACHS_NODE_NAMES),
        "consensus_dag": _consensus_manifest(),
        "archive": {
            "path": ARCHIVE_NAME,
            "format": "NumPy compressed NPZ (pickle-free)",
            "sha256": archive_sha256,
        },
        "digests": {
            "data_sha256": samples_sha256,
            "raw_bundle_sha256": raw_sha256,
            "archive_sha256": archive_sha256,
        },
    }


def _load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise SachsDataError("missing Sachs manifest: %s" % path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise SachsDataError("could not read Sachs manifest %s: %s" % (path, error)) from error
    if not isinstance(value, dict):
        raise SachsDataError("Sachs manifest must be a JSON object: %s" % path)
    return value


def _validate_manifest(manifest: Mapping[str, object], manifest_path: Path) -> None:
    prefix = "invalid Sachs manifest %s: " % manifest_path
    if manifest.get("data_format_version") != DATA_FORMAT_VERSION:
        raise SachsDataError(prefix + "unsupported data format")
    if manifest.get("dataset") != DATASET_ID:
        raise SachsDataError(prefix + "wrong dataset identifier")
    if manifest.get("generator") != GENERATOR:
        raise SachsDataError(prefix + "wrong generator")
    if manifest.get("node_names") != list(SACHS_NODE_NAMES):
        raise SachsDataError(prefix + "node order does not match the pinned contract")
    if manifest.get("sample_counts") != list(SACHS_SAMPLE_COUNTS):
        raise SachsDataError(prefix + "sample counts do not match the pinned contract")
    if manifest.get("total_samples") != SACHS_TOTAL_SAMPLES:
        raise SachsDataError(prefix + "total sample count is not 5846")
    if manifest.get("num_variables") != len(SACHS_NODE_NAMES):
        raise SachsDataError(prefix + "variable count is not 11")
    if manifest.get("environments") != _environment_manifest_entries():
        raise SachsDataError(prefix + "environment order or intended targets changed")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise SachsDataError(prefix + "source must be an object")
    if source.get("repository") != SOURCE_REPOSITORY:
        raise SachsDataError(prefix + "wrong source repository")
    if source.get("commit") != SOURCE_COMMIT:
        raise SachsDataError(prefix + "wrong source commit")
    if source.get("directory") != SOURCE_DIRECTORY:
        raise SachsDataError(prefix + "wrong source directory")
    if source.get("files") != _source_manifest_entries():
        raise SachsDataError(prefix + "source file metadata or checksums changed")
    if source.get("retrieval_mode") not in ("https_download", "offline_source_dir"):
        raise SachsDataError(prefix + "unknown retrieval mode")

    transformation = manifest.get("transformation")
    if not isinstance(transformation, dict):
        raise SachsDataError(prefix + "missing transformation declaration")
    if any(transformation.get(field) is not False for field in ("standardized", "centered", "scaled")):
        raise SachsDataError(prefix + "data must not be standardized, centered, or scaled")

    consensus = manifest.get("consensus_dag")
    if consensus != _consensus_manifest():
        raise SachsDataError(prefix + "consensus DAG changed")
    archive = manifest.get("archive")
    digests = manifest.get("digests")
    if not isinstance(archive, dict) or archive.get("path") != ARCHIVE_NAME:
        raise SachsDataError(prefix + "archive metadata is invalid")
    if not isinstance(digests, dict):
        raise SachsDataError(prefix + "digests must be an object")
    for field in ("data_sha256", "raw_bundle_sha256", "archive_sha256"):
        value = digests.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise SachsDataError(prefix + "%s is not a SHA-256 digest" % field)
        try:
            int(value, 16)
        except ValueError as error:
            raise SachsDataError(prefix + "%s is not hexadecimal" % field) from error
    if archive.get("sha256") != digests.get("archive_sha256"):
        raise SachsDataError(prefix + "archive digest fields disagree")


def _scalar_string(archive: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(archive[name])
    if value.shape != () or value.dtype.kind not in ("U", "S"):
        raise SachsDataError("archive field %r must be a string scalar" % name)
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("ascii")
    return str(item)


def _read_archive(
    archive_path: Path, expected_archive_sha256: str
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Dict[str, str]]:
    if not archive_path.is_file():
        raise SachsDataError("missing Sachs archive: %s" % archive_path)
    observed_archive_sha256 = file_sha256(archive_path)
    if observed_archive_sha256 != expected_archive_sha256:
        raise SachsDataError("archive checksum mismatch for %s" % archive_path)

    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != ARCHIVE_KEYS:
                raise SachsDataError(
                    "invalid keys in %s; missing=%r, extra=%r"
                    % (
                        archive_path,
                        sorted(ARCHIVE_KEYS - keys),
                        sorted(keys - ARCHIVE_KEYS),
                    )
                )
            version = np.asarray(archive["data_format_version"])
            if version.shape != () or version.dtype != np.dtype("int64"):
                raise SachsDataError("data_format_version must be an int64 scalar")
            if int(version.item()) != DATA_FORMAT_VERSION:
                raise SachsDataError("unsupported Sachs archive format")
            if _scalar_string(archive, "dataset_id") != DATASET_ID:
                raise SachsDataError("wrong dataset identifier in archive")

            samples = np.array(archive["samples"], copy=True)
            offsets = np.array(archive["offsets"], copy=True)
            true_dag = np.array(archive["true_dag"], copy=True)
            intended_targets = np.array(archive["intended_targets"], copy=True)
            node_names = tuple(str(value) for value in np.asarray(archive["node_names"]).tolist())
            environment_names = tuple(
                str(value) for value in np.asarray(archive["environment_names"]).tolist()
            )
            raw_filenames = tuple(
                str(value) for value in np.asarray(archive["raw_filenames"]).tolist()
            )
            stored_data_sha256 = _scalar_string(archive, "data_sha256")
            stored_raw_sha256 = _scalar_string(archive, "raw_bundle_sha256")
    except SachsDataError:
        raise
    except Exception as error:
        raise SachsDataError("could not read Sachs archive %s: %s" % (archive_path, error)) from error

    if samples.dtype != np.dtype("float64") or samples.shape != (
        SACHS_TOTAL_SAMPLES,
        len(SACHS_NODE_NAMES),
    ):
        raise SachsDataError("samples in %s have wrong dtype or shape" % archive_path)
    if not np.isfinite(samples).all():
        raise SachsDataError("samples in %s contain NaN or infinity" % archive_path)
    expected_offsets = np.asarray(
        [0] + list(np.cumsum(SACHS_SAMPLE_COUNTS)), dtype=np.int64
    )
    if offsets.dtype != np.dtype("int64") or not np.array_equal(offsets, expected_offsets):
        raise SachsDataError("environment offsets in %s are invalid" % archive_path)
    if true_dag.dtype != np.dtype("uint8") or true_dag.shape != (
        len(SACHS_NODE_NAMES),
        len(SACHS_NODE_NAMES),
    ):
        raise SachsDataError("true_dag in %s has wrong dtype or shape" % archive_path)
    _validate_binary("true_dag", true_dag)
    if np.any(np.diag(true_dag)) or not _is_acyclic(true_dag):
        raise SachsDataError("true_dag in %s is not a loop-free DAG" % archive_path)
    if not np.array_equal(true_dag, consensus_dag()):
        raise SachsDataError("true_dag in %s differs from the pinned 18-edge DAG" % archive_path)
    expected_targets = intended_target_matrix()
    if intended_targets.dtype != np.dtype("uint8") or intended_targets.shape != expected_targets.shape:
        raise SachsDataError("intended_targets in %s have wrong dtype or shape" % archive_path)
    _validate_binary("intended_targets", intended_targets)
    if not np.array_equal(intended_targets, expected_targets):
        raise SachsDataError("intended targets in %s differ from the pinned contract" % archive_path)
    if node_names != SACHS_NODE_NAMES:
        raise SachsDataError("node names in %s are out of order" % archive_path)
    if environment_names != tuple(str(env["name"]) for env in SACHS_ENVIRONMENTS):
        raise SachsDataError("environment names in %s are out of order" % archive_path)
    if raw_filenames != tuple(str(env["filename"]) for env in SACHS_ENVIRONMENTS):
        raise SachsDataError("raw filenames in %s are out of order" % archive_path)

    data = [
        np.array(samples[offsets[index] : offsets[index + 1]], copy=True)
        for index in range(len(SACHS_ENVIRONMENTS))
    ]
    observed_data_sha256 = data_sha256(data)
    if observed_data_sha256 != stored_data_sha256:
        raise SachsDataError("sample checksum mismatch inside %s" % archive_path)
    return data, true_dag, intended_targets, {
        "archive_sha256": observed_archive_sha256,
        "data_sha256": observed_data_sha256,
        "raw_bundle_sha256": stored_raw_sha256,
    }


def load_sachs_data(
    data_root: Union[Path, str] = DEFAULT_DATA_ROOT,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Dict[str, object]]:
    """Load and fully validate the prepared Sachs data package.

    The returned ``data`` list is ordered observational, Akt, PKC, PIP2, Mek,
    PIP3.  ``intended_targets`` has one row for each element of ``data[1:]``.
    """

    data_root = Path(data_root).expanduser().resolve()
    manifest_path = data_root / MANIFEST_NAME
    manifest = _load_json(manifest_path)
    _validate_manifest(manifest, manifest_path)

    archive_metadata = manifest["archive"]
    assert isinstance(archive_metadata, dict)
    expected_archive_sha256 = str(archive_metadata["sha256"])
    data, true_dag, intended_targets, archive_info = _read_archive(
        data_root / ARCHIVE_NAME, expected_archive_sha256
    )

    raw_directory = data_root / RAW_DIRECTORY_NAME
    raw_data = _read_raw_directory(raw_directory)
    observed_raw_sha256 = raw_bundle_sha256(raw_directory)
    digests = manifest["digests"]
    assert isinstance(digests, dict)
    if observed_raw_sha256 != digests["raw_bundle_sha256"]:
        raise SachsDataError("raw bundle checksum does not match the manifest")
    if archive_info["raw_bundle_sha256"] != observed_raw_sha256:
        raise SachsDataError("raw bundle checksum does not match the archive")
    if archive_info["data_sha256"] != digests["data_sha256"]:
        raise SachsDataError("sample checksum does not match the manifest")
    for index, (raw_values, archived_values) in enumerate(zip(raw_data, data)):
        if not np.array_equal(raw_values, archived_values):
            raise SachsDataError(
                "archive values differ from raw source in environment %d" % index
            )

    info = {
        "dataset": DATASET_ID,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "archive_path": str(data_root / ARCHIVE_NAME),
        "node_names": list(SACHS_NODE_NAMES),
        "environment_names": [str(env["name"]) for env in SACHS_ENVIRONMENTS],
        "sample_counts": list(SACHS_SAMPLE_COUNTS),
        "total_samples": SACHS_TOTAL_SAMPLES,
        "num_environments": len(SACHS_ENVIRONMENTS),
        "num_variables": len(SACHS_NODE_NAMES),
        "true_edge_count": len(SACHS_CONSENSUS_ARCS),
        "source_commit": SOURCE_COMMIT,
        "data_sha256": archive_info["data_sha256"],
        "raw_bundle_sha256": observed_raw_sha256,
        "archive_sha256": archive_info["archive_sha256"],
        "manifest": manifest,
    }
    return data, true_dag.astype(int), intended_targets.astype(int), info


def validate_sachs_data(data_root: Union[Path, str] = DEFAULT_DATA_ROOT) -> Dict[str, object]:
    """Fully validate a prepared package and return its information mapping."""

    _, _, _, info = load_sachs_data(data_root)
    return info


def _download(url: str, destination: Path, timeout: float) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "micodag-sachs-data-preparer/1"},
        method="GET",
    )
    context = ssl.create_default_context()
    if ssl.get_default_verify_paths().cafile is None:
        # Some standalone macOS Python installers do not populate their
        # OpenSSL cafile.  Keep certificate verification enabled and use a
        # known CA bundle when one is available; Quest normally takes the
        # default branch above.
        candidate_bundles = []  # type: List[str]
        try:
            import certifi  # type: ignore

            candidate_bundles.append(certifi.where())
        except ImportError:
            pass
        candidate_bundles.extend(
            (
                "/etc/ssl/cert.pem",
                "/etc/ssl/certs/ca-certificates.crt",
                "/opt/homebrew/etc/ca-certificates/cert.pem",
            )
        )
        for bundle in candidate_bundles:
            if Path(bundle).is_file():
                context = ssl.create_default_context(cafile=bundle)
                break
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            with destination.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
    except (OSError, urllib.error.URLError) as error:
        raise SachsDataError("could not download %s: %s" % (url, error)) from error


def _resolve_offline_file(source_directory: Path, filename: str) -> Path:
    candidates = (
        source_directory / filename,
        source_directory / RAW_DIRECTORY_NAME / filename,
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        if not existing:
            raise SachsDataError(
                "offline source is missing %s (looked in %s)"
                % (filename, ", ".join(str(path) for path in candidates))
            )
        raise SachsDataError("offline source has ambiguous copies of %s" % filename)
    return existing[0]


def _stage_raw_files(
    raw_directory: Path,
    source_directory: Optional[Path],
    timeout: float,
) -> str:
    raw_directory.mkdir(parents=True, exist_ok=False)
    for environment in SACHS_ENVIRONMENTS:
        filename = str(environment["filename"])
        destination = raw_directory / filename
        if source_directory is None:
            _download(source_url(environment), destination, timeout)
        else:
            source = _resolve_offline_file(source_directory, filename)
            try:
                shutil.copyfile(source, destination)
            except OSError as error:
                raise SachsDataError(
                    "could not copy offline source %s: %s" % (source, error)
                ) from error
        # Validate immediately, so no unverified source bytes reach the archive.
        _read_raw_file(destination, environment)
    return "https_download" if source_directory is None else "offline_source_dir"


def _write_archive(path: Path, data: Sequence[np.ndarray], raw_sha256: str) -> None:
    samples_sha256 = data_sha256(data)
    samples = np.concatenate(
        [np.asarray(values, dtype="<f8", order="C") for values in data], axis=0
    )
    offsets = np.asarray([0] + list(np.cumsum(SACHS_SAMPLE_COUNTS)), dtype="<i8")
    try:
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                data_format_version=np.asarray(DATA_FORMAT_VERSION, dtype="<i8"),
                dataset_id=np.asarray(DATASET_ID),
                samples=samples,
                offsets=offsets,
                true_dag=consensus_dag(),
                intended_targets=intended_target_matrix(),
                node_names=np.asarray(SACHS_NODE_NAMES),
                environment_names=np.asarray(
                    [str(environment["name"]) for environment in SACHS_ENVIRONMENTS]
                ),
                raw_filenames=np.asarray(
                    [str(environment["filename"]) for environment in SACHS_ENVIRONMENTS]
                ),
                data_sha256=np.asarray(samples_sha256),
                raw_bundle_sha256=np.asarray(raw_sha256),
            )
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise SachsDataError("could not write Sachs archive %s: %s" % (path, error)) from error


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise SachsDataError("could not write Sachs manifest %s: %s" % (path, error)) from error


def _manifest_recognizes_sachs_target(data_root: Path) -> bool:
    try:
        manifest = _load_json(data_root / MANIFEST_NAME)
    except SachsDataError:
        return False
    return (
        manifest.get("dataset") == DATASET_ID
        and manifest.get("generator") == GENERATOR
        and manifest.get("data_format_version") == DATA_FORMAT_VERSION
        and isinstance(manifest.get("source"), dict)
        and manifest["source"].get("commit") == SOURCE_COMMIT
        and isinstance(manifest.get("archive"), dict)
        and manifest["archive"].get("path") == ARCHIVE_NAME
    )


def _validate_target(data_root: Path, overwrite: bool) -> None:
    unresolved = data_root.expanduser().absolute()
    if unresolved.is_symlink():
        raise SachsDataError("refusing to use a symlink as the Sachs data root")
    resolved = unresolved.resolve()
    broad_targets = {
        Path(resolved.anchor),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
        (PROJECT_ROOT / "data").resolve(),
        (PROJECT_ROOT / "experiments").resolve(),
        Path.cwd().resolve(),
    }
    if resolved in broad_targets or resolved in PROJECT_ROOT.resolve().parents:
        raise SachsDataError("refusing to use broad Sachs data target %s" % resolved)
    if resolved.exists() and not resolved.is_dir():
        raise SachsDataError("Sachs data target exists but is not a directory: %s" % resolved)
    if resolved.exists() and overwrite and not _manifest_recognizes_sachs_target(resolved):
        raise SachsDataError(
            "refusing to overwrite unrelated directory %s; it lacks the pinned Sachs manifest signature"
            % resolved
        )


@contextmanager
def _preparation_lock(data_root: Path):
    data_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = data_root.parent / (".%s.preparation.lock" % data_root.name)
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SachsDataError("another process is preparing %s" % data_root) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _install_staging_directory(staging: Path, data_root: Path) -> None:
    if not data_root.exists():
        os.replace(staging, data_root)
        return

    backup = Path(
        tempfile.mkdtemp(prefix=".%s.backup-" % data_root.name, dir=str(data_root.parent))
    )
    backup.rmdir()
    os.replace(data_root, backup)
    try:
        os.replace(staging, data_root)
    except Exception:
        if not data_root.exists() and backup.exists():
            os.replace(backup, data_root)
        raise
    try:
        # The target was already checked for the pinned manifest signature.
        shutil.rmtree(backup)
    except OSError as error:
        raise SachsDataError(
            "installed new Sachs data but could not remove recognized backup %s: %s"
            % (backup, error)
        ) from error


def _recover_interrupted_backup(data_root: Path) -> None:
    """Restore one fully valid backup left between the two directory renames."""

    if data_root.exists():
        return
    backups = sorted(data_root.parent.glob(".%s.backup-*" % data_root.name))
    if not backups:
        return
    if len(backups) != 1 or not backups[0].is_dir():
        raise SachsDataError(
            "cannot safely choose among interrupted Sachs backups for %s: %r"
            % (data_root, backups)
        )
    backup = backups[0]
    try:
        validate_sachs_data(backup)
    except SachsDataError as error:
        raise SachsDataError(
            "interrupted Sachs backup is not valid and cannot be restored: %s"
            % backup
        ) from error
    os.replace(backup, data_root)


def prepare_sachs_data(
    data_root: Union[Path, str] = DEFAULT_DATA_ROOT,
    *,
    source_directory: Optional[Union[Path, str]] = None,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> Dict[str, object]:
    """Download/copy, validate, stage, and atomically install the Sachs package."""

    unresolved_data_root = Path(data_root).expanduser().absolute()
    _validate_target(unresolved_data_root, overwrite)
    data_root = unresolved_data_root.resolve()
    source_path = None
    if source_directory is not None:
        source_path = Path(source_directory).expanduser().resolve()
        if not source_path.is_dir():
            raise SachsDataError("offline source directory does not exist: %s" % source_path)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise SachsDataError("download timeout must be positive")

    with _preparation_lock(data_root):
        _recover_interrupted_backup(data_root)
        if data_root.exists() and not overwrite:
            try:
                return validate_sachs_data(data_root)
            except SachsDataError as error:
                raise SachsDataError(
                    "existing Sachs data at %s are invalid; use --overwrite only if this is a recognized Sachs target"
                    % data_root
                ) from error

        staging = Path(
            tempfile.mkdtemp(
                prefix=".%s.staging-" % data_root.name, dir=str(data_root.parent)
            )
        )
        try:
            retrieval_mode = _stage_raw_files(
                staging / RAW_DIRECTORY_NAME, source_path, float(timeout)
            )
            data = _read_raw_directory(staging / RAW_DIRECTORY_NAME)
            if [values.shape[0] for values in data] != list(SACHS_SAMPLE_COUNTS):
                raise SachsDataError("internal sample-count validation failed")
            raw_sha256 = raw_bundle_sha256(staging / RAW_DIRECTORY_NAME)
            archive_path = staging / ARCHIVE_NAME
            _write_archive(archive_path, data, raw_sha256)
            archive_sha256 = file_sha256(archive_path)
            manifest = _build_manifest(
                archive_sha256=archive_sha256,
                samples_sha256=data_sha256(data),
                raw_sha256=raw_sha256,
                retrieval_mode=retrieval_mode,
            )
            _write_manifest(staging / MANIFEST_NAME, manifest)
            validate_sachs_data(staging)
            _install_staging_directory(staging, data_root)
            return validate_sachs_data(data_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
