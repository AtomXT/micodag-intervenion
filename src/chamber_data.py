"""Frozen numeric-only Causal Chambers pilot; no Sachs or synthetic defaults.

The public reference records known effects, not verified absence of every weak
effect. Programmed actuator links and physical sensor links remain distinguished.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from itertools import combinations
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/causal_chambers/lt_camera_v1"
CONFIGURATIONS = ("scm_4", "scm_5")
ENVIRONMENTS = ("reference", "red", "green", "blue")
NODES = ("red", "green", "blue", "current", "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3")
ARCHIVE_URL = "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_camera_v1.zip"
DATA_REPO = "juangamella/causal-chamber"
GRAPH_REPO = "juangamella/causal-chamber-package"
GRAPH_PATH = "causalchamber/ground_truth/adjacencies/adjacency_lt_camera.csv"


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=serial, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def array_hash(arrays):
    h = hashlib.sha256()
    for values in arrays:
        a = np.ascontiguousarray(values, dtype="<f8")
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


class RemoteZip(io.RawIOBase):
    """Seekable HTTP range reader; refuse accidental multi-GB full downloads."""
    def __init__(self, url, opener=urllib.request.urlopen):
        self.url, self.opener, self.pos = url, opener, 0
        with opener(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.identity = {"url": url, "size": self.size, "etag": r.headers["ETag"],
                             "version_id": r.headers.get("x-amz-version-id"),
                             "last_modified": r.headers.get("Last-Modified")}
        self.transferred = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek mode")
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("negative range position")
        return self.pos

    def read(self, size=-1):
        size = max(0, self.size - self.pos) if size < 0 else min(size, max(0, self.size - self.pos))
        if size == 0: return b""
        if size > 40_000_000: raise ValueError("unexpectedly large archive range")
        end = self.pos + size - 1
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={self.pos}-{end}", "If-Match": self.identity["etag"]})
        with self.opener(request, timeout=60) as response:
            expected = f"bytes {self.pos}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("server ignored requested range; full archive refused")
            value = response.read(size + 1)
        if len(value) != size: raise ValueError("truncated or oversized archive range")
        self.pos += size
        self.transferred += size
        return value


def acquire(root=DATA_ROOT):
    root = Path(root)
    if (root / "source.json").exists():
        source = read_json(root / "source.json")
        verify_files(root, source["files"])
        return source
    root.mkdir(parents=True, exist_ok=True)
    revisions = {repo: json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/main"))["sha"]
                 for repo in (DATA_REPO, GRAPH_REPO)}
    sources = {"source/adjacency_lt_camera.csv": (GRAPH_REPO, GRAPH_PATH)}
    for name in ("README.md", "variables.csv", "generators/scm_4.py", "generators/scm_5.py"):
        sources["source/" + name.replace("/", "_")] = (DATA_REPO, "datasets/lt_camera_v1/" + name)
    sources["source/LICENSE_DATASETS.txt"] = (DATA_REPO, "datasets/lt_camera_v1/LICENSE_DATASETS.txt")
    sources["source/LICENSE_SOFTWARE.txt"] = (DATA_REPO, "datasets/lt_camera_v1/LICENSE_SOFTWARE.txt")
    records = {}
    for relative, (repo, path) in sources.items():
        url = f"https://raw.githubusercontent.com/{repo}/{revisions[repo]}/{path}"
        contents = fetch(url)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        records[relative] = {"sha256": file_hash(destination), "url": url}
    remote = RemoteZip(ARCHIVE_URL)
    with zipfile.ZipFile(remote) as archive:
        for cfg in CONFIGURATIONS:
            for environment in ENVIRONMENTS:
                experiment = f"{cfg}_{environment}"
                member = f"lt_camera_v1/{experiment}/{experiment}.csv"
                info = archive.getinfo(member)
                if info.file_size > 20_000_000: raise ValueError("unexpected CSV member size")
                contents = archive.read(info)  # ZipFile checks the member CRC.
                destination = root / "raw" / f"{experiment}.csv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(contents)
                records[str(destination.relative_to(root))] = {
                    "sha256": file_hash(destination), "archive_member": member,
                    "crc32": info.CRC, "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size}
                print(f"Acquired {experiment}: {info.file_size:,} numeric bytes", flush=True)
    source = {"version": 1, "dataset": "lt_camera_v1", "revisions": revisions,
              "archive": remote.identity, "archive_bytes_read": remote.transferred, "files": records,
              "integrity_note": "HTTPS, immutable source revisions, archive If-Match, member CRC32 and local SHA256; full-archive MD5 not checked because images were not downloaded"}
    write_json(root / "source.json", source)
    return source


def verify_files(root, records):
    for relative, record in records.items():
        if file_hash(Path(root) / relative) != record["sha256"]:
            raise ValueError(f"source/artifact identity changed: {relative}")


def targets():
    value = np.zeros((3, len(NODES)), dtype=int)
    value[:, :3] = np.eye(3, dtype=int)
    return value


def reference_graph(adjacency, configuration):
    if configuration not in CONFIGURATIONS: raise ValueError("unknown configuration")
    physical = adjacency.loc[list(NODES), list(NODES)].to_numpy(dtype=int)
    expected = np.zeros_like(physical)
    expected[:3, 3:] = 1
    if not np.array_equal(expected, physical): raise ValueError("physical reference differs from frozen 21 links")
    programmed = np.zeros_like(physical)
    programmed[0, 1] = 1
    programmed[1, 2] = 1 if configuration == "scm_4" else 0
    programmed[2, 1] = 1 if configuration == "scm_5" else 0
    return physical + programmed, physical, programmed


def common_transform(arrays):
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    if not arrays or any(a.ndim != 2 or a.shape[1] != arrays[0].shape[1] or not np.isfinite(a).all() for a in arrays):
        raise ValueError("finite arrays with identical variable order required")
    if any(np.any(a.std(axis=0) <= 0) for a in arrays): raise ValueError("positive variance required in every environment")
    mean, scale = arrays[0].mean(axis=0), arrays[0].std(axis=0)
    return [(a - mean) / scale for a in arrays], mean, scale


def ps_arrays(arrays):
    scale = arrays[0].std(axis=0)
    return [(a - a.mean(axis=0)) / scale for a in arrays]


def profile_bounds(arrays):
    """Data-only exhaustive preflight; no graph truth, penalties or target labels.

    Same moment, slope and Gamma definitions as MIP_profiled._best_local.
    Every parent set and every target mask contributes, even if later disallowed.
    """
    values = ps_arrays(arrays)
    sizes = np.array([len(a) for a in values], dtype=float)
    weights = sizes / sizes.sum()
    sigma = [a.T @ a / len(a) for a in values]
    if any(np.linalg.eigvalsh(s)[0] <= 0 for s in sigma):
        raise ValueError("positive definite environment covariances required")
    p, environments = values[0].shape[1], len(values)
    diagonals = np.concatenate([1 / np.sqrt(np.diag(s)) for s in sigma])
    low, high, coefficient_max = float(min(diagonals)), float(max(diagonals)), 0.0
    count, witness = 0, None
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for nparents in range(p):
            for parents in combinations(others, nparents):
                indices = list(parents) + [j]
                for mask in range(1 << (environments - 1)):
                    active = [0] + [e for e in range(1, environments) if not mask & (1 << (e - 1))]
                    a = weights[active].sum()
                    q = sum(weights[e] * sigma[e][np.ix_(indices, indices)] for e in active)
                    slope = np.linalg.solve(q[:-1, :-1], q[:-1, -1]) if parents else np.empty(0)
                    residual = float(q[-1, -1] - q[-1, :-1] @ slope)
                    if residual <= 0 or not np.isfinite(residual): raise ValueError("unusable profiled covariance")
                    diagonal = float(np.sqrt(a / residual))
                    coefficient = float(np.max(np.abs(slope * diagonal))) if parents else 0.0
                    if coefficient > coefficient_max:
                        witness = {"node": j, "parents": list(parents), "mask": mask}
                    low, high, coefficient_max = min(low, diagonal), max(high, diagonal), max(coefficient_max, coefficient)
                    count += 1
    bounds = {"gamma_lower": min(1e-5, low / 10), "gamma_upper": max(10., high * 10),
              "coefficient_bound": max(10., coefficient_max * 10)}
    return {"bounds": bounds, "min_diagonal": low, "max_diagonal": high,
            "max_abs_coefficient": coefficient_max, "coefficient_witness": witness,
            "local_optima_checked": count, "parent_sets": p * 2 ** (p - 1),
            "target_masks_per_parent_set": 2 ** (environments - 1),
            "input_sha256": array_hash(values), "rule": "all local optima, tenfold margin, retain looser existing bounds"}


def diagnose(arrays, frames, truth):
    """Descriptive diagnostics only; never filter rows or tune graph estimates."""
    p, results = len(truth), []
    fits = []
    for values in arrays:
        residuals, coefs, residual_variance, nonlinear_gain = [], [], [], []
        for j in range(p):
            pa = np.flatnonzero(truth[:, j])
            x = np.column_stack([np.ones(len(values)), values[:, pa]])
            beta = np.linalg.lstsq(x, values[:, j], rcond=None)[0]
            residual = values[:, j] - x @ beta
            residuals.append(residual)
            coefs.append(beta)
            residual_variance.append(float(np.var(residual)))
            if len(pa):
                x2 = np.column_stack([x, values[:, pa] ** 2])
                # Temporal halves: quadratic terms must improve out-of-sample prediction.
                errors = []
                for train, test in ((slice(None, len(x)//2), slice(len(x)//2, None)),
                                    (slice(len(x)//2, None), slice(None, len(x)//2))):
                    errors.append([float(np.mean((values[test, j] - z[test] @ np.linalg.lstsq(z[train], values[train, j], rcond=None)[0]) ** 2)) for z in (x, x2)])
                mse = np.mean(errors, axis=0)
                nonlinear_gain.append(float(1 - mse[1] / mse[0]))
            else:
                nonlinear_gain.append(None)
        fits.append((np.column_stack(residuals), coefs, np.array(residual_variance), nonlinear_gain))
    for e, (values, frame, fit) in enumerate(zip(arrays, frames, fits)):
        residuals, coefs, variances, nonlinear = fit
        correlation = np.corrcoef(residuals, rowvar=False)
        lag = [float(np.corrcoef(residuals[:-1, j], residuals[1:, j])[0, 1]) for j in range(p)]
        excluded = e - 1 if e else None
        changes = {NODES[j]: {"coefficient_max_abs_change": float(np.max(np.abs(coefs[j] - fits[0][1][j]))),
                              "residual_variance_ratio": float(variances[j] / fits[0][2][j])}
                   for j in range(p) if j != excluded}
        results.append({"environment": ENVIRONMENTS[e], "n": len(values),
            "missing_counts": frame[list(NODES)].isna().sum().to_dict(),
            "standard_deviations": np.std(values, axis=0),
            "covariance_eigenvalues": np.linalg.eigvalsh(np.cov(values, rowvar=False, ddof=0)),
            "covariance_condition_number": float(np.linalg.cond(np.cov(values, rowvar=False, ddof=0))),
            "empirical_boundary_fractions": {name: {"at_minimum": float((frame[name] == frame[name].min()).mean()),
                                                      "at_maximum": float((frame[name] == frame[name].max()).mean())} for name in NODES},
            "actuator_hardware_boundary_fractions": {name: float(frame[name].isin([0, 255]).mean()) for name in NODES[:3]},
            "residual_correlation": correlation, "residual_lag1_correlation": lag,
            "residual_standard_deviations": np.sqrt(variances),
            "quarter_residual_means": [part.mean(axis=0) for part in np.array_split(residuals, 4)],
            "quadratic_heldout_mse_fraction_improvement": dict(zip(NODES, nonlinear)),
            "non_target_changes_from_reference": changes,
            "timestamp_monotone": bool(frame.timestamp.is_monotonic_increasing),
            "counter_duplicates": int(frame.counter.duplicated().sum()),
            "caution": "descriptive, not formal tests or filtering; empirical extrema alone do not establish sensor saturation"})
    return {"role": "pre_fit_model_diagnostics_not_graph_selection", "environments": results}


def prepare(root=DATA_ROOT):
    root = Path(root)
    source = acquire(root)
    adjacency = pd.read_csv(root / "source/adjacency_lt_camera.csv", index_col=0)
    for cfg in CONFIGURATIONS:
        manifest = root / cfg / "prepared.json"
        if manifest.exists():
            load(cfg, root)
            print(f"Validated existing prepared {cfg}", flush=True)
            continue
        frames = [pd.read_csv(root / "raw" / f"{cfg}_{e}.csv") for e in ENVIRONMENTS]
        if any(len(frame) != 10000 for frame in frames): raise ValueError("expected 10,000 rows per environment")
        if any(frame.config.nunique() != 1 or frame.config.iloc[0] != "camera" for frame in frames):
            raise ValueError("unexpected chamber configuration")
        truth, physical, programmed = reference_graph(adjacency, cfg)
        fixed = {}
        for child in NODES:
            for parent in adjacency.index:
                if parent not in NODES and adjacency.loc[parent, child]:
                    values = pd.concat([f[parent] for f in frames], ignore_index=True)
                    if values.isna().any() or values.nunique() != 1:
                        raise ValueError(f"omitted documented parent is not fixed: {parent}")
                    fixed[parent] = values.iloc[0]
        arrays, mean, scale = common_transform([f[list(NODES)].to_numpy(float) for f in frames])
        preflight = profile_bounds(arrays)
        diagnostics = diagnose(arrays, frames, truth)
        destination = root / cfg
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "arrays.npz", **dict(zip(ENVIRONMENTS, arrays)))
        write_json(destination / "bounds.json", preflight)
        write_json(destination / "diagnostics.json", diagnostics)
        metadata = {"version": 1, "configuration": cfg, "nodes": NODES, "environments": ENVIRONMENTS,
                    "sample_sizes": [len(a) for a in arrays], "total_samples": sum(map(len, arrays)),
                    "reference_dag": truth, "physical_edges": physical, "programmed_edges": programmed,
                    "documented_targets": targets(), "fixed_omitted_parents": fixed,
                    "reference_mean": mean, "reference_sd": scale, "data_sha256": array_hash(arrays),
                    "source_sha256": digest(source), "preparation_code_sha256": file_hash(__file__),
                    "transformation": "common_reference_affine_no_rank_transform_no_filtering",
                    "files": {str((destination / name).relative_to(root)): {"sha256": file_hash(destination / name)}
                              for name in ("arrays.npz", "bounds.json", "diagnostics.json")}}
        write_json(manifest, metadata)
        print(f"Prepared {cfg}: {metadata['total_samples']} rows, bounds {preflight['bounds']}", flush=True)


def load(configuration, root=DATA_ROOT):
    if configuration not in CONFIGURATIONS: raise ValueError("unknown configuration")
    root = Path(root)
    metadata = read_json(root / configuration / "prepared.json")
    source = read_json(root / "source.json")
    if metadata["preparation_code_sha256"] != file_hash(__file__) or metadata["source_sha256"] != digest(source):
        raise ValueError("stale preparation or source identity")
    verify_files(root, source["files"])
    verify_files(root, metadata["files"])
    if metadata["nodes"] != list(NODES) or metadata["environments"] != list(ENVIRONMENTS) or metadata["configuration"] != configuration:
        raise ValueError("mixed variable/environment/configuration identity")
    with np.load(root / configuration / "arrays.npz", allow_pickle=False) as archive:
        arrays = [archive[e] for e in ENVIRONMENTS]
    if array_hash(arrays) != metadata["data_sha256"] or [len(a) for a in arrays] != metadata["sample_sizes"]:
        raise ValueError("common input arrays changed")
    return arrays, metadata, read_json(root / configuration / "bounds.json")
