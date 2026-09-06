#!/usr/bin/env python3
"""Run, resume, validate, and plot the unscreened scm_4 chamber experiment."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from src import chamber_data as data
from src import chamber_experiment as study
from src.main_experiment_cli import wall_time_limit

DEFAULT_ROOT = data.ROOT / "experiment_results/causal_chambers" / study.PROFILE
make_design = study.make_design


def freeze(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if data.read_json(path) != value:
                raise ValueError("frozen settings differ; preserve the old run and use a new output root")
        else:
            data.write_json(path, value)


def select_jobs(design, methods=None, setting=None):
    available = study.methods_for_profile(design["profile"])
    methods = list(available) if methods is None else methods
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(available):
        raise ValueError("choose unique chamber methods")
    if setting is not None and (len(methods) != 1 or setting < 1):
        raise ValueError("--setting requires one method and a positive one-based setting number")
    jobs = [j for j in design["jobs"] if j["method"] in methods
            and (setting is None or j["setting_index"] == setting - 1)]
    if not jobs:
        raise ValueError("setting is outside this method's grid")
    return jobs


def fit_one(root, design, job, data_root=data.DATA_ROOT):
    study.validate_identity(design, data_root)
    output = root / job["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            row = data.read_json(output)
            study.validate_result(row, job, design, allow_failure=True)
            return int(row["status"] not in study.SUCCESS)
        row = {"job": job, "design_sha256": data.digest(design)}
        started, phase = time.monotonic(), "loading"
        try:
            arrays, metadata, bounds = data.load(job["configuration"], data_root)
            phase = "fitting"
            timer = (nullcontext() if job["method"].startswith("ps_mip")
                     else wall_time_limit(job["time_limit"], job["method"]))
            with timer:
                result = study.fit_method(job, arrays, np.asarray(metadata["documented_targets"]),
                                          bounds, design["seed"], design["threads"])
            phase = "validation"
            measured = result["solve_seconds"] if job["method"].startswith("ps_mip") else result["fit_seconds"]
            row.update(status="ok_nonoptimal" if result.get("optimal") is False else "ok", result=result,
                       timing={"fit_limit_seconds": job["time_limit"],
                               "budget_overrun_seconds": max(0., measured - job["time_limit"])},
                       metrics=study.metrics(result["estimated_dag"], result["estimated_graph"], metadata))
            study.validate_result(row, job, design)
        except Exception as error:
            timeout = isinstance(error, TimeoutError) and phase == "fitting" and not job["method"].startswith("ps_mip")
            row.update(status="timeout" if timeout else "failed", phase=phase,
                       failure_kind="algorithm_timeout" if timeout else "exception", traceback=traceback.format_exc())
            print(row["traceback"], file=sys.stderr, flush=True)
        row["elapsed_seconds"] = time.monotonic() - started
        data.write_json(output, row)
        print(row["status"], row.get("metrics"), flush=True)
        return int(row["status"] not in study.SUCCESS)


def run_child(command, log, timeout):
    """A watchdog can stop only the process group created for this fit."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"\nAttempt UTC {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        stream.flush()
        child = subprocess.Popen(command, cwd=data.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            return child.wait(), True
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise


@contextmanager
def dispatch_lock(root, index):
    """Separate method/setting jobs may coexist, but duplicate jobs cannot."""
    (root / "dispatch").mkdir(parents=True, exist_ok=True)
    with (root / ".execution.lock").open("a") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with (root / "dispatch" / f"job_{index:03d}.lock").open("a") as job_lock:
            fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


def execute(root, design, data_root=data.DATA_ROOT, jobs=None):
    failed = 0
    for job in design["jobs"] if jobs is None else jobs:
        study.validate_identity(design, data_root)
        with dispatch_lock(root, job["index"]):
            output = root / job["output"]
            if output.exists():
                row = data.read_json(output)
                study.validate_result(row, job, design, allow_failure=True)
                failed += int(row["status"] not in study.SUCCESS)
                print(f"Preserved {job['method']} setting {job['setting_index']+1}: {row['status']}", flush=True)
                continue
            log = (root / job["output"].replace("/parts/", "/logs/")).with_suffix(".log")
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--output-root", str(root),
                       "--data-root", str(data_root), "--worker", "--job-index", str(job["index"])]
            started = time.monotonic()
            watchdog_seconds = design["timing"]["worker_watchdog_seconds"]
            code, watchdog = run_child(command, log, watchdog_seconds)
            elapsed = time.monotonic() - started
            if not output.exists():
                data.write_json(output, {"job": job, "design_sha256": data.digest(design),
                    "status": "watchdog_timeout" if watchdog else "process_failed", "returncode": code,
                    "elapsed_seconds": elapsed, "watchdog_seconds": watchdog_seconds})
            row = data.read_json(output)
            study.validate_result(row, job, design, allow_failure=True)
            attempt = {"job_index": job["index"], "status": row["status"], "returncode": code,
                       "watchdog": watchdog, "worker_elapsed_seconds": elapsed,
                       "watchdog_overrun_seconds": max(0., elapsed - watchdog_seconds),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
            data.write_json(root / "dispatch" / f"job_{job['index']:03d}.json", attempt)
            with (root / ".attempts.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with (root / "attempts.jsonl").open("a") as stream:
                    stream.write(json.dumps(attempt) + "\n")
            print(f"{job['method']} setting {job['setting_index']+1}: {row['status']}, {elapsed:.1f}s", flush=True)
            failed += int(code != 0 or row["status"] not in study.SUCCESS)
    return int(failed > 0)


def audit(root, data_root=data.DATA_ROOT, require_complete=True, *, provenance=None):
    design = data.read_json(root / "frozen_design.json")
    identity = study.validate_identity(design, data_root, postprocess=True)
    expected = {root / j["output"] for j in design["jobs"]}
    found = set(root.glob("*/parts/*/*.json"))
    if found - expected or (require_complete and expected - found):
        raise ValueError(f"unexpected or missing fragments: {len(found-expected)} extra, {len(expected-found)} missing")
    rows = []
    checked_sources = set()
    for job in design["jobs"]:
        path = root / job["output"]
        if path.exists():
            row = data.read_json(path)
            if "reuse" in row:
                source = row["reuse"]["source_design"]
                source_hash = data.digest(source)
                if source_hash not in checked_sources:
                    study.validate_identity(source, data_root, postprocess=True)
                    checked_sources.add(source_hash)
            study.validate_result(row, job, design, allow_failure=True)
            rows.append(row)
    if provenance is not None:
        provenance.update(identity)
    return design, rows


def reuse_unchanged(root, source_root, design, data_root=data.DATA_ROOT):
    """Import only identical PS-MIP/GIES/IGSP fits, with original rows embedded."""
    if root.resolve() == source_root.resolve() or design["profile"] != study.PROFILE:
        raise ValueError("reuse requires a separate current-profile output root")
    source, rows = audit(source_root, data_root)
    if source["profile"] != study.LEGACY_PROFILE:
        raise ValueError("reuse requires the historical seven-condition run")
    by_setting = {(r["job"]["method"], r["job"]["setting_index"]): r for r in rows}
    planned = []
    for job in design["jobs"]:
        if job["method"] not in study.REUSABLE_METHODS: continue
        original = by_setting[study.REUSABLE_METHODS[job["method"]], job["setting_index"]]
        row = copy.deepcopy(original)
        row.update(job=job, design_sha256=data.digest(design), reuse={
            "source_design": source, "source_row": original,
            "source_row_sha256": data.digest(original),
            "source_file": str(source_root / original["job"]["output"]),
            "source_file_sha256": data.file_hash(source_root / original["job"]["output"])})
        study.validate_result(row, job, design, allow_failure=True)
        planned.append((root / job["output"], row))
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(root / "frozen_design.json", design)
        for path, row in planned:
            if path.exists() and data.read_json(path) != row:
                raise ValueError(f"reuse would overwrite a different result: {path}")
        for path, row in planned:
            if not path.exists(): data.write_json(path, row)
    return {"planned": len(design["jobs"]), "reused": len(planned),
            "new_fits_needed": len(design["jobs"]) - len(planned),
            "source_design_sha256": data.digest(source)}


def completion_report(design, rows, provenance=None):
    report = {"profile": design["profile"], "planned": len(design["jobs"]), "accounted": len(rows),
            "all_planned_accounted": len(rows) == len(design["jobs"]),
            "all_planned_successful": len(rows) == len(design["jobs"]) and all(r["status"] in study.SUCCESS for r in rows),
            "status_counts": {s: sum(r["status"] == s for r in rows) for s in (*study.SUCCESS, *study.FAILURE)},
            "reused": sum("reuse" in r for r in rows),
            "design_sha256": data.digest(design)}
    if provenance is not None:
        report["postprocessing"] = {"policy": provenance["policy"],
            "runtime_matches_fit": provenance["runtime_matches_fit"],
            "runtime_sha256": data.digest(provenance["runtime"]),
            "accepted_validation_code_revisions": list(provenance["accepted_validation_code_revisions"])}
    return report


def export(root, data_root=data.DATA_ROOT, allow_partial=False):
    with (root / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = {}
        design, rows = audit(root, data_root, require_complete=not allow_partial, provenance=provenance)
        from src.chamber_plotting import export as plot
        destination = root / ("summary_preview" if allow_partial else "summary")
        data.write_json(destination / "postprocessing_identity.json", provenance)
        plot(root, data_root, design, rows, allow_partial)
        report = completion_report(design, rows, provenance)
        data.write_json(root / ("completion_preview.json" if allow_partial else "completion.json"), report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=study.METHODS)
    parser.add_argument("--setting", type=int, help="one-based setting number; requires exactly one method")
    parser.add_argument("--time-limit", type=float, default=3600, help="seconds per fit (default: 3600)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--data-root", type=Path, default=data.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--plot", action="store_true")
    modes.add_argument("--reuse-from", type=Path, help="reuse unchanged PS-MIP/GIES/IGSP fits from the historical seven-condition run; no fitting")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--allow-partial", action="store_true", help="allow an explicitly incomplete audit/plot preview")
    parser.add_argument("--job-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.allow_partial and not (args.audit or args.plot):
        parser.error("allow-partial requires audit or plot")
    if args.worker != (args.job_index is not None):
        parser.error("job-index is reserved for the internal worker")
    root, data_root = args.output_root.resolve(), args.data_root.resolve()
    if args.prepare:
        data_root.mkdir(parents=True, exist_ok=True)
        with (data_root / ".preparation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data.prepare(data_root)
        return 0
    if args.plot:
        report = export(root, data_root, args.allow_partial)
        print(json.dumps(report, indent=2))
        return int(not report["all_planned_successful"])
    if args.audit:
        provenance = {}
        design, rows = audit(root, data_root, require_complete=not args.allow_partial, provenance=provenance)
        report = completion_report(design, rows, provenance)
        print(json.dumps(report, indent=2))
        return int(not report["all_planned_successful"])
    if args.worker:
        design = data.read_json(root / "frozen_design.json")
        if not 0 <= args.job_index < len(design["jobs"]):
            parser.error("job index outside design")
        return fit_one(root, design, design["jobs"][args.job_index], data_root)
    try:
        design = make_design(data_root, args.threads, args.time_limit)
        jobs = select_jobs(design, args.methods, args.setting)
    except ValueError as error:
        parser.error(str(error))
    apis = study.check_apis()
    if args.reuse_from:
        if args.methods or args.setting:
            parser.error("reuse-from imports all unchanged methods; do not combine with methods or setting")
        print(json.dumps(reuse_unchanged(root, args.reuse_from.resolve(), design, data_root), indent=2))
        return 0
    if args.dry_run:
        print(json.dumps({"profile": study.PROFILE, "planned": len(design["jobs"]), "selected": len(jobs),
            "methods": {m: sum(j["method"] == m for j in jobs) for m in study.METHODS},
            "sample_sizes": design["configurations"]["scm_4"]["data"]["sample_sizes"],
            "screening": design["screening"], "timing": design["timing"], "apis": apis,
            "design_sha256": data.digest(design)}, indent=2))
        return 0
    freeze(root / "frozen_design.json", design)
    return execute(root, design, data_root, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
