import copy
from contextlib import redirect_stdout
import fcntl
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from experiments import run_chamber_experiment as runner
from src import chamber_data as data
from src import chamber_experiment as study
from src.main_experiment_cli import wall_time_limit


@unittest.skipUnless((data.DATA_ROOT / "scm_4/prepared.json").exists(), "prepared data unavailable")
class ChamberRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = study.make_design()

    def job(self, method="igsp_oracle"):
        return next(j for j in self.design["jobs"] if j["method"] == method)

    def result(self, method="igsp_oracle", optimal=True):
        job = self.job(method)
        graph = np.zeros((10, 10), int)
        result = {"estimated_dag": graph, "estimated_graph": graph, "estimated_targets": data.targets(),
                  "fit_seconds": .01, "common_input_sha256_before": job["data_sha256"],
                  "common_input_sha256_after": job["data_sha256"]}
        if method.startswith("ps_mip"):
            result.update(gamma=np.eye(10), screen_edges=45, screen_parent_sets=5120,
                numerical_bounds=self.design["configurations"]["scm_4"]["preflight"]["bounds"],
                bounds_inactive=True, solution_count=1, optimal=optimal,
                solver_status_code=2 if optimal else 9, solver_status="optimal" if optimal else "time_limit",
                objective_value=1., objective_bound=1. if optimal else .9,
                mip_gap=0. if optimal else .1, solve_seconds=.005 if optimal else 3600.01)
        return result

    def test_fixed_design_and_canonical_integer_float_budget(self):
        d = self.design
        self.assertEqual(len(d["jobs"]), 63)
        self.assertEqual(list(d["configurations"]), ["scm_4"])
        self.assertEqual([sum(j["method"] == m for j in d["jobs"]) for m in study.METHODS], [17,15,8,8,15])
        self.assertEqual(len({j["output"] for j in d["jobs"]}), 63)
        self.assertTrue(all(j["time_limit"] == 3600 for j in d["jobs"]))
        self.assertEqual(d["timing"]["worker_watchdog_seconds"], 3720)
        self.assertEqual(d["screening"], {"enabled": False, "candidate_pairs": 45, "parent_sets": 5120})
        self.assertEqual((d["seed"], d["threads"], d["native_threads"]), (20260901, 8, 1))
        self.assertEqual(data.digest(d), data.digest(study.make_design(time_limit=3600.0)))

    def test_setting_selection_and_method_arrays(self):
        indices = []
        for method in study.METHODS:
            selected = runner.select_jobs(self.design, [method])
            for number, job in enumerate(selected, 1):
                self.assertEqual(runner.select_jobs(self.design, [method], number), [job])
                indices.append(job["index"])
            with self.assertRaises(ValueError): runner.select_jobs(self.design, [method], len(selected)+1)
        self.assertEqual(indices, list(range(63)))
        for methods, setting in (([study.METHODS[0]], 0), (list(study.METHODS[:2]), 1), (["bad"], None),
                                 ([study.METHODS[0]]*2, None)):
            with self.assertRaises(ValueError): runner.select_jobs(self.design, methods, setting)

    def test_scientific_inputs_unchanged_from_old_five_minute_run(self):
        old_path = data.ROOT / "experiment_results/causal_chambers/scm4_unscreened_5min_v1/local/frozen_design.json"
        if not old_path.exists(): self.skipTest("historical local run not present")
        old = data.read_json(old_path)
        self.assertEqual(old["configurations"], self.design["configurations"])
        legacy_names = dict(study.REUSABLE_METHODS, utigsp_unknown="utigsp_present", gnies_unknown="gnies_present")
        lookup = {(j["method"], j["setting_index"]): j for j in old["jobs"]}
        for after in self.design["jobs"]:
            before = lookup[legacy_names[after["method"]], after["setting_index"]]
            for field in ("setting", "data_sha256", "configuration"):
                self.assertEqual(before[field], after[field])
            self.assertEqual((before["time_limit"], after["time_limit"]), (300, 3600))

    def test_dry_run_is_read_only_and_uses_selected_method(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "not-created"
            out = io.StringIO()
            with redirect_stdout(out), mock.patch.object(runner, "execute") as execute:
                self.assertEqual(runner.main(["--dry-run", "--methods", "gies_oracle", "--setting", "15",
                    "--time-limit", "3600", "--output-root", str(root)]), 0)
            self.assertFalse(root.exists())
            execute.assert_not_called()
            report = json.loads(out.getvalue())
            self.assertEqual((report["planned"], report["selected"]), (63, 1))
            self.assertEqual(report["apis"]["utigsp_invariance"], "gaussian")

    def test_rejects_legacy_and_altered_manifests(self):
        for key in ("profile", "budget", "source", "input"):
            changed = copy.deepcopy(self.design)
            if key == "profile": changed["profile"] = "scm4_unscreened_5min_v1"
            if key == "budget": changed["jobs"][0]["time_limit"] = 300
            if key == "source": changed["implementation"]["MIP.py"] = "old"
            if key == "input": changed["jobs"][0]["data_sha256"] = "old"
            with self.assertRaises(ValueError): study.validate_identity(changed)
            with self.assertRaises(ValueError): study.validate_identity(changed, postprocess=True)

    def test_runtime_portability_is_postprocessing_only(self):
        saved = copy.deepcopy(self.design)
        saved["runtime"]["python"] = "3.9.5"
        saved["runtime"]["versions"]["numpy"] = "saved-version"
        saved["runtime"]["package_source_hashes"]["graphical_models"] = "saved-package-source"
        before = copy.deepcopy(saved)
        with self.assertWarnsRegex(UserWarning, "Postprocessing runtime differs"):
            provenance = study.validate_identity(saved, postprocess=True)
        self.assertEqual(saved, before)
        self.assertFalse(provenance["runtime_matches_fit"])
        self.assertEqual(provenance["runtime"], self.design["runtime"])
        self.assertEqual(provenance["design_sha256"], data.digest(saved))
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(study, "fit_method") as fit, \
             mock.patch.object(runner, "run_child") as child:
            with self.assertRaises(ValueError): study.validate_identity(saved)
            with self.assertRaises(ValueError): runner.fit_one(Path(folder), saved, self.job())
            with self.assertRaises(ValueError): runner.execute(Path(folder), saved, jobs=[self.job()])
            fit.assert_not_called()
            child.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_only_reviewed_audit_source_revisions_are_compatible(self):
        legacy = study.make_design(profile=study.LEGACY_PROFILE)
        saved = copy.deepcopy(legacy)
        for name, hashes in study.POSTPROCESS_COMPATIBLE_SOURCES.items():
            saved["implementation"][name] = sorted(hashes)[0]
        before = copy.deepcopy(saved)
        provenance = study.validate_identity(saved, postprocess=True)
        self.assertEqual(saved, before)
        self.assertEqual(set(provenance["accepted_validation_code_revisions"]),
                         set(study.POSTPROCESS_COMPATIBLE_SOURCES))
        self.assertEqual(provenance["implementation"], legacy["implementation"])
        with self.assertRaises(ValueError): study.validate_identity(saved)
        for name in study.CORE_FILES:
            changed = copy.deepcopy(saved)
            changed["implementation"][name] = "unreviewed-source"
            with self.assertRaises(ValueError): study.validate_identity(changed, postprocess=True)
        changed = copy.deepcopy(self.design)
        for name, hashes in study.POSTPROCESS_COMPATIBLE_SOURCES.items():
            changed["implementation"][name] = sorted(hashes)[0]
        with self.assertRaises(ValueError): study.validate_identity(changed, postprocess=True)

    def test_portable_audit_keeps_original_identity_and_scientific_validation(self):
        saved = study.make_design(profile=study.LEGACY_PROFILE)
        saved["runtime"]["distribution_identity"] = "quest-runtime"
        for name, hashes in study.POSTPROCESS_COMPATIBLE_SOURCES.items():
            saved["implementation"][name] = sorted(hashes)[0]
        job = next(j for j in saved["jobs"] if j["method"] == "igsp_complete")
        row = {"job": job, "design_sha256": data.digest(saved), "status": "ok",
               "result": self.result(), "elapsed_seconds": .1,
               "timing": {"fit_limit_seconds": 3600, "budget_overrun_seconds": 0.},
               "metrics": study.metrics(np.zeros((10, 10)), np.zeros((10, 10)),
                                        saved["configurations"]["scm_4"]["data"])}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data.write_json(root / "frozen_design.json", saved)
            path = root / job["output"]
            data.write_json(path, row)
            original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            provenance = {}
            with self.assertWarns(UserWarning):
                actual, rows = runner.audit(root, require_complete=False, provenance=provenance)
            self.assertEqual(actual, saved)
            self.assertEqual(len(rows), 1)
            self.assertEqual(provenance["design_sha256"], row["design_sha256"])
            with self.assertWarns(UserWarning), redirect_stdout(io.StringIO()) as out:
                self.assertEqual(runner.main(["--audit", "--allow-partial", "--output-root", str(root)]), 1)
            report = json.loads(out.getvalue())
            self.assertFalse(report["postprocessing"]["runtime_matches_fit"])
            self.assertEqual(report["design_sha256"], row["design_sha256"])
            self.assertEqual({p: p.read_bytes() for p in root.rglob("*") if p.is_file()}, original)
            for change in ("metrics", "targets", "input", "relabel", "graph"):
                bad = copy.deepcopy(row)
                if change == "metrics": bad["metrics"]["directed_tp"] += 1
                if change == "targets": bad["result"]["estimated_targets"][0, 0] = 0
                if change == "input": bad["result"]["common_input_sha256_after"] = "wrong"
                if change == "relabel": bad["design_sha256"] = data.digest(self.design)
                if change == "graph": bad["result"]["estimated_graph"][0, 0] = 1
                data.write_json(path, bad)
                with self.assertWarns(UserWarning), self.assertRaises(ValueError):
                    runner.audit(root, require_complete=False)

    def test_parallel_jobs_coexist_but_duplicate_and_plot_locks_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with runner.dispatch_lock(root, 0), runner.dispatch_lock(root, 1):
                with self.assertRaises(BlockingIOError):
                    with runner.dispatch_lock(root, 0): pass
                with (root / ".execution.lock").open("a") as lock:
                    with self.assertRaises(BlockingIOError): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_algorithm_timeout_is_saved_without_a_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(study, "fit_method", side_effect=TimeoutError("test deadline")):
                self.assertEqual(runner.fit_one(root, self.design, self.job()), 1)
            row = data.read_json(root / self.job()["output"])
            self.assertEqual(row["status"], "timeout")
            self.assertNotIn("result", row)
            study.validate_result(row, self.job(), self.design, allow_failure=True)
            with self.assertRaises(ValueError): study.validate_result(row, self.job(), self.design)

    def test_native_limit_saves_nonoptimal_incumbent_and_gap(self):
        method = "ps_mip_unknown"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(study, "fit_method", return_value=self.result(method, optimal=False)) as fit, \
                 mock.patch.object(runner, "wall_time_limit", side_effect=AssertionError("use native limit")):
                self.assertEqual(runner.fit_one(root, self.design, self.job(method)), 0)
            self.assertEqual(fit.call_args.args[0]["time_limit"], 3600)
            row = data.read_json(root / self.job(method)["output"])
            self.assertEqual(row["status"], "ok_nonoptimal")
            self.assertAlmostEqual(row["timing"]["budget_overrun_seconds"], .01)
            self.assertEqual(row["result"]["mip_gap"], .1)
            study.validate_result(row, row["job"], self.design)

    def test_baselines_use_one_hour_algorithm_timer(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(study, "fit_method", return_value=self.result()), \
                 mock.patch.object(runner, "wall_time_limit", wraps=wall_time_limit) as timer:
                self.assertEqual(runner.fit_one(Path(folder), self.design, self.job()), 0)
            timer.assert_called_once_with(3600, "igsp_oracle")

    def test_watchdog_failure_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner.freeze(root / "frozen_design.json", self.design)
            with mock.patch.object(runner, "run_child", return_value=(-9, True)) as child:
                self.assertEqual(runner.execute(root, self.design, jobs=[self.job()]), 1)
                self.assertEqual(child.call_args.args[2], 3720)
                child.reset_mock()
                self.assertEqual(runner.execute(root, self.design, jobs=[self.job()]), 1)
                child.assert_not_called()
            with self.assertRaisesRegex(ValueError, "missing"): runner.audit(root)
            _, rows = runner.audit(root, require_complete=False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "watchdog_timeout")

    def test_export_refuses_incomplete_and_extra_fragments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner.freeze(root / "frozen_design.json", self.design)
            with self.assertRaisesRegex(ValueError, "missing"): runner.export(root)
            self.assertFalse((root / "summary").exists())
            data.write_json(root / "scm_5/parts/igsp_oracle/setting_01.json", {})
            with self.assertRaisesRegex(ValueError, "extra"): runner.audit(root, require_complete=False)

    def test_complete_but_failed_run_exports_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner.freeze(root / "frozen_design.json", self.design)
            for job in self.design["jobs"]:
                data.write_json(root / job["output"], {"job": job, "design_sha256": data.digest(self.design),
                    "status": "process_failed", "returncode": 1, "elapsed_seconds": .1})
            with mock.patch("src.chamber_plotting.export") as plot, redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["--plot", "--output-root", str(root)]), 1)
                plot.assert_called_once()
            report = data.read_json(root / "completion.json")
            provenance = data.read_json(root / "summary/postprocessing_identity.json")
            self.assertEqual(provenance["runtime"], self.design["runtime"])
            self.assertEqual(provenance["design_sha256"], data.digest(self.design))
            self.assertTrue(report["postprocessing"]["runtime_matches_fit"])
            self.assertTrue(report["all_planned_accounted"])
            self.assertFalse(report["all_planned_successful"])
            self.assertEqual(report["status_counts"]["process_failed"], 63)

    def test_timeout_timer_interrupts_and_restores(self):
        with self.assertRaises(TimeoutError):
            with wall_time_limit(.02, "unit test"):
                time.sleep(.1)
        time.sleep(.03)

    def test_simple_quest_scripts_match_method_grid_and_budget(self):
        for method in study.METHODS:
            path = data.ROOT / "experiments/quest_jobs" / f"chamber_experiment_{method}.sh"
            subprocess.run(["bash", "-n", str(path)], check=True)
            text = path.read_text()
            count = len(runner.select_jobs(self.design, [method]))
            for expected in (f"--array=1-{count}%10", f"--methods {method}", "--setting", "--time-limit 3600",
                             "#SBATCH --time=01:10:00", "--cpus-per-task=8", "source activate python39"):
                self.assertIn(expected, text)
            self.assertLessEqual(len(text.splitlines()), 32)
            self.assertNotIn("sbatch", text)
            self.assertNotIn("mktemp", text)
        self.assertEqual(len(list((data.ROOT / "experiments/quest_jobs").glob("chamber_experiment_*.sh"))), 5)

    def test_reuse_preserves_original_rows_and_only_imports_unchanged_conditions(self):
        source = study.make_design(profile=study.LEGACY_PROFILE)
        with tempfile.TemporaryDirectory() as folder:
            source_root, root = Path(folder) / "old", Path(folder) / "new"
            data.write_json(source_root / "frozen_design.json", source)
            for job in source["jobs"]:
                data.write_json(source_root / job["output"], {"job": job, "design_sha256": data.digest(source),
                    "status": "process_failed", "returncode": 1, "elapsed_seconds": .1})
            originals = {p: p.read_bytes() for p in source_root.rglob("*.json")}
            result = runner.reuse_unchanged(root, source_root, self.design)
            self.assertEqual((result["reused"], result["new_fits_needed"], result["planned"]), (40, 23, 63))
            actual, rows = runner.audit(root, require_complete=False)
            self.assertEqual(set(r["job"]["method"] for r in rows), set(study.REUSABLE_METHODS))
            self.assertEqual(runner.completion_report(actual, rows)["reused"], 40)
            before = {p: p.read_bytes() for p in root.rglob("*.json")}
            runner.reuse_unchanged(root, source_root, self.design)
            self.assertEqual({p: p.read_bytes() for p in root.rglob("*.json")}, before)
            self.assertEqual({p: p.read_bytes() for p in source_root.rglob("*.json")}, originals)
            with self.assertRaisesRegex(ValueError, "missing"): runner.audit(root)
            with mock.patch.object(runner, "run_child") as child, redirect_stdout(io.StringIO()):
                runner.execute(root, self.design, jobs=[r["job"] for r in rows])
                child.assert_not_called()
            row = rows[0]
            for change in ("payload", "method", "setting", "source_code", "source_identity"):
                bad = copy.deepcopy(row)
                if change == "payload": bad["elapsed_seconds"] = 999
                if change == "method": bad["job"] = self.job("utigsp_unknown")
                if change == "setting": bad["reuse"]["source_row"]["job"]["time_limit"] = 300
                if change == "source_code": bad["reuse"]["source_design"]["implementation"]["MIP.py"] = "wrong"
                if change == "source_identity": bad["reuse"]["source_row"]["design_sha256"] = "wrong"
                with self.assertRaises(ValueError): study.validate_result(bad, bad["job"], self.design, allow_failure=True)

    def test_policy_matches_synthetic_method_names(self):
        from experiments import run_main_experiment as synthetic
        self.assertEqual(study.METHODS, ("ps_mip_unknown", "gies_oracle", "utigsp_unknown", "igsp_oracle", "gnies_unknown"))
        for method in study.METHODS:
            self.assertIn(method, synthetic.METHODS)


if __name__ == "__main__":
    unittest.main()
