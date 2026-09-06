import copy
import io
import json
from itertools import combinations
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import chamber_data as data
from src import chamber_experiment as study
from experiments import run_chamber_experiment as runner


class ChamberUnitTests(unittest.TestCase):
    def arrays(self, p=4):
        rng = np.random.default_rng(40)
        return [rng.normal(size=(100, p)) + e for e in range(4)]

    def test_affine_is_shared_and_means_are_retained(self):
        original = self.arrays()
        arrays, mean, scale = data.common_transform(original)
        for raw, value in zip(original, arrays):
            np.testing.assert_array_equal(value, (raw - mean) / scale)
        np.testing.assert_allclose(arrays[0].mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(arrays[0].std(axis=0), 1)
        self.assertGreater(arrays[-1].mean(), 2)
        centered = data.ps_arrays(arrays)
        for x in centered:
            np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-14)

    def test_constant_and_nonfinite_data_rejected(self):
        for replacement in (0, np.nan):
            arrays = self.arrays()
            arrays[2][:, 0] = replacement
            with self.assertRaises(ValueError): data.common_transform(arrays)

    def test_reference_and_node_order(self):
        adjacency = pd.DataFrame(0, index=data.NODES, columns=data.NODES)
        adjacency.loc[list(data.NODES[:3]), list(data.NODES[3:])] = 1
        for cfg in data.CONFIGURATIONS:
            truth, physical, programmed = data.reference_graph(adjacency, cfg)
            self.assertEqual(truth.sum(), 23)
            self.assertEqual(physical.sum(), 21)
            self.assertEqual(programmed.sum(), 2)
            self.assertEqual(truth[1, 2], int(cfg == "scm_4"))
            self.assertEqual(truth[2, 1], int(cfg == "scm_5"))
        adjacency.iloc[0, 3] = 0
        with self.assertRaises(ValueError): data.reference_graph(adjacency, "scm_4")

    def test_grid_enumeration_and_penalties(self):
        settings = {m: study.settings(m, [10000] * 4) for m in study.METHODS}
        self.assertEqual(sum(map(len, settings.values())), 97)
        self.assertEqual(study.GRAPH_MULTIPLIERS[0], 1/16)
        self.assertEqual(study.GRAPH_MULTIPLIERS[-1], 16)
        self.assertEqual(study.NATIVE_MULTIPLIERS[-1], 1024)
        for m, path in settings.items():
            for setting in path:
                if m.startswith("ps_mip"):
                    self.assertAlmostEqual(setting["graph_penalty"], setting["tuning_value"] * np.log(40000)/40000)
                    np.testing.assert_allclose(setting["target_penalty"], [16*np.log(40000)/40000] * 3)
                elif m in ("utigsp_present", "igsp_complete"):
                    self.assertEqual(setting["invariance_test"], "gaussian")
                    self.assertEqual(setting["alpha_inv"], 1e-5)

    def test_ps_target_information(self):
        documented = data.targets()
        self.assertIsNone(study.target_status("ps_mip_unknown", documented))
        np.testing.assert_array_equal(study.target_status("ps_mip_complete", documented), documented)
        present = study.target_status("ps_mip_present", documented)
        self.assertTrue(np.all(present[documented == 1] == 1))
        self.assertTrue(np.all(present[documented == 0] == -1))

    def test_bounds_match_core_local_optima_even_for_tight_sensors(self):
        from MIP_profiled import _best_local
        arrays = self.arrays()
        for a in arrays:
            a[:, 3] = a[:, 0] + .001*a[:, 3]
        arrays, _, _ = data.common_transform(arrays)
        preflight = data.profile_bounds(arrays)
        self.assertGreater(preflight["bounds"]["gamma_upper"], 10)
        self.assertEqual(preflight["local_optima_checked"], 4*8*8)
        centered = data.ps_arrays(arrays)
        sigma = [a.T @ a / len(a) for a in centered]
        b = preflight["bounds"]
        for j in range(4):
            others = [i for i in range(4) if i != j]
            for k in range(4):
                for parents in combinations(others, k):
                    for mask in range(8):
                        status = np.array([(mask >> e) & 1 for e in range(3)])
                        score, actual_mask, diagonal, coefficients = _best_local(
                            j, parents, sigma, np.full(4, .25), .01, np.full(3, .1), status,
                            b["coefficient_bound"], b["gamma_lower"], b["gamma_upper"])
                        self.assertEqual(actual_mask, mask)
                        self.assertGreater(diagonal, b["gamma_lower"])
                        self.assertLess(diagonal, b["gamma_upper"])
                        self.assertTrue(np.all(np.abs(coefficients) < b["coefficient_bound"]))

    def test_numeric_bounds_cannot_fix_singular_data(self):
        arrays = self.arrays()
        for a in arrays: a[:, 3] = a[:, 0]
        with self.assertRaises(ValueError): data.profile_bounds(arrays)

    def test_dimension_safe_metrics_and_reversed_edges(self):
        truth = np.zeros((4, 4), int)
        truth[0, 1] = truth[1, 2] = 1
        dag = truth.T
        meta = {"nodes": list("abcd"), "reference_dag": truth,
                "physical_edges": truth, "programmed_edges": np.zeros_like(truth)}
        m = study.metrics(dag, dag, meta)
        self.assertEqual((m["directed_tp"], m["directed_fp"]), (0, 2))
        self.assertEqual((m["skeleton_tp"], m["skeleton_fp"]), (2, 0))
        self.assertEqual(m["directed_fpr"], 2/10)
        self.assertEqual(m["physical_reversed"], 2)
        dag[0, 1] = 1
        with self.assertRaises(ValueError): study.validate_dag(dag, 4)

    def test_method_api_compatibility(self):
        self.assertEqual(study.check_apis()["igsp_invariance"], "gaussian")

    def test_adapters_receive_correct_information_and_gaussian_tests(self):
        arrays = self.arrays()
        documented = np.zeros((3, 4), int)
        documented[:, :3] = np.eye(3)
        empty = np.zeros((4, 4), int)
        for method, module in (("utigsp_present", "UTIGSP"), ("igsp_complete", "IGSP")):
            value = (empty, documented) if method.startswith("ut") else empty
            job = {"method": method, "setting": study.settings(method, [100]*4)[0], "time_limit": 30}
            with mock.patch(module + ".fit", return_value=value) as fit:
                result = study.fit_method(job, arrays, documented, {})
                self.assertEqual(fit.call_args.kwargs["invariance_test"], "gaussian")
                self.assertEqual(fit.call_args.kwargs["nruns"], 10)
                labels = fit.call_args.kwargs["known_interventions"] if method.startswith("ut") else fit.call_args.args[1]
                np.testing.assert_array_equal(labels, documented)
                self.assertEqual(result["common_input_sha256_before"], data.array_hash(arrays))
        for method in ("gies_complete", "gnies_present"):
            job = {"method": method, "setting": study.settings(method, [100]*4)[0], "time_limit": 30}
            if method.startswith("gies"):
                with mock.patch("GIES.fit", return_value=(empty, 0.)) as fit:
                    study.fit_method(job, arrays, documented, {})
                    np.testing.assert_array_equal(fit.call_args.args[1], documented)
            else:
                with mock.patch("gnies.fit", return_value=(0., empty, {0, 1, 2, 3})) as fit:
                    result = study.fit_method(job, arrays, documented, {})
                    self.assertTrue(fit.call_args.kwargs["center"])
                    self.assertEqual(fit.call_args.kwargs["known_targets"], {0, 1, 2})
                    np.testing.assert_array_equal(result["estimated_targets"], np.ones(4))
                with mock.patch("gnies.fit", return_value=(0., empty, {1, 2})):
                    with self.assertRaisesRegex(ValueError, "dropped"):
                        study.fit_method(job, arrays, documented, {})

    def test_unknown_ps_adapter_gets_none(self):
        arrays = self.arrays()
        documented = np.zeros((3, 4), int)
        bounds = data.profile_bounds(arrays)
        job = {"method": "ps_mip_unknown", "setting": study.settings("ps_mip_unknown", [100]*4)[0], "time_limit": 30}
        value = (np.eye(4), documented, 0., 1., .01, {"optimal": True})
        with mock.patch("MIP_profiled.optimization", return_value=value) as fit:
            result = study.fit_method(job, arrays, documented, bounds)
            self.assertIsNone(fit.call_args.kwargs["target_status"])
            self.assertEqual(result["screen_parent_sets"], 32)
            self.assertTrue(result["bounds_inactive"])

    def test_external_watchdog_kills_only_owned_group(self):
        child = mock.Mock(pid=54321)
        child.wait.side_effect = [subprocess.TimeoutExpired("owned", 1), -9]
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(runner.subprocess, "Popen", return_value=child), mock.patch.object(runner.os, "killpg") as kill:
                code, watchdog = runner.run_child(["owned"], Path(folder)/"fit.log", 1)
        self.assertEqual(code, -9)
        self.assertTrue(watchdog)
        self.assertEqual(kill.call_args.args[0], 54321)

    def test_freeze_refuses_changed_design(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frozen.json"
            runner.freeze(path, {"a": 1})
            runner.freeze(path, {"a": 1})
            with self.assertRaises(ValueError): runner.freeze(path, {"a": 2})

    def test_range_reader_refuses_full_or_truncated_response(self):
        class Response(io.BytesIO):
            def __init__(self, value, status, headers):
                super().__init__(value)
                self.status, self.headers = status, headers
        for response in (Response(b"abcd", 200, {}), Response(b"a", 206, {"Content-Range": "bytes 0-3/100"})):
            opener = mock.Mock(side_effect=[Response(b"", 200, {"Content-Length": "100", "ETag": '"tag"'}), response])
            remote = data.RemoteZip("https://example.org/public.zip", opener=opener)
            with self.assertRaises(ValueError): remote.read(4)


@unittest.skipUnless((data.DATA_ROOT / "scm_4/prepared.json").exists(), "prepared chamber data unavailable")
class ChamberIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = study.make_design()

    def test_prepared_data_and_design(self):
        self.assertEqual(len(self.design["jobs"]), 97)
        self.assertEqual(len({j["output"] for j in self.design["jobs"]}), 97)
        self.assertEqual(list(self.design["configurations"]), ["scm_4"])
        for cfg in self.design["configurations"]:
            arrays, meta, bounds = data.load(cfg)
            self.assertEqual([a.shape for a in arrays], [(10000, 10)] * 4)
            self.assertEqual(sum(map(sum, meta["reference_dag"])), 23)
            self.assertEqual(bounds["parent_sets"], 5120)
            self.assertEqual(bounds["local_optima_checked"], 40960)
            self.assertEqual(sum(j["configuration"] == cfg for j in self.design["jobs"]), 97)
        self.assertLess(data.read_json(data.DATA_ROOT/"source.json")["archive_bytes_read"], 100_000_000)

    def example(self, method="igsp_complete"):
        job = next(j for j in self.design["jobs"] if j["method"] == method)
        meta = self.design["configurations"][job["configuration"]]["data"]
        dag = np.zeros((10, 10), int)
        result = {"estimated_dag": dag, "estimated_graph": dag, "estimated_targets": data.targets(), "fit_seconds": .01,
                  "common_input_sha256_before": job["data_sha256"], "common_input_sha256_after": job["data_sha256"]}
        return {"job": job, "design_sha256": data.digest(self.design), "status": "ok", "result": result,
                "timing": {"fit_limit_seconds": job["time_limit"], "budget_overrun_seconds": 0.},
                "metrics": study.metrics(dag, dag, meta)}

    def test_saved_graph_counts_targets_and_identity_validated(self):
        row = self.example()
        study.validate_result(row, row["job"], self.design)
        for change in ("counts", "target", "input", "identity"):
            bad = copy.deepcopy(row)
            if change == "counts": bad["metrics"]["directed_tp"] += 1
            if change == "target": bad["result"]["estimated_targets"][0, 0] = 0
            if change == "input": bad["result"]["common_input_sha256_after"] = "wrong"
            if change == "identity": bad["design_sha256"] = "wrong"
            with self.assertRaises(ValueError): study.validate_result(bad, row["job"], self.design)

    def test_additional_present_targets_allowed(self):
        row = self.example("utigsp_present")
        row["result"]["estimated_targets"][0, 8] = 1
        study.validate_result(row, row["job"], self.design)
        row["result"]["estimated_targets"][0, 0] = 0
        with self.assertRaises(ValueError): study.validate_result(row, row["job"], self.design)

    def test_missing_fragments_and_failures_not_presented_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data.write_json(root/"frozen_design.json", self.design)
            with self.assertRaisesRegex(ValueError, "missing"): runner.audit(root)
            row = self.example()
            row["status"] = "failed"
            study.validate_result(row, row["job"], self.design, allow_failure=True)
            with self.assertRaises(ValueError): study.validate_result(row, row["job"], self.design)

    def test_resume_preserves_existing_failure_without_refitting(self):
        row = self.example()
        row["status"] = "failed"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / row["job"]["output"]
            data.write_json(path, row)
            before = path.read_bytes()
            with mock.patch.object(study, "fit_method") as fit:
                self.assertEqual(runner.fit_one(root, self.design, row["job"], data.DATA_ROOT), 1)
                fit.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_model_failure_is_persisted(self):
        job = self.design["jobs"][0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(study, "fit_method", side_effect=ValueError("deliberate test failure")):
                self.assertEqual(runner.fit_one(root, self.design, job, data.DATA_ROOT), 1)
            row = data.read_json(root / job["output"])
            self.assertEqual(row["status"], "failed")
            self.assertIn("deliberate test failure", row["traceback"])


class ChamberPlotTests(unittest.TestCase):
    def rows(self):
        return [{"status": status, "job": {"method": "ps_mip_complete", "setting": {"tuning_value": 1}},
                 "metrics": {"directed_fp": dfp, "skeleton_fp": sfp, "directed_tp": 20, "skeleton_tp": 21,
                             "directed_fpr": dfp/67, "skeleton_fpr": sfp/22, "directed_tpr": 20/23, "skeleton_tpr": 21/23}}
                for dfp, sfp, status in ((25, 20, "ok"), (26, 21, "ok_nonoptimal"), (0, 0, "failed"))]

    def test_panel_specific_inclusive_window_does_not_mutate_rows(self):
        from src.chamber_plotting import visible
        rows = self.rows()
        before = copy.deepcopy(rows)
        self.assertEqual(len(visible(rows, "directed", "sparse")), 1)
        self.assertEqual(len(visible(rows, "skeleton", "sparse")), 2)
        self.assertEqual(len(visible(rows, "directed", "full")), 2)
        self.assertEqual(rows, before)

    def test_plot_keeps_duplicate_points_and_does_not_connect(self):
        from src.chamber_plotting import plot_panels
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        row = self.rows()[0]
        plot_panels(axes, [row, copy.deepcopy(row)], ["ps_mip_complete"], "sparse")
        self.assertEqual(axes[0].get_xlim(), (0, 25))
        self.assertEqual(len(axes[0].collections[0].get_offsets()), 2)
        self.assertEqual(len(axes[0].lines), 0)
        plt.close(fig)

    def test_summary_distinguishes_accounted_and_successful(self):
        from src.chamber_plotting import summarize
        jobs = [{"configuration": cfg, "method": method} for cfg in data.CONFIGURATIONS for method in study.METHODS]
        result = summarize({"jobs": jobs}, [])
        self.assertFalse(result["all_planned_accounted"])
        self.assertFalse(result["all_planned_successful"])
        self.assertIsNone(result["configurations"]["scm_4"]["ps_mip_complete"]["best_at_directed_fp_budget"]["3"])


if __name__ == "__main__":
    unittest.main()
