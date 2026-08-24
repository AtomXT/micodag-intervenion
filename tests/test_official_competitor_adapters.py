from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import DCDI
import GIES
from experiments import run_main_experiment as main_experiment


class DCDIOfficialSourceTests(unittest.TestCase):
    def test_vendored_author_snapshot_matches_pinned_manifest(self):
        checkout, dirty = DCDI._validate_checkout(DCDI.DEFAULT_DCDI_PATH)
        self.assertEqual(checkout, DCDI.DEFAULT_DCDI_PATH.resolve())
        self.assertFalse(dirty)

    def test_command_calls_author_entrypoint_directly(self):
        command = DCDI._official_command(
            Path("/usr/bin/python3"),
            Path("/checkout"),
            ["--train", "--model", "DCDI-G"],
        )
        self.assertEqual(command[:2], ["/usr/bin/python3", "/checkout/main.py"])
        self.assertNotIn("-c", command)
        self.assertEqual(command[2:], ["--train", "--model", "DCDI-G"])

    def test_staging_contains_no_simulation_truth(self):
        data = [np.arange(6.0).reshape(3, 2), np.ones((2, 2))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            DCDI._stage_inputs(data, path)
            np.testing.assert_array_equal(
                np.load(path / "DAG1.npy"), np.zeros((2, 2), dtype=int)
            )
            np.testing.assert_array_equal(
                np.load(path / "data_interv1.npy"), np.vstack(data)
            )
            np.testing.assert_array_equal(
                np.loadtxt(path / "regime1.csv", delimiter=",", dtype=int),
                [0, 0, 0, 1, 1],
            )
            self.assertEqual((path / "intervention1.csv").read_text(), "\n" * 5)

    def test_persistent_identity_includes_runtime_and_dependency_lock(self):
        config = DCDI._configuration(
            [np.ones((2, 2)), np.zeros((1, 2))],
            0.1,
            0.001,
            True,
            7,
            False,
            False,
            8,
            0.01,
            100,
            {
                "python_version": "3.9.0",
                "torch_version": "2.2.2",
                "resolved_environment": {
                    "distributions": {"numpy": "1.26.4"},
                    "sha256": "environment-sha",
                },
            },
            "lock-sha",
        )
        self.assertEqual(config["dependency_lock_sha256"], "lock-sha")
        self.assertEqual(config["official_runtime"]["python_version"], "3.9.0")
        self.assertEqual(
            config["official_runtime"]["resolved_environment"]["sha256"],
            "environment-sha",
        )
        self.assertEqual(
            config["runtime_compatibility"]["version"],
            DCDI.DCDI_PLOT_COMPATIBILITY_VERSION,
        )

    def test_plot_compatibility_is_staged_outside_author_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = DCDI._prepare_runtime_compatibility(Path(directory))
            source = (path / "sitecustomize.py").read_text()
        self.assertIn('kwargs.pop("padding")', source)
        self.assertIn('kwargs.setdefault("pad_inches"', source)
        self.assertNotIn("external/dcdi", source)


class GIESOfficialSourceTests(unittest.TestCase):
    def test_adapter_passes_exact_environments_targets_and_penalty_to_package(self):
        data = [
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([[1.1, 1.2]]),
            np.array([[2.1, 2.2], [2.3, 2.4]]),
        ]
        targets = np.array([[1, 0], [0, 1]])
        captured = {}

        class FakeScore:
            def __init__(self, values, interventions, lmbda):
                captured["data"] = values
                captured["interventions"] = interventions
                captured["lambda"] = lmbda

        def fake_fit(score):
            self.assertIsInstance(score, FakeScore)
            return np.array([[0, 1], [1, 0]]), -12.5

        fake_module = SimpleNamespace(fit=fake_fit, fit_bic=mock.Mock())
        with mock.patch.object(
            GIES, "_load_gies_api", return_value=(fake_module, FakeScore)
        ):
            graph, score = GIES.fit(data, targets, lmbda=2.5, timeout=10)

        for observed, expected in zip(captured["data"], data):
            np.testing.assert_array_equal(observed, expected)
        self.assertEqual(captured["interventions"], [[], [0], [1]])
        self.assertEqual(captured["lambda"], 2.5)
        np.testing.assert_array_equal(graph, [[0, 1], [1, 0]])
        self.assertEqual(score, -12.5)

    def test_default_bic_calls_package_fit_bic(self):
        fit_bic = mock.Mock(return_value=(np.zeros((2, 2)), -3.0))
        fake_module = SimpleNamespace(fit=mock.Mock(), fit_bic=fit_bic)
        with mock.patch.object(
            GIES, "_load_gies_api", return_value=(fake_module, mock.Mock())
        ):
            graph, score = GIES.fit(
                [np.ones((3, 2)), np.ones((2, 2))],
                np.array([[1, 0]]),
            )
        self.assertEqual(fit_bic.call_args.args[1], [[], [0]])
        np.testing.assert_array_equal(graph, np.zeros((2, 2), dtype=int))
        self.assertEqual(score, -3.0)

    def test_adapter_delegates_search_and_score_to_requested_package(self):
        source = Path(GIES.__file__).read_text()
        self.assertIn("gies.fit_bic(arrays, interventions)", source)
        self.assertIn("score_class(arrays, interventions", source)
        self.assertIn("return gies.fit(score)", source)
        self.assertNotIn("def insert", source)
        self.assertNotIn("def delete", source)


class MainRunnerOfficialSourceTests(unittest.TestCase):
    def test_python_preflight_imports_official_competitor_apis(self):
        completed = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(
            main_experiment.subprocess, "run", return_value=completed
        ) as run:
            main_experiment._probe_selected_python_apis(
                ["utigsp_unknown", "gnies_unknown", "gies_oracle"]
            )
        probe = run.call_args.args[0][-1]
        compile(probe, "<official-api-preflight>", "exec")
        self.assertIn("unknown_target_igsp", probe)
        self.assertIn("import ges, gnies", probe)
        self.assertIn("from gies.scores import GaussIntL0Pen", probe)
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            main_experiment.RUNTIME_PROBE_TIMEOUT,
        )

    def test_tested_requirements_match_the_install_file(self):
        pins = {}
        for line in main_experiment.DEPENDENCY_LOCK_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, version = line.split("==", 1)
                pins[name] = version
        self.assertEqual(main_experiment.TESTED_PYTHON_REQUIREMENTS, pins)

    def test_resolved_environment_fingerprint_records_transitives(self):
        runtime = main_experiment._resolved_python_environment()
        self.assertEqual(len(runtime["sha256"]), 64)
        self.assertIn("numpy", runtime["distributions"])

    def test_failed_checkpoint_advances_unless_retry_is_requested(self):
        row = {"method_config": "same"}
        previous = {
            "status": "fit_error",
            "data_sha256": "data",
            "instance_sha256": "instance",
            "seed": "7",
            "method_config": "same",
        }
        self.assertTrue(
            main_experiment._checkpoint_matches(
                previous, row, "data", "instance", 7
            )
        )
        self.assertFalse(
            main_experiment._checkpoint_matches(
                previous, row, "data", "instance", 7, retry_failures=True
            )
        )

    def test_python_baseline_version_difference_is_recorded_not_rejected(self):
        with mock.patch.object(
            main_experiment.importlib.metadata,
            "version",
            return_value="9.9",
        ):
            installed = main_experiment._identify_python_distributions(
                ["gnies"]
            )
        self.assertEqual(installed, {"gnies": "9.9"})

    def test_missing_python_distribution_still_fails_preflight(self):
        with mock.patch.object(
            main_experiment.importlib.metadata,
            "version",
            side_effect=main_experiment.importlib.metadata.PackageNotFoundError,
        ):
            with self.assertRaisesRegex(RuntimeError, "gnies is not installed"):
                main_experiment._identify_python_distributions(
                    ["gnies"]
                )

    def test_dcdi_package_version_differences_are_not_rejected(self):
        args = SimpleNamespace(
            methods=["dcdi_g_unknown"],
            dcdi_root=Path("/official/dcdi"),
            rscript="Rscript",
        )
        reported = {
            "python_packages": {"numpy": "99.0", "matplotlib": "99.0"},
            "pcalg_version": "99.0",
            "sid_version": "99.0",
        }
        with (
            mock.patch.object(
                main_experiment,
                "_identify_python_distributions",
                return_value={"numpy": "99.0"},
            ),
            mock.patch.object(main_experiment, "_probe_selected_python_apis"),
            mock.patch.object(
                main_experiment, "_resolved_python_environment", return_value={}
            ),
            mock.patch.object(
                DCDI, "validate_official_install", return_value=reported
            ),
        ):
            runtime = main_experiment._validate_competitor_runtime(args)
        self.assertEqual(runtime["dcdi_g_unknown"], reported)

    def test_quest_preflight_checks_presence_not_versions(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "quest_jobs"
            / "main_experiment_array.sh"
        ).read_text()
        self.assertIn('required <- c("pcalg", "SID")', script)
        self.assertNotIn("packageVersion", script)

    def test_gnies_uses_author_greedy_search(self):
        calls = {}

        def fake_fit(data, **kwargs):
            calls.update(kwargs)
            return 0.0, np.zeros((2, 2), dtype=int), set()

        fake_module = SimpleNamespace(fit=fake_fit)
        args = SimpleNamespace(time_limit=5)
        setting = {"graph_penalty": 1.0}
        with mock.patch.dict(sys.modules, {"gnies": fake_module}):
            main_experiment._run_gnies([np.ones((3, 2))], args, setting)
        self.assertEqual(calls["approach"], "greedy")
        self.assertEqual(calls["lmbda"], 1.0)

    def test_all_competitor_configs_record_official_source(self):
        data = [np.ones((3, 2)), np.ones((2, 2))]
        args = SimpleNamespace(
            generator_numpy_version="2.0.2",
            time_limit=10,
            metric_time_limit=10,
            threads=1,
            rscript="Rscript",
        )
        for method in (
            "dcdi_g_unknown",
            "utigsp_unknown",
            "gnies_unknown",
            "gies_oracle",
        ):
            setting = main_experiment._method_settings(method, data)[0]
            _, _, _, config = main_experiment._declared_settings(
                method, data, setting, args
            )
            self.assertIn("source_url", config)
            self.assertTrue(
                "source_version" in config or "tested_version" in config
            )

    def test_utigsp_provenance_does_not_claim_an_unverified_commit(self):
        config = main_experiment.OFFICIAL_IMPLEMENTATIONS["utigsp_unknown"]
        self.assertNotIn("algorithm_source_commit", config)


if __name__ == "__main__":
    unittest.main()
