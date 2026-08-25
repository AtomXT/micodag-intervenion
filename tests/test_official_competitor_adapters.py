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
import IGSP
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

    def test_known_target_staging_repeats_exact_zero_based_masks(self):
        data = [
            np.arange(6.0).reshape(3, 2),
            np.ones((2, 2)),
            np.zeros((1, 2)),
        ]
        targets = np.array([[1, 0], [0, 1]], dtype=int)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            DCDI._stage_inputs(data, path, targets)
            rows = (path / "intervention1.csv").read_text().splitlines()
            np.testing.assert_array_equal(
                np.load(path / "DAG1.npy"), np.zeros((2, 2), dtype=int)
            )
        self.assertEqual(rows, ["", "", "", "0", "0", "1"])

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
        self.assertEqual(config["intervention_knowledge"], "unknown")
        self.assertEqual(config["official_runtime"]["python_version"], "3.9.0")
        self.assertEqual(
            config["official_runtime"]["resolved_environment"]["sha256"],
            "environment-sha",
        )
        self.assertEqual(
            config["runtime_compatibility"]["version"],
            DCDI.DCDI_PLOT_COMPATIBILITY_VERSION,
        )

    def test_known_target_configuration_and_output_are_mode_specific(self):
        data = [np.ones((2, 2)), np.zeros((1, 2))]
        targets = np.array([[1, 0]], dtype=int)
        config = DCDI._configuration(
            data,
            0.1,
            0.0,
            True,
            7,
            False,
            False,
            8,
            0.01,
            100,
            {},
            None,
            targets,
        )
        self.assertEqual(config["intervention_knowledge"], "known")
        self.assertEqual(len(config["known_targets_sha256"]), 64)

        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "experiment"
            train = experiment / "train"
            train.mkdir(parents=True)
            np.save(train / "DAG.npy", np.array([[0, 1], [0, 0]], dtype=int))
            DCDI._write_known_targets(experiment, targets)
            dag, observed = DCDI._read_outputs(experiment, 2, 1, targets)
        np.testing.assert_array_equal(dag, [[0, 1], [0, 0]])
        np.testing.assert_array_equal(observed, targets)

    def test_optimization_calls_official_known_intervention_path(self):
        data = [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[2.0, 1.0], [3.0, 0.0]]),
        ]
        targets = np.array([[1, 0]], dtype=int)
        captured = {}

        def fake_command(_interpreter, _checkout, arguments):
            captured["arguments"] = list(arguments)
            return [sys.executable, "fake_dcdi.py", *arguments]

        def fake_run(command, **_kwargs):
            experiment = Path(command[command.index("--exp-path") + 1])
            train = experiment / "train"
            train.mkdir(parents=True, exist_ok=True)
            np.save(train / "DAG.npy", np.array([[0, 1], [0, 0]], dtype=int))
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            with (
                mock.patch.object(
                    DCDI, "_validate_checkout", return_value=(checkout, False)
                ),
                mock.patch.object(DCDI, "_resolve_python", return_value=Path(sys.executable)),
                mock.patch.object(
                    DCDI,
                    "_validate_official_runtime",
                    return_value={"rscript": "/usr/bin/Rscript"},
                ),
                mock.patch.object(DCDI, "_official_command", side_effect=fake_command),
                mock.patch.object(DCDI.subprocess, "run", side_effect=fake_run),
            ):
                dag, observed, _, metadata = DCDI.optimization(
                    data,
                    graph_penalty=0.1,
                    target_penalty=0.0,
                    known_targets=targets,
                    normalize_data=False,
                    timeout=10,
                )

        arguments = captured["arguments"]
        knowledge_index = arguments.index("--intervention-knowledge")
        self.assertEqual(arguments[knowledge_index + 1], "known")
        self.assertNotIn("--coeff-interv-sparsity", arguments)
        self.assertEqual(metadata["intervention_knowledge"], "known")
        np.testing.assert_array_equal(dag, [[0, 1], [0, 0]])
        np.testing.assert_array_equal(observed, targets)

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


class IGSPOfficialSourceTests(unittest.TestCase):
    def test_adapter_supplies_exact_known_target_sets_to_causaldag(self):
        data = [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[2.0, 1.0], [3.0, 0.0]]),
            np.array([[1.0, 2.0], [0.0, 3.0]]),
        ]
        targets = np.array([[1, 0], [0, 1]], dtype=int)
        captured = {}

        class FakeTester:
            def __init__(self, test, suffstat, **kwargs):
                self.test = test
                self.suffstat = suffstat
                self.kwargs = kwargs

        class FakeDag:
            def to_amat(self, node_list):
                self.node_list = node_list
                return np.array([[0, 1], [0, 0]], dtype=int), node_list

        def fake_igsp(settings, nodes, ci_tester, invariance_tester, **kwargs):
            captured.update(
                settings=settings,
                nodes=nodes,
                ci_tester=ci_tester,
                invariance_tester=invariance_tester,
                kwargs=kwargs,
            )
            return FakeDag()

        api = (
            FakeTester,
            FakeTester,
            lambda obs, interventions: (obs, interventions),
            object(),
            fake_igsp,
            lambda obs: obs,
            object(),
        )
        with mock.patch.object(IGSP, "_load_causaldag_api", return_value=api):
            dag = IGSP.fit(data, targets, alpha=1e-3, depth=4, nruns=2, seed=7)

        self.assertEqual(
            captured["settings"],
            [{"interventions": {0}}, {"interventions": {1}}],
        )
        self.assertEqual(captured["nodes"], {0, 1})
        self.assertEqual(captured["kwargs"], {"depth": 4, "nruns": 2})
        np.testing.assert_array_equal(dag, [[0, 1], [0, 0]])


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
        self.assertIn("igsp", probe)
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
            methods=["dcdi_g_unknown", "dcdi_g_oracle"],
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
        self.assertEqual(runtime["dcdi_g_oracle"], reported)

    def test_quest_has_one_consistent_ten_replicate_array_per_method(self):
        job_dir = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "quest_jobs"
        )
        expected = {
            method: job_dir / f"main_experiment_{method}.sh"
            for method in main_experiment.METHODS
        }
        self.assertFalse((job_dir / "main_experiment_array.sh").exists())
        self.assertEqual(
            set(job_dir.glob("main_experiment_*.sh")), set(expected.values())
        )
        expected_schedule = {
            "ps_mip_unknown": ("normal", "24:00:00"),
            "dcdi_g_unknown": ("normal", "36:00:00"),
            "utigsp_unknown": ("short", "01:00:00"),
            "gnies_unknown": ("normal", "12:00:00"),
            "ps_mip_oracle": ("normal", "24:00:00"),
            "dcdi_g_oracle": ("normal", "24:00:00"),
            "igsp_oracle": ("normal", "12:00:00"),
            "gies_oracle": ("normal", "12:00:00"),
        }
        for method, path in expected.items():
            with self.subTest(method=method):
                script = path.read_text()
                partition, wall_time = expected_schedule[method]
                self.assertIn("#SBATCH --cpus-per-task=8", script)
                self.assertIn("#SBATCH --mem=16G", script)
                self.assertIn(f"#SBATCH --partition={partition}", script)
                self.assertIn(f"#SBATCH --time={wall_time}", script)
                self.assertIn("#SBATCH --array=1-10%10", script)
                self.assertNotIn("--threads", script)
                self.assertIn("--time-limit 3600", script)
                self.assertIn(f"--methods {method}", script)
                self.assertIn(f"parts/{method}/replicate_", script)
                if method.startswith("dcdi_g_"):
                    self.assertIn("module load R/4.4.0", script)
                else:
                    self.assertNotIn("module load R/4.4.0", script)

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
            rscript="Rscript",
        )
        for method in (
            "dcdi_g_unknown",
            "utigsp_unknown",
            "gnies_unknown",
            "dcdi_g_oracle",
            "igsp_oracle",
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
