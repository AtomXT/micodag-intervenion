from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import DCDI
from experiments import run_main_experiment
from experiments import test_main_experiment_setup
from src.main_experiment_cli import main_screen_alpha
from src.main_experiment_data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MASTER_SEED,
    load_main_experiment_instance,
    load_main_experiment_manifest,
)


class PersistedStandaloneDataTests(unittest.TestCase):
    def test_generated_smallest_main_instance_loads_with_exact_design(self):
        manifest = load_main_experiment_manifest(
            DEFAULT_DATA_ROOT, master_seed=DEFAULT_MASTER_SEED
        )
        data, dag, targets, info = load_main_experiment_instance(
            DEFAULT_DATA_ROOT,
            10,
            1,
            1,
            manifest=manifest,
            master_seed=DEFAULT_MASTER_SEED,
        )
        self.assertEqual([values.shape for values in data], [(1000, 10)] + [(200, 10)] * 5)
        self.assertEqual(dag.shape, (10, 10))
        self.assertEqual(targets.shape, (5, 10))
        self.assertEqual(len(info["data_sha256"]), 64)

    def test_screening_rule_is_five_times_requested_order(self):
        data = [np.ones((1000, 10))]
        self.assertAlmostEqual(
            main_screen_alpha(data),
            5.0 * np.sqrt(np.log(10) / 1000),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            main_screen_alpha(data, float("nan"))

    def test_active_method_entrypoints_do_not_reference_legacy_data(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("MIP.py", "MIP_profiled.py", "DCDI.py", "UTIGSP.py", "GIES.py"):
            source = (root / name).read_text()
            self.assertNotIn("SyntheticData", source, name)
            self.assertNotIn("legacy_pipeline", source, name)
            self.assertIn("load_cli_instance", source, name)


class NoSaveSetupCheckTests(unittest.TestCase):
    @staticmethod
    def _namespace(root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            data_root=root,
            seed=DEFAULT_MASTER_SEED,
            threads=1,
            time_limit=10.0,
            metric_time_limit=10.0,
            dcdi_root=root / "external" / "dcdi",
            rscript="Rscript",
        )

    @staticmethod
    def _fixture():
        data = [np.ones((4, 2))] + [np.ones((3, 2)) for _ in range(5)]
        dag = np.array([[0, 1], [0, 0]], dtype=int)
        targets = np.array([[1, 0]] * 5, dtype=int)
        info = {
            "seed": 7,
            "data_sha256": "d" * 64,
            "instance_sha256": "i" * 64,
        }
        return data, dag, targets, info

    def test_runs_all_primary_methods_without_an_output_path(self):
        data, dag, targets, info = self._fixture()
        calls = []

        def fake_run(method, *_args, **_kwargs):
            calls.append(method)
            return {
                "estimated_icpdag": np.zeros((2, 2), dtype=int),
                "fit_seconds": 0.01,
            }

        def fake_metrics(row, _estimated, _truth, _limit):
            row.update(d_cpdag=0, fdp=0.0, tdp=1.0, metric_seconds=0.01)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    test_main_experiment_setup,
                    "load_main_experiment_manifest",
                    return_value={"provenance": {"numpy_version": "2.4.6"}},
                ),
                mock.patch.object(
                    test_main_experiment_setup,
                    "load_main_experiment_instance",
                    return_value=(data, dag, targets, info),
                ) as load_instance,
                mock.patch.object(
                    test_main_experiment_setup,
                    "interventional_cpdag",
                    return_value=np.zeros((2, 2), dtype=int),
                ),
                mock.patch.object(
                    run_main_experiment,
                    "_validate_competitor_runtime",
                    return_value={"shared_python": {}},
                ),
                mock.patch.object(
                    run_main_experiment, "_run_method", side_effect=fake_run
                ) as run_method,
                mock.patch.object(
                    run_main_experiment, "_add_metrics", side_effect=fake_metrics
                ),
                mock.patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                code = test_main_experiment_setup.run_check(self._namespace(root))

        self.assertEqual(code, 0)
        self.assertEqual(
            load_instance.call_args.args[1:4],
            (
                test_main_experiment_setup.TEST_P,
                test_main_experiment_setup.TEST_EDGE_MULTIPLIER,
                test_main_experiment_setup.TEST_REPLICATE,
            ),
        )
        self.assertEqual(calls, list(test_main_experiment_setup.SETUP_METHOD_ORDER))
        self.assertTrue(
            all(call.args[3].transient_artifacts for call in run_method.call_args_list)
        )
        self.assertTrue(
            all(
                call.args[4]["setting_id"]
                == run_main_experiment.PRIMARY_SETTING_IDS[call.args[0]]
                for call in run_method.call_args_list
            )
        )
        printed = output.getvalue()
        self.assertIn("Performance:", printed)
        self.assertIn("tuning_parameter", printed)
        self.assertIn("saved outputs: none", printed)

    def test_one_method_failure_does_not_prevent_later_methods(self):
        data, dag, targets, info = self._fixture()
        calls = []

        def fake_run(method, *_args, **_kwargs):
            calls.append(method)
            if method == run_main_experiment.METHODS[0]:
                raise RuntimeError("expected failure")
            return {
                "estimated_icpdag": np.zeros((2, 2), dtype=int),
                "fit_seconds": 0.01,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    test_main_experiment_setup,
                    "load_main_experiment_manifest",
                    return_value={"provenance": {"numpy_version": "2.4.6"}},
                ),
                mock.patch.object(
                    test_main_experiment_setup,
                    "load_main_experiment_instance",
                    return_value=(data, dag, targets, info),
                ),
                mock.patch.object(
                    test_main_experiment_setup,
                    "interventional_cpdag",
                    return_value=np.zeros((2, 2), dtype=int),
                ),
                mock.patch.object(
                    run_main_experiment,
                    "_validate_competitor_runtime",
                    return_value={"shared_python": {}},
                ),
                mock.patch.object(
                    run_main_experiment, "_run_method", side_effect=fake_run
                ),
                mock.patch.object(
                    run_main_experiment,
                    "_add_metrics",
                    side_effect=lambda row, *_: row.update(
                        d_cpdag=0, fdp=0.0, tdp=1.0, metric_seconds=0.01
                    ),
                ),
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = test_main_experiment_setup.run_check(self._namespace(root))

        self.assertEqual(code, 1)
        self.assertEqual(calls, list(test_main_experiment_setup.SETUP_METHOD_ORDER))

    def test_metric_failure_does_not_prevent_later_methods(self):
        data, dag, targets, info = self._fixture()
        metric_calls = []

        def fake_metrics(row, *_args):
            metric_calls.append(row["method"])
            if len(metric_calls) == 1:
                raise TimeoutError("expected metric failure")
            row.update(d_cpdag=0, fdp=0.0, tdp=1.0, metric_seconds=0.01)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    test_main_experiment_setup,
                    "load_main_experiment_manifest",
                    return_value={"provenance": {"numpy_version": "2.4.6"}},
                ),
                mock.patch.object(
                    test_main_experiment_setup,
                    "load_main_experiment_instance",
                    return_value=(data, dag, targets, info),
                ),
                mock.patch.object(
                    test_main_experiment_setup,
                    "interventional_cpdag",
                    return_value=np.zeros((2, 2), dtype=int),
                ),
                mock.patch.object(
                    run_main_experiment,
                    "_validate_competitor_runtime",
                    return_value={"shared_python": {}},
                ),
                mock.patch.object(
                    run_main_experiment,
                    "_run_method",
                    return_value={
                        "estimated_icpdag": np.zeros((2, 2), dtype=int),
                        "fit_seconds": 0.01,
                    },
                ),
                mock.patch.object(
                    run_main_experiment, "_add_metrics", side_effect=fake_metrics
                ),
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = test_main_experiment_setup.run_check(self._namespace(root))

        self.assertEqual(code, 1)
        self.assertEqual(
            metric_calls, list(test_main_experiment_setup.SETUP_METHOD_ORDER)
        )

    def test_cli_rejects_nonpositive_limits_before_loading_data(self):
        for invalid in ("0", "nan", "inf"):
            with self.subTest(invalid=invalid):
                with (
                    mock.patch(
                        "sys.argv",
                        [
                            "test_main_experiment_setup.py",
                            "--time-limit",
                            invalid,
                        ],
                    ),
                    mock.patch("sys.stderr", new_callable=io.StringIO),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        test_main_experiment_setup.main()
                self.assertEqual(raised.exception.code, 2)

    def test_dcdi_transient_mode_never_requires_or_retains_output_storage(self):
        args = SimpleNamespace(
            official_runtime={"dcdi_g_unknown": {}},
            dcdi_root=Path("/official/dcdi"),
            time_limit=10,
            rscript="Rscript",
            transient_artifacts=True,
        )
        setting = {"graph_penalty": 0.1, "target_penalty": 0.001}
        data = [np.ones((3, 2)), np.ones((2, 2))]
        with (
            mock.patch.object(
                DCDI,
                "optimization",
                return_value=(
                    np.zeros((2, 2), dtype=int),
                    np.zeros((1, 2), dtype=int),
                    0.5,
                    {"artifact_dir": None, "cached": False},
                ),
            ) as optimization,
            mock.patch(
                "src.utils.interventional_cpdag",
                return_value=np.zeros((2, 2), dtype=int),
            ),
        ):
            result = run_main_experiment._run_dcdi(data, args, setting, 2, 1, 1, 7)

        self.assertIsNone(optimization.call_args.kwargs["artifact_dir"])
        self.assertIsNone(result["artifact_dir"])


if __name__ == "__main__":
    unittest.main()
