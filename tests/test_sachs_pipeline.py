from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import UTIGSP
from analysis import aggregate_sachs_roc
from analysis import plot_sachs_roc
from experiments import run_sachs_roc
from src import sachs_data


def _dummy_sachs_data():
    return [
        np.zeros((rows, len(sachs_data.SACHS_NODE_NAMES)), dtype=float)
        for rows in sachs_data.SACHS_SAMPLE_COUNTS
    ]


def _runner_args(method, setting_index, output, **overrides):
    values = {
        "method": method,
        "setting_index": setting_index,
        "data_root": Path("/unused/sachs"),
        "output": Path(output),
        "seed": run_sachs_roc.DEFAULT_METHOD_SEED,
        "threads": 8,
        "time_limit": 60.0,
        "retry_failures": False,
        "retry_nonoptimal": False,
        "overwrite": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SachsDataContractTests(unittest.TestCase):
    def test_exact_environment_and_truth_contract(self):
        self.assertEqual(
            sachs_data.SACHS_NODE_NAMES,
            (
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
            ),
        )
        self.assertEqual(sachs_data.SACHS_SAMPLE_COUNTS, (1755, 911, 723, 810, 799, 848))
        self.assertEqual(sachs_data.SACHS_TOTAL_SAMPLES, 5846)
        self.assertEqual(int(sachs_data.consensus_dag().sum()), 18)
        self.assertEqual(
            np.flatnonzero(sachs_data.intended_target_matrix()).tolist(),
            [6, 19, 25, 34, 48],
        )
        self.assertTrue(
            all(
                sachs_data.SOURCE_COMMIT in sachs_data.source_url(environment)
                for environment in sachs_data.SACHS_ENVIRONMENTS
            )
        )

    def test_prepared_default_package_loads_without_transforming_values(self):
        if not (sachs_data.DEFAULT_DATA_ROOT / sachs_data.MANIFEST_NAME).is_file():
            self.skipTest("run experiments/prepare_sachs_data.py first")
        data, dag, targets, info = sachs_data.load_sachs_data()
        self.assertEqual(
            [values.shape for values in data],
            [(n, 11) for n in sachs_data.SACHS_SAMPLE_COUNTS],
        )
        np.testing.assert_array_equal(dag, sachs_data.consensus_dag())
        np.testing.assert_array_equal(targets, sachs_data.intended_target_matrix())
        self.assertEqual(
            info["data_sha256"],
            "e9ce21c7b59bc711fd15a108e1b2fc6b28b2df9f51750099b1dc46b524524897",
        )


class SachsRunnerTests(unittest.TestCase):
    def test_all_paths_are_predeclared_on_the_frozen_sample_size(self):
        data = _dummy_sachs_data()
        self.assertEqual(
            [len(run_sachs_roc.method_settings(method, data)) for method in run_sachs_roc.METHODS],
            [9, 8, 6, 9, 8, 9, 8, 6],
        )
        ps_unknown = run_sachs_roc.method_settings("ps_mip_unknown", data)
        ps_intended = run_sachs_roc.method_settings("ps_mip_intended", data)
        ps_base = np.log(5846) / 5846
        np.testing.assert_allclose(
            [setting["graph_penalty"] for setting in ps_unknown],
            np.asarray(run_sachs_roc.PS_BIC_MULTIPLIERS) * ps_base,
        )
        np.testing.assert_allclose(
            [setting["target_penalty"] for setting in ps_unknown],
            run_sachs_roc.PS_TARGET_BIC_MULTIPLIER * ps_base,
        )
        self.assertEqual(
            [setting["target_bic_multiplier"] for setting in ps_unknown],
            [run_sachs_roc.PS_TARGET_BIC_MULTIPLIER] * len(ps_unknown),
        )
        self.assertTrue(
            all(setting["setting_id"].endswith("_target_x16") for setting in ps_unknown)
        )
        self.assertEqual(ps_intended, ps_unknown)
        self.assertEqual(
            run_sachs_roc.target_knowledge("ps_mip_intended"),
            "intended_targets_known_present",
        )
        self.assertEqual(
            [
                setting["alpha"]
                for setting in run_sachs_roc.method_settings(
                    "utigsp_unknown", data
                )
            ],
            list(run_sachs_roc.UTIGSP_CI_ALPHAS),
        )
        gnies = run_sachs_roc.method_settings("gnies_unknown", data)
        np.testing.assert_allclose(
            [setting["lambda"] for setting in gnies],
            np.asarray(run_sachs_roc.GNIES_LAMBDAS) * np.log(5846),
        )
        gies = run_sachs_roc.method_settings("gies_oracle", data)
        np.testing.assert_allclose(
            [setting["lambda"] for setting in gies],
            np.asarray(run_sachs_roc.GIES_BIC_MULTIPLIERS) * 0.5 * np.log(5846),
        )
        self.assertEqual(run_sachs_roc.COMPLETE_SCREEN_EDGES, 55)
        self.assertEqual(run_sachs_roc.COMPLETE_PARENT_SETS, 11264)
        fingerprint = run_sachs_roc._implementation_fingerprint(
            "ps_mip_unknown"
        )
        self.assertEqual(
            fingerprint["protocol_revision"], run_sachs_roc.PROTOCOL_REVISION
        )
        self.assertEqual(len(fingerprint["sha256"]), 64)
        self.assertIn(
            "experiments/run_sachs_roc.py", fingerprint["files"]
        )
        self.assertIn("MIP_profiled.py", fingerprint["files"])

    def test_ps_preprocessing_uses_context_means_and_observational_scale(self):
        observational = np.array([[1.0, 10.0], [3.0, 14.0]])
        intervention = np.array([[102.0, 8.0], [0.0, 16.0]])
        transformed = run_sachs_roc._ps_data([observational, intervention])
        np.testing.assert_allclose(transformed[0], [[-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_allclose(transformed[1], [[51.0, -2.0], [-51.0, 2.0]])

    def test_ps_precision_scales_on_the_diagonal_are_not_graph_loops(self):
        observational = np.vstack([np.zeros(11), np.ones(11)])
        data = [observational.copy() for _ in range(6)]
        gamma = np.eye(11)
        gamma[0, 1] = 0.25
        metadata = {
            "objective_bound": 1.0,
            "solver_status": "OPTIMAL",
            "solver_status_code": 2,
            "optimal": True,
            "solution_count": 1,
            "node_count": 0,
        }
        optimization_result = (
            gamma,
            np.zeros((5, 11), dtype=np.uint8),
            0.0,
            1.0,
            0.01,
            metadata,
        )
        fake_gurobi = SimpleNamespace(setParam=mock.Mock())
        fake_mip = SimpleNamespace(
            optimization=mock.Mock(return_value=optimization_result)
        )
        setting = {"graph_penalty": 0.1, "target_penalty": 0.1}
        with (
            mock.patch.dict(
                sys.modules,
                {"gurobipy": fake_gurobi, "MIP_profiled": fake_mip},
            ),
            mock.patch("src.utils.interventional_cpdag", side_effect=lambda dag, _targets: dag),
        ):
            result = run_sachs_roc._run_ps_mip(
                data,
                None,
                setting,
                target_knowledge="unknown",
                seed=17,
                threads=1,
                time_limit=10,
            )
        self.assertEqual(int(np.diag(result["estimated_dag"]).sum()), 0)
        self.assertEqual(result["estimated_dag"][0, 1], 1)
        fake_gurobi.setParam.assert_any_call("Seed", 17)

    def test_unknown_dispatch_never_receives_intended_targets(self):
        data = _dummy_sachs_data()
        targets = sachs_data.intended_target_matrix()
        args = _runner_args("ps_mip_unknown", 0, "/tmp/unused.csv")
        setting = run_sachs_roc.method_settings("ps_mip_unknown", data)[0]
        with mock.patch.object(run_sachs_roc, "_run_ps_mip", return_value={}) as fit:
            run_sachs_roc._run_method("ps_mip_unknown", data, targets, setting, args)
        self.assertIsNone(fit.call_args.args[1])
        self.assertEqual(fit.call_args.kwargs["target_knowledge"], "unknown")

        args.method = "utigsp_unknown"
        setting = run_sachs_roc.method_settings("utigsp_unknown", data)[0]
        with mock.patch.object(run_sachs_roc, "_run_utigsp", return_value={}) as fit:
            run_sachs_roc._run_method("utigsp_unknown", data, targets, setting, args)
        self.assertIsNone(fit.call_args.kwargs["known_interventions"])

    def test_ps_mip_passes_the_partial_target_status_to_the_optimizer(self):
        observational = np.vstack([np.zeros(11), np.ones(11)])
        data = [observational.copy() for _ in range(6)]
        intended = sachs_data.intended_target_matrix()
        target_status = run_sachs_roc._known_present_target_status(intended)
        returned_targets = intended.copy()
        returned_targets[0, 0] = 1
        metadata = {
            "objective_bound": 1.0,
            "solver_status": "OPTIMAL",
            "solver_status_code": 2,
            "optimal": True,
            "solution_count": 1,
            "node_count": 0,
        }
        fake_gurobi = SimpleNamespace(setParam=mock.Mock())
        fake_mip = SimpleNamespace(
            optimization=mock.Mock(
                return_value=(
                    np.eye(11),
                    returned_targets,
                    0.0,
                    1.0,
                    0.01,
                    metadata,
                )
            )
        )
        setting = {"graph_penalty": 0.1, "target_penalty": 0.1}
        with (
            mock.patch.dict(
                sys.modules,
                {"gurobipy": fake_gurobi, "MIP_profiled": fake_mip},
            ),
            mock.patch(
                "src.utils.interventional_cpdag", side_effect=lambda dag, _targets: dag
            ),
        ):
            result = run_sachs_roc._run_ps_mip(
                data,
                target_status,
                setting,
                target_knowledge="intended_targets_known_present",
                seed=17,
                threads=1,
                time_limit=10,
            )

        np.testing.assert_array_equal(
            fake_mip.optimization.call_args.kwargs["target_status"], target_status
        )
        np.testing.assert_array_equal(result["estimated_targets"], returned_targets)

    def test_intended_ps_mip_fixes_ones_and_leaves_everything_else_unknown(self):
        data = _dummy_sachs_data()
        intended = sachs_data.intended_target_matrix()
        expected_status = np.where(intended == 1, 1, -1)
        np.testing.assert_array_equal(
            run_sachs_roc._known_present_target_status(intended),
            expected_status,
        )

        args = _runner_args("ps_mip_intended", 0, "/tmp/unused.csv")
        setting = run_sachs_roc.method_settings("ps_mip_intended", data)[0]
        with mock.patch.object(run_sachs_roc, "_run_ps_mip", return_value={}) as fit:
            run_sachs_roc._run_method(
                "ps_mip_intended", data, intended, setting, args
            )
        np.testing.assert_array_equal(fit.call_args.args[1], expected_status)
        self.assertEqual(
            fit.call_args.kwargs["target_knowledge"],
            "intended_targets_known_present",
        )

    def test_intended_utigsp_receives_only_the_declared_known_present_mask(self):
        data = _dummy_sachs_data()
        targets = sachs_data.intended_target_matrix()
        args = _runner_args("utigsp_intended", 0, "/tmp/unused.csv")
        setting = run_sachs_roc.method_settings("utigsp_intended", data)[0]
        with mock.patch.object(run_sachs_roc, "_run_utigsp", return_value={}) as fit:
            run_sachs_roc._run_method("utigsp_intended", data, targets, setting, args)
        np.testing.assert_array_equal(
            fit.call_args.kwargs["known_interventions"], targets
        )

    def test_roc_counts_use_directed_and_skeleton_universes(self):
        truth = sachs_data.consensus_dag()
        exact = run_sachs_roc.roc_counts(truth, truth)
        self.assertEqual((exact["directed_tp"], exact["directed_fp"]), (18, 0))
        self.assertEqual((exact["directed_fn"], exact["directed_tn"]), (0, 92))
        self.assertEqual((exact["skeleton_tp"], exact["skeleton_fp"]), (18, 0))
        self.assertEqual((exact["skeleton_fn"], exact["skeleton_tn"]), (0, 37))

        reversed_dag = truth.T
        reversed_counts = run_sachs_roc.roc_counts(reversed_dag, truth)
        self.assertEqual((reversed_counts["directed_tp"], reversed_counts["directed_fp"]), (0, 18))
        self.assertEqual((reversed_counts["skeleton_tp"], reversed_counts["skeleton_fp"]), (18, 0))

    def test_successful_checkpoint_skips_and_nonoptimal_can_be_retried(self):
        data = _dummy_sachs_data()
        truth = sachs_data.consensus_dag()
        targets = sachs_data.intended_target_matrix()
        info = {"data_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            args = _runner_args("ps_mip_unknown", 0, output)
            setting = run_sachs_roc.method_settings(args.method, data)[0]
            row = run_sachs_roc._base_row(args, data, info, setting)
            row.update({"status": "ok_nonoptimal", "optimal": False})
            run_sachs_roc._write_row(output, row)

            with (
                mock.patch.object(
                    run_sachs_roc,
                    "load_sachs_data",
                    return_value=(data, truth, targets, info),
                ),
                mock.patch.object(run_sachs_roc, "_run_method") as fit,
            ):
                observed = run_sachs_roc.run(args)
            self.assertEqual(observed["status"], "ok_nonoptimal")
            fit.assert_not_called()

            args.retry_nonoptimal = True
            successful = {
                "estimated_dag": np.zeros((11, 11), dtype=np.uint8),
                "estimated_graph": np.zeros((11, 11), dtype=np.uint8),
                "estimated_targets": np.zeros((5, 11), dtype=np.uint8),
                "graph_representation": "dag_and_i_cpdag",
                "fit_seconds": 0.01,
                "optimal": True,
                "preprocessing": run_sachs_roc.METHOD_PREPROCESSING[args.method],
            }
            with (
                mock.patch.object(
                    run_sachs_roc,
                    "load_sachs_data",
                    return_value=(data, truth, targets, info),
                ),
                mock.patch.object(
                    run_sachs_roc, "_run_method", return_value=successful
                ) as fit,
            ):
                observed = run_sachs_roc.run(args)
            self.assertEqual(observed["status"], "ok")
            fit.assert_called_once()

    def test_overwrite_recovers_an_unparseable_fragment(self):
        data = _dummy_sachs_data()
        truth = sachs_data.consensus_dag()
        targets = sachs_data.intended_target_matrix()
        info = {"data_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            output.write_text("truncated,old\n", encoding="utf-8")
            args = _runner_args("utigsp_unknown", 0, output, overwrite=True)
            successful = {
                "estimated_dag": np.zeros((11, 11), dtype=np.uint8),
                "estimated_graph": np.zeros((11, 11), dtype=np.uint8),
                "estimated_targets": np.zeros((5, 11), dtype=np.uint8),
                "graph_representation": "dag_and_i_cpdag",
                "fit_seconds": 0.01,
                "preprocessing": run_sachs_roc.METHOD_PREPROCESSING[args.method],
            }
            with (
                mock.patch.object(
                    run_sachs_roc,
                    "load_sachs_data",
                    return_value=(data, truth, targets, info),
                ),
                mock.patch.object(
                    run_sachs_roc, "_run_method", return_value=successful
                ),
            ):
                observed = run_sachs_roc.run(args)
            self.assertEqual(observed["status"], "ok")
            self.assertEqual(run_sachs_roc._read_one_row(output)["status"], "ok")

    def test_failed_checkpoint_cannot_report_a_false_success(self):
        data = _dummy_sachs_data()
        truth = sachs_data.consensus_dag()
        targets = sachs_data.intended_target_matrix()
        info = {"data_sha256": "d" * 64}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            args = _runner_args("utigsp_unknown", 0, output)
            setting = run_sachs_roc.method_settings(args.method, data)[0]
            row = run_sachs_roc._base_row(args, data, info, setting)
            row.update({"status": "fit_error", "error": "TimeoutError: test"})
            run_sachs_roc._write_row(output, row)
            with (
                mock.patch.object(
                    run_sachs_roc,
                    "load_sachs_data",
                    return_value=(data, truth, targets, info),
                ),
                mock.patch.object(run_sachs_roc, "_run_method") as fit,
                self.assertRaisesRegex(RuntimeError, "--retry-failures"),
            ):
                run_sachs_roc.run(args)
            fit.assert_not_called()

    def test_overlapping_attempts_for_one_fragment_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            args = _runner_args("utigsp_unknown", 0, output)
            with (
                run_sachs_roc._output_lock(output),
                self.assertRaisesRegex(RuntimeError, "already working"),
            ):
                run_sachs_roc.run(args)


class UTIGSPKnownTargetTests(unittest.TestCase):
    def test_known_ones_are_seeded_but_zeroes_remain_discoverable(self):
        captured = {}

        class FakeDag:
            def to_amat(self, node_list):
                return np.zeros((len(node_list), len(node_list)), dtype=int), node_list

        def unknown_target_igsp(settings, nodes, *_args, **_kwargs):
            captured["settings"] = settings
            captured["nodes"] = nodes
            # Deliberately omit both supplied targets, matching the fact that
            # causaldag's return value need not repeat known-present seeds.
            return FakeDag(), [{2}, set()]

        api = (
            lambda *args, **kwargs: (args, kwargs),
            lambda *args, **kwargs: (args, kwargs),
            lambda *args, **kwargs: "invariance_suffstat",
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: "ci_suffstat",
            lambda *args, **kwargs: None,
            unknown_target_igsp,
        )
        data = [np.ones((3, 3)), np.ones((2, 3)), np.ones((2, 3))]
        known = np.array([[1, 0, 0], [0, 1, 0]], dtype=int)
        with mock.patch.object(UTIGSP, "_load_causaldag_api", return_value=api):
            _dag, learned = UTIGSP.fit(
                data,
                known_interventions=known,
                alpha=0.1,
                alpha_inv=1e-5,
                nruns=1,
            )
        self.assertEqual(
            captured["settings"],
            [{"known_interventions": {0}}, {"known_interventions": {1}}],
        )
        self.assertEqual(captured["nodes"], {0, 1, 2})
        np.testing.assert_array_equal(learned, [[1, 0, 1], [0, 1, 0]])

    def test_known_target_shape_and_values_are_strict(self):
        with self.assertRaisesRegex(ValueError, "binary with shape"):
            UTIGSP._known_intervention_sets(np.zeros((3, 2)), 2, 2)
        with self.assertRaisesRegex(ValueError, "binary with shape"):
            UTIGSP._known_intervention_sets(np.array([[2, 0], [0, 1]]), 2, 2)


class SachsAggregationAndPlotTests(unittest.TestCase):
    def test_aggregation_schema_and_method_specific_target_contracts(self):
        self.assertEqual(
            aggregate_sachs_roc.REQUIRED_FIELDS,
            tuple(run_sachs_roc.RESULT_FIELDS),
        )
        with tempfile.TemporaryDirectory() as directory:
            incomplete = Path(directory) / "incomplete.csv"
            fields = [
                field
                for field in run_sachs_roc.RESULT_FIELDS
                if field != "objective_bound"
            ]
            incomplete.write_text(",".join(fields) + "\n", encoding="utf-8")
            _rows, file_errors = aggregate_sachs_roc._read_results([incomplete])
            self.assertEqual(file_errors[0]["missing"], ["objective_bound"])

        intended = sachs_data.intended_target_matrix()
        zeros = np.zeros_like(intended)

        parsed, _canonical = aggregate_sachs_roc._parse_estimated_targets(
            json.dumps(zeros.tolist()), "ps_mip_unknown", intended
        )
        np.testing.assert_array_equal(parsed, zeros)

        learned_with_off_target = intended.copy()
        learned_with_off_target[0, 0] = 1
        parsed, _canonical = aggregate_sachs_roc._parse_estimated_targets(
            json.dumps(learned_with_off_target.tolist()),
            "utigsp_intended",
            intended,
        )
        np.testing.assert_array_equal(parsed, learned_with_off_target)
        parsed, _canonical = aggregate_sachs_roc._parse_estimated_targets(
            json.dumps(learned_with_off_target.tolist()),
            "ps_mip_intended",
            intended,
        )
        np.testing.assert_array_equal(parsed, learned_with_off_target)

        missing_known = intended.copy()
        missing_known[0, 6] = 0
        with self.assertRaisesRegex(ValueError, "known-present"):
            aggregate_sachs_roc._parse_estimated_targets(
                json.dumps(missing_known.tolist()),
                "utigsp_intended",
                intended,
            )
        with self.assertRaisesRegex(ValueError, "known-present"):
            aggregate_sachs_roc._parse_estimated_targets(
                json.dumps(missing_known.tolist()),
                "ps_mip_intended",
                intended,
            )
        with self.assertRaisesRegex(ValueError, "must equal"):
            aggregate_sachs_roc._parse_estimated_targets(
                json.dumps(zeros.tolist()), "igsp_oracle", intended
            )

        union = np.zeros(11, dtype=int)
        union[[1, 6]] = 1
        parsed, _canonical = aggregate_sachs_roc._parse_estimated_targets(
            json.dumps(union.tolist()), "gnies_unknown", intended
        )
        np.testing.assert_array_equal(parsed, union)
        with self.assertRaisesRegex(ValueError, r"shape \(11,\)"):
            aggregate_sachs_roc._parse_estimated_targets(
                json.dumps(zeros.tolist()), "gnies_unknown", intended
            )

    def test_graph_contract_requires_method_label_and_consistent_extension(self):
        dag = np.zeros((11, 11), dtype=np.uint8)
        dag[0, 1] = 1
        undirected_graph = np.zeros_like(dag)
        undirected_graph[0, 1] = 1
        undirected_graph[1, 0] = 1
        targets = np.zeros((5, 11), dtype=np.uint8)

        aggregate_sachs_roc._validate_graph_contract(
            "ps_mip_unknown",
            "dag_and_i_cpdag",
            dag,
            undirected_graph,
            targets,
            recompute_icpdag=False,
        )
        with self.assertRaisesRegex(ValueError, "graph_representation"):
            aggregate_sachs_roc._validate_graph_contract(
                "ps_mip_unknown",
                "i_cpdag_and_deterministic_consistent_extension",
                dag,
                undirected_graph,
                targets,
                recompute_icpdag=False,
            )

        reversed_graph = np.zeros_like(dag)
        reversed_graph[1, 0] = 1
        with self.assertRaisesRegex(ValueError, "same orientation"):
            aggregate_sachs_roc._validate_graph_contract(
                "ps_mip_unknown",
                "dag_and_i_cpdag",
                dag,
                reversed_graph,
                targets,
                recompute_icpdag=False,
            )

        wrong_skeleton = undirected_graph.copy()
        wrong_skeleton[2, 3] = 1
        wrong_skeleton[3, 2] = 1
        with self.assertRaisesRegex(ValueError, "identical skeletons"):
            aggregate_sachs_roc._validate_graph_contract(
                "gnies_unknown",
                "i_cpdag_and_deterministic_consistent_extension",
                dag,
                wrong_skeleton,
                np.zeros(11, dtype=np.uint8),
                recompute_icpdag=False,
            )

        collider_dag = np.zeros_like(dag)
        collider_dag[0, 1] = 1
        collider_dag[2, 1] = 1
        undirected_chain = np.zeros_like(dag)
        undirected_chain[0, 1] = undirected_chain[1, 0] = 1
        undirected_chain[1, 2] = undirected_chain[2, 1] = 1
        with self.assertRaisesRegex(ValueError, "unshielded collider"):
            aggregate_sachs_roc._validate_graph_contract(
                "gnies_unknown",
                "i_cpdag_and_deterministic_consistent_extension",
                collider_dag,
                undirected_chain,
                np.zeros(11, dtype=np.uint8),
                recompute_icpdag=False,
            )

    def test_aggregation_rejects_method_config_and_seed_drift_within_path(self):
        data = _dummy_sachs_data()
        truth = sachs_data.consensus_dag()
        targets = sachs_data.intended_target_matrix()
        info = {
            "data_format_version": sachs_data.DATA_FORMAT_VERSION,
            "data_sha256": "d" * 64,
            "source_commit": sachs_data.SOURCE_COMMIT,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = root / "parts"
            summary = root / "summary"
            settings = run_sachs_roc.method_settings("utigsp_unknown", data)
            for index, setting in enumerate(settings):
                output = parts / f"setting_{index + 1:02d}.csv"
                args = _runner_args(
                    "utigsp_unknown",
                    index,
                    output,
                    seed=run_sachs_roc.DEFAULT_METHOD_SEED + int(index == 0),
                    threads=4 if index == 0 else 8,
                )
                row = run_sachs_roc._base_row(args, data, info, setting)
                result = {
                    "estimated_dag": np.zeros((11, 11), dtype=np.uint8),
                    "estimated_graph": np.zeros((11, 11), dtype=np.uint8),
                    "estimated_targets": np.zeros((5, 11), dtype=np.uint8),
                    "graph_representation": "dag_and_i_cpdag",
                    "fit_seconds": 0.01,
                    "preprocessing": run_sachs_roc.METHOD_PREPROCESSING[args.method],
                }
                run_sachs_roc._add_result(row, result, truth)
                run_sachs_roc._write_row(output, row)

            with mock.patch.object(
                aggregate_sachs_roc,
                "load_sachs_data",
                return_value=(data, truth, targets, info),
            ):
                with self.assertRaises(aggregate_sachs_roc.ValidationError):
                    aggregate_sachs_roc.aggregate_results(
                        inputs=[parts],
                        output_dir=summary,
                        data_root=root / "unused-data",
                        methods=["utigsp_unknown"],
                    )
            report = json.loads((summary / "validation_report.json").read_text())
            identity = report["method_identity_validation"]
            self.assertEqual(identity["seed_drift"][0]["method"], "utigsp_unknown")
            self.assertEqual(
                identity["method_config_drift"][0]["method"],
                "utigsp_unknown",
            )

    def test_complete_ps_path_accepts_and_marks_a_nonoptimal_incumbent(self):
        data = _dummy_sachs_data()
        truth = sachs_data.consensus_dag()
        targets = sachs_data.intended_target_matrix()
        info = {
            "data_format_version": sachs_data.DATA_FORMAT_VERSION,
            "data_sha256": "b" * 64,
            "source_commit": sachs_data.SOURCE_COMMIT,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = root / "parts"
            summary = root / "summary"
            settings = run_sachs_roc.method_settings("ps_mip_unknown", data)
            for index, setting in enumerate(settings):
                output = parts / f"setting_{index + 1:02d}.csv"
                args = _runner_args("ps_mip_unknown", index, output)
                row = run_sachs_roc._base_row(args, data, info, setting)
                result = {
                    "estimated_dag": np.zeros((11, 11), dtype=np.uint8),
                    "estimated_graph": np.zeros((11, 11), dtype=np.uint8),
                    "estimated_targets": np.zeros((5, 11), dtype=np.uint8),
                    "graph_representation": "dag_and_i_cpdag",
                    "fit_seconds": 0.01,
                    "optimal": index != 0,
                    "solver_status": "TIME_LIMIT" if index == 0 else "OPTIMAL",
                    "mip_gap": 0.1 if index == 0 else 0.0,
                    "preprocessing": run_sachs_roc.METHOD_PREPROCESSING[args.method],
                }
                run_sachs_roc._add_result(row, result, truth)
                run_sachs_roc._write_row(output, row)

            with mock.patch.object(
                aggregate_sachs_roc,
                "load_sachs_data",
                return_value=(data, truth, targets, info),
            ):
                combined, points, report = aggregate_sachs_roc.aggregate_results(
                    inputs=[parts],
                    output_dir=summary,
                    data_root=root / "unused-data",
                    methods=["ps_mip_unknown"],
                )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(combined), 9)
            self.assertEqual(report["counts"]["accepted_nonoptimal_rows"], 1)
            self.assertEqual(int(points["is_nonoptimal"].sum()), 1)
            self.assertTrue((points["directed_tp"] == 0).all())
            self.assertTrue((points["directed_fp"] == 0).all())

            outputs = plot_sachs_roc.plot_roc_points(
                points,
                output_dir=summary,
                methods=["ps_mip_unknown"],
                dpi=72,
            )
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))
            stored_report = json.loads((summary / "validation_report.json").read_text())
            self.assertEqual(stored_report["counts"]["accepted_nonoptimal_rows"], 1)


if __name__ == "__main__":
    unittest.main()
