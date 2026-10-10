import json
import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts import _runner


def configuration():
    return {"num_processes": 30, "start_process_id": 5,
            "process_batch_size": 5, "is_multi_processing": True}, [
        {"obj_func_name": name, "num_dims": dimension, "acq_func_name": "UCB",
         "hf_eval_num": 50, "optimi_run_mode": 1, "figure_run_mode": 0,
         "is_log": False}
        for name, dimension in (("Ackley", 5), ("Branin", 2))
    ]


def fake_trajectory(params, process_id, budget):
    metadata = {key: params[key] for key in ("obj_func_name", "num_dims", "acq_func_name")}
    metadata.update(seed=42 + process_id, process_id=process_id)
    model = SimpleNamespace(
        x=SimpleNamespace(shape=(budget, params["num_dims"])),
        gp_snapshot_list=[{"iter": index} for index in range(10 * params["num_dims"], budget + 1)],
    )
    return {"bo_": model, "params_dict": metadata, "stop_method_dict": {}}


class ConfigurationTests(unittest.TestCase):
    def prepare(self, argv):
        config, tasks = configuration()
        args = _runner.argument_parser("run_test.py").parse_args(argv)
        return _runner.prepare_experiment(config, tasks, args, "run_test.py")

    def test_defaults_preserve_paper_configuration(self):
        config, tasks, output = self.prepare([])
        self.assertEqual(config["num_processes"], 30)
        self.assertEqual(config["start_process_id"], 5)
        self.assertTrue(all(task["optimi_run_mode"] == 1 for task in tasks))
        self.assertTrue(all(task["hf_eval_num"] == 50 for task in tasks))
        self.assertEqual(output.parent, _runner.PROJECT_ROOT / "results")

    def test_smoke_only_reduces_workload(self):
        config, tasks, _ = self.prepare(["--smoke"])
        self.assertEqual(config["num_processes"], 1)
        self.assertEqual(config["start_process_id"], 0)
        self.assertFalse(config["is_multi_processing"])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["obj_func_name"], "Branin")
        self.assertEqual(tasks[0]["max_evals"], 22)
        self.assertEqual(tasks[0]["optimi_run_mode"], 1)

    def test_explicit_overrides_and_no_input_mutation(self):
        config, tasks = configuration()
        before = deepcopy((config, tasks))
        args = _runner.argument_parser("run_test.py").parse_args([
            "--smoke", "--runs", "2", "--workers", "2", "--start-run", "3",
            "--benchmarks", "Ackley:5", "--mode", "fresh", "--max-evals", "55",
            "--logs", "--figures", "--save-snapshots",
        ])
        config_out, tasks_out, _ = _runner.prepare_experiment(config, tasks, args, "run_test.py")
        self.assertEqual((config, tasks), before)
        self.assertEqual(config_out["num_processes"], 2)
        self.assertEqual(config_out["start_process_id"], 3)
        self.assertTrue(config_out["is_multi_processing"])
        self.assertEqual(tasks_out[0]["max_evals"], 55)
        self.assertEqual(tasks_out[0]["optimi_run_mode"], 0)
        self.assertTrue(tasks_out[0]["is_save_snapshot"])
        self.assertTrue(tasks_out[0]["is_log"])
        self.assertEqual(tasks_out[0]["figure_run_mode"], 2)

    def test_invalid_settings_fail_before_launch(self):
        for argv in (["--runs", "0"], ["--workers", "0"], ["--start-run", "-1"],
                     ["--eval-multiplier", "9"], ["--max-evals", "19"],
                     ["--benchmarks", "Missing:2"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.prepare(argv)

    def test_task_failures_produce_nonzero_exit_and_summary(self):
        results = [{"success": False, "process_id": 0, "error": "test failure"}]
        with tempfile.TemporaryDirectory() as directory, patch(
            "htslibs.utils.multi_process.process_mode", return_value=results
        ):
            status = _runner.run_script(None, configuration, "run_test.py", [
                "--smoke", "--output", directory,
            ])
            summary = json.loads((Path(directory) / "run_summary.json").read_text())
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
        self.assertEqual(status, 1)
        self.assertEqual(summary["failed_runs"], 1)
        self.assertEqual(manifest["seeds"], [42])


class TrajectoryTests(unittest.TestCase):
    def test_short_or_gapped_metric_histories_fall_back_to_recomputation(self):
        metrics = [{"iter": 21, "stop_metric_val": 0.5}, {"iter": 23, "stop_metric_val": 0.4}]
        self.assertEqual(_runner.replay_metric(metrics, 21), 0.5)
        for evaluation in (20, 22, 23, 100):
            self.assertIsNone(_runner.replay_metric(metrics, evaluation))
        self.assertIsNone(_runner.replay_metric([], 21))
        self.assertIsNone(_runner.replay_metric(None, 21))

    def test_generate_then_reuse_and_preserve_rng(self):
        params = configuration()[1][1]
        with tempfile.TemporaryDirectory() as directory:
            params.update(max_evals=22, trajectory_dir=Path(directory))
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            initial = random.getstate(), np.random.get_state(), torch.get_rng_state()

            def generate(*args):
                random.random()
                np.random.random()
                torch.rand(1)
                return fake_trajectory(*args)

            with patch.object(_runner, "_generate_trajectory", side_effect=generate) as generator:
                first = _runner.load_replay_data(params, 0)
                second = _runner.load_replay_data(params, 0)
                self.assertEqual(generator.call_count, 1)
            self.assertEqual(first["params_dict"], second["params_dict"])
            self.assertEqual(random.getstate(), initial[0])
            np.testing.assert_array_equal(np.random.get_state()[1], initial[1][1])
            self.assertEqual(np.random.get_state()[2:], initial[1][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), initial[2]))
            self.assertEqual(len(list(Path(directory).glob("*.pkl"))), 1)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_reject_missing_snapshots_and_wrong_metadata(self):
        params = configuration()[1][1]
        for invalid in ("snapshots", "seed", "objective"):
            with self.subTest(invalid=invalid):
                data = fake_trajectory(params, 0, 22)
                if invalid == "snapshots":
                    data["bo_"].gp_snapshot_list = []
                elif invalid == "seed":
                    data["params_dict"]["seed"] = 43
                else:
                    data["params_dict"]["obj_func_name"] = "Ackley"
                with self.assertRaises(ValueError):
                    _runner._validate_trajectory(data, params, 0, 22)


if __name__ == "__main__":
    unittest.main()
