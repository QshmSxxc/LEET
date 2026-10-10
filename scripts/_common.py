"""Shared experiment definitions and launch utilities for the LEET scripts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from htslibs.bayesian_optimization_lib.bo_stop_method import task_iter
from htslibs.utils.multi_process import process_mode


Benchmark = Tuple[str, int]

SYNTHETIC_BENCHMARKS: List[Benchmark] = [
    *[(name, dim) for dim in (2, 5, 10) for name in (
        "Ackley",
        "Levy",
        "Schwefel",
        "Rastrigin",
        "Griewank",
        "Michalewicz",
    )],
    ("Branin", 2),
    ("Hartmann3D", 3),
    ("Hartmann6D", 6),
]

HPO_BENCHMARKS: List[Benchmark] = [
    ("SVM_BreastCancer", 2),
    ("MLP_Digits", 5),
    ("RandomForest_California_Housing", 5),
    ("XGBoost_Diabetes", 8),
    ("LightGBM_AdultIncome", 10),
]

MAIN_BENCHMARKS: List[Benchmark] = SYNTHETIC_BENCHMARKS + HPO_BENCHMARKS

AIRFOIL_BENCHMARKS: List[Benchmark] = [
    ("Airfoil_AF1", 16),
    ("Airfoil_AF2", 16),
    ("Airfoil_AF3", 16),
]

REPRESENTATIVE_BENCHMARKS: List[Benchmark] = [
    ("Branin", 2),
    ("Ackley", 5),
    ("Hartmann6D", 6),
    ("Schwefel", 10),
    ("Rastrigin", 10),
    ("LightGBM_AdultIncome", 10),
]

THREE_LEVEL_THRESHOLDS: Dict[str, List[float]] = {
    "LR": [3.85e-3, 1.14e-2, 4.17e-2],
    "CBOS": [2.04, 2.17, 2.93],
    "CB": [3.44e-3, 1.94e-2, 1.99e-1],
    "EI": [6.41e-7, 5.29e-5, 5.37e-4],
    "PI": [4.53e-6, 5.85e-4, 6.21e-3],
    "Cov_i": [-167.0, -74.0, -37.0],
    "Ishibashi": [0.0],
    "Wilson": [0.0],
}

MODERATE_THRESHOLDS: Dict[str, List[float]] = {
    method: [values[1] if len(values) == 3 else values[0]]
    for method, values in THREE_LEVEL_THRESHOLDS.items()
}

DEFAULT_LEET_CONFIG = {
    "no_improve_patience": 10,
    "max_exploration_steps": 10,
    "eta_delta_mu": 1e-2,
    "local_region_scale": 1e-2,
    "local_sample_multiplier": 10,
    "base_confidence": 0.95,
    "exploration_increment": 0.03,
}

FULL_LEET_SWITCHES = {
    "enable_stop": True,
    "enable_exploration": True,
    "enable_local_evaluation": True,
}


def add_common_arguments(parser: argparse.ArgumentParser, default_runs: int = 30) -> None:
    parser.add_argument("--runs", type=int, default=default_runs, help="Independent runs per problem.")
    parser.add_argument("--start-run", type=int, default=0, help="First run ID; seed is 42 + run ID.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(5, (os.cpu_count() or 2) - 1)),
        help="Maximum concurrent worker processes.",
    )
    parser.add_argument(
        "--eval-multiplier",
        type=int,
        default=50,
        help="Evaluation budget multiplier in Nmax = multiplier * dimension.",
    )
    parser.add_argument("--output", type=Path, help="Output directory. A timestamped directory is used by default.")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        help="Optional benchmark filters such as Ackley:5 Branin:2.",
    )
    parser.add_argument("--figures", action="store_true", help="Save convergence figures.")
    parser.add_argument("--logs", action="store_true", help="Write one log file per run.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one short configuration for installation verification.",
    )


def parse_benchmark_spec(spec: str) -> Benchmark:
    try:
        name, dim_text = spec.rsplit(":", 1)
        return name, int(dim_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid benchmark '{spec}'. Use NAME:DIM, for example Ackley:5."
        ) from exc


def select_benchmarks(
    available: Sequence[Benchmark], requested: Optional[Sequence[str]]
) -> List[Benchmark]:
    if not requested:
        return list(available)
    selected = [parse_benchmark_spec(item) for item in requested]
    unknown = [item for item in selected if item not in available]
    if unknown:
        allowed = ", ".join(f"{name}:{dim}" for name, dim in available)
        bad = ", ".join(f"{name}:{dim}" for name, dim in unknown)
        raise SystemExit(f"Unknown benchmark(s): {bad}. Available values: {allowed}")
    return selected


def build_stop_method_dict(thresholds: Mapping[str, Sequence[float]]) -> dict:
    stop_method_dict = {}
    for method, values in thresholds.items():
        method_tolerances = {}
        for tolerance in values:
            method_tolerances[f"tol_{tolerance}"] = {
                "HP_nst": {
                    "stop_tol": tolerance,
                    "stop_iter": None,
                    "stop_x_best": None,
                    "stop_fx_best": None,
                    "stop_metrics_val": None,
                    "metrics_val_list": [],
                    "HP_val": "nst",
                    "HP_name": "nst",
                }
            }
        stop_method_dict[method] = method_tolerances
    return stop_method_dict


def timestamped_output(experiment_name: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.resolve()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return PROJECT_ROOT / "results" / f"{experiment_name}_{stamp}"


def write_manifest(output_dir: Path, payload: Mapping) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def build_task_params(
    benchmark: Benchmark,
    output_dir: Path,
    thresholds: Mapping[str, Sequence[float]],
    eval_multiplier: int,
    leet_config: Optional[Mapping] = None,
    ablation_config: Optional[Mapping] = None,
    figures: bool = False,
    logs: bool = False,
) -> dict:
    name, dimension = benchmark
    merged_leet_config = dict(DEFAULT_LEET_CONFIG)
    merged_leet_config.update(leet_config or {})
    merged_ablation_config = dict(FULL_LEET_SWITCHES)
    merged_ablation_config.update(ablation_config or {})
    return {
        "obj_func_name": name,
        "num_dims": dimension,
        "acq_func_name": "UCB",
        "hf_eval_num": eval_multiplier,
        "figure_run_mode": 2 if figures else 0,
        "is_log": logs,
        "is_log_to_console": False,
        "is_save_snapshot": False,
        "optimi_run_mode": 0,
        "stop_method_dict": build_stop_method_dict(thresholds),
        "leet_config": merged_leet_config,
        "ablation_config": merged_ablation_config,
        "ishibashi_params": {"delta": 0.1, "num_candidates": 2048},
        "wilson_params": {
            "regret_bound": 0.1,
            "total_risk": 0.05,
            "max_samples": 1000,
            "initial_samples": 64,
            "sample_growth": 1.5,
            "risk_decay": 1.1,
            "raw_samples": 256,
            "num_restarts": 8,
            "require_convergence": False,
        },
        "data_figure_dir": output_dir,
    }


def run_experiment_grid(
    *,
    experiment_name: str,
    output_dir: Path,
    configurations: Iterable[Mapping],
    runs: int,
    start_run: int,
    workers: int,
    eval_multiplier: int,
    figures: bool,
    logs: bool,
    manifest_extra: Optional[Mapping] = None,
) -> None:
    if runs <= 0:
        raise SystemExit("--runs must be positive.")
    if workers <= 0:
        raise SystemExit("--workers must be positive.")
    if eval_multiplier < 10:
        raise SystemExit("--eval-multiplier must be at least 10 because Ninit = 10d.")

    configurations = list(configurations)
    manifest = {
        "experiment": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
        "run_ids": list(range(start_run, start_run + runs)),
        "seeds": list(range(42 + start_run, 42 + start_run + runs)),
        "workers": workers,
        "evaluation_budget": f"{eval_multiplier} * dimension",
        "initial_design": "10 * dimension Latin hypercube samples",
        "configurations": configurations,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    write_manifest(output_dir, manifest)

    run_ids = list(range(start_run, start_run + runs))
    failures = []
    completed = 0

    for configuration in configurations:
        label = str(configuration["label"])
        config_output = output_dir / label
        thresholds = configuration.get("thresholds", MODERATE_THRESHOLDS)
        for benchmark in configuration["benchmarks"]:
            params = build_task_params(
                benchmark=tuple(benchmark),
                output_dir=config_output,
                thresholds=thresholds,
                eval_multiplier=eval_multiplier,
                leet_config=configuration.get("leet_config"),
                ablation_config=configuration.get("ablation_config"),
                figures=figures,
                logs=logs,
            )
            print(f"\n[{label}] {benchmark[0]} ({benchmark[1]}D), {runs} run(s)", flush=True)
            results = process_mode(
                task_iter,
                params_dict_list=[deepcopy(params) for _ in run_ids],
                multiy_process_params=run_ids,
                batch_size=min(workers, runs),
                is_multi_processing=workers > 1 and runs > 1,
            )
            completed += sum(bool(item["success"]) for item in results)
            failures.extend(
                {
                    "configuration": label,
                    "benchmark": list(benchmark),
                    "process_id": item["process_id"],
                    "error": item["error"],
                }
                for item in results
                if not item["success"]
            )

    summary = {
        "completed_runs": completed,
        "failed_runs": len(failures),
        "failures": failures,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"\nResults: {output_dir}")
    print(f"Completed: {completed}; failed: {len(failures)}")
    if failures:
        raise SystemExit(1)

