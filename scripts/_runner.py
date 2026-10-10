"""Portable launch configuration and shared baseline trajectory replay."""

import argparse
import json
import random
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def argument_parser(script_path):
    parser = argparse.ArgumentParser(description=Path(script_path).stem)
    parser.add_argument("--smoke", action="store_true", help="One problem, one run, two BO updates.")
    parser.add_argument("--runs", type=int, help="Independent runs per problem.")
    parser.add_argument("--start-run", type=int, help="First run ID; seed is 42 + run ID.")
    parser.add_argument("--workers", type=int, help="Concurrent workers; 1 selects serial execution.")
    parser.add_argument("--benchmarks", nargs="+", help="Problem filters, e.g. Ackley:2 Branin:2.")
    parser.add_argument("--eval-multiplier", type=int, help="Nmax = multiplier * dimension (default: script setting).")
    parser.add_argument("--max-evals", type=int, help="Explicit Nmax, including the 10d initial samples.")
    parser.add_argument("--mode", choices=("fresh", "replay"), help="Override the script's optimization mode.")
    parser.add_argument("--trajectory-dir", type=Path, help="Directory for shared baseline .pkl files; missing files are generated.")
    parser.add_argument("--output", type=Path, help="Result directory (default: project results/<script>_<timestamp>).")
    parser.add_argument("--save-snapshots", action="store_true", help="Also retain GP snapshots in experiment results.")
    parser.add_argument("--figures", action="store_true", help="Save convergence figures where supported by the script.")
    parser.add_argument("--logs", action="store_true", help="Write per-run log files.")
    return parser


def prepare_experiment(config, tasks, args, script_path):
    config, tasks = deepcopy(config), deepcopy(tasks)
    if args.benchmarks:
        available = {f"{task['obj_func_name']}:{task['num_dims']}" for task in tasks}
        unknown = set(args.benchmarks) - available
        if unknown:
            raise ValueError("Unknown benchmark(s): " + ", ".join(sorted(unknown)))
        tasks = [task for task in tasks if f"{task['obj_func_name']}:{task['num_dims']}" in args.benchmarks]
    if not tasks:
        raise ValueError("No experiment configurations selected.")
    if args.smoke:
        tasks = [min(tasks, key=lambda task: task["num_dims"])]
        config.update(num_processes=1, start_process_id=0, process_batch_size=1, is_multi_processing=False)
    for option, key in (("runs", "num_processes"), ("start_run", "start_process_id"), ("workers", "process_batch_size")):
        value = getattr(args, option)
        if value is not None:
            config[key] = value
    if config["num_processes"] <= 0 or config["process_batch_size"] <= 0:
        raise ValueError("--runs and --workers must be positive.")
    if config["start_process_id"] < 0:
        raise ValueError("--start-run must be nonnegative.")
    if args.workers is not None:
        config["is_multi_processing"] = args.workers > 1
    if args.eval_multiplier is not None and args.eval_multiplier < 10:
        raise ValueError("--eval-multiplier must be at least 10 (Ninit = 10d).")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = (args.output or PROJECT_ROOT / "results" / f"{Path(script_path).stem}_{stamp}").resolve()
    for task in tasks:
        if args.eval_multiplier is not None:
            task["hf_eval_num"] = args.eval_multiplier
        if args.smoke:
            task["max_evals"] = 10 * task["num_dims"] + 2
        if args.max_evals is not None:
            task["max_evals"] = args.max_evals
        budget = task.get("max_evals", task["hf_eval_num"] * task["num_dims"])
        if budget < 10 * task["num_dims"]:
            raise ValueError("Nmax must include at least 10d initial samples.")
        if args.mode is not None:
            task["optimi_run_mode"] = int(args.mode == "replay")
        task.setdefault("is_save_snapshot", False)
        if args.save_snapshots:
            task["is_save_snapshot"] = True
        if args.figures:
            task["figure_run_mode"] = 2
        if args.logs:
            task["is_log"] = True
        task["data_figure_dir"] = output
        task["trajectory_dir"] = (
            args.trajectory_dir or PROJECT_ROOT / "results" / "trajectories" / f"nmax_{budget}"
        ).resolve()
    return config, tasks, output


def run_script(task_iter, parameter_config, script_path, argv=None):
    from htslibs.utils.multi_process import process_mode

    parser = argument_parser(script_path)
    args = parser.parse_args(argv)
    try:
        config, tasks, output = prepare_experiment(*parameter_config(), args, script_path)
    except ValueError as exc:
        parser.error(str(exc))
    run_ids = list(range(config["start_process_id"], config["start_process_id"] + config["num_processes"]))
    _write_json(output / "manifest.json", {
        "launcher": Path(script_path).name,
        "smoke": args.smoke,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": config,
        "tasks": tasks,
        "run_ids": run_ids,
        "seeds": [42 + run_id for run_id in run_ids],
    })
    completed, failures = 0, []
    for params in tasks:
        print(f"\n[{Path(script_path).stem}] {params['obj_func_name']} ({params['num_dims']}D)", flush=True)
        results = process_mode(
            task_iter,
            params_dict_list=[deepcopy(params) for _ in run_ids],
            multiy_process_params=run_ids,
            batch_size=min(config["process_batch_size"], len(run_ids)),
            is_multi_processing=config["is_multi_processing"] and len(run_ids) > 1,
        )
        for result in results:
            if result["success"]:
                completed += 1
            else:
                failure = {"benchmark": params["obj_func_name"], "dimension": params["num_dims"],
                           "process_id": result["process_id"], "error": result["error"]}
                failures.append(failure)
                print(result["error"], flush=True)
        _write_json(output / "run_summary.json", {
            "completed_runs": completed, "failed_runs": len(failures), "failures": failures,
        })
    print(f"\nResults: {output}\nCompleted: {completed}; failed: {len(failures)}", flush=True)
    return int(bool(failures))


@contextmanager
def _preserve_rng():
    import numpy as np
    import torch

    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _generate_trajectory(params, process_id, budget):
    import numpy as np
    import torch
    from htslibs.bayesian_optimization_lib.bayesian_optimization import bo
    from htslibs.objective_functions.obj_functions import get_function_details
    from htslibs.utils.sample_methods import latin_hypercube_sample_tensor

    seed = 42 + process_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    objective = get_function_details(params["obj_func_name"], params["num_dims"])
    func = objective.mf_funcs.fidel_func[objective.mf_funcs.num_fidels - 1]
    bounds = objective.bounds
    initial = 10 * objective.num_dims
    # The original experiment launchers draw the initial design twice.
    latin_hypercube_sample_tensor(initial, bounds[0], bounds[1])
    x = latin_hypercube_sample_tensor(initial, bounds[0], bounds[1])
    model = bo(func, bounds, x, func(x), params["acq_func_name"], budget,
               tradeoff_param=0.95, is_save_snapshot=True)
    model.optimize()
    metadata = {key: params[key] for key in ("obj_func_name", "num_dims", "acq_func_name", "hf_eval_num")}
    metadata.update(seed=seed, process_id=process_id, max_evals=budget, trajectory_format=1)
    return {"params_dict": metadata, "bo_": model, "stop_method_dict": {}}


def _validate_trajectory(data, params, process_id, budget):
    model = data["bo_"]
    metadata = data["params_dict"]
    for key in ("obj_func_name", "num_dims", "acq_func_name"):
        if metadata.get(key) != params[key]:
            raise ValueError(f"Trajectory {key} does not match the requested experiment.")
    if metadata.get("seed", 42 + metadata.get("process_id", process_id)) != 42 + process_id:
        raise ValueError("Trajectory seed does not match the requested run ID.")
    initial = 10 * params["num_dims"]
    snapshots = {snapshot["iter"] for snapshot in getattr(model, "gp_snapshot_list", [])}
    missing = set(range(initial, budget + 1)) - snapshots
    if missing or model.x.shape[0] < budget:
        raise ValueError("Trajectory lacks the required GP snapshots. Generate it with --save-snapshots, "
                         "or select a new --trajectory-dir for automatic generation.")
    data.setdefault("stop_method_dict", {})
    return data


def replay_metric(metrics, evaluation):
    if not metrics:
        return None
    index = evaluation - metrics[0]["iter"]
    if 0 <= index < len(metrics) and metrics[index]["iter"] == evaluation:
        return metrics[index]["stop_metric_val"]
    # A stopped method may have a shorter history than the baseline trajectory.
    return None


def load_replay_data(params, process_id):
    from htslibs.utils.file_operation_lib.file_operation import load_data_from_pkl, save_data_to_pkl

    budget = int(params.get("max_evals", params["hf_eval_num"] * params["num_dims"]))
    directory = Path(params.get("trajectory_dir", PROJECT_ROOT / "results" / "trajectories" / f"nmax_{budget}"))
    path = directory / f"{params['obj_func_name']}_dims{params['num_dims']}_{params['acq_func_name']}_process_{process_id}.pkl"
    # Cache hits and first-time generation must leave identical caller RNG states.
    with _preserve_rng():
        if path.exists():
            print(f"Replaying baseline: {path}", flush=True)
            return _validate_trajectory(load_data_from_pkl(path), params, process_id, budget)
        print(f"Generating shared baseline ({budget} evaluations): {path}", flush=True)
        data = _validate_trajectory(_generate_trajectory(params, process_id, budget), params, process_id, budget)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(directory), prefix=path.stem + "_", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            save_data_to_pkl(data, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return data
