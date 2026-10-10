# LEET

This repository contains the reference implementation and experiment launchers
for **LEET: A Stagnation-Aware Bayesian Optimization Framework with Integrated
Automatic Termination**.

LEET combines stagnation detection, local surrogate verification, local
evaluation, exploration enhancement, and automatic termination. The repository
also contains implementations of the stopping criteria used as comparison
methods in the paper:

| Code key | Method in the paper |
| --- | --- |
| `LR` | LEET |
| `CBOS` | Li et al. |
| `CB` | Makarova et al. |
| `EI` | Nguyen et al. |
| `PI` | Lorenz et al. |
| `Cov_i` | Naive stagnation criterion |
| `Ishibashi` | Ishibashi et al. |
| `Wilson` | Wilson |

## Repository layout

```text
htslibs/
  bayesian_optimization_lib/    BO, GP, acquisition, and stopping implementations
  objective_functions/         synthetic, HPO, and airfoil objectives
  multi_fidelity_optimization/ local-regret utilities and bundled surrogate models
  utils/                        sampling, persistence, and multiprocessing helpers
scripts/
  run_main_experiments.py       26 synthetic and HPO benchmarks
  run_ablation_xx.py               component-ablation study
  run_sensitivity_xx.py            LEET sensitivity and interaction studies
  run_airfoil.py                three 16-dimensional airfoil scenarios
```

## Installation

Create a clean environment from the repository root and install the pinned dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch wheels are platform dependent. For a CUDA-specific installation,
install the appropriate PyTorch 2.4.1 wheel first and then install the remaining
requirements.


## Reproducing the experiments

The paper uses `Nmax = 50d` objective evaluations, `Ninit = 10d` Latin
hypercube samples, and 30 independent runs. Run IDs 0 through 29.

### Running the core script

From the repository root:

```bash
python -m htslibs.bayesian_optimization_lib.bo_stop_method
```

Direct execution from the script directory is also supported:

```bash
cd htslibs/bayesian_optimization_lib
python bo_stop_method.py
```

The direct entry point locates the bundled `htslibs` package relative to its
own file. No separate HTSlibs installation or `PYTHONPATH` setting is required.
It uses the experiment settings in `parameter_config()` in `bo_stop_method.py`.

### Main benchmark experiments

The main launcher covers 21 synthetic configurations and five hyperparameter
optimization tasks. It evaluates the three reported threshold levels for the
threshold-based methods and the configured regret-based criteria.

The Adult Income and California Housing tasks may download public datasets on
their first run and therefore require network access.

### Component ablation

The ablation launcher runs Baseline BO, EE-only, LE-only, LEET-Search, and full
LEET. Local evaluations count toward the same `Nmax` budget.


### Sensitivity studies

The `t_ni` and `eta_delta_mu` studies use all 26 non-airfoil benchmarks. The
remaining studies use the six representative problems specified in the paper.


### Airfoil experiments

The airfoil launcher runs AF-1, AF-2, and AF-3 using NeuralFoil and the moderate
threshold level transferred from the benchmark study:


This experiment is a surrogate-based aerodynamic optimization study; it is not
a high-fidelity CFD validation.

## Outputs

By default, outputs are written to a timestamped directory under `results/`.
Every launch writes:

- `manifest.json`, containing the selected problems, settings, run IDs, and seeds;
- one serialized `.pkl` result per benchmark and run;
- `run_summary.json`, containing completion and failure information;
- optional logs and figures when `--logs` or `--figures` is supplied.

Use `--output PATH` to choose a stable destination. Result files are ignored by
Git because full experiment sets can be large.

## Notes on reproducibility

- Synthetic objectives use additive Gaussian noise with standard deviation
  `0.001`.
- The default acquisition function is UCB.
- The experiment manifests preserve all launcher-level settings.
- OpenML-backed objectives require network access on their first execution.
- The optional `Q1` objective requires MATLAB and the `Q1_MATLAB_PATH`
  environment variable; it is not part of the 29 reported benchmark problems.

## Citation

Please cite the accompanying paper when using this implementation. Full
bibliographic metadata will be added after publication.

## License

The repository is released under the MIT License. See [LICENSE](LICENSE).

