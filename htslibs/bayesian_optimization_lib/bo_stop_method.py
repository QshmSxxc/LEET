import logging
import math
import os
import sys
import warnings
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

# Direct file execution must expose the bundled package to Python and workers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import torch
from pyswarm import pso
import scipy.stats as stats
import numpy as np
import random
from datetime import datetime
import time

from htslibs.bayesian_optimization_lib.pso_x1 import pso_x1
from htslibs.bayesian_optimization_lib.bayesian_optimization import bo
from htslibs.objective_functions.obj_functions import get_function_details
from htslibs.utils.sample_methods import latin_hypercube_sample_tensor
from htslibs.utils.file_operation_lib.file_operation import create_folder, delete_folder
from htslibs.utils.load_data_from_json import load_config_from_json
from htslibs.multi_fidelity_optimization.stop_method import convexity_check, Approximate_LowerBound
from htslibs.utils.file_operation_lib.file_operation import save_data_to_pkl, load_data_from_pkl
from htslibs.paper import plt_settings

# 获取参数配置
params_config = load_config_from_json()
float_precision = params_config["float_precision"]
device, dtype = torch.device(params_config["device"]), eval(params_config["dtype"])
data_figure_dir = params_config["data_figure_dir"]


stop_method_setting_dict = {
    'LR': {
        'name': "Proposed method",
        'color': "#FFAA9A",
        'linestyle': (0, (7, 1)),
        'linewidth': 3.0,
        'alpha': 1.0,
        'zorder': 1,
    },
    'CBOS': {
        'name': "Li's method",
        'color': "#A5FF19",
        'linestyle': (0, (5, 1)),
        'linewidth': 2.5,
        'alpha': 1.0,
        'zorder': 2,
    },
    'CB': {
        'name': "Makarova's method",
        'color': "#B15FD9",
        'linestyle': (1, (5, 1)),
        'linewidth': 2.0,
        'alpha': 1.0,
        'zorder': 3,
    },
    'EI': {
        'name': "Nguyen's method",
        'color': "#D9B95F",
        'linestyle': (2, (4, 2)),
        'linewidth': 1.5,
        'alpha': 1.0,
        'zorder': 4,
    },
    'PI': {
        'name': "Lorenz's method",
        'color': "#90AD63",
        'linestyle': (3, (3, 3)),
        'linewidth': 1.0,
        'alpha': 1.0,
        'zorder': 5,
    },
    'Cov_i': {
        'name': "Naïve method",
        'color': "#69CFCF",
        'linestyle': (4, (1, 1)),#(4, (5, 2, 1, 2)),
        'linewidth': 0.8,
        'alpha': 1.0,
        'zorder': 6,
    },
    'Ishibashi': {
        'name': "Ishibashi's method",
        'color': "#2F6B3C",
        'linestyle': (0, (6, 2, 1, 2)),
        'linewidth': 1.8,
        'alpha': 1.0,
        'zorder': 7,
    },
    'Wilson': {
        'name': "Wilson's method",
        'color': "#D1495B",
        'linestyle': (0, (2, 2)),
        'linewidth': 1.8,
        'alpha': 1.0,
        'zorder': 8,
    },
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_extremum(x0, func, tol=1e-2):
    x0 = x0.detach().clone().requires_grad_(True)  # 形状保持(1, d)

    # 计算梯度：func(x0)需返回(1, 1)，grad形状为(1, d)
    grad = torch.autograd.grad(func(x0).sum(), x0, create_graph=True)[0]  # 显式梯度计算
    grad_norm = grad.norm()

    # 判断是否为驻点
    if grad_norm > tol:
        return False, None, grad_norm, None

    # 计算Hessian矩阵：形状为(d, d)
    hess = torch.zeros(x0.shape[1], x0.shape[1])
    for i in range(x0.shape[1]):
        # 对梯度的第i个分量求导
        grad_i = torch.autograd.grad(grad[0, i], x0, retain_graph=True)[0][0]  # 取[0]因为x0是(1, d)
        hess[i] = grad_i

    # 特征值分解并判断极值类型
    eigvals = torch.linalg.eigvalsh(hess)
    if torch.all(eigvals > 0):
        extremum_kind = 1  # 局部极小点
    elif torch.all(eigvals < 0):
        extremum_kind = 2  # 局部极大点
    else:
        extremum_kind = 0  # 鞍点

    return True, extremum_kind, grad_norm, eigvals


def adaptive_hypercube(x_best, gp_model, n_samples=10):
    # 检测局部凸性（简化版：邻域梯度一致性）
    neighbors = x_best + 0.1 * torch.randn(n_samples, x_best.shape[1])
    grads = np.array([np.gradient(gp_model.predict(x.reshape(1, -1))) for x in neighbors])
    is_convex = np.all(np.dot(grads, grads.T) > 0)  # 粗略判断

    # 获取当前最优点的预测均值和方差
    mu, var = gp_model.predict(x_best.reshape(1, -1), return_std=True)
    sigma = np.sqrt(var)

    # 动态调整尺寸
    base_size = 0.1  # 基础尺寸
    alpha = 0.5  # 方差权重系数
    if is_convex:
        size = base_size * (1 + alpha * sigma)
    else:
        size = base_size * (1 + 2 * alpha * sigma)  # 非凸区域扩大更多

    # 各向异性扩展（可选）
    length_scales = gp_model.kernel_.length_scale
    size = size * (1 / length_scales)  # length scale越小，该方向范围越大

    return Hypercube(x_best, size)


def update_stop_data(stop_method_tol_dict, stop_metric_val, stop_iter, stop_x_best, stop_fx_best, stop_method_name=None):
    for tol_key, tol_dict in stop_method_tol_dict.items():
        for HP_key, HP_dict in tol_dict.items():
            #  1. 更新停止指标值列表
            HP_dict['metrics_val_list'].append({
                'iter': stop_iter,
                'stop_metric_val': stop_metric_val,
            })
            #  2. 判断是否满足停止条件
            if isinstance(stop_metric_val, list):
                stop_metric_val_temp = stop_metric_val[-1]
            else:
                stop_metric_val_temp = stop_metric_val
            if stop_metric_val_temp < HP_dict['stop_tol'] and HP_dict['stop_iter'] is None:
                # 更新停止数据
                HP_dict['stop_iter'] = stop_iter
                HP_dict['stop_x_best'] = stop_x_best
                HP_dict['stop_fx_best'] = stop_fx_best
                HP_dict['stop_metrics_val'] = stop_metric_val


def check_stop_method_CB(bo_model, stop_method_tol_dict, stop_metric_val):
    if stop_metric_val is None:
        pred_func = bo_model.pred_func
        tradeoff_param = 0.95
        kappa = stats.norm.ppf((1 + tradeoff_param) / 2)
        # 计算 ucb_eval_max
        mean, std, mean_scaled, std_scaled = bo_model.pred_func(bo_model.x)
        ucb_eval = mean + kappa * std
        ucb_eval_max = ucb_eval.max().detach().numpy()
        # 计算 lcb_pred_max
        # def lcb_func(x):
        #     x_torch = torch.tensor(x).to(dtype=dtype, device=device)
        #     mean, std, mean_scaled, std_scaled = bo_model.pred_func(x_torch.reshape(-1, 1))
        #     lcb_pred = mean - kappa * std
        #     return -1 * lcb_pred.detach().numpy()
        # with redirect_stdout(open(os.devnull, 'w')):
        #     pt_best, val_best = pso(
        #         lcb_func, bo_model.bounds[0,:].numpy().tolist(), bo_model.bounds[1,:].numpy().tolist(),
        #         swarmsize=300, maxiter=100
        #     )
        # lcb_pred_max = -1 * val_best
        #    计算 lcb_pred_max 的另一种方法，可用
        _, acq_val = bo_model.acq_func_optimization(pred_func, bo_model.fx_best, 'LCB', 0.95)
        lcb_pred_max = (acq_val * pred_func.fx_train_scaler.std + pred_func.fx_train_scaler.mean).detach().numpy()
        # 作差
        stop_metric_val = ucb_eval_max - lcb_pred_max
    # 更新停止数据
    update_stop_data(stop_method_tol_dict, stop_metric_val, bo_model.curr_evals, bo_model.x_best, bo_model.fx_best, 'CB')

    return stop_metric_val


def check_stop_method_EI(bo_model, stop_method_tol_dict, stop_metric_val):
    if stop_metric_val is None:
        pred_func = bo_model.pred_func
        # 计算 EI
        _, acq_val = bo_model.acq_func_optimization(pred_func, bo_model.fx_best, 'EI', 0.00)
        stop_metric_val = (acq_val * pred_func.fx_train_scaler.std).detach().numpy()
    # 更新停止数据
    update_stop_data(stop_method_tol_dict, stop_metric_val, bo_model.curr_evals, bo_model.x_best, bo_model.fx_best, 'EI')

    return stop_metric_val


def check_stop_method_PI(bo_model, stop_method_tol_dict, stop_metric_val):
    if stop_metric_val is None:
        # bo_stop_method = deepcopy(bo_)
        pred_func = bo_model.pred_func
        # 计算 PI
        _, acq_val = bo_model.acq_func_optimization(pred_func, bo_model.fx_best, 'PI', 0.00)
        stop_metric_val = acq_val.detach().numpy()
    # 更新停止数据
    update_stop_data(stop_method_tol_dict, stop_metric_val, bo_model.curr_evals, bo_model.x_best, bo_model.fx_best, 'PI')

    return stop_metric_val


def check_stop_method_CBOS(bo_model, stop_method_tol_dict, stop_metric_val):
    if stop_metric_val is None:
        convex_check_x_num = 10
        convex_check_x = bo_model.x[-convex_check_x_num:, :]
        convex_rate = convexity_check(convex_check_x, bo_model.pred_func, is_maximize=True)
        local_regret, string = Approximate_LowerBound(device, dtype, convex_check_x, bo_model.bounds, bo_model.pred_func)

        # if convex_rate > 0.9:
        update_stop_data(stop_method_tol_dict, [convex_rate, local_regret], bo_model.curr_evals, bo_model.x_best, bo_model.fx_best, 'CBOS')
    else:
        if stop_metric_val[0] > 0.9:
            update_stop_data(stop_method_tol_dict, stop_metric_val, bo_model.curr_evals, bo_model.x_best, bo_model.fx_best, 'CBOS')


def check_stop_method_Cov_i(bo_model, stop_Covi_param_dict, stop_method_tol_dict, stop_metric_val):
    if stop_metric_val is None:
        stop_Covi_param_dict['cur_x_best'], stop_Covi_param_dict['cur_fx_best'], stop_Covi_param_dict['no_improve_count'] = (
            check_for_improve(
                bo_model.x_best, bo_model.fx_best, stop_Covi_param_dict['cur_x_best'],
                stop_Covi_param_dict['cur_fx_best'], stop_Covi_param_dict['no_improve_count']))
        stop_metric_val = -1*stop_Covi_param_dict['no_improve_count']
    # 更新停止数据
    update_stop_data(stop_method_tol_dict, stop_metric_val, bo_model.curr_evals,
                     bo_model.x_best, bo_model.fx_best, 'Cov_i')


def _stop_method_candidates(bo_model, num_candidates, seed):
    """Create a deterministic domain approximation and include all evaluated points."""
    dim = bo_model.bounds.shape[1]
    engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=int(seed))
    unit_points = engine.draw(int(num_candidates)).to(
        device=bo_model.bounds.device,
        dtype=bo_model.bounds.dtype,
    )
    random_points = (
        bo_model.bounds[0] + unit_points * (bo_model.bounds[1] - bo_model.bounds[0])
    )
    return torch.unique(torch.cat((bo_model.x, random_points), dim=0), dim=0)


def _posterior_joint_in_original_scale(pred_func, points):
    scaled_points = pred_func.x_train_scaler.transform(points)
    with torch.no_grad():
        posterior = pred_func.gp_model.posterior(scaled_points)
        mean = posterior.mean.reshape(-1)
        covariance = posterior.mvn.covariance_matrix.squeeze()

    output_scale = pred_func.fx_train_scaler.std.detach().abs().reshape(-1)[0]
    mean = mean * output_scale + pred_func.fx_train_scaler.mean.detach().reshape(-1)[0]
    covariance = covariance * output_scale.square()
    return mean, covariance


def _positive_part_expectation(mean, variance):
    variance = variance.clamp_min(0.0)
    std = variance.sqrt()
    if std.item() <= torch.finfo(std.dtype).eps:
        return mean.clamp_min(0.0)

    z = mean / std
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    return mean * cdf + std * pdf


def check_stop_method_Ishibashi(
        bo_model, previous_pred_func, stop_method_tol_dict, stop_metric_val,
        delta=0.1, num_candidates=2048,
):
    """Ishibashi et al. (AISTATS 2023), converted to maximization."""
    if stop_metric_val is None:
        if previous_pred_func is None:
            raise ValueError("Ishibashi's method requires the previous GP posterior.")
        if not 0.0 < delta < 1.0:
            raise ValueError("ishibashi_delta must be in (0, 1).")

        candidates = _stop_method_candidates(
            bo_model,
            num_candidates=num_candidates,
            seed=100000 + bo_model.curr_evals,
        )
        old_mean, old_std, _, _ = previous_pred_func(candidates)
        new_mean, _, _, _ = bo_model.pred_func(candidates)
        old_best_index = torch.argmax(old_mean)
        new_best_index = torch.argmax(new_mean)
        old_best_point = candidates[old_best_index]
        new_best_point = candidates[new_best_index]

        # The original paper is stated for minimization. Under h=-f, its
        # improvement term becomes E[(f(old_best)-f(new_best))_+].
        pair = torch.stack((old_best_point, new_best_point))
        pair_mean, pair_covariance = _posterior_joint_in_original_scale(
            bo_model.pred_func, pair
        )
        improvement_mean = pair_mean[0] - pair_mean[1]
        improvement_variance = (
            pair_covariance[0, 0]
            - 2.0 * pair_covariance[0, 1]
            + pair_covariance[1, 1]
        )
        expected_improvement = _positive_part_expectation(
            improvement_mean, improvement_variance
        )

        old_evaluated_x = bo_model.x[:-1]
        old_eval_mean, old_eval_std, _, _ = previous_pred_func(old_evaluated_x)
        sqrt_beta = math.sqrt(
            2.0 * math.log(
                bo_model.bounds.shape[1]
                * old_evaluated_x.shape[0] ** 2
                * math.pi ** 2
                / (6.0 * delta)
            )
        )
        max_domain_ucb = torch.max(old_mean + sqrt_beta * old_std)
        max_evaluated_lcb = torch.max(old_eval_mean - sqrt_beta * old_eval_std)
        kappa = (max_domain_ucb - max_evaluated_lcb).clamp_min(0.0)

        new_x = bo_model.x[-1:].detach()
        new_y = bo_model.fx[-1].detach().reshape(-1)[0]
        old_new_x_mean, old_new_x_covariance = _posterior_joint_in_original_scale(
            previous_pred_func, new_x
        )
        old_new_x_variance = old_new_x_covariance.reshape(-1)[0].clamp_min(1e-16)
        output_scale = previous_pred_func.fx_train_scaler.std.detach().abs().reshape(-1)[0]
        noise_variance = (
            previous_pred_func.gp_model.likelihood.noise.detach().reshape(-1).mean()
            * output_scale.square()
        ).clamp_min(1e-16)
        innovation = new_y - old_new_x_mean[0]
        kl_bracket = (
            torch.log1p(old_new_x_variance / noise_variance)
            - old_new_x_variance / (old_new_x_variance + noise_variance)
            + old_new_x_variance
            * innovation.square()
            / (old_new_x_variance + noise_variance).square()
        ).clamp_min(0.0)
        sqrt_half_kl = 0.5 * kl_bracket.sqrt()

        delta_mu = torch.abs(old_mean[old_best_index] - new_mean[new_best_index])
        regret_gap_bound = delta_mu + expected_improvement + kappa * sqrt_half_kl

        threshold_points = torch.cat((new_best_point.reshape(1, -1), new_x), dim=0)
        _, threshold_covariance = _posterior_joint_in_original_scale(
            previous_pred_func, threshold_points
        )
        best_std = threshold_covariance[0, 0].clamp_min(0.0).sqrt()
        new_x_std = threshold_covariance[1, 1].clamp_min(0.0).sqrt()
        concentration = math.sqrt(-2.0 * math.log(delta))
        posterior_update_scale = (
            noise_variance.sqrt()
            / (threshold_covariance[1, 1].clamp_min(0.0) + noise_variance)
        )
        adaptive_threshold = (
            best_std * new_x_std + 0.5 * kappa * new_x_std
        ) * concentration * posterior_update_scale
        decision_margin = regret_gap_bound - adaptive_threshold

        stop_metric_val = [
            float(regret_gap_bound.detach().cpu()),
            float(adaptive_threshold.detach().cpu()),
            float(decision_margin.detach().cpu()),
        ]

    update_stop_data(
        stop_method_tol_dict,
        stop_metric_val,
        bo_model.curr_evals,
        bo_model.x_best,
        bo_model.fx_best,
        'Ishibashi',
    )
    return stop_metric_val


def _clopper_pearson_interval(num_successes, num_trials, risk):
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if not 0.0 < risk < 1.0:
        raise ValueError("risk must be in (0, 1).")

    lower = 0.0 if num_successes == 0 else stats.beta.ppf(
        risk / 2.0, num_successes, num_trials - num_successes + 1
    )
    upper = 1.0 if num_successes == num_trials else stats.beta.ppf(
        1.0 - risk / 2.0, num_successes + 1, num_trials - num_successes
    )
    return float(lower), float(upper)


def _sample_wilson_regret_failures(
        bo_model, test_point, num_samples, regret_bound,
        raw_samples=256, num_restarts=8,
):
    from botorch.sampling.pathwise import draw_matheron_paths
    from botorch.utils.sampling import optimize_posterior_samples

    pred_func = bo_model.pred_func
    model_dtype = pred_func.gp_model.train_inputs[0].dtype
    gp_model = deepcopy(pred_func.gp_model).to(dtype=model_dtype)
    previous_default_dtype = torch.get_default_dtype()
    try:
        # This BoTorch release creates Fourier features in the default dtype.
        torch.set_default_dtype(model_dtype)
        paths = draw_matheron_paths(gp_model, sample_shape=torch.Size([num_samples]))
    finally:
        torch.set_default_dtype(previous_default_dtype)
    unit_bounds = torch.stack((
        torch.zeros_like(bo_model.bounds[0]),
        torch.ones_like(bo_model.bounds[1]),
    ))
    _, optimum_values = optimize_posterior_samples(
        paths,
        bounds=unit_bounds,
        candidates=pred_func.x_train_scaled,
        raw_samples=max(int(raw_samples), int(num_restarts)),
        num_restarts=int(num_restarts),
        maximize=True,
    )
    test_point_scaled = pred_func.x_train_scaler.transform(test_point.reshape(1, -1))
    test_values = paths(test_point_scaled)
    optimum_values = optimum_values.reshape(num_samples, -1).max(dim=-1).values
    test_values = test_values.reshape(num_samples, -1)[:, 0]
    output_scale = pred_func.fx_train_scaler.std.detach().abs().reshape(-1)[0]
    regrets = (optimum_values - test_values) * output_scale
    return (regrets > regret_bound).detach().cpu().numpy().astype(np.int64)


def check_stop_method_Wilson(
        bo_model, stop_method_tol_dict, stop_metric_val,
        regret_bound=0.1, total_risk=0.05, num_assessments=1,
        max_samples=1000, initial_samples=64, sample_growth=1.5,
        risk_decay=1.1, raw_samples=256, num_restarts=8,
        require_convergence=False,
):
    """Wilson (NeurIPS 2024) Monte Carlo probabilistic regret bound."""
    posterior_mean, _, _, _ = bo_model.pred_func(bo_model.x)
    test_index = int(torch.argmax(posterior_mean))
    test_point = bo_model.x[test_index]
    test_value = bo_model.fx[test_index]

    if stop_metric_val is None:
        if regret_bound <= 0.0:
            raise ValueError("wilson_xi must be positive.")
        if not 0.0 < total_risk < 1.0:
            raise ValueError("wilson_rho must be in (0, 1).")
        if initial_samples <= 0 or max_samples < initial_samples:
            raise ValueError("Wilson sample limits are inconsistent.")
        if sample_growth <= 1.0 or risk_decay <= 1.0:
            raise ValueError("Wilson schedule factors must be greater than one.")

        model_risk = total_risk / 2.0
        estimation_risk = total_risk / 2.0
        assessment_risk = estimation_risk / max(1, int(num_assessments))
        failures = 0
        num_draws = 0
        failure_lower = 0.0
        failure_upper = 1.0
        converged = False
        round_index = 1

        while num_draws < max_samples:
            target_draws = min(
                int(max_samples),
                int(math.ceil(initial_samples * sample_growth ** (round_index - 1))),
            )
            if target_draws <= num_draws:
                target_draws = min(int(max_samples), num_draws + 1)
            batch_size = target_draws - num_draws
            batch_failures = _sample_wilson_regret_failures(
                bo_model,
                test_point,
                num_samples=batch_size,
                regret_bound=regret_bound,
                raw_samples=raw_samples,
                num_restarts=num_restarts,
            )
            failures += int(batch_failures.sum())
            num_draws = target_draws
            round_risk = (
                round_index ** (-risk_decay)
                * (risk_decay - 1.0)
                / risk_decay
                * assessment_risk
            )
            failure_lower, failure_upper = _clopper_pearson_interval(
                failures, num_draws, round_risk
            )
            converged = model_risk < failure_lower or model_risk > failure_upper
            if converged or num_draws >= max_samples:
                break
            round_index += 1

        failure_probability = failures / num_draws
        success_probability = 1.0 - failure_probability
        if converged and failure_probability <= model_risk:
            decision_margin = failure_upper - model_risk
        elif converged:
            decision_margin = failure_lower - model_risk
        elif require_convergence:
            decision_margin = abs(failure_probability - model_risk) + 1e-12
        else:
            # This is the capped-sample concession used in the paper's experiments.
            decision_margin = failure_probability - model_risk

        stop_metric_val = [
            float(success_probability),
            float(failure_lower),
            float(failure_upper),
            int(num_draws),
            bool(converged),
            float(decision_margin),
        ]

    update_stop_data(
        stop_method_tol_dict,
        stop_metric_val,
        bo_model.curr_evals,
        test_point,
        test_value,
        'Wilson',
    )
    return stop_metric_val

def check_stop_method_LR(
        bo_model, stop_LR_param_dict, stop_method_tol_dict,
        previous_metrics_val_list, max_evals,
        figure_run_mode, file_path, logger
):
    for tol_key, tol_dict in stop_method_tol_dict.items():
        stop_method_param_tol_dict = stop_LR_param_dict[tol_key]
        for HP_key, stop_method_param_dict in stop_method_param_tol_dict.items():
            if (stop_method_param_dict.get('terminated', False)
                    and stop_method_param_dict.get('enable_stop', True)):
                continue

            stop_start_flag = stop_method_param_dict['stop_start_flag']
            # === 判断是否独立优化 ===
            if stop_start_flag == 0:  # 复用现有优化结果
                stop_method_param_dict['bo_'] = deepcopy(bo_model)
                bo_stop = stop_method_param_dict['bo_']
            elif stop_start_flag == 1:  # 自此刻开始独立优化, 以免影响原优化过程
                bo_stop = stop_method_param_dict['bo_']
                if bo_stop.curr_evals < max_evals:  # 预算用完则不再优化
                    bo_stop.max_evals = bo_stop.curr_evals + 1  # 为了下面一句只进行一步优化
                    bo_stop.optimize()
            # === 停止检测 ===
            if bo_stop.curr_evals <= max_evals:  # 只有在没有达到最大评估次数时才进行停止检测
                # 解析参数
                cur_x_best, cur_fx_best = stop_method_param_dict['cur_x_best'], stop_method_param_dict['cur_fx_best']
                no_improve_count = stop_method_param_dict['no_improve_count']
                no_improve_patience = stop_method_param_dict['no_improve_patience']
                no_improve_patience_count = stop_method_param_dict['no_improve_patience_count']
                no_improve_patience_count_max = stop_method_param_dict['no_improve_patience_count_max']
                #   1. 检查最值是否有提高，并更新最优点和未提高的次数
                cur_x_best, cur_fx_best, no_improve_count = check_for_improve(bo_stop.x_best, bo_stop.fx_best, cur_x_best, cur_fx_best, no_improve_count)
                #   2. 如果连续没有提高的话，则在最优点附近局部增强
                if no_improve_count >= no_improve_patience:  # 连续没有提高
                    # if previous_metrics_val_list is not None and stop_start_flag == 0: # 还未独立优化过，且有之前的指标值列表
                    #     optimi_start_iter = previous_metrics_val_list[0]['iter']
                    #     incumbent_point_std = previous_metrics_val_list[bo_stop.curr_evals-optimi_start_iter]['stop_metric_val'][-1]
                    # else: # 已经独立优化过了，或者没有之前的指标值列表了，就不再判断之前的指标值了，直接进行局部搜索
                    incumbent_point_std = bo_stop.pred_func(cur_x_best.reshape(1, -1))[3].detach().item()
                    if incumbent_point_std < stop_method_param_dict['stop_tol']:  # 最值点预测方差小于阈值，该函数段已经拟合不错，局部搜索
                        # 3. 局部搜索
                        # 打印状态
                        # optimize_print(bo_stop, no_improve_count, logger)
                        log_print(logger, (
                            f"LR stop method === iter: {bo_stop.curr_evals}/{max_evals}, "
                            f"stop_tol: {stop_method_param_dict['stop_tol']}, "
                            f"incumbent_point_std: {incumbent_point_std:.6f}, "
                            f"no_improve_patience: {no_improve_count}/{no_improve_patience}, "
                            f"no_improve_patience_count: {no_improve_patience_count}/{no_improve_patience_count_max}, "
                            f"x_best: {cur_x_best.numpy()}, fx_best: {cur_fx_best.numpy()}, "
                        ))
                        # 在最优点附近局部增强
                        should_assess_local_region = (
                            stop_method_param_dict.get('enable_stop', True)
                            or stop_method_param_dict.get('enable_local_evaluation', True)
                        )
                        if should_assess_local_region:
                            x_local, fx_local, no_improve_patience = local_refine_func(
                                tol_dict[HP_key],
                                stop_method_param_dict,
                                no_improve_patience,
                                bo_stop,
                                cur_x_best,
                                max_evals,
                                logger
                            )
                        else:
                            x_local = torch.empty(
                                (0, cur_x_best.shape[0]),
                                dtype=cur_x_best.dtype,
                                device=cur_x_best.device,
                            )
                            fx_local = torch.empty(
                                (0, 1), dtype=cur_x_best.dtype, device=cur_x_best.device
                            )

                        if x_local.shape[0] > 0:
                            bo_stop.update_data(x_local, fx_local)
                            if stop_start_flag == 0:
                                stop_method_param_dict['stop_start_flag'] = 1
                        # 检查最值是否有提高，并更新最优点和未提高的次数
                        cur_x_best, cur_fx_best, no_improve_count = check_for_improve(bo_stop.x_best,
                                                                                      bo_stop.fx_best,
                                                                                      cur_x_best, cur_fx_best,
                                                                                      no_improve_count)
                        # 4. 探索增强
                        if (stop_method_param_dict.get('enable_exploration', True)
                                and no_improve_patience_count < no_improve_patience_count_max):
                            no_improve_patience_count += 1
                            bo_stop.tradeoff_param = min(
                                0.999999,
                                stop_method_param_dict['base_confidence']
                                + stop_method_param_dict['exploration_increment']
                                * (no_improve_patience_count / no_improve_patience_count_max),
                            )
                            if stop_start_flag == 0:
                                stop_method_param_dict['stop_start_flag'] = 1
                # 更新停止参数
                stop_method_param_dict['cur_x_best'], stop_method_param_dict['cur_fx_best'] = cur_x_best, cur_fx_best
                stop_method_param_dict['no_improve_count'] = no_improve_count
                stop_method_param_dict['no_improve_patience'] = no_improve_patience
                stop_method_param_dict['no_improve_patience_count'] = no_improve_patience_count
                # 更新停止状态
                tol_dict[HP_key]['metrics_val_list'].append({
                    'iter': bo_stop.curr_evals,
                    'stop_metric_val': [no_improve_count, bo_stop.pred_func(cur_x_best.reshape(1, -1))[3].detach().item()],
                })
                # 绘制图像
                # plot_results(bo_stop, 10, run_mode=figure_run_mode, file_path=file_path(bo_stop.curr_evals))

    return  stop_method_param_dict


def sub_bounds_mu_std_max_func(x_center, bounds, pred_func, cube_rate, init_pos=None, init_radius=None, local_frac=0.3):
    """
    构造局部区域，并在局部区域计算代理模型的最大均值和最大方差。
    Args:
        x_center: 中心点
        bounds: 原始搜索域
        pred_func: 代理模型的预测函数
        cube_rate: 局部区域相对于原始搜索域的大小比例
    Returns:
        mu_max: 局部区域的最大均值
        std_max: 局部区域的最大方差
        sub_lb: 局部区域的下界
        sub_ub: 局部区域的上界
    Raises:
        无。
    """
    # 计算超立方体的大小
    bound_diff = bounds[1, :] - bounds[0, :]
    cube_size = cube_rate * bound_diff
    # 计算超立方体的边界
    sub_lb = torch.clamp(x_center - cube_size / 2, min=bounds[0, :])
    sub_ub = torch.clamp(x_center + cube_size / 2, max=bounds[1, :])
    # 获取局部区域的均值最大值
    func_std = lambda x: -1 * pred_func(torch.tensor(x.reshape(1, -1), device=device, dtype=dtype))[2].detach().numpy()
    with redirect_stdout(open(os.devnull, 'w')):
        if init_pos is None:
            best_mu_position, best_value = pso(func_std, sub_lb.numpy().tolist(), sub_ub.numpy().tolist(),
                                               swarmsize=30, maxiter=100)
        else:
            best_mu_position, best_value = pso_x1(func_std, sub_lb.numpy().tolist(), sub_ub.numpy().tolist(),
                                                  swarmsize=30, maxiter=100,
                                                  init_pos=init_pos, init_radius=init_radius, local_frac=local_frac)
    mu_max = -best_value
    # 获取局部区域的方差最大值
    func_std = lambda x: -1 * pred_func(torch.tensor(x.reshape(1, -1), device=device, dtype=dtype))[3].detach().numpy()
    with redirect_stdout(open(os.devnull, 'w')):
        if init_pos is None:
            best_std_position, best_value = pso(func_std, sub_lb.numpy().tolist(), sub_ub.numpy().tolist(), swarmsize=30,
                                            maxiter=100)
        else:
            best_std_position, best_value = pso_x1(func_std, sub_lb.numpy().tolist(), sub_ub.numpy().tolist(),
                                                  swarmsize=30, maxiter=100,
                                                  init_pos=init_pos, init_radius=init_radius, local_frac=local_frac)
    std_max = -best_value
    return best_mu_position, mu_max.item(), best_std_position, std_max.item(), sub_lb, sub_ub


def local_refine_func(stop_method_tol_HP_dict, stop_method_param_dict, no_improve_patience, bo_model, x_center, max_evals, logger):
    bounds = bo_model.bounds
    pred_func = bo_model.pred_func
    obj_func = bo_model.obj_func

    def empty_eval():
        return (
            torch.empty((0, x_center.shape[0]), dtype=x_center.dtype, device=x_center.device),
            torch.empty((0, 1), dtype=x_center.dtype, device=x_center.device),
        )

    mu_max_pts, mu_max, std_max_pts, std_max, sub_lb, sub_ub = sub_bounds_mu_std_max_func(
        x_center, bounds, pred_func, cube_rate=stop_method_param_dict['k_local'])
    # 停止判断
    logger_str = f"LR stop method === iter: {bo_model.curr_evals}, "
    if std_max > stop_method_tol_HP_dict['stop_tol']:
        if stop_method_param_dict.get('enable_local_evaluation', True):
            local_evals_num = int(
                stop_method_param_dict['local_sample_multiplier'] * bounds.shape[1]
            )
            local_evals_num = min(local_evals_num, max_evals - bo_model.curr_evals)
            if local_evals_num > 0:
                x_local = latin_hypercube_sample_tensor(local_evals_num, sub_lb, sub_ub)
                fx_local = obj_func(x_local)
            else:
                x_local, fx_local = empty_eval()

            log_print(logger, f"{logger_str}"
                              f"Local evaluation {local_evals_num} points, because: "
                              f"local_std_max_{std_max:.6f} > stop_tol_{stop_method_tol_HP_dict['stop_tol']:.6f}, "
            )
        else:
            x_local, fx_local = empty_eval()
    else:
        incumbent_point_mu = pred_func(x_center.reshape(1, -1))[2].detach().item()
        delta_mu = mu_max - incumbent_point_mu
        eta_delta_mu = stop_method_param_dict['local_region_delta_mu']

        if (delta_mu < 0) and np.abs(delta_mu) > 1e-6:  # pso求得的局部mu最大点的mu值比中心点还小，且差值不太小了
            print(f'>> error: local mu max is smaller than center point mu, '
                  f'mu_max_{mu_max}, incumbent_point_mu_{incumbent_point_mu}, delta_mu_{delta_mu}, '
                  f'stop_tol_{stop_method_tol_HP_dict["stop_tol"]}, '
                  f'HP_name_{stop_method_tol_HP_dict["HP_name"]}, '
                  f'HP_val_{stop_method_tol_HP_dict["HP_val"]}, ')

            mu_max_pts, mu_max, _, _, _, _ = sub_bounds_mu_std_max_func(
                x_center, bounds, pred_func, cube_rate=stop_method_param_dict['k_local'],
                init_pos=mu_max_pts, init_radius=0.05*(sub_ub-sub_lb).numpy(), local_frac=0.5)
            delta_mu = mu_max - incumbent_point_mu

            if (delta_mu < 0) and np.abs(delta_mu) > 1e-6:
                print(f'>>> error: local mu max is smaller than center point mu again, '
                      f'mu_max_{mu_max}, incumbent_point_mu_{incumbent_point_mu}, delta_mu_{delta_mu}, '
                      f'stop_tol_{stop_method_tol_HP_dict["stop_tol"]}, '
                      f'HP_name_{stop_method_tol_HP_dict["HP_name"]}, '
                      f'HP_val_{stop_method_tol_HP_dict["HP_val"]}, ')
        if delta_mu < eta_delta_mu:
            x_local, fx_local = empty_eval()
            if stop_method_param_dict.get('enable_stop', True):
                no_improve_patience = np.inf
                stop_method_tol_HP_dict['stop_iter'] = bo_model.curr_evals
                stop_method_tol_HP_dict['stop_x_best'] = bo_model.x_best
                stop_method_tol_HP_dict['stop_fx_best'] = bo_model.fx_best
                stop_method_tol_HP_dict['stop_metrics_val'] = [std_max, delta_mu]
                stop_method_param_dict['terminated'] = True
                logger_str = logger_str + (f"Stop optimization, because: delta_mu_{delta_mu:.6f} < {eta_delta_mu} and ")
            else:
                logger_str = logger_str + (f"Continue optimization, because stopping is disabled although delta_mu_{delta_mu:.6f} < {eta_delta_mu} and ")
        else:
            if (stop_method_param_dict.get('enable_local_evaluation', True)
                    and bo_model.curr_evals < max_evals):
                x_local = torch.tensor(mu_max_pts, dtype=dtype, device=device).reshape(1, -1)
                fx_local = obj_func(x_local)
                logger_str = logger_str + f"Evaluate local mu-max point, because: delta_mu_{delta_mu:.6f} >= {eta_delta_mu} and "
            else:
                x_local, fx_local = empty_eval()
                logger_str = logger_str + f"Continue without local evaluation, because: delta_mu_{delta_mu:.6f} >= {eta_delta_mu} and "

        log_print(logger, f"{logger_str}"
                          f"local_std_max_{std_max:.6f} <= stop_tol_{stop_method_tol_HP_dict['stop_tol']:.6f}, "
                          f"local_mu_max_{mu_max:.6f}, "
                          f"incumbent_point_mu_{incumbent_point_mu:.6f}, ")

    return x_local, fx_local, no_improve_patience


def check_for_improve(bo_x_best, bo_fx_best, x_best, fx_best, no_improve_count):
    # 检查最值是否有提高
    if torch.allclose(bo_fx_best, fx_best, atol=1e-4, rtol=0, equal_nan=False):  # 未提高
        no_improve_count += 1
    else:  # 有提高
        no_improve_count = 1
    # 更新最优值
    fx_best = bo_fx_best
    x_best = bo_x_best
    # 返回更新后的最优点和未提高的次数
    return x_best, fx_best, no_improve_count


def optimize_print(bo_, no_improve_count, logger):
    print_str = (
        f"iter: {bo_.curr_evals:3d}, "
        f"best pt: {bo_.x_best.numpy()}, "
        f"best val: {bo_.fx_best.item():.{float_precision}f}, "
        f"no_improve_count: {no_improve_count}, "
        f"std: {bo_.pred_func(bo_.x_best.reshape(1, -1))[3].detach().item():.{float_precision}f}, "
    )
    log_print(logger, print_str)


def log_setup1(log_path, is_log, is_console):
    if is_log:
        # 清空 root logger 的所有 handler
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        # 设置日志文件路径
        create_folder(log_path)
        if os.path.exists(log_path):
            os.remove(log_path)
        # 获取独立的logger实例，设置日志记录器
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.INFO)  # 允许 INFO 级别及以上的日志
        # 防止重复设置
        if not logger.handlers:
            # 设置文件处理器
            handler = logging.FileHandler(log_path, encoding='utf-8')
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s >>> %(message)s')  # 设置日志格式
            handler.setFormatter(formatter)
            logger.addHandler(handler)  # 添加处理器到日志记录器

            if is_console:
                # 同时将日志输出到控制台
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            # 防止日志冒泡回 root
            logger.propagate = False
    else:
        logger = None

    return logger


def log_setup(log_path, is_log, is_console):
    if not is_log:
        return None

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 先关闭旧 handler，避免文件句柄残留
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s >>> %(message)s')

    # mode='w'：启动时清空文件，得到一个空白 txt
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if is_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def close_logger(logger):
    if logger is not None:
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)


def log_print(logger, print_str):
    if logger is not None:
        logger.info(print_str)


def task_iter(process_id = 0, params_dict = {}):
    leet_config = {
        'no_improve_patience': 10,
        'max_exploration_steps': 10,
        'eta_delta_mu': 0.01,
        'local_region_scale': 0.01,
        'local_sample_multiplier': 10,
        'base_confidence': 0.95,
        'exploration_increment': 0.03,
    }
    leet_config.update(params_dict.get('leet_config', {}))
    ablation_config = {
        'enable_stop': True,
        'enable_exploration': True,
        'enable_local_evaluation': True,
    }
    ablation_config.update(params_dict.get('ablation_config', {}))

    # 设置随机种子
    set_seed(42 + process_id)
    params_dict['process_id'] = process_id
    params_dict['seed'] = 42 + process_id
    # 记录开始时间
    # start_time = time.time()
    # 忽略所有警告
    warnings.filterwarnings("ignore")
    # 文件保存目录
    file_dir = params_dict['data_figure_dir']
    # # 开启matlab引擎
    # eng = matlab.engine.start_matlab()
    # Q1 MATLAB path is configured with the Q1_MATLAB_PATH environment variable.
    # 定义目标函数
    func_name = params_dict['obj_func_name']  # 'Ackley', 'Sin', 'Levy', 'Schwefel', 'SVM_magic_gamma_dataset', 'Swimmer', 'XGBoost', 'Hartmann6D'
    dims = params_dict['num_dims']  # Number of dimensions
    is_maximize = True  # 是否最大化目标函数, False 为最小化， True 为最大化
    obj_func = get_function_details(func_name, dims)
    bounds = obj_func.bounds
    mff = obj_func.mf_funcs
    func = mff.fidel_func[mff.num_fidels-1]
    func = func if is_maximize else lambda x: -1 * func(x)  # 使用 lambda 对 obj_func 的输出取反
    lb, ub = bounds[0, :], bounds[1, :]
    num_dims = obj_func.num_dims
    # 定义采集函数
    acq_func_name = params_dict['acq_func_name'] # 'UCB', 'EI', 'PI'
    # 定义图像参数
    data_file_dir = file_dir / f"figure/{func_name}/{acq_func_name}/process_{process_id}/"
    file_path = lambda num_evals: data_file_dir / f"image_{num_evals}.png"
    figure_run_mode = params_dict['figure_run_mode'] if num_dims == 1 else 0  # 0: 不绘制图像，1: 绘制图像，2: 绘制图像并保存
    if figure_run_mode == 2:
        delete_folder(data_file_dir)
    # 设置日志配置，将输出写入文件
    log_path = str(file_dir / f'data/log/{func_name}_dims{num_dims}_{acq_func_name}_process_{process_id}.txt')
    logger = log_setup(log_path, is_log=params_dict['is_log'], is_console=params_dict['is_log_to_console'])  # True False
    # 定义bo参数
    max_evals = params_dict['hf_eval_num'] * num_dims  # Maximum Function Evaluations
    confidence_level = leet_config['base_confidence']  # Confidence level for UCB
    # 定义初始采样点数量
    init_num = 10 * num_dims  # Number of initial samples
    # 初始化bo
    optimi_run_mode = params_dict['optimi_run_mode']  # 0: 从头开始优化，1: 继续上次的优化
    is_save_snapshot = params_dict['is_save_snapshot']
    if optimi_run_mode == 0:  # 0: 从头开始优化，1: 继续上次的优化
        # 定义初始采样点
        x_init = latin_hypercube_sample_tensor(init_num, lb, ub)
        x_init = latin_hypercube_sample_tensor(init_num, lb, ub)
        fx_init = func(x_init)

        bo_ = bo(
            func,
            bounds,
            x_init,
            fx_init,
            acq_func_name,
            x_init.shape[0] + 1,
            tradeoff_param = confidence_level,
            is_save_snapshot = is_save_snapshot
        )
        previous_metrics_val_list_dict = {}
    elif optimi_run_mode == 1:
        # 读取上次的优化结果（如果有）并更新bo的初始数据
        file_path = file_dir / f"data/{func_name}_dims{num_dims}_{acq_func_name}_process_{process_id}.pkl"
        previous_data = load_data_from_pkl(file_path)
        previous_bo = previous_data['bo_']  # 用这个来更新当前的bo对象
        previous_metrics_val_list_dict = {
            stop_method_name: next(iter(next(iter(stop_method_tol_dict.values())).values()))['metrics_val_list']
            for stop_method_name, stop_method_tol_dict in previous_data['stop_method_dict'].items()
        }

        bo_ = deepcopy(previous_bo)
        bo_.load_gp_snapshot(init_num)
        bo_.is_save_snapshot = is_save_snapshot

    # 定义停止参数
    stop_method_dict = params_dict['stop_method_dict']
    ishibashi_params = deepcopy(params_dict.get('ishibashi_params', {}))
    wilson_params = deepcopy(params_dict.get('wilson_params', {}))
    wilson_params.setdefault('num_assessments', max(1, max_evals - init_num))
    #   定义LR方法的过程数据
    stop_LR_param_dict = {}
    if 'LR' in stop_method_dict:
        for tol_key, tol_dict in stop_method_dict['LR'].items():
            stop_LR_param_dict[tol_key] = {}
            for HP_key, HP_dict in tol_dict.items():
                stop_LR_param_dict[tol_key][HP_key] = {
                    'method_name': 'LR',
                    'stop_tol': HP_dict['stop_tol'],  # 停止阈值
                    'bo_': deepcopy(bo_),  # 局部增强的bo
                    'cur_x_best': bo_.x_best,  # 最优点
                    'cur_fx_best': bo_.fx_best,  # 最优值
                    'no_improve_count': 1,  # 最值没有提升的次数
                    'no_improve_patience': leet_config['no_improve_patience'],
                    'no_improve_patience_count': 0,  # 突破容忍值的次数
                    'no_improve_patience_count_max': leet_config['max_exploration_steps'],
                    'local_region_delta_mu': leet_config['eta_delta_mu'],
                    'k_local': leet_config['local_region_scale'],
                    'local_sample_multiplier': leet_config['local_sample_multiplier'],
                    'base_confidence': leet_config['base_confidence'],
                    'exploration_increment': leet_config['exploration_increment'],
                    'enable_stop': ablation_config['enable_stop'],
                    'enable_exploration': ablation_config['enable_exploration'],
                    'enable_local_evaluation': ablation_config['enable_local_evaluation'],
                    'terminated': False,
                    'stop_start_flag': 0,  # 0表示未开始独立优化，1表示开始独立优化
                }
    #   定义Covi方法的过程数据
    stop_Covi_param_dict = {
        'method_name': 'Cov_i',
        'cur_x_best': bo_.x_best,  # 最优点
        'cur_fx_best': bo_.fx_best,  # 最优值
        'no_improve_count': 0,  # 最值没有提升的次数
    }

    # 优化开始
    print_str = f"********** process_id: {process_id}, func_name: {func_name}, num_dims: {num_dims}, acq_func_name: {acq_func_name} start **********"
    log_print(logger, print_str)
    print(print_str)

    try:
        # 运行bo
        while bo_.curr_evals < max_evals:
            # 状态打印
            if bo_.curr_evals % 10 == 0:
                print_str = (f"{func_name}/dim{num_dims}/{acq_func_name}/process_{process_id} "
                             f"[{datetime.fromtimestamp(time.time())}] >>> "
                             f"iter: {bo_.curr_evals}/{max_evals}, ")
                print(print_str)
            # 运行bo =================================================================================================
            previous_pred_func = bo_.pred_func if 'Ishibashi' in stop_method_dict else None
            if optimi_run_mode == 0:
                bo_.max_evals = bo_.curr_evals + 1
                bo_.optimize()
            elif optimi_run_mode == 1:
                curr_evals = bo_.curr_evals
                bo_ = deepcopy(previous_bo)
                bo_.load_gp_snapshot(curr_evals + 1)

            # 停止检测 start =================================================================================================
            for stop_method_name, stop_method_tol_dict in stop_method_dict.items():
                previous_metrics_val_list = previous_metrics_val_list_dict.get(stop_method_name, None)  # 获取对应方法的之前的指标值列表, 如果没有就设置为 None
                if previous_metrics_val_list is not None:  # 如果有之前的指标值列表，并且当前评估次数还没有超过之前的指标值列表的范围，就使用之前的指标值，否则就不使用
                    stop_metric_val = previous_metrics_val_list[
                        bo_.curr_evals - previous_metrics_val_list[0]['iter']
                    ]['stop_metric_val']
                else: stop_metric_val = None

                if stop_method_name in ['LR']:
                    check_stop_method_LR(bo_, stop_LR_param_dict, stop_method_tol_dict, stop_metric_val, max_evals, figure_run_mode, file_path, logger)
                elif stop_method_name in ['CBOS']:
                    check_stop_method_CBOS(bo_, stop_method_tol_dict, stop_metric_val)
                elif stop_method_name in ['EI']:
                    check_stop_method_EI(bo_, stop_method_tol_dict, stop_metric_val)
                elif stop_method_name in ['CB']:
                    check_stop_method_CB(bo_, stop_method_tol_dict, stop_metric_val)
                elif stop_method_name in ['PI']:
                    check_stop_method_PI(bo_, stop_method_tol_dict, stop_metric_val)
                elif stop_method_name in ['Cov_i']:
                    check_stop_method_Cov_i(bo_, stop_Covi_param_dict, stop_method_tol_dict, stop_metric_val)
                elif stop_method_name in ['Ishibashi']:
                    check_stop_method_Ishibashi(
                        bo_, previous_pred_func, stop_method_tol_dict, stop_metric_val,
                        **ishibashi_params,
                    )
                elif stop_method_name in ['Wilson']:
                    check_stop_method_Wilson(
                        bo_, stop_method_tol_dict, stop_metric_val,
                        **wilson_params,
                    )
            # 停止检测 end ===================================================================================================

        # 更新停止数据
        for stop_method_name, stop_method_tol_dict in stop_method_dict.items():
            for tol_key, tol_dict in stop_method_tol_dict.items():
                for HP_key, HP_dict in tol_dict.items():
                    if stop_method_name == 'LR':
                        bo_model = stop_LR_param_dict[tol_key][HP_key]['bo_']
                        # x = stop_LR_param_dict[tol_key][HP_key]['bo_'].x
                        # fx = stop_LR_param_dict[tol_key][HP_key]['bo_'].fx
                        # stop_LR_param_dict[tol_key][HP_key]['bo_'] = None
                        HP_dict['stop_process_data'] = stop_LR_param_dict[tol_key][HP_key]
                        # HP_dict['stop_process_data']['x'] = x
                        # HP_dict['stop_process_data']['fx'] = fx

                    else:
                        bo_model = bo_

                    if HP_dict['stop_iter'] is None:
                        HP_dict['stop_iter'] = max_evals
                        HP_dict['stop_x_best'] = bo_model.x_best
                        HP_dict['stop_fx_best'] = bo_model.fx_best
                        logger_str = f"{stop_method_name} 未达到停止准则："
                    else:
                        logger_str = f"{stop_method_name} 达到停止阈值："

                    logger_str = (f"{logger_str} >>> "
                                  f"stop_iter: {HP_dict['stop_iter']}, "
                                  f"stop_x_best: {HP_dict['stop_x_best'].numpy()}, "
                                  f"stop_fx_best: {HP_dict['stop_fx_best'].numpy()}, "
                                  f"stop_metrics_val: {HP_dict['stop_metrics_val']}. ")  # 0 表示非正常停止
                    log_print(logger, logger_str)
                    print(logger_str)
        # 打印结果 ===========================================================================================================
        # optimize_print(bo_ls, no_improve_count)
        print_str = (f"{func_name}/{num_dims}dim/{acq_func_name}/process_{process_id} >>> "
                     f"Opt. complete. opti_evals:{bo_.curr_evals:3d}, opti_x:{bo_.x_best.numpy()}, opti_fx:{bo_.fx_best.item():.{float_precision}f}\n")
        log_print(logger, print_str)
        print(print_str)
        # 绘制图像 ===========================================================================================================
        # plot_results(stop_LR_param_dict['bo_'], 10, run_mode=figure_run_mode, file_path=file_path(stop_LR_param_dict['bo_'].curr_evals))
        #   累积遗憾

        def plot_best_so_far(bo_model, stop_method_dict, title, file_path, figure_run_mode):
            if figure_run_mode == 0:
                return

            x1, y1 = bo_model.get_best_so_far()
            for tol_i in range(1):
                plt.plot(x1, y1.numpy().flatten(), 'b-', linewidth=3.0, label='bo_ rewares')
                for stop_method_name, stop_method_tol_dict in stop_method_dict.items():
                    stop_info_dict = list(stop_method_dict[stop_method_name].items())[tol_i][1]['HP_nst']
                    if stop_method_name == 'LR':
                        x2, y2 = stop_info_dict['stop_process_data']['bo_'].get_best_so_far()
                        plt.plot(x2, y2.numpy().flatten(), 'r--', linewidth=2.0, label='bo_lr rewares')
                    setting = stop_method_setting_dict[stop_method_name]
                    plt.axvline(
                        x=stop_info_dict['stop_iter'],
                        linestyle=setting['linestyle'],
                        linewidth=setting['linewidth'],
                        color=setting['color'],
                        alpha=setting['alpha'],
                        zorder=setting['zorder'] + 1,  # ⭐ 稍微抬高一层
                        label=f"{setting['name']}"
                    )
                plt.xlabel('eval_num')
                plt.ylabel('Rewards')
                plt.title(f'{title}_tol_{tol_i}')
                plt.legend()
                # plt_settings.grid_config(plt, grid_on=True)
                plt_settings.plt_base_settings(plt.gca())

                if figure_run_mode == 1:
                    plt.show()
                elif figure_run_mode == 2:
                    plt_settings.fig_save(plt, f'{file_path}_tol_{tol_i}', format='png')
                    plt.close()

        plot_best_so_far(
            bo_, stop_method_dict,
            title=f'{func_name} dim{num_dims} {acq_func_name} process {process_id}',
            file_path=file_dir/f"figure/cumReward_{func_name}_dim{num_dims}_{acq_func_name}_processor_{process_id}",
            figure_run_mode=figure_run_mode)
        # 保存结果 ===========================================================================================================
        save_data_to_pkl(
            {
                'params_dict': params_dict,
                'bo_': bo_,
                'stop_method_dict': stop_method_dict,
            },
            file_dir/f"data/{func_name}_dims{num_dims}_{acq_func_name}_process_{process_id}.pkl"
        )
        # 显式关闭 MATLAB 引擎
        # eng.quit()
        # 记录结束时间
        # end_time = time.time()
        # elapsed_time = end_time - start_time  # 计算耗时
        # print(f"代码执行耗时: {elapsed_time:.6f} 秒")
        # 返回最优值 =========================================================================================================
    finally:
        close_logger(logger)

    return bo_.fx_best.item()


def parameter_config():
    # 参数基本配置 =======================================================================================================
    params_config_dict = {
        "obj_func_names": [  # 目标函数名称
            # ['Airfoil_AF1', 16],
            # ['Airfoil_AF3', 16],
            # ['Airfoil_AF2', 16],
            # 可变维度s
            *[[func, dim] for dim in [
                1,
                # 2,
                # 5,
                # 10,
            ] for func in [
                'Ackley',
                'Levy',
                'Schwefel',
                'Rastrigin',
                'Griewank',
                'Michalewicz',
            ]],
            # 固定维度
            # ['Branin', 2],
            # ['Hartmann3D', 3],
            # ['Hartmann6D', 6],
            # ['SVM_BreastCancer', 2],
            # ['MLP_Digits', 5],
            # ['RandomForest_California_Housing', 5],
            # ['XGBoost_Diabetes', 8],
            # ['LightGBM_AdultIncome', 10]
        ],
        "acq_func_names": [  # 采集函数名称
            'UCB',
            # 'EI',
            # 'PI',
        ],
        # 停止准则参数
        "stop_method_settings": [
            ['LR', [1.14e-02]],
            ['CBOS', [2.17]],
            ['CB', [1.94e-02]],
            ['EI', [5.29e-05]],
            ['PI', [5.85e-04]],
            ['Cov_i', [x for x in [-74]]],
            ['Ishibashi', [0.0]],
            ['Wilson', [0.0]],
        ],
        "optimi_run_mode": 0,  # 0: 从头开始优化，1: 继续上次的优化
        "hf_eval_num": 50,  # 高保真度评估次数
        "figure_run_mode": 0,  # 0: 不绘制图像，1: 绘制图像仅显示，2: 绘制图像仅保存
        "is_log": False,  # 是否将日志输出到文件 True False
        "is_log_to_console": False,  # 是否将日志输出到控制台 True False
        "ishibashi_params": {
            "delta": 0.1,
            "num_candidates": 2048,
        },
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
        # LR停止参数配置
        # "no_improve_patience": [x * 2 for x in range(1, 10)],
        # "no_improve_patience_count_max": [x * 2 for x in range(1, 10)],
        # "k_l": [x * 0.002 for x in range(1, 11)],
        "is_save_snapshot": False,  # 是否保存bo的快照数据 True False
        'HP_val': ['nst'],  # LR停止准则的超参数  nst:没有超参数
        'HP_name': 'nst',  # 用于记录的超参数名称 local_region_delta_mu
        # 运行参数配置
        "start_process_id": 0,  # 进程起始ID
        "num_processes": 30,  # 重复次数
        "is_multi_processing": False,  # 是否多进程 True False
        "process_batch_size": 30#psutil.cpu_count(logical=False)-2,  # 多进程批处理大小 (获取CPU物理核心数)
    }
    # 参数处理 ===========================================================================================================
    #   停止准则组合
    stop_method_dict = {}
    for stop_method_setting_i in params_config_dict["stop_method_settings"]:
        stop_method_name = stop_method_setting_i[0]
        stop_method_tols = stop_method_setting_i[1]
        stop_method_tol_dict = {}
        for stop_method_tol_i in stop_method_tols:
            stop_method_tol_dict[f"tol_{stop_method_tol_i}"] = {}
            for stop_method_tol_HP_i in params_config_dict["HP_val"]:
                stop_method_tol_dict[f"tol_{stop_method_tol_i}"][f"HP_{stop_method_tol_HP_i}"] = {
                    "stop_tol": stop_method_tol_i,
                    "stop_iter": None,
                    "stop_x_best": None,
                    "stop_fx_best": None,
                    'stop_metrics_val': None,
                    "metrics_val_list": [],
                    'HP_val': stop_method_tol_HP_i,
                    'HP_name': params_config_dict['HP_name'],
                }
        stop_method_dict[stop_method_name] = stop_method_tol_dict
    #   所有参数组合
    params_dict_list = []
    # for num_dims_i in [1]:
    for obj_func_name_i in params_config_dict["obj_func_names"]:
        for acq_func_name_i in params_config_dict["acq_func_names"]:
            params_dict = {
                "obj_func_name": obj_func_name_i[0],
                "acq_func_name": acq_func_name_i,
                "hf_eval_num": params_config_dict["hf_eval_num"],
                "figure_run_mode": params_config_dict["figure_run_mode"],  # 0: 不绘制图像，1: 绘制图像，2: 绘制图像并保存
                "is_log": params_config_dict["is_log"],  # 是否将日志输出到文件 True False
                "is_log_to_console": params_config_dict["is_log_to_console"],  # 是否将日志输出到控制台 True False
                "is_save_snapshot": params_config_dict["is_save_snapshot"],  # 是否保存bo的快照数据 True False
                "optimi_run_mode": params_config_dict["optimi_run_mode"],  # 0: 从头开始优化，1: 继续上次的优化
                "ishibashi_params": deepcopy(params_config_dict["ishibashi_params"]),
                "wilson_params": deepcopy(params_config_dict["wilson_params"]),
                'stop_method_dict': deepcopy(stop_method_dict),
                'num_dims': obj_func_name_i[1],
            }
            params_dict_list.append(params_dict)

    return params_config_dict, params_dict_list


if __name__ == "__main__":
    from htslibs.utils.multi_process import process_mode

    # 设置打印精度为6位小数
    np.set_printoptions(precision=float_precision, suppress=True)
    # 设置文件保存路径
    dt = datetime.fromtimestamp(time.time())
    formatted = dt.strftime("%Y-%m-%d %H-%M-%S")
    data_figure_dir = data_figure_dir / f'{formatted}'

    # 获取参数配置
    params_config_dict, params_dict_list = parameter_config()
    is_multi_processing = params_config_dict["is_multi_processing"]
    num_processes = params_config_dict["num_processes"]
    start_process_id = params_config_dict["start_process_id"]
    # process_ids = [10,16,23,16,27]
    process_ids = range(start_process_id, num_processes + start_process_id)
    process_batch_size = params_config_dict["process_batch_size"]

    # 执行任务
    for params_dict in params_dict_list:
        # 数据缓存 ===
        # if params_dict['obj_func_name'] == 'XGBoost_CaliforniaHousing':
        #     X, y = fetch_california_housing(return_X_y=True)
        # else: X, y = None, None
        # params_dict['obj_dataset'] = [X,y]
        # ===========
        params_dict['data_figure_dir'] = data_figure_dir
        results = process_mode(
            task_iter,
            params_dict_list=[deepcopy(params_dict) for _ in range(num_processes)],
            multiy_process_params=process_ids,
            batch_size=process_batch_size,
            is_multi_processing=is_multi_processing
        )
        # 提取执行结果
        fx_best_list = [x["result"] for x in results if x["success"]]
        failed_list = [x for x in results if not x["success"]]
        print("成功数量:", len(fx_best_list))
        print("失败数量:", len(failed_list))
        for item in failed_list:
            print(f"\nprocess {item['process_id']} 失败:")
            print(item["error"])
        print('')
