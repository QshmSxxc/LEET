"""Two-factor sensitivity experiment for k_local and k_d."""

import logging
import os
import warnings
from contextlib import redirect_stdout
from copy import deepcopy
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._runner import load_replay_data, replay_metric, run_script

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

# 'FF4419', '#A5FF19', '#195BFF', '#8193C2', '#90AD63', '#AD6F61'
# ['LR   ', [0.015], ],  # 当前最优点标准差阈值
# ['CBOS ', [2.10]],  # 0.02, 0.05, 0.08
# ['CB   ', [5]],  # 0.26, 0.62, 0.97
# ['EI   ', [0.001]],  # 0.0003, 0.0005, 0.0007
# ['PI   ', [0.01]],  # 1e-13, 1e-9, 1e-5
# ['Cov_i', [-x for x in [-5]]],  # 50, 30, 10
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
        'linestyle': (0, (5, 1)),   # 比 (1,1) 更清晰
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
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证 pytorch 完全确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_extremum(x0, func, tol=1e-2):
    # 确保x0是(1, d)的2D张量且可求导
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

def check_stop_method_LR(
        bo_model, stop_LR_param_dict, stop_method_tol_dict,
        previous_metrics_val_list, max_evals,
        figure_run_mode, file_path, logger
):
    for tol_key, tol_dict in stop_method_tol_dict.items():
        stop_method_param_tol_dict = stop_LR_param_dict[tol_key]
        for HP_key, stop_method_param_dict in stop_method_param_tol_dict.items():
            stop_start_flag = stop_method_param_dict['stop_start_flag']
            # === 判断是否独立优化 ===
            if stop_start_flag == 0:  # 复用现有优化结果
                stop_method_param_dict['bo_'] = deepcopy(bo_model)
                bo_stop = stop_method_param_dict['bo_']
            elif stop_start_flag == 1:  # 自此刻开始独立优化, 以免影响原优化过程
                bo_stop = stop_method_param_dict['bo_']
                if bo_stop.curr_evals < max_evals:  # 预算用完则不再优化
                    bo_stop.max_evals = bo_stop.curr_evals + 1  # 为了下面一句只进行一步优化
                    if not hasattr(bo_stop, 'is_save_snapshot'):
                        bo_stop.is_save_snapshot = getattr(bo_model, 'is_save_snapshot', False)
                    bo_stop.optimize()
            # === 停止检测 ===
            if bo_stop.curr_evals < max_evals:  # 只有在没有达到最大评估次数时才进行停止检测
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
                        if stop_start_flag == 0:
                            stop_method_param_dict['stop_start_flag'] = 1 # 从下一次开始独立优化
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
                        x_local, fx_local, no_improve_patience = local_refine_func(
                            tol_dict[HP_key],
                            stop_method_param_dict,
                            no_improve_patience,
                            # stop_method_param_dict['k_local'],
                            bo_stop,
                            cur_x_best,
                            max_evals,
                            logger
                        )
                        bo_stop.update_data(x_local, fx_local)
                        # 检查最值是否有提高，并更新最优点和未提高的次数
                        cur_x_best, cur_fx_best, no_improve_count = check_for_improve(bo_stop.x_best,
                                                                                      bo_stop.fx_best,
                                                                                      cur_x_best, cur_fx_best,
                                                                                      no_improve_count)
                        # 4. 探索增强
                        if no_improve_patience_count < no_improve_patience_count_max:
                            no_improve_patience_count += 1
                            stop_method_param_dict['NEE'] += 1
                            # bo_stop.tradeoff_param = 0.95 * (1 + 0.03 * 0.1 * no_improve_patience_count)
                            tradeoff_param_base = stop_method_param_dict['tradeoff_param_base']
                            bo_stop.tradeoff_param = min(
                                0.999999,
                                tradeoff_param_base + 0.03 * (no_improve_patience_count / no_improve_patience_count_max)
                            )
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

    mu_max_pts, mu_max, std_max_pts, std_max, sub_lb, sub_ub = sub_bounds_mu_std_max_func(
        x_center, bounds, pred_func, cube_rate=stop_method_param_dict['k_local'])
    # 停止判断
    logger_str = f"LR stop method === iter: {bo_model.curr_evals}, "
    if std_max > stop_method_tol_HP_dict['stop_tol']:
        stop_method_param_dict['NLE'] += 1
        # 确定局部评估数量
        remaining_budget = max_evals - bo_model.curr_evals
        local_evals_num = min(
            stop_method_param_dict['k_d'] * bounds.shape[1],
            remaining_budget,
        )
        # 在局部区域内进行评估
        x_local = latin_hypercube_sample_tensor(local_evals_num, sub_lb, sub_ub)
        fx_local = obj_func(x_local)

        log_print(logger, f"{logger_str}"
                          f"Local evaluation {local_evals_num} points, because: "
                          f"local_std_max_{std_max:.6f} > stop_tol_{stop_method_tol_HP_dict['stop_tol']:.6f}, "
        )
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
            # 用来占位
            x_local = torch.empty((0, x_center.shape[0]), dtype=x_center.dtype, device=x_center.device)
            fx_local = torch.empty((0, 1), dtype=x_center.dtype, device=x_center.device)
            # 更新停止数据
            no_improve_patience = np.inf
            stop_method_tol_HP_dict['stop_iter'] = bo_model.curr_evals
            stop_method_tol_HP_dict['stop_x_best'] = bo_model.x_best
            stop_method_tol_HP_dict['stop_fx_best'] = bo_model.fx_best
            stop_method_tol_HP_dict['stop_metrics_val'] = [std_max, delta_mu]
            # 日志输出
            logger_str = logger_str + (f"Stop optimization, because: delta_mu_{delta_mu:.6f} < {eta_delta_mu} and ")
        else:
            # 评估该mu最大点
            x_local = torch.tensor(mu_max_pts, dtype=dtype, device=device).reshape(1, -1)
            fx_local = obj_func(x_local)
            # 日志输出
            logger_str = logger_str + f"Not Local evaluation, because: delta_mu_{delta_mu:.6f} >= {eta_delta_mu} and "

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
    # eng.eval("addpath(genpath('D:\A_HXR\A_Thesis\paper2\zzn\zzn_optim_min_3_var_dV'))", nargout=0)
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
    max_evals = params_dict.get('max_evals', params_dict['hf_eval_num'] * num_dims)
    confidence_level = 0.95  # Confidence level for UCB
    is_save_snapshot = params_dict.get('is_save_snapshot', False)
    # 定义初始采样点数量
    init_num = 10 * num_dims  # Number of initial samples
    # 初始化bo
    optimi_run_mode = params_dict['optimi_run_mode']  # 0: 从头开始优化，1: 继续上次的优化
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
        previous_data = load_replay_data(params_dict, process_id)
        previous_data['bo_'].is_save_snapshot = is_save_snapshot
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
    #   定义LR方法的过程数据
    stop_LR_param_dict = {}
    if 'LR' in stop_method_dict:
        for tol_key, tol_dict in stop_method_dict['LR'].items():
            stop_LR_param_dict[tol_key] = {}
            for HP_key, HP_dict in tol_dict.items():
                local_region_delta_mu = 1e-2
                tradeoff_param_base = 0.95
                k_local = 0.01
                k_d = 10
                if HP_dict['HP_name'] == 'local_region_delta_mu':
                    local_region_delta_mu = HP_dict['HP_val']
                elif HP_dict['HP_name'] in ['tradeoff_param_base', 'basic_confidence_level']:
                    tradeoff_param_base = HP_dict['HP_val']
                elif HP_dict['HP_name'] == 'k_local_k_d':
                    k_local, k_d = HP_dict['HP_val']
                else:
                    raise ValueError(f"Unknown LR hyperparameter: {HP_dict['HP_name']}")
                stop_LR_param_dict[tol_key][HP_key] = {
                    'method_name': 'LR',
                    'stop_tol': HP_dict['stop_tol'],  # 停止阈值
                    'bo_': deepcopy(bo_),  # 局部增强的bo
                    'cur_x_best': bo_.x_best,  # 最优点
                    'cur_fx_best': bo_.fx_best,  # 最优值
                    'no_improve_count': 1,  # 最值没有提升的次数
                    'no_improve_patience': 10,  # 能够容忍最值没有提升的最大次数
                    'no_improve_patience_count': 0,  # 突破容忍值的次数
                    'no_improve_patience_count_max': 10,  # 突破容忍值的最大次数
                    'local_region_delta_mu': local_region_delta_mu,  # 局部区域的mu最大值与中心点mu的差值阈值
                    'tradeoff_param_base': tradeoff_param_base,  # 探索增强的基础置信水平
                    'k_local': k_local,
                    'k_d': k_d,
                    'NEE': 0,
                    'NLE': 0,
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
                # 打印状态
                # optimize_print(bo_, stop_LR_param_dict['tol_0.01']['no_improve_count'], logger)  # stop_LR_param_dict['bo_']
                # 绘制图像
                # plot_results(stop_LR_param_dict['tol_0.01']['bo_'], 10, run_mode=figure_run_mode,
                #              file_path=file_path(stop_LR_param_dict['tol_0.01']['bo_'].curr_evals))

            # 运行bo =================================================================================================
            if optimi_run_mode == 0:
                bo_.max_evals = bo_.curr_evals + 1
                bo_.optimize()
            elif optimi_run_mode == 1:
                curr_evals = bo_.curr_evals
                bo_ = deepcopy(previous_bo)
                bo_.load_gp_snapshot(curr_evals + 1)
                bo_.is_save_snapshot = is_save_snapshot

            # 停止检测 start =================================================================================================
            for stop_method_name, stop_method_tol_dict in stop_method_dict.items():
                previous_metrics_val_list = previous_metrics_val_list_dict.get(stop_method_name, None)  # 获取对应方法的之前的指标值列表, 如果没有就设置为 None
                stop_metric_val = replay_metric(previous_metrics_val_list, bo_.curr_evals)

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
                        HP_dict['NEE'] = stop_LR_param_dict[tol_key][HP_key]['NEE']
                        HP_dict['NLE'] = stop_LR_param_dict[tol_key][HP_key]['NLE']
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
                                  f"HP_name: {HP_dict['HP_name']}, "
                                  f"HP_val: {HP_dict['HP_val']}, "
                                  f"NEE: {HP_dict.get('NEE')}, "
                                  f"NLE: {HP_dict.get('NLE')}, "
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

            x1, y1 = bo_.get_best_so_far()
            for tol_i in range(3):
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
    k_local_values = [0.005, 0.01, 0.02]
    k_d_values = [5, 10, 20]
    # 参数基本配置 =======================================================================================================
    params_config_dict = {
        # "obj_func_names": [  # 目标函数名称
        #     # 可变维度s
        #     *[[func, dim] for dim in [
        #         1,
        #         2,
        #         5,
        #         10
        #     ] for func in [
        #         'Ackley',
        #         # 'Levy',
        #         # 'Schwefel',
        #         # 'Rastrigin',
        #         # 'Griewank',
        #         # 'Michalewicz',
        #         # 'Dixon_Price',  # 剔除（停止阈值太异常）
        #     ]],
        #     # 固定维度
        #     # ['Branin', 2],
        #     # ['Hartmann3D', 3],
        #     # ['Hartmann6D', 6],
        #     # ['SVM_BreastCancer', 2],
        #     # ['MLP_Digits', 5],
        #     # ['RandomForest_California_Housing', 5],
        #     ['XGBoost_Diabetes', 8],
        #     # ['LightGBM_AdultIncome', 10]
        #     # ['Rosenbrock', 2],  # 剔除（停止阈值太异常）
        #     # ['RandomForest_Diabetes', 5],  # 剔除（没做）
        # ],
        # 用于灵敏度分析的典型 benchmark
        "obj_func_names": [
            ["Branin", 2],
            ["Hartmann6D", 6],
            ["Schwefel", 10],
            ["LightGBM_AdultIncome", 10],
        ],
        "acq_func_names": [  # 采集函数名称
            'UCB',
            # 'EI',
            # 'PI',
        ],
        # 停止准则参数
        "stop_method_settings": [
            # === 阈值选择用 ===
            # ['LR', [-1]],
            # ['CBOS', [-1]],
            # ['CB', [-1]],
            # ['EI', [-1]],
            # ['PI', [-1]],
            # ['Cov_i', [x for x in [-10000]]],
            # === 费效平衡分析用 ===
            # ['LR', [3.85e-03, 1.14e-02, 4.17e-02]],  # 当前最优点标准差阈值
            # ['CBOS', [2.04, 2.17, 2.93]],  # 0.02, 0.05, 0.08
            # ['CB', [3.44e-03, 1.94e-02, 1.99e-01]],  # 0.26, 0.62, 0.97
            # ['EI', [6.41e-07, 5.29e-05, 5.37e-04]],  # 0.0003, 0.0005, 0.0007
            # ['PI', [4.53e-06, 5.85e-04 , 6.21e-03]],  # 1e-13, 1e-9, 1e-5
            # ['Cov_i', [-x for x in [-167, -74, -37]]],  # 50, 30, 10
            # === 灵敏度分析用 ===
            ['LR', [1.14e-02]]
        ],
        "optimi_run_mode": 1,  # 0: 从头开始优化，1: 继续上次的优化
        "hf_eval_num": 50,  # 高保真度评估次数
        "figure_run_mode": 0,  # 0: 不绘制图像，1: 绘制图像仅显示，2: 绘制图像仅保存
        "is_log": False,  # 是否将日志输出到文件 True False
        "is_log_to_console": False,  # 是否将日志输出到控制台 True False
        # LR停止参数配置
        # "no_improve_patience": [x * 2 for x in range(1, 10)],
        # "no_improve_patience_count_max": [x * 2 for x in range(1, 10)],
        # "k_l": [x * 0.002 for x in range(1, 11)],
        'k_local_values': k_local_values,
        'k_d_values': k_d_values,
        'default_interaction_params': (0.01, 10),
        'HP_val': [
            (k_local, k_d)
            for k_local in k_local_values
            for k_d in k_d_values
        ],
        'HP_name': 'k_local_k_d',
        # 运行参数配置
        "start_process_id": 0,  # 进程起始ID
        "num_processes": 30,  # 重复次数
        "is_multi_processing": True,  # 是否多进程 True False
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
                "optimi_run_mode": params_config_dict["optimi_run_mode"],  # 0: 从头开始优化，1: 继续上次的优化
                'stop_method_dict': deepcopy(stop_method_dict),
                'num_dims': obj_func_name_i[1],
            }
            params_dict_list.append(params_dict)

    return params_config_dict, params_dict_list


if __name__ == "__main__":
    raise SystemExit(run_script(task_iter, parameter_config, __file__))
