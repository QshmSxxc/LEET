import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import combinations
from scipy.optimize import minimize
from pyswarm import pso
import os
from contextlib import redirect_stdout
from scipy.stats import norm
import scipy.stats as stats


def convexity_check(convex_check_x, pred_func, is_maximize=True):
    # 初始化
    number_checked = 0
    number_convex = 0
    # 生成数据对的组合
    combines = [c for c in combinations(range(len(convex_check_x)), 2)]
    # 对每一对点进行处理
    for i in range(0, len(combines)):
        # 计算均值
        x1 = convex_check_x[combines[i][0], :].reshape(1, -1)
        x2 = convex_check_x[combines[i][1], :].reshape(1, -1)
        x3 = (x1 + x2) / 2
        _, _, f1, _ = pred_func(x1)
        _, _, f2, _ = pred_func(x2)
        _, _, f3, _ = pred_func(x3)
        # 凸性判断
        if is_maximize:
            if (f1 + f2) / 2 < f3 or abs((f1 + f2) / 2 - f3) < 1e-5:
                number_convex += 1
        else:
            if (f1 + f2) / 2 < f3 or abs((f1 + f2) / 2 - f3) < 1e-5:
                number_convex += 1
        number_checked += 1

    # 计算凸性比例
    convex_rate = number_convex / number_checked
    return convex_rate


def is_within_bounds(x, bounds):
    # 遍历每个维度，检查是否在对应的 bounds 范围内
    for i in range(len(bounds)):
        lb, ub = bounds[i]  # 获取当前维度的下界和上界
        if not (lb <= x[i] <= ub):  # 检查当前维度的值是否在区间内
            return False  # 如果有一个维度不在区间内，返回 False

    return True  # 所有维度都在区间内，返回 True


def local_regret_cal(device, dtype, convex_check_x_tensor, pred_func, x_best_tensor):
    x_best = x_best_tensor.numpy()
    # 初始化局部遗憾
    # local_regret = np.array([np.inf])
    # 将全局最优点转换为 numpy 数组
    # x_best = x_best_tensor.detach().numpy()
    # 计算convex_check_x的上下界
    sub_lb = torch.min(convex_check_x_tensor, dim=0).values.cpu().numpy()  # 每列的最小值
    sub_ub = torch.max(convex_check_x_tensor, dim=0).values.cpu().numpy()  # 每列的最大值
      # 检查哪些维度的上界不大于下界
    for i in range(len(sub_lb)):
        if sub_ub[i] <= sub_lb[i]:
            # 增加上界或减少下界，确保ub[i] > lb[i]
            sub_ub[i] = sub_lb[i] + 1e-6  # 微调上界
    sub_bounds = [(sub_lb[i], sub_ub[i]) for i in range(0, len(sub_lb))]

    # # 判断x_best是否在convex_check_x围成的局部区域中
    # if is_within_bounds(x_best, sub_bounds):

    # 求convex_check_x空间内mu最小的点
    func_mu = lambda x: -1 * pred_func(torch.tensor(x.reshape(1, -1), device=device, dtype=dtype))[2].detach().numpy()
    # func_mu = lambda x: x[0]**2
    res = minimize(func_mu, x_best.ravel(), method='L-BFGS-B', bounds=sub_bounds, tol=1e-10)
    # x, y = pso(func_mu, sub_lb.tolist(), sub_ub.tolist(), swarmsize=30, maxiter=100)
    rex_y = res.fun
    distance = np.abs(rex_y - func_mu(x_best))

    # 求convex_check_x空间内std最大的点
    convex_check_x = convex_check_x_tensor.numpy()
    def Dist(x):
        x = np.tile(x, [len(convex_check_x), 1])
        x = np.array(x)
        dist = convex_check_x - x
        dist = np.square(dist)
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        dist = np.sum(dist)
        return -1 * dist
    func_std = lambda x: -1 * pred_func(torch.tensor(x.reshape(1, -1), device=device, dtype=dtype))[3].detach().numpy()

    res = minimize(Dist, x_best, method='L-BFGS-B', bounds=sub_bounds, tol=1e-5)
    var_x_init = res.x
    res = minimize(func_std, var_x_init, method='L-BFGS-B', bounds=sub_bounds, tol=1e-10)
    # with redirect_stdout(open(os.devnull, 'w')):
    #     _, std_max_op = pso(func_std, sub_lb.tolist(), sub_ub.tolist(), swarmsize=30, maxiter=100)
    std_max = -1 * res.fun

    local_regret = distance + std_max

    # test_x = torch.linspace(torch.tensor(sub_lb[0], device=device, dtype=dtype),
    #                         torch.tensor(sub_ub[0], device=device, dtype=dtype), 1000).reshape(-1, 1)
    # test_std_without_noise = std_without_noise_func(test_x).detach().numpy()
    # test_std = pred_func(test_x)[1].detach().numpy()

    # print(f'distance: {type(distance)}, std_max: {type(std_max)}, local_regret: {type(local_regret)}')
    # print(f'x_best: {x_best}, res_x: {res.x}')
    return local_regret.item(), distance.item(), std_max.item() if isinstance(std_max, np.ndarray) else std_max


def Approximate_LowerBound(device, dtype, data_set_x_tensor, bounds_tensor, pred_func):
    bounds = np.asarray(bounds_tensor)
    _, _, data_set_pred_tensor, _ = pred_func(data_set_x_tensor)
    x_best_tensor = data_set_x_tensor[torch.max(data_set_pred_tensor, dim=0)[1]]
    x_best = np.asarray(x_best_tensor).reshape(-1)
    data_set_x = np.asarray(data_set_x_tensor)
    # 观测噪声标准差
    noise = pred_func.gp_model.likelihood.noise
    # 重定义预测函数
    def pred_func_redefine(x):
        mean, std, mean_scaled, std_scaled = pred_func(x)
        std_scaled = torch.sqrt(std_scaled**2 + noise)
        std = std_scaled * pred_func.fx_train_scaler.std
        return mean, std, mean_scaled, std_scaled


    def Mu(x):
        x_torch = torch.tensor(x.reshape(1, -1)).to(device=device, dtype=dtype)
        y_means, _, _, _ = pred_func_redefine(x_torch)
        return -1 * y_means.detach().numpy()

    def Std(x):
        x_torch = torch.tensor(x.reshape(1, -1)).to(device=device, dtype=dtype)
        _, _, _, y_stds = pred_func_redefine(x_torch)  # 标准化之后的标准差是为了和原论文对齐
        return -1 * y_stds.detach().numpy()

    def UCB(x):
        x_torch = torch.tensor(x.reshape(1, -1)).to(device=device, dtype=dtype)
        y_means, _, _, y_stds = pred_func_redefine(x_torch)
        ucb = y_means + stats.norm.ppf((1 + 0.95) / 2) * y_stds
        return -1 * ucb.detach().numpy()

    def Dist(x):
        x = np.tile(x, [len(data_set_x), 1])
        x = np.array(x)
        dist = data_set_x - x
        dist = np.square(dist)
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        dist = np.sum(dist)
        return -1 * dist

    # 求convex_check_x空间内mu最大的点
    res_mu_local_max = minimize(Mu, x_best, method='L-BFGS-B', bounds=bounds.T, tol=1e-10)
    x_local_max = res_mu_local_max.x
    mu_local_max = -1 * res_mu_local_max.fun
    # 求convex_check_x空间内std最大的点
    sub_lb_torch, _ = torch.min(data_set_x_tensor, dim=0)
    sub_ub_torch, _ = torch.max(data_set_x_tensor, dim=0)
    sub_lb = np.asarray(sub_lb_torch).reshape(-1)
    sub_ub = np.asarray(sub_ub_torch).reshape(-1)
    sub_bounds = [[a, b] for a, b in zip(sub_lb, sub_ub)]
    res_dist = minimize(Dist, x_best, method='L-BFGS-B', bounds=sub_bounds, tol=1e-5)
    res_std_local_max = minimize(Std, res_dist.x, method='L-BFGS-B', bounds=sub_bounds, tol=1e-10)
    std_local_max = -1 * res_std_local_max.fun
    # 计算输入空间内的ucb最大值
    res_ucb_max = minimize(UCB, x_best, method='L-BFGS-B', bounds=bounds.T, tol=1e-10)
    ucb_max = -1 * res_ucb_max.fun
    x_ucb_max_tensor = torch.tensor(res_ucb_max.x).to(device=device, dtype=dtype)
    mu_ucb_max_tensor, _, _, std_ucb_max_tensor = pred_func_redefine(x_ucb_max_tensor.reshape(1,-1))
    mu_ucb_max, std_ucb_max = mu_ucb_max_tensor.item(), std_ucb_max_tensor.item()
    beta = (ucb_max - mu_ucb_max) / std_ucb_max
    # regret = var_lcb + var_max + (mu_min - mu_lcb)
    # rate = var_max / regret
    local_regret = (beta * (std_ucb_max + std_local_max) + (mu_ucb_max - mu_local_max)) / (beta * np.sqrt(noise.detach().numpy()))
    # rate = beta*var_max/regret

    # local_regret = regret

    parts = [
        f"x_local_max: {x_local_max}",
        f"x_ucb_max: {res_ucb_max.x}",
        f"std_ucb_max: {std_ucb_max}",
        f"std_local_max: {std_local_max}",
        f"mu_ucb_max: {mu_ucb_max}",
        f"mu_local_max: {mu_local_max}",
    ]

    return local_regret, ", ".join(parts) + ", "

# def Approximate_LowerBound(device, dtype, DataSet, bounds, pred_func):
#     x_scaler, y_scaler, gp_model = pred_func(None)
#
#     bounds_torch = torch.tensor(bounds.T).to(device=device, dtype=dtype)
#     bounds_torch = x_scaler.transform(bounds_torch)
#     bounds = np.asarray(bounds_torch).T
#
#     X_predict = DataSet
#     X_predict_torch = torch.tensor(X_predict).to(device=device, dtype=dtype)
#     X_predict_torch = x_scaler.transform(X_predict_torch)
#     Y_predict = gp_model.likelihood(gp_model(X_predict_torch)).mean.detach().numpy()
#     index2 = np.where(Y_predict == Y_predict.max())
#     X_best = X_predict_torch[index2[0][0], :]
#     x_best = np.asarray(X_best).reshape(-1)
#
#     DataSet = np.asarray(X_predict_torch)
#
#     def MU(Xc):
#         Xc_torch = torch.tensor(Xc.reshape(1, -1)).to(device=device, dtype=dtype)
#         y_means = gp_model.likelihood(gp_model(Xc_torch)).mean
#         y_means = y_scaler.inverse_transform(y_means)
#         return -1 * y_means.detach().numpy()
#
#     def Var(Xc):
#         Xc_torch = torch.tensor(Xc.reshape(1, -1)).to(device=device, dtype=dtype)
#         y_vars = gp_model.likelihood(gp_model(Xc_torch)).variance
#         return -1 * y_vars.detach().numpy()
#
#     def UCB(Xc):
#         Xc_torch = torch.tensor(Xc.reshape(1, -1)).to(device=device, dtype=dtype)
#         y_vars = gp_model.likelihood(gp_model(Xc_torch)).variance.detach().numpy()
#         y_means = gp_model.likelihood(gp_model(Xc_torch)).mean
#         y_means = y_scaler.inverse_transform(y_means).detach().numpy()
#
#         conf_intveral = norm.interval(0.95, loc=y_means, scale=y_vars)
#         return -1 * conf_intveral[1]
#
#     def Dist(Xc):
#         XC = np.tile(Xc, [len(DataSet), 1])
#         XC = np.array(XC)
#         dist = DataSet - XC
#         dist = np.square(dist)
#         dist = np.sum(dist, axis=1)
#         dist = np.sqrt(dist)
#         dist = np.sum(dist)
#         return -1 * dist
#
#     x0 = np.asarray(x_best).reshape(-1)
#     tol = 1e-10
#     res = minimize(MU, x0, method='L-BFGS-B', bounds=bounds, tol=tol)
#     mu_min = -1 * MU(res.x)
#
#     lb_a = np.min(DataSet, axis=0)
#     ub_a = np.max(DataSet, axis=0)
#     lb = np.asarray(lb_a).reshape(-1)
#     ub = np.asarray(ub_a).reshape(-1)
#     bounds_var = [[a, b] for a, b in zip(lb, ub)]
#     res0 = minimize(Dist, x0, method='L-BFGS-B', bounds=bounds_var, tol=1e-5)
#     x1 = res0.x
#     x_var = np.asarray(x1).reshape(-1)
#     tol = 1e-10
#     res = minimize(Var, x_var, method='L-BFGS-B', bounds=bounds_var, tol=tol)
#     var_max = Var(res.x) * -1
#     var_x = res.x
#
#     # Calculate the regret
#     tol = 1e-10
#     x_lcb_init = x0
#     bounds_lcb = bounds
#     res = minimize( UCB, x_lcb_init, method='L-BFGS-B', bounds=bounds_lcb, tol=tol)
#     lcb_min = -1 * UCB(res.x)
#     x_lcb = np.asarray(res.x).reshape(-1)
#     x_lcb = torch.tensor(x_lcb, requires_grad=True).to(device=device, dtype=dtype)
#     mu_lcb = gp_model.likelihood(gp_model(x_lcb.unsqueeze(dim=0))).mean
#     mu_lcb = y_scaler.inverse_transform(mu_lcb).detach().numpy() #mu + sigma * mu_lcb
#     var_lcb = gp_model.likelihood(gp_model(x_lcb.unsqueeze(dim=0))).variance.detach().numpy()
#     beta = (lcb_min - mu_lcb) / var_lcb
#     # regret = var_lcb + var_max + (mu_min - mu_lcb)
#     # rate = var_max / regret
#     regret = (beta * (var_lcb + var_max) + (mu_min - mu_lcb)) / beta
#     # rate = beta*var_max/regret
#
#     local_regret = regret/(gp_model.likelihood.noise.detach().numpy())
#
#     print(f"var_lcb: {var_lcb}, var_max: {var_max}, mu_min: {mu_min}, mu_lcb: {mu_lcb}, beta: {beta}, regret: {regret}, noise: {gp_model.likelihood.noise.detach().numpy()}, local_regret: {local_regret}")
#
#     return regret, beta, local_regret

