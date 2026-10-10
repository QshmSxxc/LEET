# We implement the Bayesian optimization algorithm in this file.
from copy import deepcopy

import torch
import torch.distributions as dist
from botorch.utils.probability.utils import (ndtr as Phi, phi)
import numpy as np
from botorch.acquisition import AcquisitionFunction
import scipy.stats as stats
from botorch.optim import optimize_acqf

from htslibs.bayesian_optimization_lib.gaussian_process_model import gp
from htslibs.utils.load_data_from_json import load_config_from_json
from htslibs.bayesian_optimization_lib.utils import StandardScaler_Torch, ToUnitCubeScaler_Torch


# 获取参数配置
params_config = load_config_from_json()
device, dtype = torch.device(params_config["device"]), eval(params_config["dtype"])
float_precision = params_config["float_precision"]
data_figure_dir = params_config["data_figure_dir"]

class ConfidenceBound_acq (AcquisitionFunction):
    def __init__(self, pred_func, confidence_level=0.95, upper_or_lower='upper'):
        super().__init__(model=None)
        self.pred_func = pred_func
        self.confidence_level = confidence_level
        self.upper_or_lower = upper_or_lower
        self.z_value = stats.norm.ppf((1 + self.confidence_level) / 2)

    def forward(self, x):
        _, _, mean_scaled, std_scaled = self.pred_func(x)
        if self.upper_or_lower == 'upper':
            cb = mean_scaled + self.z_value * std_scaled
        elif self.upper_or_lower == 'lower':
            cb = mean_scaled - self.z_value * std_scaled

        return cb


class ExpectedImprovement_acq(AcquisitionFunction):
    def __init__(self, pred_func, best_observed_value, xi=0.0):
        """
        初始化期望改进采集函数
        :param pred_func: 高斯过程预测函数，返回均值和标准差
        :param best_observed_value: 目前为止观察到的最优函数值（即已知的目标函数最小值）
        :param xi: 控制探索与利用的权衡，xi越大越偏向探索
        """
        super().__init__(model=None)
        self.pred_func = pred_func
        self.best_observed_value = best_observed_value
        self.xi = xi

    def forward(self, x):
        """
        计算期望改进
        :param x: 待评估的点（numpy数组）
        :return: 返回期望改进的值
        """
        mean, std, mean_scaled, std_scaled = self.pred_func(x)
        # 计算期望改进的公式
        improvement = mean_scaled - self.best_observed_value - self.xi
        # 将改进标准化为标准正态分布的输入
        z = improvement / std_scaled
        # 计算期望改进
        ei = improvement * Phi(z) + std_scaled * phi(z)

        return ei


class ProbabilityOfImprovement_acq(AcquisitionFunction):
    def __init__(self, pred_func, best_observed_value, kappa=0.95):
        """
        初始化Probability of Improvement (PI)采集函数
        :param pred_func: 高斯过程预测函数，返回均值和标准差
        :param best_observed_value: 当前已知的最优函数值
        :param kappa: 用于控制探索和利用的平衡，默认为0.95
        """
        super().__init__(model=None)
        self.pred_func = pred_func
        self.best_observed_value = best_observed_value
        self.kappa = kappa

    def forward(self, x):
        """
        计算给定点的Probability of Improvement (PI)
        :param x: 待评估的点（张量）
        :return: 返回该点的Probability of Improvement
        """
        # 从预测函数中获取均值和标准差
        _, _, mean_scaled, std_scaled = self.pred_func(x)

        # 计算标准化改进 (standardized improvement)
        improvement = mean_scaled - self.best_observed_value
        # 标准化改进，得到z值
        z = improvement / std_scaled

        # 创建标准正态分布对象
        normal_dist = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))  # 标准正态分布
        # 计算标准正态分布的 CDF（即改进概率）
        pi_value = normal_dist.cdf(z)

        return pi_value


class bo:
    def __init__(self,
            obj_func, bounds, x_init, fx_init,
            acq_func_name, max_evals, tradeoff_param,
            is_save_snapshot,
    ):
        self.obj_func = obj_func
        self.bounds = bounds
        self.lb = bounds[0, :]
        self.ub = bounds[1, :]
        self.x = x_init
        self.fx = fx_init
        self.acq_func_name = acq_func_name
        self.max_evals = max_evals
        self.curr_evals = x_init.shape[0]
        # self.eval_num = self.curr_evals * np.ones(self.curr_evals)  # 评估数量的累积
        self.x_best = x_init[torch.argmax(fx_init)]
        self.fx_best = torch.max(fx_init)
        self.batch_size = 1
        self.tradeoff_param = tradeoff_param  # 探索和利用的权衡参数 UCB: confidence_level, EI: xi, PI: kappa
        self.gp_snapshot_list = []
        self.pred_func = None
        self.is_save_snapshot = is_save_snapshot

        # 训练GP模型
        self.x_train_scaler = None
        self.fx_train_scaler = None
        # # 初始输入数据归一化
        # self.x_train_scaler = ToUnitCubeScaler_Torch(bounds[0, :], bounds[1, :])
        # self.x_train_scaler.fit(x_init)
        # # 初始输出数据标准化
        # self.fx_train_scaler = StandardScaler_Torch()
        # self.fx_train_scaler.fit(fx_init)

        self.pred_func = self.update_model(self.bounds, self.x, self.fx, self.x_train_scaler, self.fx_train_scaler)
        if self.is_save_snapshot:
            self.save_gp_snapshot()

    @staticmethod
    def _acq_function(pred_func, acq_func_name, fx_best_scaled, tradeoff_param):
        if acq_func_name == 'UCB':
            acq_func = ConfidenceBound_acq(pred_func, tradeoff_param, 'upper')
            # beta = stats.norm.ppf((1 + self.kappa) / 2) ** 2
            # acq_func = UpperConfidenceBound(model=model, beta=beta)
        elif acq_func_name == 'LCB':
            acq_func = ConfidenceBound_acq(pred_func, tradeoff_param, 'lower')
        elif acq_func_name == 'EI':
            acq_func = ExpectedImprovement_acq(pred_func, fx_best_scaled, tradeoff_param)
            # acq_func = ExpectedImprovement(model=model, best_f=fx_best_scaled, maximize=True)
        elif acq_func_name == 'PI':
            acq_func = ProbabilityOfImprovement_acq(pred_func, fx_best_scaled, tradeoff_param)
            # acq_func = ProbabilityOfImprovement(model, best_f=fx_best_scaled, xi=0.01)
        else:
            raise ValueError(f"Unknown acquisition function: {acq_func_name}")
        return acq_func

    def acq_func_optimization(self, pred_func, fx_best, acq_func_name, tradeoff_param):
        # 选择采集函数
        fx_best_scaled = pred_func.fx_train_scaler.transform(fx_best)
        acq_function = self._acq_function(pred_func, acq_func_name, fx_best_scaled, tradeoff_param)
        # 优化采集函数
        candidate_pts_scaled, acq_vals = optimize_acqf(
            acq_function=acq_function,
            bounds=self.bounds,  # train_x_scaler.transform(self.bounds),#to_unit_cube(self.bounds, self.lb, self.ub).T,
            q=self.batch_size,  # number of candidates to generate
            num_restarts=20,  # number of starting points for multistart optimization
            raw_samples=256,  # number of samples for initialization
        )
        return candidate_pts_scaled, acq_vals

    def update_data(self, x_new, fx_new):
        num_x_new = x_new.shape[0]
        self.x = torch.cat((self.x, x_new))
        self.fx = torch.cat((self.fx, fx_new))
        self.curr_evals += x_new.shape[0]
        # 记录当前迭代的最优值和最优点
        for i in range(num_x_new):
            if (fx_new[i, :] - self.fx_best).item() > 1e-10:
                self.x_best = x_new[i, :]  # 更新最优点
                self.fx_best = fx_new[i, :]  # 更新最优值

    @staticmethod
    def update_model(bounds, x, fx, x_train_scaler, fx_train_scaler):
        pred_func = gp(bounds, x, fx, x_train_scaler, fx_train_scaler)
        return pred_func

    def get_best_so_far(self):
        # cum_reward = torch.cumsum(self.fx, dim=0)
        cum_reward = np.maximum.accumulate(self.fx)
        eval_num = range(1, self.fx.shape[0]+1)
        return eval_num, cum_reward

    def save_gp_snapshot(self):
        if self.pred_func is None:
            raise ValueError("pred_func is None, cannot save snapshot before model is built.")

        snapshot = {
            # ===== BO 状态 =====
            'iter': int(self.curr_evals),
            # 'x': self.x.detach().cpu().clone(),
            # 'fx': self.fx.detach().cpu().clone(),
            # 'x_best': self.x_best.detach().cpu().clone(),
            # 'fx_best': self.fx_best.detach().cpu().clone(),
            # 'bounds': self.bounds.detach().cpu().clone(),
            # 'acq_func_name': self.acq_func_name,
            # 'tradeoff_param': self.tradeoff_param,
            # 'max_evals': self.max_evals,

            # ===== GP 状态 =====
            'gp_state_dict': deepcopy({
                k: v.detach().cpu().clone()
                for k, v in self.pred_func.gp_model.state_dict().items()
            }),
            'x_train_scaler': self.pred_func.x_train_scaler.state_dict(),
            'fx_train_scaler': self.pred_func.fx_train_scaler.state_dict(),
            'mean_func': self.pred_func.mean_func,
        }
        self.gp_snapshot_list.append(snapshot)

    def load_gp_snapshot(self, iter):
        """
        传整数 iter，自动在 gp_snapshot_list 中寻找
        """
        iter = int(iter)
        # 获取快照
        matched = [s for s in self.gp_snapshot_list if s['iter'] == iter]
        if len(matched) == 0:
            raise ValueError(f"No snapshot found for iter={iter}")
        snapshot = matched[-1]

        # ===== 先恢复 BO 状态 =====
        self.curr_evals = iter
        self.x = self.x[:iter, :]
        self.fx = self.fx[:iter, :]
        self.x_best = self.x[torch.argmax(self.fx)]
        self.fx_best = torch.max(self.fx)
        # self.bounds = snapshot['bounds'].to(device)
        # self.lb = self.bounds[0, :]
        # self.ub = self.bounds[1, :]
        # self.acq_func_name = snapshot['acq_func_name']
        # self.tradeoff_param = snapshot['tradeoff_param']
        # self.max_evals = snapshot['max_evals']

        # ===== 恢复 scaler =====
        x_train_scaler = ToUnitCubeScaler_Torch()
        x_train_scaler.load_state_dict(snapshot['x_train_scaler'])

        fx_train_scaler = StandardScaler_Torch()
        fx_train_scaler.load_state_dict(snapshot['fx_train_scaler'])

        # ===== 重建 GP 并加载参数，不重新训练 =====
        self.pred_func = gp(
            bounds=self.bounds,
            x_train=self.x,
            fx_train=self.fx,
            x_train_scaler=x_train_scaler,
            fx_train_scaler=fx_train_scaler,
            mean_func=snapshot['mean_func'],
            gp_model_state_dict={
                k: v.to(device) for k, v in snapshot['gp_state_dict'].items()
            },
            fit_model_flag=False,
        )

    def optimize(self):
        while self.curr_evals < self.max_evals:
            # 优化采集函数
            candidate_pts_scaled, acq_vals = self.acq_func_optimization(self.pred_func, self.fx_best, self.acq_func_name, self.tradeoff_param)
            # print(f"iter: {self.curr_evals}, Acq_vals: {acq_vals.item()}, Current best pt: {self.x_best}, Current best value: {self.fx_best.item()}")
            # 评估目标函数
            x_next = candidate_pts_scaled
            fx_next = self.obj_func(x_next)
            # 更新数据(不可置于 训练预测模型 之后)
            self.update_data(x_next, fx_next)
            # 训练预测模型
            self.pred_func = self.update_model(self.bounds, self.x, self.fx, self.x_train_scaler, self.fx_train_scaler)
            if self.is_save_snapshot:
                self.save_gp_snapshot()

        # 计算累积奖励
        # x, y = self.get_cum_rewards()
        # plt.plot(x, y.numpy().flatten(), label='Cumulative Regret')
        # 返回优化结果
        # optimize_result = {
        #     "pred_func": self.pred_func,
        #     "acq_vals": acq_vals
        # }
        # return optimize_result


################################################## 测试 ############################################################################
import warnings
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from htslibs.objective_functions.obj_functions import get_function_details
from htslibs.utils.sample_methods import latin_hypercube_sample_tensor
from htslibs.utils.file_operation_lib.file_operation import create_folder


def plot_results(bo_, convex_check_num, run_mode=1, file_path=""):
    """
    :param bo_:
    :param lb:
    :param ub:
    :param obj_func:
    :param pred_func:
    :param is_maximize:
    :param run_mode: 0:不运行图像，1:显示图形, 2:不显示图像，保存图像
    :return:
    """
    if run_mode != 0:
        # 图形化输出
        test_x = torch.linspace(bo_.bounds[0,0], bo_.bounds[1,0], 1000).reshape(-1, 1)
        test_fx_np = bo_.obj_func(test_x).numpy()
        test_mu, test_std, _, _ = bo_.pred_func(test_x)
        test_mu_np = test_mu.detach().numpy()
        test_std_np = test_std.detach().numpy()
        test_x_np = test_x.numpy()
        test_lcb_np = test_mu_np - stats.norm.ppf((1 + bo_.tradeoff_param) / 2) * test_std_np
        test_ucb_np = test_mu_np + stats.norm.ppf((1 + bo_.tradeoff_param) / 2) * test_std_np

        plt.clf()
        plt.plot(test_x_np, test_fx_np, color='r', linewidth=1.5, label='Objective Function')
        plt.plot(test_x_np, test_mu_np, color=tuple(c / 255 for c in (14, 159, 255, 255)), linewidth=1,
                 label='Predicted Function')
        plt.fill_between(test_x.reshape(-1), test_lcb_np, test_ucb_np, color="lightblue", linewidth=0, alpha=0.5,
                         label="Confidence Region (LCB-UCB)")  # 置信区间
        plt.scatter(bo_.x.flatten(), bo_.fx.flatten(), color='m', s=15, zorder=3, label='Sample Points')
        plt.scatter(bo_.x[-convex_check_num:, :].flatten(),
                    bo_.fx[-convex_check_num:, :].flatten(), color='k', s=15, zorder=3,
                    label='Convex Check Points')
        plt.scatter(bo_.x_best, bo_.fx_best, color='g', marker='*', s=50, zorder=3, label='Optimal Point')
        plt.scatter(bo_.x[-1:, :].flatten(), bo_.fx[-1:, :].flatten(),
                    color=tuple(c / 255 for c in (255, 195, 50, 255)), s=5, zorder=3, label='New Points')
        plt.text(bo_.x_best, bo_.fx_best,
                 f'({bo_.x_best.item():.4f}, {bo_.fx_best.item():.4f})',
                 color='k', fontsize=10, ha='right', va='bottom')
        plt.legend(prop=FontProperties(size=8), framealpha=0.5)
        plt.grid(color=tuple(c / 255 for c in (128, 128, 128, 50)), linestyle='--', linewidth=0.1)
        if run_mode == 1:
            plt.show()
        elif run_mode == 2:
            create_folder(file_path)
            os.remove(file_path) if os.path.exists(file_path) else None
            plt.savefig(file_path, dpi=300, bbox_inches='tight')  # dpi 设置分辨率，bbox_inches='tight' 去除多余空白
            plt.close()


def task_iter(process_id = 0, params_dict = {}):
    # 忽略所有警告
    warnings.filterwarnings("ignore")

    # 定义目标函数
    func_name = params_dict['func_name']  # 'Ackley', 'Sin', 'Levy', 'Schwefel', 'SVM_magic_gamma_dataset', 'Swimmer', 'XGBoost', 'Hartmann6D'
    dims = 1  # Number of dimensions
    is_maximize = True  # 是否最大化目标函数, False 为最小化， True 为最大化
    obj_func = get_function_details(func_name, dims)
    bounds = obj_func.bounds
    mff = obj_func.mf_funcs
    obj_func = mff.fidel_func[mff.num_fidels-1]
    func = obj_func if is_maximize else lambda x: -1 * obj_func(x)  # 使用 lambda 对 obj_func 的输出取反
    lb, ub = bounds[0, :], bounds[1, :]
    dims = lb.shape[0]
    # 定义采集函数
    acq_func_name = params_dict['acq_func_name'] # 'UCB', 'EI', 'PI'
    # 定义初始采样点
    init_num = 1#10 * dims  # Number of initial samples
    x_init = latin_hypercube_sample_tensor(init_num, lb, ub, dtype=dtype, device=device)
    fx_init = func(x_init)
    # 定义bo参数
    bo_mode = 1  # 0: 直接运行， 1: 迭代运行
    max_evals = 1 * dims  # Maximum Function Evaluations
    n_evals = max_evals - 1 if bo_mode==0 else x_init.shape[0]
    figure_run_mode = 2 if dims==1 else 0  # 0: 不绘制图像，1: 显示图像，2: 保存图像

    # 初始化BO
    if acq_func_name == 'UCB':
        tradeoff_param = 0.95
    elif acq_func_name == 'EI':
        tradeoff_param = 0.00
    else:
        tradeoff_param = 0.95
    bo_ = bo(
        func,
        bounds,
        x_init,
        fx_init,
        acq_func_name,
        n_evals + 1,
        tradeoff_param = tradeoff_param
    )

    print(
        f"********** process_id: {process_id}, func_name: {func_name}, acq_func_name: {acq_func_name} start **********\n"
        f"==============================================================================================================")
    # 运行BO
    while n_evals < max_evals:
        # pred_func = deepcopy(bo_.pred_func)
        # 运行BO
        bo_.optimize()
        # acq_val = result['acq_vals']
        # 更新评估次数
        n_evals = bo_.curr_evals
        bo_.max_evals = n_evals + 1

        # 绘制图像
        if bo_mode != 0:
            plot_results(bo_, 10, run_mode=figure_run_mode,
                         file_path=data_figure_dir / f"figure/{func_name}/{acq_func_name}/process_{process_id}/image_{n_evals}.png")
        # 打印结果
        if n_evals % 1 == 0:
            # hybzc = bo_.pred_func(bo_.x_best.reshape(1, -1))[3].item()
            # # hybzc = bo_.pred_model.posterior(bo_.x_best.reshape(1, -1)).variance.item()
            # ycbzc = bo_.pred_func.gp_model.likelihood(bo_.pred_func.gp_model(bo_.x_best.reshape(1, -1))).variance.item()
            # noise = 1#bo_.pred_model.likelihood.noise.item()
            # a = hybzc + noise - ycbzc
            # b = acq_val * pred_func.fx_train_scaler.std + pred_func.fx_train_scaler.mean
            print(
                f"process_id: {process_id}, "
                f"iter: {n_evals}, "
                # f"acq_val: {b.item()}, "
                # f"acq_val_scaled: {acq_val.item()} "
                f"output_scale: {bo_.pred_func.gp_model.covar_module.outputscale:.{float_precision}f}, "
                # f"std: {hybzc / noise:.{float_precision}f}, "
                f"Best value found: {bo_.fx_best.item():.{float_precision}f}, "
                # f"Best point found: {bo_.x_best.numpy()}"
            )
    # 绘制图像
    plot_results(bo_, 10, run_mode=figure_run_mode,
                 file_path=data_figure_dir/f"figure/{func_name}/{acq_func_name}/process_{process_id}/image_{n_evals}.png")
    # 打印结果
    print(f"Opt complete. Best value found: {bo_.fx_best.item():.{float_precision}f}, Best point found: {bo_.x_best.numpy()}\n")

    result = {
        'process_id': process_id,
        'params_dict': params_dict,
        'bo': bo_,
    }

    # # 回退到某个迭代的状态，查看该状态下的图像
    # bo_.load_gp_snapshot(12)
    # plot_results(bo_, 10, run_mode=figure_run_mode,
    #              file_path=data_figure_dir / f"figure/{func_name}/{acq_func_name}/process_{process_id}/image_{bo_.curr_evals}_111.png")

    return result


if __name__ == "__main__":
    from htslibs.utils.multi_process import process_mode
    from htslibs.objective_functions.airfoil_obj import report_all_stats, AF1_STATS

    # 设置打印精度为6位小数
    np.set_printoptions(precision=float_precision, suppress=True)
    # 其他参数
    num_processes = 1

    for func_name in [
        # 'Sin',
        # 'Ackley',
        # 'Levy',
        # 'Schwefel',
        # 'Rastrigin',
        # 'Rosenbrock',
        # 'Griewank'
        # 'SVM_BreastCancer',
        # 'MLP_Digits',
        # 'XGBoost_Diabetes',
        'Airfoil_AF1',
    ]:
        for acq_func_name in ['UCB']:  # 'UCB', 'EI', 'PI'
            # 配置参数
            params_dict = {
                'func_name': func_name,
                'acq_func_name': acq_func_name,
            }
            # 运行优化任务
            result_list = process_mode(
                task_iter,
                params_dict_list=[deepcopy(params_dict) for _ in range(num_processes)],
                multiy_process_params=range(0, num_processes),
                batch_size=48,
                is_multi_processing=False  # True, False
            )

            # 计算最优值的平均值和方差
            # print("mean: ",np.mean(fx_best_list))
            # print("std: ",np.std(fx_best_list))
            # 保存结果
            # save_data_to_pkl(
            #     result_list,
            #     f"data/BoTest_{func_name}_{acq_func_name}.pkl"
            # )
            # 发送邮件
            # add_mail_log(f'Opt. complete: {func_name}, {acq_func_name}')
            # send_email('程序运行完成', get_mail_log())





#     test_x1 = np.linspace(lb[0], ub[0], 100)
    #     test_x2 = np.linspace(lb[1], ub[1], 100)
    #     test_X1, test_X2 = np.meshgrid(test_x1, test_x2)
    #     test_x = torch.tensor(np.c_[test_X1.ravel(), test_X2.ravel()])
    #     test_fx_np = obj_func(test_x).numpy()
    #     test_fx_np = test_fx_np.reshape(test_X1.shape)
    #     test_mu, test_std, _, _ = pred_func(test_x)
    #     test_mu_np = -1 * test_mu.detach().numpy().reshape(test_X1.shape)
    #     # test_std_np = test_std.detach().numpy()
    #     # test_lcb_np = test_mu_np - 1.96 * test_std_np
    #     # test_ucb_np = test_mu_np + 1.96 * test_std_np
    #
    #     # 创建3D图像
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_surface(test_X1, test_X2, test_fx_np, cmap='viridis', alpha=0.6, label='Objective Function')
    #
    #     # # 绘制预测均值
    #     # ax.plot_surface(test_X1, test_X2, test_mu_np, cmap='coolwarm', alpha=0.6, label='Predicted Function')
    #
    #     # # 绘制置信区间
    #     # ax.plot_surface(X, Y, test_lcb_np, color='lightblue', alpha=0.3, label='LCB')
    #     # ax.plot_surface(X, Y, test_ucb_np, color='lightblue', alpha=0.3, label='UCB')
    #
    #     # # 绘制贝叶斯优化的采样点
    #     # ax.scatter(bo_.x[:, 0].numpy(), bo_.x[:, 1].numpy(), bo_.fx.numpy(), color='r', s=50, label='Samples')
    #     # ax.scatter(bo_.x_best[0].item(), bo_.x_best[1].item(), bo_.fx_best.item(), color='k', s=100, label='Best Point')
    #
    #     ax.set_xlabel('X1')
    #     ax.set_ylabel('X2')
    #     ax.set_zlabel('Objective Function')
    #     # ax.legend()
    #     plt.show()


    
