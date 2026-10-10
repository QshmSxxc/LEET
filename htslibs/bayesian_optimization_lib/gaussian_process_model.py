import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import Mean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from gpytorch.priors import GammaPrior

from htslibs.bayesian_optimization_lib.utils import StandardScaler_Torch, ToUnitCubeScaler_Torch
from htslibs.utils.load_data_from_json import load_config_from_json

# 获取参数配置
params_config = load_config_from_json()
device, dtype = torch.device(params_config["device"]), eval(params_config["dtype"])


class CustomMean(Mean):
    def __init__(self, mean_func=None):
        super().__init__()
        self.mean_func = mean_func

    def forward(self, x):
        # mean = torch.zeros(x.shape[0])
        # mean = torch.sin(x).flatten()  # 这里假设均值是 sin(x)
        mean = self.mean_func(x)
        return mean  # 这里假设均值是 sin(x)


class CustomGP(SingleTaskGP):
    def __init__(self, train_x, train_y, mean_func=None):
        super().__init__(train_x, train_y)

        # 设定均值函数
        if mean_func is None:
            self.mean_module = ZeroMean()
        else:
            self.mean_module = CustomMean(mean_func)  # 设定均值为常数
        # 设定协方差函数
        self. covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[-1],
                batch_shape=self._aug_batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=self._aug_batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )


class gp:
    def __init__(
            self,
            bounds,
            x_train, fx_train,
            x_train_scaler=None, fx_train_scaler=None,
            mean_func=None,
            gp_model_state_dict=None,
            fit_model_flag=True,
    ):
        self.bounds = bounds
        self.x_train = x_train
        self.fx_train = fx_train
        self.x_train_scaler = x_train_scaler
        self.fx_train_scaler = fx_train_scaler
        self.mean_func = mean_func

        # 训练数据预处理
        if self.x_train_scaler is None and self.fx_train_scaler is None:
            self.x_train_scaled, self.fx_train_scaled, self.x_train_scaler, self.fx_train_scaler = (
                self._preprocess_data(self.bounds, self.x_train, self.fx_train))
        else:
            self.x_train_scaled = self.x_train_scaler.transform(self.x_train)
            self.fx_train_scaled = self.fx_train_scaler.transform(self.fx_train)

        # 训练模型
        self.gp_model = self.fit_model(self.x_train_scaled, self.fx_train_scaled, self.mean_func)

        # 构造 / 训练 / 恢复模型
        if gp_model_state_dict is None:
            if fit_model_flag:
                self.gp_model = self.fit_model(self.x_train_scaled, self.fx_train_scaled, self.mean_func)
            else:
                self.gp_model = CustomGP(self.x_train_scaled, self.fx_train_scaled, self.mean_func).to(device)
                self.gp_model.eval()
                self.gp_model.likelihood.eval()
        else:
            self.gp_model = CustomGP(self.x_train_scaled, self.fx_train_scaled, self.mean_func).to(device)
            self.gp_model.load_state_dict(gp_model_state_dict)
            self.gp_model.eval()
            self.gp_model.likelihood.eval()

    def _preprocess_data(self, bounds, x_train, fx_train):
        # 输入数据归一化
        # self.train_x_scaler = MinMaxScaler_Torch(feature_range=(0, 1))
        x_train_scaler = ToUnitCubeScaler_Torch(bounds[0, :], bounds[1, :])
        x_train_scaler.fit(x_train)
        x_train_scaled = x_train_scaler.transform(x_train)
        # 输出数据标准化
        fx_train_scaler = StandardScaler_Torch()
        fx_train_scaler.fit(fx_train)
        fx_train_scaled = fx_train_scaler.transform(fx_train)

        return x_train_scaled, fx_train_scaled, x_train_scaler, fx_train_scaler

    def fit_model(self, x_train, fx_train, mean_func=None):
        # 定义高斯过程模型
        # gp_model = SingleTaskGP(x_train, fx_train)
        gp_model = CustomGP(x_train, fx_train, mean_func).to(device)  # 模型移到
        likelihood = gp_model.likelihood.to(device)  # 似然函数也移到
        mll = ExactMarginalLogLikelihood(likelihood, gp_model)
        # 训练模型
        fit_gpytorch_mll(mll)  # 训练（自动利用 GPU）
        # 评估模式
        gp_model.eval()
        likelihood.eval()

        return gp_model

    def pred_func(self, x_test):
        assert torch.is_tensor(x_test)
        # 预测标准化数据
        test_x_scaled = self.x_train_scaler.transform(x_test)
        posterior_scaled = self.gp_model.posterior(test_x_scaled)
        mean_scaled = posterior_scaled.mean.reshape(-1)  # 后验均值
        std_scaled = posterior_scaled.variance.clamp_min(1e-12).sqrt().reshape(-1)  # 后验标准差
        # 转换为原始数据
        mean = mean_scaled * self.fx_train_scaler.std + self.fx_train_scaler.mean
        std = std_scaled * self.fx_train_scaler.std

        return mean, std, mean_scaled, std_scaled

    def __call__(self, x):
        return self.pred_func(x)


if __name__ == "__main__":
    from htslibs.utils.sample_methods import latin_hypercube_sample_tensor
    from htslibs.objective_functions.obj_functions import get_function_details

    torch.manual_seed(42)
    # 设置打印精度为6位小数
    np.set_printoptions(precision=7, suppress=True)
    # 获取目标函数信息
    func_name = 'Ackley'  # 'Sin', 'Ackley', 'Levy', 'Schwefel', 'Rastrigin'
    dim = 1
    obj_func= get_function_details(func_name, dim)
    bounds = obj_func.bounds
    mff = obj_func.mf_funcs
    obj_func = mff.fidel_func[mff.num_fidels - 1]

    # 生成训练数据
    lb = bounds[0, :]
    ub = bounds[1, :]
    # train_x = torch.tensor([1.0000, 1.0000, 1.0000, 1.0000]).reshape(1, -1)
    train_x = latin_hypercube_sample_tensor(260, lb, ub, dtype=dtype, device=device)
    train_y = obj_func(train_x)  # 输出是 (n, 1)

    # 训练高斯过程模型
    mean_func = lambda x: torch.sin(x).flatten()  # 这里假设均值是 sin(x)
    # gp_model, prediction_func, _, _ = train_gp(train_x, train_y, bounds, mean_func)
    prediction_func = gp(bounds, train_x, train_y)
    covar_module = prediction_func.gp_model.covar_module
    # 获取RBF核的超参数
    length_scale = covar_module.base_kernel.lengthscale
    output_scale = covar_module.outputscale
    # print(f"func_name: {func_name}, Length Scale: {length_scale.cpu().detach().numpy()}, Output Scale: {output_scale.cpu().detach().numpy()}")

    # 绘制结果
    if dim == 1:
        # 生成测试数据
        test_x = torch.linspace(lb[0], ub[0], 5000).unsqueeze(-1).to(device)  # 测试点
        test_y = obj_func(test_x)  # 真实函数值
        # 预测测试数据
        test_y_mean, test_y_std, _, _ = prediction_func(test_x)
        # 将数据移到CPU上
        train_x, train_y = train_x.cpu().detach(), train_y.cpu().detach()
        test_x, test_y = test_x.cpu().detach(), test_y.cpu().detach()
        test_y_mean, test_y_std = test_y_mean.cpu().detach(), test_y_std.cpu().detach()
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(test_x.numpy(), test_y.numpy(), 'b-', label=f"True Function {func_name}", linewidth=2)
        plt.plot(test_x.numpy(), test_y_mean.numpy(), 'r-', label="GP Mean Prediction", linewidth=2, zorder=6)
        plt.scatter(train_x.numpy(), train_y.numpy(), color="black", s=40, label="Training Points", zorder=5)
        plt.fill_between(
            test_x.numpy().ravel(),
            (test_y_mean - 1.96 * test_y_std).numpy().ravel(),
            (test_y_mean + 1.96 * test_y_std).numpy().ravel(),
            color="lightblue",
            alpha=0.5,
            label="95% Confidence Interval",
            # zorder=6,
        )
        plt.title("Gaussian Process Regression on Ackley Function", fontsize=16)
        plt.xlabel("x", fontsize=14)
        plt.ylabel("f(x)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
