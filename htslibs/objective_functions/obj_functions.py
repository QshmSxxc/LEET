import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import math
from pathlib import Path

from htslibs.objective_functions.airfoil_obj import Airfoil_AF1, Airfoil_AF2, Airfoil_AF3
from htslibs.objective_functions.obj_func_interface import obj_func, mf_function, noise, device, dtype

parent_dir = Path(__file__).resolve().parents[1]


def get_function_details(function_name, dims=1):
    # if   function_name == 'Hartmann3D':
    #     fun = Hartmann3D()
    if function_name == 'Hartmann3D_SM':
        fun = Hartmann3D_SM()
    # elif function_name == 'Hartmann6D':
    #     fun = Hartmann6D()
    elif function_name == 'Hartmann6D_SM':
        fun = Hartmann6D_SM()
    elif function_name == 'Sin':
        fun = Sin()
    elif function_name == 'Borehole':
        fun = Borehole()
    elif function_name == 'CurrinExp':
        fun = CurrinExp()
    elif function_name == 'Park1':
        fun = Park1()
    elif function_name == 'Ackley':
        fun = Ackley(num_dims=dims)
    elif function_name == 'Levy':
        fun = Levy(num_dims=dims)
    elif function_name == 'Schwefel':
        fun = Schwefel(num_dims=dims)
    elif function_name == 'Rastrigin':
        fun = Rastrigin(num_dims=dims)
    elif function_name == 'Rosenbrock':
        fun = Rosenbrock()
    elif function_name == 'Griewank':
        fun = Griewank(num_dims=dims)
    elif function_name == 'Michalewicz':
        fun = Michalewicz(num_dims=dims)
    elif function_name == 'Hartmann3D':
        fun = Hartmann(num_dims=3)
    elif function_name == 'Hartmann6D':
        fun = Hartmann(num_dims=6)
    elif function_name == 'Branin':
        fun = Branin()
    elif function_name == 'Dixon_Price':
        fun = Dixon_Price(num_dims=dims)
    elif function_name == 'SVM_BreastCancer':
        fun = SVM_BreastCancer()
    elif function_name == 'MLP_Digits':
        fun = MLP_Digits()
    elif function_name == 'XGBoost_Diabetes':
        fun = XGBoost_Diabetes()
    elif function_name == 'RandomForest_Diabetes':
        fun = RandomForest_Regression("diabetes")
    elif function_name == 'RandomForest_California_Housing':
        fun = RandomForest_Regression("california_housing")
    elif function_name == 'LightGBM_AdultIncome':
        fun = LightGBM_AdultIncome()
    elif function_name == 'Airfoil_AF1':
        fun = Airfoil_AF1()
    elif function_name == 'Airfoil_AF2':
        fun = Airfoil_AF2()
    elif function_name == 'Airfoil_AF3':
        fun = Airfoil_AF3()
    elif function_name == 'SVM':
        fun = SVM()
    elif function_name == 'SVM_magic_gamma_dataset':
        fun = SVM_magic_gamma_dataset()
        bounds = torch.stack((fun.lb, fun.ub), dim=1)
    elif function_name == 'Swimmer':
        fun = Swimmer()
        bounds = torch.stack((fun.lb, fun.ub), dim=1)
    elif function_name == 'Lunarlanding':
        fun = Lunarlanding()  # dim = 12
        # bounds = torch.stack((fun.lb, fun.ub), dim=1)
    elif function_name == 'Hopper':
        fun = Hopper()  # dim = 33
    elif function_name == 'Walker':
        fun = Walker()
    elif function_name == 'HalfCheetah':
        fun = HalfCheetah()  # dim = 102
    elif function_name == 'XGBoost':
        fun = XGBoost()
        bounds = torch.stack((fun.lb, fun.ub), dim=1)
    elif function_name == 'Custom1D':
        fun = Custom1D()
    elif function_name == 'G04':
        fun = G04()
    elif function_name == 'G09':
        fun = G09()
    elif function_name == 'Q1':
        fun = Q1()
    else:
        raise ValueError(f"Unknown function: {function_name}")

    return fun

# try:
#     import matlab
#     import matlab.engine
# except ImportError:
#     matlab = None

def Q1():
    input_dim_num = 3
    func_name = 'Q1'
    # 开启matlab引擎
    try:
        import matlab
        import matlab.engine
    except ImportError as exc:
        raise RuntimeError("Q1 requires MATLAB Engine for Python to be installed.") from exc

    matlab_project_dir = os.environ.get("Q1_MATLAB_PATH")
    if not matlab_project_dir:
        raise RuntimeError("Q1 requires Q1_MATLAB_PATH to point to the MATLAB project directory.")
    matlab_project_dir = matlab_project_dir.replace("'", "''")

    eng = matlab.engine.start_matlab()
    eng.eval(f"addpath(genpath('{matlab_project_dir}'))", nargout=0)
    # 定义目标函数
    def function(x):
        x_dim_info = x.shape
        assert x_dim_info[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"  # 确保x的维度是符合的
        result = torch.empty_like(x)
        result = []
        x_np = np.asarray(x)
        for i in range(x_dim_info[-2]):
            result.append(eng.demo_hrx_3_var(matlab.double(x_np[i,:].reshape(1, -1).tolist())))
        result = torch.tensor(np.asarray(result).reshape(-1,1), dtype=dtype, device=device)
        # result.reshape()
        # result = -20 * torch.exp(-0.2 * torch.sqrt((x**2).mean(dim=-1).reshape(-1, 1))) \
        #          - torch.exp(torch.cos(2 * np.pi * x).mean(dim=-1).reshape(-1, 1)) \
        #          + torch.e + 20
        return -1*result + torch.randn_like(result) * noise
    # 定义约束函数
    def function_C1(x):
        return

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs = [function_C1],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((torch.tensor([10, 0, 45], dtype=dtype, device=device), torch.tensor([120, 90, 179], dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=None,
        max_value=None,
    )

def Custom1D():
    input_dim_num = 1  # 1D函数
    func_name = 'Custom1D'
    # 定义目标函数 (基础函数)
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        x = x.squeeze(-1)
        # result = -torch.sin(x) - torch.exp(x/100) + 10.7 + 0.02 * (x - 3)**2.5
        result = -torch.sin(x) - torch.exp(x / 100) + 10.3 + 0.03 * (x - 3) ** 2
        return result.unsqueeze(-1) + torch.randn_like(result.unsqueeze(-1)) * noise
    # 定义保真度函数 (代理系统输出)
    def function_F1(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        x = x.squeeze(-1)
        result = -torch.sin(x) - torch.exp(x/100) + 10.3 + 0.03 * (x - 2)**2
        return result.unsqueeze(-1) + torch.randn_like(result.unsqueeze(-1)) * noise
    def function_F0(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        x = x.squeeze(-1)  # 从 (n,1) 转换为 (n,)
        result = -torch.sin(x) - torch.exp(x/100) + 10
        return result.unsqueeze(-1) + torch.randn_like(result.unsqueeze(-1)) * noise

    # 返回函数对象 (使用与Hartmann6D相同的obj_func结构)
    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function_F0, function_F1, function]),  # 仅两个保真度层级
        bounds=torch.tensor([[0.0], [10.0]], dtype=dtype, device=device),  # x ∈ (0,10)
        min_point=None,  # 可补充真实最小值点
        min_value=None,  # 可补充真实最小值
        max_point=torch.tensor([10.0], dtype=dtype, device=device),  # 可补充真实最大值点
        max_value=torch.tensor([11.2089], dtype=dtype, device=device)   # 可补充真实最大值
    )

def Hartmann3D_SM():
    input_dim_num = 3
    func_name = 'Hartmann3D_SM'

    def function(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        # 定义参数
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=dtype, device=device)
        A = torch.tensor([
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0]
        ], dtype=dtype, device=device)
        P = torch.tensor([
            [3689.0, 1170.0, 2673.0],
            [4699.0, 4387.0, 7470.0],
            [1091.0, 8732.0, 5547.0],
            [381.0, 5743.0, 8828.0]
        ], dtype=dtype, device=device) * 1e-4
        # 计算目标函数值
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)  # 初始化result为n*1的形状
        for i in range(4):
            term = torch.sum(A[i] * (x - P[i])**2, dim=-1).reshape(-1, 1)  # 计算每一项的平方和
            result -= alpha[i] * torch.exp(-term)  # 计算目标函数的值
        # 添加噪声, 返回结果
        return -1*result + torch.randn_like(result) * noise
    # 定义保真度函数
    num_fidels = 3
    fidel_func = []
    for i in range(0, num_fidels - 1):
        model = torch.load(parent_dir / 'multi_fidelity_optimization' / 'mf_model' / f'{func_name}_fidel_{i}_model_full.pth')
        model.eval()
        def func(x, model=model):
            a = model(x).detach()
            b = torch.tensor(-1.0, dtype=dtype, device=device)
            return model(x).detach()*torch.tensor(-1.0, dtype=dtype, device=device)
        fidel_func.append(func)
    fidel_func.append(function)

    return obj_func(
        func=function,
        func_name=func_name,
        mf_funcs=mf_function(fidel_func=fidel_func),
        bounds=torch.stack((torch.zeros(input_dim_num, dtype=dtype, device=device), torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.tensor([0.114614, 0.555649, 0.852547], dtype=dtype, device=device),
        max_value=torch.tensor([3.86278], dtype=dtype, device=device)
    )

def Hartmann3D():
    input_dim_num = 3
    func_name = 'Hartmann3D'
    # 定义参数
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=dtype, device=device)
    A = torch.tensor([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ], dtype=dtype, device=device)
    P = torch.tensor([
        [3689.0, 1170.0, 2673.0],
        [4699.0, 4387.0, 7470.0],
        [1091.0, 8732.0, 5547.0],
        [381.0, 5743.0, 8828.0]
    ], dtype=dtype, device=device) * 1e-4
    delta = 0.1 * torch.tensor([0.01, -0.01, -0.1, 0.1], dtype=dtype, device=device)
    # 定义目标函数
    def function(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        # 计算目标函数值
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)  # 初始化result为n*1的形状
        for i in range(4):
            term = torch.sum(A[i] * (x - P[i])**2, dim=-1).reshape(-1, 1)  # 计算每一项的平方和
            result -= alpha[i] * torch.exp(-term)  # 计算目标函数的值
        # 添加噪声, 返回结果
        return -1 * result + torch.randn_like(result) * noise
    # 定义保真度函数
    def function_F0(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        alpha_f = alpha + delta * 2
        # 计算目标函数值
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)  # 初始化result为n*1的形状
        for i in range(4):
            term = torch.sum(A[i] * (x - P[i])**2, dim=-1).reshape(-1, 1)  # 计算每一项的平方和
            result -= alpha_f[i] * torch.exp(-term)  # 计算目标函数的值
        # 添加噪声, 返回结果
        return -1*result + torch.randn_like(result)*noise

    def function_F1(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        alpha_f = alpha + delta * 1
        # 计算目标函数值
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)  # 初始化result为n*1的形状
        for i in range(4):
            term = torch.sum(A[i] * (x - P[i]) ** 2, dim=-1).reshape(-1, 1)  # 计算每一项的平方和
            result -= alpha_f[i] * torch.exp(-term)  # 计算目标函数的值
        # 添加噪声, 返回结果
        return -1 * result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        mf_funcs=mf_function(fidel_func=[function_F0, function_F1, function]),
        bounds=torch.stack((torch.zeros(input_dim_num, dtype=dtype, device=device), torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.tensor([0.114614, 0.555649, 0.852547], dtype=dtype, device=device),
        max_value=torch.tensor([3.86278], dtype=dtype, device=device)
    )

def Hartmann6D_SM():  # 使用surrogate model训练可信度函数
    input_dim_num = 6
    func_name = 'Hartmann6D_SM'
    # 定义目标函数
    def function(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        # 定义参数
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=dtype, device=device)
        A = torch.tensor([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ], dtype=dtype, device=device)
        P = torch.tensor([
            [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
            [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
            [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
            [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]
        ], dtype=dtype, device=device) * 1e-4
        # 计算目标函数值
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)  # 初始化result为n*1的形状
        for i in range(4):
            term = torch.sum(A[i] * (x - P[i])**2, dim=-1).reshape(-1, 1)  # 计算每一项的平方和
            result -= alpha[i] * torch.exp(-term)  # 计算目标函数的值
        # 添加噪声, 返回结果
        return -1*result + torch.randn_like(result) * noise
    # 定义保真度函数
    num_fidels = 4
    fidel_func = []
    for i in range(0, num_fidels - 1):
        model = torch.load(parent_dir / 'multi_fidelity_optimization' / 'mf_model' / f'{func_name}_fidel_{i}_model_full.pth')
        model.eval()
        def func(x, model=model):
            a = model(x).detach()
            b = torch.tensor(-1.0, dtype=dtype, device=device)
            return model(x).detach()*torch.tensor(-1.0, dtype=dtype, device=device)
        fidel_func.append(func)
    fidel_func.append(function)

    return obj_func(
        func=function,
        func_name=func_name,
        mf_funcs=mf_function(fidel_func=fidel_func),
        bounds=torch.stack((torch.zeros(input_dim_num, dtype=dtype, device=device), torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.tensor([0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573], dtype=dtype, device=device),
        max_value=torch.tensor([3.32237], dtype=dtype, device=device),
    )

def Hartmann6D():
    input_dim_num = 6
    func_name = 'Hartmann6D'
    # 定义参数
    delta = 0.1 * torch.tensor([0.01, -0.01, -0.1, 0.1], dtype=dtype, device=device)
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=dtype, device=device)
    A = torch.tensor([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ], dtype=dtype, device=device)
    P = torch.tensor([
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]
    ], dtype=dtype, device=device) * 1e-4
    # 定义目标函数
    def function(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        # 计算目标函数值
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)  # 初始化result为n*1的形状
        for i in range(4):
            term = torch.sum(A[i] * (x - P[i])**2, dim=-1).reshape(-1, 1)  # 计算每一项的平方和
            result -= alpha[i] * torch.exp(-term)  # 计算目标函数的值
        # 添加噪声, 返回结果
        return -1*result + torch.randn_like(result) * noise
    # 定义保真度函数
    def function_F0(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        alpha_f = alpha + delta*3
        # 计算目标函数值
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)  # 初始化result为n*1的形状
        for i in range(4):
            term = torch.sum(A[i] * (x - P[i])**2, dim=-1).reshape(-1, 1)  # 计算每一项的平方和
            result -= alpha_f[i] * torch.exp(-term)  # 计算目标函数的值
        # 添加噪声, 返回结果
        return -1*result + torch.randn_like(result) * noise
    def function_F1(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        alpha_f = alpha + delta*2
        # 计算目标函数值
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)  # 初始化result为n*1的形状
        for i in range(4):
            term = torch.sum(A[i] * (x - P[i])**2, dim=-1).reshape(-1, 1)  # 计算每一项的平方和
            result -= alpha_f[i] * torch.exp(-term)  # 计算目标函数的值
        # 添加噪声, 返回结果
        return -1*result + torch.randn_like(result) * noise
    def function_F2(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        alpha_f = alpha + delta*1
        # 计算目标函数值
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)  # 初始化result为n*1的形状
        for i in range(4):
            term = torch.sum(A[i] * (x - P[i])**2, dim=-1).reshape(-1, 1)  # 计算每一项的平方和
            result -= alpha_f[i] * torch.exp(-term)  # 计算目标函数的值
        # 添加噪声, 返回结果
        return -1*result + torch.randn_like(result) * noise
    
    return obj_func(
        func=function,
        func_name=func_name,
        mf_funcs=mf_function(fidel_func=[function_F0, function_F1, function_F2, function]),
        bounds=torch.stack((torch.zeros(input_dim_num, dtype=dtype, device=device), torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.tensor([0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573], dtype=dtype, device=device),
        max_value=torch.tensor([3.32237], dtype=dtype, device=device),
    )

def Borehole():
    input_dim_num = 8
    func_name = 'Borehole'

    def function(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"

        rw = x[:, 0]    # 半径rw
        r  = x[:, 1]    # 半径r
        Tu = x[:, 2]    # Upper aquifer transmissivity
        Hu = x[:, 3]    # Upper aquifer potentiometric head
        Ti = x[:, 4]    # Lower aquifer transmissivity
        Hi = x[:, 5]    # Lower aquifer potentiometric head
        L  = x[:, 6]    # Borehole length
        Kw = x[:, 7]    # Hydraulic conductivity

        # 计算Borehole目标函数的值
        frac1 = 2 * torch.pi * Tu * (Hu-Hi)
        frac2a = 2*L*Tu / (torch.log(r/rw)*(rw**2)*Kw)
        frac2b = Tu / Ti
        frac2 = torch.log(r/rw) * (1+frac2a+frac2b)

        result = frac1 / frac2
        result = result.reshape(-1, 1)

        return result + torch.randn_like(result) * noise  # 添加噪声
    
    def function_F0(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"

        rw = x[:, 0]    # 半径rw
        r  = x[:, 1]    # 半径r
        Tu = x[:, 2]    # Upper aquifer transmissivity
        Hu = x[:, 3]    # Upper aquifer potentiometric head
        Ti = x[:, 4]    # Lower aquifer transmissivity
        Hi = x[:, 5]    # Lower aquifer potentiometric head
        L  = x[:, 6]    # Borehole length
        Kw = x[:, 7]    # Hydraulic conductivity

        # 计算Borehole目标函数的值
        frac1 = 5 * Tu * (Hu-Hi)
        frac2a = 2*L*Tu / (torch.log(r/rw)*(rw**2)*Kw)
        frac2b = Tu / Ti
        frac2 = torch.log(r/rw) * (1.5+frac2a+frac2b)

        result = frac1 / frac2
        result = result.reshape(-1, 1)

        return result + torch.randn_like(result) * noise  # 添加噪声

    return obj_func(
        func=function,
        func_name=func_name,
        mf_funcs=mf_function(fidel_func=[function_F0, function]),
        bounds=torch.stack((torch.tensor([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855], dtype=dtype, device=device), torch.tensor([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045], dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=None,
        max_value=torch.tensor([309.5756], dtype=dtype, device=device),
    )

def CurrinExp():
    input_dim_num = 2
    func_name = 'CurrinExp'

    def function(x):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"

        x1 = x[:, 0]
        x2 = x[:, 0]
        
        # 计算目标函数值
        fact1 = 1 - torch.exp(-1/(2*x2))
        fact2 = 2300*(x1**3) + 1900*(x1**2) + 2092*x1 + 60
        fact3 = 100*(x1**3) + 500*(x1**2) + 4*x1 + 20

        result = fact1 * fact2/fact3
        result = result.reshape(-1, 1)
        return result + torch.randn_like(result) * noise
    
    def function_F0(x, func = function):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"

        x1 = x[:, 0]
        x2 = x[:, 1]
        
        # 计算目标函数值
        maxarg = torch.max(torch.tensor([0.0], dtype=dtype, device=device), x2-1/20)

        yh1 = func(torch.stack([x1+1/20, x2+1/20], dim=1))
        yh2 = func(torch.stack([x1+1/20, maxarg], dim=1))
        yh3 = func(torch.stack([x1-1/20, x2+1/20], dim=1))
        yh4 = func(torch.stack([x1-1/20, maxarg], dim=1))

        result = (yh1 + yh2 + yh3 + yh4) / 4
        result = result.reshape(-1, 1)
        return result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        mf_funcs=mf_function(fidel_func=[function_F0, function]),
        bounds=torch.stack((torch.zeros(input_dim_num, dtype=dtype, device=device), torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=None,
        max_value=torch.tensor([12.8127], dtype=dtype, device=device),
    )

def Park1():
    input_dim_num = 4
    func_name = 'Park1'

    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        # 避免除以0的安全处理
        eps = 1e-12
        # 结果初始化: n x 1
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)
        # f2(x) = x1/2 * [ sqrt(1 + (x2 + x3^2)* x4^2 / x1^2 ) + (x1 + 3x4)*exp(1 + sin(x3)) ]
        x1 = x[:, 0].unsqueeze(-1)
        x2 = x[:, 1].unsqueeze(-1)
        x3 = x[:, 2].unsqueeze(-1)
        x4 = x[:, 3].unsqueeze(-1)
        # 1) x1/2
        term1 = x1 / 2.0
        # 2) sqrt(...)
        term2 = torch.sqrt(
            1.0 + (x2 + x3 ** 2) * (x4 ** 2) / (x1 ** 2 + eps)
        )
        # 3) (x1 + 3*x4)*exp(1 + sin(x3))
        term3 = (x1 + 3.0 * x4) * torch.exp(1.0 + torch.sin(x3))
        # 合并
        f2_val = term1 * (term2 + term3)
        # 赋值给 result
        result[:, 0] = f2_val.squeeze(-1)
        # 添加噪声
        return result + torch.randn_like(result) * noise

    def function_F0(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        # 结果初始化: n x 1
        result = torch.zeros((x.shape[0], 1), dtype=dtype, device=device)
        # 先计算高保真 f2(x) 的数值
        f2_val = function(x)  # 直接调用高保真函数 (也可重写以减小计算量)
        # f1(x) = 1 + (sin(x1)/10)*f2(x) - 2*x1 + x2^2 + x3^2 + 0.5
        x1 = x[:, 0].unsqueeze(-1)
        x2 = x[:, 1].unsqueeze(-1)
        x3 = x[:, 2].unsqueeze(-1)
        val = 1.0 + (torch.sin(x1) / 10.0) * f2_val - 2.0 * x1 + x2 ** 2 + x3 ** 2 + 0.5
        result[:, 0] = val.squeeze(-1)
        return result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        mf_funcs=mf_function(fidel_func=[function_F0, function]),
        bounds=torch.stack((
            torch.zeros(input_dim_num, dtype=dtype, device=device),
            torch.ones(input_dim_num, dtype=dtype, device=device)
        ), dim=0),
        min_point=None,  # 如果有已知最小值可填写
        min_value=None,  # 若是最大化问题，这里可留空
        max_point=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device),
        max_value=torch.tensor([13.4776], dtype=dtype, device=device),
    )

def Ackley(num_dims=1):
    input_dim_num = num_dims
    func_name = 'Ackley'
    # 定义目标函数
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"  # 确保x的维度是符合的
        result = -20 * torch.exp(-0.2 * torch.sqrt((x**2).mean(dim=-1).reshape(-1, 1))) \
                 - torch.exp(torch.cos(2 * np.pi * x).mean(dim=-1).reshape(-1, 1)) \
                 + torch.e + 20
        return -1*result + torch.randn_like(result) * noise
    # 定义约束函数
    def function_C1(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        if len(x.shape) == 3:
            result = x[:, :, 1] + 2 * x[:, :, 0] - 2
        elif len(x.shape) == 2:
            result = x[:, 1] + 2 * x[:, 0] - 2
        return -1*result.reshape(-1)

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs = [function_C1],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((
            -5*torch.ones(input_dim_num, dtype=dtype, device=device),
            10*torch.ones(input_dim_num, dtype=dtype, device=device)
        ), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.full((input_dim_num,), 0.0, dtype=dtype, device=device),
        max_value=torch.full((1,), 0.0, dtype=dtype, device=device),
    )

def Levy(num_dims=1):
    input_dim_num = num_dims
    func_name = 'Levy'
    # 定义目标函数
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"  # 确保x的维度是符合的
        # x 是一个 n*d 维张量
        n, d = x.size()  # n 为样本数，d 为每个样本的维度
        w = 1 + (x - 1) / 4  # 转换为 w_i, 对 x 的每一维执行这个操作
        # 初始化 term1 和 term2
        term1 = torch.sin(torch.pi * w[:, 0]) ** 2  # 对每一行的第一个元素进行计算
        term2 = torch.zeros(n, dtype=dtype, device=device)  # 初始化 term2 为 0, 适应每个样本
        # 逐个样本计算其对应的 term2
        for i in range(d - 1):
            term2 += (w[:, i] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w[:, i] + 1) ** 2)
        term3 = (w[:, -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[:, -1]) ** 2)  # 对每一行的最后一个元素进行计算
        # 返回每个样本的 Levy 函数值
        result = term1 + term2 + term3
        result = result.reshape(-1, 1)
        return -1 * result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((-10 * torch.ones(input_dim_num, dtype=dtype, device=device),
                            10 * torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.full((input_dim_num,), 1.0, dtype=dtype, device=device),
        max_value=torch.full((1,), 0.0, dtype=dtype, device=device),
    )

def Schwefel(num_dims=1):
    input_dim_num = num_dims
    func_name = 'Schwefel'

    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"  # 确保x的维度是符合的
        # x 是一个 n*d 维张量
        d = x.shape[1]  # 获取输入张量的维度
        result = 418.9829 * d - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=1).reshape(-1, 1)
        return -1 * result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((-500 * torch.ones(input_dim_num, dtype=dtype, device=device),
                            500 * torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.full((input_dim_num,), 420.9687, dtype=dtype, device=device),
        max_value=torch.full((1,), 0.0, dtype=dtype, device=device),
    )

def Rastrigin(num_dims=1):
    input_dim_num = num_dims
    func_name = 'Rastrigin'

    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        A = 10
        result = A * input_dim_num + (x**2 - A * torch.cos(2 * np.pi * x)).sum(dim=-1, keepdim=True)
        return -1 * result + torch.randn_like(result) * noise  # 最大化问题

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((-5.12 * torch.ones(input_dim_num, dtype=dtype, device=device),
                            5.12 * torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.full((input_dim_num,), 0.0, dtype=dtype, device=device),  # Rastrigin 在原点取得最小值，因此最大值未知
        max_value=torch.full((1,), 0.0, dtype=dtype, device=device),
    )

def Rosenbrock():
    input_dim_num = 2
    func_name = 'Rosenbrock'

    # 定义目标函数
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        result = torch.sum(
            100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (x[..., :-1] - 1) ** 2,
            dim=-1,
            keepdim=True
        )
        return -1 * result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((
            -5 * torch.ones(input_dim_num, dtype=dtype, device=device),
            10 * torch.ones(input_dim_num, dtype=dtype, device=device)
        ), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.ones(input_dim_num, dtype=dtype, device=device),
        max_value=torch.full((1,), 0.0, dtype=dtype, device=device),
    )

def Griewank(num_dims=1):
    input_dim_num = num_dims
    func_name = 'Griewank'

    # 定义目标函数
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"

        sum_term = torch.sum(x ** 2, dim=-1, keepdim=True) / 4000

        i = torch.arange(1, input_dim_num + 1, dtype=dtype, device=device)
        prod_term = torch.prod(torch.cos(x / torch.sqrt(i)), dim=-1, keepdim=True)

        result = 1 + sum_term - prod_term
        return -1 * result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((
            -50 * torch.ones(input_dim_num, dtype=dtype, device=device),
            50 * torch.ones(input_dim_num, dtype=dtype, device=device)
        ), dim=0),
        min_point=torch.zeros(input_dim_num, dtype=dtype, device=device),
        min_value=torch.full((1,), 0.0, dtype=dtype, device=device),
        max_point=None,
        max_value=None,
    )

def Michalewicz(num_dims=2, m=10):
    input_dim_num = num_dims
    func_name = 'Michalewicz'

    if input_dim_num == 2:
        min_point = torch.tensor([2.20, 1.57], dtype=dtype, device=device)
        min_value = torch.tensor([-1.8013], dtype=dtype, device=device)
    elif input_dim_num == 5:
        min_point = None
        min_value = torch.tensor([-4.687658], dtype=dtype, device=device)
    elif input_dim_num == 10:
        min_point = None
        min_value = torch.tensor([-9.66015], dtype=dtype, device=device)
    else:
        min_point = None
        min_value = None

    # 定义目标函数（与 Ackley 风格一致：返回 -1 * result）
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"

        i = torch.arange(1, input_dim_num + 1, dtype=dtype, device=device)

        if len(x.shape) == 2:
            term = torch.sin(x) * torch.sin(i * x**2 / np.pi) ** (2 * m)
            result = -torch.sum(term, dim=-1, keepdim=True)
        elif len(x.shape) == 3:
            term = torch.sin(x) * torch.sin(i.view(1, 1, -1) * x**2 / np.pi) ** (2 * m)
            result = -torch.sum(term, dim=-1, keepdim=True)
        else:
            raise ValueError("Input x must be 2D or 3D tensor.")

        return -1 * result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((
            torch.zeros(input_dim_num, dtype=dtype, device=device),
            torch.pi * torch.ones(input_dim_num, dtype=dtype, device=device)
        ), dim=0),
        min_point=min_point,
        min_value=min_value,
        max_point=None,
        max_value=None,
    )

def Dixon_Price(num_dims=2):
    input_dim_num = num_dims
    func_name = 'Dixon_Price'

    # 已知全局最优点
    min_point = torch.zeros(input_dim_num, dtype=dtype, device=device)
    min_point[0] = 1.0
    for i in range(2, input_dim_num + 1):
        min_point[i - 1] = torch.sqrt(torch.tensor(2.0 ** (-(2**i - 2) / (2**i)), dtype=dtype, device=device))

    min_value = torch.tensor([0.0], dtype=dtype, device=device)

    # 定义目标函数
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"

        if len(x.shape) == 2:
            x1_term = (x[:, 0] - 1) ** 2
            i = torch.arange(2, input_dim_num + 1, dtype=dtype, device=device)
            sum_term = torch.sum(
                i * (2 * x[:, 1:] ** 2 - x[:, :-1]) ** 2,
                dim=-1
            )
            result = (x1_term + sum_term).reshape(-1, 1)

        elif len(x.shape) == 3:
            x1_term = (x[:, :, 0] - 1) ** 2
            i = torch.arange(2, input_dim_num + 1, dtype=dtype, device=device).view(1, 1, -1)
            sum_term = torch.sum(
                i * (2 * x[:, :, 1:] ** 2 - x[:, :, :-1]) ** 2,
                dim=-1
            )
            result = (x1_term + sum_term).reshape(-1, 1)

        else:
            raise ValueError("Input x must be 2D or 3D tensor.")

        return -1 * result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((
            -10 * torch.ones(input_dim_num, dtype=dtype, device=device),
            10 * torch.ones(input_dim_num, dtype=dtype, device=device)
        ), dim=0),
        min_point=min_point,
        min_value=min_value,
        max_point=None,
        max_value=None,
    )

def Hartmann(num_dims=6):
    input_dim_num = num_dims
    func_name = f'Hartmann{num_dims}'

    assert input_dim_num in [3, 6], "Hartmann function is only defined for 3 or 6 dimensions."

    if input_dim_num == 3:
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=dtype, device=device)

        A = torch.tensor([
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35]
        ], dtype=dtype, device=device)

        P = 1e-4 * torch.tensor([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828]
        ], dtype=dtype, device=device)

        min_point = torch.tensor(
            [0.114614, 0.555649, 0.852547],
            dtype=dtype, device=device
        )
        min_value = torch.tensor([-3.86278], dtype=dtype, device=device)

    elif input_dim_num == 6:
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=dtype, device=device)

        A = torch.tensor([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ], dtype=dtype, device=device)

        P = 1e-4 * torch.tensor([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ], dtype=dtype, device=device)

        min_point = torch.tensor(
            [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
            dtype=dtype, device=device
        )
        min_value = torch.tensor([-3.32237], dtype=dtype, device=device)

    # 目标函数
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"

        if len(x.shape) == 2:
            x_expanded = x.unsqueeze(1)
            inner = torch.sum(A.unsqueeze(0) * (x_expanded - P.unsqueeze(0))**2, dim=-1)
            result = -torch.sum(alpha.unsqueeze(0) * torch.exp(-inner), dim=-1, keepdim=True)
        elif len(x.shape) == 3:
            x_expanded = x.unsqueeze(2)
            inner = torch.sum(
                A.view(1, 1, 4, input_dim_num) * (x_expanded - P.view(1, 1, 4, input_dim_num))**2,
                dim=-1
            )
            result = -torch.sum(
                alpha.view(1, 1, 4) * torch.exp(-inner),
                dim=-1,
                keepdim=True
            )
        else:
            raise ValueError("Input x must be 2D or 3D tensor.")

        return -result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((
            torch.zeros(input_dim_num, dtype=dtype, device=device),
            torch.ones(input_dim_num, dtype=dtype, device=device)
        ), dim=0),
        min_point=min_point,
        min_value=min_value,
        max_point=None,
        max_value=None,
    )

def Branin(num_dims=2):
    assert num_dims == 2, "Branin function is only defined for 2 dimensions!"

    input_dim_num = num_dims
    func_name = 'Branin'

    # 定义目标函数
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"

        x1 = x[..., 0:1]
        x2 = x[..., 1:2]

        a = 1.0
        b = 5.1 / (4 * torch.pi ** 2)
        c = 5.0 / torch.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * torch.pi)

        result = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s

        return -1 * result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.tensor([
            [-5.0, 0.0],
            [10.0, 15.0]
        ], dtype=dtype, device=device),
        min_point=torch.tensor(
            [-torch.pi, 12.275],
            dtype=dtype,
            device=device
        ),
        min_value=torch.full((1,), 0.397887, dtype=dtype, device=device),
        max_point=None,
        max_value=None,
    )


def Sin():
    input_dim_num = 1
    func_name = 'Sin'
    # 定义目标函数
    def function(x):
        result = torch.sin(2*np.pi*x).reshape(-1, 1)
        return result + torch.randn_like(result) * noise
    # 定义保真度函数
    coe = 2*torch.pi
    def function_F0(x,):
        # 使用 sin(x) 的泰勒展开的前几项来近似
        y1 = coe * x - ((coe * x) ** 3) / 6
        y2 = y1 + ((coe * x) ** 5) / math.factorial(5)
        y3 = y2 - ((coe * x) ** 7) / math.factorial(7)
        y4 = y3 + ((coe * x) ** 9) / math.factorial(9)
        y5 = y4 - ((coe * x) ** 11) / math.factorial(11)
        result  = y5 + ((coe * x) ** 13) / math.factorial(13)
        return result + torch.randn_like(result) * noise

    def function_F1(x, func = function_F0):
        result = func(x) - ((coe * x) ** 15) / math.factorial(15)
        return result + torch.randn_like(result) * noise
    
    def function_F2(x, func = function_F1):
        result = func(x) + ((coe * x) ** 17) / math.factorial(17)
        return result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        mf_funcs=mf_function(fidel_func=[function_F0, function_F1, function_F2, function]),
        bounds=torch.stack((torch.zeros(input_dim_num, dtype=dtype, device=device), torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=torch.tensor([0.75], dtype=dtype, device=device),
        min_value=torch.tensor([-1.0], dtype=dtype, device=device),
        max_point=torch.tensor([0.25], dtype=dtype, device=device),
        max_value=torch.tensor([1.0], dtype=dtype, device=device),
    )

def SVM():
    input_dim_num = 2
    func_name = 'SVM'
    # 定义参数
    data = fetch_openml('magic', version=1)
    x_data = data.data  # 特征
    y_data = data.target  # 标签
    fidelity = 0.9
    train_size = int(fidelity * x_data.shape[0])  # 训练集大小
    x_train_data, _, y_train_data, _ = train_test_split(x_data, y_data, train_size=train_size, random_state=42)
    scaler = StandardScaler()
    x_train_data = scaler.fit_transform(x_train_data)  # 数据标准化
    # 定义目标函数
    def function(x, input_data=x_train_data, output_data=y_train_data):
        # 确保x的维度是符合的
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        # 计算目标函数值
        result = torch.zeros(x.shape[0], 1, dtype=dtype, device=device)
        for i in range(x.shape[0]):
            # 获取输入的 C 和 gamma
            soft_margin_coefficient = x[i, 0].item()
            kernel_bandwidth = x[i, 1].item()
            # 创建一个SVM模型
            svm_model = SVC(kernel='rbf', C=soft_margin_coefficient, gamma=kernel_bandwidth)
            # 使用5折交叉验证评估模型
            scores = cross_val_score(svm_model, input_data, output_data, cv=5, scoring='accuracy')
            result[i, 0] = torch.tensor(scores.mean(), dtype=dtype, device=device)
        # 添加噪声, 返回结果
        return result + torch.randn_like(result) * noise
    # 定义保真度函数
    fidelity = 0.6
    # train_size = int(fidelity * x_data.shape[0])  # 训练集大小
    train_size = 500
    x_train_data, _, y_train_data, _ = train_test_split(x_data, y_data, train_size=train_size, random_state=42)
    scaler = StandardScaler()
    x_train_data = scaler.fit_transform(x_train_data)  # 数据标准化
    def function_F0(x, input_data=x_train_data, output_data=y_train_data):
        return function(x, input_data, output_data)

    fidelity = 0.3
    # train_size = int(fidelity * x_data.shape[0])  # 训练集大小
    train_size = 2000
    x_train_data, _, y_train_data, _ = train_test_split(x_data, y_data, train_size=train_size, random_state=42)
    scaler = StandardScaler()
    x_train_data = scaler.fit_transform(x_train_data)  # 数据标准化
    def function_F1(x, input_data=x_train_data, output_data=y_train_data):
        return function(x, input_data, output_data)

    return obj_func(
        func=function,
        func_name=func_name,
        mf_funcs=mf_function(fidel_func=[function_F0, function_F1, function]),
        bounds=torch.stack((torch.tensor([0.1, 0.001], dtype=dtype, device=device),
                            torch.tensor([100000, 10], dtype=dtype, device=device)), dim=0),  # [C, gamma]
        min_point=None,
        min_value=None,
        max_point=None,
        max_value=None,
    )

def G04():
    input_dim_num = 5
    func_name = 'G04'
    # 定义目标函数
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"  # 确保x的维度是符合的
        if len(x.shape) == 3:
            x1 = x[:, :, 0]
            x3 = x[:, :, 2]
            x5 = x[:, :, 4]
        elif len(x.shape) == 2:
            x1 = x[:, 0]
            x3 = x[:, 2]
            x5 = x[:, 4]
        result = (5.3578547 * (x3**2) + 0.8356891*x1*x5 + 37.293239*x1 - 40792.141).reshape(-1, 1)
        return -1*result + torch.randn_like(result) * noise
    # 定义约束函数
    def function_C1(x):
        pass

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs = [function_C1],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((torch.tensor([78, 33, 27, 27, 27], dtype=dtype, device=device), torch.tensor([102, 45, 45, 45, 45], dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.full((input_dim_num,), 0.0, dtype=dtype, device=device),
        max_value=torch.full((1,), 0.0, dtype=dtype, device=device),
    )

def G09():
    input_dim_num = 7
    func_name = 'G09'
    # 定义目标函数
    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"  # 确保x的维度是符合的
        if len(x.shape) == 3:
            x1 = x[:, :, 0]
            x2 = x[:, :, 1]
            x3 = x[:, :, 2]
            x4 = x[:, :, 3]
            x5 = x[:, :, 4]
            x6 = x[:, :, 5]
            x7 = x[:, :, 6]
        elif len(x.shape) == 2:
            x1 = x[:, 0]
            x2 = x[:, 1]
            x3 = x[:, 2]
            x4 = x[:, 3]
            x5 = x[:, 4]
            x6 = x[:, 5]
            x7 = x[:, 6]
        result = ((x1-10)**2 + 5*(x2-12)**2 + x3**4 + 3*(x4-11)**2 + 10*x5**6 + 7*x6**2 + x7**4 - 4*x6*x7 - 10*x6 - 8*x7).reshape(-1, 1)
        return -1*result + torch.randn_like(result) * noise
    # 定义约束函数
    def function_C1(x):
        pass

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs = [function_C1],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.stack((-10*torch.ones(input_dim_num, dtype=dtype, device=device),
                            10*torch.ones(input_dim_num, dtype=dtype, device=device)), dim=0),
        min_point=None,
        min_value=None,
        max_point=torch.full((input_dim_num,), 0.0, dtype=dtype, device=device),
        max_value=torch.full((1,), 0.0, dtype=dtype, device=device),
    )


from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer, load_digits, load_diabetes, fetch_california_housing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import fetch_california_housing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

def _to_python_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)

def _tensor_result(values, dtype, device):
    arr = np.array(values, dtype=np.float64).reshape(-1, 1)
    return torch.tensor(arr, dtype=dtype, device=device)

# SVM + Breast Cancer
def SVM_BreastCancer():
    func_name = "SVM_BreastCancer"
    input_dim_num = 2

    # 固定数据划分，减少噪声
    X, y = load_breast_cancer(return_X_y=True)  # 数据集加载与固定划分
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25,  # 25% 做验证集
        random_state=42,  # 固定随机种子，保证每次划分一致
        stratify=y,  # 固定随机种子，保证每次划分一致
    )

    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        x_np = x.detach().cpu().numpy()

        vals = []
        for row in x_np:  # 遍历每个候选超参数点
            # 提取超参数
            log10_C, log10_gamma = row.tolist()  # 读出两个优化变量
            C = 10 ** log10_C
            gamma = 10 ** log10_gamma
            # 标准化 + SVM
            model = Pipeline([
                ("scaler", StandardScaler()),  # 标准化
                ("clf", SVC(  # SVM 分类器
                    C=C,
                    gamma=gamma,
                    kernel="rbf",
                    random_state=42
                ))
            ])
            model.fit(X_train, y_train)  # 训练模型
            y_pred = model.predict(X_val)  # 在验证集上预测
            score = accuracy_score(y_val, y_pred)   # 计算目标值：准确率 （越大越好）
            # 保存每个点的结果
            vals.append(score)

        result = _tensor_result(vals, dtype, device)
        return result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.tensor([
            [-3.0, -4.0],   # lower
            [ 3.0,  1.0],   # upper
        ], dtype=dtype, device=device),
        min_point=None,
        min_value=None,
        max_point=None,
        max_value=None,
    )

# SVM_BreastCancer
def MLP_Digits():
    func_name = "MLP_Digits"
    input_dim_num = 5

    X, y = load_digits(return_X_y=True)  # 读取 digits 数据集，并固定划分
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        x_np = x.detach().cpu().numpy()

        vals = []  # 保存每个候选点的得分
        for row in x_np:
            log10_alpha, log10_lr, hidden_units1, hidden_units2, log2_batch = row.tolist()

            # 连续变量 -> 实际超参数
            alpha = 10 ** log10_alpha
            learning_rate_init = 10 ** log10_lr

            # 离散结构参数
            hidden_units1 = int(np.clip(round(hidden_units1), 20, 200))
            hidden_units2 = int(np.clip(round(hidden_units2), 10, 150))
            batch_size = int(2 ** np.clip(round(log2_batch), 4, 7))  # 16, 32, 64, 128

            # 如果第二层太小，可退化为单隐层；否则用双隐层
            if hidden_units2 <= 10:
                hidden_layer_sizes = (hidden_units1,)
            else:
                hidden_layer_sizes = (hidden_units1, hidden_units2)

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    batch_size=batch_size,
                    max_iter=200,
                    early_stopping=True,
                    random_state=42
                ))
            ])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)

            vals.append(score)

        result = _tensor_result(vals, dtype, device)
        return result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.tensor([
            [-6.0, -5.0, 20.0, 10.0, 4.0],
            [-1.0, -1.0, 200.0, 150.0, 7.0],
        ], dtype=dtype, device=device),
        min_point=None,
        min_value=None,
        max_point=None,
        max_value=None,
    )

def XGBoost_Diabetes():  # 回归
    func_name = "XGBoost_Diabetes"
    input_dim_num = 8

    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    def function(x):
        # start = time.time()
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        x_np = x.detach().cpu().numpy()

        vals = []
        for row in x_np:
            (
                max_depth,
                log10_lr,
                subsample,
                colsample_bytree,
                min_child_weight,
                log10_reg_alpha,
                log10_reg_lambda,
                gamma
            ) = row.tolist()

            # ===== 参数处理 =====
            max_depth = int(np.clip(round(max_depth), 2, 10))
            learning_rate = 10 ** np.clip(log10_lr, -2.5, -0.5)

            subsample = float(np.clip(subsample, 0.5, 1.0))
            colsample_bytree = float(np.clip(colsample_bytree, 0.5, 1.0))

            min_child_weight = float(np.clip(min_child_weight, 1.0, 10.0))

            reg_alpha = 10 ** np.clip(log10_reg_alpha, -5.0, 1.0)
            reg_lambda = 10 ** np.clip(log10_reg_lambda, -5.0, 2.0)

            gamma = float(np.clip(gamma, 0.0, 5.0))

            # ===== 模型 =====
            model = XGBRegressor(
                n_estimators=200,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                gamma=gamma,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=42,
                n_jobs=1,
                verbosity=0
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # 最大化 R2
            score = r2_score(y_val, y_pred)

            vals.append(score)

        result = _tensor_result(vals, dtype, device)
        # end = time.time()
        # print(f"运行时间: {(end - start)/60:.6f} min")
        return result + torch.randn_like(result) * noise

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=torch.tensor([
            [2.0, -2.5, 0.5, 0.5, 1.0, -5.0, -5.0, 0.0],
            [10.0, -0.5, 1.0, 1.0, 10.0, 1.0, 2.0, 5.0],
        ], dtype=dtype, device=device),
        min_point=None,
        min_value=None,
        max_point=None,
        max_value=None,
    )

def RandomForest_Regression(dataset_name="diabetes"):  # 回归
    func_name = f"RandomForest_{dataset_name}"
    input_dim_num = 5

    # ===== 数据集选择 =====
    dataset_name = dataset_name.lower()

    if dataset_name == "diabetes":
        X, y = load_diabetes(return_X_y=True)
    elif dataset_name in ["california", "california_housing", "housing"]:
        X, y = fetch_california_housing(return_X_y=True)
    else:
        raise ValueError(
            f"Unsupported dataset_name: {dataset_name}. "
            f"Supported choices are ['diabetes', 'california_housing']."
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        x_np = x.detach().cpu().numpy()

        vals = []
        for row in x_np:
            (
                n_estimators,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features,
            ) = row.tolist()

            # ===== 参数处理 =====
            if dataset_name == "diabetes":
                n_estimators = int(np.clip(round(n_estimators), 50, 300))
                max_depth = int(np.clip(round(max_depth), 2, 20))
            else:  # california_housing
                n_estimators = int(np.clip(round(n_estimators), 50, 400))
                max_depth = int(np.clip(round(max_depth), 2, 30))

            min_samples_split = int(np.clip(round(min_samples_split), 2, 20))
            min_samples_leaf = int(np.clip(round(min_samples_leaf), 1, 10))
            max_features = float(np.clip(max_features, 0.1, 1.0))

            # ===== 模型 =====
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=True,
                random_state=42,
                n_jobs=1,
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # 最大化 R2
            score = r2_score(y_val, y_pred)
            vals.append(score)

        result = _tensor_result(vals, dtype, device)
        return result + torch.randn_like(result) * noise

    # ===== 搜索边界 =====
    if dataset_name == "diabetes":
        bounds = torch.tensor([
            [50.0,  2.0,  2.0, 1.0, 0.1],
            [300.0, 20.0, 20.0, 10.0, 1.0],
        ], dtype=dtype, device=device)
    else:  # california_housing
        bounds = torch.tensor([
            [50.0,  2.0,  2.0, 1.0, 0.1],
            [400.0, 30.0, 20.0, 10.0, 1.0],
        ], dtype=dtype, device=device)

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=bounds,
        min_point=None,
        min_value=None,
        max_point=None,
        max_value=None,
    )

def LightGBM_AdultIncome():  # 二分类
    func_name = "LightGBM_AdultIncome"
    input_dim_num = 10

    X, y = fetch_openml(name="adult", version=2, as_frame=True, return_X_y=True)

    # 标签转成 0/1
    y = (y == ">50K").astype(int)

    # 数值列 / 类别列
    categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["category", "object"]).columns.tolist()

    # 预处理
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    def function(x):
        assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
        x_np = x.detach().cpu().numpy()

        vals = []
        for row in x_np:
            (
                num_leaves,
                learning_rate,
                max_depth,
                min_child_samples,
                feature_fraction,
                bagging_fraction,
                bagging_freq,
                lambda_l1,
                lambda_l2,
                min_split_gain,
            ) = row.tolist()

            # ===== 参数处理 =====
            num_leaves = int(np.clip(round(num_leaves), 16, 256))
            learning_rate = float(np.clip(learning_rate, 0.01, 0.3))
            max_depth = int(np.clip(round(max_depth), -1, 15))
            min_child_samples = int(np.clip(round(min_child_samples), 5, 100))
            feature_fraction = float(np.clip(feature_fraction, 0.6, 1.0))
            bagging_fraction = float(np.clip(bagging_fraction, 0.6, 1.0))
            bagging_freq = int(np.clip(round(bagging_freq), 0, 10))
            lambda_l1 = float(np.clip(lambda_l1, 0.0, 10.0))
            lambda_l2 = float(np.clip(lambda_l2, 0.0, 10.0))
            min_split_gain = float(np.clip(min_split_gain, 0.0, 1.0))

            # LightGBM 要求 bagging_fraction < 1 时才有 bagging_freq 的意义
            if bagging_fraction >= 0.999:
                bagging_freq = 0

            # ===== 模型 =====
            model = LGBMClassifier(
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                feature_fraction=feature_fraction,
                bagging_fraction=bagging_fraction,
                bagging_freq=bagging_freq,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                min_split_gain=min_split_gain,
                n_estimators=200,
                objective="binary",
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            )

            clf = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            # 最大化 accuracy
            score = accuracy_score(y_val, y_pred)
            vals.append(score)

        result = _tensor_result(vals, dtype, device)
        return result + torch.randn_like(result) * noise

    # ===== 搜索边界 =====
    bounds = torch.tensor([
        [16.0, 0.01, -1.0,  5.0, 0.6, 0.6,  0.0, 0.0, 0.0, 0.0],
        [256.0, 0.3, 15.0, 100.0, 1.0, 1.0, 10.0, 10.0, 10.0, 1.0],
    ], dtype=dtype, device=device)

    return obj_func(
        func=function,
        func_name=func_name,
        c_funcs=[],
        mf_funcs=mf_function(fidel_func=[function]),
        bounds=bounds,
        min_point=None,
        min_value=None,
        max_point=None,
        max_value=None,
    )


# def XGBoost_CaliforniaHousing(X, y):  # 回归
#     func_name = "XGBoost_CaliforniaHousing"
#     input_dim_num = 5
#
#     X, y = fetch_california_housing(return_X_y=True)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.25, random_state=42
#     )
#
#     def function(x):
#         assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
#         x_np = x.detach().cpu().numpy()
#
#         vals = []
#         for row in x_np:
#             max_depth, log10_learning_rate, subsample, colsample_bytree, min_child_weight = row.tolist()
#
#             max_depth = int(np.clip(round(max_depth), 2, 10))
#             learning_rate = 10 ** float(np.clip(log10_learning_rate, -2.0, -0.5))
#             subsample = float(np.clip(subsample, 0.5, 1.0))
#             colsample_bytree = float(np.clip(colsample_bytree, 0.5, 1.0))
#             min_child_weight = float(np.clip(min_child_weight, 1.0, 10.0))
#
#             model = XGBRegressor(
#                 n_estimators=200,
#                 max_depth=max_depth,
#                 learning_rate=learning_rate,
#                 subsample=subsample,
#                 colsample_bytree=colsample_bytree,
#                 min_child_weight=min_child_weight,
#                 objective="reg:squarederror",
#                 eval_metric="rmse",
#                 random_state=42,
#                 n_jobs=1,
#                 verbosity=0
#             )
#
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_val)
#             score = r2_score(y_val, y_pred)
#             vals.append(score)
#
#         result = _tensor_result(vals, dtype, device)
#         return result + torch.randn_like(result) * noise
#
#     return obj_func(
#         func=function,
#         func_name=func_name,
#         c_funcs=[],
#         mf_funcs=mf_function(fidel_func=[function]),
#         bounds=torch.tensor([
#             [2.0, -2.0, 0.5, 0.5, 1.0],
#             [10.0, -0.5, 1.0, 1.0, 10.0],
#         ], dtype=dtype, device=device),
#         min_point=None,
#         min_value=None,
#         max_point=None,
#         max_value=None,
#     )


# def MLP_Digits1():
#     func_name = "MLP_Digits"
#     input_dim_num = 4
#
#     X, y = load_digits(return_X_y=True)  # 读取digits数据集，并固定划分
#     X_train, X_val, y_train, y_val = train_test_split(  # 划分训练集和验证集
#         X, y, test_size=0.25, random_state=42, stratify=y
#     )
#
#     def function(x):
#         assert x.shape[-1] == input_dim_num, f"Input dimension must be {input_dim_num}!"
#         x_np = x.detach().cpu().numpy()
#
#         vals = []  # 保存每个候选点的得分
#         for row in x_np:
#             log10_alpha, log10_lr, hidden_units, log2_batch = row.tolist()
#
#             alpha = 10 ** log10_alpha  # L2 正则化系数
#             learning_rate_init = 10 ** log10_lr  # 学习率
#             hidden_units = int(np.clip(round(hidden_units), 20, 200))  # 隐藏层神经元个数
#             batch_size = int(2 ** np.clip(round(log2_batch), 4, 7))  # batch size
#
#             model = Pipeline([
#                 ("scaler", StandardScaler()),
#                 ("clf", MLPClassifier(
#                     hidden_layer_sizes=(hidden_units,),
#                     alpha=alpha,
#                     learning_rate_init=learning_rate_init,
#                     batch_size=batch_size,
#                     max_iter=200,
#                     early_stopping=True,
#                     random_state=42
#                 ))
#             ])
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_val)
#             score = accuracy_score(y_val, y_pred)
#
#             vals.append(score)
#
#         result = _tensor_result(vals, dtype, device)
#         return result + torch.randn_like(result) * noise
#
#     return obj_func(
#         func=function,
#         func_name=func_name,
#         c_funcs=[],
#         mf_funcs=mf_function(fidel_func=[function]),
#         bounds=torch.tensor([
#             [-6.0, -5.0, 20.0, 4.0],
#             [-1.0, -1.0, 200.0, 7.0],
#         ], dtype=dtype, device=device),
#         min_point=None,
#         min_value=None,
#         max_point=None,
#         max_value=None,
#     )

# XGBoost / LightGBM + California Housing



# 神经网络模型
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # 导入SVM分类器
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


class SVM_magic_gamma_dataset:
    def __init__(self, fidelity=0.5):
        self.dims = 2  # the soft margin coefficient (C) and the kernel bandwidth (gamma)
        self.lb = torch.tensor([0.1, 0.001], dtype=dtype, device=device)
        self.ub = torch.tensor([100000, 10], dtype=dtype, device=device)
        self.min_point = None  # 最小值点
        self.min_value = None  # 最小值

        # 加载Magic Gamma数据集 ==============================================================
        data = fetch_openml('magic', version=1)
        X = data.data  # 特征
        y = data.target  # 标签
        # 随机选择2000个样本
        self.fidelity = fidelity
        train_size = int(fidelity * X.shape[0])
        X_subsample, _, self.y_data, _ = train_test_split(X, y, train_size=train_size, random_state=42)
        # 数据标准化
        scaler = StandardScaler()
        self.x_data = scaler.fit_transform(X_subsample)

    def __call__(self, x):
        result = torch.zeros(x.shape[0], 1, dtype=dtype, device=device)
        for i in range(x.shape[0]):
            # 训练SVM ===========================================================================
            # 获取输入的 C 和 gamma
            soft_margin_coefficient = x[i, 0].item()
            kernel_bandwidth = x[i, 1].item()
            # 创建一个SVM模型
            svm_model = SVC(kernel='rbf', C=soft_margin_coefficient, gamma=kernel_bandwidth)
            # 使用5折交叉验证评估模型
            scores = cross_val_score(svm_model, self.x_data, self.y_data, cv=5, scoring='accuracy')
            result[i, 0] = torch.tensor(scores.mean(), dtype=dtype, device=device)

        return result  #


class Swimmer:

    def __init__(self):
        self.policy_shape = (2, 8)
        self.mean = 0
        self.std = 0.1
        self.dims = 16
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.env = gym.make('Swimmer-v2')
        self.num_rollouts = 3
        self.Cp = 20
        self.leaf_size = 10
        self.kernel_type = "poly"
        self.gamma_type = "scale"
        self.ninits = 40
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape)

        self.render = False

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        # assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x = np.array(x)

        M = x.reshape(self.policy_shape)

        returns = []
        observations = []
        actions = []

        for i in range(self.num_rollouts):
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:

                # action = np.dot(M, (obs - self.mean) / self.std)
                action = np.dot(M, obs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if self.render:
                    self.env.render()
            returns.append(totalr)

        return np.mean(returns) * -1
    

class Swimmer:

    def __init__(self):
        self.policy_shape = (2, 8)
        self.mean = 0
        self.std = 0.1
        self.dims = 16
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.env = gym.make('Swimmer-v2')
        self.num_rollouts = 3
        self.num_steps = 1000
        self.Cp = 20
        self.leaf_size = 10
        self.kernel_type = "poly"
        self.gamma_type = "scale"
        self.ninits = 40
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape)

        self.render = True

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        # assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x = np.array(x)

        M = x.reshape(self.policy_shape)

        returns = []
        observations = []
        actions = []

        for i in range(self.num_rollouts):
            print("rollout:", i)
            obs = self.env.reset()[0]
            done = False
            totalr = 0.
            steps = 0

            while not done:
                print(f"step:{steps}\r", end='')
                # action = np.dot(M, (obs - self.mean) / self.std)
                action = np.dot(M, obs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _, _ = self.env.step(action)
                totalr += r
                steps += 1
                if steps > self.num_steps:
                    done = True
                if self.render:
                    self.env.render()
            returns.append(totalr)
        print(returns)
        return np.mean(returns) * -1
    

class Lunarlanding:
    def __init__(self):
        self.dims = 12
        self.lb = np.zeros(12)
        self.ub = 2 * np.ones(12)
        self.counter = 0
        self.env = gym.make('LunarLander-v2')
        self.Cp = 50
        self.leaf_size = 10
        self.kernel_type = "poly"
        self.ninits = 40
        self.gamma_type = "scale"

        self.render = False

    def heuristic_Controller(self, s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a

    def __call__(self, x):
        x = np.array(x)
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        total_rewards = []
        for i in range(0, 3):  # controls the number of episode/plays per trial
            state = self.env.reset()[0]
            rewards_for_episode = []
            num_steps = 2000

            for step in range(num_steps):
                if self.render:
                    self.env.render()
                received_action = self.heuristic_Controller(state, x)
                next_state, reward, done, info, _ = self.env.step(received_action)
                rewards_for_episode.append(reward)
                state = next_state
                if done:
                    break

            rewards_for_episode = np.array(rewards_for_episode)
            total_rewards.append(np.sum(rewards_for_episode))
        total_rewards = np.array(total_rewards)
        mean_rewards = np.mean(total_rewards)

        print(f"\nmean_rewards: {mean_rewards}")

        return mean_rewards * -1
    

class Hopper:

    def __init__(self):
        self.mean = np.array([1.41599384, -0.05478602, -0.25522216, -0.25404721,
                              0.27525085, 2.60889529, -0.0085352, 0.0068375,
                              -0.07123674, -0.05044839, -0.45569644])
        self.std = np.array([0.19805723, 0.07824488, 0.17120271, 0.32000514,
                             0.62401884, 0.82814161, 1.51915814, 1.17378372,
                             1.87761249, 3.63482761, 5.7164752])
        self.dims = 33
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.env = gym.make('Hopper-v2')
        self.num_rollouts = 3
        self.render = False
        self.policy_shape = (3, 11)

        self.Cp = 10
        self.leaf_size = 100
        self.kernel_type = "poly"
        self.gamma_type = "auto"
        self.ninits = 150

        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape)

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        # assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x = np.array(x)

        M = x.reshape(self.policy_shape)

        returns = []
        observations = []
        actions = []

        for i in range(self.num_rollouts):
            obs = self.env.reset()[0]
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # M      = self.policy
                # inputs = (obs - self.mean) / self.std
                inputs = obs
                action = np.dot(M, inputs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _, _ = self.env.step(action)
                totalr += r
                steps += 1
                if self.render:
                    self.env.render()
            returns.append(totalr)

        return np.mean(returns) * -1


class Walker:

    def __init__(self):
        self.mean = 0
        self.std = 1
        self.dims = 102
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.env = gym.make('Walker2d')
        self.num_rollouts = 3
        self.render = False
        self.policy_shape = (6, 17)

        self.Cp = 10
        self.leaf_size = 100
        self.kernel_type = "poly"
        self.gamma_type = "auto"
        self.ninits = 150

        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape)

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        # assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x = np.array(x)

        M = x.reshape(self.policy_shape)

        returns = []
        observations = []
        actions = []

        for i in range(self.num_rollouts):
            obs = self.env.reset()[0]
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # M      = self.policy
                # inputs = (obs - self.mean) / self.std
                inputs = obs
                action = np.dot(M, inputs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _, _ = self.env.step(action)
                totalr += r
                steps += 1
                if self.render:
                    self.env.render()
            returns.append(totalr)
        print(returns)
        return np.mean(returns) * -1


class HalfCheetah:
    
    def __init__(self):
        self.mean = 0
        self.std = 0.1
        self.dims = 102
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.env = gym.make('HalfCheetah-v3')
        # self.env = TimeLimit(self.env, max_episode_steps=1000)  # 设置最大步数为 1000
        self.num_rollouts = 3
        self.num_steps = 200000
        self.render = False
        self.policy_shape = (6, 17)

        self.Cp = 10
        self.leaf_size = 100
        self.kernel_type = "poly"
        self.gamma_type = "auto"
        self.ninits = 150

        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape)

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        # assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x = np.array(x)

        M = x.reshape(self.policy_shape)

        returns = []
        observations = []
        actions = []

        for i in range(self.num_rollouts):
            obs = self.env.reset()[0]
            done = False
            totalr = 0.
            steps = 0
            while not done:
                print("step:", steps, "\r", end='')
                inputs = obs
                action = np.dot(M, inputs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _, _ = self.env.step(action)
                totalr += r
                steps += 1
                if steps >= self.num_steps:
                    done = True
                if self.render:
                    self.env.render()
            returns.append(totalr)
        print(returns)
        return np.mean(returns) * -1


# from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
# from hpobench.util.openml_data_manager import get_openmlcc18_taskids


class XGBoost:

    def __init__(self, task_id=167149):
        self.benchmark = XGBoostBenchmark(task_id=task_id,
                             container_name='xgboost_benchmark',
                             container_source='library://phmueller/automl')
        
        cs = self.benchmark.get_configuration_space()
        self.hypermeters_config_space = cs
        self.hypermeters_configuration = self.hypermeters_config_space.get_default_configuration()
        self.hypermeters = self.hypermeters_config_space.get_hyperparameters()
        # 获取所有超参数的上下界并组成一个字典
        self.hyperparameters_bounds = {}
        for hyperparameter in self.hypermeters:
            self.hyperparameters_bounds[hyperparameter.name] = {
                'lower': hyperparameter.lower,
                'upper': hyperparameter.upper
            }
        self.lb = torch.tensor([self.hyperparameters_bounds['colsample_bylevel']['lower'], self.hyperparameters_bounds['colsample_bytree']['lower'], self.hyperparameters_bounds['eta']['lower'], self.hyperparameters_bounds['min_child_weight']['lower'], self.hyperparameters_bounds['reg_alpha']['lower'], self.hyperparameters_bounds['reg_lambda']['lower']], dtype=dtype, device=device)
        self.ub = torch.tensor([self.hyperparameters_bounds['colsample_bylevel']['upper'], self.hyperparameters_bounds['colsample_bytree']['upper'], self.hyperparameters_bounds['eta']['upper'], self.hyperparameters_bounds['min_child_weight']['upper'], self.hyperparameters_bounds['reg_alpha']['upper'], self.hyperparameters_bounds['reg_lambda']['upper']], dtype=dtype, device=device)

        self.dims = self.hyperparameters_bounds.__len__()

    def __call__(self, x):
        x = x.numpy()
        test_loss_narray = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            hypermeters_value = x[i, :]
            self.hypermeters_configuration['colsample_bylevel'] = float(hypermeters_value[0])
            self.hypermeters_configuration['colsample_bytree'] = float(hypermeters_value[1])
            self.hypermeters_configuration['eta'] = float(hypermeters_value[2])
            self.hypermeters_configuration['min_child_weight'] = float(hypermeters_value[3])
            self.hypermeters_configuration['reg_alpha'] = float(hypermeters_value[4])
            self.hypermeters_configuration['reg_lambda'] = float(hypermeters_value[5])
            

            for n_estimator in [64]:
                for subsample in [1]:
                    fidelity = {'n_estimators': n_estimator, 'dataset_fraction': subsample}
                    result_dict = self.benchmark.objective_function(self.hypermeters_configuration.get_dictionary(),
                                                        fidelity=fidelity)
                    valid_loss = result_dict['function_value']
                    train_loss = result_dict['info']['train_loss']
                    assert result_dict['info']['fidelity'] == fidelity

                    result_dict = self.benchmark.objective_function_test(self.hypermeters_configuration, fidelity={'n_estimators': n_estimator})
                    test_loss = result_dict['function_value']

                    test_loss_narray[i, :] = test_loss

                # print(f'No Estimator: {n_estimator:3d} - '
                #       f'Subsample Rate: {subsample:.1f} - Test {test_loss:.4f} '
                #       f'- Valid {valid_loss:.4f} - Train {train_loss:.4f}')
        # self.benchmark.__del__()
        # print(test_loss_narray)
        return torch.tensor(test_loss_narray)
    

if __name__ == '__main__':
    # 与Levy一样的调用方式
    af1_problem = Airfoil_AF1()

    # 现在可以像调用Levy一样调用它
    print(af1_problem.func_name)  # 'Airfoil_AF1'
    print(af1_problem.bounds.shape)  # torch.Size([2, 16])

    # 评估一个batch
    x_test = torch.rand(5, 16, dtype=dtype, device=device)
    # 别忘了把x缩放到bounds内：
    lb, ub = af1_problem.bounds[0], af1_problem.bounds[1]
    x_test = lb + x_test * (ub - lb)

    y = af1_problem.func(x_test)
    print(y.shape)  # torch.Size([5, 1])
    print(y)  # 越大越好（最大化形式）



    from htslibs.utils.plotting_lib.plotting import boxplot_draw
    from htslibs.utils.sample_methods import latin_hypercube_sample_tensor
    # from time import time

    # 绘制保真度函数的误差箱线图
    func_name_list = ['Airfoil_AF1']  # 'Hartmann7D', 'Hartmann6D', 'Hartmann3D', 'Borehole', 'CurrinExp'
    boxplot_data = {}
    for func_name in func_name_list:
        obj_func = get_function_details(func_name, 1)  # Hartmann3D
        mf_funcs = obj_func.mf_funcs
        num_fidels = mf_funcs.num_fidels
        bounds = obj_func.bounds
        lb = bounds[0, :]
        ub = bounds[1, :]
        test_x = latin_hypercube_sample_tensor(100000, lb, ub)
        true_vals = obj_func(test_x)

        # 绘制保真度函数对比图
        marker_size = 1
        for i in range(mf_funcs.num_fidels):
            plt.scatter(test_x.numpy(), mf_funcs(i, test_x).numpy(), marker='.', s=marker_size, label=f'F{i}')
        plt.legend()
        plt.grid()
        plt.show()

        boxplot_data[func_name] = {}
        for i in range(num_fidels-1):
            pred_vals = mf_funcs(i, test_x)
            diff_vals = (true_vals.detach().numpy() - pred_vals.detach().numpy()).ravel()
            boxplot_data[func_name][f'fidel_{i}'] = diff_vals
    boxplot_draw(boxplot_data, ['#87CEEB', '#4682B4', '#191970'])




    # boxplot_data = {'Label1': {'A1': diff_vals.ravel()}}
    # boxplot_draw(boxplot_data, ['#87CEEB', '#4682B4', '#191970'])

    # start = time()
    # bb = XGBoost(167149)
    # bb(torch.tensor([1, 1, 0.3, 1, 1e-04, 1]).reshape(1, -1))
    # bb(torch.tensor([1, 1, 0.3, 1, 1e-04, 1]).reshape(1, -1))
    # bb.delete_benchmark()
    # print(f'Done, took totally {time()-start:.2f}')

    # a = 0
