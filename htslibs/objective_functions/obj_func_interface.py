import numpy as np
import torch

from htslibs.utils.load_data_from_json import load_config_from_json

# 获取参数配置
params_config = load_config_from_json()
device, dtype = torch.device(params_config["device"]), eval(params_config["dtype"])

noise = 0.001  # 噪声


class mf_function:
    def __init__(self, fidel_func=[]):
        self.num_fidels = len(fidel_func)
        self.costs = np.round(10 ** np.arange(0, self.num_fidels)).astype(int)
        self.fidel_func = fidel_func

    def __call__(self, fidel, x):
        return self.fidel_func[fidel](x)  # 1-based index

class obj_func:
    def __init__(self,
            func, func_name, c_funcs, mf_funcs, bounds,
            min_point=None, min_value=None, max_point=None, max_value=None,
    ):
        self.func_name = func_name
        self.func = func
        self.c_funcs = c_funcs  # 约束函数
        self.mf_funcs = mf_funcs  # 多可信度函数
        self.bounds = bounds
        self.num_dims = bounds.shape[1]
        self.min_point = min_point
        self.min_value = min_value
        self.max_point = max_point
        self.max_value = max_value

    def __call__(self, x):
        return self.func(x)
