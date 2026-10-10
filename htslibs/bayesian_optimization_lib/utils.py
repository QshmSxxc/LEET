# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import numpy as np
import torch

from htslibs.utils.load_data_from_json import load_config_from_json

# 获取参数配置
params_config = load_config_from_json()
device, dtype = torch.device(params_config["device"]), eval(params_config["dtype"])


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    # assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(point, lb, ub):
    # assert np.all(lb < ub)
    # assert lb.ndim == 1
    # assert ub.ndim == 1
    assert point.ndim  == 2
    new_point = point * (ub - lb) + lb
    return new_point


def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points


class ToUnitCubeScaler_Torch:
    def __init__(self, lb_tensor=None, ub_tensor=None):
        self.lb = lb_tensor
        self.ub = ub_tensor

    def fit(self, tensor_data):
        if self.lb is None or self.ub is None:
            raise ValueError("请提供下界和上界")
        if (self.lb >= self.ub).any():
            raise ValueError("下界必须小于上界")

    def transform(self, tensor_data):
        return (tensor_data - self.lb) / (self.ub - self.lb)

    def inverse_transform(self, tensor_data):
        return tensor_data * (self.ub - self.lb) + self.lb

    def state_dict(self):
        return {
            "lb": self.lb.detach().cpu().clone(),
            "ub": self.ub.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict):
        self.lb = state_dict["lb"].to(device)
        self.ub = state_dict["ub"].to(device)

class MinMaxScaler_Torch:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None

    def fit(self, tensor_data):
        """
        计算并保存训练数据每个特征的最小值和最大值
        """
        self.data_min = tensor_data.min(dim=0).values  # 每一列的最小值
        self.data_max = tensor_data.max(dim=0).values  # 每一列的最大值
        # 对于最大值和最小值相等的列，将最小值设为 0，最大值设为 1
        zero_mask = self.data_max == self.data_min  # 找到那些最大值等于最小值的列
        self.data_min[zero_mask] = 0
        self.data_max[zero_mask] = 1

    def transform(self, tensor_data):
        """
        使用保存的最小值和最大值对新的数据进行归一化
        """
        # 归一化操作，避免除零错误
        # if self.data_max == self.data_min:
        #     normalized_data = tensor_data
        # else:
        if self.data_max is None or self.data_min is None:
            # 如果没有拟合数据，直接返回原始数据
            normalized_data = tensor_data
        else:
            normalized_data = (tensor_data - self.data_min) / (self.data_max - self.data_min)

        return normalized_data

    def inverse_transform(self, tensor_data):
        """
        反归一化，将归一化的数据恢复到原始数据的尺度
        """
        # 如果最小值和最大值相等，直接返回原始数据
        # if torch.all(self.data_max == self.data_min):
        #     return normalized_data
        if self.data_max is None or self.data_min is None:
            # 如果没有拟合数据，直接返回原始数据
            original_data = tensor_data
        else:
            # 反归一化操作
            original_data = tensor_data * (self.data_max - self.data_min) + self.data_min
        return original_data

    def fit_transform(self, tensor_data):
        """
        先调用 fit() 计算最小值和最大值，然后对数据进行归一化
        """
        self.fit(tensor_data)
        return self.transform(tensor_data)


class StandardScaler_Torch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor_data):
        self.mean = tensor_data.mean(dim=0)
        self.std = tensor_data.std(dim=0, unbiased=False)

        zero_std_mask = self.std < 1e-3
        self.std[zero_std_mask] = 1
        self.mean[zero_std_mask] = 0

        if tensor_data.shape[0] == 1:
            self.std = torch.ones_like(self.std)

    def transform(self, tensor_data):
        return (tensor_data - self.mean) / self.std

    def inverse_transform(self, tensor_data):
        return tensor_data * self.std + self.mean

    def fit_transform(self, tensor_data):
        self.fit(tensor_data)
        return self.transform(tensor_data)

    def state_dict(self):
        return {
            "mean": self.mean.detach().cpu().clone(),
            "std": self.std.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].to(device)
        self.std = state_dict["std"].to(device)