import numpy as np
import torch
import os

# from htslibs.utils.load_data_from_json import load_config_from_json

# 获取参数配置
# params_config = load_data_from_json()
# device, dtype = torch.device(params_config["device"]), eval(params_config["dtype"])

def latin_hypercube_sample(num, dim, lb, ub):
    """
    Latin Hypercube Sampling
    :param num:
    :param dim:
    :param bounds:
    :return:
    """
    assert len(lb) == len(ub)
    assert len(lb) == dim
    assert len(ub) == dim

    # 单位空间上的拉丁超立方抽样
    unit_cube_points = np.zeros((num, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, num))
    centers = centers / float(2 * num)
    for i in range(0, dim):
        unit_cube_points[:, i] = centers[np.random.permutation(num)]

    perturbation = np.random.uniform(-1.0, 1.0, (num, dim))
    perturbation = perturbation / float(2 * num)
    unit_cube_points += perturbation

    # 将单位空间上的拉丁超立方抽样映射到实际空间
    points = unit_cube_points * (np.array(ub) - np.array(lb)) + np.array(lb)

    return points


def latin_hypercube_sample_tensor(num, lb, ub, dtype=torch.float32, device='cpu'):
    """
    Latin Hypercube Sampling using PyTorch (assuming lb and ub are tensors)
    :param num: Number of samples
    :param dim: Dimension of the samples
    :param lb: Lower bounds (torch.Tensor of shape (dim,))
    :param ub: Upper bounds (torch.Tensor of shape (dim,))
    :return: A tensor of shape (num, dim) with sampled points
    """
    assert lb.shape == ub.shape  # 确保 lb 和 ub 的形状一致
    # assert len(lb) == dim  # 确保维度匹配
    dim = len(lb)

    # 单位空间上的拉丁超立方采样
    unit_cube_points = torch.zeros((num, dim), dtype=dtype, device=device)
    centers = (1.0 + 2.0 * torch.arange(0.0, num, dtype=dtype, device=device)) / (2 * num)  # 生成中心点
    for i in range(dim):
        unit_cube_points[:, i] = centers[torch.randperm(num)]  # 随机排列

    # 生成扰动，并加入到中心点
    perturbation = torch.rand((num, dim), dtype=dtype, device=device) / (2 * num)
    unit_cube_points += perturbation

    # 将单位空间上的拉丁超立方采样映射到实际空间
    points = unit_cube_points * (ub - lb) + lb

    return points