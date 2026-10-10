"""
翼型气动优化目标函数
接口与Levy/Schwefel等benchmark函数完全一致，可被BO-LEET直接调用
"""
import torch
import numpy as np
import neuralfoil as nf
from scipy.special import comb

from htslibs.objective_functions.obj_func_interface import obj_func, mf_function, noise, device, dtype

# ==================== 诊断 ====================
class FeasibilityStats:
    """可行性统计器（线程不安全，但单线程BO够用）"""

    def __init__(self):
        self.total_calls = 0
        self.feasible_calls = 0
        self.infeasibility_reasons = {}

    def record(self, is_feasible: bool, reason: str = "success"):
        self.total_calls += 1
        if is_feasible:
            self.feasible_calls += 1
        self.infeasibility_reasons[reason] = \
            self.infeasibility_reasons.get(reason, 0) + 1

    @property
    def feasibility_rate(self):
        if self.total_calls == 0:
            return 0.0
        return self.feasible_calls / self.total_calls

    def report(self):
        print(f"  Total calls: {self.total_calls}")
        print(f"  Feasible: {self.feasible_calls} "
              f"({self.feasibility_rate * 100:.1f}%)")
        print(f"  Reasons breakdown:")
        for reason, count in sorted(
                self.infeasibility_reasons.items(),
                key=lambda x: -x[1]
        ):
            print(f"    {reason}: {count}")

    def reset(self):
        self.total_calls = 0
        self.feasible_calls = 0
        self.infeasibility_reasons = {}


AF1_STATS = FeasibilityStats()
AF2_STATS = FeasibilityStats()
AF3_STATS = FeasibilityStats()


def reset_all_stats():
    AF1_STATS.reset()
    AF2_STATS.reset()
    AF3_STATS.reset()


def report_all_stats():
    print("\n" + "=" * 60)
    print("  可行性统计报告")
    print("=" * 60)
    print("AF1:")
    AF1_STATS.report()
    print()
    print("AF2:")
    AF2_STATS.report()
    print()
    print("AF3:")
    AF3_STATS.report()
    print("=" * 60)


# ==================== 配置常量 ====================
N_UPPER_COEFFS = 8
N_LOWER_COEFFS = 8
DESIGN_DIM = N_UPPER_COEFFS + N_LOWER_COEFFS  # 16

FIXED_LE_WEIGHT = 0.0
FIXED_TE_THICKNESS = 0.0

UPPER_LOW_VAL, UPPER_HIGH_VAL = 0.05, 0.30
LOWER_LOW_VAL, LOWER_HIGH_VAL = -0.30, -0.05

NF_MODEL_SIZE = "medium"
NF_CONFIDENCE_THRESHOLD = 0.5

THICKNESS_MIN = 0.05
THICKNESS_MAX = 0.20
CAMBER_MAX = 0.08
N_GEO_CHECK_POINTS = 51

penalty_dict = {
    'AF1': {
        'PENALTY_GEOM_SEVERE':   -30.0,
        'PENALTY_GEOM_MODERATE': -25.0,
        'PENALTY_GEOM_MINOR':    -20.0,
        'PENALTY_AERO_FAIL':     -27.0,
    },
    'AF2': {
        'PENALTY_GEOM_SEVERE':   600.0,
        'PENALTY_GEOM_MODERATE': 400.0,
        'PENALTY_AERO_FAIL':     300.0,
        'PENALTY_GEOM_MINOR':    200.0,
    },
    'AF3': {
        'PENALTY_GEOM_SEVERE':   -45.0,
        'PENALTY_GEOM_MODERATE': -30.0,
        'PENALTY_AERO_FAIL':     -25.0,
        'PENALTY_GEOM_MINOR':    -15.0,
    },
}

# 每个场景对应的 stats 对象，方便 _core 函数内部统一查找
AIRFOIL_PENALTY_LAMDA = 2.0


def set_airfoil_penalty_lamda(lamda):
    global AIRFOIL_PENALTY_LAMDA
    lamda = float(lamda)
    if lamda < 0:
        raise ValueError(f"AIRFOIL_PENALTY_LAMDA must be non-negative, got {lamda}")
    AIRFOIL_PENALTY_LAMDA = lamda


def get_airfoil_penalty_lamda():
    return AIRFOIL_PENALTY_LAMDA


def _scaled_penalty(scene, penalty_key):
    return AIRFOIL_PENALTY_LAMDA * penalty_dict[scene][penalty_key]


_STATS_MAP = {
    'AF1': AF1_STATS,
    'AF2': AF2_STATS,
    'AF3': AF3_STATS,
}


# ==================== 内部工具函数 ====================

def _to_scalar(value, default=None):
    if value is None:
        return default
    try:
        arr = np.asarray(value, dtype=float).ravel()
        if len(arr) == 0:
            return default
        return float(arr[0])
    except Exception:
        return default


def _design_vars_to_kulfan(x_np):
    return {
        "upper_weights": x_np[:N_UPPER_COEFFS].copy(),
        "lower_weights": x_np[N_UPPER_COEFFS:].copy(),
        "leading_edge_weight": FIXED_LE_WEIGHT,
        "TE_thickness": FIXED_TE_THICKNESS,
    }


def _manual_cst_to_coords(kulfan_params, n_points=51):
    upper_w = np.asarray(kulfan_params["upper_weights"])
    lower_w = np.asarray(kulfan_params["lower_weights"])
    n_order_u = len(upper_w) - 1
    n_order_l = len(lower_w) - 1

    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    C = (x ** 0.5) * ((1 - x) ** 1.0)

    def shape(weights, n):
        S = np.zeros_like(x)
        for i, w in enumerate(weights):
            S += w * comb(n, i) * (x ** i) * ((1 - x) ** (n - i))
        return S

    y_upper = C * shape(upper_w, n_order_u)
    y_lower = C * shape(lower_w, n_order_l)

    return x, y_upper, y_lower


def _check_geometry(kulfan_params):
    """几何检查 → (is_valid, violation_degree)"""
    try:
        x, y_upper, y_lower = _manual_cst_to_coords(
            kulfan_params, n_points=N_GEO_CHECK_POINTS
        )
    except Exception:
        return False, 1.0

    if np.any(y_upper < y_lower - 1e-6):
        return False, 1.0

    thickness = y_upper - y_lower
    max_t = np.max(thickness)
    if max_t < THICKNESS_MIN or max_t > THICKNESS_MAX:
        return False, 1.0

    camber = 0.5 * (y_upper + y_lower)
    if np.max(np.abs(camber)) > CAMBER_MAX:
        return False, 0.6

    if len(y_upper) >= 5:
        d2u = np.diff(y_upper, 2)
        if np.sum(np.diff(np.sign(d2u + 1e-12)) != 0) > 4:
            return False, 0.3
    if len(y_lower) >= 5:
        d2l = np.diff(y_lower, 2)
        if np.sum(np.diff(np.sign(d2l + 1e-12)) != 0) > 4:
            return False, 0.3

    return True, 0.0


def _get_geom_penalty(violation, scene="AF1"):
    if violation <= 0.0:
        return 0.0
    elif violation <= 0.35:
        return _scaled_penalty(scene, 'PENALTY_GEOM_MINOR')
    elif violation <= 0.65:
        return _scaled_penalty(scene, 'PENALTY_GEOM_MODERATE')
    else:
        return _scaled_penalty(scene, 'PENALTY_GEOM_SEVERE')


def _get_geom_reason(violation):
    """将 violation 程度转为 reason 字符串，供 stats 记录使用"""
    if violation <= 0.35:
        return "geom_minor"
    elif violation <= 0.65:
        return "geom_moderate"
    else:
        return "geom_severe"


def _evaluate_aero(kulfan_params, alpha, Re):
    try:
        result = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan_params,
            alpha=alpha,
            Re=Re,
            model_size=NF_MODEL_SIZE,
        )
    except Exception:
        return None

    CL = _to_scalar(result.get("CL"))
    CD = _to_scalar(result.get("CD"))
    if CL is None or CD is None or np.isnan(CL) or np.isnan(CD) or CD <= 1e-7:
        return None

    conf = _to_scalar(result.get("analysis_confidence"), default=1.0)
    if conf is not None and conf < NF_CONFIDENCE_THRESHOLD:
        return None

    return {"CL": CL, "CD": CD, "L_over_D": CL / CD}


def _find_alpha_for_CL(kulfan_params, CL_target, Re,
                       alpha_range=(-3.0, 12.0), tol=0.02, max_iter=12):
    a_low, a_high = alpha_range
    r_low = _evaluate_aero(kulfan_params, a_low, Re)
    r_high = _evaluate_aero(kulfan_params, a_high, Re)
    if r_low is None or r_high is None:
        return None
    if not (r_low["CL"] - tol <= CL_target <= r_high["CL"] + tol):
        return None

    r_mid = None
    for _ in range(max_iter):
        a_mid = 0.5 * (a_low + a_high)
        r_mid = _evaluate_aero(kulfan_params, a_mid, Re)
        if r_mid is None:
            return None
        if abs(r_mid["CL"] - CL_target) < tol:
            return r_mid
        if r_mid["CL"] < CL_target:
            a_low = a_mid
        else:
            a_high = a_mid
    return r_mid


# ==================== 单样本核心评估函数 ====================

def _af1_core(x_np):
    """AF-1: 最大化 L/D @ alpha=2°, Re=1e6"""
    scene = 'AF1'
    stats = _STATS_MAP[scene]

    if np.any(np.isnan(x_np)) or np.any(np.isinf(x_np)):
        stats.record(False, "sanity_failed")
        return _scaled_penalty(scene, 'PENALTY_AERO_FAIL')

    kp = _design_vars_to_kulfan(x_np)
    is_valid, viol = _check_geometry(kp)
    if not is_valid:
        stats.record(False, _get_geom_reason(viol))
        return _get_geom_penalty(viol, scene)

    r = _evaluate_aero(kp, alpha=2.0, Re=1e6)
    if r is None:
        stats.record(False, "aero_failed")
        return _scaled_penalty(scene, 'PENALTY_AERO_FAIL')

    stats.record(True, "success")
    return r["L_over_D"]


def _af2_core(x_np):
    """AF-2: 在CL=0.5约束下最小化CD, Re=1e6"""
    scene = 'AF2'
    stats = _STATS_MAP[scene]  # ← 修复：补全 stats 记录

    if np.any(np.isnan(x_np)) or np.any(np.isinf(x_np)):
        stats.record(False, "sanity_failed")
        return _scaled_penalty(scene, 'PENALTY_AERO_FAIL')

    kp = _design_vars_to_kulfan(x_np)
    is_valid, viol = _check_geometry(kp)
    if not is_valid:
        stats.record(False, _get_geom_reason(viol))
        return _get_geom_penalty(viol, scene)

    r = _find_alpha_for_CL(kp, CL_target=0.5, Re=1e6)
    if r is None:
        stats.record(False, "aero_failed")
        return _scaled_penalty(scene, 'PENALTY_AERO_FAIL')

    stats.record(True, "success")
    return r["CD"] * 1e4


_AF3_OPS = [
    {"alpha": 6.0, "Re": 5e5, "weight": 0.25},
    {"alpha": 3.0, "Re": 2e6, "weight": 0.50},
    {"alpha": 1.0, "Re": 5e6, "weight": 0.25},
]


def _af3_core(x_np):
    """AF-3: 三工况加权 L/D 最大化"""
    scene = 'AF3'
    stats = _STATS_MAP[scene]  # ← 修复：补全 stats 记录

    if np.any(np.isnan(x_np)) or np.any(np.isinf(x_np)):
        stats.record(False, "sanity_failed")
        return _scaled_penalty(scene, 'PENALTY_AERO_FAIL')

    kp = _design_vars_to_kulfan(x_np)
    is_valid, viol = _check_geometry(kp)
    if not is_valid:
        stats.record(False, _get_geom_reason(viol))
        return _get_geom_penalty(viol, scene)

    weighted = 0.0
    for op in _AF3_OPS:
        r = _evaluate_aero(kp, alpha=op["alpha"], Re=op["Re"])
        if r is None:
            stats.record(False, "aero_failed")
            return _scaled_penalty(scene, 'PENALTY_AERO_FAIL')
        weighted += op["weight"] * (r["L_over_D"])

    stats.record(True, "success")
    return weighted


# ==================== 三个目标函数（与Levy同接口） ====================

def Airfoil_AF1():
    """
    翼型气动优化 - 场景1: 单工况升阻比最大化
    设计变量: 16维 (Kulfan CST 上下表面各8系数)
    工况: alpha=2°, Re=1e6
    优化目标: 最大化 L/D
    """
    input_dim_num = DESIGN_DIM
    func_name = 'Airfoil_AF1'

    def function(x):
        assert x.shape[-1] == input_dim_num, \
            f"Input dimension must be {input_dim_num}!"

        n, d = x.size()
        x_np_batch = x.detach().cpu().numpy()

        results = np.zeros(n, dtype=np.float64)
        for i in range(n):
            results[i] = _af1_core(x_np_batch[i])

        result = torch.tensor(
            results, dtype=dtype, device=device
        ).reshape(-1, 1)

        return result + torch.randn_like(result) * noise

    bounds = torch.stack((
        torch.tensor(
            [UPPER_LOW_VAL] * N_UPPER_COEFFS + [LOWER_LOW_VAL] * N_LOWER_COEFFS,
            dtype=dtype, device=device,
        ),
        torch.tensor(
            [UPPER_HIGH_VAL] * N_UPPER_COEFFS + [LOWER_HIGH_VAL] * N_LOWER_COEFFS,
            dtype=dtype, device=device,
        ),
    ), dim=0)

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


def Airfoil_AF2():
    """
    翼型气动优化 - 场景2: 约束阻力最小化
    设计变量: 16维
    约束: CL = 0.5
    工况: Re=1e6
    优化目标: 最小化 CD * 1e4（返回 -CD*1e4 以转为最大化）
    """
    input_dim_num = DESIGN_DIM
    func_name = 'Airfoil_AF2'

    def function(x):
        assert x.shape[-1] == input_dim_num, \
            f"Input dimension must be {input_dim_num}!"

        n, d = x.size()
        x_np_batch = x.detach().cpu().numpy()

        results = np.zeros(n, dtype=np.float64)
        for i in range(n):
            results[i] = _af2_core(x_np_batch[i])
            # results[i] = -af2_min_obj  # 转为最大化目标

        result = torch.tensor(
            results, dtype=dtype, device=device
        ).reshape(-1, 1)

        return -1*result + torch.randn_like(result)*noise

    bounds = torch.stack((
        torch.tensor(
            [UPPER_LOW_VAL] * N_UPPER_COEFFS + [LOWER_LOW_VAL] * N_LOWER_COEFFS,
            dtype=dtype, device=device,
        ),
        torch.tensor(
            [UPPER_HIGH_VAL] * N_UPPER_COEFFS + [LOWER_HIGH_VAL] * N_LOWER_COEFFS,
            dtype=dtype, device=device,
        ),
    ), dim=0)

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


def Airfoil_AF3():
    """
    翼型气动优化 - 场景3: 多工况加权升阻比最大化
    设计变量: 16维
    工况: 3个加权工况 (climb / cruise / dash)
    优化目标: 最大化 加权 L/D
    """
    input_dim_num = DESIGN_DIM
    func_name = 'Airfoil_AF3'

    def function(x):
        assert x.shape[-1] == input_dim_num, \
            f"Input dimension must be {input_dim_num}!"

        n, d = x.size()
        x_np_batch = x.detach().cpu().numpy()

        results = np.zeros(n, dtype=np.float64)
        for i in range(n):
            results[i] = _af3_core(x_np_batch[i])
            # results[i] = -af3_min_obj  # 转为最大化目标

        result = torch.tensor(
            results, dtype=dtype, device=device
        ).reshape(-1, 1)

        return result + torch.randn_like(result) * noise

    bounds = torch.stack((
        torch.tensor(
            [UPPER_LOW_VAL] * N_UPPER_COEFFS + [LOWER_LOW_VAL] * N_LOWER_COEFFS,
            dtype=dtype, device=device,
        ),
        torch.tensor(
            [UPPER_HIGH_VAL] * N_UPPER_COEFFS + [LOWER_HIGH_VAL] * N_LOWER_COEFFS,
            dtype=dtype, device=device,
        ),
    ), dim=0)

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
