# Author: Andrew Janke
# Created on: 2024-06-17
# Description: 复用自pyswarm的粒子群优化算法(from pyswarm import pso)，修改初始化部分
import numpy as np

def pso_x1(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
        minstep=1e-8, minfunc=1e-8, debug=False,
        init_pos=None, # 先验中心点；如果为 None，则完全随机初始化
        init_radius=None, # 局部初始化半径；None 时默认取 0.05 * (ub - lb)
        local_frac=0.3 # 局部初始化粒子的比例，建议 0.2 ~ 0.4
        ):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)

    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified,
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    init_pos : array or None
        先验中心点；如果为 None，则完全随机初始化
    init_radius : scalar or array or None
        局部初始化半径；None 时默认取 0.05 * (ub - lb)
    local_frac : float
        局部初始化粒子的比例，建议 0.2 ~ 0.4

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    """

    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Check for constraint function(s) #########################################
    obj = lambda x: func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = lambda x: np.array([0])
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = lambda x: np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = lambda x: np.array(f_ieqcons(x, *args, **kwargs))

    def is_feasible(x):
        check = np.all(cons(x) >= 0)
        return check

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has

    if init_pos is None:
        # 完全随机初始化
        x = np.random.rand(S, D)
        x = lb + x * (ub - lb)
    else:
        # 保守版：局部 + 全局 混合初始化
        init_pos = np.array(init_pos, dtype=float)
        assert len(init_pos) == D, 'init_pos must have the same dimension as lb and ub'
        init_pos = np.clip(init_pos, lb, ub)

        if init_radius is None:
            init_radius = 0.05 * (ub - lb)
        else:
            init_radius = np.array(init_radius, dtype=float)
            if init_radius.ndim == 0:
                init_radius = np.ones(D) * init_radius
            assert len(init_radius) == D, 'init_radius must be scalar or have the same dimension as lb and ub'

        init_radius = np.maximum(init_radius, 1e-12)

        n_local = max(1, int(S * local_frac))
        n_global = S - n_local

        # 局部粒子：围绕先验点
        x_local = init_pos + (2 * np.random.rand(n_local, D) - 1) * init_radius
        x_local = np.clip(x_local, lb, ub)

        # 全局粒子：保持探索能力
        if n_global > 0:
            x_global = np.random.rand(n_global, D)
            x_global = lb + x_global * (ub - lb)
            x = np.vstack([x_local, x_global])
        else:
            x = x_local

        # 第一个粒子直接放在先验点
        x[0, :] = init_pos

    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)      # best particle function values
    g = []                # best swarm position
    fg = 1e100            # artificial best swarm position starting value

    for i in range(S):
        # Initialize the particle's best known position
        p[i, :] = x[i, :]

        # Calculate the objective's value at the current particle's
        fp[i] = obj(p[i, :])

        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        if i == 0:
            g = p[0, :].copy()

        # If the current particle's position is better than the swarm's,
        # update the best swarm position
        if fp[i] < fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()

        # Initialize the particle's velocity
        v[i, :] = vlow + np.random.rand(D) * (vhigh - vlow)

    # Iterate until termination criterion met ##################################
    it = 1
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        for i in range(S):

            # Update the particle's velocity
            v[i, :] = omega * v[i, :] + phip * rp[i, :] * (p[i, :] - x[i, :]) + \
                      phig * rg[i, :] * (g - x[i, :])

            # Update the particle's position, correcting lower and upper bound
            # violations, then update the objective function value
            x[i, :] = x[i, :] + v[i, :]
            mark1 = x[i, :] < lb
            mark2 = x[i, :] > ub
            x[i, mark1] = lb[mark1]
            x[i, mark2] = ub[mark2]
            fx = obj(x[i, :])

            # Compare particle's best position (if constraints are satisfied)
            if fx < fp[i] and is_feasible(x[i, :]):
                p[i, :] = x[i, :].copy()
                fp[i] = fx

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx < fg:
                    if debug:
                        print('New best for swarm at iteration {:}: {:} {:}'.format(it, x[i, :], fx))

                    tmp = x[i, :].copy()
                    stepsize = np.sqrt(np.sum((g - tmp) ** 2))
                    if np.abs(fg - fx) <= minfunc:
                        print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                        return tmp, fx
                    elif stepsize <= minstep:
                        print('Stopping search: Swarm best position change less than {:}'.format(minstep))
                        return tmp, fx
                    else:
                        g = tmp.copy()
                        fg = fx

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))

    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    return g, fg