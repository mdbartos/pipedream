import numpy as np

def interpolate_sample(x, xp, fp):
    n = xp.shape[0]
    m = fp.shape[1]
    ix = np.searchsorted(xp, x)
    if (ix == 0):
        result = fp[0]
    elif (ix >= n):
        result = fp[n - 1]
    else:
        dx_0 = x - xp[ix - 1]
        dx_1 = xp[ix] - x
        frac = dx_0 / (dx_0 + dx_1)
        result = (1 - frac) * fp[ix - 1] + (frac) * fp[ix]
    return result

def bounded_newton_raphson(f, df, x0, x_lb, x_ub, args,
                           max_iter=1000, eps=1e-8):
    # 1. Initial steps
    # 1a.
    x = x0
    f_lx = f(x_lb, *args)
    f_ux = f(x_ub, *args)
    if (f_lx > f_ux):
        x_lb, x_ub = x_ub, x_lb
    # 1b.
    if (x < x_lb) or (x > x_ub):
        x = (x_lb + x_ub) / 2
    # 1c.
    dx = np.abs(x_ub - x_lb)
    # 1d.
    fx = f(x, *args)
    dfx = df(x, *args)
    # 2.
    for _ in range(max_iter):
        cond_0 = (((x - x_ub) * dfx - fx) * ((x - x_lb) * dfx - fx)) >= 0.
        cond_1 = (np.abs(2 * fx) > np.abs(dx * dfx))
        if (cond_0 or cond_1):
            dx = 0.5 * (x_ub - x_lb)
            x = x_lb + dx
        # 3.
        else:
            dx = fx / dfx
            x = x - dx
        # 4.
        if np.abs(dx) < eps:
            return x
        # 5.
        fx = f(x, *args)
        dfx = df(x, *args)
        if fx < 0:
            x_lb = x
        else:
            x_ub = x

def _kalman_semi_implicit(Z_next, P_x_k_k, A_1, A_2, b, H, C,
                          Qcov, Rcov):
    I = np.eye(A_1.shape[0])
    y_k1_k = b
    A_1_inv = np.linalg.inv(A_1)
    H_1 = H @ A_1_inv
    P_y_k1_k = A_2 @ P_x_k_k @ A_2.T + C @ Qcov @ C.T
    L_y_k1 = P_y_k1_k @ H_1.T @ np.linalg.inv((H_1 @ P_y_k1_k @ H_1.T) + Rcov)
    P_y_k1_k1 = (I - L_y_k1 @ H_1) @ P_y_k1_k
    b_hat = y_k1_k + L_y_k1 @ (Z_next - H_1 @ y_k1_k)
    P_x_k1_k1 = A_1_inv @ P_y_k1_k1 @ A_1_inv.T
    return b_hat, P_x_k1_k1

