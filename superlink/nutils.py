import numpy as np
from numba import njit

@njit
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

@njit
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

