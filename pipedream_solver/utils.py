import numpy as np

def interpolate_sample(x, xp, fp, method=1):
    """
    Interpolate a sample `fp` over domain `xp` at point `x`.

    Inputs:
    -------
    x : float
        The x-coordinate at which to evaluate the interpolated value
    xp: np.ndarray (float)
        The x-coordinates of the data points
    fp: np.ndarray (float)
        The y-coordinates of the data points
    method: int [0 or 1]
        Use nearest neighbor (0) or linear (1) interpolation.
    """
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
        if method == 1:
            frac = dx_0 / (dx_0 + dx_1)
            result = (1 - frac) * fp[ix - 1] + (frac) * fp[ix]
        elif method == 0:
            if abs(dx_0) <= abs(dx_1):
                result = fp[ix - 1]
            else:
                result = fp[ix]
    return result

def bounded_newton_raphson(f, df, x0, x_lb, x_ub, args,
                           max_iter=1000, eps=1e-8):
    """
    Perform the Newton-Raphson iteration to find the zero of a function over a
    bounded interval.

    Inputs:
    -------
    f : function
        Function to evaluate
    df: function
        Derivative of the function to evaluate
    x0: float
        Initial estimate of the zero
    x_lb: float
        Lower bound
    x_ub: float
        Upper bound
    args: sequence
        Extra arguments to function `f`
    max_iter: int
        Maximum number of Newton-Raphson iterations
    eps: float
        Allowable error of the zero value
    """
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
    """
    Perform Kalman filtering to estimate state and error covariance.

    Inputs:
    -------
    Z_next : np.ndarray (b x 1)
        Observed data
    P_x_k_k : np.ndarray (M x M)
        Posterior error covariance estimate at previous timestep
    A_1 : np.ndarray (M x M)
        Left state transition matrix
    A_2 : np.ndarray (M x M)
        Right state transition matrix
    b : np.ndarray (M x 1)
        Right-hand side solution vector
    H : np.ndarray (M x b)
        Observation matrix
    C : np.ndarray (a x M)
        Signal-input matrix
    Qcov : np.ndarray (M x M)
        Process noise covariance
    Rcov : np.ndarray (M x M)
        Measurement noise covariance
    """
    I = np.eye(A_1.shape[0])
    A_1_inv = np.linalg.inv(A_1)
    
    x_k1_k = A_1_inv @ b
    P_x_k1_k = A_1_inv @ A_2 @ P_x_k_k @ A_2.T @ A_1_inv.T + C @ Qcov @ C.T
    L_x_k1 = P_x_k1_k @ H.T @ np.linalg.inv((H @ P_x_k1_k @ H.T) + Rcov)
    P_zz = (H @ P_x_k1_k @ H.T) + Rcov
    P_x_k1_k1 = (I - L_x_k1 @ H) @ P_x_k1_k
    x_hat = x_k1_k + L_x_k1 @ (Z_next - H @ x_k1_k)
    b_hat = A_1 @ x_hat
    return b_hat, P_x_k1_k1, P_zz

def _square_root_kalman_semi_implicit(Z_next, P_x_k_k, A_1, A_2, b, H, C,
                          Qcov, Rcov):
    """
    Perform Kalman filtering to estimate state and error covariance.
    Inputs:
    -------
    Z_next : np.ndarray (b x 1)
        Observed data
    P_x_k_k : np.ndarray (M x M)
        Posterior error covariance estimate at previous timestep
    A_1 : np.ndarray (M x M)
        Left state transition matrix
    A_2 : np.ndarray (M x M)
        Right state transition matrix
    b : np.ndarray (M x 1)
        Right-hand side solution vector
    H : np.ndarray (M x b)
        Observation matrix
    C : np.ndarray (a x M)
        Signal-input matrix
    Qcov : np.ndarray (M x M)
        Process noise covariance
    Rcov : np.ndarray (M x M)
        Measurement noise covariance
    """
    I = np.eye(A_1.shape[0])
    A_1_inv = np.linalg.inv(A_1)
    Rq = np.linalg.cholesky(Qcov)
    Rr = np.linalg.cholesky(Rcov)
    F = np.linalg.cholesky(P_x_k_k)
    
    x_k1_k = A_1_inv @ b
    Fbar = np.linalg.qr(np.vstack((F@A_2.T@A_1_inv.T, Rq)), mode='r')
    
    G = np.linalg.qr(np.block([[Fbar@H.T], [Rr]]), mode='r')
    
    L_x_k1 = (np.linalg.inv(G)@(np.linalg.inv(G).T@H)@Fbar.T@Fbar).T
    
    Fhat = np.linalg.qr(np.block([[Fbar@(I - L_x_k1@H).T], [Rr@L_x_k1.T]]), mode='r')
    x_hat = x_k1_k + L_x_k1 @ (Z_next - H @ x_k1_k)
    
    P_x_k1_k1 = Fhat.T@Fhat
    b_hat = A_1 @ x_hat
    return b_hat, P_x_k1_k1
