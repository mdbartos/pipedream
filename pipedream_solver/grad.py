import numpy as np

def jac_prod_Ik(model, Ik, ik, uk, dk, j):
    jac_prod = np.zeros(model._I.size)
    _I_internal = (~model._I_start) & (~model._I_end)
    jac_prod += model._E_Ik * Ik
    jac_prod[model._I_start] -= uk
    jac_prod[model._I_start] += ik[model._i_1k]
    jac_prod[model._I_end] -= ik[model._i_nk]
    jac_prod[model._I_end] += dk
    jac_prod[_I_internal] -= ik[model.backward_I_i[_I_internal]]
    jac_prod[_I_internal] += ik[model.forward_I_i[_I_internal]]
    return jac_prod

def jac_prod_ik(model, Ik, ik, uk, dk, j):
    jac_prod = np.zeros(model._i.size)
    _i_1k = model._i_1k
    _i_nk = model._i_nk
    _i_is_start = np.zeros(model._i.size, dtype=np.bool_)
    _i_is_start[_i_1k] = True
    _i_is_end = np.zeros(model._i.size, dtype=np.bool_)
    _i_is_end[_i_nk] = True
    _i_is_internal = (~_i_is_start) & (~_i_is_end)
    _im1 = (model._i - 1)[_i_is_internal]
    _ip1 = (model._i + 1)[_i_is_internal]
    g = 9.81
    jac_prod -= g * model._A_ik * (Ik[model._Ik] - Ik[model._Ip1k])
    jac_prod += model._b_ik * ik
    jac_prod[_i_is_start] += model._a_ik[_i_is_start] * uk
    jac_prod[_i_is_end] += model._c_ik[_i_is_end] * dk
    if _i_is_internal.any():
        jac_prod[_i_is_start] += model._c_ik[_i_is_start] * ik[_i_1k + 1]
        jac_prod[_i_is_end] += model._a_ik[_i_is_end] * ik[_i_nk - 1]
        jac_prod[_i_is_internal] += model._a_ik[_i_is_internal] * ik[_im1]
        jac_prod[_i_is_internal] += model._c_ik[_i_is_internal] * ik[_ip1]
    return jac_prod

# TODO: Error function in diagnostics needs to be corrected for theta
def jac_prod_uk(model, Ik, ik, uk, dk, j):
    g = 9.81
    jac_prod = (model._b_uk * uk + model._c_uk * ik[model._i_1k] 
                - g * model._A_uk * (model._theta_uk * j[model._J_uk] - Ik[model._I_1k]))
    return jac_prod

def jac_prod_dk(model, Ik, ik, uk, dk, j):
    g = 9.81
    jac_prod = (model._b_dk * dk + model._a_dk * ik[model._i_nk] 
                - g * model._A_dk * (Ik[model._I_Np1k] - model._theta_dk * j[model._J_dk]))
    return jac_prod

def jac_prod_j(model, Ik, ik, uk, dk, j):
    jac_prod = np.zeros(model.M)
    bc = model.bc
    dt = model._dt
    jac_prod += model.A_sj / dt
    np.add.at(jac_prod, model._J_uk, model._B_uk * model._dx_uk * model._theta_uk / 2 / dt)
    np.add.at(jac_prod, model._J_dk, model._B_dk * model._dx_dk * model._theta_dk / 2 / dt)
    jac_prod *= j
    np.add.at(jac_prod, model._J_uk, uk)
    np.subtract.at(jac_prod, model._J_dk, dk)
    #np.add.at(jac_prod, model._J_uo, o)
    #np.subtract.at(jac_prod, model._J_do, o)
    #np.add.at(jac_prod, model._J_uw, w)
    #np.subtract.at(jac_prod, model._J_dw, w)
    #np.add.at(jac_prod, model._J_up, p)
    #np.subtract.at(jac_prod, model._J_dp, p)
    jac_prod[bc] = j[bc]
    return jac_prod

def grad_Ik(model, err_Ik, err_ik, err_uk, err_dk, err_j):
    g = 9.81
    _is_internal = (~model._is_start) & (~model._is_end)
    _grad_Ik = np.zeros(err_Ik.size, dtype=np.float64)
    _grad_Ik[model._is_start] = (err_uk * g * model._A_uk) + (err_Ik[model._is_start] * model._E_Ik[model._is_start]) + (err_ik[model._link_start] * -g * model._A_ik[model._link_start])
    _grad_Ik[_is_internal] = (err_ik * g * model._A_ik)[~model._link_end] + (err_Ik[_is_internal] * model._E_Ik[_is_internal]) + (err_ik * -g * model._A_ik)[~model._link_start]
    _grad_Ik[model._is_end] = (err_ik * g * model._A_ik)[model._link_end] + (err_Ik[model._is_end] * model._E_Ik[model._is_end]) + (err_dk * -g * model._A_dk)
    return _grad_Ik

def grad_ik(model, err_Ik, err_ik, err_uk, err_dk, err_j):
    g = 9.81
    _grad_ik = np.zeros(err_ik.size, dtype=np.float64)
    _grad_ik[:] += (1 * err_Ik[model._Ik]) + (-1 * err_Ik[model._Ip1k])
    _grad_ik[:] += err_ik * model._b_ik
    _grad_ik[~model._link_end] += err_ik[~model._link_start] * model._a_ik[~model._link_start]
    _grad_ik[model._link_end] += err_dk * model._a_dk
    _grad_ik[~model._link_start] += err_ik[~model._link_end] * model._c_ik[~model._link_end]
    _grad_ik[model._link_start] += err_uk * model._c_uk
    return _grad_ik

def grad_uk(model, err_Ik, err_ik, err_uk, err_dk, err_j):
    g = 9.81
    _grad_uk = np.zeros(err_uk.size, dtype=np.float64)
    _grad_uk[:] = (1 * err_j)[model._J_uk] + (-1 * err_Ik)[model._is_start] + (err_uk * model._b_uk) + (err_ik * model._a_ik)[model._link_start]
    return _grad_uk

def grad_dk(model, err_Ik, err_ik, err_uk, err_dk, err_j):
    g = 9.81
    _grad_dk = np.zeros(err_dk.size, dtype=np.float64)
    _grad_dk[:] = (1 * err_Ik)[model._is_end] + (-1 * err_j)[model._J_dk] + (err_dk * model._b_dk) + (err_ik * model._c_ik)[model._link_end]
    return _grad_dk

def grad_j(model, err_Ik, err_ik, err_uk, err_dk, err_j, bc):
    g = 9.81
    _grad_j = np.zeros(err_j.size, dtype=np.float64)
    np.add.at(_grad_j, model._J_uk, err_j[model._J_uk] * ((model._A_sj[model._J_uk] + model._theta_uk * model._B_uk * model._dx_uk / 2) / model._dt) + (err_uk * (-g * model._A_uk * model._theta_uk)))
    np.add.at(_grad_j, model._J_dk, err_j[model._J_dk] * ((model._A_sj[model._J_dk] + model._theta_dk * model._B_dk * model._dx_dk / 2) / model._dt) + (err_dk * (g * model._A_dk * model._theta_dk)))
    _grad_j[bc] = err_j[bc]
    return _grad_j
