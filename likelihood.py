import numpy as np
from scipy import integrate


Mpiv = 1e14


def f_polynomial(M, a, b, c):
    return a + b * np.log(M/Mpiv) + c * np.log(M/Mpiv)**2


def f_integrand(M, a, b, c):
    return 1e9 * (1/M) * np.exp(f_polynomial(M, a, b, c))


def integrate_params(param_list, zeroth_bin=False):
    res = np.ones((len(param_list), 1))

    for i in range(res.size):
        # 0th bin integration
        if zeroth_bin and i == 0:
            a, b, M_min, M_max = param_list[i][1:]
            res[i] = integrate.quad(
                f_integrand, M_min, M_max, args=(a, b, 0))[0]
        # i-th bin integration
        else:
            a, b, c, M_min, M_max = param_list[i][1:]
            res[i] = integrate.quad(
                f_integrand, M_min, M_max, args=(a, b, c))[0]

    return res


def likelihood_sim(N_sim, N_model, C, idx):
    assert N_sim.shape == N_model.shape
    assert C.shape[0] == C.shape[1]
    # assert len(idx) == N_sim.size - C[:, 0].size
    D = N_sim - N_model
#    print(" Product: ", D.T @ np.linalg.inv(C) @ D)
    for i in idx[::-1]:
        D = np.delete(D, i, axis=0)
    size = C.shape[0]
    D = D[:size]
    # print(D)
    # print(D.shape, C.shape)
    return (1/2) * D.T.dot(np.linalg.inv(C)).dot(D)


def likelihood_var(lambd, c_arr):
    c_sum = 0
    for i in range(c_arr.size-1):
        c_sum += (c_arr[i] - c_arr[i+1])**2
    c_sum /= (2 * lambd**2)

    return (c_arr.size - 1) * np.log(lambd) + c_sum


def likelihood_total(params, overwrite=True):
    '''[lambd, a4, b4, c1, c2, c3,..., cN]'''
    global toc
    global likelihood_min

    if params.size == 0 or params.shape == (1, 23):
        fname = "current_optimizer_" + m_nu + '_' + o_m + '_' + A_s + ".npy"
        params = np.load(fname)

    lambd = params[0]
    p = params[1:]
    print(p)
    _, hmf_piecewise_params = get_HMF_piecewise(
        p, reg_bins=19, Mpiv=Mpiv, offset=0)
    N_model = integrate(hmf_piecewise_params, zeroth_bin=False)
    print("var: ", likelihood_var(lambd, params[3:]))
    print("sim: ", likelihood_sim(Y_curr, N_model, C, idx))

    res = likelihood_sim(Y_curr, N_model, C, idx) + \
        likelihood_var(lambd, params[3:])

    if overwrite and res < likelihood_min:
        likelihood_min = res
        fname = "current_optimizer_" + m_nu + '_' + o_m + '_' + A_s + ".npy"
        np.save(fname, params)
        print("Saved to ", fname)

    cost_overrun = 0

    for c in params[2:]:
        if c > 0.:
            cost_overrun += (1000*c)**8

    if params[0] < 1:
        cost_overrun += (1000*(1-params[0]))**8

    if not overwrite:
        print(N_model)
        print("overrun: ", cost_overrun)

    # toc = time.perf_counter()
    return res + cost_overrun


# def likelihood_weights(weights):
#   M_logspace = np.logspace(13, 15.5, 1001)
#   vals = np.zeros((20,))
#   for i in range(20):
#       ind_lo = 50*i
#       ind_hi = 50*(i+1)+1
#       x = M_logspace[ind_lo:ind_hi]
#       dn_dlnM = np.e**(mean.flatten()[ind_lo:ind_hi] + pca.u[ind_lo:ind_hi, 0])
#       y = 1e9 * dn_dlnM / x
#       vals[i] = integrate.trapz(y, x)


#     _, hmf_piecewise_params = get_HMF_piecewise(
#         params[1:], reg_bins=19, Mpiv=Mpiv, offset=0)
#     N_model = integrate_params(hmf_piecewise_params, zeroth_bin=False)

#     assert N_sim.shape == N_model.shape
#     assert C.shape[0] == C.shape[1]

#     # assert len(idx) == N_sim.size - C[:, 0].size
#     D = N_sim - N_model
#     # print(" Product: ", D.T @ np.linalg.inv(C) @ D)
#     for i in idx[::-1]:
#         D = np.delete(D, i, axis=0)
#     size = C.shape[0]
#     D = D[:size]
#     # print(D)
#     # print(D.shape, C.shape)

#     return (1/2) * D.T.dot(np.linalg.inv(C)).dot(D)
