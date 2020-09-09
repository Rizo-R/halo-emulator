import glob
import numpy as np
import os
import pickle
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats
import sys


class HaloEmulator:

    def __init__(self, path='./', mass_type='M200c'):
        self.path = path
        self.mass_type = mass_type
        self.point_list = HaloEmulator.extract_data(path, mass_type)
        self.X, self.Y = HaloEmulator.convert_data(self.point_list)

    def extract_data(path, mass_type):
        filelist = glob.glob(os.path.join(path, 'dndm_' + mass_type + '*.pkl'))
        points = []
        for filename in filelist:
            with open(filename, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            points.append(HaloEmulator.reshape_data(data))
        return np.concatenate(points, axis=0)

    def convert_data(data):
        size = len(data)
        X = np.zeros((size, data[0].size-1))
        Y = np.zeros((size, 1))
        for i in range(size):
            X[i] = np.copy(data[i][:5])
            Y[i][0] = data[i][5]
        return (X, Y)

    def reshape_data(point_list):
        theta, a, m, counts = np.array(
            point_list[0]), point_list[1], point_list[2], point_list[3]
        z = 1/a - 1
        theta_reshaped = np.broadcast_to(theta, counts.shape + theta.shape)
        z_reshaped = np.moveaxis(np.broadcast_to(z, m.shape + z.shape), 1, 0,)
        m_reshaped = np.broadcast_to(m, z.shape + m.shape)
        back_half_of_array = np.stack((z_reshaped, m_reshaped, counts), axis=2)
        return np.concatenate((theta_reshaped, back_half_of_array), axis=2).reshape(-1, 6)


class RedshiftTester(HaloEmulator):

    def __init__(self, path='./', mass_type='M200c', M_low=0, M_high=None, n_chunks=None, redshift=None):
        super().__init__(path, mass_type)
        self.M_low = M_low
        self.M_high = M_high
        self.n_chunks = n_chunks
        self.redshift = redshift
        self.X, self.Y = RedshiftTester.set_limits(
            self.X, self.Y, self.M_low, self.M_high, self.n_chunks)

    def set_limits(X, Y, M_low=0, M_high=None, n_chunks=None):
        limits = []
        n = 0

        if M_high is None:
            limits = np.where((X[:, 4] >= M_low))[0]
            M_high = X[:, 4].max()
        else:
            limits = np.where((X[:, 4] >= M_low) & (X[:, 4] <= M_high))[0]

        try:
            n = np.multiply(n_chunks, int((M_high - M_low) / 0.05))
            if isinstance(n, np.int64):
                return(X[limits][:n], Y[limits][:n])
            elif isinstance(n, np.ndarray):
                assert n.shape == (2,), "[n_chunks] has to have size 2!"
                return(X[limits][n[0]:n[-1]], Y[limits][n[0]:n[-1]])
            else:
                raise IOError("Input mismatch!")
        except TypeError:
            if n_chunks is None:
                return(X[limits], Y[limits])
            else:
                raise IOError(
                    "[n] should be either NoneType, an integer, or a size-2 tuple!")


a = HaloEmulator()
b = RedshiftTester(M_low=12, M_high=16)
print("Input shape: ", b.X.shape)

n = 46
redshift = b.X[n*20, 3]
print("Redshift: ", redshift)

redshift_ind = np.where(b.X[:, 3] == redshift)
X = b.X[redshift_ind][:, :5]
Y = b.Y[redshift_ind]

filename = './covmat_M200c_mnu0.00000_om0.30000_As2.100020bins.pkl'

pic_in = open(filename, "rb")
cov_matrices = pickle.load(pic_in, encoding="latin1")

Mpiv = 1e14
likelihood_min = 1e8
C = cov_matrices[n]

idx = np.argwhere(np.all(C == 0, axis=0) | np.all(C == 0, axis=1))
for i in idx[::-1]:
    C = np.delete(C, i, axis=0)
    C = np.delete(C, i, axis=1)
print("Covariance matrix shape: ", C.shape)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


print("Covariance matrix is positive definite: ", is_pos_def(C))

if not is_pos_def(C):
    raise Exception("Covariance matrix not positive definite.")


def get_HMF_piecewise(p, **kwargs):

    M_arr_full = np.logspace(13, 15.5, 1001)
    ln_out_HMF = np.zeros(1001)
    res = []

    chunk_size = 50

    reg_bins = kwargs['reg_bins']
    offset = kwargs['offset']

    N_bin = reg_bins+1
    edge_der = np.zeros(N_bin+1)
    Mpiv = kwargs['Mpiv']
    ln_M_Mpiv = np.log(M_arr_full/Mpiv)

    if len(p) != reg_bins+3:
        print("Wrong number of params")
        return -np.inf

    # ln_HMF = a + b*np.log(M_arr_full) + c*np.log(M_arr_full)^2
    # dln_HMF/dlnM = b + 2*c*np.log(M_arr_full)
    # d^2ln_HMF/dlnM^2 = 2*c

    # Pivot bin: 4th bin
    a, b, c = p[0], p[1], p[5]
    idx_lo, idx_hi = 3*chunk_size+offset, 4*chunk_size+offset
    ln_out_HMF[idx_lo:idx_hi] = a + b * \
        ln_M_Mpiv[idx_lo:idx_hi] + c * ln_M_Mpiv[idx_lo:idx_hi]**2
    edge_der[3] = b + 2*c*ln_M_Mpiv[idx_lo]
    edge_der[4] = b + 2*c*ln_M_Mpiv[idx_hi]
    res.append((4, a, b, c, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    # Go "left"
    for i in range(0, 3)[::-1]:
        idx_lo = i*chunk_size+offset
        idx_hi = (i+1)*chunk_size+offset
        c = p[i+2]
        b = edge_der[i+1] - 2*c*ln_M_Mpiv[idx_hi+1]
        a = ln_out_HMF[idx_hi+1] - b * \
            ln_M_Mpiv[idx_hi+1] - c*ln_M_Mpiv[idx_hi+1]**2
        edge_der[i] = b + 2*c*ln_M_Mpiv[idx_lo]
        ln_out_HMF[idx_lo:idx_hi] = a + b * \
            ln_M_Mpiv[idx_lo:idx_hi] + c*ln_M_Mpiv[idx_lo:idx_hi]**2

        res.append((i+1, a, b, c, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    res.reverse()

    # Extend first bin (no curvature c=0)
    idx_lo = 0
    idx_hi = offset
    b = edge_der[0]
    a = ln_out_HMF[idx_hi+1] - b*ln_M_Mpiv[idx_hi+1]
    ln_out_HMF[idx_lo:idx_hi] = a + b*ln_M_Mpiv[idx_lo:idx_hi]

    if offset != 0:
        res.insert(0, (0, a, b, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    # Go "right"
    for i in range(4, N_bin-1):
        idx_lo = i*chunk_size+offset
        idx_hi = (i+1)*chunk_size+offset
        c = p[i+2]
        b = edge_der[i] - 2*c*ln_M_Mpiv[idx_lo]
        a = ln_out_HMF[idx_lo-1] - b * \
            ln_M_Mpiv[idx_lo-1] - c*ln_M_Mpiv[idx_lo-1]**2
        edge_der[i+1] = b + 2*c*ln_M_Mpiv[idx_hi]
        ln_out_HMF[idx_lo:idx_hi] = a + b * \
            ln_M_Mpiv[idx_lo:idx_hi] + c*ln_M_Mpiv[idx_lo:idx_hi]**2

        res.append((i+1, a, b, c, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    # Extend last bin with same curvature
    idx_lo = (N_bin-1)*chunk_size+offset
    idx_hi = len(ln_out_HMF)-1
    c = p[-1]
    b = edge_der[-2] - 2*c*ln_M_Mpiv[idx_lo]
    a = ln_out_HMF[idx_lo-1] - b*ln_M_Mpiv[idx_lo-1] - c*ln_M_Mpiv[idx_lo-1]**2
    edge_der[-1] = b + 2*c*ln_M_Mpiv[idx_hi]
    ln_out_HMF[idx_lo:idx_hi+1] = a + b * \
        ln_M_Mpiv[idx_lo:idx_hi+1] + c*ln_M_Mpiv[idx_lo:idx_hi+1]**2

    if idx_lo < idx_hi:
        res.append((N_bin, a, b, c, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    return (ln_out_HMF, res)


def f_polynomial(M, a, b, c):
    return a + b * np.log(M/Mpiv) + c * np.log(M/Mpiv)**2


def f_integrand(M, a, b, c):
    return 1e9 * (1/M) * np.e**(f_polynomial(M, a, b, c))


def integrate_dn_dlnM(param_list, zeroth_bin=False):
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
    global likelihood_min

    if params.size == 0:
        fname = "current_optimizer_n_" + str(n) + ".npy"
        params = np.load(fname)

    lambd = params[0]
    p = params[1:]
    _, hmf_piecewise_params = get_HMF_piecewise(
        p, reg_bins=19, Mpiv=Mpiv, offset=0)
    N_model = integrate_dn_dlnM(hmf_piecewise_params, zeroth_bin=False)
    print("var: ", likelihood_var(lambd, params[3:]))
    print("sim: ", likelihood_sim(Y, N_model, C, idx))

    res = likelihood_sim(Y, N_model, C, idx) + \
        likelihood_var(lambd, params[3:])

    if overwrite and res < likelihood_min:
        # global params_trial
        # params_trial = params
        likelihood_min = res
        np.save("current_optimizer", params)

    if not overwrite:
        print(N_model)

    cost_overrun = 0

    for c in params[2:]:
        if c > 0.:
            cost_overrun += (1000*c)**8

    if params[0] < 1:
        cost_overrun += (1000*(1-params[0]))**8

    return res + cost_overrun


arr = np.tile(np.linspace(-0.08,  -0.13, 4), 5)
params = np.concatenate((np.ones((3, 1)), np.reshape(arr, (-1, 1))), axis=0)
params[0] = 1.01
params[1] = -13
params[2] = -1.2

print(likelihood_total(params, overwrite=False))


x_max = []
x_max.append(100.)
x_max.append(20.)
x_max.append(20.)
for i in range(20):
    x_max.append(0.)

x_min = []
x_min.append(1.)
x_min.append(-20.)
x_min.append(-20.)
for i in range(20):
    x_min.append(-20.)


class MyBounds(object):
    def __init__(self, x_max=x_max, x_min=x_min):
        self.x_max = np.array(x_max)
        self.x_min = np.array(x_min)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        t_max = bool(np.all(x <= self.x_max))
        t_min = bool(np.all(x >= self.x_min))
        return t_max and t_min


bounds = MyBounds()

#q = np.load("current_optimizer.npy")

optimize_res = optimize.basinhopping(
    likelihood_total, params, disp=True, niter_success=3)

print("Done!")
