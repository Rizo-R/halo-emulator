from IPython.display import display
import glob
import GPy
import errno
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats
import sys
import urllib.request as request
import yt

stdout = sys.stdout


class HaloEmulator:

    def __init__(self, path='/Users/rizo/Documents/ASTRO 4940/Halo_emu/', mass_type='M500c'):
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

    def GaussianProcesses(X, Y, num_dim=2):
        """
        Takes in an array of values in the range values_range and conducts a Gaussian Processes analysis on it.
        Displays and plots the result if plot is enabled.
        Returns a GPy model.
        """
        GPy.plotting.change_plotting_library('plotly_offline')

        kernel = GPy.kern.RBF(input_dim=num_dim, variance=1., lengthscale=1.)
        m = GPy.models.GPRegression(X, Y, kernel)
        # Create a text trap and redirect stdout to avoid unnecessary printing.
        text_trap = io.StringIO()
        sys.stdout = text_trap
        # Execute our now mute function.
        m.optimize_restarts(num_restarts=10)
        # Restore the stdout function.
        sys.stdout = stdout

        return m


class RedshiftTester(HaloEmulator):

    def __init__(self, path='/Users/rizo/Documents/ASTRO 4940/Halo_emu/', mass_type='M500c', M_low=0, M_high=None, n_chunks=None, redshift=None):
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

    def test_without_redshift(X, Y, redshift=None):
        redshift_removed = np.random.choice(
            X[:, 3]) if redshift is None else redshift
        indices = np.where(X[:, 3] == redshift_removed)[0]

        X_train = np.delete(X, indices, axis=0)
        Y_train = np.delete(Y, indices, axis=0)
        X_test = X[indices]
        Y_test = Y[indices]

        m = HaloEmulator.GaussianProcesses(X_train, Y_train, num_dim=5)
        Y_predict = m.predict(X_test)[0]

        fig = plt.figure()
        ax = plt.axes()

        plt.plot(X_test[:, 4], Y_test)
        plt.plot(X_test[:, 4], Y_predict)

        ax.set_title(
            'Predicted (Orange) vs Experimental (Blue) at z = {}'.format(redshift_removed))
        ax.set_xlabel('Galactic Halo Mass, $M_{\odot}$')
        ax.set_ylabel('N')

        plt.show()

        return (m, redshift, Y_test, Y_predict)


class ThetaTester(HaloEmulator):

    def __init__(self, path='/Users/rizo/Documents/ASTRO 4940/Halo_emu/', mass_type='M500c', M_low=0, M_high=None, redshift_lim=0., n_chunks=None, theta=np.ndarray(())):
        super().__init__(path, mass_type)
        self.M_low = M_low
        self.M_high = M_high
        self.redshift_lim = redshift_lim
        self.n_chunks = n_chunks
        self.theta = theta
        self.X, self.Y = ThetaTester.set_limits(
            self.X, self.Y, self.M_low, self.M_high, self.redshift_lim, self.n_chunks)

    def test_without_theta(X, Y, theta=None):
        theta_removed = ThetaTester.choose_random_theta(
            X) if theta is None else theta
        indices = np.where((X[:, 0] == theta_removed[0]) & (
            X[:, 1] == theta_removed[1]) & (X[:, 2] == theta_removed[2]))[0]

        X_train = np.delete(X, indices, axis=0)
        Y_train = np.delete(Y, indices, axis=0)
        X_test = X[indices]
        Y_test = Y[indices]

        m = HaloEmulator.GaussianProcesses(X_train, Y_train, num_dim=5)

        ThetaTester.plot_theta_single(X_test, Y_test, m)

    def plot_theta_single(X, Y, m, save_file=""):
        Y_predict = m.predict(X)[0]

        fig = plt.figure()
        ax = plt.axes()

        print(X.shape)
        print(X)
        print(Y.shape)
        print(Y)

        plt.plot(X[:, 4], Y)
        plt.plot(X[:, 4], Y_predict)

        ax.set_title('Mass Distribution, Predicted (Orange) vs Experimental (Blue) at $\Theta$ = ' +
                     str(X[0][0]) + ' ' + str(X[0][1]) + ' ' + str(X[0][2]))
        ax.set_xlabel('Galactic Halo Mass, $M_{\odot}$')
        ax.set_ylabel('N')

        plt.show()

    def plot_theta(X, Y, m, save_file=""):
        x_arrays, ind = ThetaTester.split_by_value(X, 0)
        y_arrays = ThetaTester.split_by_index(Y, ind)

        for i in range(len(x_arrays)):
            x = x_arrays[i]
            y = y_arrays[i]
            print(x.shape)
            print(y.shape)
            y_predict = m.predict(x)[0]

            fig = plt.figure()
            ax = plt.axes()

            plt.plot(x[:, 4], y)
            plt.plot(x[:, 4], y_predict)
        ax.set_title('Mass Distribution, Predicted (Orange) vs Experimental (Blue) at $\Theta$ = ' +
                     str(x[0][0]) + ' ' + str(x[0][1]) + ' ' + str(x[0][2]))
        ax.set_xlabel('Galactic Halo Mass, $M_{\odot}$')
        ax.set_ylabel('N')

        plt.show()
        if len(save_file) != 0:
            plt.savefig(save_file)

    def choose_random_theta(X):
        i = np.random.randint(0, X.shape[0])
        return X[i, 0:3]

    def set_limits(X, Y, M_low=0, M_high=None, redshift=0., n_chunks=None):
        limits = []
        n = 0

        if M_high is None:
            limits = np.where((X[:, 4] >= M_low) & (X[:, 3] == redshift))
        else:
            limits = np.where((X[:, 4] >= M_low) & (
                X[:, 4] <= M_high) & (X[:, 3] == redshift))

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
                    "[n_chunks] should be either NoneType, an integer, or a size-2 tuple!")


    def split_by_value(arr, n):
        ind = ThetaTester.value_indices(arr, n)
        return ThetaTester.split_by_index(arr, ind), ind

    def split_by_index(arr, ind):
        res = []
        if len(ind) == 1:
            return arr
        for i in range(len(ind)):
            if i == len(ind)-1:
                res.append(arr[ind[i]:])
            else:
                first = ind[i]
                last = ind[i+1]
                res.append(arr[first:last])
        return res

    def value_indices(arr, n):
        # Helper function for split_by_value().
        # Returns a list of indicies at the nth column of a non-empty array arr.
        curr = arr[0][n]

        res = [0]
        for i in range(arr.shape[0]):
            if arr[i][n] != curr:
                res.append(i)
                curr = arr[i][n]
        return res


a = HaloEmulator()
b = RedshiftTester(M_low=12, M_high=16)


def test_redshift(path='/Users/rizo/Documents/ASTRO 4940/Halo_emu/', mass_type='M500c', M_low=0, M_high=None, n_chunks=None, redshift=None, iterations=1):
    obj = RedshiftTester(path=path, mass_type=mass_type, M_low=M_low,
                         M_high=M_high, n_chunks=n_chunks, redshift=redshift)
    for i in range(iterations):
        _, _, _, _ = RedshiftTester.test_without_redshift(
            obj.X_limit, obj.Y_limit, obj.redshift)


def test_theta(path='/Users/rizo/Documents/ASTRO 4940/Halo_emu/', mass_type='M500c', M_low=0, M_high=None, redshift_lim=0., n_chunks=None, theta=None):
    obj = ThetaTester(path=path, mass_type=mass_type, M_low=M_low, M_high=M_high,
                      redshift_lim=redshift_lim, n_chunks=n_chunks, theta=theta)
    ThetaTester.test_without_theta(obj.X_limit, obj.Y_limit, obj.theta)


def get_HMF_piecewise(p, **kwargs):
    ''' res is a list of tuples containing (i, a_n, b_n, c_n, M_min_n, M_max_n)
    i is the index of the tuple (starts with 1); included for convenience.'''

    ln_out_HMF = np.zeros(4001)
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
    return 1e9 * (1/M) * np.e**f_polynomial(M, a, b, c)


def integrate_dn_dlnM(param_list, zeroth_bin=False):
    '''Assumes param_list[0] doesn't have a c-parameter.'''
    res = np.ones((len(param_list), 1))

    for i in range(res.size):
        # 0th bin integration
        if zeroth_bin and (i == 0):
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
    return (1/2) * D.T @ np.linalg.inv(C) @ D


def likelihood_var(lambd, c_arr):
    c_sum = 0
    for i in range(c_arr.size-1):
        c_sum += (c_arr[i] - c_arr[i+1])**2
    c_sum /= (2 * lambd**2)

    return (c_arr.size - 1) * np.log(lambd) + c_sum


def likelihood_total(params, overwrite=True):
    '''[lambd, a4, b4, c1, c2, c3,..., cN]'''
    lambd = params[0]
    p = params[1:]
    _, hmf_piecewise_params = get_HMF_piecewise(
        p, reg_bins=len(p)-3, Mpiv=Mpiv, offset=0)
    N_model = integrate_dn_dlnM(hmf_piecewise_params, zeroth_bin=False)
    # print(N_model)
    print("var: ", likelihood_var(lambd, params[3:]))
    print("sim: ", likelihood_sim(Y, N_model, C2, idx))
    # print(likelihood_sim(Y, N_model, C2) + likelihood_var(lambd, params[3:]))
    if overwrite:
        np.save("current_optimizer", params)
    return likelihood_sim(Y, N_model, C2, idx)

# https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def plot_params(params):
    print(likelihood_total(params, overwrite=False))
    _, res = get_HMF_piecewise(params[1:], reg_bins=79, Mpiv=1e14, offset=0)
    N_model = integrate_dn_dlnM(res, False)
    plt.loglog(np.logspace(12.025, 15.975, 80), N_model)
    plt.loglog(np.power(10, X[:, 4]), Y)
    plt.show()


filename = '/Users/rizo/Documents/ASTRO 4940/Halo_Emu_Cov_Mat/covmat_M200b_mnu0_00000_om0_30000_As2_1000.pkl'

pic_in = open(filename, "rb")
cov_matrices = pickle.load(pic_in, encoding="latin1")

C = cov_matrices[47]

idx = np.argwhere(np.all(C == 0, axis=0) | np.all(C == 0, axis=1))
for i in idx[::-1]:
    C = np.delete(C, i, axis=0)
    C = np.delete(C, i, axis=1)
print(C.shape)

C1 = nearestPD(C)
C2 = C[:63, :63]


n = 77600
indices = np.where((b.X[:, 3] == 0) & (b.X[:, 0] == b.X[n, 0]) & (
    b.X[:, 1] == b.X[n, 1]) & (b.X[:, 2] == b.X[n, 2]))
X = b.X[indices]
Y = b.Y[indices]


Mpiv = 1e14
n = 77600
M_arr_full = np.logspace(12, 16, 4001)

arr = np.tile(np.linspace(-0.1,  -0.12, 10), 8)
params = np.concatenate((np.ones((3, 1)), np.reshape(arr, (-1, 1))), axis=0)
params[0] = 1.01
params[1] = -13
params[2] = -1.5

print(likelihood_total(params))
a, res = get_HMF_piecewise(params[1:], reg_bins=79, Mpiv=1e14, offset=0)
plt.loglog(M_arr_full, np.exp(a))
plt.loglog(np.logspace(12, 16, 80), Y/(np.log(10) * 10**9))
plt.show()

# Constraints:
# lambd >= 1


def constraint1(params):
    lambd = params[0].item()
    return lambd - 1
# c_n <= 0


def constraint2(params):
    sum = 0
    for el in params[3:]:
        if el > 0:
            sum += el
    return sum


# Ignore the second constraint for now and use it as a boundary.
con1 = {'type': 'ineq', 'fun': constraint1}
# con2 = {'type': 'eq', 'fun': constraint2}
# cons = [con1, con2]

# Bounds:
# c_n <= 0
b_lambd = (None, None)
b_a = (None, None)
b_b = (None, None)
b_c = (-np.inf, 0.)
bnds = [b_lambd, b_a, b_b]
for i in range(80):
    bnds.append(b_c)


optimize_res = optimize.minimize(
    likelihood_total, params, method='SLSQP', bounds=bnds, constraints=con1)
print(optimize_res)
