import glob
import numpy as np
import os
import pickle
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats
import sys
import time


from emulator import *
from HMF_piecewise import *
from likelihood import *
from pca import *


a = HaloEmulator()
b = RedshiftTester(M_low=12, M_high=16)


def sort_data(X, Y):
    ''' Sorts input by mu_n, in ascending order.'''
    # Concatenate X and Y to make sorting easier.
    M = np.concatenate((X, Y), axis=1)
    M_new = M[M[:, 0].argsort()]

    # Sorting by mu_n messed up the order within each cosmology -
    # mass is no longer sorted in ascending order. This is fixed in the
    # for-loop below.
    for i in range(M_new.shape[0]//20):
        idx_lo = 20*i
        idx_hi = 20*(i+1)
        chunk = M_new[idx_lo:idx_hi]
        M_new[idx_lo:idx_hi] = chunk[chunk[:, 4].argsort()]

    X_new = M_new[:, :5]
    Y_new = np.expand_dims(M_new[:, 5], axis=1)

    return X_new, Y_new


Mpiv = 1e14

redshift = 0.
redshift_ind = np.where(b.X[:, 3] == redshift)
X_unsorted = b.X[redshift_ind]
Y_unsorted = b.Y[redshift_ind]
X, Y = sort_data(X_unsorted, Y_unsorted)


def remove_zeros(C):
    idx = np.argwhere(np.all(C == 0, axis=0) | np.all(C == 0, axis=1))
    for i in idx[::-1]:
        C = np.delete(C, i, axis=0)
        C = np.delete(C, i, axis=1)
    # print("Covariance matrix shape: ", C.shape)
    return C, idx


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def remove_positives(params):
    # Remove positive entries for the c-parameter.
    # ind_h, ind_v = np.where(params[:, 1:] > 0)
    # params[ind_h, ind_v+1] = 0
    ind = np.where(params[1:] > 0)[0]
    params[ind+1] = 0
    return params


def set_initial_guess():
    arr = np.tile(np.linspace(-0.08,  -0.13, 4), 5)
    params = np.concatenate((np.ones((3, 1)), arr[:, np.newaxis]), axis=0)
    params[0] = 1.01
    params[1] = -13
    params[2] = -1.2

    return params


def load_cov_matrices(path):
    pic_in = open(path, "rb")
    cov_matrices = pickle.load(pic_in, encoding="latin1")

    return cov_matrices

# (1, 2, 3, 4, 5, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 25, 26)


bnds = [(1., 100.), (-100., 100.), (-100., 100.)]
for i in range(20):
    bnds.append((-20., 20.))

for n in range(Y.size//20):
    if True:
        continue

    if n not in (2, 3, 4, 5, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 25, 26):
        continue

    likelihood_min = 1e8
    params = set_initial_guess()

    X_curr = X[20*n:20*(n+1)]
    Y_curr = Y[20*n:20*(n+1)]
    # Round to 5 decimal places; add decimal places if needed.
    m_nu = "{:.5f}".format(X_curr[0, 0])
    o_m = "{:.5f}".format(X_curr[0, 1])
    A_s = "{:.4f}".format(X_curr[0, 2])

    path = './covmat/covmat_M200c_mnu' + m_nu + \
        '_om' + o_m + '_As' + A_s + '20bins.pkl'
    cov_matrices = load_cov_matrices(path)
    C = cov_matrices[-1]
    C, idx = remove_zeros(C)

    # print(likelihood_total(params, False))

    if C.size == 0:
        countinue

    print("Covariance matrix is positive definite: ", is_pos_def(C))

    if not is_pos_def(C):
        raise Exception("Covariance matrix not positive definite.")

    fname = "current_optimizer_" + m_nu + "_" + o_m + "_" + A_s + ".npy"
    path = "./"
    np.save(fname, params)

    # Multiple iterations to achieve the accuracy needed.
    for i in range(4):
        params = np.load(fname)
        try:
            optimizer = optimize.basinhopping(likelihood_total, params)
        except ValueError:
            pass

    optimized = np.load(path + fname)
    optimized = remove_positives(optimized)

    print(optimized.shape)

    print("Finished iteration ", n)
    print("Total likelihood: ", likelihood_total(optimized, overwrite=False))
    # print("Time: ", str(toc - tic))
    print("\n")


def pca_refit(weights):
    # Use the first four eigenvectors
    res = np.zeros((pca.basis_vectors.shape[0],))
    for i in range(4):
        res += weights[i] * pca.basis_vectors[:, i]
    return likelihood_weights(res, overwrite=True)
