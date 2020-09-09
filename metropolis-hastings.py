import glob
import numpy as np
import os
import pickle
import scipy.integrate as integrate
import scipy.linalg
import scipy.optimize as optimize
import scipy.stats as stats
import sys
import time


from emulator import *
from HMF_piecewise import *
from likelihood import *
from pca import *


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


a = HaloEmulator()
b = RedshiftTester(M_low=12, M_high=16)
print("Input shape: ", b.X.shape)

redshift = 0.
redshift_ind = np.where(b.X[:, 3] == redshift)
X = b.X[redshift_ind]
Y = b.Y[redshift_ind]
X, Y = sort_data(X, Y)

filelist = glob.glob('./hmf_params/*.npy')
filelist.sort()
filelist

# Params is a numpy array of shape (27, 23).
params = np.ones((len(filelist), 23))
for i in range(params.shape[0]):
    params[i] = np.load(filelist[i])


ind_h, ind_v = np.where(params[:, 1:] > 0)
params[ind_h, ind_v+1] = 0

Mpiv = 1e14

Y_fit = np.ones((1001, params.shape[0]))

for i in range(params.shape[0]):
    Y_fit[:, i], _ = get_HMF_piecewise(
        params[i][1:], reg_bins=19, offset=0, Mpiv=1e14)

HMF_mean = np.average(Y_fit, axis=1).reshape(-1, 1)
Y_res = Y_fit - np.broadcast_to(HMF_mean, (HMF_mean.size, len(filelist)))
pca = Pca.calculate(Y_res)


var_sum = 0
for i in range(4):
    var_sum += pca.explained_variance[i]
explained_percentage = 100 * var_sum / np.sum(pca.explained_variance)
print("The first %d basis vectors explain %f %% of the variance." %
      (i+1, explained_percentage))


def fit_weights(weights, evectors):
    '''Returns HMF that is based on the given N weights and the first N 
    given eigenvectors. Returns a Numpy array.
    Prerequisites: len(evectors) >= len(weights).'''
    res = np.zeros((1001,))
    for i in range(len(weights)):
        res += weights[i] * evectors[:, i]
    return res


def integrate_dn_dlnM(HMF):
    '''Integrates the given HMF for a given cosmology. The cosmology is 
    provided by the number in the (sorted) list of cosmologies, starting at 
    [0., 0.3, 2.1].'''
    # 20 bins.
    M_logspace = np.logspace(13, 15.5, 1001)
    vals = np.zeros((20, 1))
    for i in range(20):
        ind_lo = 50*i
        ind_hi = 50*(i+1)+1
        x = M_logspace[ind_lo:ind_hi]
        dn_dlnM = np.exp(HMF_mean.flatten()
                         [ind_lo:ind_hi] + HMF[ind_lo:ind_hi])
        y = 1e9 * dn_dlnM / x
        vals[i] = integrate.trapz(y, x)
    return vals


def load_cov_matrices(path):
    pic_in = open(path, "rb")
    cov_matrices = pickle.load(pic_in, encoding="latin1")

    return cov_matrices


def remove_zeros(C):
    idx = np.argwhere(np.all(C == 0, axis=0) | np.all(C == 0, axis=1))
    for i in idx[::-1]:
        C = np.delete(C, i, axis=0)
        C = np.delete(C, i, axis=1)
    print("Covariance matrix shape: ", C.shape)
    return C, idx


def flip_coin(r):
    u = np.random.uniform(0, 1)
    if r > 1 or r > u:
        return True
    else:
        return False


def log_prior(weights):
    for w in weights:
        if -500. < w < 500.:
            return 0.
    return -np.inf


def log_P(weights, pca, y, C, idx):
    HMF = fit_weights(weights, pca.u[:, :3])
    N = integrate_dn_dlnM(HMF)

    lp = log_prior(weights)
    if not np.isfinite(lp):
        return -np.inf
    return lp + likelihood_sim(N, y, C, idx)


def mcmc_step(N_hops, pca, std_dev=0.5, n=0):
    '''My attempt to run MCMC using the Metropolis-Hastings algorithm. Returns 
    the last 1000 states in the chain.'''
    # Load the covariance matrix.
    filelist_mat = glob.glob("./covmat/covmat_M200c_*.pkl")
    filelist_mat.sort()
    filelist_mat
    cov_matrices = load_cov_matrices(filelist_mat[n])[-1]

    states = np.zeros((N_hops, 4))
    acc = 0
    tot = 0
    cur = []
    C, idx = remove_zeros(cov_matrices)

    # Create a random initial state.
    for i in range(4):
        cur.append(np.random.uniform(0, 400))

    for i in range(N_hops):
        tot += 1
        states[i] = cur

        # Initialize the next state, based on the current state.
        next_step = []
        for j in range(4):
            next_step.append(np.random.normal(cur[j], std_dev))

        # Calculate and compare the two likelihood function values for the
        # current and the next states.
        # HMF_cur = fit_weights(cur, pca.u[:, :4])
        # HMF_next = fit_weights(next, pca.u[:, :4])
        # N_cur = integrate_dn_dlnM(HMF_cur, 0)
        # N_next = integrate_dn_dlnM(HMF_next, 0)
        log_likelihood_cur = - log_P(cur, pca, Y[20*n:20*(n+1)], C, idx)
        # -likelihood_sim(N_cur, Y[20*n:20*(n+1)], C, idx)
        log_likelihood_next = - log_P(next_step, pca, Y[20*n:20*(n+1)], C, idx)
        #  -likelihood_sim(N_next, Y[20*n:20*(n+1)], C, idx)

        if i % 100 == 0:
            print(log_likelihood_cur)
            print(log_likelihood_next)

        # The (log) ratio of the likelihoods determines the probability of
        # switching to the new state.
        lnR = log_likelihood_next - log_likelihood_cur
        if flip_coin(np.exp(lnR)):
            cur = next_step
            acc += 1

    print("Acceptance rate: %f" % (acc/tot))
    return states[-1000:]


tic = time.perf_counter()
states = mcmc_step(N_hops=100000, pca=pca, std_dev=0.5, n=0)
toc = time.perf_counter()
print(states[-100:])
print("Time: %f seconds." % (toc-tic))