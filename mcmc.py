import emcee
import glob
import numpy as np
import scipy.integrate as integrate
import scipy.linalg
import time


from emulator import *
from HMF_fit import *
from HMF_piecewise import *
from likelihood import *
from pca import *


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

# Following the constraint c_i <= 0, remove positive entries.
ind_h, ind_v = np.where(params[:, 1:] > 0)
params[ind_h, ind_v+1] = 0

Mpiv = 1e14

# Put HMF data into a single matrix.
Y_fit = np.ones((1001, params.shape[0]))

for i in range(params.shape[0]):
    Y_fit[:, i], _ = get_HMF_piecewise(
        params[i][1:], reg_bins=19, offset=0, Mpiv=1e14)

# Calculate the residuals and conduct PCA analysis.
HMF_mean = np.average(Y_fit, axis=1).reshape(-1, 1)
Y_res = Y_fit - np.broadcast_to(HMF_mean, (HMF_mean.size, len(filelist)))
pca = Pca.calculate(Y_res)


def fit_weights(weights, evectors):
    '''Returns HMF that is based on the given N weights and the first N 
    given eigenvectors. Returns a Numpy array.
    Prerequisites: len(evectors) >= len(weights).'''
    res = np.zeros((1001,))
    for i in range(len(weights)):
        res += weights[i] * evectors[:, i]
    return res


def integrate_dn_dlnM(HMF):
    '''Integrates the given HMF. Returns a (20,1) array of integrated values 
    (not rounded).'''
    # 20 bins.
    M_logspace = np.logspace(13, 15.5, 1001)
    vals = np.zeros((20, 1))
    for i in range(20):
        ind_lo = 50*i
        ind_hi = 50*(i+1)+1
        x = M_logspace[ind_lo:ind_hi]
        dn_dlnM = np.exp(HMF[ind_lo:ind_hi])
        y = 1e9 * dn_dlnM / x
        vals[i] = integrate.trapz(y, x)
    return vals


def log_likelihood_mcmc(weights, y, n):
    '''The cosmology is provided by the number in the (sorted) list of 
    cosmologies, starting at [0., 0.3, 2.1].'''
    filelist_mat = glob.glob("./covmat/covmat_M200c_*.pkl")
    filelist_mat.sort()
    cov_matrix = load_cov_matrices(filelist_mat[n])[-1]
    C, idx = remove_zeros(cov_matrix)

    N_weights = integrate_dn_dlnM(
        HMF_mean[:, 0] + fit_weights(weights, pca.u))
    N_y = integrate_dn_dlnM(HMF_mean[:, 0] + y)

    return -likelihood_sim(N_weights, N_y, C, idx)


def log_prior(weights):
    for w in weights:
        if -700. < w < 700.:
            return 0.
    return -np.inf


def log_probability(weights, **kwargs):
    y = kwargs['y']
    n = kwargs['n']
    lp = log_prior(weights)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_mcmc(weights, y, n)


def mcmc(Y, n, Nsteps=10000, ndim=4, nwalkers=32, pca=pca,
         log_probability=log_probability):
    # Initial position is a Gaussian ball around the PCA weights.
    initial_weights = np.dot(np.diagflat(pca.s), pca.v)[:4, n]
    p0 = np.broadcast_to(initial_weights, (nwalkers, initial_weights.size))
    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

    tic = time.perf_counter()
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, kwargs=({'y': Y[:, n], 'n': n}))
    sampler.run_mcmc(pos, Nsteps, progress=True)
    toc = time.perf_counter()

    path = "./mcmc_samples/n=" + str(n) + "/"
    # Create the direction if it doesn't exist.
    if not os.path.exists(path):
        os.makedirs(path)
    samples_flat = []

    try:
        # Obtain tau and calculate burnin and thin values.
        tau = sampler.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))

        samples_flat = sampler.get_chain()
        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        log_prob_samples = sampler.get_log_prob(
            discard=burnin, flat=True, thin=thin)
        log_prior_samples = sampler.get_blobs(
            discard=burnin, flat=True, thin=thin)

        np.save(path + "samples_flat", sampler.get_chain())
        np.save(path + "samples", samples)
        np.save(path + "log_prob_samples", log_prob_samples)
        np.save(path + "log_prior_samples", log_prior_samples)
    # If unable to obtain tau, simply record the chains without discarding any steps
    # nor thinning the chain.
    except emcee.autocorr.AutocorrError:
        samples = sampler.get_chain(flat=True)
        log_prob_samples = sampler.get_log_prob(flat=True)
        log_prior_samples = sampler.get_blobs(flat=True)

        np.save(path + "samples_no_tau", samples)
        np.save(path + "log_prob_samples_no_tau", log_prob_samples)
        np.save(path + "log_prior_samples_no_tau", log_prior_samples)

    print("Time: %f seconds." % (toc-tic))

    return samples_flat, samples, log_prob_samples, log_prior_samples


ndim = 4
nwalkers = 32
Nsteps = 10000

mcmc_samples = []

for i in range(8, 27):
    print("Current sample: %d" % i)
    _, samples, _, _ = mcmc(
        Y_res, i, Nsteps=Nsteps, ndim=ndim, nwalkers=nwalkers, pca=pca, log_probability=log_probability)
    mcmc_samples.append(samples)


# samples_new = sampler.get_chain(discard=burnin, flat=True, thin=thin)
