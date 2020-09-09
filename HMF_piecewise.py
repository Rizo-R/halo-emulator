import numpy as np


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
