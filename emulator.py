import glob
import numpy as np
import os
import pickle


class HaloEmulator:

    def __init__(self, path='./hmf/', mass_type='M200c'):
        self.path = path
        self.mass_type = mass_type
        self.point_list = HaloEmulator.extract_data(path, mass_type)
        self.X, self.Y = HaloEmulator.convert_data(self.point_list)

    @staticmethod
    def extract_data(path, mass_type):
        filelist = glob.glob(os.path.join(path, 'dndm_' + mass_type + '*.pkl'))
        points = []
        for filename in filelist:
            with open(filename, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            points.append(HaloEmulator.reshape_data(data))
        return np.concatenate(points, axis=0)

    @staticmethod
    def convert_data(data):
        size = len(data)
        X = np.zeros((size, data[0].size-1))
        Y = np.zeros((size, 1))
        for i in range(size):
            X[i] = np.copy(data[i][:5])
            Y[i][0] = data[i][5]
        return (X, Y)

    @staticmethod
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

    def __init__(self, path='./hmf/', mass_type='M200c', M_low=0, M_high=None, n_chunks=None, redshift=None):
        super().__init__(path, mass_type)
        self.M_low = M_low
        self.M_high = M_high
        self.n_chunks = n_chunks
        self.redshift = redshift
        self.X, self.Y = RedshiftTester.set_limits(
            self.X, self.Y, self.M_low, self.M_high, self.n_chunks)

    @staticmethod
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
