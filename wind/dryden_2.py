import math
from typing import ValuesView
from dryden import DrydenGustModel
import numpy as np
import numba
from scipy import signal
import wandb
from time import time

from scipy.signal import lti, TransferFunction
from math import pi, sqrt
import matplotlib.pyplot as plt

class DrydenGustModel2:
    def __init__(self, dt = 0.01, b=1.5, Va=13, intensity='light'):
        """
        Python realization of the continuous Dryden Turbulence Model (MIL-F-8785C).
        :param dt: (float) band-limited white noise input sampling time.
        :param b: (float) wingspan of aircraft
        :param h: (float) Altitude of aircraft
        :param self.Va: (float) Airspeed of aircraft
        :param intensity: (str) Intensity of turbulence, one of ["light", "moderate", "severe"]
        """
        # For fixed (nominal) altitude and airspeed
        self.Va = Va  # airspeed [m/s]
        self.intensity = intensity

        if self.intensity == 'light':
            self.sigma_w = 0.7
            self.sigma_u = 1.06
            self.sigma_v = self.sigma_u
        elif self.intensity == 'moderate':
            self.sigma_u = 2.12
            self.sigma_v = self.sigma_u
            self.sigma_w =  1.4

        # Turbulence length scales
        self.Lu = 200
        self.Lv = self.Lu
        self.Lw = 50

        self._A = np.array([[-Va/self.Lu, 0, 0, 0, 0],
                            [0, -2*(Va/self.Lv), -(Va/self.Lv)**2, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 0, -2*(Va/self.Lw), -(Va/self.Lw)**2],
                            [0, 0, 0, 1, 0]])
        self._B = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
        self._C = np.array([[self.sigma_u * np.sqrt((2*Va)/(np.pi * self.Lu)), 0, 0, 0, 0],
                            [0, self.sigma_v * np.sqrt((3*Va)/(np.pi * self.Lv)), np.sqrt((Va/self.Lv)**3), 0, 0],
                            [0, 0, 0, self.sigma_w * np.sqrt((3*Va)/(np.pi* self.Lv)), np.sqrt((Va/self.Lw)**3)]])
        self._gust_state = np.zeros((5, 1))

    def update(self, Va, dt = 0.01):
        
        self._A = np.array([[-Va/self.Lu, 0, 0, 0, 0],
                            [0, -2*(Va/self.Lv), -(Va/self.Lv)**2, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 0, -2*(Va/self.Lw), -(Va/self.Lw)**2],
                            [0, 0, 0, 1, 0]])
        self._C = np.array([[self.sigma_u * np.sqrt((2*Va)/(np.pi * self.Lu)), 0, 0, 0, 0],
                            [0, self.sigma_v * np.sqrt((3*Va)/(np.pi  * self.Lv)), np.sqrt((Va/self.Lv)**3), 0, 0],
                            [0, 0, 0, self.sigma_w * np.sqrt((3*Va)/(np.pi * self.Lv)), np.sqrt((Va/self.Lw)**3)]])

        # calculate wind gust using Dryden model.  Gust is defined in the body frame
        w = np.sqrt(np.pi / dt) * self.np_random.standard_normal(3)
        w = [[x] for x in w]
        # propagate Dryden model (Euler method): x[k+1] = x[k] + Ts*( A x[k] + B w[k] )
        self._gust_state += dt * (self._A @ self._gust_state + self._B @ w)

        _gusts = self._C @ self._gust_state

        self.gust = np.array([[_gusts.item(0), 0.0, _gusts.item(2)]])

        return self.gust

    def seed(self, seed=None):
        """
        Seed the random number generator.
        :param seed: (int) seed.
        :return:
        """
        self.np_random = np.random.RandomState(seed)


@numba.njit
def get_gust(Va, dt = 0.01, _gust_state = 0):

    sigma_w = 0.7
    sigma_u = 1.06
    sigma_v = sigma_u
    
    Lu = 200
    Lv = Lu
    Lw = 50
    
    _A = np.array([[-Va/Lu, 0, 0, 0, 0],
                        [0, -2*(Va/Lv), -(Va/Lv)**2, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, -2*(Va/Lw), -(Va/Lw)**2],
                        [0, 0, 0, 1, 0]])

    _B = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
    _C = np.array([[sigma_u * np.sqrt((2*Va)/(np.pi * Lu)), 0, 0, 0, 0],
                        [0, sigma_v * np.sqrt((3*Va)/(np.pi  * Lv)), np.sqrt((Va/Lv)**3), 0, 0],
                        [0, 0, 0, sigma_w * np.sqrt((3*Va)/(np.pi * Lv)), np.sqrt((Va/Lw)**3)]])

    # calculate wind gust using Dryden model.  Gust is defined in the body frame
    w = np.sqrt(np.pi / dt) * np.random.standard_normal(3)
    w = [[x] for x in w]
    # propagate Dryden model (Euler method): x[k+1] = x[k] + Ts*( A x[k] + B w[k] )
    # _gust_state += dt * (_A @ _gust_state + _B @ w)
    _gust_state += dt * (np.matmul(_A, _gust_state) + np.matmul(_B, w))

    _gusts = _C @ _gust_state

    gust = np.array([[_gusts.item(0), 0.0, _gusts.item(2)]])

    return gust

if __name__ == "__main__":

    dryden2 = DrydenGustModel2()
    # dryden = DrydenGustModel()
    # dryden.seed(23341)
    dryden2.seed(23341)


    values = np.linspace(0, 10, 1001)
    # print(values)

    x_gusts = []
    x_gusts_2 = []
    start = time()
    # for i, value  in enumerate(values):
    #     x_gusts.append( dryden.update(Va =13, dt = 0.01)[0][0])
    #     # print(dryden.update(Va = 13, dt = 0.01))
    #     # x_gusts_2.append(dryden2._gust(13)[0])

    # # end = time()
    # print(f'Old took {end - start} seconds!')

    start = time()
    _gust_state = 0
    for i, value  in enumerate(values):
        # x_gusts_2.append(dryden2._gust(13)[0])
    #    x_gusts_2.append(dryden2.update(Va= 13, dt= 0.01)[0][0])
        x_gusts_2 = get_gust(Va = 13, dt = 0.01)


    end = time()
    print(f'New took {end - start} seconds!')
    # print(x_gusts)
    # print(x_gusts_2)
    plt.plot(values, x_gusts)
    plt.plot(values, x_gusts_2)
    plt.show()