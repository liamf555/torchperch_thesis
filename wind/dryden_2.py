import math
from typing import ValuesView
import numpy as np
from scipy import signal
import wandb

from scipy.signal import lti, TransferFunction
from math import sqrt
import matplotlib.pyplot as plt

class Filter:
    def __init__(self, num, den):
        """
        Wrapper for the scipy LTI system class.
        :param num: numerator of transfer function
        :param den: denominator of transfer function
        """
        self.filter = lti(num, den)
        self.x = None
        self.y = None
        self.t = None

    def simulate(self, u, t):
        """
        Simulate filter
        :param u: filter input
        :param t: time steps for which to simulate
        :return: filter output
        """
        if self.x is None:
            x_0 = None
        else:
            x_0 = self.x[-1]

        self.t, self.y, self.x = self.filter.output(U=u, T=t, X0=x_0)

        return self.y

    def reset(self):
        """
        Reset filter
        :return:
        """
        self.x = None
        self.y = None
        self.t = None


class DrydenGustModel:
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
        self.L_u = 200
        self.L_v = self.L_u
        self.L_w = 50

        K_u = self.sigma_u * math.sqrt((2 * self.L_u) / (math.pi * self.Va))
        K_v = self.sigma_v * math.sqrt((self.L_v) / (math.pi * self.Va))
        K_w = self.sigma_w * math.sqrt((self.L_w) / (math.pi * self.Va))

        T_u = self.L_u / self.Va
        T_v1 = math.sqrt(3.0) * self.L_v / self.Va
        T_v2 = self.L_v / self.Va
        T_w1 = math.sqrt(3.0) * self.L_w / self.Va
        T_w2 = self.L_w / self.Va

        K_p = self.sigma_w * math.sqrt(0.8 / self.Va) * ((math.pi / (4 * b)) ** (1 / 6)) / ((self.L_w) ** (1 / 3))
        K_q = 1 / self.Va
        K_r = K_q

        T_p = 4 * b / (math.pi * self.Va)
        T_q = T_p
        T_r = 3 * b / (math.pi * self.Va)

        # self.filters = {"H_u": Filter(K_u, [T_u, 1]),
        #                 "H_w": Filter([K_w * T_w1, K_w], [T_w2 ** 2, 2 * T_w2, 1]),}
        self.filters = {"H_u": Filter(K_u, [T_u, 1]),
                        "H_v": Filter([ K_v * T_v1,K_v], [T_v2 ** 2, 2 * T_v2, 1]),
                        "H_w": Filter([K_w * T_w1, K_w], [T_w2 ** 2, 2 * T_w2, 1]),
                        "H_p": Filter(K_p, [T_p, 1]),
                        "H_q": Filter([-K_w * K_q * T_w1, -K_w * K_q, 0], [T_q * T_w2 ** 2, T_w2 ** 2 + 2 * T_q * T_w2, T_q + 2 * T_w2, 1]),
                        "H_r": Filter([K_v * K_r * T_v1, K_v * K_r, 0], [T_r * T_v2 ** 2, T_v2 ** 2 + 2 * T_r * T_v2, T_r + 2 * T_v2, 1]),}

        self.sim_length = 0
        self.turbulence_sim_length = 1000
        self.dt = dt

        self.noise = None
        self.np_random = None
        self.seed()

        self.vel_lin = None
        self.vel_ang = None

        # self._gust_u = 0.0
        # self._gust_v = 0.0
        # self._gust_w = 0.0
        # # self._gust_q = 0.0

        # self.gust_u = 0.0
        # self.gust_v = 0.0
        # self.gust_w = 0.0
        # self.gust_q = 0.0

    def seed(self, seed=None):
        """
        Seed the random number generator.
        :param seed: (int) seed.
        :return:
        """
        self.np_random = np.random.RandomState(seed)

    def _generate_noise(self, size):
        self.noise = np.sqrt(np.pi / self.dt) * self.np_random.standard_normal(size=(4, size)) 

        # print(self.noise)
        return self.noise

    def reset(self, noise=None):
        """
        Reset model.
        :param noise: (np.array) Input to filters, should be four sequences of Gaussianly distributed numbers.
        :return:
        """
        self.vel_lin = None
        # self.vel_ang = None
        self.sim_length = 0

        if noise is not None:
            assert len(noise.shape) == 2
            assert noise.shape[0] == 4
            noise = noise * math.sqrt(math.pi / self.dt)
        self.noise = noise

        for filter in self.filters.values():
            filter.reset()


    def simulate(self, length):
        """
        Simulate turbulence by passing band-limited Gaussian noise of length length through the shaping filters.
        :param length: (int) the number of steps to simulate.
        :return:
        """
        t_span = [self.sim_length, self.sim_length + length]

        t = np.linspace(t_span[0] * self.dt, t_span[1] * self.dt, length)
        # t = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) / self.dt)

        # print(len(t))

        if self.noise is None:
            noise = self._generate_noise(t.shape[0])
        else:
            if self.noise.shape[-1] >= t_span[1]:
                noise = self.noise[:, t_span[0]:t_span[1]]
            else:
                noise_start_i = t_span[0] % self.noise.shape[-1]
                remaining_noise_length = self.noise.shape[-1] - noise_start_i
                if remaining_noise_length >= length:
                    noise = self.noise[:, noise_start_i:noise_start_i + length]
                else:
                    if length - remaining_noise_length > self.noise.shape[-1]:
                        concat_noise = np.pad(self.noise,
                                              ((0, 0), (0, length - remaining_noise_length - self.noise.shape[-1])),
                                              mode="wrap")
                    else:
                        concat_noise = self.noise[:, :length - remaining_noise_length]
                    noise = np.concatenate((self.noise[:, noise_start_i:], concat_noise), axis=-1)


        vel_lin = np.array([self.filters["H_u"].simulate(noise[0], t),
                            self.filters["H_v"].simulate(noise[1], t),
                            self.filters["H_w"].simulate(noise[2], t)])

        vel_ang = np.array([self.filters["H_p"].simulate(noise[3], t),
                            self.filters["H_q"].simulate(noise[1], t),
                            self.filters["H_r"].simulate(noise[2], t)])

        if self.vel_lin is None:
            self.vel_lin = vel_lin
            self.vel_ang = vel_ang
        else:
            self.vel_lin = np.concatenate((self.vel_lin, vel_lin), axis=1)
            self.vel_ang = np.concatenate((self.vel_ang, vel_ang), axis=1)

        self.sim_length += length


    def get_turbulence_linear(self, timestep):
            """
            Get linear component of turbulence model at given timestep
            :param timestep: (int) timestep
            :return: ([float]) linear component of turbulence at given timestep
            """
            return self._get_turbulence(timestep, "linear")

    def get_turbulence_angular(self, timestep):
        """
        Get angular component of turbulence model at given timestep
        :param timestep: (int) timestep
        :return: ([float]) angular component of turbulence at given timestep
        """
        return self._get_turbulence(timestep, "angular")

    def _get_turbulence(self, timestep, component):
        """
        Get turbulence at given timestep.
        :param timestep: (int) timestep
        :param component: (string) which component to return, linear or angular.
        :return: ([float]) turbulence component at timestep
        """
        if timestep >= self.sim_length:
            self.simulate(self.turbulence_sim_length)

        if component == "linear":
            return self.vel_lin[:, timestep]
        else:
            return self.vel_ang[:, timestep]

if __name__ == "__main__":

    dryden = DrydenGustModel()

    values = np.linspace(0, 10, 1001)
    print(values)

    
    x_gusts = []
    for i, value  in enumerate(values):
        x_gusts.append(dryden.get_turbulence_linear(i)[0])
    plt.plot(values, x_gusts)
    plt.show()
    