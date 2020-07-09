import math
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
    def __init__(self, dt = 0.1, b=1.5, Va=13, intensity='light'):
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

        if intensity == 'light':
            self.sigma_w = 0.7
            self.sigma_u = 1.06
            self.sigma_v = self.sigma_u
        elif intensity == 'moderate':
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

        self.np_random = None
        self.seed()

        self.sim_length = 0
        self.dt = dt

        self.noise = None

        self.vel_lin = None
        self.vel_ang = None

        self._gust_u = 0.0
        # self._gust_v = 0.0
        self._gust_w = 0.0
        # self._gust_q = 0.0

        self.gust_u = 0.0
        # self.gust_v = 0.0
        self.gust_w = 0.0
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

    # def reset(self):
    #     self._gust_u = 0.0
    #     self._gust_w = 0.0
    #     self.gust = []

    def simulate(self, length):
        """
        Simulate turbulence by passing band-limited Gaussian noise of length length through the shaping filters.
        :param length: (int) the number of steps to simulate.
        :return:
        """
        t_span = [self.sim_length, self.sim_length + length]

        # t = np.linspace(t_span[0] * self.dt, t_span[1] * self.dt, length)
        t = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) / self.dt)

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

        print(noise)

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

    def _euler(self, tfs):

        self.ss_u = tfs["H_u"].to_ss()
        # self.ss_v = self.filters["H_v"].filter.to_ss()
        self.ss_w = tfs["H_w"].to_ss()
        # self.ss_q = self.filters["H_q"].filter.to_ss()

    def _gust(self, dt):

        print(dt)
        noise = np.sqrt(np.pi / dt) * self.np_random.standard_normal(2)

        # print(noise)
      
        # propagate Dryden model (Euler method): x[k+1] = x[k] + Ts*( A x[k] + B w[k] )
        #self._gust_state += self._Ts * (self._A @ self._gust_state + self._B * w)
        self._gust_u += dt * (self.ss_u.A * self._gust_u + self.ss_u.B * noise[0])
        gust_u = self.ss_u.C @ self._gust_u

        # self._gust_v += self.dt * (self.ss_v.A * self._gust_v + self.ss_v.B * noise[1])
        # gust_v = self.ss_v.C @ self._gust_v

        self._gust_w += dt * (self.ss_w.A * self._gust_w + self.ss_w.B * noise[1])
        gust_w = self.ss_w.C @ self._gust_w

        # self._gust_q += self.dt * (self.ss_q.A * self._gust_q + self.ss_q.B * noise[1])
        # gust_q = self.ss_q.C @ self._gust_q

        # self.gust = np.array([[gust_u.item(0),gust_v.item(0),gust_w.item(0), gust_q.item(0)]]).T

        # print(gust_u.item(0))

        wandb.log({'dryden_u': gust_u.item(0)})

        self.gust = np.array([[gust_u.item(0), 0.0,  gust_w.item(0)]])

        # self.gust = np.array([[0.0, 0.0,  0.0]])

        return self.gust

    def update(self, Va, dt):

        print(Va)
        K_u = self.sigma_u * math.sqrt((2 * self.L_u) / (math.pi * Va))
        # K_v = sigma_v * math.sqrt((L_v) / (math.pi * self.Va))
        K_w = self.sigma_w * math.sqrt((self.L_w) / (math.pi * Va))

        T_u = self.L_u / Va
        # T_v1 = math.sqrt(3.0) * L_v / self.Va
        # T_v2 = L_v / self.Va
        T_w1 = math.sqrt(3.0) * self.L_w / Va
        T_w2 = self.L_w / Va

        tfs = {"H_u": TransferFunction(K_u, [T_u, 1]),
                        "H_w": TransferFunction([K_w * T_w1, K_w], [T_w2 ** 2, 2 * T_w2, 1]),}

        self._euler(tfs)

        return self._gust(dt)

if __name__ == "__main__":

    dryden = DrydenGustModel()

    print('goat')

    # dryden.euler()

    u_vel = []
    v_vel = []
    w_vel = []
    q = []
    sim_time = 0
    times = []

    dryden.seed(0)

    # dryden.simulate(1)

    # print(list(zip(*dryden.noise)))

    i = 0

    while sim_time < 0.9 :
        sim_time += 0.1
        # noise = list(zip(*dryden.noise))[i]
        # print(noise)
        noise=0
        dryden.update(13)
        u_vel.append(dryden.gust.item(0))
        # v_vel.append(dryden.gust.item(1))
        w_vel.append(dryden.gust.item(1))
        # q.append(dryden.gust.item(3))
        times.append(sim_time)
        i += 1
        # print(sim_time)


    # print(u_vel)

    

    plt.plot(times, u_vel, label = 1)
    # # plt.plot(times, v_vel, label = 2)
    plt.plot(times, w_vel, label = 3)
    # plt.plot(times, q, label = 4)

    # # plt.plot(times[:-1], dryden_4.vel_lin[0], label = 2)
    # plt.plot(times, dryden.vel_lin[0], label = 'u')
    # # plt.plot(times, dryden.vel_lin[1], label = 'v')
    # plt.plot(times, dryden.vel_lin[2], label = 'w')

    # plt.plot(times, dryden.vel_lin[0], label = 'u')
    # plt.plot(times, dryden.vel_lin[1], label = 'v')
    # plt.plot(times, dryden.vel_ang[1], label = 'q')

    plt.legend()

    plt.show()


# class Dryden(object):

#     ''' Class to generate Dryden gusts. Initial application applies to velocity spectra only. ''' 

#     def __init__(self, Ts = 0.1, self.Va = 17):

#         # model parameters
        
#         self.sigma_u = 1.06 
#         self.sigma_v = self.sigma_u  
#         self.sigma_w = 0.7


#         self.l_u = 200.0
#         self.l_v = self.l_u
#         self.l_w = 50.0

#         num_u = self.sigma_u * math.sqrt(2*self.l_u / (np.pi * self.Va)) 

#         num_v1 = self.sigma_v * math.sqrt(self.l_v / self.Va) 
#         num_v2 = (math.sqrt(3)*self.l_v) / self.Va
#         num_v = [num_v1 * num_v2, num_v1]
           
#         num_w1 = self.sigma_w * math.sqrt(self.l_w / self.Va) 
#         num_w2 = (math.sqrt(3) * self.l_w) / self.Va
#         num_w = [num_w1 * num_w2, num_w1]

#         den_u = [self.l_u / self.Va, 1]

#         den_v1 = (self.l_v / self.Va) ** 2
#         den_v2 = (self.l_v / self.Va) * 2
#         den_v = [den_v1, den_v2, 1]

#         den_w1 = (self.l_w / self.Va) ** 2
#         den_w2 = (self.l_w / self.Va) * 2
#         den_w = [den_w1, den_w2, 1]

#         H_u = signal.TransferFunction(num_u, den_u) 
#         H_v = signal.TransferFunction(num_v, den_v)
#         H_w = signal.TransferFunction(num_w, den_w)

#         self.ss_u = H_u.to_ss()
#         self.ss_v = H_v.to_ss()
#         self.ss_w = H_w.to_ss()

#         self._gust_u = 0.0
#         self._gust_v = 0.0
#         self._gust_w = 0.0

#         self.np_random = None

#         self._Ts = Ts

#         self.gust = np.zeros((5, 1))


#     def update(self):
        
#           # zero mean unit variance Gaussian (white noise)
#         noise = self.np_random.randn(3,1) * sqrt(np.pi / 0.1)

#         # print(noise)

#         # noise = [10, 10, 10]
        
#         # propagate Dryden model (Euler method): x[k+1] = x[k] + Ts*( A x[k] + B w[k] )
#         #self._gust_state += self._Ts * (self._A @ self._gust_state + self._B * w)
#         self._gust_u += self._Ts * (self.ss_u.A * self._gust_u + self.ss_u.B * noise[0])
#         gust_u = self.ss_u.C @ self._gust_u

#         self._gust_v += self._Ts * (self.ss_v.A * self._gust_v + self.ss_v.B * noise[1])
#         gust_v = self.ss_v.C @ self._gust_v

#         self._gust_w += self._Ts * (self.ss_w.A * self._gust_w + self.ss_w.B * noise[2])
#         gust_w = self.ss_w.C @ self._gust_w

#         self.gust = np.array([[gust_u.item(0),gust_v.item(0),gust_w.item(0)]]).T


#         # output the current gust: y[k] = C x[k]
#         return self.gust


#     def seed(self, seed=0):

#         self.np_random = np.random.RandomState(seed)


    
















    














































        
        