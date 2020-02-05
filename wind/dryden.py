import math
import numpy as np
from scipy import signal

from scipy.signal import lti

from math import sqrt

import matplotlib.pyplot as plt

class Dryden(object):

    ''' Class to generate Dryden gusts. Initial application applies to velocity spectra only. ''' 

    def __init__(self, Ts = 0.1, V_a = 17):

        # model parameters
        
        self.sigma_u = 1.06 
        self.sigma_v = self.sigma_u  
        self.sigma_w = 0.7


        self.l_u = 200.0
        self.l_v = self.l_u
        self.l_w = 50.0

        num_u = [self.sigma_u * math.sqrt(2*self.l_u / (np.pi * V_a))] # factor of root(pi) removed

        num_v1 = self.sigma_v * math.sqrt(self.l_v / V_a) # factor of root(pi) removed
        num_v2 = (math.sqrt(3)*self.l_v) / V_a
        num_v = [num_v1 * num_v2, num_v1]
           
        num_w1 = self.sigma_w * math.sqrt(self.l_w / V_a) # factor of root(pi) removed
        num_w2 = (math.sqrt(3) * self.l_w) / V_a
        num_w = [num_w1 * num_w2, num_w1]

        den_u = [self.l_u / V_a, 1]

        den_v1 = (self.l_v / V_a) ** 2
        den_v2 = (self.l_v / V_a) * 2
        den_v = [den_v1, den_v2, 1]

        den_w1 = (self.l_w / V_a) ** 2
        den_w2 = (self.l_w / V_a) * 2
        den_w = [den_w1, den_w2, 1]

        H_u = signal.TransferFunction(num_u, den_u) 
        H_v = signal.TransferFunction(num_v, den_v)
        H_w = signal.TransferFunction(num_w, den_w)

        self.ss_u = H_u.to_ss()
        self.ss_v = H_v.to_ss()
        self.ss_w = H_w.to_ss()

        self._gust_u = 0.0
        self._gust_v = 0.0
        self._gust_w = 0.0

        self.np_random = None

        self._Ts = Ts

        self.gust = np.zeros((5, 1))


    def update(self):
        
          # zero mean unit variance Gaussian (white noise)
        noise = self.np_random.randn(3,1) * sqrt(np.pi / 0.1)

        print(noise)

        # noise = [10, 10, 10]




        
        # propagate Dryden model (Euler method): x[k+1] = x[k] + Ts*( A x[k] + B w[k] )
        #self._gust_state += self._Ts * (self._A @ self._gust_state + self._B * w)
        self._gust_u += self._Ts * (self.ss_u.A * self._gust_u + self.ss_u.B * noise[0])
        gust_u = self.ss_u.C @ self._gust_u

        self._gust_v += self._Ts * (self.ss_v.A * self._gust_v + self.ss_v.B * noise[1])
        gust_v = self.ss_v.C @ self._gust_v

        self._gust_w += self._Ts * (self.ss_w.A * self._gust_w + self.ss_w.B * noise[2])
        gust_w = self.ss_w.C @ self._gust_w

        self.gust = np.array([[gust_u.item(0),gust_v.item(0),gust_w.item(0)]]).T


        # output the current gust: y[k] = C x[k]
        return self.gust


    def seed(self, seed):

        self.np_random = np.random.RandomState(seed)

class dryden4:
    def __init__(self, size=100, dt=0.1, b=0, h=50, V_a=17, intensity=None):
        # For fixed (nominal) altitude and airspeed
      
        V_a = V_a  # airspeed [m/s]

        sigma_w = 0.7
        sigma_u = 1.06
        sigma_v = sigma_u

        # Turbulence length scales
        L_u = 200
        L_v = L_u
        L_w = 50

        K_u = sigma_u * math.sqrt((2 * L_u) / (math.pi * V_a))
        K_v = sigma_v * math.sqrt((L_v) / (math.pi * V_a))
        K_w = sigma_w * math.sqrt((L_w) / (math.pi * V_a))

        T_u = L_u / V_a
        T_v1 = math.sqrt(3.0) * L_v / V_a
        T_v2 = L_v / V_a
        T_w1 = math.sqrt(3.0) * L_w / V_a
        T_w2 = L_w / V_a

        # Convert back to m/s in the numerator (NB: this should not carry over to angular rates below)
        self.H_u = lti( K_u, [T_u, 1])
        self.H_v = lti([ K_v * T_v1,  K_v], [T_v2 ** 2, 2 * T_v2, 1])
        self.H_w = lti([ K_w * T_w1,  K_w], [T_w2 ** 2, 2 * T_w2, 1])

        # K_p = sigma_w * math.sqrt(0.8 / V_a) * ((math.pi / (4 * b)) ** (1 / 6)) / ((L_w) ** (1 / 3))
        # K_q = 1 / V_a
        # K_r = K_q

        # T_p = 4 * b / (math.pi * V_a)
        # T_q = T_p
        # T_r = 3 * b / (math.pi * V_a)

        # self.H_p = lti(K_p, [T_p, 1])
        # self.H_q = lti([-K_w * K_q * T_w1, -K_w * K_q, 0], [T_q * T_w2 ** 2, T_w2 ** 2 + 2 * T_q * T_w2, T_q + 2 * T_w2, 1])
        # self.H_r = lti([K_v * K_r * T_v1, K_v * K_r, 0], [T_r * T_v2 ** 2, T_v2 ** 2 + 2 * T_r * T_v2, T_r + 2 * T_v2, 1])

        self.np_random = None
        self.seed()

        self.dt = dt
        self.sim_length = None

        #self.noise = None
        self.vel_lin = None
        self.vel_ang = None

    def seed(self, seed=0):
        self.np_random = np.random.RandomState(seed)

    

    def generate_noise(self, size):


        # noise = [math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
        #            math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
        #            math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
        #            math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
                #    ]

        noise = [math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
                   math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
                   math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size)]
        

        return noise

    def reset(self):
        self.vel_lin = None
        self.vel_ang = None
        self.sim_length = 0

    def simulate(self, length, noise=None):
        t_span = [self.sim_length, self.sim_length + length]

        t = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) / self.dt)

        if noise is None:
            noise = self.generate_noise(t.shape[0])

        vel_lin = np.array([self.H_u.output(noise[0], t)[1],
                                   self.H_v.output(noise[1], t)[1],
                                   self.H_w.output(noise[2], t)[1]])

        # vel_ang = np.array([self.H_p.output(noise[3], t)[1],
        #                              self.H_q.output(noise[3], t)[1],
        #                              self.H_r.output(noise[3], t)[1]])

        if self.vel_lin is None:
            self.vel_lin = vel_lin
            # self.vel_ang = vel_ang
        else:
            self.vel_lin = np.concatenate((self.vel_lin, vel_lin), axis=1)
            # self.vel_ang = np.concatenate((self.vel_ang, vel_ang), axis=1)

        self.sim_length += length

dryden = Dryden()
dryden.seed(0)
dryden_4 =  dryden4() 

# dryden_4.seed()
dryden_4.reset()

sim_time = 0

u_speeds = []
v_speeds = []
w_speeds = []
times = []

while sim_time < 100:
    dryden.update()
    u_speeds.append(dryden.gust.item(0))
    times.append(sim_time)

    sim_time += 0.1

dryden_4.simulate(100)
# dryden.update()
# plt.plot(times[:-1], dryden.gust_u, label = '1')

plt.plot(times, u_speeds, label = 2)

plt.plot(times[:-1], dryden_4.vel_lin[0], label = 3)
plt.legend()

plt.show()



    
















    














































        
        