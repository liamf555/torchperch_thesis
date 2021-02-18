
import numpy as np

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
        self.gust = np.zeros((1, 3))

    def seed(self, seed=None):
        """
        Seed the random number generator.
        :param seed: (int) seed.
        :return:
        """
        self.np_random = np.random.RandomState(seed)


    def reset(self):
        self._gust_state = np.zeros((5, 1))

    def update(self, Va, dt):
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

if __name__ == "__main__":

    dryden = DrydenGustModel()



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

    while sim_time < 99.9 :
        sim_time += 0.1
        # noise = list(zip(*dryden.noise))[i]
        # print(noise)
        noise=0
        dryden.update(13, 0.01)
        u_vel.append(dryden.gust.item(0))
        # v_vel.append(dryden.gust.item(1))
        w_vel.append(dryden.gust.item(2))
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


    
















    














































        
        