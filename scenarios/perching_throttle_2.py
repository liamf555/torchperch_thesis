import numpy as np
# import wandb

# Perching scenario

state_dims = 9
actions = 49
failReward = 0.0
h_min = -30


def wrap_class(BixlerClass, parameters):
    class PerchingBixler(BixlerClass):
        def __init__(self):
            super(PerchingBixler, self).__init__(parameters)
            self.variable_start = parameters.get("variable_start")
            self.start_config = tuple(parameters.get("start_config"))
            self.max_throttle = (self.mass*9.81)/2

        def is_out_of_bounds(self):
            def is_in_range(x, lower, upper):
                return lower < x and x < upper
            # Prevent flipping over
            if self.orientation_e[1, 0] > np.pi / 2:
                return True
            # Check x remains sensible
            if not is_in_range(self.position_e[0, 0], -50, 10):
                return True
            # Check not trying to turn throttle back on
            # if self.throttle_change and self.throttle_on:
            #     return True
            if self.throttle_on and abs(self.alpha) > 10:
                return True
            if self.throttle_on and np.rad2deg(abs(self.orientation_e[1, 0])) > 10:
                return True
            # Check z remains sensible (i.e. not crashed)
            if not is_in_range(self.position_e[2, 0], h_min, 1):
                return True
            # Check u remains sensible, > 0
            if not is_in_range(self.velocity_b[0, 0], 0, 20):
                return True
            # airspeed remains sensible
            if self.airspeed > 100:
                return True
            return False

        @staticmethod
        def gaussian(x, sig=0.4, mu=0):
            return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2)/2)

        # throttle 1
        # def get_reward(self):
        #     if self.is_terminal():
        #         if self.is_out_of_bounds():
        #             return failReward
        #         if self.throttle_change == 0:
        #             return failReward
        #         obs = np.array([self.position_e[0,0], self.position_e[2,0], self.orientation_e[1,0], self.velocity_b[0,0], self.velocity_b[2,0]])
        #         # print(obs)
        #         target_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        #         bound = np.array([15, 5, np.deg2rad(20),10,10])
        #         cost = (target_state - obs)/bound
        #         cost = list(map(self.gaussian, cost))
        #         reward = np.prod(cost) * ((1/np.exp(self.throttle_change) * np.exp(1)))
        #         # print(self.throttle_change)
        #         wandb.log({"throttle_change": self.throttle_change})
        #         return reward
        #     return 0.0

        # throttle 2

        def get_reward(self):
            if self.is_terminal():
                if self.is_out_of_bounds():
                    return failReward
                if self.throttle_change == 0:
                    return failReward
                obs = np.array([self.position_e[0, 0], self.position_e[2, 0],
                               self.orientation_e[1, 0], self.velocity_b[0, 0], self.velocity_b[2, 0]])
                # print(obs)
                target_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                bound = np.array([15, 5, np.deg2rad(20), 10, 10])
                cost = (target_state - obs)/bound
                cost = list(map(self.gaussian, cost))
                reward = np.prod(cost)
                wandb.log({"throttle_change": self.throttle_change})
                return reward
            return 0.0

        def is_terminal(self):
            # Terminal point is floor
            if self.position_e[2, 0] > 0:
                return True
            return self.is_out_of_bounds()

        def get_state(self):
            return super(PerchingBixler, self).get_state()[0:14].T

        def get_normalized_obs(self, state=None):
            if state is None:
                state = self.get_state()

            # # throttle 1
            # obs = np.float64(np.delete(state, [1, 3, 5, 7, 9, 11], axis=1))
            # obs = np.concatenate((obs, [[self.throttle_change]]), axis=1)

            # throttle 2
            obs = np.float64(np.delete(state, [1, 3, 5, 7, 9, 11], axis=1))
            obs = np.concatenate((obs, [[float(self.throttle_on)]]), axis=1)

            # print(obs)

            # pb2 = np.pi*2
            # mins = np.array([ -50,  h_min,  -pb2, -10, -10,  -pb2,  self.sweep_limits[0], self.elev_limits[0]])
            # maxs = np.array([  10,       1,  pb2, 20,   10,   pb2,  self.sweep_limits[1], self.elev_limits[1]])
            # return (obs-mins)/(maxs-mins)

            # using SB normalize wrapper
            # obs = {"vec": obs, "throttle": int(self.throttle_on)}

            # print(f"perching_obs {obs}")

            return obs

        def reset_scenario(self):

            self.wind_sim.update()
            wind = self.wind_sim.get_wind()

            if self.variable_start:
                theta = self.np_random.normal(
                    0, 0.0524)  # shift +-3.5degs in theta
                # theta = 0.0
                self.airspeed = self.np_random.normal(13, 1)
            else:
                self.airspeed = 13  # m/s
                theta = 0.0

            # print(self.airspeed)
            u = (self.airspeed + wind[0])[0]
            self.velocity_e = np.array([[0], [0], [0]])
            self.velocity_e[0][0] = u
            self.throttle_on = True
            self.throttle_change = 0
            self.prev_throttle = True
            self.time_log = True
            self.change_time = 0
            self.first_flag = True
            self.alpha = 0

            initial_state = np.array(
                [[self.start_config[0], 0, self.start_config[1], 0, theta, 0, u, 0, 0, 0, 0, 0, 0, 0, 0]], dtype="float32")

            # print(f"u: {start_shift_u}, wind: {wind[0]}, airspeed: {target_airspeed}")

            self.set_state(initial_state)

            Q = 0.5 * self.rho * (self.airspeed**2)

            C_D0, C_Dalpha = self._get_coefficients_C_D()
            C_D = C_D0 + C_Dalpha * self.alpha
            D = Q * self.S * C_D

            C_L0, C_Lalpha, C_Lq = self._get_coefficients_C_L()
            q = np.rad2deg(self.omega_b[1, 0])
            # print(q)
            C_L = C_L0 + (C_Lalpha * self.alpha) + (C_Lq * q)
            L = Q * self.S * C_L

            drag = np.matmul(self.dcm_wind2body,
                             np.array([[-D], [0], [-L]]))[0]

            self.throttle_val = np.clip(-drag[0], 0, self.max_throttle)

            # print(self.dcm_wind2body)
            # print("ep start")

            # print(f"alpha: {self.alpha}")
            # print(f"drag: {drag}", D, u, self.alpha, self.airspeed)

    return PerchingBixler
