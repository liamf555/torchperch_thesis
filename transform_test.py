import numpy as np

class Transform():

    def __init__(self):
        self.o_agent_ekf = np.zeros((3,1))
        self.rotation = np.identity(3)
        self.offset_vector = np.array([[40], [0], [2]])

    def setup(self, state):

        position_ekf = np.transpose(state[:,0:3])
        yaw = state[0,5]

        cy = np.cos(yaw)
        sy = np.sin(yaw)

        rot = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
            ])

        offset_vector_rot = np.matmul(rot, self.offset_vector)

        self.o_agent_ekf = position_ekf + offset_vector_rot

        self.rotation = np.transpose(rot)

    def apply(self, state):

        position_ekf = np.transpose(state[:,0:3])

        rel_pos_ekf = position_ekf - self.o_agent_ekf

        pos_agent = np.matmul(self.rotation, rel_pos_ekf)

        pos_agent[1] = 0

        state[:,0:3] = np.transpose(pos_agent)

        return state


state = np.array([[65,     0,        -10,
                      0,     0,        np.deg2rad(45),
                      0,     0,        0,
                      0,     0,        0,
                      0, 0
	                  ]])

transform = Transform()

transform.setup(state)

state = np.array([[75,     0,        -9,
                      0,     0,        np.deg2rad(45),
                      0,     0,        0,
                      0,     0,        0,
                      0, 0
	                  ]])



print(transform.apply(state))