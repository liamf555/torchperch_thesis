import gym
import numpy as np
from pymavlink import mavutil

class RealBixlerEnv(gym.EnV):

	def __init__(self,
			connectionPath='/dev/ttyACM0',
			baud=57600,
			source_system=1,
			source_component=1):
		self.metadata = {'render.modes':[]} # None?
		self.reward_range = (0.0,1.0)       # Depends on the scenario
		self.spec = None                    # What?

		self.action_space = None            # Depends on scenario
        self.observation_space = None       # Depends on scenario

		# Establish link with autopilot
		self.master = mavutil.mavlink_connection(connectionPath, baud=57600, souce_system=1, source_component=158)
		self.master.wait_heartbeat()

	def step(self,action):
		observation = self._wait_for_MLAGENT_STATE()
		reward = scenario.get_reward(observation)
		done = scenario.is_done()
		info = {}
		return observation, reward, done, info

	def reset(self):
		# Hmm... Not really sure here...
        # I guess just hang around until the
		#  auto mission is back in the right position?
		self._wait_for_waypoint(X)

	def render(self,mode):
		# If we're having any render mode, just print the state?
		# Let it be someone else's problem for now:
		super(RealBixlerEnv,self).render(mode=mode)

	def close(self):
		# Let the autopilot know that we're giving up...
		pass

	def _wait_for_MLAGENT_STATE(self):
		# Wait for next MLAGENT_STATE message
		awaitingMessage = True
		msg = None
		while awaitingMessage:
			msg = self.master.recv_msg()
			if msg is None or not hasattr(msg,'name'):
				# Either no message, or message is noise
				continue
			if msg.name is not 'MLAGENT_STATE':
				if msg.name is 'PARAM_REQUEST_LIST':
					# If an attempt to get parameters is made, return a PARAM_VALUE message indicating no parameters
					master.mav.param_value_send("",0,master.mavlink.MAV_PARAM_TYPE_UINT8,0,0)
				continue
			
			# Got a new MLAGENT_STATE message
			awaitingMessage = False
		return np.array([[msg.x,     msg.y,        msg.z,
		                  msg.phi,   msg.theta,    msg.psi,
		                  msg.u,     msg.v,        msg.w,
		                  msg.p,     msg.q,        msg.r,
		                  msg.sweep, msg.elevator, msg.tip ]])

