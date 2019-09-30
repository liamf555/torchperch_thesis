#import logging
from gym.envs.registration import register

#logger = logging.getLogger(__name__)

register(
    id='Bixler-v0',
    entry_point='gym_bixler.envs:BixlerEnv',
    # timestep_limit=1000,
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='Bixler-v2',
    entry_point='gym_bixler.envs:BixlerEnvContRate',
)

register(
    id='Bixler-v1',
    entry_point='gym_bixler.envs:BixlerEnvCont',
)

register(
	id='rBixler-v0',
	entry_point='gym_bixler.envs:RobsBixlerEnv'
)
