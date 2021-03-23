#import logging
from gym.envs.registration import register

#logger = logging.getLogger(__name__)

register(
    id='Bixler-v0',
    entry_point='gym_bixler.envs:BixlerEnv',
    # kwargs = {'noise': 0.0,'latency': 0.0, 'var_start': True},
)


register(
    id='Bixler-v1',
    entry_point='gym_bixler.envs:BixlerEnvLive',
)

register(
    id='Bixler-v3',
    entry_point='gym_bixler.envs:BixlerHEREnv',
    # kwargs = {'noise': 0.0,'latency': 0.0, 'var_start': True},
)

# register(
#     id='Bixler-v0',
#     entry_point='gym_bixler.envs:BixlerEnv',
#     # kwargs = {'noise': 0.0,'latency': 0.0, 'var_start': True},
# )