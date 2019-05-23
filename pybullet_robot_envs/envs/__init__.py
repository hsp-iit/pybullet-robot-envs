import gym
from gym.envs.registration import registry, make, spec

def register(id,*args,**kvargs):
	if id in registry.env_specs:
		return
	else:
		return gym.envs.registration.register(id,*args,**kvargs)

# --------------------------- #
# add one register entry for each Gym environment

register(
        id='iCubPush-v0',
        entry_point='envs.iCub:iCubPushGymEnv',
        max_episode_steps=1000,
        reward_threshold=1000.0,
)

# --------------------------- #
def getList():
	btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('iCub')>=0]
	return btenvs
