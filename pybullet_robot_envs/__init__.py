import logging
import gym
from gym.envs.registration import register

register(
        id='iCubPush-v0',
        entry_point='pybullet_robot_envs.envs.icub_envs:iCubPushGymEnv',
        max_episode_steps=1000,
        reward_threshold=1000.0,
)

# --------------------------- #
def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('iCub')>=0]
    return btenvs

getList()
