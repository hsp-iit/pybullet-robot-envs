

import logging
import gym
from gym.envs.registration import register
"""
register(
        id='iCubReach-v0',
        entry_point='pybullet_robot_envs.envs.icub_envs:iCubReachGymEnv',
        max_episode_steps=1000,
        kwargs={ 'useIK':1, 'isDiscrete':0, 'control_arm':'l', 'useOrientation':0, 'rnd_obj_pose':1, 'maxSteps':1000},
)

register(
        id='iCubPush-v0',
        entry_point='pybullet_robot_envs.envs.icub_envs:iCubPushGymEnv',
        max_episode_steps=1000,
        kwargs={ 'useIK':1, 'isDiscrete':0, 'control_arm':'l', 'useOrientation':0, 'rnd_obj_pose':1, 'maxSteps':1000},
)

# --------------------------- #
def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('iCub')>=0]
    return btenvs

getList()
