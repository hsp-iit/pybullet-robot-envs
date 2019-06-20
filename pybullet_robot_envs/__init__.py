

import logging
import gym
from gym.envs.registration import register
"""
register(
        id='iCubReach-v0',
        entry_point='pybullet_robot_envs.envs.icub_envs:iCubReachGymEnv',
        max_episode_steps=1000,
        kwargs={ 'useIK':1, 'isDiscrete':0, 'control_arm':'l', 'useOrientation':0, 'rnd_obj_pose':1, 'maxSteps':500},
)
"""
register(
        id='pandaReach-v0',
        entry_point='pybullet_robot_envs.envs.panda_envs.panda_reach_gym_env:pandaReachGymEnv',
        max_episode_steps=1000,
        kwargs={
                 'useIK':0,
                 'isDiscrete':0,
                 'actionRepeat':1,
                 'renders':False,
                 'maxSteps':1000,
                 'dist_delta':0.03, 'numControlledJoints':2, 'fixedPositionObj':True, 'includeVelObs':True},
)

# --------------------------- #
def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('iCub')>=0]
    return btenvs

getList()