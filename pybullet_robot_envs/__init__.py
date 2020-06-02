

import logging
import gym
from gym.envs.registration import register

register(
        id='iCubReach-v0',
        entry_point='pybullet_robot_envs.envs:iCubReachGymEnv',
        max_episode_steps=1000,
        kwargs={ 'use_IK':1,
                 'control_arm': 'l',
                 'control_orientation': 0,
                 'obj_pose_rnd_std': 0,
                 'max_steps': 1000,
                 'renders': True},
)

register(
        id='iCubPush-v0',
        entry_point='pybullet_robot_envs.envs:iCubPushGymEnv',
        max_episode_steps=1000,
        kwargs={ 'use_IK': 1,
                 'control_arm': 'l',
                 'control_orientation': 0,
                 'obj_pose_rnd_std': 0.05,
                 'tg_pose_rnd_std': 0,
                 'max_steps': 1000,
                 'reward_type': 0,
                 'renders': True},
)

register(
        id='iCubPushGoal-v0',
        entry_point='pybullet_robot_envs.envs:iCubPushGymGoalEnv',
        max_episode_steps=1000,
        kwargs={ 'use_IK': 1,
                 'control_arm': 'r',
                 'control_orientation': 1,
                 'obj_pose_rnd_std': 0.05,
                 'tg_pose_rnd_std': 0,
                 'max_steps': 1000,
                 'renders': True},
)


register(
        id='pandaReach-v0',
        entry_point='pybullet_robot_envs.envs:pandaReachGymEnv',
        max_episode_steps=1000,
        kwargs={'use_IK':0,
                'obj_pose_rnd_std': 0.05,
                'max_steps': 1000,
                'includeVelObs': True,
                'renders': True},
)

register(
        id='pandaPush-v0',
        entry_point='pybullet_robot_envs.envs:pandaPushGymEnv',
        max_episode_steps=1000,
        kwargs={'use_IK': 0,
                'obj_pose_rnd_std': 0.05,
                'tg_pose_rnd_std': 0,
                'includeVelObs': True,
                'max_steps': 1000,
                'renders': True},
)

register(
        id='pandaPushGoal-v0',
        entry_point='pybullet_robot_envs.envs:pandaPushGymGoalEnv',
        max_episode_steps=1000,
        kwargs={'use_IK': 0,
                'obj_pose_rnd_std': 0.05,
                'tg_pose_rnd_std': 0,
                'includeVelObs': True,
                'max_steps': 1000,
                'renders': True},
)

# --------------------------- #
def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('iCub')>=0]
    return btenvs

getList()
