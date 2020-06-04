# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0,currentdir)

import gym
from gym import spaces
import numpy as np
import math as m

from pybullet_robot_envs.envs.panda_envs.panda_push_gym_env import pandaPushGymEnv
from pybullet_robot_envs.envs.world_envs.world_env import get_objects_list
from pybullet_robot_envs.envs.utils import goal_distance, scale_gym_data


class pandaPushGymGoalEnv(gym.GoalEnv, pandaPushGymEnv):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self,
                 numControlledJoints=7,
                 use_IK=0,
                 action_repeat=1,
                 obj_name=get_objects_list()[1],
                 renders=False,
                 max_steps=1000,
                 obj_pose_rnd_std=0, tg_pose_rnd_std=0.2,
                 includeVelObs=True):

        super().__init__(numControlledJoints, use_IK, action_repeat, obj_name,
                                                 renders, max_steps, obj_pose_rnd_std, tg_pose_rnd_std,
                                                 includeVelObs)

        # Define spaces
        self.observation_space, self.action_space = self.create_gym_spaces()

    def create_gym_spaces(self):
        # Configure observation limits
        obs, obs_lim = self.get_extended_observation()
        observation_low = []
        observation_high = []
        for el in obs_lim:
            observation_low.extend([el[0]])
            observation_high.extend([el[1]])

        goal_obs = self.get_goal_observation()

        # Configure the observation space
        observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-10, 10, shape=goal_obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-10, 10, shape=goal_obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(np.array(observation_low), np.array(observation_high), dtype='float32'),
        ))

        # Configure action space
        action_dim = self._robot.get_action_dim()
        action_bound = 1
        action_high = np.array([action_bound] * action_dim)
        action_space = spaces.Box(-action_high, action_high, dtype='float32')

        return observation_space, action_space

    def reset(self):
        self.reset_simulation()

        # --- sample target pose --- #
        world_obs, _ = self._world.get_observation()
        self._target_pose = self.sample_tg_pose(world_obs[:3])
        self.debug_gui()

        obs = self.get_goal_observation()
        scaled_obs = scale_gym_data(self.observation_space['observation'], obs['observation'])
        obs['observation'] = scaled_obs
        return obs

    def get_goal_observation(self):
        obs, _ = self.get_extended_observation()
        world_observation, _ = self._world.get_observation()

        return {
            'observation': np.array(obs),
            'achieved_goal': np.array(world_observation[:3]),
            'desired_goal': np.array(self._target_pose),
        }

    def step(self, action):
        # apply action on the robot
        self.apply_action(action)

        obs = self.get_goal_observation()
        scaled_obs = scale_gym_data(self.observation_space['observation'], obs['observation'])
        obs['observation'] = scaled_obs

        info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
        }

        done = self._termination() or info['is_success']
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

        return obs, reward, done, info

    def _termination(self):
        if self._env_step_counter > self._max_steps:
            return np.float32(1.0)

        return np.float32(0.)

    def _is_success(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal[:3], goal[:3])

        return d <= self._target_dist_min

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal[:3], goal[:3])

        return -(d > self._target_dist_min).astype(np.float32)
