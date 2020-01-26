# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0,currentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import math as m
import pybullet as p

from pybullet_robot_envs.envs.icub_envs.icub_env import iCubEnv
from pybullet_robot_envs.envs.icub_envs.icub_push_gym_env import iCubPushGymEnv
from pybullet_robot_envs.envs.world_envs.fetch_env import get_objects_list, WorldFetchEnv

from pybullet_robot_envs.envs.utils import goal_distance


class iCubPushGymGoalEnv(gym.GoalEnv, iCubPushGymEnv):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self,
                 action_repeat=1,
                 use_IK=1,
                 discrete_action=0,
                 control_arm='l',
                 control_orientation=0,
                 obj_name=get_objects_list()[1],
                 obj_pose_rnd_std=0,
                 tg_pose_rnd_std=0,
                 renders=False,
                 max_steps=2000,
                 reward_type=1):

        self._time_step = 1. / 240.

        self._control_arm = control_arm
        self._discrete_action = discrete_action
        self._use_IK = 1 if self._discrete_action else use_IK
        self._control_orientation = control_orientation
        self._action_repeat = action_repeat
        self._observation = []
        self._hand_pose = []

        self._env_step_counter = 0
        self._renders = renders
        self._max_steps = max_steps
        self._last_frame_time = 0
        self.terminated = 0

        self._tg_pose = np.zeros(3)
        self._tg_pose_rnd_std = tg_pose_rnd_std
        self._target_dist_min = 0.03
        self._reward_type = reward_type

        # Initialize PyBullet simulator
        self._p = p
        if self._renders:
            self._cid = p.connect(p.SHARED_MEMORY)
            if (self._cid<0):
                self._cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5, 90, -60, [0.0, -0.0, -0.0])
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        else:
            self._cid = p.connect(p.DIRECT)

        # Load robot
        self._robot = iCubEnv(use_IK=self._use_IK, control_arm=self._control_arm,
                              control_orientation=self._control_orientation)

        # Load world environment
        self._world = WorldFetchEnv(obj_name=obj_name, obj_pose_rnd_std=obj_pose_rnd_std,
                                    workspace_lim=self._robot._workspace_lim)

        # limit iCub workspace to table plane
        self._robot._workspace_lim[2][0] = self._world.get_table_height()

        # Define spaces
        self.observation_space, self.action_space = self.create_spaces()

        # initialize simulation environment
        self.seed()
        self.reset()

    def reset(self):
        self.terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)
        self._env_step_counter = 0

        p.setGravity(0, 0, -9.8)

        self._robot.reset()
        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation()

        self._world.reset()
        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation()

        self._tg_pose = np.array(self._sample_pose())

        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation()

        self._robot.debug_gui()
        self._world.debug_gui()
        self.debug_gui()

        if self._use_IK:
            self._hand_pose = self._robot._home_hand_pose

        return self.get_extended_observation()

    def create_spaces(self):
        # Configure observation limits
        obs = self.get_extended_observation()

        # Configure the observation space
        observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        # Configure action space
        action_dim = self._robot.get_action_dim()
        if self._discrete_action:
            action_space = spaces.Discrete(action_dim*2+1)
        else:
            action_bound = 0.005
            action_high = np.array([action_bound] * action_dim)
            action_space = spaces.Box(-action_high, action_high, dtype='float32')

        return observation_space, action_space

    def get_extended_observation(self):
        self._observation = []

        # get observation form robot and world
        robot_observation, _ = self._robot.get_observation()
        world_observation, _ = self._world.get_observation()

        # relative object position wrt hand c.o.m. frame
        inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3],
                                                       p.getQuaternionFromEuler(robot_observation[3:6]))
        obj_pos_in_hand, obj_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn, world_observation[:3],
                                                                p.getQuaternionFromEuler(world_observation[3:6]))
        obj_euler_in_hand = p.getEulerFromQuaternion(obj_orn_in_hand)

        self._observation.extend(list(robot_observation))
        self._observation.extend(list(world_observation))

        self._observation.extend(list(obj_pos_in_hand))
        self._observation.extend(list(obj_euler_in_hand))

        self._observation.extend(list(self._tg_pose))

        return {
            'observation': np.array(self._observation),
            'achieved_goal': np.array(world_observation[:3]),
            'desired_goal': self._tg_pose.copy(),
        }

    def step(self, action):
        # apply action on the robot
        self.apply_action(action)

        obs = self.get_extended_observation()

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self._tg_pose.copy()),
        }

        done = self._termination() or info['is_success']
        reward = self._compute_reward(obs['achieved_goal'], self._tg_pose.copy(), info)

        return obs, reward, done, info

    def _termination(self):
        if self._env_step_counter > self._max_steps:
            return np.float32(1.0)

        return np.float32(0.)

    def _is_success(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal[:3], goal[:3])

        return d <= self._target_dist_min

    def _compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal[:3], goal[:3])

        return -(d > self._target_dist_min).astype(np.float32)