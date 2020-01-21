# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv
from pybullet_robot_envs.envs.world_envs.fetch_env import get_objects_list, WorldFetchEnv

import pybullet_data
import pybullet_robot_envs.robot_data
from pkg_resources import parse_version

from pybullet_robot_envs.envs.utils import goal_distance

import math as m
import random
import numpy as np
import time

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class pandaPushGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self,
                 use_IK=0,
                 discrete_action=0,
                 action_repeat=1,
                 obj_name="cube_small",
                 renders=False,
                 max_steps=1000,
                 dist_delta=0.03, joint_action_space=7,
                 obj_pose_rnd_std=0, tg_pose_rnd_std=0,
                 includeVelObs=True):

        self._timeStep = 1. / 240.

        self.action_dim = []
        self._discrete_action = discrete_action
        self._use_IK = use_IK
        self._action_repeat = action_repeat
        self._observation = []
        self._env_step_counter = 0
        self._renders = renders
        self._max_steps = max_steps
        self.terminated = False

        self._target_pose = []
        self._target_dist_max = 0.3
        self._target_dist_min = 0.1
        self._p = p
        self._tg_pose_rnd_std = tg_pose_rnd_std
        self.includeVelObs = includeVelObs

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5, 90, -60, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        # Load robot
        self._robot = pandaEnv(use_IK=self._use_IK, basePosition=[0.2, 0, 0.625],
                               joint_action_space=joint_action_space,
                               includeVelObs=self.includeVelObs)

        # Load world environment
        self._world = WorldFetchEnv(obj_name=obj_name, obj_pose_rnd_std=obj_pose_rnd_std,
                                    workspace_lim=self._robot._workspace_lim)

        # limit robot workspace to table plane
        self._robot._workspace_lim[2][0] = self._world.get_table_height()

        # Define spaces
        self._observation_space, self._action_space = self.create_spaces()

        self.seed()
        self.reset()

    def create_spaces(self):
        # Configure observation limits
        obs, obs_lim = self.get_extended_observation()
        observation_dim = len(obs)

        observation_low = []
        observation_high = []
        for el in obs_lim:
            observation_low.extend([el[0]])
            observation_high.extend([el[1]])

        # Configure the observation space
        observation_space = spaces.Box(np.array(observation_low), np.array(observation_high), dtype='float32')

        # Configure action space
        self.action_dim = self._robot.get_action_dimension()

        if self._discrete_action:
            self.action_space = spaces.Discrete(self.action_dim)
        else:
            action_bound = 0.05
            action_high = np.array([action_bound] * self.action_dim)
            action_space = spaces.Box(-action_high, action_high, dtype='float32')

        return observation_space, action_space

    def reset(self):
        self.terminated = False

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        self._env_step_counter = 0

        self._robot.reset()
        self._world.reset()

        self._target_pose = self._sample_pose()

        p.setGravity(0, 0, -9.8)

        # Let the world run for a bit
        for _ in range(500):
            p.stepSimulation()

        self._robot.debug_gui()
        self._world.debug_gui()
        self._debug_gui()

        self._observation = self.get_extended_observation()
        return np.array(self._observation)

    def get_extended_observation(self):
        self._observation = []
        observation_lim = []

        # get observation form robot and world
        robot_observation, robot_obs_lim = self._robot.get_observation()
        world_observation, world_obs_lim = self._world.get_observation()

        self._observation.extend(list(robot_observation))
        self._observation.extend(list(world_observation))
        observation_lim.extend(robot_obs_lim)
        observation_lim.extend(world_obs_lim)

        # target position
        self._observation.extend(list(self._target_pose))
        observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])

        return np.array(self._observation), observation_lim

    def step(self, action):
        # set new action
        new_action = np.clip(action, self._action_space.low, self._action_space.high)

        for _ in range(self._action_repeat):
            if self._use_IK:
                robot_obs, _ = self._robot.get_observation()

                if self._robot._control_orientation:
                    new_action = np.add(robot_obs[:6], new_action)
                else:
                    new_action = np.add(robot_obs[:3], new_action)

            self._robot.apply_action(new_action)
            p.stepSimulation()

            if self._termination():
                break

            self._env_step_counter += 1

            if self._renders:
                time.sleep(self._timeStep)

        self._observation, _ = self.get_extended_observation()

        done = self._termination()
        reward = self._compute_reward()

        return self._observation, np.array([reward]), np.array(done), {}

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return np.array([])

        base_pos, _ = self._p.getBasePositionAndOrientation(self._robot.robot_id)

        cam_dist = 1.3
        cam_yaw = 180
        cam_pitch = -40
        RENDER_HEIGHT = 720
        RENDER_WIDTH = 960

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=cam_dist,
                                                                yaw=cam_yaw,
                                                                pitch=cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)

        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        world_obs, _ = self._world.get_observation()

        d = goal_distance(np.array(world_obs[:3]), np.array(self._target_pose))

        if d <= self._target_dist_min:
            self.terminated = True
            print('------------->>> success!')
            print('final reward')
            print(self._compute_reward())

        if self.terminated or self._env_step_counter > self._max_steps:
            return True

        return False

    def _compute_reward(self):

        reward = np.float(0.0)
        robot_obs, _ = self._robot.get_observation()
        world_obs, _ = self._world.get_observation()

        d1 = goal_distance(np.array(robot_obs[:3]), np.array(world_obs[:3]))
        d2 = goal_distance(np.array(world_obs[:3]), np.array(self._target_pose))

        reward = -d1 - d2
        # print("--------")
        # print(reward)
        # print("--------")
        if d2 <= self._target_dist_min:
            reward = np.float32(1000.0) + (100 - d2*80)
        return reward

    def _sample_pose(self):
        # ws_lim = self._ws_lim
        x_min = self._world._ws_lim[0][0] + 0.064668
        x_max = self._world._ws_lim[0][1] - 0.05

        px = x_min + 0.2 * (x_max - x_min)
        py = self._world._ws_lim[1][0] + 0.5 * (self._world._ws_lim[1][1] - self._world._ws_lim[1][0])
        pz = self._world.get_table_height()

        if self._tg_pose_rnd_std > 0:
            # Add a Gaussian noise to position
            mu, sigma = 0, self._tg_pose_rnd_std
            noise = np.random.normal(mu, sigma, 2)

            px = px + noise[0]
            px = np.clip(px, x_min, x_max)

            py = py + noise[1]
            py = np.clip(py, self._ws_lim[1][0], self._ws_lim[1][1])

        pose = (px, py, pz)

        return pose

    def _debug_gui(self):
        p.addUserDebugLine(self._target_pose, [self._target_pose[0] + 0.1, self._target_pose[1], self._target_pose[2]], [1, 0, 0])
        p.addUserDebugLine(self._target_pose, [self._target_pose[0], self._target_pose[1] + 0.1, self._target_pose[2]], [0, 1, 0])
        p.addUserDebugLine(self._target_pose, [self._target_pose[0], self._target_pose[1], self._target_pose[2] + 0.1], [0, 0, 1])
