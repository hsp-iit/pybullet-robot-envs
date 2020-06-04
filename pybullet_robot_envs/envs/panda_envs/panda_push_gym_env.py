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
from pybullet_robot_envs.envs.world_envs.world_env import get_objects_list, WorldEnv
from pybullet_robot_envs.envs.utils import goal_distance, scale_gym_data

import math as m
import numpy as np
import time


class pandaPushGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self,
                 numControlledJoints=7,
                 use_IK=0,
                 action_repeat=1,
                 obj_name=get_objects_list()[1],
                 renders=False,
                 max_steps=1000,
                 obj_pose_rnd_std=0.0,
                 tg_pose_rnd_std=0.0,
                 includeVelObs=True):

        self._timeStep = 1. / 240.

        self.action_dim = []
        self._use_IK = use_IK
        self._action_repeat = action_repeat
        self._observation = []
        self._env_step_counter = 0
        self._renders = renders
        self._max_steps = max_steps
        self.terminated = False

        self._target_pose = [0.0] * 3
        self._target_dist_max = 0.3
        self._target_dist_min = 0.1
        self._tg_pose_rnd_std = tg_pose_rnd_std
        self.includeVelObs = includeVelObs

        if self._renders:
            self._physics_client_id = p.connect(p.SHARED_MEMORY)
            if self._physics_client_id < 0:
                self._physics_client_id = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5, 90, -60, [0.52, -0.2, -0.33], physicsClientId=self._physics_client_id)
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        # Load robot
        self._robot = pandaEnv(self._physics_client_id, use_IK=self._use_IK, joint_action_space=numControlledJoints)

        # Load world environment
        self._world = WorldEnv(self._physics_client_id,
                                    obj_name=obj_name, obj_pose_rnd_std=obj_pose_rnd_std,
                                    workspace_lim=self._robot.get_workspace())

        # limit robot workspace to table plane
        workspace = self._robot.get_workspace()
        workspace[2][0] = self._world.get_table_height()-0.2
        self._robot.set_workspace(workspace)

        # Define spaces
        self.observation_space, self.action_space = self.create_gym_spaces()

        self.seed()
        # self.reset()

    def create_gym_spaces(self):
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

        obs, _ = self.get_extended_observation()
        scaled_obs = scale_gym_data(self.observation_space, obs)
        return scaled_obs

    def reset_simulation(self):
        self.terminated = 0

        # --- reset simulation --- #
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._physics_client_id)
        p.setTimeStep(self._timeStep, physicsClientId=self._physics_client_id)
        self._env_step_counter = 0

        p.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)

        # --- reset robot --- #
        self._robot.reset()

        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        # --- reset world --- #
        self._world.reset()

        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        if self._use_IK:
            self._hand_pose = self._robot._home_hand_pose

        # --- draw some reference frames in the simulation for debugging --- #
        self._robot.debug_gui()
        self._world.debug_gui()
        p.stepSimulation(physicsClientId=self._physics_client_id)

    def get_extended_observation(self):
        self._observation = []
        observation_lim = []

        # ----------------------------------- #
        # --- Robot and world observation --- #
        # ----------------------------------- #
        robot_observation, robot_obs_lim = self._robot.get_observation()
        world_observation, world_obs_lim = self._world.get_observation()

        self._observation.extend(list(robot_observation))
        self._observation.extend(list(world_observation))
        observation_lim.extend(robot_obs_lim)
        observation_lim.extend(world_obs_lim)

        # ----------------------------------------- #
        # --- Object pose wrt hand c.o.m. frame --- #
        # ----------------------------------------- #
        inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3],
                                                       p.getQuaternionFromEuler(robot_observation[3:6]))

        obj_pos_in_hand, obj_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn, world_observation[:3],
                                                                p.getQuaternionFromEuler(world_observation[3:6]))

        obj_euler_in_hand = p.getEulerFromQuaternion(obj_orn_in_hand)

        self._observation.extend(list(obj_pos_in_hand))
        self._observation.extend(list(obj_euler_in_hand))
        observation_lim.extend([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
        observation_lim.extend([[0, 2*m.pi], [0, 2*m.pi], [0, 2*m.pi]])

        # ------------------- #
        # --- Target pose --- #
        # ------------------- #
        self._observation.extend(self._target_pose)
        observation_lim.extend(world_obs_lim[:3])

        return np.array(self._observation), observation_lim

    def apply_action(self, action):
        # process action and send it to the robot

        action = scale_gym_data(self.action_space, np.array(action))

        for _ in range(self._action_repeat):
            robot_obs, _ = self._robot.get_observation()

            if self._use_IK:
                if not self._robot._control_orientation:
                    action *= 0.005
                    new_action = np.add(self._hand_pose[:3], action)

                else:
                    action[:3] *= 0.005
                    action[3:6] *= 0.01

                    new_action = np.add(self._hand_pose, action)

                    # constraint rotation inside limits
                    eu_lim = self._robot.get_rotation_lim()
                    new_action[3:6] = [min(eu_lim[0][1], max(eu_lim[0][0], new_action[3])),
                                       min(eu_lim[1][1], max(eu_lim[1][0], new_action[4])),
                                       min(eu_lim[2][1], max(eu_lim[2][0], new_action[5]))]

                # constraint position inside workspace
                ws_lim = self._robot.get_workspace()
                new_action[:3] = [
                    min(ws_lim[0][1], max(ws_lim[0][0], new_action[0])),
                    min(ws_lim[1][1], max(ws_lim[1][0], new_action[1])),
                    min(ws_lim[2][1], max(ws_lim[2][0], new_action[2]))]

                # Update hand_pose to new pose
                self._hand_pose = new_action

            else:
                action *= 0.05

                n_tot_joints = len(self._robot._joint_name_to_ids.items())  # arm  + fingers
                n_joints_to_control = self._robot.get_action_dim()  # only arm
                joint_curr_pos = robot_obs[-n_tot_joints:]
                new_action = np.add(joint_curr_pos[:n_joints_to_control], action)

            # -------------------------- #
            # --- send pose to robot --- #
            # -------------------------- #
            self._robot.apply_action(new_action)
            p.stepSimulation(physicsClientId=self._physics_client_id)
            time.sleep(self._timeStep)

            if self._termination():
                break

            self._env_step_counter += 1

    def step(self, action):

        # apply action on the robot
        self.apply_action(action)

        obs, _ = self.get_extended_observation()
        scaled_obs = scale_gym_data(self.observation_space, obs)

        done = self._termination()
        reward = self._compute_reward()

        return scaled_obs, np.array(reward), np.array(done), {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._world.seed(seed)
        self._robot.seed(seed)
        return [seed]

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return np.array([])

        base_pos, _ = self._p.getBasePositionAndOrientation(self._robot.robot_id,
                                                            physicsClientId=self._physics_client_id)

        cam_dist = 1.3
        cam_yaw = 180
        cam_pitch = -40
        RENDER_HEIGHT = 640
        RENDER_WIDTH = 480

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=cam_dist,
                                                                yaw=cam_yaw,
                                                                pitch=cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2,
                                                                physicsClientId=self._physics_client_id)

        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1, farVal=100.0,
                                                         physicsClientId=self._physics_client_id)

        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                                                  physicsClientId=self._physics_client_id)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        world_obs, _ = self._world.get_observation()
        d = goal_distance(np.array(world_obs[:3]), np.array(self._target_pose))

        if d <= self._target_dist_min:
            self.terminated = 1
            print('------------->>> success!')
            print('final reward')
            print(self._compute_reward())

            return np.float32(1.0)

        if self.terminated or self._env_step_counter > self._max_steps:
            return np.float32(1.0)

        return np.float32(0.0)

    def _compute_reward(self):
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

    def sample_tg_pose(self, obj_pos):
        # get workspace limits
        ws_lim = self._world.get_workspace()
        x_min = ws_lim[0][0] + 0.07
        x_max = ws_lim[0][1] - 0.07
        y_min = ws_lim[1][0]
        y_max = ws_lim[1][1]

        # fixed target pose wrt object pose ...
        px = obj_pos[0] + 0.05
        py = obj_pos[1] + 0.05
        pz = obj_pos[2]

        # ... or random target pose
        if self._tg_pose_rnd_std > 0:
            # Add a Gaussian noise to position
            mu, sigma = 0, self._tg_pose_rnd_std
            noise = np.random.normal(mu, sigma, 2)

            px = obj_pos[0] + noise[0]
            py = obj_pos[1] + noise[1]

        px = np.clip(px, x_min, x_max)
        py = np.clip(py, y_min, y_max)

        pose = (px, py, pz)

        return pose

    def change_physics_params(self, obj_mass, obj_friction, obj_dumping, robot_damping):
        p.changeDynamics(self._world.obj_id, linkIndex=-1, mass=obj_mass, lateralFriction=obj_friction,
                         linearDamping=obj_dumping)

        for i in range(self._robot._num_dof_no_fingers):
            p.changeDynamics(self._robot.robot_id, linkIndex=i, linearDamping=robot_damping)
        return 0

    def debug_gui(self):
        p.addUserDebugLine(self._target_pose, [self._target_pose[0] + 0.1, self._target_pose[1], self._target_pose[2]], [1, 0, 0],
                           physicsClientId=self._physics_client_id)
        p.addUserDebugLine(self._target_pose, [self._target_pose[0], self._target_pose[1] + 0.1, self._target_pose[2]], [0, 1, 0],
                           physicsClientId=self._physics_client_id)
        p.addUserDebugLine(self._target_pose, [self._target_pose[0], self._target_pose[1], self._target_pose[2] + 0.1], [0, 0, 1],
                           physicsClientId=self._physics_client_id)
