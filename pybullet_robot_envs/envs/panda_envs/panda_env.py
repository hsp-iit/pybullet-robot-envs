# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import pybullet as p
from gym.utils import seeding
from pybullet_robot_envs.robot_data import franka_panda

import numpy as np
import math as m


class pandaEnv:

    initial_positions = {
        'panda_joint1': 0.0, 'panda_joint2': -0.54, 'panda_joint3': 0.0,
        'panda_joint4': -2.6, 'panda_joint5': -0.30, 'panda_joint6': 2.0,
        'panda_joint7': 1.0, 'panda_finger_joint1': 0.02, 'panda_finger_joint2': 0.02,
    }

    def __init__(self, physicsClientId, use_IK=0, base_position=(0.0, 0, 0.625), control_orientation=1, control_eu_or_quat=0,
                 joint_action_space=9, includeVelObs=True):

        self._physics_client_id = physicsClientId
        self._use_IK = use_IK
        self._control_orientation = control_orientation
        self._base_position = base_position

        self.joint_action_space = joint_action_space
        self._include_vel_obs = includeVelObs
        self._control_eu_or_quat = control_eu_or_quat

        self._workspace_lim = [[0.3, 0.65], [-0.3, 0.3], [0.65, 1.5]]
        self._eu_lim = [[-m.pi, m.pi], [-m.pi, m.pi], [-m.pi, m.pi]]

        self.end_eff_idx = 11  # 8

        self._home_hand_pose = []

        self._num_dof = 7
        self._joint_name_to_ids = {}
        self.robot_id = None

        self.seed()
        self.reset()

    def reset(self):
        # Load robot model
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        self.robot_id = p.loadURDF(os.path.join(franka_panda.get_data_path(), "panda_model.urdf"),
                                   basePosition=self._base_position, useFixedBase=True, flags=flags,
                                   physicsClientId=self._physics_client_id)

        assert self.robot_id is not None, "Failed to load the panda model"

        # reset joints to home position
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
        idx = 0
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()

                self._joint_name_to_ids[joint_name] = i

                p.resetJointState(self.robot_id, i, self.initial_positions[joint_name], physicsClientId=self._physics_client_id)
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                        targetPosition=self.initial_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0,
                                        physicsClientId=self._physics_client_id)

                idx += 1

        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

        if self._use_IK:

            self._home_hand_pose = [0.2, 0.0, 0.8,
                                    min(m.pi, max(-m.pi, m.pi)),
                                    min(m.pi, max(-m.pi, 0)),
                                    min(m.pi, max(-m.pi, 0))]

            self.apply_action(self._home_hand_pose)
            p.stepSimulation(physicsClientId=self._physics_client_id)

    def delete_simulated_robot(self):
        # Remove the robot from the simulation
        p.removeBody(self.robot_id, physicsClientId=self._physics_client_id)

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

        for joint_name in self._joint_name_to_ids.keys():
            jointInfo = p.getJointInfo(self.robot_id, self._joint_name_to_ids[joint_name], physicsClientId=self._physics_client_id)

            ll, ul = jointInfo[8:10]
            jr = ul - ll
            # For simplicity, assume resting state == initial state
            rp = self.initial_positions[joint_name]
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(jr)
            rest_poses.append(rp)

        return lower_limits, upper_limits, joint_ranges, rest_poses

    def get_action_dim(self):
        if not self._use_IK:
            return self.joint_action_space

        if self._control_orientation and self._control_eu_or_quat is 0:
            return 6  # position x,y,z + roll/pitch/yaw of hand frame

        elif self._control_orientation and self._control_eu_or_quat is 1:
            return 7  # position x,y,z + quat of hand frame

        return 3  # position x,y,z

    def get_observation_dim(self):
        return len(self.get_observation())

    def get_workspace(self):
        return [i[:] for i in self._workspace_lim]

    def set_workspace(self, ws):
        self._workspace_lim = [i[:] for i in ws]

    def get_rotation_lim(self):
        return [i[:] for i in self._eu_lim]

    def set_rotation_lim(self, eu):
        self._eu_lim = [i[:] for i in eu]

    def get_observation(self):
        # Create observation state
        observation = []
        observation_lim = []

        # Get state of the end-effector link
        state = p.getLinkState(self.robot_id, self.end_eff_idx, computeLinkVelocity=1,
                               computeForwardKinematics=1, physicsClientId=self._physics_client_id)

        # ------------------------- #
        # --- Cartesian 6D pose --- #
        # ------------------------- #
        pos = state[0]
        orn = state[1]

        observation.extend(list(pos))
        observation_lim.extend(list(self._workspace_lim))

        # cartesian orientation
        if self._control_eu_or_quat is 0:
            euler = p.getEulerFromQuaternion(orn)
            observation.extend(list(euler))  # roll, pitch, yaw
            observation_lim.extend(self._eu_lim)
        else:
            observation.extend(list(orn))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        # --------------------------------- #
        # --- Cartesian linear velocity --- #
        # --------------------------------- #
        if self._include_vel_obs:
            # standardize by subtracting the mean and dividing by the std

            vel_std = [0.04, 0.07, 0.03]
            vel_mean = [0.0, 0.01, 0.0]

            vel_l = np.subtract(state[6], vel_mean)
            vel_l = np.divide(vel_l, vel_std)

            observation.extend(list(vel_l))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])

        # ------------------- #
        # --- Joint poses --- #
        # ------------------- #

        jointStates = p.getJointStates(self.robot_id, self._joint_name_to_ids.values(), physicsClientId=self._physics_client_id)
        jointPoses = [x[0] for x in jointStates]

        observation.extend(list(jointPoses))
        observation_lim.extend([[self.ll[i], self.ul[i]] for i in range(0, len(self._joint_name_to_ids.values()))])

        return observation, observation_lim

    def pre_grasp(self):
        self.apply_action_fingers([0.04, 0.04])

    def grasp(self, obj_id=None):
        self.apply_action_fingers([0.0, 0.0], obj_id)

    def apply_action_fingers(self, action, obj_id=None):
        # move finger joints in position control
        assert len(action) == 2, ('finger joints are 2! The number of actions you passed is ', len(action))

        idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

        # use object id to check contact force and eventually stop the finger motion
        if obj_id is not None:
            _, forces = self.check_contact_fingertips(obj_id)
            # print("contact forces {}".format(forces))

            if forces[0] >= 20.0:
                action[0] = p.getJointState(self.robot_id, idx_fingers[0], physicsClientId=self._physics_client_id)[0]

            if forces[1] >= 20.0:
                action[1] = p.getJointState(self.robot_id, idx_fingers[1], physicsClientId=self._physics_client_id)[0]

        for i, idx in enumerate(idx_fingers):
            p.setJointMotorControl2(self.robot_id,
                                    idx,
                                    p.POSITION_CONTROL,
                                    targetPosition=action[i],
                                    force=10,
                                    maxVelocity=1,
                                    physicsClientId=self._physics_client_id)

    def apply_action(self, action, max_vel=-1):

        if self._use_IK:
            # ------------------ #
            # --- IK control --- #
            # ------------------ #

            if not (len(action) == 3 or len(action) == 6 or len(action) == 7):
                raise AssertionError('number of action commands must be \n- 3: (dx,dy,dz)'
                                     '\n- 6: (dx,dy,dz,droll,dpitch,dyaw)'
                                     '\n- 7: (dx,dy,dz,qx,qy,qz,w)'
                                     '\ninstead it is: ', len(action))

            # --- Constraint end-effector pose inside the workspace --- #

            dx, dy, dz = action[:3]
            new_pos = [dx, dy,
                       min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], dz))]

            # if orientation is not under control, keep it fixed
            if not self._control_orientation:
                new_quat_orn = p.getQuaternionFromEuler(self._home_hand_pose[3:6])

            # otherwise, if it is defined as euler angles
            elif len(action) == 6:
                droll, dpitch, dyaw = action[3:]

                eu_orn = [min(m.pi, max(-m.pi, droll)),
                          min(m.pi, max(-m.pi, dpitch)),
                          min(m.pi, max(-m.pi, dyaw))]

                new_quat_orn = p.getQuaternionFromEuler(eu_orn)

            # otherwise, if it is define as quaternion
            elif len(action) == 7:
                new_quat_orn = action[3:7]

            # otherwise, use current orientation
            else:
                new_quat_orn = p.getLinkState(self.robot_id, self.end_eff_idx, physicsClientId=self._physics_client_id)[5]

            # --- compute joint positions with IK --- #
            jointPoses = p.calculateInverseKinematics(self.robot_id, self.end_eff_idx, new_pos, new_quat_orn,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001,
                                                      physicsClientId=self._physics_client_id)

            # --- set joint control --- #
            if max_vel == -1:
                p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                            jointIndices=self._joint_name_to_ids.values(),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=jointPoses,
                                            positionGains=[0.2] * len(jointPoses),
                                            velocityGains=[1.0] * len(jointPoses),
                                            physicsClientId=self._physics_client_id)

            else:
                for i in range(self._num_dof):
                    p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                            jointIndex=i,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=jointPoses[i],
                                            maxVelocity=max_vel,
                                            physicsClientId=self._physics_client_id)

        else:
            # --------------------- #
            # --- Joint control --- #
            # --------------------- #

            assert len(action) == self.joint_action_space, ('number of motor commands differs from number of motor to control', len(action))

            joint_idxs = tuple(self._joint_name_to_ids.values())
            for i, val in enumerate(action):
                motor = joint_idxs[i]
                new_motor_pos = min(self.ul[i], max(self.ll[i], val))

                p.setJointMotorControl2(self.robot_id,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        positionGain=0.5, velocityGain=1.0,
                                        physicsClientId=self._physics_client_id)

    def check_collision(self, obj_id):
        # check if there is any collision with an object

        contact_pts = p.getContactPoints(obj_id, self.robot_id, physicsClientId=self._physics_client_id)

        # check if the contact is on the fingertip(s)
        n_fingertips_contact, _ = self.check_contact_fingertips(obj_id)

        return (len(contact_pts) - n_fingertips_contact) > 0

    def check_contact_fingertips(self, obj_id):
        # check if there is any contact on the internal part of the fingers, to control if they are correctly touching an object

        idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

        p0 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[0], physicsClientId=self._physics_client_id)
        p1 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[1], physicsClientId=self._physics_client_id)

        p0_contact = 0
        p0_f = [0]
        if len(p0) > 0:
            # get cartesian position of the finger link frame in world coordinates
            w_pos_f0 = p.getLinkState(self.robot_id, idx_fingers[0], physicsClientId=self._physics_client_id)[4:6]
            f0_pos_w = p.invertTransform(w_pos_f0[0], w_pos_f0[1])

            for pp in p0:
                # compute relative position of the contact point wrt the finger link frame
                f0_pos_pp = p.multiplyTransforms(f0_pos_w[0], f0_pos_w[1], pp[6], f0_pos_w[1])

                # check if contact in the internal part of finger
                if f0_pos_pp[0][1] <= 0.001 and f0_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p0_contact += 1
                    p0_f.append(pp[9])

        p0_f_mean = np.mean(p0_f)

        p1_contact = 0
        p1_f = [0]
        if len(p1) > 0:
            w_pos_f1 = p.getLinkState(self.robot_id, idx_fingers[1], physicsClientId=self._physics_client_id)[4:6]
            f1_pos_w = p.invertTransform(w_pos_f1[0], w_pos_f1[1])

            for pp in p1:
                # compute relative position of the contact point wrt the finger link frame
                f1_pos_pp = p.multiplyTransforms(f1_pos_w[0], f1_pos_w[1], pp[6], f1_pos_w[1])

                # check if contact in the internal part of finger
                if f1_pos_pp[0][1] >= -0.001 and f1_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p1_contact += 1
                    p1_f.append(pp[9])

        p1_f_mean = np.mean(p0_f)

        return (p0_contact > 0) + (p1_contact > 0), (p0_f_mean, p1_f_mean)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def debug_gui(self):
        ws = self._workspace_lim
        p1 = [ws[0][0], ws[1][0], ws[2][0]]  # xmin,ymin
        p2 = [ws[0][1], ws[1][0], ws[2][0]]  # xmax,ymin
        p3 = [ws[0][1], ws[1][1], ws[2][0]]  # xmax,ymax
        p4 = [ws[0][0], ws[1][1], ws[2][0]]  # xmin,ymax

        p.addUserDebugLine(p1, p2, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)
        p.addUserDebugLine(p2, p3, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)
        p.addUserDebugLine(p3, p4, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)
        p.addUserDebugLine(p4, p1, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
