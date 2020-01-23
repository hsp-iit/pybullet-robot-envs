# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import icub_model_pybullet

import numpy as np
import quaternion
import math as m


class iCubEnv:

    def __init__(self, use_IK=0, control_arm='l', control_orientation=0):

        self._use_IK = use_IK
        self._control_orientation = control_orientation
        self._use_simulation = 1

        self._indices_torso = range(12, 15)
        self._indices_left_arm = range(15, 22)
        self._indices_right_arm = range(25, 32)
        self._indices_head = range(22, 25)
        self._end_eff_idx = []

        self._home_pos_torso = [0.0, 0.0, 0.0]  # degrees
        self._home_pos_head = [0.47, 0, 0]

        self._home_left_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]
        self._home_right_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]

        self._home_hand_pose = []

        self._workspace_lim = [[0.25, 0.52], [-0.3, 0.3], [0.5, 1.0]]
        self._eu_lim = [[-m.pi, m.pi], [-m.pi, m.pi], [-m.pi, m.pi]]

        self._control_arm = control_arm if control_arm == 'r' or control_arm == 'l' else 'l'  # left arm by default

        self.robot_id = None

        self._num_joints = 0
        self._motor_idxs = 0
        self.ll, self.ul, self.jr, self.rs = None, None, None, None

        self.reset()

    def reset(self):

        self.robot_id = p.loadSDF(os.path.join(icub_model_pybullet.get_data_path(), "icub_model.sdf"))[0]
        assert self.robot_id is not None, "Failed to load the icub model"

        self._num_joints = p.getNumJoints(self.robot_id)

        # set constraint between base_link and world
        constr_id = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                                       parentFramePosition=[0, 0, 0],
                                       childFramePosition=[p.getBasePositionAndOrientation(self.robot_id)[0][0],
                                                           p.getBasePositionAndOrientation(self.robot_id)[0][1],
                                                           p.getBasePositionAndOrientation(self.robot_id)[0][2] * 1.2],
                                       parentFrameOrientation=p.getBasePositionAndOrientation(self.robot_id)[1])

        # Set all joints initial values
        for count, i in enumerate(self._indices_torso):
            p.resetJointState(self.robot_id, i, self._home_pos_torso[count] / 180 * m.pi)

        for count, i in enumerate(self._indices_head):
            p.resetJointState(self.robot_id, i, self._home_pos_head[count] / 180 * m.pi)

        for count, i in enumerate(self._indices_left_arm):
            p.resetJointState(self.robot_id, i, self._home_left_arm[count] / 180 * m.pi)

        for count, i in enumerate(self._indices_right_arm):
            p.resetJointState(self.robot_id, i, self._home_right_arm[count] / 180 * m.pi)

        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

        # save indices of only the joints to control
        control_arm_indices = self._indices_left_arm if self._control_arm == 'l' else self._indices_right_arm
        self._motor_idxs = [i for i in self._indices_torso] + [j for j in control_arm_indices]

        self._end_eff_idx = self._indices_left_arm[-1] if self._control_arm == 'l' else self._indices_right_arm[-1]

        self._motor_names = []
        for i in self._indices_torso:
            jointInfo = p.getJointInfo(self.robot_id, i)
            if jointInfo[3] > -1:
                self._motor_names.append(str(jointInfo[1]))
        for i in control_arm_indices:
            jointInfo = p.getJointInfo(self.robot_id, i)
            if jointInfo[3] > -1:
                self._motor_names.append(str(jointInfo[1]))

        # set initial hand pose
        if self._control_arm == 'l':
            self._home_hand_pose = [0.26, 0.3, 0.9, 0, 0, 0]  # x,y,z, roll,pitch,yaw
        else:
            self._home_hand_pose = [0.26, -0.3, 0.9, 0, 0, m.pi]

        # self.eu_lim[0] = np.add(self.eu_lim[0], self.home_hand_pose[3])
        # self.eu_lim[1] = np.add(self.eu_lim[1], self.home_hand_pose[4])
        # self.eu_lim[2] = np.add(self.eu_lim[2], self.home_hand_pose[5])

        if self._use_IK:
            self.apply_action(self._home_hand_pose)

    def delete_simulated_robot(self):
        # Remove the robot from the simulation
        p.removeBody(self.robot_id)

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
        for i in range(self._num_joints):
            jointInfo = p.getJointInfo(self.robot_id, i)

            if jointInfo[3] > -1:
                ll, ul = jointInfo[8:10]
                jr = ul - ll
                # For simplicity, assume resting state == initial state
                rp = p.getJointState(self.robot_id, i)[0]
                lower_limits.append(ll)
                upper_limits.append(ul)
                joint_ranges.append(jr)
                rest_poses.append(rp)

        return lower_limits, upper_limits, joint_ranges, rest_poses

    def get_ws_lim(self):
        return self._workspace_lim

    def get_action_dim(self):
        if not self._use_IK:
            return len(self._motor_idxs)
        if self._control_orientation:
            return 6  # position x,y,z + roll/pitch/yaw of hand frame
        return 3  # position x,y,z

    def get_observation_dim(self):
        return len(self.getObservation())

    def get_observation(self):
        # Cartesian world pos/orn of left hand center of mass
        observation = []
        observation_lim = []
        state = p.getLinkState(self.robot_id, self._end_eff_idx, computeLinkVelocity=1)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)
        vel_l = state[6]
        vel_a = state[7]

        observation.extend(list(pos))
        observation.extend(list(euler))  # roll, pitch, yaw
        #observation.extend(list(vel_l))
        #observation.extend(list(vel_a))

        observation_lim.extend(list(self._workspace_lim))
        observation_lim.extend(list(self._eu_lim))

        joint_states = p.getJointStates(self.robot_id, self._motor_idxs)
        joint_poses = [x[0] for x in joint_states]
        observation.extend(list(joint_poses))
        observation_lim.extend([[self.ll[i], self.ul[i]] for i in self._motor_idxs])

        return observation, observation_lim

    def apply_action(self, action):
        if self._use_IK:

            if not len(action) >= 3:
                raise AssertionError('number of action commands must be minimum 3: (dx,dy,dz)', len(action))

            dx, dy, dz = action[:3]

            new_pos = [min(self._workspace_lim[0][1], max(self._workspace_lim[0][0], dx)),
                       min(self._workspace_lim[1][1], max(self._workspace_lim[1][0], dy)),
                       min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], dz))]

            if not self._control_orientation:
                new_quat_orn = p.getQuaternionFromEuler(self._home_hand_pose[3:6])

            elif len(action) >= 6:
                droll, dpitch, dyaw = action[3:6]

                new_eu_orn = [min(self._eu_lim[0][1], max(self._eu_lim[0][0], droll)),
                              min(self._eu_lim[1][1], max(self._eu_lim[1][0], dpitch)),
                              min(self._eu_lim[2][1], max(self._eu_lim[2][0], dyaw))]

                new_quat_orn = p.getQuaternionFromEuler(new_eu_orn)

            else:
                new_quat_orn = p.getLinkState(self.robot_id, self._end_eff_idx)[5]

            # transform the new pose from COM coordinate to link coordinate
            if self._control_arm is 'r':
                COM_t0_link_hand_pos = (0.064668, -0.0056, -0.022681)
            else:
                COM_t0_link_hand_pos = (-0.064768, -0.00563, -0.02266)

            link_hand_pose = p.multiplyTransforms(new_pos, new_quat_orn,
                                                  COM_t0_link_hand_pos, p.getQuaternionFromEuler((0, 0, 0)))

            # compute joint positions with IK
            jointPoses = p.calculateInverseKinematics(self.robot_id, self._end_eff_idx,
                                                      link_hand_pose[0], link_hand_pose[1],
                                                      lowerLimits=self.ll, upperLimits=self.ul,
                                                      jointRanges=self.jr, restPoses=self.rs)

            # workaround to block joints of not-controlled arm
            joints_to_block = self._indices_left_arm if self._control_arm == 'r' else self._indices_right_arm

            if self._use_simulation:
                for i in range(self._num_joints):
                    if i in joints_to_block:
                        continue

                    jointInfo = p.getJointInfo(self.robot_id, i)
                    if jointInfo[3] > -1:
                        # minimize error is:
                        # error = position_gain * (desired_position - actual_position) +
                        #         velocity_gain * (desired_velocity - actual_velocity)

                        p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=jointPoses[i],
                                                force=50)
            else:
                # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self._num_joints):
                    if i in joints_to_block:
                        continue
                    p.resetJointState(self.robot_id, i, jointPoses[i])

        else:
            if not len(action) == len(self._motor_idxs):
                raise AssertionError('number of motor commands differs from number of motor to control',
                                     len(action), len(self._motor_idxs))

            for idx, val in enumerate(action):
                motor = self._motor_idxs[idx]

                # curr_motor_pos = p.getJointState(self.robot_id, motor)[0]
                new_motor_pos = min(self.ul[motor], max(self.ll[motor], val))

                p.setJointMotorControl2(self.robot_id,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        force=50)

    def debug_gui(self):

        ws = self._workspace_lim
        p1 = [ws[0][0], ws[1][0], ws[2][0]]  # xmin,ymin
        p2 = [ws[0][1], ws[1][0], ws[2][0]]  # xmax,ymin
        p3 = [ws[0][1], ws[1][1], ws[2][0]]  # xmax,ymax
        p4 = [ws[0][0], ws[1][1], ws[2][0]]  # xmin,ymax

        p.addUserDebugLine(p1, p2, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p2, p3, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p3, p4, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p4, p1, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)

        p.addUserDebugLine([0, 0, 0], [0.3, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id, parentLinkIndex=-1)
        p.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id, parentLinkIndex=-1)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.3], [0, 0, 1], parentObjectUniqueId=self.robot_id, parentLinkIndex=-1)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_right_arm[-1])
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_right_arm[-1])
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_right_arm[-1])

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_left_arm[-1])
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_left_arm[-1])
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_left_arm[-1])
