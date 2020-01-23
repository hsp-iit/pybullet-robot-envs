# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import pybullet as p
import pybullet_data

import numpy as np
import math as m

class pandaEnv:

    def __init__(self, use_IK=0, basePosition=[-0.6,-0.4,0.625], joint_action_space=7, includeVelObs = True):

        self._use_IK = use_IK
        self._control_orientation = 1
        self._use_simulation = 1
        self._base_position = basePosition
        self._workspace_lim = [[0.1, 1], [-0.4, 0.4], [0.65, 2]]
        self.endEffLink = 11  # 8
        self.joint_action_space = joint_action_space
        self._include_vel_obs = includeVelObs

        self._home_pos_joints = [0, -0.54, 0, -2.6, -0.30, 2, 1, 0.02, 0.02]

        self._num_dof = 7
        self.robot_id = None

        self.reset()

    def reset(self):
        # load robot
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
        self.robot_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),
                                  basePosition=self._base_position, useFixedBase=True, flags=flags)

        # reset joints to home position
        num_joints = p.getNumJoints(self.robot_id)
        idx = 0
        for i in range(num_joints):
            jointInfo = p.getJointInfo(self.robot_id, i)
            jointType = jointInfo[2]
            if jointType is p.JOINT_REVOLUTE or jointType is p.JOINT_PRISMATIC:
                p.resetJointState(self.robot_id, i, self._home_pos_joints[idx])
                idx += 1

        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

        if self._use_IK:
            self._home_endEff_pose = [0.6, 0, 1, m.pi, 0.,  0]  # x,y,z,roll,pitch,yaw

            self.apply_action(self._home_endEff_pose)
            p.stepSimulation()

        self.debug_gui()

    def delete_simulated_robot(self):
        # Remove the robot from the simulation
        p.removeBody(self.robot_id)

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

        for i in range(p.getNumJoints(self.robot_id)):
            jointInfo = p.getJointInfo(self.robot_id, i)
            jointType = jointInfo[2]

            if jointType is p.JOINT_REVOLUTE or jointType is p.JOINT_PRISMATIC:
                ll, ul = jointInfo[8:10]
                jr = ul - ll
                # For simplicity, assume resting state == initial state
                rp = p.getJointState(self.robot_id, i)[0]
                lower_limits.append(ll)
                upper_limits.append(ul)
                joint_ranges.append(jr)
                rest_poses.append(rp)

        return lower_limits, upper_limits, joint_ranges, rest_poses

    def get_action_dimension(self):
        if not self._use_IK:
            return self.joint_action_space
        if self._control_orientation:
            return 6  # position x,y,z + roll/pitch/yaw of hand frame
        return 3  # position x,y,z

    def get_observation_dimension(self):
        return len(self.get_observation())

    def get_observation(self):
        observation = []
        observation_lim = []
        state = p.getLinkState(self.robot_id, self.endEffLink, computeLinkVelocity=1)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))  # roll, pitch, yaw

        observation_lim.extend(list(self._workspace_lim))
        # CHECK THIS! -pi,pi or 0,2*pi?
        observation_lim.extend([[-m.pi, m.pi], [-m.pi, m.pi], [-m.pi, m.pi]])

        if self._include_vel_obs:
            velL = state[6]
            velA = state[7]
            observation.extend(list(velL))
            observation.extend(list(velA))
            observation_lim.extend([[0, 10], [0, 10], [0, 10]])
            observation_lim.extend([[0, 10], [0, 10], [0, 10]])

        jointStates = p.getJointStates(self.robot_id, range(self._num_dof))
        jointPoses = [x[0] for x in jointStates]
        fingerStates = p.getJointStates(self.robot_id, [9, 10])
        fingerPoses = [x[0] for x in fingerStates]

        observation.extend(list(jointPoses))
        observation.extend(list(fingerPoses))
        observation_lim.extend([[self.ll[i], self.ul[i]] for i in range(self._num_dof)])
        observation_lim.extend([[self.ll[i], self.ul[i]] for i in range(self._num_dof, self._num_dof+2)])

        return observation, observation_lim

    def apply_action_fingers(self, action):
        assert len(action) == 2, ('finger joints are 2! The number of actions you passed is ', len(action))

        for i in [9, 10]:
            p.setJointMotorControl2(self.robot_id,
                                    i,
                                    p.POSITION_CONTROL,
                                    targetPosition=action[0],
                                    force=10)

    def apply_action(self, action):

        if self._use_IK:
            assert len(action) == 3 or len(action) == 6, ('IK dim differs from ', len(action))

            dx, dy, dz = action[:3]
            new_pos = [min(self._workspace_lim[0][1], max(self._workspace_lim[0][0], dx)),
                       min(self._workspace_lim[1][1], max(self._workspace_lim[1][0], dy)),
                       min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], dz))]

            if not self._control_orientation:
                quat_orn = p.getQuaternionFromEuler(self._home_endEff_pose[3:6])

            elif len(action) is 6:
                droll, dpitch, dyaw = action[3:]

                eu_orn = [min(m.pi, max(-m.pi, droll)),
                          min(m.pi, max(-m.pi, dpitch)),
                          min(m.pi, max(-m.pi, dyaw))]

                quat_orn = p.getQuaternionFromEuler(eu_orn)

            else:
                quat_orn = p.getLinkState(self.robot_id, self.endEffLink)[5]

            jointPoses = p.calculateInverseKinematics(self.robot_id, self.endEffLink, new_pos, quat_orn)

            if self._use_simulation:
                    for i in range(self._num_dof):
                        jointInfo = p.getJointInfo(self.robot_id, i)
                        if jointInfo[3] > -1:
                            p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                                    jointIndex=i,
                                                    controlMode=p.POSITION_CONTROL,
                                                    targetPosition=jointPoses[i],
                                                    force=500)
            else:
                for i in range(self._num_dof):
                    p.resetJointState(self.robot_id, i, jointPoses[i])

        else:
            assert len(action) == self.joint_action_space, ('number of motor commands differs from number of motor to control', len(action))

            for a in range(len(action)):
                curr_motor_pos = p.getJointState(self.robot_id, a)[0]
                new_motor_pos = curr_motor_pos + action[a]  # supposed to be a delta

                p.setJointMotorControl2(self.robot_id,
                                        a,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        targetVelocity=0,
                                        force=500)

    def debug_gui(self):

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=8)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=8)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=8)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=11)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=11)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=11)
