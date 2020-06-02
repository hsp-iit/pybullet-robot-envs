# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
from pybullet_robot_envs.robot_data import iCub
from gym.utils import seeding

import math as m
import numpy as np

class iCubEnv:

    initial_positions = {
        # Left leg
        'l_knee': 0.0,
        'l_ankle_pitch': 0.0, 'l_ankle_roll': 0.0,
        'l_hip_pitch': 0.0, 'l_hip_roll': 0.0, 'l_hip_yaw': 0.0,
        # Right leg
        'r_knee': 0.0,
        'r_ankle_pitch': 0.0, 'r_ankle_roll': 0.0,
        'r_hip_pitch': 0.0, 'r_hip_roll': 0.0, 'r_hip_yaw': 0.0,
        # Head
        'neck_pitch': 0.008, 'neck_roll': 0.0, 'neck_yaw': 0.0,
        # Torso
        'torso_pitch': 0.0, 'torso_roll': 0.0, 'torso_yaw': 0.0,
        # Left arm
        'l_shoulder_pitch': -0.51, 'l_shoulder_roll': 0.7, 'l_shoulder_yaw': 0,
        'l_elbow': 1.22,
        'l_wrist_pitch': 0.0, 'l_wrist_prosup': 0.0, 'l_wrist_yaw': 0.0,
        # Right arm
        'r_shoulder_pitch': -0.51, 'r_shoulder_roll': 0.7, 'r_shoulder_yaw': 0,
        'r_elbow': 1.22,
        'r_wrist_pitch': 0.0, 'r_wrist_prosup': 0.0, 'r_wrist_yaw': 0.0,
    }

    joint_groups = {'l_leg': ['l_knee', 'l_ankle_pitch', 'l_hip_pitch'],
                    'r_leg': ['r_knee', 'r_ankle_pitch', 'r_hip_pitch'],
                    'head': ['neck_pitch', 'neck_roll', 'neck_yaw'],
                    'torso': ['torso_pitch', 'torso_roll', 'torso_yaw'],
                    'l_arm': ['l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw',
                              'l_elbow', 'l_wrist_pitch', 'l_wrist_prosup', 'l_wrist_yaw'],
                    'r_arm': ['r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw',
                              'r_elbow', 'r_wrist_pitch', 'r_wrist_prosup', 'r_wrist_yaw'],
                    }

    def __init__(self, physicsClientId, use_IK=0, control_arm='l', control_orientation=1, control_eu_or_quat=0):

        self._physics_client_id = physicsClientId
        self._use_IK = use_IK
        self._control_orientation = control_orientation
        self._control_eu_or_quat = control_eu_or_quat
        self._control_arm = control_arm if control_arm == 'r' or control_arm == 'l' else 'l'  # left arm by default

        self.end_eff_idx = []

        self._workspace_lim = [[0.1, 0.45], [-0.3, 0.3], [0.5, 1.0]]
        self._eu_lim = [[-m.pi/2, m.pi/2], [-m.pi/2, m.pi/2], [-m.pi/2, m.pi/2]]

        # set initial hand pose
        if self._control_arm == 'l':
            self._home_hand_pose = [0.3, 0.26, 0.8, 0, 0, 0]  # x, y, z, roll, pitch, yaw
            self._eu_lim = [[-m.pi / 2, m.pi / 2], [-m.pi / 2, m.pi / 2], [-m.pi / 2, m.pi / 2]]

        else:
            self._home_hand_pose = [0.3, -0.26, 0.8, 0, 0, m.pi]
            self._eu_lim = [[-m.pi / 2, m.pi / 2], [-m.pi / 2, m.pi / 2], [m.pi / 2, 3 / 2 * m.pi]]

        self._joints_to_control = []
        self._joints_to_block = []
        self._joint_name_to_ids = {}

        self.robot_id = None

        self.ll, self.ul, self.jr, self.rs, self.jd = None, None, None, None, None

        self.seed()
        self.reset()

    def reset(self):
        # TODO: would it be better to reset the pose of the robot instead of re-loading the sdf model? The problem could be then related to the objects that changes at every reset(). check

        # Load robot model
        self.robot_id = p.loadSDF(os.path.join(iCub.get_data_path(), "icub_model.sdf"),
                                  physicsClientId=self._physics_client_id)[0]

        assert self.robot_id is not None, "Failed to load the icub model"

        # set constraint between base_link and world
        constr_id = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                                       parentFramePosition=[0, 0, 0],
                                       childFramePosition=[p.getBasePositionAndOrientation(self.robot_id)[0][0],
                                                           p.getBasePositionAndOrientation(self.robot_id)[0][1],
                                                           p.getBasePositionAndOrientation(self.robot_id)[0][2] * 1.2],
                                       parentFrameOrientation=p.getBasePositionAndOrientation(self.robot_id)[1],
                                       physicsClientId=self._physics_client_id)

        # Set all joints initial values
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
        for i in range(num_joints):

            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()

                self._joint_name_to_ids[joint_name] = i

                p.resetJointState(self.robot_id, i, self.initial_positions[joint_name])
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=self.initial_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0,
                                        physicsClientId=self._physics_client_id)

        # save indices of the joints to control:
        if len(self._joints_to_control) is 0:

            for joint_name in self._joint_name_to_ids.keys():

                # - torso
                if joint_name in self.joint_groups['torso']:
                    self._joints_to_control.append(self._joint_name_to_ids[joint_name])

                # - right or left arm, depending on the arm to control
                elif (joint_name in self.joint_groups['l_arm'] and self._control_arm == 'l') or \
                     (joint_name in self.joint_groups['r_arm'] and self._control_arm == 'r'):

                    self._joints_to_control.append(self._joint_name_to_ids[joint_name])

                else:
                    self._joints_to_block.append(self._joint_name_to_ids[joint_name])

                # - Save end-effector index:
                if (self._control_arm == 'l' and joint_name == 'l_wrist_yaw') or \
                   (self._control_arm == 'r' and joint_name == 'r_wrist_yaw'):

                    self.end_eff_idx = self._joint_name_to_ids[joint_name]

        # get joint ranges
        self.ll, self.ul, self.jr, self.rs, self.jd = self.get_joint_ranges()

        if self._use_IK:
            self.apply_action(self._home_hand_pose)

        p.stepSimulation()

    def delete_simulated_robot(self):
        # Remove the robot from the simulation
        p.removeBody(self.robot_id, physicsClientId=self._physics_client_id)

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses, joint_dumping = [], [], [], [], []

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
            joint_dumping.append(0.1 if self._joint_name_to_ids[joint_name] in self._joints_to_control else 100.)

        return lower_limits, upper_limits, joint_ranges, rest_poses, joint_dumping

    def get_workspace(self):
        return [i[:] for i in self._workspace_lim]

    def set_workspace(self, ws):
        self._workspace_lim = [i[:] for i in ws]

    def get_rotation_lim(self):
        return [i[:] for i in self._eu_lim]

    def set_rotation_lim(self, eu):
        self._eu_lim = [i[:] for i in eu]

    def get_action_dim(self):
        if not self._use_IK:
            return len(self._joints_to_control)

        if self._control_orientation and self._control_eu_or_quat is 0:
            return 6  # position x,y,z + roll/pitch/yaw of hand frame

        elif self._control_orientation and self._control_eu_or_quat is 1:
            return 7  # position x,y,z + quat of hand frame

        return 3  # position x,y,z

    def get_observation_dim(self):
        return len(self.get_observation())

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
        quat = state[1]

        observation.extend(list(pos))
        observation_lim.extend(list(self._workspace_lim))

        # cartesian orientation
        if self._control_eu_or_quat is 0:
            euler = p.getEulerFromQuaternion(quat)

            observation.extend(list(euler))
            observation_lim.extend(self._eu_lim)

        else:
            observation.extend(list(quat))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        # --------------------------------- #
        # --- Cartesian linear velocity --- #
        # --------------------------------- #
        vel_l = state[6]

        observation.extend(list(vel_l))
        observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])

        # ------------------- #
        # --- Joint poses --- #
        # ------------------- #
        joint_states = p.getJointStates(self.robot_id, self._joints_to_control, physicsClientId=self._physics_client_id)
        joint_poses = [x[0] for x in joint_states]

        observation.extend(list(joint_poses))
        observation_lim.extend([[self.ll[i], self.ul[i]] for i, idx in enumerate(self._joint_name_to_ids.values())
                                if idx in self._joints_to_control])

        return observation, observation_lim

    def _com_to_link_hand_frame(self):
        if self._control_arm is 'r':
            com_T_link_hand = ((0.064668, -0.0056, -0.022681), (0., 0., 0., 1.))
        else:
            com_T_link_hand = ((-0.064768, -0.00563, -0.02266), (0., 0., 0., 1.))

        return com_T_link_hand

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
            new_pos = [min(self._workspace_lim[0][1], max(self._workspace_lim[0][0], dx)),
                       min(self._workspace_lim[1][1], max(self._workspace_lim[1][0], dy)),
                       min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], dz))]

            # if orientation is not under control, keep it fixed
            if not self._control_orientation:
                new_quat_orn = p.getQuaternionFromEuler(self._home_hand_pose[3:6])

            # otherwise, if it is defined as euler angles
            elif len(action) == 6:
                droll, dpitch, dyaw = action[3:6]

                new_eu_orn = [min(self._eu_lim[0][1], max(self._eu_lim[0][0], droll)),
                              min(self._eu_lim[1][1], max(self._eu_lim[1][0], dpitch)),
                              min(self._eu_lim[2][1], max(self._eu_lim[2][0], dyaw))]

                new_quat_orn = p.getQuaternionFromEuler(new_eu_orn)

            # otherwise, if it is define as quaternion
            elif len(action) == 7:
                new_quat_orn = action[3:7]

            # otherwise, use current orientation
            else:
                new_quat_orn = p.getLinkState(self.robot_id, self.end_eff_idx, physicsClientId=self._physics_client_id)[5]

            # --- compute joint positions with IK --- #
            # transform the new pose from COM coordinate to link coordinate, because calculateInverseKinematics() wants the link coordinate
            com_T_link_hand = self._com_to_link_hand_frame()

            link_hand_pose = p.multiplyTransforms(new_pos, new_quat_orn, com_T_link_hand[0], com_T_link_hand[1])

            jointPoses = p.calculateInverseKinematics(self.robot_id, self.end_eff_idx,
                                                      link_hand_pose[0], link_hand_pose[1],
                                                      jointDamping=self.jd,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001,
                                                      physicsClientId=self._physics_client_id)
            jointPoses = np.asarray(jointPoses)
            self.rs = np.asarray(self.rs)

            idxs_to_block = [i for i, idx in enumerate(self._joint_name_to_ids.values()) if idx in self._joints_to_block]
            jointPoses[idxs_to_block] = self.rs[idxs_to_block]
            # --- set joint control --- #
            if max_vel == -1:
                p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                            jointIndices=self._joint_name_to_ids.values(),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=jointPoses,
                                            positionGains=[0.2]*len(jointPoses),
                                            velocityGains=[1.0]*len(jointPoses),
                                            physicsClientId=self._physics_client_id)

            else:
                for i, idx in enumerate(self._joint_name_to_ids.values()):
                    p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                            jointIndex=idx,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=jointPoses[i],
                                            positionGain=0.2, velocityGain=1.0,
                                            maxVelocity=max_vel,
                                            physicsClientId=self._physics_client_id)

        else:
            # --------------------- #
            # --- Joint control --- #
            # --------------------- #

            if not len(action) == len(self._joints_to_control):
                raise AssertionError('number of motor commands differs from number of motor to control',
                                     len(action), len(self._joints_to_control))

            for i, val in enumerate(action):
                motor = self._joints_to_control[i]

                for j, idx in enumerate(self._joint_name_to_ids.values()):
                    if motor == idx:
                        new_motor_pos = min(self.ul[j], max(self.ll[j], val))
                        break

                p.setJointMotorControl2(self.robot_id,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        positionGain=0.5, velocityGain=1.0,
                                        maxVelocity=max_vel,
                                        physicsClientId=self._physics_client_id)

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
        p.addUserDebugLine(p3, p4, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0,physicsClientId=self._physics_client_id)
        p.addUserDebugLine(p4, p1, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.3, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id, parentLinkIndex=-1,
                           physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id, parentLinkIndex=-1,
                           physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.3], [0, 0, 1], parentObjectUniqueId=self.robot_id, parentLinkIndex=-1,
                           physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
