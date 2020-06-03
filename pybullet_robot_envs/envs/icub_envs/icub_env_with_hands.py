# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
from pybullet_robot_envs.robot_data import iCub
from pybullet_robot_envs.envs.icub_envs.icub_env import iCubEnv

import numpy as np
import math as m
import time


class iCubHandsEnv(iCubEnv):

    initial_positions = {
        'l_hand::l_aij6': 0.0, 'l_hand::l_aij3': 0.0, 'l_hand::l_aij4': 0.0, 'l_hand::l_aij5':0.0,
        'l_hand::l_lij6': 0.0, 'l_hand::l_lij3': 0.0, 'l_hand::l_lij4': 0.0, 'l_hand::l_lij5': 0.0,
        'l_hand::l_mj6': 0.0, 'l_hand::l_mj3': 0.0, 'l_hand::l_mj4': 0.0, 'l_hand::l_mj5': 0.0,
        'l_hand::l_rij6': 0.0, 'l_hand::l_rij3': 0.0, 'l_hand::l_rij4': 0.0, 'l_hand::l_rij5': 0.0,
        'l_hand::l_tj2': 0.0, 'l_hand::l_tj4': 0.0, 'l_hand::l_tj5': 0.0, 'l_hand::l_tj6': 0.0,
        'r_hand::r_aij6': 0.0, 'r_hand::r_aij3': 0.0, 'r_hand::r_aij4': 0.0, 'r_hand::r_aij5': 0.0,
        'r_hand::r_lij6': 0.0, 'r_hand::r_lij3': 0.0, 'r_hand::r_lij4': 0.0, 'r_hand::r_lij5': 0.0,
        'r_hand::r_mj6': 0.0, 'r_hand::r_mj3': 0.0, 'r_hand::r_mj4': 0.0, 'r_hand::r_mj5': 0.0,
        'r_hand::r_rij6': 0.0, 'r_hand::r_rij3': 0.0, 'r_hand::r_rij4': 0.0, 'r_hand::r_rij5': 0.0,
        'r_hand::r_tj2': 0.0, 'r_hand::r_tj4': 0.0, 'r_hand::r_tj5': 0.0, 'r_hand::r_tj6': 0.0,
        }

    initial_positions.update(iCubEnv.initial_positions)

    joint_groups = {
        'l_hand': ['l_hand::l_aij6', 'l_hand::l_aij3', 'l_hand::l_aij4', 'l_hand::l_aij5',
                    'l_hand::l_lij6', 'l_hand::l_lij3', 'l_hand::l_lij4', 'l_hand::l_lij5',
                    'l_hand::l_mj6', 'l_hand::l_mj3', 'l_hand::l_mj4', 'l_hand::l_mj5',
                    'l_hand::l_rij6', 'l_hand::l_rij3', 'l_hand::l_rij4', 'l_hand::l_rij5',
                    'l_hand::l_tj2', 'l_hand::l_tj4', 'l_hand::l_tj5', 'l_hand::l_tj6'],
        'r_hand': ['r_hand::r_aij6', 'r_hand::r_aij3', 'r_hand::r_aij4', 'r_hand::r_aij5',
                    'r_hand::r_lij6', 'r_hand::r_lij3', 'r_hand::r_lij4', 'r_hand::r_lij5',
                    'r_hand::r_mj6', 'r_hand::r_mj3', 'r_hand::r_mj4', 'r_hand::r_mj5',
                    'r_hand::r_rij6', 'r_hand::r_rij3', 'r_hand::r_rij4', 'r_hand::r_rij5',
                    'r_hand::r_tj2', 'r_hand::r_tj4', 'r_hand::r_tj5', 'r_hand::r_tj6']
        }

    joint_groups.update(iCubEnv.joint_groups)

    def __init__(self, physicsClientId, use_IK=0, control_arm='l', control_orientation=1, control_eu_or_quat=0):

        self._physics_client_id = physicsClientId
        self._use_IK = use_IK
        self._control_orientation = control_orientation
        self._control_eu_or_quat = control_eu_or_quat

        self._home_hand_pose = []
        self._home_motor_pose = []

        self._grasp_pos = [0, 0.75, 0.5, 0.5, 0, 0.75, 0.5, 0.5, 0, 0.75, 0.5, 0.5, 0, 0.75, 0.5, 0.5, 1.57, 0.4, 0.2, 0.07]

        self._workspace_lim = [[0.15, 0.50], [-0.3, 0.3], [0.5, 1.0]]
        self._eu_lim = [[-m.pi, m.pi], [-m.pi, m.pi], [-m.pi, m.pi]]

        self._control_arm = control_arm if control_arm == 'r' or control_arm == 'l' else 'l'  # left arm by default
        self._joints_to_control = []
        self._joints_to_block = []
        self._joint_name_to_ids = {}

        self.robot_id = None

        # set initial hand pose
        if self._control_arm == 'l':
            self._home_hand_pose = [0.2, 0.3, 0.8, -m.pi, 0, -m.pi/2]   # x,y,z, roll,pitch,yaw
            self._eu_lim = [[-3/2*m.pi, -m.pi/2], [-m.pi / 2, m.pi / 2], [0, -m.pi]]
        else:
            self._home_hand_pose = [0.2, -0.3, 0.8, 0, 0,  m.pi/2]
            self._eu_lim = [[-m.pi / 2, m.pi / 2], [-m.pi / 2, m.pi / 2], [0, m.pi]]

        self.seed()
        self.reset()

    def reset(self):

        self.robot_id = p.loadSDF(os.path.join(iCub.get_data_path(), "icub_model_with_hands.sdf"),
                                  physicsClientId=self._physics_client_id)[0]
        assert self.robot_id is not None, "Failed to load the icub model"

        self._num_joints = p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)

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

                # torso
                if joint_name in self.joint_groups['torso']:
                    self._joints_to_control.append(self._joint_name_to_ids[joint_name])

                # + left arm and left hand, depending on the arm to control
                elif joint_name in self.joint_groups['l_arm'] or joint_name in self.joint_groups['l_hand'] \
                        and self._control_arm == 'l':

                    self._joints_to_control.append(self._joint_name_to_ids[joint_name])

                # or + right arm and right hand, depending on the arm to control
                elif joint_name in self.joint_groups['r_arm'] or joint_name in self.joint_groups['r_hand'] \
                        and self._control_arm == 'r':

                    self._joints_to_control.append(self._joint_name_to_ids[joint_name])

                else:
                    self._joints_to_block.append(self._joint_name_to_ids[joint_name])

                # - Save end-effector index:
                if (self._control_arm == 'l' and joint_name == 'l_wrist_yaw') or \
                        (self._control_arm == 'r' and joint_name == 'r_wrist_yaw'):

                    self.end_eff_idx = self._joint_name_to_ids[joint_name]

        self.ll, self.ul, self.jr, self.rs, self.jd = self.get_joint_ranges()

        # self.eu_lim[0] = np.add(self.eu_lim[0], self.home_hand_pose[3])
        # self.eu_lim[1] = np.add(self.eu_lim[1], self.home_hand_pose[4])
        # self.eu_lim[2] = np.add(self.eu_lim[2], self.home_hand_pose[5])

        if self._use_IK:
            self.apply_action(self._home_hand_pose)
            p.stepSimulation(physicsClientId=self._physics_client_id)

    def _com_to_link_hand_frame(self):
        if self._control_arm is 'r':
            com_T_link_hand = ((-0.011682, 0.051682, -0.000577), (0.0, 0.0, 0.0, 1.0))
        else:
            com_T_link_hand = ((-0.011682, 0.051355, 0.000577), (0.0, 0.0, 0.0, 1.0))

        return com_T_link_hand

    def open_hand(self):
        # open fingers

        if self._control_arm is 'l':
            idx_fingers = [self._joint_name_to_ids[jn] for jn in self.joint_groups['l_hand']]
        else:
            idx_fingers = [self._joint_name_to_ids[jn] for jn in self.joint_groups['r_hand']]

        pos = [0.0] * len(idx_fingers)

        p.setJointMotorControlArray(self.robot_id, idx_fingers, p.POSITION_CONTROL,
                                    targetPositions=pos,
                                    positionGains=[0.1] * len(idx_fingers),
                                    velocityGains=[1.0] * len(idx_fingers),
                                    physicsClientId=self._physics_client_id)

    def pre_grasp(self):
        # move fingers to pre-grasp configuration

        if self._control_arm is 'l':
            idx_thumb = self._joint_name_to_ids['l_hand::l_tj2']
            idx_fingers = [self._joint_name_to_ids[jn] for jn in self.joint_groups['l_hand']]
        else:
            idx_thumb = self._joint_name_to_ids['r_hand::r_tj2']
            idx_fingers = [self._joint_name_to_ids[jn] for jn in self.joint_groups['r_hand']]

        pos = [0.0] * len(idx_fingers)
        for i, idx in enumerate(idx_fingers):
            if idx == idx_thumb:
                pos[i] = 1.57

        p.setJointMotorControlArray(self.robot_id, idx_fingers, p.POSITION_CONTROL,
                                    targetPositions=pos,
                                    positionGains=[0.1] * len(idx_fingers),
                                    velocityGains=[1.0] * len(idx_fingers),
                                    physicsClientId=self._physics_client_id)

    def grasp(self, pos=None):
        # close fingers

        if self._control_arm is 'l':
            idx_thumb = self._joint_name_to_ids['l_hand::l_tj2']
            idx_fingers = [self._joint_name_to_ids[jn] for jn in self.joint_groups['l_hand']]
        else:
            idx_thumb = self._joint_name_to_ids['r_hand::r_tj2']
            idx_fingers = [self._joint_name_to_ids[jn] for jn in self.joint_groups['r_hand']]

        # # set also position to other joints to avoid weird movements
        # not_idx_fingers = [idx for idx in self._joints_to_control if idx not in idx_fingers]
        #
        # joint_states = p.getJointStates(self.robot_id, not_idx_fingers)
        # joint_poses = [x[0] for x in joint_states]
        # p.setJointMotorControlArray(self.robot_id, not_idx_fingers, p.POSITION_CONTROL,
        #                             targetPositions=joint_poses,
        #                             positionGains=[0.1] * len(not_idx_fingers),
        #                             velocityGains=[1.0] * len(not_idx_fingers),
        #                             physicsClientId=self._physics_client_id)

        position_control = True
        if position_control:
            if pos is None:
                pos = self._grasp_pos
            p.setJointMotorControlArray(self.robot_id, idx_fingers, p.POSITION_CONTROL,
                                        targetPositions=pos,
                                        positionGains=[0.1] * len(idx_fingers),
                                        velocityGains=[1.0] * len(idx_fingers),
                                        forces=[10] * len(idx_fingers),
                                        physicsClientId=self._physics_client_id)

        else:
            # vel = [0, 1, 1, 1.2, 0, 1, 1, 1.2, 0, 1, 1, 1.2, 0, 1, 1, 1.2, 1.57, 1.5, 1.1, 1.1]
            vel = [0, 0.5, 0.6, 0.6, 0, 0.5, 0.6, 0.6, 0, 0.5, 0.6, 0.6, 0, 0.5, 0.6, 0.6, 0, 0.5, 0.6, 0.6]

            p.setJointMotorControlArray(self.robot_id, idx_fingers, p.VELOCITY_CONTROL,
                                        targetVelocities=vel,
                                        positionGains=[0.1] * len(idx_fingers),
                                        velocityGains=[1.0] * len(idx_fingers),
                                        physicsClientId=self._physics_client_id)

    def check_contact_fingertips(self, obj_id):
        # finger tips
        tips_idxs = [3, 7, 11, 15, 19]

        if self._control_arm is 'l':
            idx_fingers = [self._joint_name_to_ids[jn] for jn in self.joint_groups['l_hand']]
        else:
            idx_fingers = [self._joint_name_to_ids[jn] for jn in self.joint_groups['r_hand']]

        p0 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[tips_idxs[0]], physicsClientId=self._physics_client_id)
        p1 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[tips_idxs[1]], physicsClientId=self._physics_client_id)
        p2 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[tips_idxs[2]], physicsClientId=self._physics_client_id)
        p3 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[tips_idxs[3]], physicsClientId=self._physics_client_id)
        p4 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[tips_idxs[4]], physicsClientId=self._physics_client_id)

        fingers_in_contact = 0

        p0_f = 0
        if len(p0) > 0:
            fingers_in_contact += 1
            # print("p0! {}".format(len(p0)))
            for pp in p0:
                p0_f += pp[9]
            p0_f /= len(p0)
            # print("\t\t p0 normal force! {}".format(p0_f))

        p1_f = 0
        if len(p1) > 0:
            fingers_in_contact += 1
            # print("p1! {}".format(len(p1)))
            for pp in p1:
                p1_f += pp[9]
            p1_f /= len(p1)
            # print("\t\t p1 normal force! {}".format(p1_f))

        p2_f = 0
        if len(p2) > 0:
            fingers_in_contact += 1
            # print("p2! {}".format(len(p2)))
            for pp in p2:
                p2_f += pp[9]
            p2_f /= len(p2)
            # print("\t\t p2 normal force! {}".format(p2_f))

        p3_f = 0
        if len(p3) > 0:
            fingers_in_contact += 1
            # print("p3! {}".format(len(p3)))
            for pp in p3:
                p3_f += pp[9]
            p3_f /= len(p3)
            # print("\t\t p3 normal force! {}".format(p3_f))

        p4_f = 0
        if len(p4) > 0:
            fingers_in_contact += 1
            # print("p4! {}".format(len(p4)))
            for pp in p4:
                p4_f += pp[9]
            p4_f /= len(p4)
            # print("\t\t p4 normal force! {}".format(p4_f))

        return fingers_in_contact, [p0_f, p1_f, p2_f, p3_f, p4_f]

    def check_collision(self, obj_id):
        # check if there is any collision with an object

        contact_pts = p.getContactPoints(obj_id, self.robot_id, physicsClientId=self._physics_client_id)

        # check if the contact is on the fingertip(s)
        n_fingertips_contact, _ = self.check_contact_fingertips(obj_id)

        return (len(contact_pts) - n_fingertips_contact) > 0

