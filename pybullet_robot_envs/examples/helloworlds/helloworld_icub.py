# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import pybullet as p
import pybullet_data
from icub_model_pybullet import model_with_hands
from ycb_objects_models_sim import objects

import time
import math as m
import numpy as np


def main():
    # Open GUI and set pybullet_data in the path
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(1.8, 140, -50, [0.0, -0.0, -0.0])
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(1/240.)

    # Load plane contained in pybullet_data
    p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"))

    # Set gravity for simulation
    p.setGravity(0,0,-9.8)

    # load robot model
    icubId = p.loadSDF(os.path.join(model_with_hands.get_data_path(), "icub_model_with_hands.sdf"))[0]

    # set constraint between base_link and world
    p.createConstraint(icubId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[p.getBasePositionAndOrientation(icubId)[0][0],
                                           p.getBasePositionAndOrientation(icubId)[0][1],
                                           p.getBasePositionAndOrientation(icubId)[0][2] * 1.2],
                       parentFrameOrientation=p.getBasePositionAndOrientation(icubId)[1])

    # --- init_pos for standing --- #
    home_pos_torso = [0.0, 0.0, 0.0]  # degrees
    home_pos_head = [0.47, 0, 0]
    home_left_arm = [-29.4, 28.8, 0, 44.59, 0, 0, 0]
    home_right_arm = [-29.4, 30.4, 0, 44.59, 0, 0, 0]
    home_left_hand = [0] * 20
    home_right_hand = [0] * 20

    init_pos = [0]*12 + home_pos_torso + home_left_arm + home_left_hand + home_pos_head + home_right_arm + home_right_hand

    hand_idx = 51
    # Load other objects
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0])
    p.loadURDF(os.path.join(objects.getDataPath(), 'YcbFoamBrick',  "model.urdf"), [0.5, 0.0, 0.8])

    p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=icubId,
                       parentLinkIndex=hand_idx)
    p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=icubId,
                       parentLinkIndex=hand_idx)
    p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=icubId,
                       parentLinkIndex=hand_idx)

    # add debug slider
    jointIds = []
    paramIds = []
    joints_num = p.getNumJoints(icubId)

    for i in range(joints_num):
        p.resetJointState(icubId, i, init_pos[i]/180*m.pi)

    for i in range(joints_num):
        info = p.getJointInfo(icubId, i)
        jointName = info[1]
        jointIds.append(i)
        # paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), info[8], info[9], init_pos[i]/180*m.pi))

    for _ in range(100):
        p.stepSimulation()

    # pregrasp gesture of the right hand
    p.setJointMotorControlArray(icubId, [50, 68], p.POSITION_CONTROL, targetPositions=[1.4, 1.57], forces=[50, 50])
    for _ in range(70):
        p.stepSimulation()
        time.sleep(0.02)

    # go above the object
    pos_1 = [0.5, 0.0, 0.8]
    quat_1 = p.getQuaternionFromEuler([0, m.pi/2, m.pi/2])
    jointPoses = p.calculateInverseKinematics(icubId, hand_idx, pos_1, quat_1)
    p.setJointMotorControlArray(icubId, jointIds, p.POSITION_CONTROL, targetPositions=jointPoses,
                                forces=[50] * len(jointIds))
    p.setJointMotorControl2(icubId, 50, p.POSITION_CONTROL, targetPosition=1.4, force=50)
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.02)

    # go down to the object
    pos_2 = [0.5, 0.0, 0.675 + 0.064]
    quat_2 = p.getQuaternionFromEuler([0, m.pi/2, m.pi/2])
    jointPoses = p.calculateInverseKinematics(icubId, hand_idx, pos_2, quat_2)
    p.setJointMotorControlArray(icubId, jointIds, p.POSITION_CONTROL, targetPositions=jointPoses,
                                forces=[50] * len(jointIds))
    p.setJointMotorControl2(icubId, 50, p.POSITION_CONTROL, targetPosition=1.4, force=50)
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.02)

    # close fingers
    pos = [0, 0.3, 0.5, 0.9, 0,  0.3, 0.5, 0.9, 0,  0.3, 0.5, 0.9, 0,  0.3, 0.5, 0.9, 1.57, 0.6, 0.4, 0.7]

    steps = [i/100 for i in range(0, 101, 1)]
    for s in steps:
        next_pos = np.multiply(pos, s)
        p.setJointMotorControlArray(icubId, range(52, 72), p.POSITION_CONTROL, targetPositions=next_pos,
                                    forces=[20] * len(range(52, 72)))
        p.setJointMotorControlArray(icubId, [50, 68], p.POSITION_CONTROL, targetPositions=[0.5, 1.57], forces=[50, 50])
        for _ in range(5):
            p.stepSimulation()

    # go up to the object
    pos_2 = [0.5, 0, 0.9]
    quat_2 = p.getQuaternionFromEuler([0, m.pi/2, m.pi/2])
    jointPoses = list(p.calculateInverseKinematics(icubId, hand_idx, pos_2, quat_2))
    jointPoses[-20:] = pos
    p.setJointMotorControlArray(icubId, jointIds, p.POSITION_CONTROL, targetPositions=jointPoses,
                                forces=[50] * len(jointIds))
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.03)

    # go up to the object
    pos_2 = [0.3, -0.2, 0.9]
    quat_2 = p.getQuaternionFromEuler([0, 0, m.pi/2])
    jointPoses = list(p.calculateInverseKinematics(icubId, hand_idx, pos_2, quat_2))
    jointPoses[-20:] = pos
    p.setJointMotorControlArray(icubId, jointIds, p.POSITION_CONTROL, targetPositions=jointPoses,
                                forces=[50] * len(jointIds))
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.03)

    # open fingers
    pos = [0, 0.3, 0.5, 0.9, 0, 0.3, 0.5, 0.9, 0, 0.3, 0.5, 0.9, 0, 0.3, 0.5, 0.9, 1.57, 0.6, 0.4, 0.7]

    steps = [i / 100 for i in range(100, -1, -1)]
    for s in steps:
        next_pos = np.multiply(pos, s)
        p.setJointMotorControlArray(icubId, range(52, 72), p.POSITION_CONTROL, targetPositions=next_pos,
                                    forces=[20] * len(range(52, 72)))
        for _ in range(4):
            p.stepSimulation()

    jointPoses[-20:] = [0]*20

    # reset joints to home position
    jointIds = []
    num_joints = p.getNumJoints(icubId)
    idx = 0
    for i in range(num_joints):
        jointInfo = p.getJointInfo(icubId, i)
        jointName = jointInfo[1]
        jointType = jointInfo[2]

        if jointType is p.JOINT_REVOLUTE or jointType is p.JOINT_PRISMATIC:
            jointIds.append(i)
            paramIds.append(
                p.addUserDebugParameter(jointName.decode("utf-8"), jointInfo[8], jointInfo[9], jointPoses[idx]))
            idx += 1


    while True:
        new_pos = []
        for i in paramIds:
            new_pos.append(p.readUserDebugParameter(i))
        p.setJointMotorControlArray(icubId, jointIds, p.POSITION_CONTROL, targetPositions=new_pos, forces=[50]*len(jointIds))

        p.stepSimulation()
        time.sleep(0.01)


if __name__ == '__main__':
    main()
