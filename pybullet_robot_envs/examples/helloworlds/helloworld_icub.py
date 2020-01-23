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
import time
import math as m


def main():
    # Open GUI and set pybullet_data in the path
    p.connect(p.GUI)
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

    # Load other objects
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1.1, 0.0, 0.0])
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "lego/lego.urdf"), [0.5, 0.0, 0.8])

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
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), info[8], info[9], init_pos[i]/180*m.pi))

    while True:
        new_pos = []
        for i in jointIds:
            new_pos.append(p.readUserDebugParameter(i))
        p.setJointMotorControlArray(icubId, jointIds, p.POSITION_CONTROL, targetPositions=new_pos, forces=[50]*joints_num)

        p.stepSimulation()
        time.sleep(0.01)


if __name__ == '__main__':
    main()
