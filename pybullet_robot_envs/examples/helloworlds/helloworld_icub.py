# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import pybullet_data
from pybullet_robot_envs.envs.icub_envs.icub_env_with_hands import iCubHandsEnv
from pybullet_object_models import ycb_objects

import time
import math as m


def main():

    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(1.8, 120, -50, [0.0, -0.0, -0.0])
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    sim_timestep = 1.0/240
    p.setTimeStep(sim_timestep)

    # Load plane contained in pybullet_data
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))

    # Set gravity for simulation
    p.setGravity(0, 0, -9.8)

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    robot = iCubHandsEnv(physics_client_id, use_IK=1, control_arm='r')
    p.stepSimulation()

    # -------------------------- #
    # --- Load other objects --- #
    # -------------------------- #

    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0])
    p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbFoamBrick', "model.urdf"), [0.5, -0.03, 0.7])

    # Run the world for a bit
    for _ in range(100):
        p.stepSimulation()

    # ------------------ #
    # --- Start Demo --- #
    # ------------------ #

    robot.pre_grasp()

    for _ in range(10):
        p.stepSimulation()
        time.sleep(sim_timestep)

    # 1: go above the object
    pos_1 = [0.49, 0.0, 0.8]
    quat_1 = p.getQuaternionFromEuler([0, 0, m.pi/2])

    robot.apply_action(pos_1 + list(quat_1), max_vel=5)
    robot.pre_grasp()

    for _ in range(60):
        p.stepSimulation()
        time.sleep(sim_timestep)

    # 2: turn hand above the object
    pos_2 = [0.485, 0.0, 0.72]
    quat_2 = p.getQuaternionFromEuler([m.pi/2, 1/3*m.pi, -m.pi])

    robot.apply_action(pos_2 + list(quat_2), max_vel=5)
    robot.pre_grasp()

    for _ in range(60):
        p.stepSimulation()
        time.sleep(sim_timestep)

    # 3: close fingers
    pos_cl = [0, 0.6, 0.8, 1.0, 0,  0.6, 0.8, 1.0, 0,  0.6, 0.8, 1.0, 0,  0.6, 0.8, 1.0, 1.57, 0.8, 0.5, 0.8]

    robot.grasp(pos_cl)

    for _ in range(60):
        p.stepSimulation()
        time.sleep(sim_timestep)

    # 4: go up
    pos_4 = [0.45, 0, 0.9]
    quat_4 = p.getQuaternionFromEuler([m.pi/2, 1/3*m.pi, -m.pi])

    robot.apply_action(pos_4 + list(quat_4), max_vel=5)
    robot.grasp(pos_cl)

    for _ in range(60):
        p.stepSimulation()
        time.sleep(sim_timestep)

    # 5: go right
    pos_5 = [0.3, -0.2, 0.9]
    quat_5 = p.getQuaternionFromEuler([0.0, 0.0, m.pi/2])

    robot.apply_action(pos_5 + list(quat_5), max_vel=5)
    robot.grasp(pos_cl)

    for _ in range(60):
        p.stepSimulation()
        time.sleep(sim_timestep)

    # 6: open hand
    robot.pre_grasp()

    for _ in range(50):
        p.stepSimulation()
        time.sleep(sim_timestep)

    # ------------------------ #
    # --- Play with joints --- #
    # ------------------------ #

    param_ids = []
    joint_ids = []
    num_joints = p.getNumJoints(robot.robot_id)

    joint_states = p.getJointStates(robot.robot_id, range(0, num_joints))
    joint_poses = [x[0] for x in joint_states]
    idx = 0
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot.robot_id, i)
        joint_name = joint_info[1]
        joint_type = joint_info[2]

        if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
            param_ids.append(p.addUserDebugParameter(joint_name.decode("utf-8"), joint_info[8], joint_info[9], joint_poses[i]))
            joint_ids.append(i)
            idx += 1

    while True:
        new_pos = []
        for i in param_ids:
            new_pos.append(p.readUserDebugParameter(i))
        p.setJointMotorControlArray(robot.robot_id, joint_ids, p.POSITION_CONTROL, targetPositions=new_pos)

        p.stepSimulation()
        time.sleep(sim_timestep)


if __name__ == '__main__':
    main()
