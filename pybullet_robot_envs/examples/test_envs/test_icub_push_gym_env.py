# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

#!/usr/bin/env python
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_robot_envs.envs.icub_envs.icub_reach_gym_env import iCubReachGymEnv
from pybullet_robot_envs.envs.icub_envs.icub_push_gym_env import iCubPushGymEnv
from pybullet_robot_envs.envs.icub_envs.icub_push_gym_goal_env import iCubPushGymGoalEnv

import argparse
import pybullet as p


def main(joint_control, arm, random_policy):

    use_IK = 0 if joint_control else 1

    env = iCubReachGymEnv(renders=True, control_arm=arm, use_IK=use_IK, control_orientation=1, obj_pose_rnd_std=0.05)
    env.reset()

    motorsIds = []

    if not random_policy:
        if use_IK:
            dv = 1
            motorsIds.append(p.addUserDebugParameter("lhPosX", -dv, dv, 0.0))
            motorsIds.append(p.addUserDebugParameter("lhPosY", -dv, dv, 0.0))
            motorsIds.append(p.addUserDebugParameter("lhPosZ", -dv, dv, 0.0))
            motorsIds.append(p.addUserDebugParameter("lhRollx", -dv, dv, 0.0))
            motorsIds.append(p.addUserDebugParameter("lhPitchy", -dv, dv, 0.0))
            motorsIds.append(p.addUserDebugParameter("lhYawz", -dv, dv, 0.0))

        else:
            dv = 1
            joints_idx = env._robot._joints_to_control

            for j in joints_idx:
                info = p.getJointInfo(env._robot.robot_id, j)
                jointName = info[1]
                motorsIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -dv, dv, 0.0))

    done = False
    for t in range(10000000):
        # env.render()

        if not random_policy:
            action = []
            for motorId in motorsIds:
                action.append(p.readUserDebugParameter(motorId))

        else:
            action = env.action_space.sample()

        state, reward, done, info = env.step(action)

        if t % 100 == 0:
            print("reward ", reward)

        if done:
            print("done ", done)
            env.reset()


def parser_args():
    """
    parse the arguments for running the experiment
    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--joint', action='store_const', const=1, dest="joint_control",
                        help='action is in joint position space. Default: cartesian control space')

    parser.add_argument('--arm', action='store', default='l', dest="arm",
                        help="choose arm to control: 'l' - left or 'r'-right")

    parser.add_argument('--random_policy', action='store_const', const=1, dest="random_policy",
                        help="Simulate a random policy instead of using sliders to control the action")

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parser_args()
    print('args')
    print(args)
    main(**args)
