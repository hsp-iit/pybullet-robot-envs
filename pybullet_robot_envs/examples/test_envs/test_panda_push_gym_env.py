# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_robot_envs.envs.panda_envs.panda_push_gym_env import pandaPushGymEnv
from pybullet_robot_envs import robot_data
import pybullet_data


import time
import math as m

def main():

    use_IK = 0
    discreteAction = 0
    use_IK = 1 if discreteAction else use_IK

    env = pandaPushGymEnv(renders=True, use_IK=use_IK, discrete_action=0, obj_pose_rnd_std=0.0)
    motorsIds = []

    if (env._discrete_action):
        dv = 12
        motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0))
    elif use_IK:
        dv = 1
        motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPosY", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPosZ", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhRollx", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPitchy", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhYawz", -dv, dv, 0.0))
    else:
        dv = 1
        #joints_idx = env._icub.motorIndices

        for j in range(7):
            info = env._p.getJointInfo(env._robot.robot_id, j)
            jointName = info[1]
            motorsIds.append(env._p.addUserDebugParameter(jointName.decode("utf-8"), -dv, dv, 0.0))

    done = False

    for t in range(10000000):
        #env.render()
        action = []
        for motorId in motorsIds:
            action.append(env._p.readUserDebugParameter(motorId))

        action = int(action[0]) if discreteAction else action

        #print(env.step(action))

        state, reward, done, _ = env.step(action)
        if t%10==0:
            print("reward ", reward)
            print("done ", done)

if __name__ == '__main__':
    main()
