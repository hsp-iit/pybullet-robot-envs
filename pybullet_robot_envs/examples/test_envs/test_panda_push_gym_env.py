# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_robot_envs.envs.panda_envs.panda_reach_gym_env import pandaReachGymEnv
from pybullet_robot_envs import robot_data
import pybullet_data


import time
import math as m

def main():

    use_IK = 0
    discreteAction = 0
    use_IK = 1 if discreteAction else use_IK

    env = pandaReachGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, useIK=use_IK, isDiscrete=discreteAction)
    motorsIds = []

    if (env._isDiscrete):
        dv = 12
        motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0))
    else :
        dv = 1
        #joints_idx = env._icub.motorIndices

        for j in range(7):
            info = env._p.getJointInfo(env._panda.pandaId,j)
            jointName = info[1]
            motorsIds.append(env._p.addUserDebugParameter(jointName.decode("utf-8"), -dv, dv, 0.0))

    done = False
    env._p.addUserDebugText('current hand position',[0,-0.5,1.4],[1.1,0,0])
    idx = env._p.addUserDebugText(' ',[0,-0.5,1.2],[1,0,0])

    for t in range(10000000):
        #env.render()
        action = []
        for motorId in range(7):
            action.append(env._p.readUserDebugParameter(motorId))

        action = int(action[0]) if discreteAction else action

        #print(env.step(action))

        state, reward, done, _ = env.step(action)
        if t%10==0:
            print("reward ", reward)
            print("done ", done)
            env._p.addUserDebugText(' '.join(str(round(e,2)) for e in state[:6]),[0,-0.5,1.2],[1,0,0],replaceItemUniqueId=idx)

if __name__ == '__main__':
    main()
