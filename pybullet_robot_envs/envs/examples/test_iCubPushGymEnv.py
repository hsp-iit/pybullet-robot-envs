#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from envs.iCub.iCubPushGymEnv import iCubPushGymEnv
import pybullet_data
import robot_data

import time
import math as m

def main():

    use_IK = 1
    discreteAction = 0
    use_IK = 1 if discreteAction else use_IK

    env = iCubPushGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, useIK=use_IK, isDiscrete=discreteAction)
    motorsIds = []

    if (env._isDiscrete):
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
        joints_idx = env._icub.motorIndices

        for count,j in enumerate(joints_idx):
            info = env._p.getJointInfo(env._icub.icubId,j)
            jointName = info[1]
            motorsIds.append(env._p.addUserDebugParameter(jointName.decode("utf-8"), -dv, dv, 0.0))

    done = False
    env._p.addUserDebugText('current hand position',[0,-0.5,1.4],[1.1,0,0])
    idx = env._p.addUserDebugText(' ',[0,-0.5,1.2],[1,0,0])

    for t in range(10000000 ):
        #env.step0()
        #env.render()
        action = []
        for motorId in motorsIds:
            action.append(env._p.readUserDebugParameter(motorId))

        action = int(action[0]) if discreteAction else action

        state, reward, done, _ = env.step(action)
        if t%10==0:
            print("reward ", reward)
            print("done ", done)
            env._p.addUserDebugText(' '.join(str(round(e,2)) for e in state[:6]),[0,-0.5,1.2],[1,0,0],replaceItemUniqueId=idx)

if __name__ == '__main__':
    main()
