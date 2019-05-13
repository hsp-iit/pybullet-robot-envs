import os, inspect
from envs.iCubGymEnv import iCubGymEnv
import pybullet_data
import time
import math as m

def main():
    use_IK = 1

    # Find path to icub sdf models
    sdfPath = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for root, dirs, files in os.walk(os.path.dirname(dir_path)):
        for file in files:
            if file.endswith('icub_fixed_model.sdf'):
                sdfPath = root+'/'+str(file)

    env = iCubGymEnv(urdfRoot=sdfPath, renders=True, useInverseKinematics=use_IK)

    motorsIds = []
    if use_IK:
        dv = 1
        motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0.41))
        motorsIds.append(env._p.addUserDebugParameter("lhPosY", -dv, dv, 0.2))
        motorsIds.append(env._p.addUserDebugParameter("lhPosZ", -dv, dv, 0.8))
    else:
        ##init_pos for standing
        # without FT_sensors
        init_pos = [0.0, 0.0, 0.0, -29.4, 28.8, 0, 44.59, 0, 0, 0]
        joints_idx = list(env._icub.indices_torso) + list(env._icub.indices_left_arm)
        for count,j in enumerate(joints_idx):
            info = env._p.getJointInfo(env._icub.icubId,j)
            jointName = info[1]
            motorsIds.append(env._p.addUserDebugParameter(jointName.decode("utf-8"), info[8], info[9], init_pos[count]/180*m.pi))

    done = False

    for t in range(10000000 ):
        #env.step0()
        #env.render()
        action = []
        for motorId in motorsIds:
            action.append(env._p.readUserDebugParameter(motorId))
        state, reward, done, _ = env.step2(action)

if __name__ == '__main__':
    main()
