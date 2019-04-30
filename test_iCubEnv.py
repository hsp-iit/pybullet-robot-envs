
from envs.iCubGymEnv import iCubGymEnv
import time

def main():

    env = iCubGymEnv(renders=True)

    dv = 0.5
    motorsIds = []
    motorsIds.append(env._p.addUserDebugParameter("posX", -dv, dv, 0))
    motorsIds.append(env._p.addUserDebugParameter("posY", -dv, dv, 0))
    motorsIds.append(env._p.addUserDebugParameter("posZ", -dv, dv, 0))
    motorsIds.append(env._p.addUserDebugParameter("yaw", -dv, dv, 0))

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
