#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


from pybullet_robot_envs.envs.icub_envs.icub_push_gym_env import iCubPushGymEnv
from pybullet_robot_envs import robot_data

#from baselines import ddpg
from baselines.ddpg import ddpg
print(ddpg)
import datetime
import pybullet_data

import time
import math as m
import gym

def callback(lcl, glb):
  # stop training if reward exceeds 199
  total = sum(lcl['episode_rewards'][-101:-1]) / 100
  totalt = lcl['t']
  #print("totalt")
  #print(totalt)
  is_solved = totalt > 100 and total >= 100
  return is_solved


def main():

    use_IK = 1
    discreteAction = 0
    use_IK = 1 if discreteAction else use_IK

    #icubenv = KukaGymEnv(renders=True, isDiscrete=False)
    icubenv = iCubPushGymEnv(urdfRoot=robot_data.getDataPath(), renders=False, useIK=use_IK, isDiscrete=discreteAction)

    logger.configure(dir='../pybullet_logs/icubpush_ddpg', format_strs=['stdout','log','csv','tensorboard'])

    act = ddpg.learn(env=icubenv,
                    network='mlp',
                    total_timesteps=1000,
                    )
 ##! the trained model cannot be saved for the moment

if __name__ == '__main__':
  main()
