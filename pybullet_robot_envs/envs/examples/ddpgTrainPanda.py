#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from envs.panda.pandaPushGymEnv import pandaPushGymEnv

from baselines.ddpg import ddpg
print(ddpg)
import datetime
import pybullet_data
import robot_data

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
  
  #icubenv = KukaGymEnv(renders=True, isDiscrete=False)
  pandaenv = pandaPushGymEnv(urdfRoot=robot_data.getDataPath(), renders=False, useIK=0, isDiscrete=discreteAction)

  act = ddpg.learn(env=pandaenv,
                    network='mlp',
                    total_timesteps=100000,
                    )

  print("Saving model to panda.pkl")
  act.save("panda.pkl")


if __name__ == '__main__':
  main()