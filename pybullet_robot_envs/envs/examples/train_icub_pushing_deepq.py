#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from envs.iCub.iCubPushGymEnv import iCubPushGymEnv

from baselines import deepq

import datetime
import pybullet_data
import robot_data

import time
import math as m

def callback(lcl, glb):
  is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
  return is_solved

def main():

  use_IK = 1
  discreteAction = 1
  use_IK = 1 if discreteAction else use_IK

  icubenv = iCubPushGymEnv(urdfRoot=robot_data.getDataPath(), renders=False, useIK=use_IK, isDiscrete=discreteAction)

  act = deepq.learn(env=icubenv,
                    network='mlp',
                    lr=1e-3,
                    total_timesteps=100000,
                    buffer_size=50000,
                    exploration_fraction=0.1,
                    exploration_final_eps=0.02,
                    print_freq=10,
                    callback=callback
                    )

  print("Saving model to icub_pushing_model_deepg.pkl")
  act.save("icub_pushing_model_deepq.pkl")


if __name__ == '__main__':
  main()
