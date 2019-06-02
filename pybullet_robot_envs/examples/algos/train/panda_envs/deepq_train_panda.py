#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))))
os.sys.path.insert(0, parentdir)
print(parentdir)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

#import pybullet_robot_envs
import pybullet_data

from envs.panda_envs.panda_push_gym_env import pandaPushGymEnv
import robot_data

import datetime
import time
import math as m

import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule


def callback(lcl, glb):
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    if is_solved:
        print("is solved!")
    return is_solved

def main():

  use_IK = 0
  discreteAction = 1
  #use_IK = 1 if discreteAction else use_IK

  pandaenv = env = pandaPushGymEnv(urdfRoot=robot_data.getDataPath(), renders=False, useIK=use_IK, isDiscrete=discreteAction)

  #logger.configure(dir='../pybullet_logs/icubpush_deepq', format_strs=['stdout','log','csv','tensorboard'])


  act = deepq.learn(env=pandaenv,
                    network='mlp',
                    lr=1e-3,
                    total_timesteps=100000,
                    buffer_size=50000,
                    exploration_fraction=0.1,
                    exploration_final_eps=0.02,
                    print_freq=10,
                    callback=callback
                    )



  print("Saving model to panda_pushing_model_deepg.pkl")
  act.save("../pybullet_logs/panda_push_deepq_tot/model.pkl")


if __name__ == '__main__':
  main()
