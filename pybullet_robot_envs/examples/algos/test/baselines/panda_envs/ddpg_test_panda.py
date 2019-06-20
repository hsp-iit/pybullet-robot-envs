#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))))
os.sys.path.insert(0, parentdir)
print(parentdir)
from envs.panda_envs.panda_reach_gym_env import pandaReachGymEnv


from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

import datetime
import pybullet_data
import robot_data
import numpy as np
import time
import math as m
import gym


def main():
    model = DDPG.load("panda_reaching_2D_fixed_extended_obs_MlpPolicy")

    pandaenv = pandaReachGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, useIK=0, numControlledJoints = 2, fixedPositionObj = True, includeVelObs = True)
    obs = pandaenv.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = pandaenv.step(action)
        #pandaenv.render()


if __name__ == '__main__':
    main()