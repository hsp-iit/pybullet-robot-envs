#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
os.sys.path.insert(0, parentdir)
print(parentdir)
from envs.panda_envs.panda_push_gym_env import pandaPushGymEnv


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
    model = DDPG.load("panda_pushing_7DOF_4")

    pandaenv = pandaPushGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, useIK=0, numControlledJoints = 7, fixedPositionObj = False, includeVelObs = True)
    obs = pandaenv.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = pandaenv.step(action)
        #pandaenv.render()


if __name__ == '__main__':
    main()