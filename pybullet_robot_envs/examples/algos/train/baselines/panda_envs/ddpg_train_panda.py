#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))))
os.sys.path.insert(0, parentdir)
print(parentdir)


from envs.panda_envs.panda_reach_gym_env import pandaReachGymEnv
from stable_baselines import logger
from stable_baselines.ddpg.policies import LnMlpPolicy
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

    
    discreteAction = 0 
    rend = False
    pandaenv = pandaReachGymEnv(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0, isDiscrete=discreteAction, numControlledJoints = 6, fixedPositionObj = False, includeVelObs = True)
    n_actions = pandaenv.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    
    pandaenv = DummyVecEnv([lambda: pandaenv])

    model = DDPG(LnMlpPolicy, pandaenv,normalize_observations = False, gamma=0.995,batch_size=16,memory_limit=1000000, normalize_returns = True, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log="./panda_reaching_ddpg/", reward_scale = 1)
    model.learn(total_timesteps=10000000)

    #logger.configure(folder='../pybullet_logs/panda_reaching_ddpg', format_strs=['stdout','log','csv','tensorboard'])

    print("Saving model to panda.pkl")
    model.save("../../../test/baselines/panda_envs/panda_reaching_6D_")
    del model # remove to demonstrate saving and loading

if __name__ == '__main__':
    main()

