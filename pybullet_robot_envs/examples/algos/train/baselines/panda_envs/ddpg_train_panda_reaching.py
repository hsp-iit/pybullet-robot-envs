#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
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
import sys, getopt

def main(argv):

    # -j
    numControlledJoints = 6
    # -p
    fixed = True
    # -o
    normalize_observations = False
    # -g
    gamma = 0.9
    # -b
    batch_size = 16
    # -m
    memory_limit = 1000000
    # -r
    normalize_returns = True


    try:
        opts, args = getopt.getopt(argv,"hj:p:g:b:m:r:o:",["j=","p=","g=","b=","m=","r=","o="])
    except getopt.GetoptError:
        print ('test.py -j <numJoints> -p <fixedPoseObject> -g <gamma> -b <batchsize> -m <memory_limit> -r <norm_ret> -o <norm_obs> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('------------------')
            print('test.py -j <numJoints> -p <fixedPoseObject> -g <gamma> -b <batchsize> -m <memory_limit> -r <norm_ret> -o <norm_obs> ')
            print('------------------')
            return 0
            sys.exit()
        elif opt in ("-j", "--j"):
            if(numControlledJoints >7):
                print("check dim state")
                return 0
            else:
                numControlledJoints = int(arg)
        elif opt in ("-p", "--p"):
            fixed = bool(arg)
        elif opt in ("-g", "--g"):
            gamma = float(arg)
        elif opt in ("-o", "--o"):
            normalize_observations = bool(arg)
        elif opt in ("-b", "--b"):
            batch_size = int(arg)
        elif opt in ("-m", "--m"):
            memory_limit = int(arg)
        elif opt in ("-r", "--r"):
            normalize_returns = bool(arg)
        
        
        
            

    
    discreteAction = 0 
    rend = False
    pandaenv = pandaReachGymEnv(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0, isDiscrete=discreteAction, numControlledJoints = numControlledJoints, fixedPositionObj = fixed, includeVelObs = True)
    n_actions = pandaenv.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    
    pandaenv = DummyVecEnv([lambda: pandaenv])

    model = DDPG(LnMlpPolicy, pandaenv,normalize_observations = normalize_observations, gamma=gamma,batch_size=batch_size,memory_limit=memory_limit, normalize_returns = normalize_returns, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log="./panda_reaching_ddpg/", reward_scale = 1)
    model.learn(total_timesteps=10000000)

    #logger.configure(folder='../pybullet_logs/panda_reaching_ddpg', format_strs=['stdout','log','csv','tensorboard'])

    print("Saving model to panda.pkl")
    model.save("../../../test/baselines/panda_envs/panda_reaching_6D_")
    del model # remove to demonstrate saving and loading

if __name__ == '__main__':
    main(sys.argv[1:])

