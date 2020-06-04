# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
from os import path
#print(currentdir)
parentdir =path.abspath(path.join(__file__ ,"../../../../../.."))
os.sys.path.insert(0, parentdir)
print(parentdir)

from envs.panda_envs.panda_reach_gym_env import pandaReachGymEnv
from stable_baselines import logger
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from termcolor import colored

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
    numControlledJoints = 7
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
    # -t
    timesteps = 10000000

    policy_name = "reaching_policy"

    # COMMAND LINE PARAMS MANAGEMENT:
    try:
        opts, args = getopt.getopt(argv,"hj:p:g:b:m:r:o:t:n:",["j=","p=","g=","b=","m=","r=","o=","t=","n="])
    except getopt.GetoptError:
        print ('test.py -t <timesteps> -j <numJoints> -p <fixedPoseObject> -g <gamma> -b <batchsize> -m <memory_limit> -r <norm_ret> -o <norm_obs> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('------------------ Default values:')
            print('train.py -t <timesteps: 10000000> -j <numJoints: 7> -p <fixedPoseObject: False> -n <policy_name:"reaching_policy"> -g <gamma: 0.9> -b <batch_size: 16> -m <memory_limit: 1000000> -r <norm_ret: True> -o <norm_obs: False> ')
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
        elif opt in ("-t", "--t"):
            timesteps = int(arg)
        elif opt in ("-n","--n"):
            policy_name = str(arg)



    rend = False
    obj_pose_rnd_std = 0 if fixed == True else 0.05
    pandaenv = pandaReachGymEnv(renders=rend, use_IK=0, numControlledJoints = numControlledJoints, obj_pose_rnd_std = obj_pose_rnd_std, includeVelObs = True)
    n_actions = pandaenv.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))


    pandaenv = DummyVecEnv([lambda: pandaenv])

    model = DDPG(LnMlpPolicy, pandaenv,normalize_observations = normalize_observations, gamma=gamma,batch_size=batch_size,
                    memory_limit=memory_limit, normalize_returns = normalize_returns, verbose=1, param_noise=param_noise,
                    action_noise=action_noise, tensorboard_log="../pybullet_logs/pandareach_ddpg/", reward_scale = 1)

    print(colored("-----Timesteps:","red"))
    print(colored(timesteps,"red"))
    print(colored("-----Number Joints Controlled:","red"))
    print(colored(numControlledJoints,"red"))
    print(colored("-----Object Position Fixed:","red"))
    print(colored(fixed,"red"))
    print(colored("-----Policy Name:","red"))
    print(colored(policy_name,"red"))
    print(colored("------","red"))
    print(colored("Launch the script with -h for further info","red"))

    model.learn(total_timesteps=timesteps)

    print("Saving model to panda.pkl")
    model.save("../pybullet_logs/pandareach_ddpg/"+ policy_name)

    del model # remove to demonstrate saving and loading

if __name__ == '__main__':
    main(sys.argv[1:])
