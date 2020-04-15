# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
from os import path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir =path.abspath(path.join(__file__ ,"../../../../../.."))
os.sys.path.insert(0, parentdir)
print(parentdir)
from envs.panda_envs.panda_reach_gym_env import pandaReachGymEnv


from stable_baselines.ddpg.policies import MlpPolicy
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
    # -p
    fixed = False
    # -j
    numControlledJoints = 7
    # -n
    policy_name = "reaching_policy"

    # COMMAND LINE PARAMS MANAGEMENT:
    try:
        opts, args = getopt.getopt(argv,"hj:p:n:",["j=","p=","n="])
    except getopt.GetoptError:
        print ('test.py -j <numJoints> -p <fixedPoseObject> -p <policy_name> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('------------------ Default values:')
            print('test.py  -j <numJoints: 7> -p <fixedPoseObject: False> -n <policy_name:"pushing_policy"> ')
            print('------------------')
            return 0
            sys.exit()
        elif opt in ("-j", "--j"):
            if(numControlledJoints >7):
                print("Check dimension state")
                return 0
            else:
                numControlledJoints = int(arg)
        elif opt in ("-p", "--p"):
            fixed = bool(arg)
        elif opt in ("-n","--n"):
            policy_name = str(arg)


    print(colored("-----Number Joints Controlled:","red"))
    print(colored(numControlledJoints,"red"))
    print(colored("-----Object Position Fixed:","red"))
    print(colored(fixed,"red"))
    print(colored("-----Policy Name:","red"))
    print(colored(policy_name,"red"))
    print(colored("------","red"))
    print(colored("Launch the script with -h for further info","red"))

    model = DDPG.load("../../../train/baselines/pybullet_logs/pandareach_ddpg/"+ policy_name)

    pandaenv = pandaReachGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, useIK=0, numControlledJoints = numControlledJoints, fixedPositionObj = fixed, includeVelObs = True)
    obs = pandaenv.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = pandaenv.step(action)
        #pandaenv.render()


if __name__ == '__main__':
    main(sys.argv[1:])
