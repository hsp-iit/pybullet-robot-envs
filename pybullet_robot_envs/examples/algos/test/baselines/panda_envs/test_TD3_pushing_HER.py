# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
os.sys.path.insert(0, parentdir)
print(parentdir)


from stable_baselines import HER, DQN, SAC, DDPG, TD3

from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from envs.panda_envs.panda_push_gym_env_HER import pandaPushGymEnvHER
import robot_data

model_class = DDPG  # works also with SAC and DDPG

# -j

action_space = 7
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
timesteps = 1000000
policy_name = "pushing_policy"
discreteAction = 0
rend = True

env = pandaPushGymEnvHER(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0,
        isDiscrete=discreteAction, action_space = action_space,
        fixedPositionObj = fixed, includeVelObs = True, object_position=1)

goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# Wrap the model
model = HER.load("../policies/PUSHING_TD3+HER_FIXED_POSITION_PHASE_1best_model.pkl", env=env)

obs = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
