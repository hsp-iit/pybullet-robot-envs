# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# import RL agent
from stable_baselines import DDPG

from pybullet_robot_envs.envs.icub_envs.icub_push_gym_env import iCubPushGymEnv
from pybullet_robot_envs import robot_data

import time

log_dir = '../pybullet_logs/icubpush_ddpg'

def evaluate(env, model, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)
      obs, reward, done, info = env.step(action)

      # Stats
      episode_rewards[-1] += reward
      if done:
          print("Episode reward: ", episode_rewards[-1])
          time.sleep(1)
          obs = env.reset()
          episode_rewards.append(0.0)

  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

  return mean_100ep_reward

def main():

    # create Environment
    env = iCubPushGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, useIK=1,
                        isDiscrete=0, rnd_obj_pose=0, maxSteps=2000, reward_type=0)


    model = DDPG.load(os.path.join(log_dir,'final_model.pkl'), env=env)

    #model.learn(total_timesteps=10000)
    # Evaluate the trained agent
    mean_reward = evaluate(env, model, num_steps=6000)


if __name__ == '__main__':
  main()
