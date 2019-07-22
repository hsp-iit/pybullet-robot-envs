#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
os.sys.path.insert(0, parentdir)
print(parentdir)


from stable_baselines import HER, DQN, SAC, DDPG
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from envs.panda_envs.panda_push_gym_env_HER import pandaPushGymEnvHER
import robot_data
import tensorflow as tf
from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import numpy as np



class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[128,128,128],
                                           layer_norm=False,
                                           act_fun=tf.nn.relu,
                                           feature_extraction="lnmlp")

model_class = DDPG  # works also with SAC and DDPG

# -j
numControlledJoints = 7
# -p
fixed = False
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
policy_name = "pushing_policy"
discreteAction = 0
rend = False
env = pandaPushGymEnvHER(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0,
        isDiscrete=discreteAction, numControlledJoints = numControlledJoints,
        fixedPositionObj = fixed, includeVelObs = True)

# Available strategies (cf paper): future, final, episode, random
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
# Wrap the model

model = HER(CustomPolicy, env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
            verbose=1,tensorboard_log="../pybullet_logs/panda_push_ddpg/stable_baselines/DDPG+HER", buffer_size=1000000,batch_size=256,
            random_exploration=0.3, action_noise=action_noise)
# Train the model
model = HER.load("../policies/pushing_fixed_HER.pkl", env=env)
model.learn(timesteps)
model.save("../policies/HERPolicy")
