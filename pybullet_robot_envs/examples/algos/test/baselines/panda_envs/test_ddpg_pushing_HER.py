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

model_class = DDPG  # works also with SAC and DDPG

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
policy_name = "pushing_policy"
discreteAction = 0
rend = True

env = pandaPushGymEnvHER(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0,
        isDiscrete=discreteAction, numControlledJoints = numControlledJoints,
        fixedPositionObj = fixed, includeVelObs = True)

goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# Wrap the model
model = HER.load("../policies/pushing_fixed_HER.pkl", env=env)

obs = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
