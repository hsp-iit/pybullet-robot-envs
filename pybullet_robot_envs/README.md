<p align="center">
<h1 align="center">Environments</h1>
</p>

## Overview
 - [Robot environments](#robot-environments)
    - [iCub Env](#icub-env)
    - [Panda Env](#panda-env)
 - [Task environments](#task-environments)
    - [iCub Reach Env](#icub-reach-env)
    - [iCub Push Env](#icub-push-env)
    - [Panda Reach Env](#panda-reach-env)
    - [Panda Push Env](#panda-push-env)
---

## Robot environments
### iCub Env

- `Action space`:

    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)` (if `use_IK=1`)
    - Joints space: torso and arm joint positions  (if `use_IK=0`)
    - You can choose which arm to control with flag `control_arm`, when instatiating `iCub Env`


-  `Observation space`:

   - hand 6D pose
   - hand linear velocity
   - arm and torso joint positions

### Panda Env
- `Action space`: 
    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`  (if `use_IK=1`)
    - Joints space  (if `use_IK=0`)


-  `Observation space`:
   - gripper 6D pose
   - gripper linear velocity
   - joint positions


## Task environments

The different task environments concerning the same robot differ mostly for:
  - the extended observations provided;
  - the reward function.

### iCub Reach Env

- `Action space`:
    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`  (if `use_IK=1`)
    - Joints space: torso and arm joint positions  (if `use_IK=0`)

-  `Observation space`:
    - iCub's observation space
    - object absolute 6D pose
    - object relative 6D pose wrt hand

#### Reward function
The reward function is given by:
- distance between hand and the object (target) position
- plus a bonus when the hand is close to the object

Here is the code used to compute the reward function:

```python
robot_obs, _ = self._robot.get_observation()
world_obs, _ = self._world.get_observation()

d = goal_distance(np.array(robot_obs[:3]), np.array(world_obs[:3]))

reward = -d
if d <= self._target_dist_min:
    reward += np.float32(1000.0) + (100 - d*80)
```

### iCub Push Env

- `Action space`:
    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`  (if `use_IK=1`)
    - Joints space: torso and arm joint positions  (if `use_IK=0`)

-  `Observation space`:
    - iCub's observation space
    - object absolute 6D pose
    - object linear velocity
    - object relative 6D pose wrt hand
    - target absolute pose

#### Reward function
The reward function is given by:
- distance `d1` between hand and object (target) position
- distance `d2` between object position and target position
- plus a bonus when the hand is close to the object

The distances are encapsulated into the reward functions in two ways.

**Note:** current implementation of pushing task works only when initial positions of object and target are fixed. The implementation may changed. 

##### #0: negative reward
Here is the code used to compute the reward function **#0**:

```python
handPos = self._icub.getObservation()[0:3]
objPos, _ = p.getBasePositionAndOrientation(self._objID)
d1 = goal_distance(np.array(handPos), np.array(objPos))
d2 = goal_distance(np.array(objPos), np.array(self._tg_pose))

reward = -d1 -d2
if d2 <= self._target_dist_min:
    reward += np.float32(1000.0)
```

##### #1: positive reward
Here is the code used to compute the reward function **#1**:

```python
handPos = self._icub.getObservation()[0:3]
objPos, _ = p.getBasePositionAndOrientation(self._objID)
d1 = goal_distance(np.array(handPos), np.array(objPos))
d2 = goal_distance(np.array(objPos), np.array(self._tg_pose))

rew1 = 0.125
rew2 = 0.25

  if d1 > 0.1:
      reward = rew1*(1 - d1/self._init_dist_hand_obj)
  else:
      reward = rew1*(1 - d1/self._init_dist_hand_obj) + rew2*(1 - d2/self._max_dist_obj_tg)

  if d2 <= self._target_dist_min:
      reward += np.float32(1000.0)
```

### iCub Push Goal Env
Hindsight Experience Replay (HER) implementation of the iCub Push Env.

- `Action space`:
    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`  (if `use_IK=1`)
    - Joints space: torso and arm joint positions  (if `use_IK=0`)


-  `Observation space`:
    - iCub's observation space
    - object 6D pose
    - object linear velocity
    - object relative 6D pose wrt hand
    - target 3D pose

#### Goals

-  `Achieved goal`:
    - object 3D pose
    
-  `Desired goal`:
    - target 3D pose


### Panda Reach Env

- `Action space`: 
    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`  (if `use_IK=1`)
    - Joints space  (if `use_IK=0`)
    
-  `Observation space`:
    - Panda's observation space
    - object absolute 6D pose
    - object relative 6D pose wrt hand

#### Reward function
In this environment the reward function is given by:
- the distance between the end-effector and the desired position
- plus a bonus when the end-effector is close to the desired position

Here is the code used to compute the reward function:

```python
robot_obs, _ = self._robot.get_observation()
world_obs, _ = self._world.get_observation()

d = goal_distance(np.array(robot_obs[:3]), np.array(world_obs[:3]))

reward = -d
if d <= self._target_dist_min:
    reward = np.float32(1000.0) + (100 - d*80)
```

### Panda Push Env

- `Action space`: 
    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`  (if `use_IK=1`)
    - Joints space  (if `use_IK=0`)
    
-  `Observation space`:
    - Panda's observation space
    - object absolute 6D pose
    - object relative 6D pose wrt hand
    - target 3D pose

#### Reward function
The reward function is given by:
- distance `d1` between hand and object (target) position
- distance `d2` between object position and target position
- plus a bonus when the hand is close to the object

Here is the code used to compute the reward function:

```python
robot_obs, _ = self._robot.get_observation()
world_obs, _ = self._world.get_observation()

d1 = goal_distance(np.array(robot_obs[:3]), np.array(world_obs[:3]))
d2 = goal_distance(np.array(world_obs[:3]), np.array(self._target_pose))

reward = -d1 - d2
if d2 <= self._target_dist_min:
    reward = np.float32(1000.0) + (100 - d2*80)
```

### Panda Push Goal Env
Hindsight Experience Replay (HER) implementation of the Panda Push Env.

- `Action space`: 
    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`  (if `use_IK=1`)
    - Joints space  (if `use_IK=0`)
    
-  `Observation space`:
    - Panda's observation space
    - object absolute 6D pose
    - object relative 6D pose wrt hand
    - target 3D pose

#### Goals

-  `Achieved goal`:
    - object 3D pose
    
-  `Desired goal`:
    - target 3D pose