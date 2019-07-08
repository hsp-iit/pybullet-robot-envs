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

    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`
    - joints space: torso and arm joint positions
    - You can choose which arm to control with flag `control_arm`, when instatiating `iCub Env`


-  `Observation space`:

   - hand pose
   - hand linear velocity
   - arm and torso joint positions

### Panda Env
- `Action space`: joints space

- `Action space dimensions`: 7 DOF maximum (the no. of joints can be specified when calling the training script)
    - **Note**: use the same number of joints when calling the test scripts.


-  `Observation space`:
   - joint positions
   - end-effector pose
   - end-effector velocity


## Task environments

The different task environments concerning the same robot differ mostly for:
  - the extended observations provided;
  - the reward function.

### iCub Reach Env

- `Action space`:
    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`


-  `Observation space`:
    - iCub's observation space
    - object absolute 6D pose
    - object relative 6D pose (wrt hand)

#### Reward function
The reward function is given by:
- distance between hand and the object (target) position
- plus a bonus when the hand is close to the object

Here is the code used to compute the reward function:

```python
reward = np.float(0.0)
objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
handPos = self._icub.getObservation()[0:3]
d = goal_distance(np.array(handPos), np.array(objPos))
reward = -d
if d <= self._target_dist_min:
    reward = np.float32(1000.0)
    return reward
```

### iCub Push Env

- `Action space`:
    - Cartesian space: hand 6D pose `(x,y,z),(roll,pitch,yaw)`


-  `Observation space`:
    - iCub's observation space
    - object absolute 6D pose
    - object linear velocity
    - object relative 6D pose (wrt hand)
    - object relative velocity (wrt hand)
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

### Panda Reach Env

#### Observation space
- Panda's observation space
- plus object position and orientation

#### Reward function
In this environment the rewad function is given by:
- the distance between the end-effector and the desired position
- plus a bonus when the end-effector is close to the desired position

Here is the code used to compute the reward function:

```python
reward = np.float(32.0)
objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
endEffAct = self._panda.getObservation()[0:3]
d = goal_distance(np.array(endEffAct), np.array(objPos))
reward = -d
if d <= self._target_dist_min:
    reward = np.float32(1000.0) + (100 - d*80)
    return reward
```

### Panda Push Env

#### Observation space
- Panda's observation space
- plus object position and orientation

#### Reward function
Coming soon...
