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
### Panda Env
- `Action space`: joints space
- `Action space dimensions`: 7 DOF maximum (the no. of joints can be specified when calling the training script) 
    - **Note**: use the same number of joints when calling the test scripts.
-  `Observation space`: 
   - joint positions
   - end-effector pose
   - end-effector velocity

## Task environments

Thw different task environments concerning the same robot differ mostly for:
  - the extended observations provided;
  - the reward function.
  
### iCub Reach Env
### iCub Push Env

### Panda Reach Env
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
