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
In the environments related to panda you can control the robot only by acting on robot's joints.
The `action space` has dimension 7 by default, but you can specify the number of joints to be used when you call one of the training scripts.
BE CAREFUL to use the same number of joints when you call the test scripts.
The main  differences between the environments consists of the reward function.
## Task environments

### iCub Reach Env
### iCub Push Env

### Panda Reach Env
In this environment the rewad function has been modeled in the following way:

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
