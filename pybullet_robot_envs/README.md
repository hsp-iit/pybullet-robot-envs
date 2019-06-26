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
The action space has dimension 7 by default, but you can specify the number of joints to be used when you call one of the training scripts.
BE CAREFUL to use the same number of joints when you cal the test scripts.
The main  differences between the environments consists of the reward function.
## Task environments

### iCub Reach Env
### iCub Push Env

### Panda Reach Env
In this environment the rewad function has been modeled in
### Panda Push Env
