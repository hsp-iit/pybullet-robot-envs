## pybullet-workbook

### PyBullet installation
1. Create a virtual environment:
```
virtualenv pybullet
```
2. Activate the virtual environment:
```
source pybullet/bin/activate
```
3. Install all the dependencies necessary for doing RL training with py bullet:
```
pip install -r requirements.txt
```
4. Test installation running:
```
python helloworld.py
```
5. Test RL framework running:
  ```
  python -m pybullet_envs.examples.enjoy_TF_HumanoidBulletEnv_v0_2017may
  ```
or
```
python -m pybullet_envs.examples.kukaGymEnvTest
```
6. Get OpenAI baselines:
```
git clone https://github.com/openai/baselines.git
```
7. Test everything working with:
```
cd baselines/
python -m pybullet_envs.agents.train_ppo --config=pybullet_pendulum --logdir=pendulum
```

### Pybullet summary

Pybullet tutorial is available [here](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#).
Main features:

- It is possible to load **urdf/sdf** models;
- Interactive OpenGl gui;
- It is possible to control robots (**inverse dynamics and kinematics**);
- Options in camera rendering;
- RL framework:
    - **a suit for gym environments is installed together with PyBullet**;
    - It includes pybullet version of openai gym envs;
    - To train we can use
       - [openai baselines](https://github.com/openai/baselines);
       - [agents](https://pypi.org/project/agents/): a python package for tensorflow implementation of RL algorithms;
- Also VR is available.

### Examples
- `helloworld.py`: a basic script for loading one of pybullet models
- `helloworld_icub.py`: a basic script loading an [iCub sdf model](https://github.com/giuliavezzani/pybullet-workbook/blob/master/envs/icub_fixed_model.sdf). 

This model has the **base fixed to the ground**, suitable for manipulation tasks.
However, **no accurate models for the iCub hand are available** so far in a format suitable for `pybullet`.



### Envs
Gym-like environments  for the iCub and the Franka Panda are under development in [/envs](https://github.com/giuliavezzani/pybullet-workbook/tree/master/envs).

The iCub model has been obtained from the urdf including hands manually fixed the joint reference frames.

Tha Panda model has been obtained by converting the .urdf.xarco model provided by [franka_ros](https://github.com/frankaemika/franka_ros/blob/kinetic-devel/franka_description/robots/panda_arm.urdf.xacro) using [Ros](https://answers.ros.org/question/10401/how-to-convert-xacro-file-to-urdf-file/).

The structureof the environments takes inspiration from [pybullet kuka env](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kukaGymEnv.py).
