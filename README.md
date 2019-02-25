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
- `helloworld_icub.py`: a basic script loading an [iCub sdf model](https://github.com/giuliavezzani/icub-models/blob/master/iCub/robots/iCubGazeboV2_5/model.sdf). 
In order to run the script, download the model by:
```
git clone https://github.com/giuliavezzani/icub-models
```
This model has the **base fixed to the ground**, suitable for manipulation tasks.
However, **no accurate models for the iCub hand are available** so far in a format suitable for `pybullet`.


### Issues with our icub_models
- Problems in importing [sdf model with hands](https://github.com/robotology/icub-gazebo/blob/master/icub_with_hands/icub_with_hands.sdf)

