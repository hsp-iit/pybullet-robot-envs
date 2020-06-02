# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os
from setuptools import setup, find_packages

setup_py_dir = os.path.dirname(os.path.realpath(__file__))
need_files = []
datadir = "pybullet_robot_envs"

hh = setup_py_dir + "/" + datadir

for root, dirs, files in os.walk(hh):
  for fn in files:
    ext = os.path.splitext(fn)[1][1:]
    if ext and ext in 'urdf sdf xml yaml stl ini dae'.split(
    ):
      fn = root + "/" + fn
      need_files.append(fn[1 + len(hh):])

print("packages")
print(find_packages())
print("-----")

setup(
    name='pybullet_robot_envs',
    version='0.0.1',
    author="Elena Rampone, Andrea Romdhana, Giulia Vezzani",
    author_email="elena.rampone@iit.it, andrea.romdhana@iit.it, giulia.vezzani@iit.it",
    description="PyBullet Robot Envs: A collection of OpenAI Gym RL robotic environments in PyBullet simulator",
    license="LGPL",
    url='https://github.com/robotology-playground/pybullet-robot-envs',
    python_requires='>=3.5',
    #install_requires=['gym==0.12.5','pybullet==2.5.0'],
    package_dir={'': '.'},
    packages=find_packages(),
    package_data={'pybullet_robot_envs': need_files},
)
