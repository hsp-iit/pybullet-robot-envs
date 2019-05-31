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
print(find_packages('pybullet_robot_envs'))
print("-----")

# Environment-specific dependencies.
extras = {
    'pandas':['pandas'],
    'matplotlib':['matplotlib']
}

setup(
    name='pybullet_robot_envs',
    version='0.0.1',
    description="PyBullet Robot Envs: A toolkit for developing OpenAI Gym environments in PyBullet simulator",
    install_requires=['pybullet','gym','tensorflow','ruamel.yaml'],
    extras_require=extras,
    package_dir={'': '.'},
    packages=find_packages(),
    package_data={'pybullet_robot_envs': need_files},
)
