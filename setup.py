from setuptools import setup


setup(
    name='pybullet_robot_envs',
    version='0.0.1',
    #install_requires=[x for x in requirements.txt]
    package_dir={'': 'pybullet_robot_envs'},
    packages=[x for x in find_packages('pybullet_robot_envs')],
)
