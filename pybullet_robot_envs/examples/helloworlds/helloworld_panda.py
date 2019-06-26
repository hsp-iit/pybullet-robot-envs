import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import pybullet as p
import pybullet_data
import time
import os
import math as m
import robot_data

# Open GUI and set pybullet_data in the path
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane contained in pybullet_data
planeId = p.loadURDF("plane.urdf")

# Set gravity for simulation
p.setGravity(0,0,-9.8)

dir_path = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(os.path.dirname(dir_path)):
    for file in files:
        if file.endswith('.urdf'):
            print (root)
            p.setAdditionalSearchPath(root)

pandaId = p.loadURDF(os.path.join(robot_data.getDataPath(),"franka_description/robots/panda_arm_physics.urdf"))

# set constraint between base_link and world
cid = p.createConstraint(pandaId,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],
						 [p.getBasePositionAndOrientation(pandaId)[0][0],
						  p.getBasePositionAndOrientation(pandaId)[0][1],
						  p.getBasePositionAndOrientation(pandaId)[0][2]*1.2],
						 p.getBasePositionAndOrientation(pandaId)[1])


while True:
	p.stepSimulation()
	time.sleep(0.01)
