import pybullet as p
import pybullet_data
import time
import os
import math as m

# Open GUI and set pybullet_data in the path
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane contained in pybullet_data
planeId = p.loadURDF("plane.urdf")

# Set gravity for simulation
p.setGravity(0,0,-9.8)

# Add path to icub sdf models
dir_path = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(os.path.dirname(dir_path)):
    for file in files:
        if file.endswith('.urdf'):
            print (root)
            p.setAdditionalSearchPath(root)

pandaId = p.loadURDF("panda_arm.urdf")

# set constraint between base_link and world
cid = p.createConstraint(pandaId,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],
						 [p.getBasePositionAndOrientation(pandaId)[0][0],
						  p.getBasePositionAndOrientation(pandaId)[0][1],
						  p.getBasePositionAndOrientation(pandaId)[0][2]*1.2],
						 p.getBasePositionAndOrientation(pandaId)[1])


while True:
	p.stepSimulation()
	time.sleep(0.01)
