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
        if file.endswith('.sdf'):
            print (root+'/'+str(file))
            p.setAdditionalSearchPath(root)

robotIds = p.loadSDF("icub_fixed_model.sdf")
icubId = robotIds[0]

# set constraint between base_link and world
cid = p.createConstraint(icubId,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],
						 [p.getBasePositionAndOrientation(icubId)[0][0],
						  p.getBasePositionAndOrientation(icubId)[0][1],
						  p.getBasePositionAndOrientation(icubId)[0][2]*1.2],
						 p.getBasePositionAndOrientation(icubId)[1])

##init_pos for standing
# without FT_sensors
init_pos = [0]*15 + [-29.4, 28.8, 0, 44.59, 0, 0, 0, 0.47, 0, 0, -29.4, 30.4, 0, 44.59, 0, 0, 0]

# with FT_sensors
#init_pos = [0]*19 + [-29.4, 28.8, 0, 0, 44.59, 0, 0, 0, 0.47, 0, 0, -29.4, 30.4, 0, 0, 44.59, 0, 0, 0]

# all set to zero
#init_pos = [0]*p.getNumJoints(icubId)

# add debug slider
jointIds=[]
paramIds=[]
joints_num = p.getNumJoints(icubId)
for j in range (joints_num):
	info = p.getJointInfo(icubId,j)
	jointName = info[1]
	jointType = info[2]
	jointIds.append(j)
	paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), info[8], info[9], init_pos[j]/180*m.pi))


while True:
	for i in range(joints_num):
		p.setJointMotorControl2(icubId, i, p.POSITION_CONTROL,
								targetPosition=p.readUserDebugParameter(i),
								targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

	p.stepSimulation()
	time.sleep(0.01)
