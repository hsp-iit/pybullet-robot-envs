import pybullet as p
import pybullet_data
import time
import math as m

# X-red Y-green Z-blue

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#gravId = p.addUserDebugParameter("gravity",-10,10,-9.8)
p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")

#modeledit.sdf modelV2_5
#obUids = p.loadSDF("modeledit.sdf") #, useMaximalCoordinates=True
obUids = p.loadSDF("modeledit.sdf")
humanoid = obUids[0]
print(p.getBodyInfo(humanoid))    # get (base_link, iCub)


# set constraint between base_link and world
cid = p.createConstraint(humanoid,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],
						 [p.getBasePositionAndOrientation(humanoid)[0][0],
						  p.getBasePositionAndOrientation(humanoid)[0][1],
						  p.getBasePositionAndOrientation(humanoid)[0][2]*1.2],
						 p.getBasePositionAndOrientation(humanoid)[1])

##initpos for standing

# without FT_sensors
initpos = [0]*15 + [-29.4, 28.8, 0, 44.59, 0, 0, 0, 0.47, 0, 0, -29.4, 30.4, 0, 44.59, 0, 0, 0]

# with FT_sensors
#initpos = [0]*19 + [-29.4, 28.8, 0, 0, 44.59, 0, 0, 0, 0.47, 0, 0, -29.4, 30.4, 0, 0, 44.59, 0, 0, 0]

# all set to zero
#initpos = [0]*p.getNumJoints(humanoid)

# add debug slider
jointIds=[]
paramIds=[]
totoaljointsnum = p.getNumJoints(humanoid)
for j in range (totoaljointsnum):
	info = p.getJointInfo(humanoid,j)
	jointName = info[1]
	jointType = info[2]
	jointIds.append(j)
	paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), info[8], info[9], initpos[j]/180*m.pi))


while 1:
	#p.setGravity(0,0,p.readUserDebugParameter(gravId))
	for i in range(totoaljointsnum):
		p.setJointMotorControl2(humanoid, i, p.POSITION_CONTROL,
								targetPosition=p.readUserDebugParameter(i),
								targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

	p.stepSimulation()
	time.sleep(0.01)