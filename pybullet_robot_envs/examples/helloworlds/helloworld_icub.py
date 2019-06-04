import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import pybullet as p
import pybullet_data
import robot_data
import time
import math as m


def main():
    # Open GUI and set pybullet_data in the path
    p.connect(p.GUI)
    #p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load plane contained in pybullet_data
    p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"))

    # Set gravity for simulation
    p.setGravity(0,0,-9.8)

    # load robot model
    robotIds = p.loadSDF(os.path.join(robot_data.getDataPath(), "iCub/icub_fixed_model.sdf"))
    icubId = robotIds[0]

    # set constraint between base_link and world
    p.createConstraint(icubId,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],
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

    # Load other objects
    p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), [1.1, 0.0, 0.0])
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "lego/lego.urdf"), [0.5,0.0,0.8])

    # add debug slider
    jointIds=[]
    paramIds=[]
    joints_num = p.getNumJoints(icubId)

    print("len init_pos ",len(init_pos))
    print("len num joints", joints_num)

    for j in range (joints_num):
    	info = p.getJointInfo(icubId,j)
    	jointName = info[1]
    	#jointType = info[2]
    	jointIds.append(j)
    	paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), info[8], info[9], init_pos[j]/180*m.pi))


    while True:
    	for i in range(joints_num):
    		p.setJointMotorControl2(icubId, i, p.POSITION_CONTROL,
    								targetPosition=p.readUserDebugParameter(i),
    								targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

    	p.stepSimulation()
    	time.sleep(0.01)

if __name__ == '__main__':
    main()
