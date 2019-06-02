import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import robot_data
import math as m
import time


class pandaEnv:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Load plane contained in pybullet_data
        planeId = p.loadURDF("plane.urdf", useFixedBase=True)
        # Set gravity for simulation
        p.setGravity(0, 0, -9.8)
        self.numJoints = 0
        self.initPos = []

    def loadRobot(self, pathURDF):
        path = os.path.join(pathURDF, "franka_description/robots/panda_arm_no_params.urdf")
        self.pandaId = p.loadURDF(path, basePosition = [-0.6,-0.4, 0.625], useFixedBase=True)
        self.numJoints = p.getNumJoints(self.pandaId)
        #pathArm = os.path.join(pathURDF, "franka_description/robots/hand.urdf")
        #self.handId = p.loadURDF(pathArm)
        self.init_pos = [0 for i in range(self.numJoints)]
        tableId = p.loadURDF(os.path.join(pathURDF, "franka_description/table.urdf"), useFixedBase=True)
        cubeId = p.loadURDF( os.path.join(pathURDF, "franka_description/cube_small.urdf"),[0,0,0.8] )

    def loadSliders(self):
        jointIds = []
        paramIds = []
        joints_num = p.getNumJoints(self.pandaId)
        for j in range(joints_num):
            info = p.getJointInfo(self.pandaId, j)
            jointName = info[1]
            jointType = info[2]
            jointIds.append(j)
            paramIds.append(p.addUserDebugParameter(jointName.decode(
                "utf-8"), info[8], info[9], self.init_pos[j]/180*m.pi))

    def runSimulation(self):
        while True:
            for i in range(self.numJoints):
                p.setJointMotorControl2(self.pandaId, i, p.POSITION_CONTROL, targetPosition=0, targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)
            
            #print(p.getLinkState(self.pandaId, 8))
            #print(str(p.getLinkState(self.pandaId, 10)[0][2] - p.getLinkState(self.pandaId, 9)[0][2]) + '\n')
            p.stepSimulation()
            time.sleep(0.01)


def main():
    panda = pandaEnv()
    urdfRootPath=robot_data.getDataPath()
    panda.loadRobot(urdfRootPath)
    #panda.loadSliders()
    panda.runSimulation()


if __name__ == "__main__":
    main()
