import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import math as m

class iCub:

    def __init__(self, urdfRootPath=currentdir+'/icub_fixed_model.sdf', timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.useInverseKinematics = 1

        self.indices_torso = range(12,15)
        self.indices_left_arm = range(15,22)
        self.indices_right_arm = range(25,32)
        self.indices_head = range(22,25)

        self.home_pos_torso = [0.0, 0.0, 0.0] #degrees
        self.home_pos_head = [0.47, 0, 0]

        self.home_left_arm = [-29.4, 28.8, 0, 44.59, 0, 0, 0]
        self.home_right_arm = [-29.4, 30.4, 0, 44.59, 0, 0, 0]

        self.reset()

    def reset(self):
        self.icubId = p.loadSDF(currentdir+"/icub_fixed_model.sdf")
        self.icubId = self.icubId[0]
        self.numJoints = p.getNumJoints(self.icubId)

        ## for j in range (self.numJoints):
        ##    print(p.getJointInfo(self.icubId,j))

        # set constraint between base_link and world
        self.constr_id = p.createConstraint(self.icubId,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],
                                 [p.getBasePositionAndOrientation(self.icubId)[0][0],
                                  p.getBasePositionAndOrientation(self.icubId)[0][1],
                                  p.getBasePositionAndOrientation(self.icubId)[0][2]*1.2],
                                  p.getBasePositionAndOrientation(self.icubId)[1])

        ## Set all joints initial values
        for count,i in enumerate(self.indices_torso):
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_pos_torso[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        for count,i in enumerate(self.indices_head):
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_pos_head[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        for count,i in enumerate(self.indices_left_arm):
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_left_arm[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        for count,i in enumerate(self.indices_right_arm):
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_right_arm[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        l_state = p.getLinkState(self.icubId, self.indices_left_arm[-1],computeForwardKinematics=1)
        self.handPos = list(l_state[4])
        self.handOrn = list(l_state[5])
        print("iCub reset(): handPos")
        print(self.handPos)
        j_state = p.getJointState(self.icubId, self.indices_left_arm[-1])
        self.handYaw = j_state[0]
        print("iCub reset(): handYaw")
        print(self.handYaw)
        self.motorNames = []
        self.motorIndices = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.icubId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

    def getActionDimension(self):
        ## TO CHECK
         if not self.useInverseKinematics:
             return len(self.motorIndices)
         return 6 #position x,y,z (of hand link?) + roll/pitch/yaw angles of wrist joints?

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.icubId, self.indices_left_arm[-1])
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def applyAction(self, motorCommands):
        if(self.useInverseKinematics):
            #print("motorCommands[0]=", motorCommands[0])

            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            da = motorCommands[3]

            #state = p.getLinkState(self.icubId, self.indices_left_arm[-1])
            #currHandPos = state[0]
            self.handPos[0] = self.handPos[0] - dx
            self.handPos[1] = self.handPos[1] - dy
            self.handPos[2] = self.handPos[2] + dz

            self.handYaw += da
            jointPoses = p.calculateInverseKinematics(self.icubId, self.indices_left_arm[-1], self.handPos,self.handOrn)

            for i in range(self.numJoints):
                #print(i)
                jointInfo = p.getJointInfo(self.icubId, i)
                #print(jointInfo)
                p.setJointMotorControl2(bodyUniqueId=self.icubId,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[i],
                                        targetVelocity=0,
                                        positionGain=0.25,
                                        velocityGain=0.75,
                                        force=50)
