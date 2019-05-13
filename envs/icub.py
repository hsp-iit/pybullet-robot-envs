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

    def __init__(self, urdfRootPath=currentdir+'/icub_fixed_model.sdf',
                    timeStep=0.01,
                    useInverseKinematics=0):

        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.useInverseKinematics = useInverseKinematics
        self.useNullSpace = 0
        self.useSimulation = 1

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
        self.icubId = p.loadSDF(self.urdfRootPath)
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

        # Get cartesian world position of left hand link frame
        l_state = p.getLinkState(self.icubId, self.indices_left_arm[-1], computeForwardKinematics=1)
        self.handPos = list(l_state[4])
        self.handOrn = list(l_state[5])
        print("PRIMA: iCub reset(): handPos")
        print(self.handPos)

        ## Set all joints initial values
        for count,i in enumerate(self.indices_torso):
            p.resetJointState(self.icubId, i, self.home_pos_torso[count]/180*m.pi)
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_pos_torso[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        for count,i in enumerate(self.indices_head):
            p.resetJointState(self.icubId, i, self.home_pos_head[count]/180*m.pi)
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_pos_head[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        for count,i in enumerate(self.indices_left_arm):
            p.resetJointState(self.icubId, i, self.home_left_arm[count]/180*m.pi)
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_left_arm[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        for count,i in enumerate(self.indices_right_arm):
            p.resetJointState(self.icubId, i, self.home_right_arm[count]/180*m.pi)
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_right_arm[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        self.ll, self.ul, self.jr, self.rs = self.getJointRanges()
        print(self.ll)

        # Get cartesian world position of left hand link frame
        l_state = p.getLinkState(self.icubId, self.indices_left_arm[-1], computeForwardKinematics=1)
        self.handPos = [0.41, 0.0, 0.8]#list(l_state[4])
        self.handOrn = [0.0,0.0,0.0]#[list(l_state[5])]
        print("DOPO: iCub reset(): handPos")
        print(self.handPos)
        j_state = p.getJointState(self.icubId, self.indices_left_arm[-1])
        self.handYaw = self.home_left_arm[-1]/180*m.pi

        self.motorNames = []
        self.motorIndices = []
        for i in self.indices_torso:
            jointInfo = p.getJointInfo(self.icubId, i)
            if jointInfo[3] > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)
        for i in self.indices_left_arm:
            jointInfo = p.getJointInfo(self.icubId, i)
            if jointInfo[3] > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

    def getJointRanges(self):
        """
        Returns
        -------
        lowerLimits : [ float ] * numDofs
        upperLimits : [ float ] * numDofs
        jointRanges : [ float ] * numDofs
        restPoses : [ float ] * numDofs
        """

        lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.icubId, i)

            if jointInfo[3] > -1:
                ll, ul = jointInfo[8:10]
                jr = ul - ll
                # For simplicity, assume resting state == initial state
                rp = p.getJointState(self.icubId, i)[0]
                lowerLimits.append(ll)
                upperLimits.append(ul)
                jointRanges.append(jr)
                restPoses.append(rp)

        return lowerLimits, upperLimits, jointRanges, restPoses

    def getActionDimension(self):
        ## TO CHECK
         if not self.useInverseKinematics:
             return len(self.motorIndices)
         return 6 #position x,y,z of hand link + roll/pitch/yaw angles of wrist joints?

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        #Cartesian wolrd pos/orn of left hand center of mass
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

            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]

            #state = p.getLinkState(self.icubId, self.indices_left_arm[-1])
            #currHandPos = state[3]
            self.handPos[0] = dx #self.handPos[0] + dx
            self.handPos[1] = dy #self.handPos[1] + dy
            self.handPos[2] = dz #self.handPos[2] + dz

            #if (self.handPos[1] > 0.2):
            #    self.handPos[1] = 0.2
            #if (self.handPos[1] < -0.2):
            #    self.handPos[1] = -0.2

            #if (self.handPos[0] > 0.6):
            #    self.handPos[0] = 0.6
            #if (self.handPos[0] < 0.2):
            #    self.handPos[0] = 0.2

            #if (self.handPos[2] > 1):
            #    self.handPos[2] = 1
            #if (self.handPos[2] < 0.5):
            #    self.handPos[2] = 0.5

            print(self.handPos[0])

            self.handOrn = p.getQuaternionFromEuler([0, 0, 0])

            if self.useNullSpace:
                jointPoses = p.calculateInverseKinematics(self.icubId, self.indices_left_arm[-1],
                                                        self.handPos,
                                                        lowerLimits=self.ll,
                                                        upperLimits=self.ul,
                                                        jointRanges=self.jr,
                                                        restPoses=self.jr)
            else:
                jointPoses = p.calculateInverseKinematics(self.icubId, self.indices_left_arm[-1],self.handPos,self.handOrn)

            if (self.useSimulation):
                for i in range(self.numJoints):
                    jointInfo = p.getJointInfo(self.icubId, i)
                    #print(jointInfo[1])
                    p.setJointMotorControl2(bodyUniqueId=self.icubId,
                                            jointIndex=i,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=jointPoses[i],
                                            targetVelocity=0,
                                            positionGain=0.25,
                                            velocityGain=0.75,
                                            force=50)
            else:
                #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.numJoints):
                    p.resetJointState(self.icubId, i, jointPoses[i])

        else:
            assert len(motorCommands)==len(self.motorIndices), 'number of motor commands differs from number of motor to control (10)'
            for action in range(len(motorCommands)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.icubId,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=motorCommands[action],
                                        targetVelocity=0,
                                        positionGain=0.25,
                                        velocityGain=0.75,
                                        force=50)
