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

class pandaEnv:

    def __init__(self, urdfRootPath=robot_data.getDataPath(), timeStep=0.01, useInverseKinematics=0, basePosition=[-0.6,-0.4,0.625], useFixedBase= True, actionSpace = 7, includeVelObs = True):

        self.urdfRootPath = os.path.join(urdfRootPath, "franka_description/robots/panda_arm_physics.urdf")
        self.timeStep = timeStep
        self.useInverseKinematics = useInverseKinematics
        self.useNullSpace = 0
        self.useOrientation = 1
        self.useSimulation = 1
        self.basePosition = basePosition
        self.useFixedBase = useFixedBase
        self.workspace_lim = [[0.2,0.60],[-0.60,0.60],[0,0.40]]
        self.endEffLink = 8
        self.actionSpace = actionSpace
        self.includeVelObs = includeVelObs
        self.reset()

    def reset(self):

        self.pandaId = p.loadURDF(self.urdfRootPath, basePosition = self.basePosition, useFixedBase = self.useFixedBase)

        for i in range(self.actionSpace):
            p.resetJointState(self.pandaId, i, 0)
            p.setJointMotorControl2(self.pandaId, i, p.POSITION_CONTROL,targetPosition=0,targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

    def getJointsRanges(self):
        #to-be-defined
        return 0


    def getActionDimension(self):
        if self.useInverseKinematics == 1:
            return 6
        else:
            return self.actionSpace #p.getactionSpace(self.pandaId)


    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.pandaId, self.endEffLink, computeLinkVelocity = 1)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(list(pos))
        observation.extend(list(euler)) #roll, pitch, yaw
        if (self.includeVelObs) :
            velL = state[6]
            velA = state[7]
            observation.extend(list(velL))
            observation.extend(list(velA))

        jointStates = p.getJointStates(self.pandaId,range(11))
        jointPoses = [x[0] for x in jointStates]
        observation.extend(list(jointPoses))

        return observation

    def applyAction(self, action):

        if(self.useInverseKinematics):
            print("Inverse Kinematic")
            #TO BE DEFINED
        else:

            assert len(action)==self.actionSpace, ('number of motor commands differs from number of motor to control',len(action))

            for a in range(len(action)):

                curr_motor_pos = p.getJointState(self.pandaId, a)[0]
                new_motor_pos = curr_motor_pos + action[a] #supposed to be a delta

                p.setJointMotorControl2(self.pandaId,
                                        a,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        targetVelocity=0,
                                        positionGain=0.25,
                                        velocityGain=0.75,
                                        force=100)
