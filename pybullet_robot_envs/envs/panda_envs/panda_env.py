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

    def __init__(self, urdfRootPath=robot_data.getDataPath(), timeStep=0.01, useInverseKinematics=0, basePosition=[-0.6,-0.4,0.625], useFixedBase= True):

        self.urdfRootPath = os.path.join(urdfRootPath, "franka_description/robots/panda_arm_no_params.urdf")
        self.timeStep = timeStep
        self.useInverseKinematics = useInverseKinematics
        self.useNullSpace = 0
        self.useOrientation = 1
        self.useSimulation = 1
        self.basePosition = basePosition
        self.useFixedBase = useFixedBase
        self.workspace_lim = [[0.2,0.70],[-0.70,0.70],[0,0.40]]
        #number of the end effector
        self.endEffLink = 8 
        self.reset()

    def reset(self):

        self.pandaId = p.loadURDF(self.urdfRootPath, basePosition = self.basePosition, useFixedBase = self.useFixedBase)
        
        #self.numJoints = p.getNumJoints(self.pandaId)
        self.numJoints = 8 
        ## Set all joints initial values
        for i in range(self.numJoints):
            p.resetJointState(self.pandaId, i, 0)
            p.setJointMotorControl2(self.pandaId, i, p.POSITION_CONTROL,targetPosition=0,targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

    def getJointsRanges(self):
        #to-be-defined
        return 0 
    

    def getActionDimension(self):
        if self.useInverseKinematics == 1:
            return 6
        else:
            return p.getNumJoints(self.pandaId)


    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        #Cartesian world pos/orn of endEff
        observation = []
        #REVIEW TO DO 
        state = p.getLinkState(self.pandaId, self.endEffLink)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler)) #roll, pitch, yaw

        return observation

    def applyAction(self, action):

        if(self.useInverseKinematics):
            print("Inverse Kinematic")
            #TO BE DEFINED            
        else:
            assert len(action)==self.numJoints, ('number of motor commands differs from number of motor to control',len(action))

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
                                        force=50)

    