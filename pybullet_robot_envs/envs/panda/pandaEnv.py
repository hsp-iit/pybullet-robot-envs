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
        self.workspace_lim = [[0,0.80],[-0.80,0.80],[0,0.40]]
        #number of the end effector
        self.endEffLink = 8 
        self.reset()

    def reset(self):

        self.pandaId = p.loadURDF(self.urdfRootPath, basePosition = self.basePosition, useFixedBase = self.useFixedBase)
        
        self.numJoints = p.getNumJoints(self.pandaId)
        ## Set all joints initial values
        for i in range(self.numJoints):
            p.resetJointState(self.pandaId, i, 0)
            p.setJointMotorControl2(self.pandaId, i, p.POSITION_CONTROL,targetPosition=0,targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        """# set initial hand pose
        if self.useInverseKinematics:
                self.endEffPos = [0.3,0.2,0.3] # x,y,z
                self.endEffOrn = [0.3,0.4,0.35] # roll,pitch,yaw

        self.motorNames = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.pandaId, i)
            if jointInfo[3] > -1:
                self.motorNames.append(str(jointInfo[1]))
        """
    def getJointsRanges(self):
        #to-be-defined
        return 0 
    

    def getActionDimension(self):
        if not self.useInverseKinematics:
            return self.numJoints
            #position x,y,z of hand link + roll/pitch/yaw angles of wrist joints?
        return 6 #only end-eff control 

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
            """
            assert len(action)>=3, ('number of action commands must be equal to 3 at least (dx,dy,dz)',len(action))

            dx, dy, dz = action[:3]

            self.handPos[0] = min(self.workspace_lim[0][1], max(self.workspace_lim[0][0], self.handPos[0] + dx))
            self.handPos[1] = min(self.workspace_lim[1][1], max(self.workspace_lim[1][0], self.handPos[1] + dy))
            self.handPos[2] = min(self.workspace_lim[2][1], max(self.workspace_lim[2][0], self.handPos[2] + dz))

            if not self.useOrientation:
                quat_orn = []

            elif len(action) is 6:
                droll, dpitch, dyaw = action[3:]

                self.handOrn[0] = min(m.pi, max(-m.pi, self.handOrn[0] + droll))
                self.handOrn[1] = min(m.pi, max(-m.pi, self.handOrn[1] + dpitch))
                self.handOrn[2] = min(m.pi, max(-m.pi, self.handOrn[2] + dyaw))

                quat_orn = p.getQuaternionFromEuler(self.handOrn)

            else:
                quat_orn = p.getLinkState(self.icubId,self.motorIndices[-3])[5]

            if self.useNullSpace:
                jointPoses = p.calculateInverseKinematics(self.icubId, self.motorIndices[-1],
                                                        self.handPos,
                                                        jointRanges=self.jr,
                                                        restPoses=self.jr,
                                                        )
            else:
                jointPoses = p.calculateInverseKinematics(self.icubId, self.motorIndices[-1],self.handPos,quat_orn)

            if (self.useSimulation):
                for i in range(self.numJoints):
                    jointInfo = p.getJointInfo(self.icubId, i)
                    if jointInfo[3] > -1:
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
        """
        else:
            assert len(action)==len(self.motorIndices), ('number of motor commands differs from number of motor to control',len(action))

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

    