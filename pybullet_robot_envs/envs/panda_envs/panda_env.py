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

    def __init__(self, urdfRootPath=robot_data.getDataPath(), timeStep=0.01, useInverseKinematics=0, basePosition=[-0.6,-0.4,0.625], useFixedBase= True, action_space = 7, includeVelObs = True):

        self.urdfRootPath = os.path.join(urdfRootPath, "franka_description/robots/panda_arm_physics.urdf")
        self.timeStep = timeStep
        self.useInverseKinematics = useInverseKinematics
        self.useNullSpace = 0
        self.useOrientation = 1
        self.useSimulation = 1
        self.basePosition = basePosition
        self.useFixedBase = useFixedBase
        self.workspace_lim = [[0.3,0.60],[-0.3,0.3],[0,1]]
        self.workspace_lim_endEff = [[0.1,0.70],[-0.4,0.4],[0.65,1]]
        self.endEffLink = 8
        self.action_space = action_space
        self.includeVelObs = includeVelObs
        self.numJoints = 7
        self.reset()

    def reset(self):

        self.pandaId = p.loadURDF(self.urdfRootPath, basePosition = self.basePosition, useFixedBase = self.useFixedBase)

        for i in range(self.numJoints):
            p.resetJointState(self.pandaId, i, 0)
            p.setJointMotorControl2(self.pandaId, i, p.POSITION_CONTROL,targetPosition=0,targetVelocity=0.0,
            positionGain=0.25, velocityGain=0.75, force=50)
        if self.useInverseKinematics:
            self.endEffPos = [0.4,0,0.85] # x,y,z
            self.endEffOrn = [0.3,0.4,0.35] # roll,pitch,yaw

    def getJointsRanges(self):
        #to-be-defined
        return 0


    def getActionDimension(self):
            return self.action_space


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
            assert len(action)>=3, ('IK dim differs from ',len(action))
            assert len(action)<=6, ('IK dim differs from ',len(action))

            dx, dy, dz = action[:3]

            self.endEffPos[0] = min(self.workspace_lim_endEff[0][1], max(self.workspace_lim_endEff[0][0], self.endEffPos[0] + dx))
            self.endEffPos[1] = min(self.workspace_lim_endEff[1][1], max(self.workspace_lim_endEff[1][0], self.endEffPos[1] + dx))
            self.endEffPos[2] = min(self.workspace_lim_endEff[2][1], max(self.workspace_lim_endEff[2][0], self.endEffPos[2] + dx))

            if not self.useOrientation:
                quat_orn = p.getQuaternionFromEuler(self.handOrn)

            elif len(action) is 6:
                droll, dpitch, dyaw = action[3:]
                self.endEffOrn[0] = min(m.pi, max(-m.pi, self.endEffOrn[0] + droll))
                self.endEffOrn[1] = min(m.pi, max(-m.pi, self.endEffOrn[1] + dpitch))
                self.endEffOrn[2] = min(m.pi, max(-m.pi, self.endEffOrn[2] + dyaw))
                quat_orn = p.getQuaternionFromEuler(self.endEffOrn)

            else:
                quat_orn = p.getLinkState(self.pandaId, self.endEffLink)[5]

            jointPoses = p.calculateInverseKinematics(self.pandaId, self.endEffLink, self.endEffPos, quat_orn)

            if (self.useSimulation):
                    for i in range(self.numJoints):
                        jointInfo = p.getJointInfo(self.pandaId, i)
                        if jointInfo[3] > -1:
                            p.setJointMotorControl2(bodyUniqueId=self.pandaId,
                                                    jointIndex=i,
                                                    controlMode=p.POSITION_CONTROL,
                                                    targetPosition=jointPoses[i],
                                                    targetVelocity=0,
                                                    positionGain=0.25,
                                                    velocityGain=0.75,
                                                    force=50)
            else:
                for i in range(self.numJoints):
                    p.resetJointState(self.pandaId, i, jointPoses[i])


        else:
            assert len(action)==self.action_space, ('number of motor commands differs from number of motor to control',len(action))

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
