# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import robot_data
import math as m

class iCubEnv:

    def __init__(self, urdfRootPath=robot_data.getDataPath(),
                    timeStep=0.01,
                    useInverseKinematics=0, arm='l', useOrientation=0):

        self.urdfRootPath = os.path.join(urdfRootPath, "iCub/icub_fixed_model.sdf")
        self.timeStep = timeStep
        self.useInverseKinematics = useInverseKinematics
        self.useOrientation = useOrientation
        self.useSimulation = 1

        self.indices_torso = range(12,15)
        self.indices_left_arm = range(15,22)
        self.indices_right_arm = range(25,32)
        self.indices_head = range(22,25)

        self.home_pos_torso = [0.0, 0.0, 0.0] #degrees
        self.home_pos_head = [0.47, 0, 0]

        self.home_left_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]
        self.home_right_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]

        self.workspace_lim = [[0.25,0.52],[-0.2,0.2],[0.5,1.0]]

        self.control_arm = arm if arm =='r' or arm =='l' else 'l' #left arm by default

        self.reset()

    def reset(self):
        self.icubId = p.loadSDF(self.urdfRootPath)[0]
        self.numJoints = p.getNumJoints(self.icubId)

        # set constraint between base_link and world
        self.constr_id = p.createConstraint(self.icubId,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],
                                 [p.getBasePositionAndOrientation(self.icubId)[0][0],
                                  p.getBasePositionAndOrientation(self.icubId)[0][1],
                                  p.getBasePositionAndOrientation(self.icubId)[0][2]*1.2],
                                  p.getBasePositionAndOrientation(self.icubId)[1])

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

        # set initial hand pose
        if self.useInverseKinematics:
            if self.control_arm =='l':
                self.handPos = [0.25,0.35,0.85] # x,y,z
                self.handOrn = [0.3,0.4,0.35] # roll,pitch,yaw
            else:
                self.handPos = [0.25,-0.35,0.85] # x,y,z
                self.handOrn = [0.3, -0.4, 2.8] # roll,pitch,yaw


        # save indices of only the joints to control
        control_arm_indices = self.indices_left_arm if self.control_arm =='l' else self.indices_right_arm
        self.motorIndices = [i for i in self.indices_torso] + [j for j in control_arm_indices]

        self.motorNames = []
        for i in self.indices_torso:
            jointInfo = p.getJointInfo(self.icubId, i)
            if jointInfo[3] > -1:
                self.motorNames.append(str(jointInfo[1]))
        for i in control_arm_indices:
            jointInfo = p.getJointInfo(self.icubId, i)
            if jointInfo[3] > -1:
                self.motorNames.append(str(jointInfo[1]))

    def getJointRanges(self):

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
        if not self.useInverseKinematics:
            return len(self.motorIndices)
        if self.useOrientation:
            return 6 #position x,y,z + roll/pitch/yaw of hand frame
        return 3 #position x,y,z

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        #Cartesian world pos/orn of left hand center of mass
        observation = []
        state = p.getLinkState(self.icubId, self.motorIndices[-1],computeLinkVelocity=1)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)
        velL = state[6]
        velA = state[7]

        observation.extend(list(pos))
        observation.extend(list(euler)) #roll, pitch, yaw
        observation.extend(list(velL))
        #observation.extend(list(velA))

        jointStates = p.getJointStates(self.icubId,self.motorIndices)
        jointPoses = [x[0] for x in jointStates]
        observation.extend(list(jointPoses))

        return observation

    def applyAction(self, action):

        if(self.useInverseKinematics):

            if not len(action)>=3:
                raise AssertionError('number of action commands must be equal to 3 at least (dx,dy,dz)',len(action))

            dx, dy, dz = action[:3]

            self.handPos[0] = min(self.workspace_lim[0][1], max(self.workspace_lim[0][0], self.handPos[0] + dx))
            self.handPos[1] = min(self.workspace_lim[1][1], max(self.workspace_lim[1][0], self.handPos[1] + dy))
            self.handPos[2] = min(self.workspace_lim[2][1], max(self.workspace_lim[2][0], self.handPos[2] + dz))

            if not self.useOrientation:
                quat_orn = p.getQuaternionFromEuler(self.handOrn) #[]

            elif len(action) is 6:
                droll, dpitch, dyaw = action[3:]

                self.handOrn[0] = min(m.pi, max(-m.pi, self.handOrn[0] + droll))
                self.handOrn[1] = min(m.pi, max(-m.pi, self.handOrn[1] + dpitch))
                self.handOrn[2] = min(m.pi, max(-m.pi, self.handOrn[2] + dyaw))

                quat_orn = p.getQuaternionFromEuler(self.handOrn)

            else:
                quat_orn = p.getLinkState(self.icubId,self.motorIndices[-3])[5]

            # compute joint positions with IK
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

        else:
            if not len(action)==len(self.motorIndices):
                raise AssertionError('number of motor commands differs from number of motor to control', len(action),len(self.motorIndices))

            for idx,val in enumerate(action):
                motor = self.motorIndices[idx]

                curr_motor_pos = p.getJointState(self.icubId, motor)[0]
                new_motor_pos = min(self.ul[motor], max(self.ll[motor], curr_motor_pos + val))

                p.setJointMotorControl2(self.icubId,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        targetVelocity=0,
                                        positionGain=0.25,
                                        velocityGain=0.75,
                                        force=50)
