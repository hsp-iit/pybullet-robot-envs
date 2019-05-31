import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import math as m
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from icub_env import iCubEnv
import random
import pybullet_data
import robot_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class iCubPushGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self,
                 urdfRoot=robot_data.getDataPath(),
                 useIK=1,
                 control_arm='l',
                 isDiscrete=0,
                 actionRepeat=1,
                 renders=False,
                 maxSteps = 1000):

        self._control_arm=control_arm
        self._isDiscrete = isDiscrete
        self._timeStep = 1./240.
        self._useIK = 1 if self._isDiscrete else useIK
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._h_table = []
        self._target_dist_max = 0.1
        self._target_dist_min = 0.07

        # Initialize PyBullet simulator
        self._p = p
        if self._renders:
          cid = p.connect(p.SHARED_MEMORY)
          if (cid<0):
             cid = p.connect(p.GUI)
          p.resetDebugVisualizerCamera(2.5,90,-60,[0.52,-0.2,-0.33])
        else:
            p.connect(p.DIRECT)

        self.seed()
        # initialize simulation environment
        self.reset()

        observationDim = len(self._observation)
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype='float32')

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(13) if self._icub.useOrientation else spaces.Discrete(7)

        else:
            action_dim = self._icub.getActionDimension()
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype='float32')

        self.viewer = None

    def reset(self):

        self.terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        self._envStepCounter = 0

        p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"),[0,0,0])

        # Load robot
        self._icub = iCubEnv(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, useInverseKinematics=self._useIK, arm=self._control_arm)

        ## Load table and object for simulation
        tableId = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), [0.85, 0.0, 0.0])
        table_info = p.getVisualShapeData(tableId,-1)[0]
        self._h_table =table_info[5][2] + table_info[3][2]

        #limit iCub workspace to table plane
        self._icub.workspace_lim[2][0] = self._h_table

        # Randomize start position of object and target.
        obj_pose, self.target_pose = self._sample_pose()
        self._objID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "lego/lego.urdf"),obj_pose)
        self._targetID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "domino/domino.urdf"), self.target_pose)

        #limit iCub workspace to table plane
        self._icub.workspace_lim[2][0] = self._h_table+0.01

        self._debugGUI()
        p.setGravity(0,0,-9.8)
        # Let the world run for a bit
        for _ in range(10):
            p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def getExtendedObservation(self):
        # get robot observation
        self._observation = self._icub.getObservation()
        # read hand position/velocity
        handState = p.getLinkState(self._icub.icubId, self._icub.motorIndices[-1], computeLinkVelocity=1)
        handPos = handState[0]
        handOrn = handState[1]
        handLinkPos = handState[4]
        handLinkOrn = handState[5]
        handLinkVelL = handState[6]
        handLinkVelA = handState[7]

        # get object position and transform it wrt hand c.o.m. frame
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        invHandPos, invHandOrn = p.invertTransform(handPos, handOrn)
        handEul = p.getEulerFromQuaternion(handOrn)

        objPosInHand, objOrnInHand = p.multiplyTransforms(invHandPos, invHandOrn,
                                                                objPos, objOrn)

        self._observation.extend(list(objPos))
        self._observation.extend(list(objOrn))
        #self._observation.extend(list(handLinkVelL))
        #self._observation.extend(list(handLinkVelA))
        return self._observation

    def step(self, action):
        ws_lim = self._icub.workspace_lim
        if (self._isDiscrete):

            dv = 0.003
            if not self._icub.useOrientation:
                dx = [0, -dv, dv, 0, 0, 0, 0][action]
                dy = [0, 0, 0, -dv, dv, 0, 0][action]
                dz = [0, 0, 0, 0, 0, -dv, dv][action]
                realAction = [dx, dy, dz]
            else:
                dv1 = 0.05
                dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][action]
                dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0][action]
                dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
                droll = [0, 0, 0, 0, 0, 0, 0, -dv1, dv1, 0, 0, 0, 0][action]
                dpitch = [0, 0, 0, 0, 0, 0, 0, 0, 0, -dv1, dv1, 0, 0][action]
                dyaw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dv1, dv1][action]
                #dz = - abs(dz) #force arm to go down
                realAction = [dx, dy, dz, droll, dpitch, dyaw]

            return self.step2(realAction)

        elif self._useIK:

            dv = 0.01
            realPos = [a*0.005 for a in action[:3]]
            realOrn = []
            #realPos[2] = - abs(realPos[2])

            if self.action_space.shape[-1] is 6:
                realOrn = [a*0.05 for a in action[3:]]

            return self.step2(realPos+realOrn)

        else:
            return self.step2([a*0.05 for a in action])

    def step2(self, action):
        for i in range(self._actionRepeat):
            self._icub.applyAction(action)
            p.stepSimulation()

            if self._termination():
                break

            self._envStepCounter += 1

        if self._renders:
            time.sleep(self._timeStep)

        self._observation = self.getExtendedObservation()

        done = self._termination()

        reward = self._compute_reward()

        ws_lim = self._icub.workspace_lim
        cost = (not ws_lim[0][0]<self._observation[0]<ws_lim[0][1]) or \
                (not ws_lim[1][0]<self._observation[1]<ws_lim[1][1]) or \
                (not ws_lim[2][0]<self._observation[2]<ws_lim[2][1])

        #reward -= cost*1
        #print("reward")
        #print(reward)

        return np.array(self._observation), np.array(reward), np.array(done), {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="rgb_array", close=False):
        ## TODO Check the behavior of this function
        if mode != "rgb_array":
          return np.array([])

        base_pos,orn = self._p.getBasePositionAndOrientation(self._icub.icubId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
            #renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def __del__(self):
        p.disconnect()

    # TODO
    def _termination(self):

        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return np.float32(1.0)

        handState = p.getLinkState(self._icub.icubId, self._icub.motorIndices[-1], computeLinkVelocity=0)
        handPos = handState[0]
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        d = goal_distance(np.array(handPos), np.array(objPos))

        if d <= self._target_dist_min:
            self.terminated = 1
        return (d <= self._target_dist_min)

    def _compute_reward(self):

        reward = np.float32(0.0)

        handState = p.getLinkState(self._icub.icubId, self._icub.motorIndices[-1], computeLinkVelocity=0)
        handPos = handState[0]
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        tgPos, tgOrn = p.getBasePositionAndOrientation(self._targetID)
        d1 = goal_distance(np.array(handPos), np.array(objPos))
        d2 = goal_distance(np.array(objPos), np.array(tgPos))
        #print("distance")
        #print(d1)
        #if d < self._target_dist_max:
        reward = -d1#1/3*-d1+2/3*-d2
        if d1 <= self._target_dist_min:
            reward = np.float32(1000.0)

        return reward

    def _sample_pose(self):
        ws_lim = self._icub.workspace_lim
        px,tx =  0.2+0.2*np.random.random(), 0.5#np.random.uniform(low=ws_lim[0][0], high=ws_lim[0][1], size=(2))
        py,ty =  0.2-0.2*np.random.random(),-0.1#np.random.uniform(low=ws_lim[1][0], high=ws_lim[1][1], size=(2))
        pz,tz = self._h_table, self._h_table
        obj_pose = [px,py,pz]
        tg_pose = [tx,ty,tz]

        return obj_pose, tg_pose

    def _debugGUI(self):
        ws = self._icub.workspace_lim
        p1 = [ws[0][0],ws[1][0],ws[2][0]] # xmin,ymin
        p2 = [ws[0][1],ws[1][0],ws[2][0]] # xmax,ymin
        p3 = [ws[0][1],ws[1][1],ws[2][0]] # xmax,ymax
        p4 = [ws[0][0],ws[1][1],ws[2][0]] # xmin,ymax

        p.addUserDebugLine(p1,p2,lineColorRGB=[0,0,1],lineWidth=2.0,lifeTime=0)
        p.addUserDebugLine(p2,p3,lineColorRGB=[0,0,1],lineWidth=2.0,lifeTime=0)
        p.addUserDebugLine(p3,p4,lineColorRGB=[0,0,1],lineWidth=2.0,lifeTime=0)
        p.addUserDebugLine(p4,p1,lineColorRGB=[0,0,1],lineWidth=2.0,lifeTime=0)

        p.addUserDebugLine([-0.07,0,0],[0,0,0],[1,0,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_right_arm[-1])
        p.addUserDebugLine([-0.07,0,0.03],[-0.07,0.1,0],[0,1,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_right_arm[-1])
        p.addUserDebugLine([-0.07,0,0.03],[-0.07,0,0.1],[0,0,1],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_right_arm[-1])

        p.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_left_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_left_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_left_arm[-1])

        p.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=self._objID)
        p.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=self._objID)
        p.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=self._objID)
