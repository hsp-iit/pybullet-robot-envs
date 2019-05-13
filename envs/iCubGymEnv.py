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
import icub
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class iCubGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self,
                 urdfRoot=currentdir+'/icub_fixed_model.sdf', ## TODO
                 useInverseKinematics=1,
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps = 1000):

        self._isDiscrete = isDiscrete
        self._timeStep = 1./240.
        self._useInverseKinematics = useInverseKinematics
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40

        self._p = p
        if self._renders:
          cid = p.connect(p.SHARED_MEMORY)
          if (cid<0):
             cid = p.connect(p.GUI)
          p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
        else:
            p.connect(p.DIRECT)

        self.seed()
        self.reset()
        observationDim = len(self.getExtendedObservation())
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        action_dim = self._icub.getActionDimension()
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)


        self.viewer = None

    def reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)

        ## Load table and object for simulation
        p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"),[0,0,0])
        tablePos = [0.85, 0.0, 0.0]
        p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), tablePos)
        objPos = [0.41, 0.0, 0.8]
        self.objID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube_small.urdf"), objPos)

        p.setGravity(0,0,-10)

        # Load robot
        self._icub = icub.iCub(urdfRootPath=self._urdfRoot,
                                timeStep=self._timeStep,
                                useInverseKinematics=self._useInverseKinematics)

        self._envStepCounter = 0
        # Let the world run for a bit
        for _ in range(50):
            p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def getExtendedObservation(self):
        self._observation = self._icub.getObservation()
        handState = p.getLinkState(self._icub.icubId, self._icub.indices_left_arm[-1])
        handPos = handState[0]
        handOrn = handState[1]

        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.objID)
        invHandPos, invHandOrn = p.invertTransform(handPos, handOrn)
        handEul = p.getEulerFromQuaternion(handOrn)

        cubePosInHand, cubeOrnInHand = p.multiplyTransforms(invHandPos, invHandOrn,
                                                                cubePos, cubeOrn)
        projectedCubePos2D = [cubePosInHand[0], cubePosInHand[1]]
        cubeEulerInHand = p.getEulerFromQuaternion(cubeOrnInHand)

        #we return the relative x,y position and euler angle of cube in hand space
        cubeInHandPosXYEulZ = [cubePosInHand[0], cubePosInHand[1], cubeEulerInHand[2]]

        self._observation.extend(list(cubeInHandPosXYEulZ))
        return self._observation

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

    def step0(self):
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation), 0, 0, {}

    def step(self, action):

        if self._icub.useInverseKinematics:
            dv = 0.005
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[1] * 0.002
            realAction = [dx, dy, dz, dy]
        else:
            dv = 0.01
            realAction = [i * dv for i in action]
        return self.step2(realAction)

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

        reward = 0
        return np.array(self._observation), reward, done, {}

    # TODO
    def _termination(self):
        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return True

        #if target as reached:
            #self.terminated = 1
            #...
            #self._observation = self.getExtendedObservation()
            #return True
        return False

    # TODO
    def _reward(self):
        reward = -1000
        return reward
