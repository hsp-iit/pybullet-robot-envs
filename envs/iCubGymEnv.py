import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import math
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
                 urdfRoot='/home/gvezzani/giulia_code/icub_models_to_test/icub-models/iCub/robots/iCubGenova01', ## TODO
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps = 1000):

        self._isDiscrete = isDiscrete
        self._timeStep = 1./240.
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
        observationDim = 3
        print("observationDim")
        print(observationDim)

        observation_high = np.array([largeValObservation] * observationDim)

        action_dim = 10
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def reset(self):
        p.setGravity(0,0,-10)
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)

        p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"),[0,0,0])
        objPos = [0.41, 0.0, 0.8]
        objID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube_small.urdf"), objPos)

        p.setGravity(0,0,-10)
        self._icub = icub.iCub(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

        tablePos = [0.8, 0.0, 0.0]
        p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), tablePos)

        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = [0, 0, 0]
        return np.array(self._observation)

    def render(self, mode="rgb_array", close=False):
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

    def step(self):
        p.stepSimulation()
