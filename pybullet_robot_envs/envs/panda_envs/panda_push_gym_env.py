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
from panda_env import pandaEnv
import random
import pybullet_data
import robot_data
from pkg_resources import parse_version


largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis = -1)

class pandaPushGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self, urdfRoot=robot_data.getDataPath(),
                 useIK = 0,
                 isDiscrete = 0,
                 actionRepeat = 1,
                 renders = False,
                 maxSteps = 1000,
                 dist_delta = 0.03, numControlledJoints = 7, fixedPositionObj = True, includeVelObs = True):

        self.action_dim = numControlledJoints
        self._isDiscrete = isDiscrete
        self._timeStep = 1./240.
        self._useIK = useIK
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = False
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._h_table = []
        self._target_dist_max = 0.3
        self._target_dist_min = 0.1
        self._p = p
        self.fixedPositionObj = fixedPositionObj
        self.includeVelObs = includeVelObs

        if self._renders:
          cid = p.connect(p.SHARED_MEMORY)
          if (cid<0):
             cid = p.connect(p.GUI)
          p.resetDebugVisualizerCamera(2.5,90,-60,[0.52,-0.2,-0.33])
        else:
            p.connect(p.DIRECT)

        # self.seed()
        # initialize simulation environment
        self.reset()

        observationDim = len(self._observation)
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype='float32')

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(self._panda.getActionDimension())

        else:
            #self.action_dim = 2 #self._panda.getActionDimension()
            self._action_bound = 1
            action_high = np.array([self._action_bound] * self.action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype='float32')

        self.viewer = None


    def reset(self):
        self.terminated = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        self._envStepCounter = 0


        p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"), useFixedBase= True)
        # Load robot
        self._panda = pandaEnv(self._urdfRoot, timeStep=self._timeStep, basePosition =[0,0,0.625],
            useInverseKinematics= self._useIK, actionSpace = self.action_dim, includeVelObs = self.includeVelObs)


        # Load table and object for simulation
        tableId = p.loadURDF(os.path.join(self._urdfRoot, "franka_description/table.urdf"), useFixedBase=True)


        table_info = p.getVisualShapeData(tableId,-1)[0]
        self._h_table =table_info[5][2] + table_info[3][2]

        #limit panda workspace to table plane
        self._panda.workspace_lim[2][0] = self._h_table
        # Randomize start position of object and target.

        #we take the target point
        self.obj_pose, self.target_pose = self._sample_pose()
        if (self.fixedPositionObj):
            #we use a fixed starting position for the cube
            self._objID = p.loadURDF( os.path.join(self._urdfRoot,"franka_description/cube_small.urdf"), basePosition = [0.7,0.0,0.64])
        else:
            self._objID = p.loadURDF( os.path.join(self._urdfRoot,"franka_description/cube_small.urdf"), basePosition= self.obj_pose)
        #useful to see where is the taget point
        self._targetID = p.loadURDF(os.path.join(self._urdfRoot, "franka_description/domino/domino.urdf"), self.target_pose)

        self._debugGUI()
        p.setGravity(0,0,-9.8)
        # Let the world run for a bit
        for _ in range(10):
            p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)


    def getExtendedObservation(self):

        #get robot observations
        self._observation = self._panda.getObservation()
        #read EndEff position/velocity
        #endEffState = p.getLinkState(self._panda.pandaId, self._panda.endEffLink, computeLinkVelocity = 1)
        #endEffPos = endEffState[0]
        #endEffOrn = endEffState[1]
        #endEffLinkPos = endEffState[4]
        #endEffLinkOrn = endEffState[5]
        #endEffLinkVelL = endEffState[6]
        #endEffLinkVelA = endEffState[7]

        #get object position in space
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)

        #invEndEffPos, invEndEffOrn = p.invertTransform(endEffPos, endEffOrn)
        #endEffEul = p.getEulerFromQuaternion(endEffOrn)

        #objPosInEndEff, objOrnInEndEff = p.multiplyTransforms(invEndEffPos, invEndEffOrn,
        #objPos, objOrn)

        self._observation.extend(list(objPos))
        self._observation.extend(list(objOrn))

        #target position
        self._observation.extend(list(self.target_pose))
        return self._observation


    def step(self, action):
        if self._useIK:
            #TO DO
            return 0

        else:
            action = [float(i*0.05) for i in action]
            return self.step2(action)

    def step2(self,action):

        for i in range(self._actionRepeat):
            self._panda.applyAction(action)
            p.stepSimulation()

            if self._termination():
                break

            self._envStepCounter += 1

        if self._renders:
            time.sleep(self._timeStep)

        self._observation = self.getExtendedObservation()

        reward = self._compute_reward()

        done = self._termination()

        return np.array(self._observation), np.array([reward]), np.array(done), {}



    def render(self, mode="rgb_array", close=False):
        ## TODO Check the behavior of this function
        if mode != "rgb_array":
          return np.array([])

        base_pos,orn = self._p.getBasePositionAndOrientation(self._panda.pandaId)
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


    def _termination(self):

        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        d = goal_distance(np.array(objPos), np.array(self.target_pose))

        if d <= self._target_dist_min:
            self.terminated = True


        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return [True]

        return [False]


    def _compute_reward(self):

        reward = np.float(32.0)
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        endEffAct = self._panda.getObservation()[0:3]
        d1 = goal_distance(np.array(endEffAct), np.array(objPos))
        d2 = goal_distance(np.array(objPos), np.array(self.target_pose))
        reward = - d1 - d2
        #print("--------")
        #print(reward)
        #print("--------")
        if d2 <= self._target_dist_min:
            reward = np.float32(1000.0) + (100 - d2*80)
        return reward


    def _sample_pose(self):
        ws_lim = self._panda.workspace_lim
        px1= np.random.uniform(low=ws_lim[0][0]+0.005*np.random.rand(), high=ws_lim[0][1]-0.005*np.random.rand())
        px2 = np.random.uniform(low=ws_lim[0][0]+0.005*np.random.rand(), high=ws_lim[0][1]-0.005*np.random.rand())
        py1 = np.random.uniform(low=ws_lim[1][0]+0.005*np.random.rand(), high=ws_lim[1][1]-0.005*np.random.rand())
        py2 = np.random.uniform(low=ws_lim[1][0]+0.005*np.random.rand(), high=ws_lim[1][1]-0.005*np.random.rand())

        pz = 0.625
        pose1  = [px1,py1,pz]
        pose2 = [px2,py2,pz]
        return pose1, pose2




    def _debugGUI(self):
        #TO DO
        return 0
