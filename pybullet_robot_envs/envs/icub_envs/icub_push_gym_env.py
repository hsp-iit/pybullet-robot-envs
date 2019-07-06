import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0,currentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import math as m
import pybullet as p
from icub_env import iCubEnv
import pybullet_data
import robot_data

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

def goal_distance(goal_a, goal_b):
    if not goal_a.shape == goal_b.shape:
        raise AssertionError("shape of goals mismatch")
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class iCubPushGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self,
                 urdfRoot=robot_data.getDataPath(),
                 actionRepeat=4,
                 useIK=1,
                 isDiscrete=0,
                 control_arm='l',
                 useOrientation=0,
                 rnd_obj_pose=1,
                 renders=False,
                 maxSteps = 2000,
                 reward_type = 1):

        self._control_arm=control_arm
        self._isDiscrete = isDiscrete
        self._timeStep = 1./240.
        self._useIK = 1 if self._isDiscrete else useIK
        self._useOrientation = useOrientation
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
        self._target_dist_min = 0.03
        self._rnd_obj_pose = rnd_obj_pose
        self._reward_type = reward_type

        # Initialize PyBullet simulator
        self._p = p
        if self._renders:
          self._cid = p.connect(p.SHARED_MEMORY)
          if (self._cid<0):
             self._cid = p.connect(p.GUI)
          p.resetDebugVisualizerCamera(2.5,90,-60,[0.0,-0.0,-0.0])
        else:
            self._cid = p.connect(p.DIRECT)

        # initialize simulation environment
        self.seed()
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
        self._icub = iCubEnv(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, useInverseKinematics=self._useIK,
                             arm=self._control_arm, useOrientation=self._useOrientation)

        ## Load table and object for simulation
        self._tableId = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), [0.85, 0.0, 0.0])

        #limit iCub workspace to table plane
        table_info = p.getVisualShapeData(self._tableId,-1)[0]
        self._h_table =table_info[5][2] + table_info[3][2]
        self._icub.workspace_lim[2][0] = self._h_table

        # Randomize start position of object
        obj_pose, self._tg_pose = self._sample_pose()
        self._objID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "lego/lego.urdf"),obj_pose)

        self._debugGUI()
        p.setGravity(0,0,-9.8)
        # Let the world run for a bit
        for _ in range(10):
            p.stepSimulation()

        self._observation = self.getExtendedObservation()

        self._init_dist_hand_obj = goal_distance(np.array(self._observation[0:3]), np.array(obj_pose))
        self._max_dist_obj_tg = goal_distance(np.array(obj_pose), np.array(self._tg_pose))

        return np.array(self._observation)

    def getExtendedObservation(self):
        # get robot observation
        self._observation = self._icub.getObservation()

        # get object position and transform it wrt hand c.o.m. frame
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        objEuler = p.getEulerFromQuaternion(objOrn) #roll, pitch, yaw

        objLVel, objAVel = p.getBaseVelocity(self._objID)

        self._observation.extend(list(objPos))
        self._observation.extend(list(objEuler))
        self._observation.extend(list(objLVel))
        self._observation.extend(list(objAVel))

        # relative object position wrt hand c.o.m. frame
        invHandPos, invHandOrn = p.invertTransform(self._observation[:3], p.getQuaternionFromEuler(self._observation[3:6]))

        objPosInHand, objOrnInHand = p.multiplyTransforms(invHandPos, invHandOrn,
                                                                objPos, objOrn)

        objEulerInHand = p.getEulerFromQuaternion(objOrnInHand)
        self._observation.extend(list(objPosInHand))
        self._observation.extend(list(objEulerInHand))

        relLVel = np.array(objAVel) - np.array(self._observation[6:9])
        self._observation.extend(list(relLVel))

        self._observation.extend(list(self._tg_pose))

        return np.array(self._observation)

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
                dv1 = 0.005
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

            realPos = [a*0.003 for a in action[:3]]
            realOrn = []
            if self.action_space.shape[-1] is 6:
                realOrn = [a*0.01 for a in action[3:]]

            return self.step2(realPos+realOrn)

        else:
            return self.step2([a*0.05 for a in action])

    def step2(self, action):
        for _ in range(self._actionRepeat):
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

        return self._observation, np.array(reward), np.array(done), {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="rgb_array", close=False):
        ## TODO Check the behavior of this function
        if mode != "rgb_array":
          return np.array([])

        base_pos,_ = self._p.getBasePositionAndOrientation(self._icub.icubId)
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
        self._p.disconnect(self._cid)

    def _termination(self):

        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return np.float32(1.0)

        objPos, _ = p.getBasePositionAndOrientation(self._objID)
        d = goal_distance(np.array(objPos), np.array(self._tg_pose))

        if d <= self._target_dist_min:
            self.terminated = 1
            print('------------->>> success!')
            print('final reward')
            print(self._compute_reward())

        return (d <= self._target_dist_min)

    def _compute_reward(self):

        reward = np.float32(0.0)

        handState = p.getLinkState(self._icub.icubId, self._icub.motorIndices[-1], computeLinkVelocity=0)
        handPos = handState[0]
        objPos, _ = p.getBasePositionAndOrientation(self._objID)
        d1 = goal_distance(np.array(handPos), np.array(objPos))
        d2 = goal_distance(np.array(objPos), np.array(self._tg_pose))

        if self._reward_type is 0:
            reward = -d1 -d2
            if d2 <= self._target_dist_min:
                reward += np.float32(1000.0)
        #normalized reward
        elif self._reward_type is 1:

            rew1 = 0.125
            rew2 = 0.25
            if d1 > 0.1:
                reward = rew1 * (1 - d1/self._init_dist_hand_obj)
                #print("reward 1 ", reward)
            else:
                reward = rew1 * (1 - d1/self._init_dist_hand_obj) + rew2 * (1 - d2/self._max_dist_obj_tg)
                #print("reward 2 ", reward)

            if d2 <= self._target_dist_min:
                reward += np.float32(1000.0)

        return reward

    def _sample_pose(self):

        ws_lim = self._icub.workspace_lim

        px1 = ws_lim[0][0] + 0.6*(ws_lim[0][1]-ws_lim[0][0])
        py1 = ws_lim[1][0] + 0.5*(ws_lim[1][1]-ws_lim[1][0])
        pz = self._h_table

        px2, py2 = px1, py1
        if self._rnd_obj_pose:

            while goal_distance(np.array([px1,py1,pz]), np.array([px2,py2,pz])) < self._target_dist_min:
                px2 = self.np_random.uniform(low=ws_lim[0][0]+0.08*self.np_random.rand(), high=ws_lim[0][1]-0.005*self.np_random.rand())
                py2 = self.np_random.uniform(low=ws_lim[1][0]+0.005*self.np_random.rand(), high=ws_lim[1][1]-0.005*self.np_random.rand())

        else:
            px2 = ws_lim[0][0] + 0.6*(ws_lim[0][1]-ws_lim[0][0]), ws_lim[0][0] + 0.2*(ws_lim[0][1]-ws_lim[0][0])
            py2 = ws_lim[1][0] + 0.5*(ws_lim[1][1]-ws_lim[1][0]), ws_lim[1][0] + 0.5*(ws_lim[1][1]-ws_lim[1][0])


        pose1  = [px1,py1,pz]
        pose2 = [px2,py2,pz]

        return pose1, pose2

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

        p.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_right_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_right_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_right_arm[-1])

        p.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_left_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_left_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_left_arm[-1])

        p.addUserDebugLine([0.0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=self._objID)
        p.addUserDebugLine([0.0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=self._objID)
        p.addUserDebugLine([0.0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=self._objID)

        p.addUserDebugLine(self._tg_pose,[self._tg_pose[i]+n for i,n in enumerate([0.1,0,0])],[1,0,0],lineWidth=2.0,lifeTime=0)
        p.addUserDebugLine(self._tg_pose,[self._tg_pose[i]+n for i,n in enumerate([0,0.1,0])],[0,1,0],lineWidth=2.0,lifeTime=0)
        p.addUserDebugLine(self._tg_pose,[self._tg_pose[i]+n for i,n in enumerate([0,0,0.1])],[0,0,1],lineWidth=2.0,lifeTime=0)
