import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0,currentdir)

from collections import OrderedDict
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
import csv


largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

#used in save_data_test
test_steps = 0
test_done = False


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis = -1)


#inherit from different class
class pandaPushGymEnvHERRand(gym.GoalEnv):

    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self, urdfRoot=robot_data.getDataPath(),
                 useIK = 0,
                 isDiscrete = 0,
                 actionRepeat = 1,
                 renders = False,
                 maxSteps = 1000,
                 dist_delta = 0.03,
                 action_space = 7,
                 fixedPositionObj = True,
                 includeVelObs = True,
                 object_position=0,
                 test_phase = False,
                 alg = 'ddpg' ,
                 max_episode_steps = 1000,
                 type_physics = 0):

        self.object_position = object_position
        self.action_dim = action_space
        self._isDiscrete = isDiscrete
        self._param_lambda = 1/np.random.uniform(125,1000)
        self._timeStep = (1.0/240.0) + np.random.exponential(self._param_lambda)
        self._useIK = useIK
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._h_table = []
        self._target_dist_max = 0.3
        self._target_dist_min = 0.1
        self._p = p
        self.fixedPositionObj = fixedPositionObj
        self.includeVelObs = includeVelObs
        self.test_phase = test_phase
        self.alg = alg
        self.max_episode_steps = max_episode_steps
        self.type_physics = type_physics


        if self._renders:
          cid = p.connect(p.SHARED_MEMORY)
          if (cid<0):
             cid = p.connect(p.GUI)
          p.resetDebugVisualizerCamera(2.5,90,-60,[0.52,-0.2,-0.33])
        else:
            p.connect(p.DIRECT)

        # self.seed()
        # initialize simulation environment
        self._observation = self.reset()

        observation_dim = len(self._observation['observation'])

        self.observation_space = spaces.Dict({

            'observation': spaces.Box(-largeValObservation, largeValObservation, shape=(observation_dim,), dtype=np.float32),

            #the archieved goal is the position reached with the object in space
            'achieved_goal': spaces.Box(-largeValObservation, largeValObservation, shape=(3,), dtype=np.float32),

            #the desired goal is the desired position in space
            'desired_goal': spaces.Box(-largeValObservation, largeValObservation, shape=(3,), dtype=np.float32)

            })

        print(self.observation_space)

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(self._panda.getActionDimension())

        else:
            #self.action_dim = 2 #self._panda.getActionDimension()
            self._action_bound = 1
            action_high = np.array([self._action_bound] * self.action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.viewer = None


    def reset(self):


        if (self.test_phase):
            global test_steps,test_done
            if (test_steps != 0 ):
                self.save_data_test()
                test_steps = 0



        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        self._envStepCounter = 0

        p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"), useFixedBase= True)
        # Load robot
        self._panda = pandaEnv(self._urdfRoot, timeStep=self._timeStep, basePosition =[0,0,0.625],
            useInverseKinematics= self._useIK, action_space = self.action_dim, includeVelObs = self.includeVelObs)

        # Load table and object for simulation
        tableId = p.loadURDF(os.path.join(self._urdfRoot, "franka_description/table.urdf"), useFixedBase=True)


        table_info = p.getVisualShapeData(tableId,-1)[0]
        self._h_table =table_info[5][2] + table_info[3][2]

        #limit panda workspace to table plane
        self._panda.workspace_lim[2][0] = self._h_table
        # Randomize start position of object and target.

        #we take the target point

        if (self.fixedPositionObj):
            if(self.object_position==0):
                #we have completely fixed position
                self.obj_pose = [0.6,0.1,0.64]
                self.target_pose = [0.4,0.45,0.64]
                self._objID = p.loadURDF( os.path.join(self._urdfRoot,"franka_description/cube_small.urdf"), basePosition = self.obj_pose)
                self._targetID = p.loadURDF(os.path.join(self._urdfRoot, "franka_description/domino/domino.urdf"), basePosition= self.target_pose)

            elif(self.object_position==1):
                #we have completely fixed position
                self.obj_pose = [np.random.uniform(0.5,0.6),np.random.uniform(0,0.1),0.64]
                self.target_pose = [0.4,0.45,0.64]
                # self.target_pose = [np.random.uniform(0.4,0.5),np.random.uniform(0.45,0.55),0.64]
                self._objID = p.loadURDF( os.path.join(self._urdfRoot,"franka_description/cube_small.urdf"), basePosition = self.obj_pose)
                self._targetID = p.loadURDF(os.path.join(self._urdfRoot, "franka_description/domino/domino.urdf"), basePosition= self.target_pose)

            elif(self.object_position==2):
                #we have completely fixed position
                self.obj_pose = [np.random.uniform(0.4,0.6),np.random.uniform(0,0.2),0.64]
                self.target_pose = [np.random.uniform(0.3,0.5),np.random.uniform(0.35,0.55),0.64]
                self._objID = p.loadURDF( os.path.join(self._urdfRoot,"franka_description/cube_small.urdf"), basePosition = self.obj_pose)
                self._targetID = p.loadURDF(os.path.join(self._urdfRoot, "franka_description/domino/domino.urdf"), basePosition= self.target_pose)

            elif(self.object_position==3):
                print("")
        else:
            self.obj_pose, self.target_pose = self._sample_pose()
            self._objID = p.loadURDF( os.path.join(self._urdfRoot,"franka_description/cube_small.urdf"), basePosition= self.obj_pose)
            #useful to see where is the taget point
            self._targetID = p.loadURDF(os.path.join(self._urdfRoot, "franka_description/domino/domino.urdf"), basePosition= self.target_pose)



        if self.type_physics==1:
            # Randomizing the physics of the object...
            self.currentMass = np.random.uniform(0.1,0.8)
            self.currentFriction = np.random.uniform(0.1,0.7)
            self.currentDamping = np.random.uniform(0.01,0.2)
            p.changeDynamics(self._objID, linkIndex=-1, mass=self.currentMass, lateralFriction=self.currentFriction,
                            linearDamping=self.currentDamping)

            # Randomizing the physics of the robot... (only joints damping and controller gains)
            for i in range(7):
                p.changeDynamics(self._panda.pandaId, linkIndex=i, linearDamping=np.random.uniform(0.25,20))

        elif self.type_physics==2:
            # Randomizing the physics of the object...
            self.currentMass = 0.8
            self.currentFriction = 0.2
            self.currentDamping = 0.2
            p.changeDynamics(self._objID, linkIndex=-1, mass=self.currentMass, lateralFriction=self.currentFriction,
                            linearDamping=self.currentDamping)

            # Randomizing the physics of the robot... (only joints damping and controller gains)
            for i in range(7):
                p.changeDynamics(self._panda.pandaId, linkIndex=i, linearDamping= 0.25)

        self._debugGUI()
        p.setGravity(0,0,-9.8)
        # Let the world run for a bit
        for _ in range(10):
            p.stepSimulation()

        #we take the dimension of the observation

        return self.getExtendedObservation()


    def getExtendedObservation(self):

        observation = self._panda.getObservation()
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self._objID)

        list_obj_pos = list(obj_pos)
        list_obj_orn = list(obj_orn)

        #object velocity
        obj_vel, obj_vel_ang = p.getBaseVelocity(self._objID)

        observation.extend(list_obj_pos)
        observation.extend(list_obj_orn)
        observation.extend(list(obj_vel))
        observation.extend(list(obj_vel_ang))

        #object target position
        observation.extend(list(self.target_pose))

        return OrderedDict([
            ('observation', np.asarray(observation.copy())),
            ('achieved_goal', np.asarray(list_obj_pos.copy())),
            ('desired_goal', np.asarray(list(self.target_pose).copy()))
            ])


    def step(self, action):


        #check if we are running a test
        if(self.test_phase):
            global test_steps, test_done
            test_steps = test_steps + 1
            if (test_steps == self.max_episode_steps):
                test_done = False
                self.reset()


        if self._useIK:
            realPos = [a*0.003 for a in action[:3]]
            realOrn = []
            if self.action_space.shape[-1] is 6:
                realOrn = [a*0.01 for a in action[3:]]

            return self.step2(realPos+realOrn)

        else:
            action = [float(i*0.05) for i in action]
            return self.step2(action)

    def step2(self,action):

        for i in range(self._actionRepeat):
            self._panda.applyAction(action)
            p.stepSimulation()
            self._envStepCounter += 1

        if self._renders:
            time.sleep(self._timeStep)

        self._observation = self.getExtendedObservation()
        reward = self.compute_reward(self._observation ['achieved_goal'], self._observation ['desired_goal'], None)
        #if the reward is zero done = TRUE
        done = reward == 0
        info = {'is_success': done}
        done = done or self._envStepCounter >= self._maxSteps
        return self._observation, reward, done, info



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

    def compute_reward(self, achieved_goal, desired_goal, info):

        global test_done
        #evaluating the distance
        d = goal_distance(np.array(achieved_goal), np.array(desired_goal))
        if d <= self._target_dist_min:
            #reward = 0, good boy
            test_done = True
            return 0
        else:
            #negative reward, objective not achieved
            return -1


    def _sample_pose(self):
        ws_lim = self._panda.workspace_lim
        px1= np.random.uniform(low=ws_lim[0][0]+0.005*np.random.rand(), high=ws_lim[0][1]-0.005*np.random.rand())
        py1 = np.random.uniform(low=ws_lim[1][0]+0.005*np.random.rand(), high=ws_lim[1][1]-0.005*np.random.rand())

        if px1 < 0.45:
            px2 = px1 + np.random.uniform(0.1,0.2)
        else:
            px2 =  px1 - np.random.uniform(0.1,0.2)
        if py1 < 0:
            py2 = py1 + np.random.uniform(0.2,0.3)
        else:
            py2 = py1 - np.random.uniform(0.2,0.3)

        pz = 0.625
        pose1  = [px1,py1,pz]
        pose2 = [px2,py2,pz]
        return pose1, pose2

    def save_data_test(self):

        global test_steps, test_done
        row = [test_steps, test_done, self.currentMass, self.currentFriction , self.currentDamping, self._timeStep]
        with open('test_panda_push_'+ self.alg+'.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
            csvFile.close()

    def _debugGUI(self):
        #TO DO
        return 0
