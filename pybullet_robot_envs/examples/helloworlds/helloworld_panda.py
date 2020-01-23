# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import pybullet as p
import pybullet_data
import time
import os
import math as m
import robot_data


def main():
	# Open GUI and set pybullet_data in the path
	p.connect(p.GUI)
	p.resetSimulation()
	p.setPhysicsEngineParameter(numSolverIterations=150)
	p.setTimeStep(1/240.)

	# Load plane contained in pybullet_data
	planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"))

	# Set gravity for simulation
	p.setGravity(0, 0, -9.8)

	flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE

	pandaId = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),
						basePosition=[0.6, 0, 0.625], useFixedBase=True, flags=flags)

	init_pos = [0, -0.54, 0, -2.6, -0.30, 2, 1, 0.02, 0.02]

	# Load other objects
	p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0])
	p.loadURDF(os.path.join(pybullet_data.getDataPath(), "lego/lego.urdf"), [1, 0.0, 0.8])

	# add debug slider
	jointIds = []
	paramIds = []

	# reset joints to home position
	num_joints = p.getNumJoints(pandaId)
	idx = 0
	for i in range(num_joints):
		jointInfo = p.getJointInfo(pandaId, i)
		jointName = jointInfo[1]
		jointType = jointInfo[2]

		if jointType is p.JOINT_REVOLUTE or jointType is p.JOINT_PRISMATIC:
			p.resetJointState(pandaId, i, init_pos[idx])
			jointIds.append(i)
			paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), jointInfo[8], jointInfo[9], init_pos[idx]))
			idx += 1

	while True:
		new_pos = []
		for i in range(len(jointIds)):
			new_pos.append(p.readUserDebugParameter(i))
		p.setJointMotorControlArray(pandaId, jointIds, p.POSITION_CONTROL, targetPositions=new_pos, forces=[50]*len(jointIds))

		p.stepSimulation()
		time.sleep(0.01)


if __name__ == '__main__':
    main()
