import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
tablePos = [1.0, 0.0, 0.0]
objPos = [0.41, 0.0, 0.8]
tableID = p.loadURDF("table/table.urdf", tablePos)
objID = p.loadURDF("cube_small.urdf", objPos)


icubId = p.loadSDF("/home/gvezzani/giulia_code/icub_models_to_test/icub-models/iCub/robots/iCubGazeboV2_5/model.sdf")

for i in range (10000):
    p.stepSimulation()
    maxForce = 5
    p.setJointMotorControl2(bodyUniqueId=icubId[0],
        jointIndex=21,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity = 2,
        force = maxForce)
    time.sleep(1./240.)



p.disconnect()
