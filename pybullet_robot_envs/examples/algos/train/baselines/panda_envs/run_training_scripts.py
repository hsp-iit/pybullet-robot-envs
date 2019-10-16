# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
os.sys.path.insert(0, parentdir)



import train_TD3_pushing_HER as train_phase1
import train_TD3_pushing_HER_Dyn_Rand as trainDynRand
import train_TD3_pushing_HER_IK as trainIK


#trainDynRand.main(False)
#trainDynRand.main(True)
#train_phase1.main(True)
trainIK.main(False)
