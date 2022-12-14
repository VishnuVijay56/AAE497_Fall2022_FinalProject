"""
AAE497_project.py: testing primal-dual algos
    - Author: Vishnu Vijay
    - Created: 11/12/22
"""

import mav_body_parameter as MBP
import AAE497_project_functions as F

from mav_state import MAV_State


skip_calculation = False
model_known = False
disp_conv = True
eps = 1e-5

init_state = MAV_State(-MBP.down0, MBP.phi0, MBP.theta0, MBP.psi0, 
                        MBP.p0, MBP.q0, MBP.r0, MBP.Va0)

### CALCULATE OPTIMAL FEEDBACK GAINS - MODEL KNOWN ###
if ((not skip_calculation) and (model_known)):
    F.imp_MK_algo(eps, disp_conv, 3)

# RUN SIMULATION #
if (model_known):
    F.run_sim(1, init_state, True)


### CALCULATE OPTIMAL FEEDBACK GAINS - MODEL FREE ###
if ((not skip_calculation) and (not model_known)):
    F.imp_MF_algo(eps, disp_conv)

if (not model_known):
    F.run_sim(2, init_state, True)
