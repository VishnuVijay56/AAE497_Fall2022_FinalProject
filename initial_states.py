"""
initial_states.py: stores all the ICs to be used in the MF PD Algorithm
                   to estimate S and W matrices
    - Author: Vishnu Vijay
    - Created: 11/25/22
"""

from mav_state import MAV_State
import numpy as np

alt_opt = [0, 5, 10]
phi_opt = [0, 10]
theta_opt = [0, 10]
psi_opt = [-10, 0, 10]
p_opt = [0.1]
q_opt = [0.1]
r_opt = [0.1]
Va_opt = [18, 20, 25]

de_opt = [-10, 10]
dt_opt = [0.5, 1]
da_opt = [-10, 10]
dr_opt = [-10, 10]


IC_list = []

for o1 in range(len(alt_opt)):
    for o2 in range(len(phi_opt)):
        for o3 in range(len(theta_opt)):
            for o4 in range(len(psi_opt)):
                for o5 in range(len(p_opt)):
                    for o6 in range(len(q_opt)):
                        for o7 in range(len(r_opt)):
                            for o8 in range(len(Va_opt)):
                                ms = MAV_State(alt_opt[o1],
                                               phi_opt[o2] * 3.14 / 180,
                                               theta_opt[o3] * 3.14 / 180,
                                               psi_opt[o4] * 3.14 / 180,
                                               p_opt[o5] * 3.14 / 180,
                                               q_opt[o6] * 3.14 / 180,
                                               r_opt[o7] * 3.14 / 180,
                                               Va_opt[o8])
                                IC_list.append(ms)

num_ICs = 13 #len(IC_list)


x_lon_IC = []
x_lat_IC = []
u_lon_IC = []
u_lat_IC = []
for i in range(num_ICs):
    x_lon_IC.append(IC_list[i].get_lon_state())
    x_lat_IC.append(IC_list[i].get_lat_state())

    IC_input_lon = np.array([[de_opt[0]], [dt_opt[0]]])
    IC_input_lat = np.array([[da_opt[0]], [dr_opt[0]]])

    if (i % 2 == 0):
        IC_input_lon[0] = de_opt[1]
    if (i % 3 == 0):
        IC_input_lon[1] = dt_opt[1]
    if (i % 4 == 0):
        IC_input_lat[1] = da_opt[1]
    if (i % 5 == 0):
        IC_input_lat[1] = dr_opt[1]

    u_lon_IC.append(IC_input_lon)
    u_lat_IC.append(IC_input_lat)

N = 10