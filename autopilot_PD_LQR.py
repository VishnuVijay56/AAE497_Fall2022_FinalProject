"""
autopilot_PD_LQR.py: Takes set of gains calculated from LQR theory and 
    - Author: Vishnu Vijay
    - Created: 11/22/22
    - History:
        - 
"""

import numpy as np

import control_parameters as AP
from wrap import wrap
import model_coef as M
from helper import QuaternionToEuler

from mav_state import MAV_State
from delta_state import Delta_State


class Autopilot:
    def __init__(self, ts_control, F_lon, F_lat):
        # set time step
        self.Ts = ts_control

        # Trim state
        self.trim_d_e = M.u_trim.item(0)
        self.trim_d_a = M.u_trim.item(1)
        self.trim_d_r = M.u_trim.item(2)
        self.trim_d_t = M.u_trim.item(3)

        # Set State Gains
        self.F_lon = F_lon
        self.F_lat = F_lat

        self.commanded_state = MAV_State()


    def update(self, case, cmd, state):
        # Lateral Autopilot

        if (case != 1):
            x_lat = state.get_lat_state()
            temp = self.F_lat @ x_lat

            delta_a = self.saturate(1*temp.item(0) + 1*self.trim_d_a, -np.radians(30), np.radians(30))
            delta_r = self.saturate(1*temp.item(1) + 1*self.trim_d_r, -np.radians(30), np.radians(30))

        else:
            err_Va = state.Va - cmd.airspeed_command

            chi_c = wrap(cmd.course_command, state.chi)
            err_chi = self.saturate(state.chi - chi_c, -np.radians(15), np.radians(15))

            try:
                x_lat = np.array([[err_Va * np.sin(state.beta)],
                                [state.p],
                                [state.r],
                                [state.phi],
                                [err_chi]])
            except Exception as e:
                print(e)

            # print("F_lat shape: ", self.F_lat.shape)
            # print("x_lat: ", x_lat)

            x_lat = x_lat.reshape((5, 1))
            temp = self.F_lat @ x_lat

            delta_a = self.saturate(temp.item(0) + 1*self.trim_d_a, -np.radians(30), np.radians(30))
            delta_r = self.saturate(temp.item(1) + 1*self.trim_d_r, -np.radians(30), np.radians(30))


        # Longitudinal Autopilot

        if (case != 1):
            x_lon = state.get_lon_state()
            temp = self.F_lon @ x_lon

            delta_e = self.saturate(1*temp.item(0) + 1*self.trim_d_e, -np.radians(30), np.radians(30))
            delta_t = self.saturate(1*temp.item(1) + 1*self.trim_d_t, 0., 1.)


        else:
            alt_c = self.saturate(cmd.altitude_command, state.altitude - 0.2*AP.altitude_zone, state.altitude + 0.2*AP.altitude_zone)
            err_alt = state.altitude - alt_c
            err_down = -err_alt
            
            try:
                x_lon = np.array([[err_Va * np.cos(state.alpha)], # u
                                [err_Va * np.sin(state.alpha)], # w
                                [state.q], # q
                                [state.theta], # theta
                                [err_down]]) # downward pos
            except Exception as e:
                print(e)
            
            # print("F_lon shape: ", self.F_lon.shape)
            # print("x_lon shape: ", x_lon.shape)

            x_lon = x_lon.reshape((5, 1))
            temp = self.F_lon @ x_lon

            delta_e = self.saturate(temp.item(0) + 1*self.trim_d_e, -np.radians(30), np.radians(30))
            delta_t = self.saturate(temp.item(1) + 1*self.trim_d_t, 0., 1.)

        
        
        # construct output and commanded states
        delta = Delta_State(d_e = delta_e,
                            d_a = delta_a,
                            d_r = delta_r,
                            d_t = delta_t)
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = 0 # phi_c
        self.commanded_state.theta = 0 # theta_c
        self.commanded_state.chi = cmd.course_command

        return delta, self.commanded_state


    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
