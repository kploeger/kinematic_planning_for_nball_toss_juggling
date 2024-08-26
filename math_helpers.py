"""
    A few simple helper functions to reduce the number of dependencies.

    mail@kaiploeger.net
"""

import numpy as np
import casadi as cas


def Rx(theta):
    """ Rotation matrix around x-axis """
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def Ry(theta):
    """ Rotation matrix around y-axis """
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    """ Rotation matrix around z-axis """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotation_matrix_to_axis_angle(R):
    """ Convert a rotation matrix to a three angle rotation direction """
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))  # clip to avoid numerical errors
    if np.isclose(angle, 0):
        return np.zeros(3)
    axis = (1 / (2 * np.sin(angle))) * np.array([R[2, 1] - R[1, 2],
                                                 R[0, 2] - R[2, 0],
                                                 R[1, 0] - R[0, 1]])
    return angle * axis

def cas_cross_product(x, y):
    """ Cross product of two 3D casadi vectors """
    return cas.vertcat(x[1]*y[2] - x[2]*y[1],
                       x[2]*y[0] - x[0]*y[2],
                       x[0]*y[1] - x[1]*y[0])
