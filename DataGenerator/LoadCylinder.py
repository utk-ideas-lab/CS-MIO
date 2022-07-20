import numpy as np
from scipy.integrate import odeint


def loadMatData():
    run_x = np.loadtxt('./DataGenerator/run_x.txt', delimiter=',')
    run_y = np.loadtxt('./DataGenerator/run_dx.txt', delimiter=',')

    simple = 1
    if simple == 0:
        run_x_1 = np.concatenate((run_x[1000:4000], run_x[5500:7500]), axis=0)
        run_y_1 = np.concatenate((run_y[1000:4000], run_y[5500:7500]), axis=0)
        return run_x_1, run_y_1
    else:
        return run_x, run_y


def simulate(initial_state, time_points):
    def cylinder_system(current_state, t):
        # positions of x, y, z in space at the current time point
        x, y, z = current_state

        # define the 3 ordinary differential equations known as the lorenz equations
        dx_dt = -0.00921683 * x - 1.02249789 * y + 0.00021198 * x * z - 0.00193227 * y * z
        dy_dt = 1.0346432 * x + 0.00463942 * y + 0.00218698 * x * z - 0.00175402 * y * z
        dz_dt = -21.90134429 - 0.3117679 * z + 0.00113786 * x * x + 0.00022177 * x * y + 0.00091433 * y * y - 0.00109604 * z * z

        # return a list of the equations that describe the system
        return [dx_dt, dy_dt, dz_dt]

    N = 3

    # use odeint() to solve a system of ordinary differential equations
    # the arguments are:
    # 1, a function - computes the derivatives
    # 2, a vector of initial system conditions (aka x, y, z positions in space)
    # 3, a sequence of time points to solve for
    # returns an array of x, y, and z value arrays for each time point, with the initial values in the first row
    xyz = odeint(cylinder_system, initial_state, time_points)

    return xyz
