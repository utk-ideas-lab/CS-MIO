import numpy as np
from scipy.integrate import odeint
from Util import tvregdiff
import matplotlib.pyplot as plt


def lorenz3generator(
        seed=40,
        data_noise=10,
        end_time=60,
        noise_type=1):
    '''
    Parameters
    ----------
    seed: random seed of generator
    data_noise: Gaussian noise of the data
    end_time: End time of lorenz: end_time * 1000
    noise_type: noise type of the data 1: noise data with noise added to accurate dx/dt
                                       2: noise data with noise added to x, then use total variation differentiation

    Returns
    -------
    x: feature
    y: response
    SNR: signal-to-noise ratio
    '''

    # define the lorenz system
    # x, y, and z make up the system state, t is time, and sigma, rho, beta are the system parameters
    def lorenz_system(current_state, t):
        # positions of x, y, z in space at the current time point
        x, y, z = current_state

        # define the 3 ordinary differential equations known as the lorenz equations
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z

        # return a list of the equations that describe the system
        return [dx_dt, dy_dt, dz_dt]

    # define the system parameters sigma, rho, and beta
    sigma = 10.
    rho = 28.
    beta = 8. / 3.
    N = 3

    # define the time points to solve for, evenly spaced between the start and end times
    start_time = 0
    time_points = np.linspace(start_time, end_time, int(end_time * 1000))
    nt = int(end_time * 1000)
    # dt = (end_time - start_time) / int(end_time * 1000 - 1)
    dt = 0.001

    # define the initial system state (aka x, y, z positions in space)
    initial_state = [-8.0, 8.0, 27.0]

    # noise scale
    eps = data_noise

    # use odeint() to solve a system of ordinary differential equations
    # the arguments are:
    # 1: a function - computes the derivatives
    # 2: a vector of initial system conditions (aka x, y, z positions in space)
    # 3: a sequence of time points to solve for
    # returns an array of x, y, and z value arrays for each time point, with the initial values in the first row
    xyz = odeint(lorenz_system, initial_state, time_points)

    # seed used to add noise into the accurate differential
    np.random.seed(seed)

    accurate_output = np.array(
        [lorenz_system(xyz[i], time_points) for i in range(nt - 1)]
    )

    # noise data with noise added to dxdt
    if noise_type == 1:
        print('data with noise added to dxdt')
        accurate_output = np.array(
            [lorenz_system(xyz[i], time_points) for i in range(nt - 1)]
        )

        noise = eps * np.random.normal(0, 1, accurate_output.shape)
        noise_output = accurate_output + noise

        # calculate the SNR for each dx
        snrs = [np.var(accurate_output[:, i]) / np.var(noise[:, i]) for i in range(accurate_output.shape[1])]
        SNR = sum(snrs) / len(snrs)

        return xyz[:-1], noise_output, SNR

    # total variation differentiation
    elif noise_type == 2:
        print('noise data with noise added to x, then use total variation differentiation')
        noise = eps * np.random.normal(0, 1, xyz.shape)
        xyz = xyz + noise
        total_variation_diff = []

        for i in range(N):
            # init =np.concatenate((np.diff(xyz[:, i]), [0]), axis=0)
            total_variation_diff.append(tvregdiff.TVRegDiff(data=xyz[:, i], itern=10, alph=0.00002,
                                                            scale='small', diffkernel='sq', dx=dt,
                                                            ep=1e12, plotflag=False, diagflag=False))
        total_variation_diff = np.asarray(total_variation_diff).T

        xt = np.asarray([np.cumsum(total_variation_diff[:, i]) * dt for i in range(N)]).T
        xt = np.asarray([xt[:, i] - (np.mean(xt[1000: -1000, i]) - np.mean(xyz[1000: -1000, i])) for i in range(N)]).T
        xt = np.asarray(xt[1000:-1000])
        total_variation_diff = total_variation_diff[1000:-1000]

        snrs = [np.var(xyz[:, i]) / np.var(noise[:, i]) for i in range(N)]
        SNR=sum(snrs)/len(snrs)

        return xt, total_variation_diff, SNR

    else:
        print('default: accurate data')
        accurate_output = np.array(
            [lorenz_system(xyz[i], time_points) for i in range(nt - 1)]
        )
        return xyz[:-1], accurate_output


def plot_trajectory(model):
    xyz0 = simulate_originalSystem()

    plt.figure(figsize=(8, 8))
    plt.rc('xtick', labelsize=18)
    plt.rc('legend', fontsize=25)

    xyz = simulate_learnedSystem(model)

    # plot the lorenz attractor in three-dimensional phase space
    ax = plt.axes(projection='3d')
    ax.grid(False)

    ax.plot(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2], color='green', alpha=0.6, linewidth=2, label='Ground Truth')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '--', color='red', alpha=0.6, linewidth=2,
            label='CS-MIO Discovered System')
    ax.set_xlabel('$x$', fontsize=34)
    ax.set_ylabel('$y$', fontsize=34)
    ax.set_zlabel('$z$', fontsize=34)
    ax.legend(loc='upper right')

    plt.show()


def simulate_learnedSystem(model):
    # define the lorenz system
    # x, y, and z make up the system state, t is time, and sigma, rho, beta are the system parameters
    def lorenz_system(current_state, t):
        # positions of x, y, z in space at the current time point
        x, y, z = current_state

        coefficients = model.get_coefficients()
        features = model.get_features()
        terms = []
        for i in [0, 1, 2]:
            values = []
            for feature in features[i]:
                value = 1
                for term in feature:
                    if term == 0:
                        value = x * value
                    elif term == 1:
                        value = value * y
                    else:
                        value = value * z
                values.append(value)

            terms.append(values)

        dx_dt = np.array(coefficients[0][1:]).dot(np.array(terms[0])) + coefficients[0][0]
        dy_dt = np.array(coefficients[1][1:]).dot(np.array(terms[1])) + coefficients[1][0]
        dz_dt = np.array(coefficients[2][1:]).dot(np.array(terms[2])) + coefficients[2][0]

        # return a list of the equations that describe the system
        return [dx_dt, dy_dt, dz_dt]

    dt = 0.001
    start_time = 0
    end_time = 20
    time_points = np.linspace(start_time, end_time, int(end_time * 1 / dt))
    nt = int(end_time * 1 / dt)

    # define the initial system state (aka x, y, z positions in space)
    initial_state = [-8.0, 8, 4]
    xyz = odeint(lorenz_system, initial_state, time_points)

    return xyz


def simulate_originalSystem():
    # define the lorenz system
    # x, y, and z make up the system state, t is time, and sigma, rho, beta are the system parameters
    def lorenz_system(current_state, t):
        # positions of x, y, z in space at the current time point
        x, y, z = current_state

        dx_dt = 10 * (y - x)
        dy_dt = x * (28 - z) - y
        dz_dt = x * y - 8 / 3 * z

        # return a list of the equations that describe the system
        return [dx_dt, dy_dt, dz_dt]

    dt = 0.001
    start_time = 0
    end_time = 20
    time_points = np.linspace(start_time, end_time, int(end_time * 1 / dt))
    nt = int(end_time * 1 / dt)
    # dt = (end_time - start_time) / int(end_time * 1000 - 1)

    # define the initial system state (aka x, y, z positions in space)
    initial_state = [-8.0, 8, 4]
    xyz = odeint(lorenz_system, initial_state, time_points)

    return xyz
