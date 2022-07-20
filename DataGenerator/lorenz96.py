from scipy.integrate import odeint
import numpy as np
from Util import tvregdiff
import matplotlib.pyplot as plt


# datatype: 1:accurate data, 2:numerical differential, 3: noise data
def lorenz96generator(
        seed=40,
        dimension=96,
        data_noise=10,
        end_time=60,
        noise_type=1
):
    '''

    Parameters
    ----------
    dimension
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

    def Lorenz96(x, t):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    # These are our constants
    N = dimension  # Number of variables
    F = 8  # Forcing

    x0 = np.ones(N)  # Initial state (equilibrium)
    x0[0] += 0.01  # Add small perturbation to the first variable

    # define the time points to solve for, evenly spaced between the start and end times
    start_time = 0
    end_time = end_time * 10
    time_points = np.linspace(start_time, end_time, int(end_time * 100))
    nt = int(end_time * 100)
    dt = (end_time - start_time) / int(end_time * 100 - 1)

    # noise scale
    eps = data_noise

    X = odeint(Lorenz96, x0, time_points)

    accurate_output = np.array(
        [Lorenz96(X[i], time_points) for i in range(nt - 1)]
    )

    # seed used to add noise into the accurate differential
    np.random.seed(seed)

    # noise data with noise added to dxdt
    if noise_type == 1:
        print('data with noise added to dx/dt')
        accurate_output = np.array([Lorenz96(X[i], time_points) for i in range(nt - 1)]
                                   )

        noise = eps * np.random.normal(0, 1, accurate_output.shape)
        noise_output = accurate_output + noise

        # calculate the SNR for each dx
        snrs = [np.var(accurate_output[:, i]) / np.var(noise[:, i]) for i in range(accurate_output.shape[1])]
        SNR = sum(snrs) / len(snrs)

        return X[:-1], noise_output, SNR
    elif noise_type == 2:
        print('noise data with noise added to x, then use total variation differentiation')
        noise = eps * np.random.normal(0, 1, X.shape)
        xyz = X + noise
        total_variation_diff = []

        for i in range(N):
            total_variation_diff.append(tvregdiff.TVRegDiff(data=xyz[:, i], itern=10, alph=0.00002,
                                                            scale='small', diffkernel='sq', dx=dt,
                                                            ep=1e12, plotflag=False, diagflag=False))
        total_variation_diff = np.asarray(total_variation_diff).T

        xt = np.asarray([np.cumsum(total_variation_diff[:, i]) * dt for i in range(N)]).T
        xt = np.asarray([xt[:, i] - (np.mean(xt[1000: -1000, i]) - np.mean(xyz[1000: -1000, i])) for i in range(N)]).T
        xt = np.asarray(xt[1000:-1000])
        total_variation_diff = total_variation_diff[1000:-1000]

        # calculate the SNR for each x
        snrs = [np.var(X[:, i]) / np.var(noise[:, i]) for i in range(N)]
        SNR = sum(snrs) / len(snrs)

        return xt, total_variation_diff, SNR
    # default: accurate data is returned
    else:
        print('default: accurate data')
        accurate_output = np.array(
            [Lorenz96(X[i], time_points) for i in range(nt - 1)]
        )

        return X[:-1], accurate_output


def plot_trajectory(model, dimension):
    x0 = simulate_originalSystem(dimension)
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    fig, axs = plt.subplots(1, 1, figsize=(10, 8))

    x = simulate_learnedSystem(model, dimension)
    levels = np.linspace(x.min(), x.max(), 6)

    axs.grid(False)
    delta_x = x - x0

    cs = axs.contourf(delta_x.T, cmap='bwr', levels=levels)
    fig.colorbar(cs, ax=axs)
    axs.set_ylabel('Index of State Variable', fontsize=20)

    plt.xlabel('Time t (unit: 0.01 sec)', fontsize=20)
    plt.show()


def simulate_originalSystem(dimension):
    N = dimension

    def lorenz96_system(current_state, t):
        # positions of x, y, z in space at the current time point
        x = current_state
        F = 8  # Forcing
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F

        return d

    dt = 0.01
    start_time = 0
    end_time = 10
    time_points = np.linspace(start_time, end_time, int(end_time * 1 / dt))
    nt = int(end_time * 1 / dt)
    # dt = (end_time - start_time) / int(end_time * 1000 - 1)

    # define the initial system state (aka x, y, z positions in space)
    x0 = np.ones(N)  # Initial state (equilibrium)
    x0[0] += 0.01  # Add small perturbation to the first variable
    x = odeint(lorenz96_system, x0, time_points)

    return x


def simulate_learnedSystem(model, dimension):
    N = dimension

    def lorenz96_system(current_state, t):
        x = current_state
        d = np.zeros(N)

        coefficients = model.get_coefficients()
        features = model.get_features()

        for i in range(N):
            terms = []
            for feature in features[i]:
                value = 1
                for term in feature:
                    value = value * x[term]

                terms.append(value)
            d[i] = np.array(coefficients[i][1:]).dot(np.array(terms)) + coefficients[i][0]

        return d

    dt = 0.01
    start_time = 0
    end_time = 10
    time_points = np.linspace(start_time, end_time, int(end_time * 1 / dt))
    nt = int(end_time * 1 / dt)
    # dt = (end_time - start_time) / int(end_time * 1000 - 1)

    # define the initial system state (aka x, y, z positions in space)
    x0 = np.ones(N)  # Initial state (equilibrium)
    x0[0] += 0.01  # Add small perturbation to the first variable
    x = odeint(lorenz96_system, x0, time_points)

    return x