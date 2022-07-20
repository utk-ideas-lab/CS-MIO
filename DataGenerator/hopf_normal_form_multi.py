import numpy as np
from Util import tvregdiff
from scipy.integrate import odeint


def hopf_normal_form_multi(
        seed=40,
        data_noise=10,
        noise_type=1,
        dimension=3):
    # Generate training data
    def hopf(x, mu, omega, A):
        return [
            mu * x[0] - omega * x[1] - A * x[0] * (x[0] ** 2 + x[1] ** 2),
            omega * x[0] + mu * x[1] - A * x[1] * (x[0] ** 2 + x[1] ** 2),
        ]

    np.random.seed(seed)

    omega = 1
    A = 1
    dt = 0.0025
    t_train = np.arange(0, 75, dt)
    mu_stable = np.array([-0.15, -0.05])
    mu_unstable = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55])
    n_ics = mu_stable.size + 2 * mu_unstable.size
    x_train = [np.zeros((t_train.size, 3)) for i in range(n_ics)]
    x_accurate = [np.zeros((t_train.size, 3)) for i in range(n_ics)]

    noise_type = noise_type
    eps = 0
    eps1 = 0
    if noise_type == 2:
        eps = data_noise
        eps1 = 0
    else:
        eps = 0
        eps1 = data_noise

    ic_idx = 0

    for mu in mu_stable:
        x0_train = [2, 0]
        x = odeint(lambda x, t: hopf(x, mu, omega, A), x0_train, t_train)
        x_train[ic_idx][:, 0:2] = x + eps * np.random.normal(size=x.shape)
        x_train[ic_idx][:, 2] = mu
        x_accurate[ic_idx][:, 0:2] = x
        x_accurate[ic_idx][:, 2] = mu
        ic_idx += 1

    for mu in mu_unstable:
        x0_train = [0.01, 0]
        x = odeint(lambda x, t: hopf(x, mu, omega, A), x0_train, t_train)
        x_train[ic_idx][:, 0:2] = x + eps * np.random.normal(size=x.shape)
        x_train[ic_idx][:, 2] = mu
        x_accurate[ic_idx][:, 0:2] = x
        x_accurate[ic_idx][:, 2] = mu
        ic_idx += 1

        x0_train = [2, 0]
        x = odeint(lambda x, t: hopf(x, mu, omega, A), x0_train, t_train)
        x_train[ic_idx][:, 0:2] = x + eps * np.random.normal(size=x.shape)
        x_train[ic_idx][:, 2] = mu
        x_accurate[ic_idx][:, 0:2] = x
        x_accurate[ic_idx][:, 2] = mu
        ic_idx += 1
        # plt.figure()
        # plt.plot(x_train[ic_idx-1][:, 0], x_train[ic_idx-1][:, 1])
        # plt.show()

    x_train = np.asarray(x_train)
    x_accurate = np.asarray(x_accurate)
    x_noise = x_train - x_accurate

    if noise_type == 1:
        all_noise_data = []
        all_noise = []
        all_accurate = []
        for mu in range(x_train.shape[0]):
            accurate_output = np.array(
                [hopf([x_train[mu, i, 0], x_train[mu, i, 1]], x_train[mu, i, 2], omega, A) for i in
                 range(x_train.shape[1])])

            noise = eps1 * np.random.normal(0, 1, accurate_output.shape)
            noise_output = accurate_output + noise

            # SNR
            all_noise.append(noise)
            all_accurate.append(accurate_output)

            zeros_diff = eps1 * np.random.normal(0, 1, (noise_output.shape[0], 1))
            noise_output = np.concatenate((noise_output, zeros_diff), axis=1)
            noise_data = np.concatenate((x_train[mu, :], noise_output), axis=1)

            all_noise_data.append(noise_data)

        all_noise_data = np.asarray(all_noise_data)
        all_noise = np.asarray(all_noise)
        all_accurate = np.asarray(all_accurate)

        all_noise_data = all_noise_data.reshape(all_noise_data.shape[0] * all_noise_data.shape[1],
                                                all_noise_data.shape[2])

        all_noise = all_noise.reshape(all_noise.shape[0] * all_noise.shape[1],
                                      all_noise.shape[2])
        all_accurate = all_accurate.reshape(all_accurate.shape[0] * all_accurate.shape[1],
                                            all_accurate.shape[2])
        snrs = [np.var(all_accurate[:, i]) / np.var(all_noise[:, i]) for i in range(all_noise.shape[1])]
        SNR = np.average(snrs)

        data = np.split(all_noise_data, 2, axis=1)

        return data[0], data[1], SNR

    elif noise_type == 2:
        print('noise data with noise added to x, then use total variation differentiation')

        tvd_data = []
        all_accurate = []
        all_noise = []

        for mu in range(x_train.shape[0]):
            tvd_diff = []
            for i in range(dimension):
                tvd_diff.append(
                    tvregdiff.TVRegDiff(data=x_train[mu, :, i], itern=5, alph=2, scale='small', diffkernel='sq', dx=dt,
                                        ep=1e2, plotflag=False, diagflag=False))

            tvd_diff = np.asarray(tvd_diff).T[1000:-500]
            tvd_diff = np.concatenate((x_train[mu, 1000:-500], tvd_diff), axis=1)
            tvd_data.append(tvd_diff)

            all_accurate.append(x_accurate[mu, 1000:-500])
            all_noise.append(x_noise[mu, 1000:-500])

        tvd_data = np.asarray(tvd_data)
        all_accurate = np.asarray(all_accurate)
        all_noise = np.asarray(all_noise)

        tvd_data = tvd_data.reshape(tvd_data.shape[0] * tvd_data.shape[1],
                                    tvd_data.shape[2])

        all_accurate = all_accurate.reshape(all_accurate.shape[0] * all_accurate.shape[1], all_accurate.shape[2])
        all_noise = all_noise.reshape(all_noise.shape[0] * all_noise.shape[1], all_noise.shape[2])

        snrs = [np.var(all_accurate[:, i]) / np.var(all_noise[:, i]) for i in range(all_accurate.shape[1] - 1)]
        SNR = np.average(snrs)

        data = np.split(tvd_data, 2, axis=1)
        return data[0], data[1], SNR

    return np.asarray(x_train)
