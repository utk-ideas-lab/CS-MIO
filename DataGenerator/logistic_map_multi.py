import numpy as np


def logistic_map_multi(
        seed=40,
        data_noise=10
            ):
    # Generate training data
    np.random.seed(seed)

    N = 1000
    mus = [2.5, 2.75, 3, 3.25, 3.5, 3.75, 3.8, 3.85, 3.9, 3.95]
    eps = data_noise
    x = [np.zeros((N, 2)) for i in range(len(mus))]
    accurate_x = [np.zeros((N, 2)) for i in range(len(mus))]
    for i, mu in enumerate(mus):
        x[i][0] = [0.5, mu]
        for k in range(1, N):
            x[i][k, 0] = np.maximum(
                np.minimum(
                    mu * x[i][k - 1, 0] * (1 - x[i][k - 1, 0])
                    + eps * np.random.randn(),
                    1.0,
                ),
                0.0,
            )
            accurate_x[i][k, 0] = np.maximum(
                np.minimum(
                    mu * x[i][k - 1, 0] * (1 - x[i][k - 1, 0]),
                    1.0,
                ),
                0.0,
            )
            x[i][k, 1] = mu
    x_train = x

    x_train = np.asarray(x_train)
    accurate_x = np.asarray(accurate_x)

    train_data = []
    all_accurate_dx = []
    all_noise_dx = []
    for mu in range(x_train.shape[0]):
        dx_train_temp = x_train[mu][1:]
        x_train_temp = x_train[mu][0:-1]
        train_temp = np.concatenate((x_train_temp, dx_train_temp), axis=1)
        all_noise_dx.append(x_train[mu][1:, 0:1])
        all_accurate_dx.append(accurate_x[mu][1:, 0:1])
        train_data.append(train_temp)

    train_data = np.asarray(train_data)
    all_noise_dx = np.asarray(all_noise_dx)
    all_accurate_dx = np.asarray(all_accurate_dx)

    train_data = train_data.reshape(train_data.shape[0] * train_data.shape[1],
                                    train_data.shape[2])

    all_accurate = all_accurate_dx.reshape(all_accurate_dx.shape[0] * all_accurate_dx.shape[1],
                                           all_accurate_dx.shape[2])
    all_noise = all_noise_dx.reshape(all_noise_dx.shape[0] * all_noise_dx.shape[1],
                                     all_noise_dx.shape[2])
    snrs = [np.var(all_accurate[:, i]) / np.var(
        all_noise[:, i] - all_accurate[:, i]) for i in range(all_accurate.shape[1])]
    SNR = sum(snrs) / len(snrs)

    train_data = np.split(train_data, 2, axis=1)
    return train_data[0], train_data[1], SNR