from CSMIO import CSMIO
from DataGenerator import lorenz96
from Optimizers.csmio_optimizer import CSMIOOptimizer

# Experiment 2: Lorenz96 System
if __name__ == '__main__':
    dimension = 96
    # Generate training data
    x, y, snr = lorenz96.lorenz96generator(seed=40, dimension=dimension, noise_type=1, end_time=60, data_noise=50)
    print('SNR: %.4f' % snr)

    # Discover equations
    model = CSMIO(optimizer=CSMIOOptimizer(dimension=dimension, order=2, term_ks=[4]))
    model.fit(x, y)

    lorenz96.plot_trajectory(model, dimension=dimension)
