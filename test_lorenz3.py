from CSMIO import CSMIO
from DataGenerator import lorenz3
from Optimizers.csmio_optimizer import CSMIOOptimizer

# Experiment 1: Lorenz3 System
if __name__ == '__main__':
    # Generate training data
    x, y, snr = lorenz3.lorenz3generator(seed=40, noise_type=1, data_noise=300, end_time=60)
    print('SNR: %.4f' % snr)

    # Discover equations
    model = CSMIO(optimizer=CSMIOOptimizer(dimension=3, order=5, term_ks=[2, 3, 2]))
    model.fit(x, y)

    # show the trajectory of identified equations
    lorenz3.plot_trajectory(model)
