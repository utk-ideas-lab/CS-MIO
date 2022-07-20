from CSMIO import CSMIO
from Optimizers.csmio_optimizer import CSMIOOptimizer
from DataGenerator import logistic_map_multi

# Experiment 4: Logistic_map System
if __name__ == '__main__':
    # Generate training data
    x, y, snr = logistic_map_multi.logistic_map_multi(seed=40, data_noise=0.1)
    print('SNR: %.4f' % snr)

    # Discover equations
    model = CSMIO(optimizer=CSMIOOptimizer(dimension=2, order=5, term_ks=[2, 1]))
    model.fit(x, y)
