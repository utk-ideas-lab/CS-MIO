import numpy as np
from CSMIO import CSMIO
from Optimizers.csmio_optimizer import CSMIOOptimizer
from DataGenerator import hopf_normal_form_multi

# Experiment 5: Hopf_normal_form System
if __name__ == '__main__':
    # Generate training data
    x, y, snr = hopf_normal_form_multi.hopf_normal_form_multi(seed=40, noise_type=1, data_noise=1, dimension=3)
    print('SNR: %.4f' % snr)

    # Discover equations
    model = CSMIO(optimizer=CSMIOOptimizer(dimension=3, order=5, term_ks=[4, 4, 1]))
    model.fit(x, y)
