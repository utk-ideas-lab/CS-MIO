from CSMIO import CSMIO
from DataGenerator import LoadCylinder
from Optimizers.csmio_optimizer import CSMIOOptimizer

# Experiment 3: Cylinder System
if __name__ == '__main__':
    # Generate training data
    x, y = LoadCylinder.loadMatData()

    # Discover equations
    # order = 2
    model = CSMIO(optimizer=CSMIOOptimizer(dimension=3, order=2, term_ks=[4, 4, 5], intercept=True))
    model.fit(x, y)

    # order = 3
    model = CSMIO(optimizer=CSMIOOptimizer(dimension=3, order=3, term_ks=[6, 5, 9], intercept=True))
    model.fit(x, y)
