from Optimizers.csmio_optimizer import CSMIOOptimizer


class CSMIO:

    def __init__(self, optimizer=None):
        if optimizer is None:
            self.optimizer = CSMIOOptimizer()
        self.optimizer = optimizer

    def fit(self, x, y):
        self.optimizer.fit(x, y)

    def get_coefficients(self):
        return self.optimizer.coefficients

    def get_features(self):
        return self.optimizer.features
