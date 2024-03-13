

from ._base_optimizer import _BaseOptimizer


class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum
        self.velocity = {}

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                if idx not in self.velocity:
                    self.velocity[idx] = {"weight": 0, "bias": 0}

                self.velocity[idx]["weight"] = self.momentum * self.velocity[idx]["weight"] - self.learning_rate * m.dw
                m.weight += self.velocity[idx]["weight"]

            if hasattr(m, 'bias'):
                self.velocity[idx]["bias"] = self.momentum * self.velocity[idx]["bias"] - self.learning_rate * m.db
                m.bias += self.velocity[idx]["bias"]

