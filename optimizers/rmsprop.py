"""
Implementation of RMSProp optimizer.
"""

import numpy as np
from optimizers.optimizer import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer.

    Parameters:
        params (list): Model parameters.
        learning_rate (float): Learning rate for the optimizer.
        rho (float): Exponential decay rate.
        epsilon (float): Term added to the denominator to improve numerical stability.
    """

    def __init__(
            self,
            params: np.ndarray,
            learning_rate: float = 0.01,
            beta: float = 0.99,
            epsilon: float = 1e-8
        ) -> None:
        super().__init__(params, learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.state = None

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.params}, \
            learning_rate={self.learning_rate}, beta={self.beta}, epsilon={self.epsilon})"

    def step(self, grads: np.ndarray) -> list:
        """
        Updates the model parameters based on the gradients.

        Args:
        grads (np.ndarray): Gradients of the model parameters.

        Returns:
        list: Updated model parameters.
        """
        if self.state is None:
            self.state = {'cache': [np.zeros_like(p) for p in self.params]}

        cache = self.state['cache']
        updated_params = []

        for i, param in enumerate(self.params):
            cache[i] = self.beta * cache[i] + (1 - self.beta) * grads[i] ** 2
            param_update = self.learning_rate * grads[i] / (np.sqrt(cache[i]) + self.epsilon)
            updated_param = param - param_update
            updated_params.append(updated_param)

        self.state['cache'] = cache
        self.params = updated_params
        return updated_params
