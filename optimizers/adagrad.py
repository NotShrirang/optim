"""
Implementation of the AdaGrad optimizer.
"""

import numpy as np
from optimizers.optimizer import Optimizer


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer.

    Parameters:
        params (list): Model parameters.
        learning_rate (float): Learning rate for the optimizer.
        epsilon (float): A small constant for numerical stability.
    """
    def __init__(
            self,
            params: np.ndarray,
            learning_rate: float = 0.02,
            epsilon: float = 1e-10
        ):
        super().__init__(params, learning_rate)
        self.epsilon = epsilon
        self.state = None

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.params}, \
            learning_rate={self.learning_rate}, epsilon={self.epsilon})"


    def step(self, grads: np.ndarray) -> np.ndarray:
        """
        Updates the model parameters based on the gradients.
        
        Args:
        grads (np.ndarray): Gradients of the model parameters.

        Returns:
        np.ndarray: Updated model parameters.
        """
        if self.state is None:
            self.state = [np.zeros_like(param) for param in self.params]

        updated_params = []
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.state[i] += grad ** 2
            param_update = self.learning_rate * grad / (np.sqrt(self.state[i]) + self.epsilon)
            updated_param = param - param_update
            updated_params.append(updated_param)

        self.params = updated_params
        return updated_params
