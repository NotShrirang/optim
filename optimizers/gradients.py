"""
Implementation of the Gradient Descent optimizer.
"""

import numpy as np
from optimizers.optimizer import Optimizer


class GradientDescent(Optimizer):
    """
    Gradient Descent optimizer.

    Parameters:
        params (list): Model parameters.
        learning_rate (float): Learning rate for the optimizer
    """

    def __init__(
            self,
            params: np.ndarray,
            learning_rate: float = 1e-3
        ) -> None:
        super().__init__(params, learning_rate)

    def step(self, grads: np.ndarray) -> list:
        """
        Updates the model parameters based on the gradients.

        Args:
        grads (np.ndarray): Gradients of the model parameters.

        Returns:
        list: Updated model parameters.
        """
        updated_params = []
        for param, grad in zip(self.params, grads):
            param_update = self.learning_rate * grad
            updated_param = param - param_update
            updated_params.append(updated_param)

        self.params = updated_params
        return updated_params
