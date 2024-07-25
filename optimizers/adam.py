"""
Implementation of the Adam optimizer.
"""

import numpy as np
from optimizers.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.

    Parameters:
        params (list): Model parameters.
        learning_rate (float): Learning rate for the optimizer
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): A small constant for numerical stability.
    """

    def __init__(
            self,
            params: np.ndarray,
            learning_rate: float = 1e-3,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8
        ) -> None:
        super().__init__(params, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params}, \
            learning_rate={self.learning_rate}, beta1={self.beta1}, \
            beta2={self.beta2}, epsilon={self.epsilon})"

    def step(self, grads: np.ndarray) -> list:
        """
        Updates the model parameters based on the gradients.
        
        Args:
        grads (np.ndarray): Gradients of the model parameters.

        Returns:
        list: Updated model parameters.
        """
        if self.state is None:
            self.state = {
                "t": 0,
                "m": [np.zeros_like(param) for param in self.params],
                "v": [np.zeros_like(param) for param in self.params],
            }

        self.state["t"] += 1
        t = self.state["t"]
        updated_params = []
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.state["m"][i] = self.beta1 * self.state["m"][i] + (1 - self.beta1) * grad
            self.state["v"][i] = self.beta2 * self.state["v"][i] + (1 - self.beta2) * grad ** 2
            m_hat = self.state["m"][i] / (1 - self.beta1 ** t)
            v_hat = self.state["v"][i] / (1 - self.beta2 ** t)
            param_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_param = param - param_update
            updated_params.append(updated_param)

        self.params = updated_params
        return updated_params
