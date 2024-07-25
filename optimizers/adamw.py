"""
Implementation of AdamW optimizer.
"""

import numpy as np
from optimizers.optimizer import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer.

    Parameters:
        params (list): Model parameters.
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Weight decay rate.
    """

    def __init__(
            self,
            params: np.ndarray,
            learning_rate: float = 1e-3,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8,
            weight_decay: float = 1e-2
        ) -> None:
        super().__init__(params, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.state = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params}, \
            learning_rate={self.learning_rate}, beta1={self.beta1}, \
            beta2={self.beta2}, epsilon={self.epsilon}, \
            weight_decay={self.weight_decay})"

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
                "m": [np.zeros_like(param) for param in self.params],
                "v": [np.zeros_like(param) for param in self.params],
                "t": 0,
            }

        updated_params = []
        self.state["t"] += 1
        t = self.state["t"]

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            m = self.state["m"][i]
            v = self.state["v"][i]

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            param_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon) + self.learning_rate * self.weight_decay * param
            updated_param = param - param_update

            self.state["m"][i] = m
            self.state["v"][i] = v

            updated_params.append(updated_param)

        self.params = updated_params
        return updated_params
