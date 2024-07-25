"""
This module contains the base class for optimizers.
"""

import numpy as np


class Optimizer:
    """
    Base class for optimizers.

    Not meant to be used directly. Subclasses should implement the `step` method.
    """
    def __init__(self, params, learning_rate) -> None:
        self.params = params
        self.learning_rate = learning_rate
        self.state = None

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.params}, \
            learning_rate={self.learning_rate})"

    def step(self, grads: np.ndarray) -> np.ndarray:
        """
        Updates the model parameters based on the gradients.
        
        Args:
        grads (np.ndarray): Gradients of the model parameters.
        
        Returns:
        np.ndarray: Updated model parameters.
        """
        raise NotImplementedError
