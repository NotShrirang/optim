"""
Functional implementation of Gradient Descent optimizer.
"""

import numpy as np

def gradient_descent(params: np.ndarray, grads: np.ndarray, learning_rate: float = 0.01) -> list:
    """
    Simple Gradient Descent optimizer.
    
    Args:
    params (np.ndarray): List of parameter arrays to update.
    grads (np.ndarray): List of gradient arrays corresponding to the parameters.
    learning_rate (float): Learning rate for the optimizer.
    
    Returns:
    updated_params (list): List of updated parameter arrays.
    """
    
    updated_params = []
    for param, grad in zip(params, grads):
        updated_param = param - learning_rate * grad
        updated_params.append(updated_param)
    return updated_params