import numpy as np

def adagrad(params: np.ndarray, grads: np.ndarray, learning_rate: float = 0.01, epsilon: float = 1e-8, state: dict = None) -> np.ndarray:
    """
    Adagrad optimizer.
    
    Args:
    params (np.ndarray): List of parameter arrays to update.
    grads (np.ndarray): List of gradient arrays corresponding to the parameters.
    learning_rate (float): Learning rate for the optimizer.
    epsilon (float): Small constant to prevent division by zero.
    state (dict): Dictionary to store the state of the optimizer (cache).
    
    Returns:
    updated_params (np.ndarray): List of updated parameter arrays.
    state (dict): Updated state of the optimizer.
    """
    
    if state is None:
        state = {'cache': [np.zeros_like(p) for p in params]}
    
    cache = state['cache']
    updated_params = []

    for i in range(len(params)):
        cache[i] += grads[i] ** 2
        param_update = learning_rate * grads[i] / (np.sqrt(cache[i]) + epsilon)
        updated_param = params[i] - param_update
        updated_params.append(updated_param)
    
    state['cache'] = cache
    
    return updated_params, state