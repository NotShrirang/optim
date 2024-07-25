from optimizers.functional import gradient_descent, adagrad, rmsprop, adam, adamw
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 1)
y = (3 * X) + 4

weights = np.random.randn(1, 1)
bias = np.zeros((1, 1))

params = [weights, bias]

def linear_model(X, params):
    weights, bias = params
    return np.dot(X, weights) + bias

def compute_loss(y, y_pred):
    return np.mean((y_pred - y) ** 2)

def compute_gradients(X, y, y_pred):
    m = X.shape[0]
    error = y_pred - y
    dw = (2 / m) * np.dot(X.T, error)
    db = (2 / m) * np.sum(error)
    return [dw, db]

learning_rate = 0.005
epochs = 2500
state = None

for epoch in range(epochs):
    y_pred = linear_model(X, params)
    
    loss = compute_loss(y, y_pred)
    
    grads = compute_gradients(X, y, y_pred)
    
    # params = gradient_descent(params, grads, learning_rate=learning_rate)
    # params, state = adagrad(params, grads, learning_rate=learning_rate, state=state)
    # params, state = rmsprop(params, grads, learning_rate=learning_rate, state=state, beta=0.9, epsilon=1e-8)
    # params, state = adam(params, grads, learning_rate=learning_rate, state=state, beta1=0.9, beta2=0.999, epsilon=1e-8, t=epoch+1)
    # params, state = adamw(params, grads, learning_rate=learning_rate, state=state, beta1=0.9, beta2=0.999, epsilon=1e-8, t=epoch+1, weight_decay=0.01)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss}")


weights, bias = params
print("Trained weights:", weights)
print("Trained bias:", bias)

X_new = np.array([[0], [2]])
y_pred = linear_model(X_new, params)
print("Predictions:", y_pred)