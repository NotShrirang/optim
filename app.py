"""
A simple Streamlit app to visualize different optimizers.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from optimizers import adagrad, adam, adamw, rmsprop, gradients

st.set_page_config(
    page_title="Optimizer Visualizer",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

st.title("Optimizer Visualizer")

st.sidebar.title("Settings")

st.write("Generate random data points for linear regression.")

c1, c2 = st.columns(2)
slope = c1.number_input("Slope", -10.0, 10.0, 3.0, 1.0)
intercept = c2.number_input("Intercept", -10.0, 10.0, 2.0, 1.0)

st.sidebar.subheader("Optimizer Settings")

optimizer_type = st.sidebar.selectbox("Optimizer", ["SGD", "Adam", "RMSprop", "Adagrad", "AdamW"])

learning_rate = st.sidebar.number_input("Learning rate", 0.0001, 1.0, 0.01, 0.0001, format="%.4f")

if optimizer_type != "SGD":
    if optimizer_type == "RMSprop":
        beta = st.sidebar.number_input(
                    "Beta",
                    min_value=0.1,
                    max_value=0.999,
                    value=0.9,
                    step=0.01,
                    format="%.4f"
                )

    if optimizer_type in ["Adam", "AdamW"]:
        beta1 = st.sidebar.number_input(
                    "Beta 1",
                    min_value=0.1,
                    max_value=0.999,
                    value=0.9,
                    step=0.01,
                    format="%.4f"
                )
        beta2 = st.sidebar.number_input(
                    "Beta 2",
                    min_value=0.001,
                    max_value=0.999,
                    value=0.999,
                    step=0.001,
                    format="%.4f"
                )

    epsilon = st.sidebar.number_input(
                    "Epsilon",
                    min_value=1e-9,
                    max_value=0.1,
                    value=1e-8,
                    step=1e-9,
                    format="%.9f"
                )

    if optimizer_type == "AdamW":
        weight_decay = st.sidebar.number_input(
                    "Weight Decay",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.01,
                    step=0.0001,
                    format="%.4f"
                )

number_of_data_points = st.number_input("Number of data points", 0, 100000, 500, 1)

X = np.random.randn(number_of_data_points, 1)
Y = intercept + (slope * X) + np.random.uniform(-2, 2, (number_of_data_points, 1)) * 0.5

weights = np.random.rand(1, 1)
bias = np.random.rand(1, 1)

params = [weights, bias]

train_losses = []
test_losses = []

train_x = X[:int(number_of_data_points*0.8)]
train_y = Y[:int(number_of_data_points*0.8)]

test_x = X[int(number_of_data_points*0.8):]
test_y = Y[int(number_of_data_points*0.8):]

st.write("X shape:", X.shape)
st.write("Y shape:", Y.shape)

loss_function = st.sidebar.selectbox("Loss function",
                    ["Mean Squared Error", "Root Mean Squared Error", "Mean Absolute Error"])

steps = st.sidebar.number_input("Number of steps", 1, 10000, 200, 1)

plot_scatter = st.checkbox("Show Data")

def scatter(x, y, title, x_label="X", y_label="Y"):
    """
    Scatter plot.
    """
    figure, axes = plt.subplots()
    axes.scatter(x, y)
    axes.grid(linestyle="--", alpha=0.7)
    axes.set_xticks(np.arange(np.round(y.min(), 0), int(y.max())+1, 1), minor=True)
    axes.set_yticks(np.arange(int(y.min())-1, int(y.max())+1, 1), minor=True)
    if slope == 0:
        axes.set_yticks(np.arange(int(y.min())-10, int(y.max())+10, 1), minor=True)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    return figure, axes

if plot_scatter:
    fig, ax = scatter(X, Y, f"Line: y={slope:.0f}x + {intercept:.0f}", "X", "Y")
    st.pyplot(fig, clear_figure=True, use_container_width=True)

if optimizer_type == "SGD":
    optimizer = gradients.GradientDescent(params, learning_rate)
elif optimizer_type == "Adam":
    optimizer = adam.Adam(params, learning_rate, beta1, beta2, epsilon)
elif optimizer_type == "RMSprop":
    optimizer = rmsprop.RMSProp(params, learning_rate, beta, epsilon)
elif optimizer_type == "Adagrad":
    optimizer = adagrad.AdaGrad(params, learning_rate, epsilon)
elif optimizer_type == "AdamW":
    optimizer = adamw.AdamW(params, learning_rate, beta1, beta2, epsilon, weight_decay)
else:
    optimizer = gradients.GradientDescent(params, learning_rate)

st.subheader("Training")

def linear_model(x, current_params):
    """
    Linear model.
    """
    w, b = current_params
    return np.dot(x, w) + b

def mse_loss(y, y_pred):
    """
    Mean Squared Error loss function.
    """
    return np.mean((y_pred - y) ** 2)

def mae_loss(y, y_pred):
    """
    Mean Absolute Error loss function.
    """
    return np.mean(np.abs(y_pred - y))

def rmse_loss(y, y_pred):
    """
    Root Mean Squared Error loss function.
    """
    return np.sqrt(np.mean((y_pred - y) ** 2))

def compute_gradients(x, y_true, y_pred):
    """
    Computes gradients.
    """
    m = x.shape[0]
    error = y_pred - y_true
    if loss_function in ["Mean Squared Error", "Root Mean Squared Error"]:
        dw = (2 / m) * np.dot(x.T, error)
        db = (2 / m) * np.sum(error)
    if loss_function == "Mean Absolute Error":
        dw = (1 / m) * np.dot(x.T, np.sign(error))
        db = (1 / m) * np.sum(np.sign(error))
    return [dw, db]

if loss_function == "Mean Squared Error":
    calculate_loss = mse_loss
elif loss_function == "Mean Absolute Error":
    calculate_loss = mae_loss
elif loss_function == "Root Mean Squared Error":
    calculate_loss = rmse_loss
else:
    calculate_loss = mse_loss

if "train_losses" not in st.session_state:
    st.session_state["train_losses"] = []
    st.session_state["test_losses"] = []

if "show_training_progress" not in st.session_state:
    st.session_state["show_training_progress"] = True

if "show_trained_params" not in st.session_state:
    st.session_state["show_trained_params"] = True

if "show_loss_curve" not in st.session_state:
    st.session_state["show_loss_curve"] = False

if "show_model_predictions" not in st.session_state:
    st.session_state["show_model_predictions"] = False

def update_checkbox_state(checkbox):
    st.session_state[checkbox] = not st.session_state[checkbox]

c1, c2, c3, c4 = st.columns(4)
show_training_progress = c1.checkbox("Show training progress", value=st.session_state.get("show_training_progress"), on_change=lambda: update_checkbox_state("show_training_progress"))
show_trained_params = c2.checkbox("Show trained parameters", value=st.session_state.get("show_trained_params"), on_change=lambda: update_checkbox_state("show_trained_params"))
show_loss_curve = c3.checkbox("Show loss curve", value=st.session_state.get("show_loss_curve"), on_change=lambda: update_checkbox_state("show_loss_curve"))
show_model_predictions = c4.checkbox("Show model predictions", value=st.session_state.get("show_model_predictions"), on_change=lambda: update_checkbox_state("show_model_predictions"))

if st.button("Train"):
    for epoch in range(steps):
        predictions = linear_model(train_x, params)
        train_loss = calculate_loss(train_y, predictions)

        test_predictions = linear_model(test_x, params)
        test_loss = calculate_loss(test_y, test_predictions)

        grads = compute_gradients(train_x, train_y, predictions)
        params = optimizer.step(grads)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if not st.session_state["show_training_progress"]:
            continue

        if steps <= 20:
            st.write(f"Epoch {epoch + 1}, Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
        elif steps <= 100:
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch + 1}, Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
        elif steps <= 1000:
            if (epoch + 1) % 100 == 0:
                st.write(f"Epoch {epoch + 1}, Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
            continue
        elif steps <= 10000:
            if (epoch + 1) % 1000 == 0:
                st.write(f"Epoch {epoch + 1}, Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
            continue
        else:
            if (epoch + 1) % 10000 == 0:
                st.write(f"Epoch {epoch + 1}, Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
            continue

    if st.session_state["show_trained_params"]:
        st.write("Trained weights:", params[0].flatten()[0])
        st.write("Trained bias:", params[1].flatten()[0])

    if st.session_state["show_loss_curve"]:
        st.subheader("Loss Curves")
        fig, ax = plt.subplots()
        ax.plot(train_losses, label="Train Loss")
        ax.plot(test_losses, label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{optimizer_type} Optimizer: Loss vs Epoch")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.7)

        st.pyplot(fig, clear_figure=True, use_container_width=True)

    if st.session_state["show_model_predictions"]:
        st.subheader("Model Predictions")
        fig, ax = plt.subplots()
        ax.scatter(train_x, train_y, label="Train Data Points", c="blue")
        ax.scatter(test_x, test_y, label="Test Data Points", c="orange")
        random_x = np.random.uniform(np.round(X.min(), 0)-1, int(X.max())+1, (number_of_data_points, 1))
        ax.plot(random_x, linear_model(random_x, params), label=f"Predicted line: y = {params[0].flatten()[0]:.4f}x + {params[1].flatten()[0]:.4f}", c="red")
        ax.plot(X, ((slope * X) + intercept), label=f"Ideal Line: y = {slope:.0f}x + {intercept:.0f}", c="green")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xticks(np.arange(np.round(Y.min(), 0), int(Y.max())+1, 1), minor=True)
        ax.set_yticks(np.arange(int(Y.min())-1, int(Y.max())+1, 1), minor=True)
        if slope == 0:
            ax.set_yticks(np.arange(int(Y.min())-10, int(Y.max())+10, 1), minor=True)
        ax.set_title("Regression Plot")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.7)

        st.pyplot(fig, clear_figure=True, use_container_width=True)
