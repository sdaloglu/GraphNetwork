# This script is for fitting a symbolic regression model to the data

from pysr import PySRRegressor 


default_pysr_params = dict(
    populations = 30,
    model_selection = "best",
)


# Create the symbolic regression model and train it
model = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log", "abs", "sqrt"],
    **default_pysr_params
)


model.fit(X_train, y_train)

