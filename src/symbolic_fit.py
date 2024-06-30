# This script applies a symbolic regression by using the PySR package
# to the learned message embeddings

import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pysr import PySRRegressor 
sys.path.append('../utils')  
from my_models import GN, get_edge_index
from data_loading import test_loader
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse



# Create a figure and axes
fig, ax = plt.subplots()
dim = 2

title_spring = 'spring_n=4_dim=2'
title_r1 = 'r1_n=4_dim=2'
title_r2 = 'r2_n=4_dim=2'
title_charge = 'charge_n=4_dim=2'


# Choose the title of the simulation and the regularizer type
parser = argparse.ArgumentParser(description='Symbolic regression fit for different simulations.')
parser.add_argument('--data', type=str, help='Title of the simulation to run.')
parser.add_argument('--regularizer', type=str, help='Type of regularizer to use.')
args = parser.parse_args()
title = args.data    # Choose the title of the simulation from the command line
regularizer = args.regularizer  # Choose the type of regularizer from the command line

#title = title_r1
#regularizer = 'bottleneck'


if regularizer in ['linear_l1', 'standard','kl','const_l1']:
    msg_dim = 100
elif regularizer == 'bottleneck':
    msg_dim = 2

# Load the message data from the trained model - this also includes the node embedding for receiving and sending nodes
script_dir = os.path.dirname(__file__)
messages_over_time = pkl.load(open(os.path.join(script_dir, '..', 'models', title[:2], 'pruned_messages_{title}_{regularizer}.pkl'), "rb"))

# Ensure the number of epochs is 50
print(len(messages_over_time))
assert len(messages_over_time) == 50 or len(messages_over_time) == 100

# Load the model data
final_model = pkl.load(open(os.path.join(script_dir, '..', 'models', title[:2], 'pruned_models_{title}_{regularizer}.pkl'), "rb"))

# Ensure only the fully trained last model is recorded
assert len(final_model) == 1

# Select the last element of the list corresponding to the final epoch
last_message = messages_over_time[-1]

try:
    msg_columns = ['e%d'%(k) for k in range(1, msg_dim+1)]
    msg_array = np.array(last_message[msg_columns])
except:
    msg_columns = ['e%d'%(k) for k in range(msg_dim)]
    msg_array = np.array(last_message[msg_columns])
    
    
if regularizer in ['linear_l1', 'bottleneck', 'standard', "const_l1"]:
    msg_importance = msg_array.std(axis=0)    # Extract the standard deviation of each message element
    most_important = np.argsort(msg_importance)[-dim:]    # Find the indices of the most important message elements
    
elif regularizer == 'kl':
    msg_importance = np.sum(msg_array**2, axis=0)
    most_important = np.argsort(msg_importance)[-dim:]    # Find the indices of the most important message elements

print("Most important message elements are:", most_important)
print("Most important message elements have std:", msg_importance[most_important])


# Extract the training loss values from each epoch
losses_train = [x['train_loss'][:1][0] for x in messages_over_time]
losses_test = [x['test_loss'][:1][0] for x in messages_over_time]

# Epochs
epochs = np.arange(1, len(losses_train)+1)


# Plot the training loss over time
# MAE loss of the batch of size train_batch_size over 30 epochs
ax.plot(epochs, losses_train, label='Training Loss')

# Plot the test loss over time
# MAE loss of the batch of size test_batch_size over 30 epochs
ax.plot(epochs, losses_test, label='Test Loss')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'Loss for the {title} with {regularizer} Regularization')
ax.legend()


# Save the figure
fig.savefig(os.path.join(script_dir, '..', 'symbolic_fit_results', 'loss_{title}_{regularizer}.png'))


###############################################################
# Extract the final test loss after full training
###############################################################
final_test_loss = losses_test[-1]
print("Final test loss after full training:", final_test_loss)



###############################################################
# Fit the symbolic regression model to the edge model
###############################################################

# Create the symbolic regression model and train it
model1 = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    #unary_operators=["exp", "abs", "sqrt"],
    elementwise_loss="f(x, y) = abs(x - y)",
)

model2 = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    #unary_operators=["exp", "abs", "sqrt"],
    elementwise_loss="f(x, y) = abs(x - y)",
)


F1 = last_message['e%d'%(most_important[0],)]
F2 = last_message['e%d'%(most_important[1],)]

inputs = last_message[['dx', 'dy', 'r', 'm1', 'm2', 'q1', 'q2']]

# First fit the highest standard deviation message element
model1.fit(inputs, F1)

# Then fit the second highest standard deviation message element
model2.fit(inputs, F2)






###############################################################
# Fit the symbolic regression to the node model
###############################################################



# Assert that the final model array has a length of 1
assert len(final_model) == 1

# Extract the final trained model from which the output of the node model will be fit
final_model = final_model[-1]


# Initialize the model (ensure the parameters match those used during training)

if regularizer == 'kl':
    model = GN(input_dim=6, message_dim=200, output_dim=2, hidden_units=300, aggregation='add')
else:
    model = GN(input_dim=6, message_dim=msg_dim, output_dim=2, hidden_units=300, aggregation='add')


# Load the model state
model.load_state_dict(final_model)  # Load the last model state for the fully trained model


# Ensure the model is in evaluation mode
model.eval()



# First aggregate the edge messages for each node
# Adding the 3 back to back rows since we have 4 nodes and each node has 3 edges
# msg_array has shape [12000, 100]
# Reshape the array to (4000, 3, 100) and then sum over the second axis to aggregate every 3 rows


# Convert msg_array to a tensor
msg_array = torch.tensor(msg_array)
msg_aggr_tensor = msg_array.reshape(-1, 3, msg_dim).sum(axis=1)

# Get the node embeddings
node_embeddings = last_message[['x1', 'y1', 'vx1', 'vy1', 'm1', 'q1']][::3]
node_embeddings_df = node_embeddings.reset_index(drop=True)
node_embeddings_tensor = torch.tensor(node_embeddings_df.values, dtype=torch.float32)
# print(msg_aggr_tensor.shape)
# print(msg_array.shape)
# print(node_embeddings_tensor.shape)

output = model.node_model(torch.cat([node_embeddings_tensor, msg_aggr_tensor], dim=1))
outputs_numpy = output.detach().numpy()

# Determine the standard deviation of each column in the outputs_numpy array
std_devs = np.std(outputs_numpy, axis=0)

# Get indices of the columns sorted by standard deviation in descending order
sorted_indices_by_std = np.argsort(std_devs)[::-1]

# Select the first and second highest varied columns based on standard deviation
acc1 = outputs_numpy[:, sorted_indices_by_std[0]]
acc2 = outputs_numpy[:, sorted_indices_by_std[1]]




# Feed in the aggregated edge messages to the node model, add three back to back rows
F1_aggr = F1.values.reshape(-1, 3).sum(axis=1)
F1_aggr_series = pd.Series(F1_aggr, name='F1_aggr').reset_index(drop=True)
F2_aggr = F2.values.reshape(-1, 3).sum(axis=1)
F2_aggr_series = pd.Series(F2_aggr, name='F2_aggr').reset_index(drop=True)

# print(acc1.shape)
# print(acc2.shape)
# print(F1_aggr.shape)
# print(F2_aggr.shape)

# Create a symbolic regression model to fit the output of the node model
model3 = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    variable_names=["f1"],
    elementwise_loss="f(x, y) = abs(x - y)",
)

model4 = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    variable_names=["f2"],
    elementwise_loss="f(x, y) = abs(x - y)",
)


input1 = pd.concat([F1_aggr_series, last_message[['x1', 'y1', 'm1']][::3].reset_index(drop=True)], axis=1)
input2 = pd.concat([F2_aggr_series, last_message[['x1', 'y1', 'm1']][::3].reset_index(drop=True)], axis=1)

# Save the dataframe 
#input1.to_csv(f'symbolic_fit/input_dataframe_{title}_{regularizer}.csv', index=False)
#input2.to_csv(f'symbolic_fit/input_dataframe_{title}_{regularizer}.csv', index=False)


# First fit to the first dimension of the acceleration (node model output)
model3.fit(input1, acc1)

# Then fit to the second dimension of the acceleration (node model output)
model4.fit(input2, acc2)


# Define the symbolic formulas for the edge models
def model1_formula(dx, dy, r, m1, m2, q1, q2):
    return [model1.sympy().subs({'dx': x, 'dy': y, 'r': z, 'm1': w, 'm2': v, 'q1': u, 'q2': t}).evalf() for x, y, z, w, v, u, t in zip(dx, dy, r, m1, m2, q1, q2)]

def model2_formula(dx, dy, r, m1, m2, q1, q2):
    return [model2.sympy().subs({'dx': x, 'dy': y, 'r': z, 'm1': w, 'm2': v, 'q1': u, 'q2': t}).evalf() for x, y, z, w, v, u, t in zip(dx, dy, r, m1, m2, q1, q2)]

# Define the symbolic formulas for the node models
def model3_formula(F1_aggr, x1, y1, vx1, vy1, m1, q1):
    return [model3.sympy().subs({'F1_aggr': a, 'x1': b, 'y1': c, 'vx1': d, 'vy1': e, 'm1': f, 'q1': g}).evalf() for a, b, c, d, e, f, g in zip(F1_aggr, x1, y1, vx1, vy1, m1, q1)]

def model4_formula(F2_aggr, x1, y1, vx1, vy1, m1, q1):
    return [model4.sympy().subs({'F2_aggr': a, 'x1': b, 'y1': c, 'vx1': d, 'vy1': e, 'm1': f, 'q1': g}).evalf() for a, b, c, d, e, f, g in zip(F2_aggr, x1, y1, vx1, vy1, m1, q1)]

# Define an overall symbolic formula that combines the edge and node models
def NewtonLaw_x(dx, dy, r, m1, m2, x1, y1, vx1, vy1, q1, q2):
    F1 = model1_formula(dx, dy, r, m1, m2, q1, q2)
    F1_aggr = pd.Series(F1).values.reshape(-1, 3).sum(axis=1)
    m1, x1, y1, vx1, vy1, q1 = [var[::3].reset_index(drop=True) for var in [m1, x1, y1, vx1, vy1, q1]]
    
    # Assert that the length of the mass array is the same as the length of the F1_aggr array
    assert len(m1) == len(F1_aggr)
    
    acc1 = model3_formula(F1_aggr, x1, y1, vx1, vy1, m1, q1)
    return acc1

# Define an overall symbolic formula that combines the edge and node models
def NewtonLaw_y(dx, dy, r, m1, m2, x1, y1, vx1, vy1, q1, q2):
    F2 = model2_formula(dx, dy, r, m1, m2, q1, q2)
    F2_aggr = pd.Series(F2).values.reshape(-1, 3).sum(axis=1)
    m1, x1, y1, vx1, vy1, q1 = [var[::3].reset_index(drop=True) for var in [m1, x1, y1, vx1, vy1, q1]]
    
    # Assert that the length of the mass array is the same as the length of the F2_aggr array
    assert len(m1) == len(F2_aggr)
    
    acc2 = model4_formula(F2_aggr, x1, y1, vx1, vy1, m1, q1)
    return acc2



# Feeding the test data to the symbolic equations
inputs = last_message[['dx', 'dy', 'r', 'm1', 'm2', 'x1', 'y1', 'vx1', 'vy1', 'q1', 'q2']]
acc1 = NewtonLaw_x(inputs['dx'], inputs['dy'], inputs['r'], inputs['m1'], inputs['m2'], inputs['x1'], inputs['y1'], inputs['vx1'], inputs['vy1'], inputs['q1'], inputs['q2'])
acc2 = NewtonLaw_y(inputs['dx'], inputs['dy'], inputs['r'], inputs['m1'], inputs['m2'], inputs['x1'], inputs['y1'], inputs['vx1'], inputs['vy1'], inputs['q1'], inputs['q2'])


# Save acc1 and acc2
#pd.DataFrame({'acc1': acc1, 'acc2': acc2}).to_csv('symbolic_fit/acceleration_data_symbolic.csv')

# Convert acc1 and acc2 to a PyTorch tensor to match the type of acc1_true, and acc2_true
acc1 = torch.tensor(acc1, dtype=torch.float32)
acc2 = torch.tensor(acc2, dtype=torch.float32)

# Loading the data to access the actual values of the acceleration
# Assert that the size of the test loader is one since there is a single batch of size 1000
assert len(test_loader) == 1

y_test = []
for test_batch in test_loader:
    y_test.append(test_batch.y)
    
y_test = torch.cat(y_test, dim=0)

acc1_true = y_test[:, 0]
acc2_true = y_test[:, 1]

print(acc1_true.shape)
print(acc2_true.shape)

# Calculate the MAE for the first dimension of the acceleration
mae1 = torch.sum(torch.abs(acc1_true - acc1))/1000
mae1_other = torch.sum(torch.abs(acc1_true - acc2))/1000
print("MAE for the first dimension of the acceleration by using the symbolic equation:", mae1)
print("MAE for the first dimension of the acceleration by using the other symbolic equation:", mae1_other)
# Compare to the MAE of the node model

# Calculate the MAE for the second dimension of the acceleration
mae2 = torch.mean(torch.abs(acc2_true - acc2))
mae2_other = torch.mean(torch.abs(acc2_true - acc1))
print("MAE for the second dimension of the acceleration by using the symbolic equation:", mae2)
print("MAE for the second dimension of the acceleration by using the other symbolic equation:", mae2_other)


print("MAE with the GN model is:", final_test_loss)




# Save the best symbolic formulation in the symbolic_fit directory
with open(os.path.join(script_dir, '..', 'symbolic_fit_results', '{}_{}.txt'.format(title, regularizer)), "w") as f:
    f.write("Symbolic equation for \( F_x \):\n")
    f.write(str(model1.sympy()))
    f.write("\n\n")
    f.write("Symbolic equation for \( F_y \):\n")
    f.write(str(model2.sympy()))
    f.write("\n\n")
    f.write("Symbolic equation for \( a_x \):\n")
    f.write(str(model3.sympy()))
    f.write("\n\n")
    f.write("Symbolic equation for \( a_y \):\n")
    f.write(str(model4.sympy()))
    
    # Save the final test loss
    f.write("\n\n")
    f.write(f"Final test loss of the GN after full training: {final_test_loss}")
    
    # Save the MAE of the symbolic equations
    f.write("\n")
    f.write(f"MAE for the first dimension of the acceleration by using the symbolic equation: {mae1}")
    f.write("\n")
    f.write(f"MAE for the second dimension of the acceleration by using the symbolic equation: {mae2}")
    
print("Symbolic regression fit complete")

