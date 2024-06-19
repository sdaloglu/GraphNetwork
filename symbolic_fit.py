# This script applies a symbolic regression by using the PySR package
# to the learned message embeddings

import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from analyze import linear_transformation_2d, out_linear_transformation_2d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pysr import PySRRegressor 
sys.path.append('utils')  
from my_models import GN, get_edge_index
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Create a figure and axes
fig, ax = plt.subplots()

msg_dim = 100
dim = 2

title_spring = 'spring_n=4_dim=2'
title_r1 = 'r1_n=4_dim=2'
title_r2 = 'r2_n=4_dim=2'
title_charge = 'charge_n=4_dim=2'

title = title_spring


regularizer = 'standard'

# Load the message data from the trained model - this also includes the node embedding for receiving and sending nodes
messages_over_time = pkl.load(open(f"models/pruned_messages_{title}_{regularizer}.pkl", "rb"))


# Load the model data
final_model = pkl.load(open(f"models/pruned_models_{title}_{regularizer}.pkl", "rb"))

# Select the last element of the list corresponding to the final epoch
last_message = messages_over_time[-1]

try:
    msg_columns = ['e%d'%(k) for k in range(1, msg_dim+1)]
    msg_array = np.array(last_message[msg_columns])
except:
    msg_columns = ['e%d'%(k) for k in range(msg_dim)]
    msg_array = np.array(last_message[msg_columns])
msg_importance = msg_array.std(axis=0)    # Extract the standard deviation of each message element
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
ax.set_title('Loss for the Spring System with L1 Regularization')
ax.legend()


# Save the figure
fig.savefig('plots/loss_spring_n=4_dim=2_l1.png')


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
    elementwise_loss="f(x, y) = abs(x - y)"
)

model2 = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    #unary_operators=["exp", "abs", "sqrt"],
    elementwise_loss="f(x, y) = abs(x - y)"
)


F1 = last_message['e%d'%(most_important[0],)]
F2 = last_message['e%d'%(most_important[1],)]

inputs = last_message[['dx', 'dy', 'r', 'm1', 'm2']]

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
final_model = final_model[0]


# Initialize the model (ensure the parameters match those used during training)
model = GN(input_dim=6, message_dim=100, output_dim=2, hidden_units=300, aggregation='add')


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
msg_aggr_tensor = msg_array.reshape(-1, 3, 100).sum(axis=1)

# Get the node embeddings
node_embeddings = last_message[['x1', 'y1', 'vx1', 'vy1', 'q1', 'm1']][::3]
node_embeddings_df = node_embeddings.reset_index(drop=True)
node_embeddings_tensor = torch.tensor(node_embeddings_df.values, dtype=torch.float32)
print(msg_aggr_tensor.shape)
print(msg_array.shape)
print(node_embeddings_tensor.shape)

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

print(acc1.shape)
print(acc2.shape)
print(F1_aggr.shape)
print(F2_aggr.shape)

# Create a symbolic regression model to fit the output of the node model
model3 = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    variable_names=["f1"],
    elementwise_loss="f(x, y) = abs(x - y)"
)

model4 = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    variable_names=["f2"],
    elementwise_loss="f(x, y) = abs(x - y)"
)


input1 = pd.concat([F1_aggr_series, node_embeddings_df], axis=1)
input2 = pd.concat([F2_aggr_series, node_embeddings_df], axis=1)

# Save the dataframe 
input1.to_csv('input_dataframe.csv', index=False)
input2.to_csv('input_dataframe.csv', index=False)


# First fit to the first dimension of the acceleration (node model output)
model3.fit(input1, acc1)

# Then fit to the second dimension of the acceleration (node model output)
model4.fit(input2, acc2)




# Save the best symbolic formulation
with open(f"{title}_{regularizer}.txt", "w") as f:
    f.write(str(model1.sympy()))
    f.write("\n")
    f.write(str(model2.sympy()))
    f.write("\n")
    f.write(str(model3.sympy()))
    f.write("\n")
    f.write(str(model4.sympy()))
    
print("Symbolic regression fit complete")



def model1_formula(dx, dy, r):
    return [model1.sympy().subs({'dx': x, 'dy': y, 'r': z}) for x, y, z in zip(dx, dy, r)]

def model2_formula(dx, dy, r):
    return [model2.sympy().subs({'dx': x, 'dy': y, 'r': z}) for x, y, z in zip(dx, dy, r)]

def model3_formula(F1_aggr, m1):
    return [model3.sympy().subs({'F1_aggr': x, 'm1': y}) for x, y in zip(F1_aggr, m1)]

def model4_formula(F2_aggr, m1):
    return [model4.sympy().subs({'F2_aggr': x, 'm1': y}) for x, y in zip(F2_aggr, m1)]


def NewtonLaw_x(dx, dy, r, m1):
    F1 = model1_formula(dx, dy, r)
    F1_aggr = pd.Series(F1).values.reshape(-1, 3).sum(axis=1)
    acc1 = model3_formula(F1_aggr, m1[::3])
    acc1_array = np.array(acc1)
    return acc1_array, acc1_array.shape

def NewtonLaw_y(dx, dy, r, m1):
    F2 = model2_formula(dx, dy, r)
    F2_aggr = pd.Series(F2).values.reshape(-1, 3).sum(axis=1)
    acc2 = model4_formula(F2_aggr, m1[::3])
    acc2_array = np.array(acc2)
    return acc2_array, acc2_array.shape



# Feedin the test data to the symbolic equations

inputs = last_message[['dx', 'dy', 'r', 'm1']]
acc1, acc1_shape = NewtonLaw_x(inputs['dx'], inputs['dy'], inputs['r'], inputs['m1'])
acc2, acc2_shape = NewtonLaw_y(inputs['dx'], inputs['dy'], inputs['r'], inputs['m1'])

print(acc1_shape)
print(acc2_shape)



from training_stand import y_test

print(y_test.shape)
torch.Size([3011850, 4, 2])


acc1_true = y_test[:, :, 0]
acc2_true = y_test[:, :, 1]

print(acc1_true.shape)
print(acc2_true.shape)
