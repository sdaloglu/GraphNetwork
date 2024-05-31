# This script applies a symbolic regression by using the PySR package
# to the learned message embeddings

#from pysr import PySRRegressor 
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import sys
import os
from analyze import linear_transformation_2d, out_linear_transformation_2d
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Create a figure and axes
fig, ax = plt.subplots()

msg_dim = 100
dim = 2


# Load the message data from the trained model - this also includes the node embedding for receiving and sending nodes
messages_over_time = pkl.load(open("models/messages_spring_n=4_dim=2_l1.pkl", "rb"))

# Select the last element of the list corresponding to the final epoch
last_message = messages_over_time[-1]


try:
        msg_columns = ['e%d'%(k) for k in range(1, msg_dim+1)]
        msg_array = np.array(last_message[msg_columns])
except:
    msg_columns = ['e%d'%(k) for k in range(msg_dim)]
    msg_array = np.array(last_message[msg_columns])
msg_importance = msg_array.std(axis=0)
most_important = np.argsort(msg_importance)[-dim:]

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