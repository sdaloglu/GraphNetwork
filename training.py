from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import sys
sys.path.append('utils')    # Add the utils directory to the path
from my_models import loss_function, get_edge_index, GN
from messages import get_messages
from copy import deepcopy as copy
import pickle as pkl
import subprocess


# Extract the PyTorch version
version_nums = torch.__version__.split('.')
# Torch Geometric seems to always build for *.*.0 of torch :
version_nums[-1] = '0' + version_nums[-1][1:]
os.environ['TORCH'] = '.'.join(version_nums)

# The pip install command as a string
pip_command = f"pip install --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-{os.environ['TORCH']}.html && pip install --upgrade torch-sparse -f https://pytorch-geometric.com/whl/torch-{os.environ['TORCH']}.html && pip install --upgrade torch-geometric"

# Use subprocess to run the command
subprocess.check_call(pip_command, shell=True)


# Number of nodes
n_particles = 4

# Open the simulated data from the data directory
title_spring = 'spring_n=4_dim=2'
title_r1 = 'r1_n=4_dim=2'
title_r2 = 'r2_n=4_dim=2'
title_charge = 'charge_n=4_dim=2_'


# Choose the title of the simulation
title = title_spring


data = np.load('data/{}_data.npy'.format(title), allow_pickle=True)
a_vals = np.load('data/{}_acc.npy'.format(title), allow_pickle=True)


# Checking if the NVIDIA GPU is available (on the cloud service if run on Colab)
if torch.cuda.is_available():
  device = torch.device("cuda")
  x = torch.rand(3)
  x = x.to(device)
  print(x)
elif torch.backends.mps.is_available():
  device = torch.device("mps")
  print("Using the Mac's GPU")
else:
  device = torch.device("cpu")
  print("CUDA or MPS is not available using CPU")
  

# Creating torch tensors from numpy arrays from simulation
#X_ = torch.from_numpy(np.concatenate([data[:, i] for i in range(0, data.shape[1], 5)]))   # Use time data with step size 5 (record 1 event in 5 events)
#y_ = torch.from_numpy(np.concatenate([a_vals[:, i] for i in range(0, data.shape[1], 5)]))
X_ = torch.from_numpy(np.concatenate([data[:, i] for i in range(data.shape[1])]))   # Use time data with step size 1
y_ = torch.from_numpy(np.concatenate([a_vals[:, i] for i in range(data.shape[1])]))


# Move data to GPU
X = X_.to(device)
y = y_.to(device)


# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle=False)

# Number of dimensions in each node embedding (vector)
n_features = X.shape[2]

# Number of particles/nodes
n = n_particles

# Dimension of the simulation
dim = 2

# Specify the type of regularization methods used to constrain the learned message embedding
regularizer = 'l1'

# Based on the regularization type, determine the message dimension (number of message features)
if regularizer == 'l1' or regularizer == 'standard':
  message_dim = 100
elif regularizer == 'kl':
  message_dim = 200
elif regularizer == 'bottleneck':
  message_dim = dim    # Dimension of the true force

# Get the edge index matrix
edge_indices = get_edge_index(n)
# Move to GPU
edge_indices = edge_indices.to(device)

# Import the GN model defined in a seperate .py file
model = GN(input_dim=n_features, # 6 features
           message_dim=message_dim,   # Dimension of the latent space representation (hopefully force) -- 
           output_dim=dim,   # Dimension of the acceleration -- set by the choice of the physics simulation
           hidden_units = 300,   # Intermediate latent space dimension during the forward pass.
           aggregation = 'add')
# Move to GPU
model = model.to(device)

# Creating a graph (storing its value)
data = Data(x = X_train[0], edge_index = edge_indices, y = y_train[0])



##################################################################
###################### Data Management ###########################
##################################################################
train_batch_size = 64

# Create a list of 750,000 (100x10,000)*(0.75) graph data type for the simulation -- Training Data
train_data = []
for i in range(len(X_train)):
  # Create a graph data type
  data = Data(x = X_train[i].requires_grad_(True), edge_index=edge_indices, y =y_train[i].requires_grad_(True))
  train_data.append(data)

# Create a loader to batch from the train_data
train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

# len(X_train) = 750,000. --> Number of training data points
# len(train_data) = 750,000. --> Number of training data points
# len(train_loader) = 11,719. --> Number of batches = [total data points]/[batch size]



# Create a list of 250,000 (100x10,000)*(0.25) graph data type for the simulation -- Testing Data
# This time we shuffle by creating random indices, since there is only one batch

np.random.seed(42)
test_indices = np.random.randint(0,len(X_test),1000)  # Sample 1,000 random data
test_data = []
for i in test_indices:
  # Create a graph data type
  data = Data(x = X_test[i], edge_index=edge_indices, y=y_test[i])
  test_data.append(data)

# Create a loader to batch from the test_data, batch size is larger since no gradient calculation is required for evalution
test_batch_size = 1000
test_loader =  DataLoader(test_data, batch_size=int(test_batch_size), shuffle=False)

# len(X_test) = 250,000. --> Number of testing data points
# len(test_data) = 1,000. --> Number of testing data points chosen randomly
# len(test_loader) = 1 --> Number of batches = [total data points]/[batch size]



##################################################################
###################### Training Loop #############################
##################################################################




# Define epochs
epochs = 30

# set a learning rate but should adjust it to decaying learning schedule (higher to lower)
learning_rate = 1e-3

# Define the optimizer and specify which parameters should be updated during the training process
optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate, weight_decay=1e-8)    # This also includes the weight regularization

# batch_per_epoch = len(train_loader)
batch_per_epoch = 10000    # Limiting the number of batches to 5000 per epoch

# Define learning rate scheduler (start with low rate, gradually increasing to max, then lower than the initial learning rate)
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=batch_per_epoch , epochs=epochs, final_div_factor=1e5)

# Define an empty array for the messages
messages_over_time = []

# Training Loop
for epoch in tqdm(range(epochs)):

  model = model.to(device)
  # The amount of times looping over the full dataset to train the model
  cum_loss = 0.0
  print("__________________")
  print("Epoch number: "+ str(epoch))
  

  i = 0
  while i< batch_per_epoch:    # Limiting the number of batches to 5000 per epoch
    for batch in train_loader:
      if i >= batch_per_epoch:
                break
      i += 1
      # Move the batch of data to GPU
      batch = batch.to(device)
 

      # Backward pass and optimize
      optimizer.zero_grad()

      if regularizer == 'l1' or regularizer == 'kl':
        
        # Calculate the loss
        base_loss, message_reg = loss_function(model=model, graph=batch, augmentation = True, regularizer=regularizer)
        total_loss = base_loss + message_reg
        
      elif regularizer == 'bottleneck' or regularizer == 'standard':
        total_loss = model.loss(batch, augmentation=True)
        base_loss = total_loss  # No regularization

      # Back-propagation algorithm to calculate the gradient of loss w.r.t. all model parameters
      total_loss.backward()

      # Step the optimizer
      optimizer.step()

      # Step the scheduler
      scheduler.step()

      # Calculate the cumulative loss for the batch
      cum_loss += base_loss.item()
      
  print("__________________")
  train_loss = cum_loss/(batch_per_epoch)
  print("Train Loss: ",train_loss)   #Averaging over the epoch

  # Set the test loss to zero for the next epoch (beginning of an epoch)
  test_loss = 0.0
  for test_batch in test_loader:    # We need multiple batches per epoch to plot y=x
    test_batch = test_batch.to(device)  # Move the test_batch to GPU
    
    # Calculate the test loss after each epoch
    loss = model.loss(test_batch, augmentation = False).item()
    test_loss += loss 
    
  test_loss = test_loss/(len(test_loader)) #Averaging over the epoch
  print("Test Loss: ", test_loss)
  print("__________________")
  
  # After each epoch get the learned messages of the trained model on a unseen test data
  current_message = get_messages(model,test_loader,msg_dim=message_dim,dim=dim)
  
  # Adding epoch and loss information
  current_message['epoch'] = epoch
  current_message['train_loss'] = train_loss
  current_message['test_loss'] = test_loss
  messages_over_time.append(current_message)    # Record the messages over each epoch
  
  
 

  


##################################################################
################## Save the Trained Model ########################
##################################################################

recorded_models = []

# Append the trained model
model.cpu()
recorded_models.append(model.state_dict())

# Save the data to a file
# Create a new directory called models if it does not exist and save the data
if not os.path.exists('models'):
  os.mkdir('models')
  
# Save the trained model with the name models_{title}_{regularizer}.pkl
pkl.dump(recorded_models, open('models/models_{}_{}.pkl'.format(title, regularizer), 'wb'))

# Save the messages
pkl.dump(messages_over_time, open('models/messages_{}_{}.pkl'.format(title, regularizer), 'wb'))



# Print if the model is saved
print("Model saved successfully")

