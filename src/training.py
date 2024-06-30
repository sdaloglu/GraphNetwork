from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import sys
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../utils'))    # Add the utils directory to the path
from my_models import loss_function, get_edge_index, GN, update_l1_alpha_linear, update_l1_alpha_triangle
from messages import get_messages
from copy import deepcopy as copy
import pickle as pkl
import subprocess
import argparse


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


title_spring = 'spring_n=4_dim=2'
title_r1 = 'r1_n=4_dim=2'
title_r2 = 'r2_n=4_dim=2'
title_charge = 'charge_n=4_dim=2'

# Choose the title of the simulation and the regularizer type
parser = argparse.ArgumentParser(description='GN training for different simulations and regularizations.')
parser.add_argument('--data', type=str, help='Title of the simulation to run.')
parser.add_argument('--regularizer', type=str, help='Type of regularizer to use.')
args = parser.parse_args()
title = args.data    # Choose the title of the simulation from the command line
regularizer = args.regularizer  # Choose the type of regularizer from the command line





################## Load the data ##################
script_dir = os.path.dirname(__file__)
data = np.load(os.path.join(script_dir, '..', 'data', '{}_data.npy'.format(title)), allow_pickle=True)
a_vals = np.load(os.path.join(script_dir, '..', 'data', '{}_acc.npy'.format(title)), allow_pickle=True)

# Check for NaNs and Infinites in the loaded data
if np.isnan(data).any() or not np.isfinite(data).all():
    raise ValueError("Data contains NaNs or Infinite values which are not suitable for conversion.")
if np.isnan(a_vals).any() or not np.isfinite(a_vals).all():
    raise ValueError("Acceleration contains NaNs or Infinite values which are not suitable for conversion.")


################## Data Preprocessing ##################
# Calculate whiskers for boxplot to determine outliers
q1 = np.percentile(a_vals.flatten(), 25)
q3 = np.percentile(a_vals.flatten(), 75)
iqr = q3 - q1
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr

# Create a mask for each timestep in each simulation based on the whisker boundaries
mask = (a_vals >= lower_whisker) & (a_vals <= upper_whisker)
# Reduce the mask to ensure all particles' accelerations are within the whisker values for each timestep
valid_timesteps_mask = np.all(mask, axis=(2, 3))

# Apply the mask to each simulation's data and acceleration values and convert to tensors
filtered_data = [torch.tensor(data[i][valid_timesteps_mask[i]], dtype=torch.float32) for i in range(data.shape[0])]
filtered_a_vals = [torch.tensor(a_vals[i][valid_timesteps_mask[i]], dtype=torch.float32) for i in range(a_vals.shape[0])]

# Convert the filtered and subsampled data to torch tensors
X_ = torch.cat(filtered_data, dim=0)
y_ = torch.cat(filtered_a_vals, dim=0)

# Select only the first 1 million data points
X_ = X_[:1000000]
y_ = y_[:1000000]

# Print shape of X_ and y_
print(X_.shape)
print(y_.shape)

# Check if the filtered data and acceleration values are the same length
assert len(filtered_data) == len(filtered_a_vals)

# Check if there are any acceleration values outside the whisker boundaries
assert all(torch.all((a >= lower_whisker) & (a <= upper_whisker)) for a in filtered_a_vals)

######################################################



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


# Based on the regularization type, determine the message dimension (number of message features)
if regularizer == 'linear_l1' or regularizer == 'standard':
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





##################################################################
###################### Data Management ###########################
##################################################################
train_batch_size = 64

# Create a list of 800,000 (100x10,000)*(0.80) graph data type for the simulation -- Training Data
train_data_graphs = []
for i in range(len(X_train)):
  # Create a graph data type
  train_data = Data(x = X_train[i].requires_grad_(True), edge_index=edge_indices, y =y_train[i].requires_grad_(True))
  train_data_graphs.append(train_data)

# Create a loader to batch from the train_data_graphs
train_loader = DataLoader(train_data_graphs, batch_size=train_batch_size, shuffle=True)

# len(X_train) = 800,000. --> Number of training data points
# len(train_data_graphs) = 800,000. --> Number of training data points
# len(train_loader) = 12,500. --> Number of batches = [total data points]/[batch size]



# Create a list of 200,000 (100x10,000)*(0.20) graph data type for the simulation -- Testing Data
# This time we shuffle by creating random indices, since there is only one batch

np.random.seed(42)
test_indices = np.random.randint(0,len(X_test),1000)  # Sample 1,000 random data
test_data_graphs = []
for i in test_indices:
  # Create a graph data type
  test_data = Data(x = X_test[i], edge_index=edge_indices, y=y_test[i])
  test_data_graphs.append(test_data)

# Create a loader to batch from the test_data_graphs, batch size is larger since no gradient calculation is required for evalution
test_batch_size = 1000
test_loader =  DataLoader(test_data_graphs, batch_size=int(test_batch_size), shuffle=False)

# len(X_test) = 200,000. --> Number of testing data points
# len(test_data_graphs) = 1,000. --> Number of testing data points chosen randomly
# len(test_loader) = 1 --> Number of batches = [total data points]/[batch size]



##################################################################
###################### Training Loop #############################
##################################################################




# Define epochs
epochs = 50

# set a learning rate but should adjust it to decaying learning schedule (higher to lower)
learning_rate = 1e-3

# Define the optimizer and specify which parameters should be updated during the training process
optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate, weight_decay=1e-8)    # This also includes the weight regularization

batch_per_epoch = len(train_loader)
# batch_per_epoch = 5000    # Limiting the number of batches to 5000 per epoch
total_steps = batch_per_epoch * epochs
current_step = 0 

# Define learning rate scheduler (start with low rate, gradually increasing to max, then lower than the initial learning rate)
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=batch_per_epoch , epochs=epochs, final_div_factor=1e5)


# Initialize the l1_alpha with a small value
base_l1_alpha = 1e-2
max_l1_alpha = 1e-1
l1_alpha = base_l1_alpha



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

      if regularizer == 'l1' or regularizer == 'kl' or regularizer == 'linear_l1' or regularizer == 'triangle_l1':
        total_loss = 0
        # Calculate the loss
        l1_alpha = update_l1_alpha_linear(current_step, total_steps, base_l1_alpha, max_l1_alpha)
        current_step += 1
        
        base_loss, message_reg = loss_function(model=model, graph=batch, augmentation = True, regularizer=regularizer, l1_alpha=1e-2)
        total_loss = base_loss + message_reg
        
      elif regularizer == 'bottleneck' or regularizer == 'standard':
        total_loss = 0
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
  
  # After each epoch get the learned messages of the trained model on a unseen test data
  current_message = get_messages(model,test_loader,msg_dim=message_dim,dim=dim)
  # Adding epoch and loss information
  current_message['epoch'] = epoch
  current_message['train_loss'] = train_loss


  ############### Evaluation ###############
  model.eval()    #Set the model to evaluation mode
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
  model.train()    #Set the model back to training mode
  ############### Evaluation ###############


  current_message['test_loss'] = test_loss
  messages_over_time.append(current_message)    # Record the messages over each epoch
  
  
 

  


##################################################################
################## Save the Trained Model ########################
##################################################################

recorded_models = []

# Append the fully trained final model
model.cpu()
recorded_models.append(model.state_dict())

# Save the data to a file
# Create a new directory called models in the parent directory of script_dir if it does not exist and save the data
models_dir = os.path.join(os.path.dirname(script_dir), 'models')
if not os.path.exists(models_dir):
  os.mkdir(models_dir)

# Save the trained model with the name models_{title}_{regularizer}.pkl
model_path = os.path.join(models_dir, f"{title[:2]}/pruned_models_{title}_{regularizer}.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as f:
  pkl.dump(recorded_models, f)

# Save the messages
messages_path = os.path.join(models_dir, f"{title[:2]}/pruned_messages_{title}_{regularizer}.pkl")
os.makedirs(os.path.dirname(messages_path), exist_ok=True)
with open(messages_path, 'wb') as f:
  pkl.dump(messages_over_time, f)


# Print if the model is saved
print("Model saved successfully")

