from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import sys
sys.path.append('utils')    # Add the utils directory to the path
from my_models import loss_function, edge_index, GN
from copy import deepcopy as copy
import pickle as pkl

# Open the simulated data from the data directory
data = np.load('data/spring_sim_4_particles_data.npy', allow_pickle=True)
a_vals = np.load('data/spring_sim_4_particles_acc.npy', allow_pickle=True)


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
  

# Creating torch tensors from nupy arrays from simulation
X_ = torch.from_numpy(np.concatenate([data[:, i] for i in range(0, data.shape[1], 5)]))   # Use time data with step size 5 (record 1 event in 5 events)
y_ = torch.from_numpy(np.concatenate([a_vals[:, i] for i in range(0, data.shape[1], 5)]))


# Move data to GPU
X = X_.to(device)
y = y_.to(device)


# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

# Number of dimensions in each node embedding (vector)
n_features = X.shape[2]

# Number of particles/nodes
n = 4

# Dimension of the simulation
dim = 2

# Get the edge index matrix
edge_indices = edge_index(n)
# Move to GPU
edge_indices = edge_indices.to(device)

# Import the GN model defined in a seperate .py file
model = GN(input_dim=n_features, # 6 features
           message_dim=100,   # Dimension of the latent space representation (hopefully force) -- Can be 
           output_dim=dim,   # Dimension of the acceleration -- set by the choice of the physics simulation
           hidden_units = 100,   # Intermediate latent space dimension during the forward pass.
           aggregation = 'add',
           edge_index=edge_indices)
# Move to GPU
model = model.to(device)

# Creating a graph (storing its value)
data = Data(x = X_train[0], edge_index=edge_indices, y =y_train[0])



##################################################################
###################### Data Management ###########################
##################################################################
batch_size = 60

# Create a list of 1,000,000 (100x10,000) graph data type for the simulation -- Training Data
train_data = []
for i in range(len(X_train)):

  data = Data(x = X_train[i], edge_index=edge_indices, y =y_train[i])
  train_data.append(data)

# Create a loader to batch from the train_data
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)



# Create a list of 1,000,000 (100x10,000) graph data type for the simulation -- Testing Data
test_data = []
for i in range(len(X_test)):

  data = Data(x = X_test[i], edge_index=edge_indices, y=y_test[i])
  test_data.append(data)

# Create a loader to batch from the test_data, batch size is larger since no gradient calculation is required for evalution
test_loader =  DataLoader(test_data, batch_size=int(20*batch_size), shuffle=True)



# len(train_data) = 800,000. --> Number of training data points
# len(train_loader) = 13,334. --> Number of bathces = [total data points]/[batch size]




##################################################################
###################### Training Loop #############################
##################################################################



#Regularization strength
lambd = 0.01


# Define epochs
epochs = 20

# set a learning rate but should adjust it to decaying learning schedule (higher to lower)
learning_rate = 0.001

# Define the optimizer and specify which parameters should be updated during the training process
optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate, weight_decay=1e-8)

# Define learning rate scheduler (start with low rate, gradually increasing to max, then lower than the initial learning rate)
scheduler = OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader) , epochs=epochs, final_div_factor=1e5)




# Training Loop
for epoch in tqdm(range(epochs)):

  model = model.to(device)
  # The amount of times looping over the full dataset to train the model
  cum_loss = 0.0
  print("__________________")
  print("Epoch number: "+ str(epoch))
  

  i = 0
  while i< 5000:
    for batch in train_loader:
      if i >= 5000:
                break
      i += 1
      # Move the batch of data to GPU
      batch.x = batch.x.to(device)
      batch.y = batch.y.to(device)
      batch.edge_index = batch.edge_index.to(device)
      batch.batch = batch.batch.to(device)

      

      

      # Backward pass and optimize
      optimizer.zero_grad()


      # Calculate the loss
      base_loss, message_reg = loss_function(model=model,graph=batch,edge_index=edge_indices)

      # Adding the weight regularization L2 for the model during training

      l2_reg = torch.tensor(0.).to(device)
      for param in model.parameters():
        l2_reg += torch.norm(param)**2

      # Normalize the loss
      total_loss = (base_loss+message_reg)/batch_size + (lambd/2)*l2_reg

      #Backpropagation algorithm to calculate the gradient of loss w.r.t. all model parameters
      total_loss.backward()

      # Step the optimizer
      optimizer.step()

      # Step the scheduler
      scheduler.step()

      # Calculate the cumilative loss for the batch
      cum_loss += total_loss.item()



  print(cum_loss/(batch_size*5000))   #Averaging over the epoch
  print("__________________")


##################################################################
################## Save the Trained Model ########################
##################################################################

recorded_models = []

# Append the trained model
model.cpu()
recorded_models.append(model.state_dict())

# Save the trained model
pkl.dump(recorded_models,
         open('models_over_time.pkl', 'wb'))

# Print if the model is saved
print("Model saved successfully")