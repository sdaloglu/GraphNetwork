import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys
sys.path.append('utils')    # Add the utils directory to the path
from my_models import get_edge_index


title_spring = 'spring_n=4_dim=2'
title_r1 = 'r1_n=4_dim=2'
title_r2 = 'r2_n=4_dim=2'
title_charge = 'charge_n=4_dim=2'

title = title_spring


data = np.load('data/{}_data.npy'.format(title), allow_pickle=True)
a_vals = np.load('data/{}_acc.npy'.format(title), allow_pickle=True)

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

# Check if the filtered data and acceleration values are the same length
assert len(filtered_data) == len(filtered_a_vals)

# Check if there are any acceleration values outside the whisker boundaries
assert all(torch.all((a >= lower_whisker) & (a <= upper_whisker)) for a in filtered_a_vals)

######################################################

edge_indices = get_edge_index(4)
# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.25, shuffle=False)

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



