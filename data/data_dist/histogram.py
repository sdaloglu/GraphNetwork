# Load the data and check if it has NaN values and also make a histogram of the acceleration values and other data components

import numpy as np
import matplotlib.pyplot as plt
import os

# Open the simulated data from the data directory
title_spring = 'spring_n=4_dim=2'
title_r1 = 'r1_n=4_dim=2'
title_r2 = 'r2_n=4_dim=2'
title_charge = 'charge_n=4_dim=2'

# Choose the title of the simulation
title = title_charge

# Load the data
script_dir = os.path.dirname(os.path.realpath(__file__))
original_data = np.load(os.path.join(os.path.dirname(script_dir), '{}_data.npy'.format(title)), allow_pickle=True)
original_a_vals = np.load(os.path.join(os.path.dirname(script_dir), '{}_acc.npy'.format(title)), allow_pickle=True)

# Check for NaNs and Infinites in the loaded data
if np.isnan(original_data).any() or not np.isfinite(original_data).all():
    raise ValueError("Data contains NaNs or Infinite values which are not suitable for conversion.")
if np.isnan(original_a_vals).any() or not np.isfinite(original_a_vals).all():
    raise ValueError("Acceleration contains NaNs or Infinite values which are not suitable for conversion.")

################## Data Preprocessing ##################
# Calculate whiskers for boxplot to determine outliers
q1 = np.percentile(original_a_vals.flatten(), 25)
q3 = np.percentile(original_a_vals.flatten(), 75)
iqr = q3 - q1
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr

# Create a mask for each timestep in each simulation based on the whisker boundaries
mask = (original_a_vals >= lower_whisker) & (original_a_vals <= upper_whisker)
# Reduce the mask to ensure all particles' accelerations are within the whisker values for each timestep
valid_timesteps_mask = np.all(mask, axis=(2, 3))

# Apply the mask to each simulation's data and acceleration values
filtered_data = [original_data[i][valid_timesteps_mask[i]] for i in range(original_data.shape[0])]
filtered_a_vals = [original_a_vals[i][valid_timesteps_mask[i]] for i in range(original_a_vals.shape[0])]
# Concatenate the filtered and subsampled data
filtered_a_vals = np.concatenate(filtered_a_vals, axis=0)
q1 = np.percentile(filtered_a_vals.flatten(), 25)
q3 = np.percentile(filtered_a_vals.flatten(), 75)
iqr = q3 - q1
lower_whisker_filtered = q1 - 1.5 * iqr
upper_whisker_filtered = q3 + 1.5 * iqr

######################################################

# Function to create histograms
def create_histograms(data, a_vals, title, suffix, lower_whisker=lower_whisker, upper_whisker=upper_whisker):
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    fig.suptitle(f'Data Distribution {suffix} for Charge System', fontsize=20)

    # Plotting histogram of the acceleration values
    axs[0, 0].hist(a_vals.flatten(), bins=50, alpha=0.7)
    axs[0, 0].set_title('Histogram of Acceleration Values', fontsize=16)
    axs[0, 0].set_xlabel('$a_x$ and $a_y$', fontsize=14)
    axs[0, 0].set_ylabel('Frequency', fontsize=14)
    axs[0, 0].grid(True)

    # Box plot of acceleration values with whiskers and outliers
    axs[0, 1].boxplot(a_vals.flatten(), notch=True, vert=True, patch_artist=True, showfliers=True, flierprops=dict(marker='o', color='red', alpha=0.7))
    axs[0, 1].set_title('Boxplot of Acceleration Values', fontsize=16)
    axs[0, 1].set_xlabel('$a_x$ and $a_y$', fontsize=14)
    axs[0, 1].set_ylabel('Value', fontsize=14)
    axs[0, 1].grid(True)
    num_outliers = np.sum((a_vals.flatten() < lower_whisker) | (a_vals.flatten() > upper_whisker))
    total_data_points = a_vals.size
    axs[0, 1].legend([f'Lower whisker: {lower_whisker:.5g}', f'Upper whisker: {upper_whisker:.5g}', f'Outliers: {num_outliers}', f'Total data points: {total_data_points}'], loc='center right', bbox_to_anchor=(1.4, 0.5), fontsize=14)
    
    # Histograms for x, y, Vx, and Vy values
    variables = ['x', 'y', '$V_x$', '$V_y$']
    for i, var in enumerate(variables):
        if suffix == "Before Pruning":
            all_var_values = data[:,:,:, i].flatten()  # Concatenate all array slices for the variable

        else:
            all_var_values = np.concatenate([d[:,:, i].flatten() for d in data])  # Concatenate all array slices for the variable
            
        axs[i//2+1, i%2].hist(all_var_values, bins=50, alpha=0.7)
        axs[i//2+1, i%2].set_title(f'Histogram of {var} Values', fontsize=16)
        axs[i//2+1, i%2].set_xlabel(var, fontsize=14)
        axs[i//2+1, i%2].set_ylabel('Frequency', fontsize=14)
        axs[i//2+1, i%2].grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(script_dir, f'all_histograms_{title}_{suffix}.png'))
    plt.close(fig)
# Create histograms before pruning
create_histograms(original_data, original_a_vals, title, "Before Pruning")

# Create histograms after pruning
create_histograms(filtered_data, filtered_a_vals, title, "After Pruning", lower_whisker_filtered, upper_whisker_filtered)
