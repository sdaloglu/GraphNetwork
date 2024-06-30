"""
Message vectors analysis for the edge models trained in the study.
The following functions are adapted from Dr. Miles Cranmer's public repository at:

https://github.com/MilesCranmer/symbolic_deep_learning

- linear_transformation_2d()
- out_linear_transformation_2d()
- percentile_sum()

"""

import pickle as pkl
import matplotlib.pyplot as plt
from celluloid import Camera
from copy import deepcopy as copy
import numpy as np
import os
from IPython.display import HTML
from scipy.optimize import minimize
import sys
import argparse

# Open the simulated data from the data directory
title_spring = 'spring_n=4_dim=2'
title_r1 = 'r1_n=4_dim=2'
title_r2 = 'r2_n=4_dim=2'
title_charge = 'charge_n=4_dim=2'


# Choose the title of the simulation and the regularizer type
parser = argparse.ArgumentParser(description='Edge message analysis for different simulations and regularizations.')
parser.add_argument('--data', type=str, help='Title of the simulation to run.')
parser.add_argument('--regularizer', type=str, help='Type of regularizer to use.')
args = parser.parse_args()
title = args.data    # Choose the title of the simulation from the command line
regularizer = args.regularizer  # Choose the type of regularizer from the command line

# Load messages_over_time pkl file
script_dir = os.path.dirname(__file__)
messages_over_time = pkl.load(open(os.path.join(script_dir, f'../models/{title[:2]}/pruned_messages_{title}_{regularizer}.pkl'), 'rb'))

# Load model pkl file
recorded_models = pkl.load(open(os.path.join(script_dir, f'../models/{title[:2]}/pruned_models_{title}_{regularizer}.pkl'), 'rb'))
dim = 2

if regularizer == 'linear_l1' or regularizer == 'standard' or regularizer == 'kl':
  msg_dim = 100
elif regularizer == 'bottleneck':
  msg_dim = dim    # Dimension of the true force


fig = plt.figure(figsize=(8, 6))

# Dynamically adjust the height ratios based on msg_dim
if msg_dim == 2:
    height_ratios = [0.1, 1, 0.1]  # Less vertical stretch for msg_dim 2
else:
    height_ratios = [0.1, 1, 1]  # Default case
    
gs = fig.add_gridspec(3, 2, height_ratios=height_ratios)  # Adjust grid spec for the title axis

# Create an axis for the title
title_ax = fig.add_subplot(gs[0, :])
title_ax.axis('off')  # Turn off axis

# Top row: force components
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
# Bottom row: single sparsity plot spanning two columns
ax3 = fig.add_subplot(gs[2, :])

# Adjust the aspect ratio based on msg_dim
aspect_ratio = 'auto' if msg_dim in [2, 3] else 'equal'
ax3.set_aspect(aspect_ratio)

cam = Camera(fig)


for i in t(range(0, len(messages_over_time), 1)):
    msgs = copy(messages_over_time[i])
    
    # Update the title in the dedicated title axis
    title_ax.text(0.5, 0.5, f'Epoch {i+1}', ha='center', va='center', fontsize=16, transform=title_ax.transAxes)
    title_ax.axis('off')  # Ensure axis remains off
    
    msgs['bd'] = msgs.r + 1e-2

    try:
        msg_columns = ['e%d' % (k) for k in range(1, msg_dim + 1)]
        msg_array = np.array(msgs[msg_columns])
    except:
        msg_columns = ['e%d' % (k) for k in range(msg_dim)]
        msg_array = np.array(msgs[msg_columns])

    msg_importance = msg_array.std(axis=0)
    most_important = np.argsort(msg_importance)[-dim:]
    msgs_to_compare = msg_array[:, most_important]
    msgs_to_compare = (msgs_to_compare - np.average(msgs_to_compare, axis=0)) / np.std(msgs_to_compare, axis=0)

    pos_cols = ['dx', 'dy']

    if dim == 3:
        pos_cols.append('dz')

    if title[:2] == 'sp':
        force_fnc = lambda msg: -2 * (msg.bd.values - 1)[:, None] * np.array(msg[pos_cols]) / msg.bd.values[:, None]
    elif title[:2] == 'ch':
        force_fnc = lambda msg: (msg.q1.values[:, None] * msg.q2.values[:, None]) * np.array(msg[pos_cols]) / (msg.bd.values[:, None]**3)
    elif title[:2] == 'r1':
        force_fnc = lambda msg: -(msg.m1.values[:, None] * msg.m2.values[:, None] * np.array(msg[pos_cols]) ) / (msg.bd.values[:, None]**2)
    elif title[:2] == 'r2':
        force_fnc = lambda msg: (msg.m1.values[:, None] * msg.m2.values[:, None] * np.array(msg[pos_cols]) ) / (msg.bd.values[:, None]**3)


    expected_forces = force_fnc(msgs)
    
    # Function to calculate R2 value
    def calculate_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def percentile_sum(x):
        x_flat = x.flatten()
        lower_bound = x_flat.min()
        upper_bound = np.percentile(x_flat, 90)
        valid_mask = (x_flat >= lower_bound) & (x_flat <= upper_bound)
        valid_elements = x_flat[valid_mask]
        fraction_of_valid_elements = valid_mask.sum() / len(x_flat)
        return valid_elements.sum() / fraction_of_valid_elements

    def linear_transformation_2d(alpha):
        lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
        lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]

        score = (
                percentile_sum(np.square(msgs_to_compare[:, 0] - lincomb1)) +
                percentile_sum(np.square(msgs_to_compare[:, 1] - lincomb2))
        ) / 2.0

        return score

    def out_linear_transformation_2d(alpha):
        lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
        lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]

        return lincomb1, lincomb2


    min_result = minimize(linear_transformation_2d, np.ones(dim ** 2 + dim), method='Powell')
 
    for j in range(dim):
        px = out_linear_transformation_2d(min_result.x)[j]
        r2_values = calculate_r2(msgs_to_compare[:, j], px)

        py = msgs_to_compare[:, j]
        ax = ax1 if j == 0 else ax2

        ax.scatter(px, py, alpha=0.1, s=0.1, color='k')
        ax.set_xlabel('Linear combination of forces')
        ax.set_ylabel(f'Message Element {j+1}')
        # Place R2 value with 3 significant figures
        ax.text(0.5, 0.95, f'$R^2$: {r2_values:.3f}', ha='center', va='top', transform=ax.transAxes, fontsize=12, color='red')

        xlim = np.array([np.percentile(px, q) for q in [10, 90]])
        ylim = np.array([np.percentile(py, q) for q in [10, 90]])
        xlim[0], xlim[1] = xlim[0] - (xlim[1] - xlim[0]) * 0.05, xlim[1] + (xlim[1] - xlim[0]) * 0.05
        ylim[0], ylim[1] = ylim[0] - (ylim[1] - ylim[0]) * 0.05, ylim[1] + (ylim[1] - ylim[0]) * 0.05

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Add a dashed y=x line
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--')

    # Plot sparsity and add standard deviation values
    sorted_indices = np.argsort(msg_importance)[::-1]
    top_indices = sorted_indices[:15]
    sorted_msg_importance = msg_importance[top_indices]

    ax3.pcolormesh(sorted_msg_importance[None, :], cmap='gray_r', edgecolors='k')
    ax3.axis('off')
    ax3.grid(True)
    ax3.set_aspect('equal')
    ax3.text(15.5, 0.5, '...', fontsize=30)

    for k, idx in enumerate(top_indices):
        ax3.text(k + 0.5, -0.5, f'{sorted_msg_importance[k]:.3f}', ha='center', va='top', fontsize=8)

    plt.tight_layout()
    

    cam.snap()


ani = cam.animate()
HTML(ani.to_jshtml())


# Save the video to a file
html_content = ani.to_jshtml()

# Specify the directory and file name
directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'gifs')
file_name = f"pruned_{title}_{regularizer}.html"

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Combine directory and file name to get the full file path
file_path = os.path.join(directory, file_name)

# Write the HTML content to the file
with open(file_path, 'w') as file:
    file.write(html_content)

# Print confirmation message
print(f"Animation saved to: {file_path}")

