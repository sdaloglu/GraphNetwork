import pickle as pkl
import matplotlib.pyplot as plt
from celluloid import Camera
from copy import deepcopy as copy
import numpy as np
import os
from IPython.display import HTML


# Open the simulated data from the data directory
title_spring = 'spring_n=4_dim=2'
title_r1 = 'r1_n=4_dim=2'
title_r2 = 'r2_n=4_dim=2'
title_charge = 'charge_n=4_dim=2'


# Choose the title of the simulation
title = title_r1

regularizer = 'l1'

# Load messages_over_time pkl file
messages_over_time = pkl.load(open('models/messages_{}_{}.pkl'.format(title, regularizer), 'rb'))

# Load model pkl file
recorded_models = pkl.load(open('models/models_{}_{}.pkl'.format(title, regularizer), 'rb'))

dim = 2
msg_dim = 100
sim = 'r1'

fig = plt.figure(figsize=(8, 6))

# Dynamically adjust the height ratios based on msg_dim
if msg_dim == 2:
    height_ratios = [0.1, 1, 0.1]  # Less vertical stretch for msg_dim 2
elif msg_dim == 3:
    height_ratios = [0.1, 1, 0.1]  # Slightly more vertical stretch for msg_dim 3
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


t = lambda _: _  # tqdm
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

    if sim == 'spring':
        force_fnc = lambda msg: -(msg.bd.values - 1)[:, None] * np.array(msg[pos_cols]) / msg.bd.values[:, None]
    elif sim == 'charge':
        force_fnc = lambda msg: (msg.q1.values[:, None] * msg.q2.values[:, None]) * np.array(msg[pos_cols]) / (msg.bd.values[:, None]**3)
    elif sim == 'r1':
        force_fnc = lambda msg: -(msg.m1.values[:, None] * msg.m2.values[:, None] * np.array(msg[pos_cols]) ) / (msg.bd.values[:, None]**2)

    expected_forces = force_fnc(msgs)

    def percentile_sum(x):
        x = x.ravel()
        bot = x.min()
        top = np.percentile(x, 90)
        msk = (x >= bot) & (x <= top)
        frac_good = (msk).sum() / len(x)
        return x[msk].sum() / frac_good

    from scipy.optimize import minimize

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

    def linear_transformation_3d(alpha):
        lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2] * expected_forces[:, 2]) + alpha[3]
        lincomb2 = (alpha[0 + 4] * expected_forces[:, 0] + alpha[1 + 4] * expected_forces[:, 1] + alpha[2 + 4] * expected_forces[:, 2]) + alpha[3 + 4]
        lincomb3 = (alpha[0 + 8] * expected_forces[:, 0] + alpha[1 + 8] * expected_forces[:, 1] + alpha[2 + 8] * expected_forces[:, 2]) + alpha[3 + 8]

        score = (
                percentile_sum(np.square(msgs_to_compare[:, 0] - lincomb1)) +
                percentile_sum(np.square(msgs_to_compare[:, 1] - lincomb2)) +
                percentile_sum(np.square(msgs_to_compare[:, 2] - lincomb3))
        ) / 3.0

        return score

    def out_linear_transformation_3d(alpha):
        lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2] * expected_forces[:, 2]) + alpha[3]
        lincomb2 = (alpha[0 + 4] * expected_forces[:, 0] + alpha[1 + 4] * expected_forces[:, 1] + alpha[2 + 4] * expected_forces[:, 2]) + alpha[3 + 4]
        lincomb3 = (alpha[0 + 8] * expected_forces[:, 0] + alpha[1 + 8] * expected_forces[:, 1] + alpha[2 + 8] * expected_forces[:, 2]) + alpha[3 + 8]

        return lincomb1, lincomb2, lincomb3

    if dim == 2:
        min_result = minimize(linear_transformation_2d, np.ones(dim ** 2 + dim), method='Powell')
    if dim == 3:
        min_result = minimize(linear_transformation_3d, np.ones(dim ** 2 + dim), method='Powell')

    for j in range(dim):
        if dim == 3:
            px = out_linear_transformation_3d(min_result.x)[j]
        else:
            px = out_linear_transformation_2d(min_result.x)[j]

        py = msgs_to_compare[:, j]
        ax = ax1 if j == 0 else ax2
        ax.scatter(px, py, alpha=0.1, s=0.1, color='k')
        ax.set_xlabel('Linear combination of forces')
        ax.set_ylabel(f'Message Element {j+1}')

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
        ax3.text(k + 0.5, -0.5, f'{sorted_msg_importance[k]:.2f}', ha='center', va='top', fontsize=8)

    plt.tight_layout()
    

    cam.snap()

ani = cam.animate()
HTML(ani.to_jshtml())

# Save the video to a file
html_content = ani.to_jshtml()

# Specify the directory and file name
directory = 'data'
file_name = 'animation.html'

# Combine directory and file name to get the full file path
file_path = os.path.join(directory, file_name)

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Write the HTML content to the file
with open(file_path, 'w') as file:
    file.write(html_content)

# Print confirmation message
print(f"Animation saved to: {file_path}")
