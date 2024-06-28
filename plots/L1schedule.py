import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os

x = np.linspace(1, 50, 50)
y_constant = np.ones(50) * 0.01

def compute_y_linear(x):
    return x * (0.1 - 0.01) / 50 + 0.01

def compute_y_triangle(x):
    midpoint = 25
    max_value = 0.1
    
    x_increase = x[:midpoint]
    x_decrease = x[midpoint:]
    y_increase = x_increase * (max_value - 0.01) / midpoint + 0.01
    y_decrease = (50 - x_decrease) * (max_value - 0.01) / (50 - midpoint) + 0.01
    y_triangle = np.concatenate((y_increase, y_decrease))
    return y_triangle

y_linear = compute_y_linear(x)
y_triangle = compute_y_triangle(x)

plt.figure(figsize=(14, 10), tight_layout=True)
plt.plot(x, y_constant, label='Constant L1 = 0.01', linewidth=2.5)
plt.plot(x, y_linear, label='Linear L1 Schedule', linewidth=2.5)
plt.plot(x, y_triangle, label='Triangle L1 Schedule', linewidth=2.5)
plt.axhline(y=0.1, color='k', linestyle='--', linewidth=2.5)
plt.text(22, 0.105, 'Max L1 = 0.1', verticalalignment='center', color='k', fontsize=24)

plt.xlabel('Epochs', fontsize=24)
plt.ylabel('L1 coefficient', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.ylim(0.01, 0.12)  # Set the y-axis limits to the specified minimum and maximum
plt.yticks(list(plt.yticks()[0]) + [0.01], labels=[f'{"{:.2f}".format(i)}' if i != 0.01 else r'$\mathbf{0.01}$' for i in plt.yticks()[0]] + [r'$\mathbf{0.01}$'])

# Adding vertical and horizontal grid lines
plt.grid(True, which='both', linestyle='--', linewidth=1)

plt.legend(bbox_to_anchor=(-0.008, 1.01), loc='upper left', fontsize=21)
script_dir = os.path.dirname(__file__)
plt.savefig(os.path.join(script_dir, 'L1_schedule.png'), bbox_inches='tight')
### Extract the final test losses from pruned_messages and print on terminal ###

# titles = ["spring_n=4_dim=2", "charge_n=4_dim=2", "r1_n=4_dim=2", "r2_n=4_dim=2"]
# regularizers = ["l1","triangle_l1","linear_l1", "bottleneck", "kl", "standard"]

# for title in titles:
#     for regularizer in regularizers:
#         messages_over_time = pkl.load(open(f"models/{title[:2]}/pruned_messages_{title}_{regularizer}.pkl", "rb"))
#         losses_test = [x['test_loss'][:1][0] for x in messages_over_time]
#         print(f"{title}_{regularizer}: {losses_test[-1]}")
