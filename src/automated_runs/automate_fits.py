import subprocess
import os

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the directory above the current script
parent_dir = os.path.dirname(current_script_dir)

# Define the array of titles for different simulations
titles = ["spring_n=4_dim=2", "charge_n=4_dim=2", "r1_n=4_dim=2", "r2_n=4_dim=2"]
regularizers = ["linear_l1", "bottleneck", "kl", "standard"]


# Loop through the titles and run the training script for each simulation
for title in titles:
    for regularizer in regularizers:
        print(f"Running symbolic fitting for {title} with {regularizer} regularizer")
        script_path = os.path.join(parent_dir, "symbolic_fit.py")
        subprocess.run(f"python {script_path} --data {title} --regularizer {regularizer}", shell=True)