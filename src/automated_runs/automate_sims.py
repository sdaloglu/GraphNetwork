import subprocess
import os

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the directory above the current script
parent_dir = os.path.dirname(current_script_dir)

# Define the array of simulations
simulations = ["spring", "charge", "r1", "r2"]


# Loop through the titles and run the training script for each simulation
for sim in simulations:
    print(f"Running simulation for {sim}")
    script_path = os.path.join(parent_dir, "simulate_systems.py")
    subprocess.run(f"python {script_path} --sim {sim}", shell=True)