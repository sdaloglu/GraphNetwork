import subprocess

# Define the array of titles for different simulations
titles = ["spring_n=4_dim=2", "charge_n=4_dim=2", "r1_n=4_dim=2", "r2_n=4_dim=2"]

# Loop through the titles and run the training script for each simulation
for title in titles:
    print(f"Running training for {title}")
    subprocess.run(f"python training_kl.py --data {title}", shell=True)
    
    # Maybe put the my_analyze.py run after each training script
    subprocess.run(f"python my_analyze_kl.py --data {title}", shell=True)