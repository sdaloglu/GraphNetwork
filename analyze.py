# This file contains the analysis of the learned messages over time 
# with respect to the true force of the simulated system

import pickle as pkl

# Let's plot the scatter plot of the messages and the true force


# Load messages_over_time pkl file
messages_over_time = pkl.load(open('data/messages_over_time.pkl', 'rb'))

# Load model pkl file
recorded_models = pkl.load(open('data/models_over_time.pkl', 'rb'))



