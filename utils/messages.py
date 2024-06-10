# This file contains the function to organize and record the learned messages over time

import numpy as np
import sys
import pandas as pd
sys.path.append('utils')    # Add the utils directory to the path
from my_models import loss_function, GN
import torch 

# We want to check if the learned model has messages that correspond to the true force of the simulated system

# Define a function to extract the messages from the model

def get_messages(model, test_loader, msg_dim, dim = 2):
    """
    Args:
        model (torch.nn.Module): The model to extract the messages from

        
    Returns:
        messages (np.array): The messages extracted from the model
    """
    
    # Move the model to the CPU
    #model.cpu()
    
    # Initialize an empty list to store the messages
    messages = []
    
    for batch in test_loader:
        
        # Extract the node features of the source nodes
        x_source = batch.x[batch.edge_index[0]]   # We want the graph connectivity info for all graphs in the batch, hence the batch.edge_index[0]
        
        # Extract the node features of the target nodes
        x_target = batch.x[batch.edge_index[1]] 
        
        # Get the messages
        if msg_dim == 200:    # KL regularization
            message = model.edge_model(torch.cat([x_source, x_target], dim = 1))
            mean = message[:,:100]
            message = mean
            
        else:    # L1 regularization, bottleneck, or no regularization (Standard)
            message = model.edge_model(torch.cat([x_source, x_target], dim = 1))
        
        
        # Append the node features to the messages list
        message_with_node_features = torch.cat((x_source,x_target,message), dim = 1)
        
        
        if msg_dim == 200: # KL regularization
            if dim == 2:
                columns = [elem%(k) for k in range(1, 3) for elem in 'x%d y%d vx%d vy%d q%d m%d'.split(' ')]
                columns += ['e%d'%(k,) for k in range(100)]
            elif dim == 3:
                columns = [elem%(k) for k in range(1, 3) for elem in 'x%d y%d z%d vx%d vy%d vz%d q%d m%d'.split(' ')]
                columns += ['e%d'%(k,) for k in range(100)]
        else:
            if dim == 2:
                columns = [elem%(k) for k in range(1, 3) for elem in 'x%d y%d vx%d vy%d q%d m%d'.split(' ')]
                columns += ['e%d'%(k,) for k in range(msg_dim)]
            elif dim == 3:
                columns = [elem%(k) for k in range(1, 3) for elem in 'x%d y%d z%d vx%d vy%d vz%d q%d m%d'.split(' ')]
                columns += ['e%d'%(k,) for k in range(msg_dim)]

        
        # List of messages -- Pandas dataframe
        messages.append(pd.DataFrame(
            data=message_with_node_features.cpu().detach().numpy(),
            columns=columns
        ))
        
    messages = pd.concat(messages)
    
    # Adding the extra information to the messages dataframe --> delta x and delta y and r
    # These information are used for calculating the true force
    messages['dx'] = messages.x1 - messages.x2
    messages['dy'] = messages.y1 - messages.y2
    
    if dim == 2:    
        messages['r'] = np.sqrt(
            (messages.dx)**2 + (messages.dy)**2
        )
    elif dim == 3:
        # Add the third dimension delta z
        messages['dz'] = messages.z1 - messages.z2
        messages['r'] = np.sqrt(
            (messages.dx)**2 + (messages.dy)**2 + (messages.dz)**2
        )
    
    return messages

        

        
        
        
        
        