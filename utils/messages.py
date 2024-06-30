"""
This file contains the function to organize and record the learned messages over time.
"""

import numpy as np
import sys
import pandas as pd
import torch
from my_models import loss_function, GN

# Add the utils directory to the path
sys.path.append('utils')

def get_messages(model, test_loader, msg_dim, dim=2):
    """
    Extracts messages from the model.

    Args:
        model (torch.nn.Module): The model to extract the messages from.
        test_loader (DataLoader): DataLoader for the test dataset.
        msg_dim (int): Dimension of the messages.
        dim (int, optional): Dimensionality of the data (2 or 3). Defaults to 2.

    Returns:
        pd.DataFrame: The messages extracted from the model.
    """
    
    model.eval()
    
    # Initialize an empty list to store the messages
    messages = []
    
    for batch in test_loader:
        
        # Extract the node features of the source nodes
        x_source = batch.x[batch.edge_index[0]]  # Graph connectivity info for all graphs in the batch
        
        # Extract the node features of the target nodes
        x_target = batch.x[batch.edge_index[1]] 
        
        # Get the messages
        if msg_dim == 200:  # KL regularization
            message_mu_log_var = model.edge_model(torch.cat([x_source, x_target], dim=1))
            mu = message_mu_log_var[:, 0::2]
            log_var = message_mu_log_var[:, 1::2]
            message = torch.cat([mu, log_var], dim=1)
        else:  # L1 regularization, bottleneck, or no regularization (Standard)
            message = model.edge_model(torch.cat([x_source, x_target], dim=1))
        
        # Append the node features to the messages list
        message_with_node_features = torch.cat((x_source, x_target, message), dim=1)
        
        # Define columns based on dimensionality and message dimension
        if msg_dim == 200:  # KL regularization
            if dim == 2:
                columns = [f'{elem}{k}' for k in range(1, 3) for elem in 'x y vx vy q m'.split()]
                columns += [f'e{k}' for k in range(100)]
                columns += [f'log_var{k}' for k in range(100)]
        else:
            if dim == 2:
                columns = [f'{elem}{k}' for k in range(1, 3) for elem in 'x y vx vy q m'.split()]
                columns += [f'e{k}' for k in range(msg_dim)]
        
        # Append the messages to the list as a Pandas DataFrame
        messages.append(pd.DataFrame(
            data=message_with_node_features.cpu().detach().numpy(),
            columns=columns
        ))
        
    # Concatenate all message DataFrames
    messages = pd.concat(messages)
    
    # Adding extra information to the messages DataFrame: delta x, delta y, and r
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
    
    model.train()
    
    return messages
