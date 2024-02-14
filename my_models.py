import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing    # Message Passing Graph Neural Network (what we are using)
from torch.nn import ReLU, Softplus

class GN(MessagePassing):
    # Using the MessagePassing base class from PyTorch Geometric
    def __init__(self, input_dim, n_layers = 5, hidden_units = 100, message_dim=2, node_dim=6, aggregation = 'add'):
       
        # Specify the aggregation method from the temporary object of the superclass
        super(GN,self).__init__(aggr = aggregation)   # Adding forces as an inductive bias of the GN model
        
        self.edge_model = nn.Sequential( 
            # Edge model aiming to learn the true force
            nn.Linear(input_dim, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, message_dim),           
           
        )   

        self.node_model = nn.Sequential(
            # Node model aiming to learn the true acceleration
            nn.Linear(input_dim, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, node_dim),           
            
        )
        
    
    def message(self, source_node, target_node):
        # Compute messages from the source node to target node
        # the message function takes an input of the concatenation of the features of the two nodes
        return self.edge_model(torch.cat([source_node, target_node], dim = 1))
        
        
    # The default aggregate function is used from the superclass (summing the messages)
    
    def update(self, aggr_out, target_node):
        
        """First we concatenate the aggregated messages and the input of the target node, 
        then we pass it through the node model.

        Returns:
            _type_: _description_
        """
        
        return self.node_model(torch.cat([aggr_out, target_node], dim = 1))
        
        
    def forward(self, x, edge_index):
        """
        Args:
            x (_type_): node features
            edge_index (_type_): edge indices

        Returns:
            _type_: _description_
        """
        # forward pass of the neural network
        # Calling propagate() will in turn call message(), aggregate(), and update()
        # size argument is optional and can be used to specify the dimensions of the source and target node feature matrices
        return self.propagate(edge_index, x = x, size = (x.size(0),x.size(0)))