import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing    # Message Passing Graph Neural Network (what we are using)
from torch.nn import ReLU

class GN(MessagePassing):
    # Using the MessagePassing base class from PyTorch Geometric
    def __init__(self, edge_index, message_dim, input_dim=6, output_dim=2, hidden_units = 100, aggregation = 'add'):
       
        # Specify the aggregation method from the temporary object of the superclass
        super(GN,self).__init__(aggr = aggregation)   # Adding forces as an inductive bias of the GN model
        self.edge_index = edge_index
        
        self.edge_model = nn.Sequential( 
            # Edge model aiming to learn the true force
            nn.Linear(2*input_dim, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, message_dim),           
           
        )   

        self.node_model = nn.Sequential(
            # Node model aiming to learn the true acceleration
            # The input is the concatenation of the aggregated messages and the input of the target node
            nn.Linear(message_dim+input_dim, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, output_dim),           
            
        )
        
    
    def message(self, x_i, x_j):
        # Compute messages from the source node to target node
        # the message function takes an input of the concatenation of the features of the two nodes
        return self.edge_model(torch.cat([x_i, x_j], dim = 1))
        
        
    # The default aggregate function is used from the superclass (summing the messages)
    
    def update(self, aggr_out, x):
        
        """First we concatenate the aggregated messages and the input of the target node, 
        then we pass it through the node model.

        Returns:
            _type_: _description_
        """
        
        return self.node_model(torch.cat([aggr_out, x], dim = 1))
        
        
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
    
    def loss(self, graph):
  
        # Compare the ground truth acceleration with the predicted acceleration (output of the node model)
        # Using MAE as the loss function
        
        y = graph.y
        x = graph.x
        edge_index = graph.edge_index
        
        # Ensure tensors are on the correct device
        y = y.to(next(self.parameters()).device)
        x = x.to(next(self.parameters()).device)
        edge_index = edge_index.to(next(self.parameters()).device)
        
        return torch.sum(torch.abs(y - self.forward(x, edge_index)))
     
    
def edge_index(n):
    """
    Creating an adjacency tensor for a fully connected graph with n nodes
    
    Args:
        n (_type_): number of nodes

    Returns:
        _type_: _description_
    """
    
    # Create an adjacency matrix for a fully connected graph, excluding self-loops
    ones = torch.ones(n,n, dtype = torch.int32)
    adjacency_matrix = ones - torch.eye(n, dtype = torch.int32)
    
    
    # Find indices of non-zero elements (edges)
    edge_index = (adjacency_matrix == 1).nonzero(as_tuple=False).t()

    # Now, edge_index is a [2, num_edges] tensor 

    
    return edge_index


def loss_function(model, graph, edge_index, n, batch_size, regularizer = 'l1'):
    """
    Loss function for the Graph Neural Network
    
    Args:
        model (_type_): Graph Neural Network model
        graph (_type_): Graph object

    Returns:
        _type_: _description_
    """

    base_loss = model.loss(graph)
    source_node = graph.x[model.edge_index[0]]
    target_node = graph.x[model.edge_index[1]]
        
    if regularizer == 'l1':
        alpha = 0.01
        
        message = model.message(target_node, source_node)
        message_reg = alpha * torch.sum(torch.abs(message)) * batch_size
        message_reg = message_reg / (n*(n-1))  # Normalizing the regularizer by dividing by the number of edges times 2 (edge_index is directed)
        return base_loss, message_reg
    
    elif regularizer == 'kl':
        alpha = 1.0
        
        message = model.message(target_node, source_node)    # Message tensor of shape [num_edges, message_dim]
        
        # Calculate the KL divergence of the message distribution
        
        mu = torch.mean(message, dim = 0)
        sigma = torch.std(message, dim = 0)
        
        
        
        # Add the KL divergence term to the loss
        
        return 0 
    
