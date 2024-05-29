import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing    # Message Passing Graph Neural Network (what we are using)
from torch.nn import ReLU
from torch_geometric.nn import aggr

class GN(MessagePassing):
    # Using the MessagePassing base class from PyTorch Geometric
    def __init__(self, message_dim, input_dim=6, output_dim=2, hidden_units = 300, aggregation = 'add'):
       
        # Specify the aggregation method from the temporary object of the superclass
        super(GN,self).__init__(aggr = aggregation)   # Adding forces as an inductive bias of the GN model

        self.message_dim = message_dim
        
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

        if message_dim == 200:  # Meaning KL regularization is used
            node_model_layers = [
                nn.Linear(int(message_dim // 2) + input_dim, hidden_units),
                ReLU(),
                nn.Linear(hidden_units, hidden_units),
                ReLU(),
                nn.Linear(hidden_units, hidden_units),
                ReLU(),
                nn.Linear(hidden_units, output_dim),
            ]
        elif message_dim == 100 or message_dim == 2 or message_dim == 3:    # message_dim == 100 for L1 and standard, or message_dim == 2/3 for Bottleneck
            node_model_layers = [
                nn.Linear(message_dim + input_dim, hidden_units),
                ReLU(),
                nn.Linear(hidden_units, hidden_units),
                ReLU(),
                nn.Linear(hidden_units, hidden_units),
                ReLU(),
                nn.Linear(hidden_units, output_dim),
            ]
            
        else:
            raise ValueError("Message dimension has to be 100 or 200. Invalid dimension is assigned")

        self.node_model = nn.Sequential(*node_model_layers)
        

        
    def message(self, x_i, x_j):
        
        # Compute messages from the source node to target node
        message = self.edge_model(torch.cat([x_i, x_j], dim = 1))
        
        if self.message_dim == 100:    # When we use L1 regularization, message with 100 dimensions is returned
            return message
        
        elif self.message_dim == 2 or self.message_dim == 3:  # When we use bottleneck model, message with 2/3 dimensions is returned
            return message
        
        
        elif self.message_dim == 200:    # When we use KL regularization, sample from the predicted distributions is returned
            
            # Take the first half of the features as the mean and the second half as the variance
            mean = message[:,:100]
            log_variance = message[:,100:]
            
            # Sample from the predicted distribution
            sample = torch.randn_like(mean).to(x_i.device) * torch.exp(0.5*log_variance) + mean
            
            return sample
        
        else:
            raise ValueError("Message dimension has to be 100, 200, 2 or 3. Invalid dimensions is assigned")
                
    

    # The default aggregate function is used from the superclass (summing the messages)
    
    def update(self, aggr_out, x):
        
        """First we concatenate the aggregated messages and the input of the target node, 
        then we pass it through the node model.

        Returns:
            _type_: _description_
        """
        # aggr_ouput is the added message outputs for each node, thus has the shape [n_particles, message_dim]
        
        if aggr_out.size(0) != x.size(0):    # This should correspond to the number of nodes in the batch
            raise ValueError("Number of rows in the aggregated message and node feature matrix are not the same")
        

        
        return self.node_model(torch.cat([aggr_out, x], dim = 1))
        

    def forward(self, x, edge_index, system_dimension, augmentation):
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

        x = x
        if augmentation:
            # Perform data augmentation in training loop in real time
            
            # Generate random noise to add to the node features
            # Make sure the noise is the same for each node feature to simulate system noise
            noise = torch.randn(1, system_dimension) * 3
            noise = noise.repeat(x.size(0),1).to(x.device)
            x = x.index_add(1, torch.arange(system_dimension).to(x.device), noise)
            
        return self.propagate(edge_index=edge_index, x = x, size = (x.size(0),x.size(0)))
   
    
    def loss(self, graph, augmentation):
  
        # Compare the ground truth acceleration with the predicted acceleration (output of the node model)
        # Using MAE as the loss function
        
        x = graph.x    # Node feature matrix of the batch
        edge_index = graph.edge_index    # Graph connectivity matrix of the batch
        y = graph.y    # Output - acceleration matrix of the batch
        system_dimension = y.shape[1]    # Dimension of the system
        
        return torch.sum(torch.abs(y - self.forward(x, edge_index, system_dimension, augmentation)))/y.shape[0]    # Normalize by dividing the loss by the number of nodes in the batch
    
    
    
    
def get_edge_index(n):
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



def loss_function(model, graph, augmentation, regularizer):
    """
    Loss function for the Graph Neural Network
    
    Args:
        model (_type_): Graph Neural Network model
        graph (_type_): Graph object

    Returns:
        _type_: _description_
    """

    base_loss = model.loss(graph, augmentation = augmentation)
    source_node = graph.x[graph.edge_index[0]]
    target_node = graph.x[graph.edge_index[1]]
        
    if regularizer == 'l1':
        alpha = 1e-2   
        
        message = model.message(source_node, target_node)
        
        message_reg = alpha * torch.sum(torch.abs(message))    # Multiply by the regularizer coefficient
        message_reg_normalized = message_reg / message.shape[0]  # Normalizing the regularizer by the number of edges in the batch
        return base_loss, message_reg_normalized
    
    elif regularizer == 'kl':
        alpha = 1.0
        
        message = model.edge_model(torch.cat([source_node, target_node],dim=1))   # Message tensor of shape [num_edges, message_dim]
        
        # Calculate the KL divergence of the message distribution
        
        mu = message[:,:100]    # Take the first half of the features as the mean of the message distribution
        log_var = message[:,100:]   # Take the second half of the features as the log variance of the message distribution
        
        kl_reg = alpha * torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var - 1))
        kl_reg = kl_reg / message.shape[0]    # Normalizing the regularizer by dividing by the number of edges in the batch
        
        return base_loss, kl_reg
    
    else:
        return base_loss
    
