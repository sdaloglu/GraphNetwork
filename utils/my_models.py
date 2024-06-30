import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing  # Message Passing Graph Neural Network (what we are using)
from torch.nn import ReLU
from torch_geometric.nn import aggr

class GN(MessagePassing):
    """
    Graph Neural Network using the MessagePassing base class from PyTorch Geometric.
    
    Args:
        message_dim (int): Dimension of the message.
        input_dim (int, optional): Dimension of the input features. Default is 6.
        output_dim (int, optional): Dimension of the output features. Default is 2.
        hidden_units (int, optional): Number of hidden units in the neural network layers. Default is 300.
        aggregation (str, optional): Aggregation method to use. Default is 'add'.
    """
    def __init__(self, message_dim, input_dim=6, output_dim=2, hidden_units=300, aggregation='add'):
        # Specify the aggregation method from the temporary object of the superclass
        super(GN, self).__init__(aggr=aggregation)  # Adding forces as an inductive bias of the GN model

        self.message_dim = message_dim
        
        # Define the edge model
        self.edge_model = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, hidden_units),
            ReLU(),
            nn.Linear(hidden_units, message_dim),
        )

        # Define the node model based on the message dimension
        if message_dim == 200:  # KL regularization
            node_model_layers = [
                nn.Linear(int(message_dim // 2) + input_dim, hidden_units),
                ReLU(),
                nn.Linear(hidden_units, hidden_units),
                ReLU(),
                nn.Linear(hidden_units, hidden_units),
                ReLU(),
                nn.Linear(hidden_units, output_dim),
            ]
        elif message_dim in [100, 2, 3]:  # L1 regularization, Standard or Bottleneck
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
            raise ValueError("Message dimension has to be 100, 200, 2 or 3. Invalid dimension is assigned")

        self.node_model = nn.Sequential(*node_model_layers)

    def message(self, x_i, x_j):
        """
        Compute messages from the source node to the target node.
        
        Args:
            x_i (Tensor): Features of the target nodes.
            x_j (Tensor): Features of the source nodes.
        
        Returns:
            Tensor: Computed messages.
        """
        message = self.edge_model(torch.cat([x_i, x_j], dim=1))
        
        if self.message_dim == 100:  # L1 regularization
            return message
        elif self.message_dim in [2, 3]:  # Bottleneck model
            return message
        elif self.message_dim == 200:  # KL regularization
            mu = message[:, 0::2]
            log_var = message[:, 1::2]
            sample = torch.randn_like(mu).to(x_i.device) * torch.exp(0.5 * log_var) + mu
            return sample
        else:
            raise ValueError("Message dimension has to be 100, 200, 2 or 3. Invalid dimensions is assigned")

    def update(self, aggr_out, x=None):
        """
        Update node features by concatenating the aggregated messages and the input of the target node,
        then passing it through the node model.
        
        Args:
            aggr_out (Tensor): Aggregated messages.
            x (Tensor, optional): Node features. Default is None.
        
        Returns:
            Tensor: Updated node features.
        """
        if aggr_out.size(0) != x.size(0):
            raise ValueError("Number of rows in the aggregated message and node feature matrix are not the same")
        
        return self.node_model(torch.cat([x, aggr_out], dim=1))

    def forward(self, x, edge_index, system_dimension, augmentation):
        """
        Forward pass of the neural network.
        
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            system_dimension (int): Dimension of the system.
            augmentation (bool): Whether to perform data augmentation.
        
        Returns:
            Tensor: Output of the propagate function.
        """

        # Calling propagate() will in turn call message(), aggregate(), and update()
        # size argument is optional and can be used to specify the dimensions of the source and target node feature matrices

        if augmentation:
            # Perform data augmentation in training loop in real time
            # Generate random noise to add to the node features
            # Make sure the noise is the same for each node feature to simulate system noise
            noise = torch.randn(1, system_dimension) * 3
            noise = noise.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(system_dimension).to(x.device), noise)
        
        return self.propagate(edge_index=edge_index, x=x, size=(x.size(0), x.size(0)))

    def loss(self, graph, augmentation):
        """
        Compute the loss by comparing the ground truth acceleration with the predicted acceleration.
        
        Args:
            graph (Data): Graph object containing node features, edge indices, and ground truth.
            augmentation (bool): Whether to perform data augmentation.
        
        Returns:
            Tensor: Computed loss.
        """
        x = graph.x    # Node feature matrix of the batch
        edge_index = graph.edge_index    # Graph connectivity matrix of the batch
        y = graph.y    # Output - acceleration matrix of the batch
        system_dimension = y.shape[1]    # Dimension of the system
        
        return torch.sum(torch.abs(y - self.forward(x, edge_index, system_dimension, augmentation))) / y.shape[0]  # Normalize by dividing the loss by the number of nodes in the batch

def get_edge_index(n):
    """
    Create an adjacency tensor for a fully connected graph with n nodes.
    
    Args:
        n (int): Number of nodes.
    
    Returns:
        Tensor: Edge index tensor of shape [2, num_edges].
    """
    
    # Create an adjacency matrix for a fully connected graph, excluding self-loops
    ones = torch.ones(n, n, dtype=torch.int32)
    adjacency_matrix = ones - torch.eye(n, dtype=torch.int32)
    edge_index = (adjacency_matrix == 1).nonzero(as_tuple=False).t()
    # Now, edge_index is a [2, num_edges] tensor 

    
    return edge_index

def loss_function(model, graph, augmentation, regularizer, l1_alpha):
    """
    Loss function for the Graph Neural Network.
    
    Args:
        model (GN): Graph Neural Network model.
        graph (Data): Graph object containing node features, edge indices, and ground truth.
        augmentation (bool): Whether to perform data augmentation.
        regularizer (str): Type of regularizer ('l1' or 'kl').
        l1_alpha (float): Regularization coefficient for L1 regularization.
    
    Returns:
        tuple: Base loss and regularization loss.
    """
    base_loss = model.loss(graph, augmentation=augmentation)
    source_node = graph.x[graph.edge_index[0]]
    target_node = graph.x[graph.edge_index[1]]
        
    if regularizer == 'l1' or regularizer == 'linear_l1' or regularizer == 'triangle_l1':
        message = model.message(source_node, target_node)
        message_reg = l1_alpha * torch.sum(torch.abs(message))
        message_reg_normalized = message_reg / message.shape[0]    # Normalizing the regularizer by the number of edges in the batch
        return base_loss, message_reg_normalized
    elif regularizer == 'kl':
        alpha = 1.0
        message = model.edge_model(torch.cat([source_node, target_node], dim=1))
        mu = message[:, 0::2]
        log_var = message[:, 1::2]
        kl_reg = alpha * torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var))
        kl_reg = kl_reg / message.shape[0]    # Normalizing the regularizer by dividing by the number of edges in the batch
        
        return base_loss, kl_reg
    else:
        return base_loss

def update_l1_alpha_linear(current_step, total_steps, base_alpha, max_alpha):
    """
    Adjust l1_alpha to continuously increase throughout the training process.
    This might be a better option for small number of epochs
    Args:
        current_step (int): Current training step.
        total_steps (int): Total number of training steps.
        base_alpha (float): Initial value of l1_alpha.
        max_alpha (float): Maximum value of l1_alpha.
    
    Returns:
        float: Updated l1_alpha value.
    """
    return base_alpha + (max_alpha - base_alpha) * (current_step / total_steps)

def update_l1_alpha_triangle(current_step, total_steps, base_alpha, max_alpha):
    """
    Adjust l1_alpha to first increase and then decrease throughout the training process.
    This might be a better option for larger number of epochs
    Args:
        current_step (int): Current training step.
        total_steps (int): Total number of training steps.
        base_alpha (float): Initial value of l1_alpha.
        max_alpha (float): Maximum value of l1_alpha.
    
    Returns:
        float: Updated l1_alpha value.
    """
    midpoint = total_steps / 2
    if current_step <= midpoint:
        return base_alpha + (max_alpha - base_alpha) * (current_step / midpoint)
    else:
        return max_alpha - (max_alpha - base_alpha) * ((current_step - midpoint) / midpoint)