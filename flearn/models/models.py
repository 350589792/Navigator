import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from flearn.models.rgnn.rgnn import RGNN
from flearn.models.rgnn.data_load import create_federated_data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class Linear_Regression(nn.Module):
    def __init__(self, input_dim = 60, output_dim = 1):
        super(Linear_Regression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class FedUAVGNN(nn.Module):
    """
    Wrapper class for RGNN model to support federated learning with UAV networks.
    Handles proper initialization and input/output adaptation for federated clients.
    """
    def __init__(self, model_config):
        super(FedUAVGNN, self).__init__()
        self.n_hidden = model_config.hidden_dim
        self.n_feature = 2  # UAV position coordinates (x, y)
        self.device = model_config.device if hasattr(model_config, 'device') else 'cpu'
        self.gnn_core = RGNN(n_feature=self.n_feature, n_hidden=self.n_hidden)
        self.to(self.device)
        
    def forward(self, x, edge_index):
        """
        Adapts input data format for RGNN model and processes it.
        
        Args:
            x: Node features tensor (n_nodes, n_features)
            edge_index: Graph connectivity in COO format (2, n_edges)
        
        Returns:
            Tuple of (prob_dist, hidden_state, cell_state, latent)
                prob_dist: Probability distribution over next nodes
                hidden_state: LSTM hidden state
                cell_state: LSTM cell state
                latent: Latent pointer vector for next layer
        """
        # Ensure input tensors are properly formatted
        if x.dim() == 2:
            # Current node features (first node in batch)
            x_current = x[0].unsqueeze(0)  # (1, n_feature)
            
            # All node features reshaped for batch
            X_all = x.unsqueeze(0)  # (1, n_nodes, n_feature)
        else:
            # Handle case where input is already batched
            x_current = x[:, 0].unsqueeze(1)  # (batch_size, 1, n_feature)
            X_all = x
        
        # Create mask for unvisited nodes (all False initially)
        mask = torch.zeros(X_all.size(0), X_all.size(1), device=x.device)
        
        # Process through RGNN
        prob_dist, h, c, latent = self.gnn_core(x_current, X_all, mask)
        
        # Process through RGNN and get the tuple output
        prob_dist, h, c, latent = self.gnn_core(x_current, X_all, mask)
        
        # Access relay decisions and environmental features from RGNN
        self.relay_decisions = self.gnn_core.relay_decisions
        self.environmental_features = self.gnn_core.environmental_features
        
        return prob_dist, h, c, latent
