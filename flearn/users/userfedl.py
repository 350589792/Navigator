import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from flearn.users.userbase import User
from flearn.optimizers.fedoptimizer import *
from flearn.models.models import FedUAVGNN
from flearn.data.data_load_new import create_federated_data
import copy
# Implementation for FedAvg clients with UAV-GNN support

class UserFEDL(User):
    def __init__(self, numeric_id, train_data, test_data, model_config, batch_size, learning_rate, hyper_learning_rate, L,
                 local_epochs, optimizer, hidden_dim=128):
        # Initialize model configuration
        self.model_config = model_config
        if not hasattr(self.model_config, 'hidden_dim'):
            self.model_config.hidden_dim = hidden_dim
        if not hasattr(self.model_config, 'device'):
            self.model_config.device = 'cpu'
            
        # Initialize FedUAVGNN model
        self.model = FedUAVGNN(self.model_config)
        
        # Store dataset sizes
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        
        super().__init__(numeric_id, train_data, test_data, self.model, batch_size, learning_rate, hyper_learning_rate, L,
                         local_epochs)

        # Using MSE loss for regression task
        self.loss = F.mse_loss
        
        # Initialize optimizer
        self.optimizer = FEDLOptimizer(self.model.parameters(), lr=self.learning_rate, hyper_lr=hyper_learning_rate, L=L)
        
        # Store UAV network configuration
        self.model_config = model_config
    
    def get_full_grad(self):
        """Compute full gradient for the UAV-GNN model."""
        for data in self.trainloaderfull:
            self.model.zero_grad()
            prob_dist, _, _, _ = self.model(data.x, data.edge_index)
            
            # Ensure output and target shapes match
            if prob_dist.shape != data.y.shape:
                if prob_dist.dim() == 2:
                    if prob_dist.size(0) == 1:
                        prob_dist = prob_dist.squeeze(0)
                    else:
                        prob_dist = prob_dist.transpose(0, 1)
                        prob_dist = prob_dist.reshape(-1)
                
                min_size = min(prob_dist.size(0), data.y.size(0))
                prob_dist = prob_dist[:min_size]
                data.y = data.y[:min_size]
            
            loss = F.mse_loss(prob_dist, data.y)
            loss.backward()

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]


    def prepare_data(self, features, edge_index, targets):
        """Prepare UAV network data for training."""
        dataset = TensorDataset(features, edge_index, targets)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, epochs):
        """Train the UAV-GNN model locally."""
        self.clone_model_paramenter(self.model.parameters(), self.server_grad)
        self.get_grads(self.pre_local_grad)
        self.model.train()
        
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for epoch in range(1, self.local_epochs + 1):
            epoch_loss = 0
            for batch_idx, data in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                
                # Forward pass through GNN
                prob_dist, h, c, latent = self.model(data.x, data.edge_index)
                
                # Ensure output and target shapes match
                if prob_dist.shape != data.y.shape:
                    if prob_dist.dim() == 2:
                        if prob_dist.size(0) == 1:
                            prob_dist = prob_dist.squeeze(0)
                        else:
                            prob_dist = prob_dist.transpose(0, 1)
                            prob_dist = prob_dist.reshape(-1)
                    
                    min_size = min(prob_dist.size(0), data.y.size(0))
                    prob_dist = prob_dist[:min_size]
                    data.y = data.y[:min_size]
                
                # Compute loss using MSE
                loss = F.mse_loss(prob_dist, data.y)
                epoch_loss += loss.item()
                
                # Calculate accuracy (within 10% threshold)
                correct += (torch.abs(prob_dist - data.y) < 0.1).sum().item()
                total_samples += data.y.size(0)
                
                # Backward pass
                loss.backward()
                self.optimizer.step(self.server_grad, self.pre_local_grad)
            
            total_loss += epoch_loss
        
        # Store metrics
        self.train_loss = total_loss / (self.local_epochs * len(self.trainloader))
        self.train_accuracy = correct / total_samples
        
        # Evaluate on test set
        self.model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data in self.testloader:
                prob_dist, _, _, _ = self.model(data.x, data.edge_index)
                
                # Ensure output and target shapes match
                if prob_dist.shape != data.y.shape:
                    if prob_dist.dim() == 2:
                        if prob_dist.size(0) == 1:
                            prob_dist = prob_dist.squeeze(0)
                        else:
                            prob_dist = prob_dist.transpose(0, 1)
                            prob_dist = prob_dist.reshape(-1)
                    
                    min_size = min(prob_dist.size(0), data.y.size(0))
                    prob_dist = prob_dist[:min_size]
                    data.y = data.y[:min_size]
                
                test_loss += F.mse_loss(prob_dist, data.y).item()
                test_correct += (torch.abs(prob_dist - data.y) < 0.1).sum().item()
                test_total += data.y.size(0)
        
        self.test_loss = test_loss / len(self.testloader)
        self.test_accuracy = test_correct / test_total
        
        self.optimizer.zero_grad()
        self.get_full_grad()
        return self.train_loss

