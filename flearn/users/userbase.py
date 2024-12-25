import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, id, train_data, test_data, model, batch_size = 0, learning_rate = 0, hyper_learning_rate = 0 , L = 0, local_epochs = 0):
        # from fedprox
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        if(batch_size == 0):
            self.batch_size = len(train_data)
        else:
            self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hyper_learning_rate = hyper_learning_rate
        self.L = L
        self.local_epochs = local_epochs
        # Create dataloaders with custom collate function for GNN data
        collate_fn = lambda data_list: data_list[0]  # For GNN, we process one graph at a time
        self.trainloader = DataLoader(train_data, self.batch_size, collate_fn=collate_fn)
        self.testloader = DataLoader(test_data, self.batch_size, collate_fn=collate_fn)
        self.testloaderfull = DataLoader(test_data, self.test_samples, collate_fn=collate_fn)
        self.trainloaderfull = DataLoader(train_data, self.train_samples, collate_fn=collate_fn)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # those parameters are for FEDL.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.server_grad    = copy.deepcopy(list(self.model.parameters()))
        self.pre_local_grad = copy.deepcopy(list(self.model.parameters()))

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
            if(new_param.grad != None):
                if(old_param.grad == None):
                    old_param.grad = torch.zeros_like(new_param.grad)

                if(local_param.grad == None):
                    local_param.grad = torch.zeros_like(new_param.grad)

                old_param.grad.data = new_param.grad.data.clone()
                local_param.grad.data = new_param.grad.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
            if(param.grad != None):
                if(clone_param.grad == None):
                    clone_param.grad = torch.zeros_like(param.grad)
                clone_param.grad.data = param.grad.data.clone()
                
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()
            param.grad.data = new_param.grad.data.clone()

    def get_grads(self, grads):
        self.optimizer.zero_grad()
        
        for data in self.trainloaderfull:
            # Get probability distribution from model output tuple
            output_tuple = self.model(data.x, data.edge_index)
            prob_dist = output_tuple[0] if isinstance(output_tuple, tuple) else output_tuple
            
            # Ensure output and target shapes match before loss calculation
            if prob_dist.shape != data.y.shape:
                # Handle case where output is [1, N] shaped
                if prob_dist.dim() == 2:
                    if prob_dist.size(0) == 1:
                        prob_dist = prob_dist.squeeze(0)  # Remove batch dimension
                    else:
                        prob_dist = prob_dist.transpose(0, 1)  # Transpose if needed
                        prob_dist = prob_dist.reshape(-1)  # Flatten to 1D
                
                # Ensure both tensors have same size
                min_size = min(prob_dist.size(0), data.y.size(0))
                prob_dist = prob_dist[:min_size]
                data.y = data.y[:min_size]
            
            loss = F.mse_loss(prob_dist, data.y)
            loss.backward()
            
        self.clone_model_paramenter(self.model.parameters(), grads)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        total_samples = 0
        for data in self.testloaderfull:
            # Get probability distribution from model output tuple
            output_tuple = self.model(data.x, data.edge_index)
            prob_dist = output_tuple[0] if isinstance(output_tuple, tuple) else output_tuple
            
            # Debug tensor shapes
            print(f"[Debug] Test - Output shape: {prob_dist.shape}, Target shape: {data.y.shape}")
            
            # Ensure output and target shapes match before comparison
            if prob_dist.shape != data.y.shape:
                # Handle case where output is [1, N] shaped
                if prob_dist.dim() == 2:
                    if prob_dist.size(0) == 1:
                        prob_dist = prob_dist.squeeze(0)  # Remove batch dimension
                    else:
                        prob_dist = prob_dist.transpose(0, 1)  # Transpose if needed
                        prob_dist = prob_dist.reshape(-1)  # Flatten to 1D
                
                # Ensure both tensors have same size
                min_size = min(prob_dist.size(0), data.y.size(0))
                prob_dist = prob_dist[:min_size]
                data.y = data.y[:min_size]
                    
            # Verify shapes match after adjustments
            if prob_dist.shape != data.y.shape:
                raise ValueError(f"Shape mismatch after adjustments: output {prob_dist.shape} vs target {data.y.shape}")
                
            # Compare only the probability distribution output with target
            test_acc += (torch.sum(torch.abs(prob_dist - data.y) < 0.1)).item()  # Within 10% threshold
            total_samples += data.y.shape[0]
        return test_acc, total_samples

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for data in self.trainloaderfull:
            # Get probability distribution from model output tuple
            output_tuple = self.model(data.x, data.edge_index)
            prob_dist = output_tuple[0] if isinstance(output_tuple, tuple) else output_tuple
            
            # Debug tensor shapes
            print(f"[Debug] Train - Output shape: {prob_dist.shape}, Target shape: {data.y.shape}")
            
            # Ensure output and target shapes match before comparison
            if prob_dist.shape != data.y.shape:
                # Handle case where output is [1, N] shaped
                if prob_dist.dim() == 2:
                    if prob_dist.size(0) == 1:
                        prob_dist = prob_dist.squeeze(0)  # Remove batch dimension
                    else:
                        prob_dist = prob_dist.transpose(0, 1)  # Transpose if needed
                        prob_dist = prob_dist.reshape(-1)  # Flatten to 1D
                
                # Ensure both tensors have same size
                min_size = min(prob_dist.size(0), data.y.size(0))
                prob_dist = prob_dist[:min_size]
                data.y = data.y[:min_size]
                    
            # Verify shapes match after adjustments
            if prob_dist.shape != data.y.shape:
                raise ValueError(f"Shape mismatch after adjustments: output {prob_dist.shape} vs target {data.y.shape}")
                
            # Compare only the probability distribution output with target
            train_acc += (torch.sum(torch.abs(prob_dist - data.y) < 0.1)).item()  # Within 10% threshold
            loss += F.mse_loss(prob_dist, data.y)
        return train_acc, loss.item(), self.train_samples
    
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for personalizing
            data = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            data = next(self.iter_trainloader)
        return data
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            data = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            data = next(self.iter_testloader)
        return data

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
