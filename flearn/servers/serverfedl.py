import torch
import os
import time
import numpy as np
import psutil
from collections import OrderedDict

from flearn.users.userfedl import UserFEDL
from flearn.utils.metrics_logger import MetricsLogger
from flearn.servers.serverbase import Server
from flearn.utils.model_utils import read_data, read_user_data
from flearn.models.models import FedUAVGNN
from flearn.data.data_load_new import create_federated_data

# Implementation for FedAvg Server with UAV-GNN support

class FEDL(Server):
    def __init__(self, dataset, algorithm, model_config, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters,
                 local_epochs, optimizer, num_users, rho, times, hidden_dim=128, use_relay=True, use_resource_opt=False):
        # Store model configuration
        self.model_config = model_config
        self.use_relay = use_relay
        self.use_resource_opt = use_resource_opt
        
        # Initialize global GNN model with configuration
        self.global_model = FedUAVGNN(model_config)
        super().__init__(dataset, algorithm, self.global_model, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters,
                         local_epochs, optimizer, num_users, rho, times)
        
        # Initialize metrics tracking
        self.rs_train_loss = []
        self.rs_train_acc = []
        self.rs_test_loss = []
        self.rs_test_acc = []
        self.communication_overhead = []
        self.training_times = []
        
        # Initialize metrics logger
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_logger = MetricsLogger(log_dir, f'uav_fed_{time.strftime("%Y%m%d_%H%M%S")}')

        # Create UAV network data
        self.model_config = model_config
        from flearn.data.gnn_dataset import GNNDataset
        
        features, edge_indices, targets = create_federated_data(
            n_users=model_config.n_users_medium,  # Default to medium size
            n_uavs=model_config.n_uavs_medium
        )
        
        # Create GNN datasets
        train_data = GNNDataset(features, edge_indices, targets)
        test_data = GNNDataset(features, edge_indices, targets)  # Using same data for testing in this example
        
        # Initialize users with UAV-specific data partitions
        total_users = model_config.n_uavs_medium
        for i in range(total_users):
            # Create user-specific GNN datasets
            user_train_data = GNNDataset([features[i]], [edge_indices[i]], [targets[i]])
            user_test_data = GNNDataset([features[i]], [edge_indices[i]], [targets[i]])  # Using same data for testing
            
            # Determine if this UAV should be an aggregator-forward device
            # For medium/large networks: make every 3rd UAV an AF device
            is_af = (total_users >= 5) and (i % 3 == 0)
            
            # Initialize user with GNN model configuration
            user = UserFEDL(i, user_train_data, user_test_data, model_config, batch_size, learning_rate, 
                           hyper_learning_rate, L, local_epochs, optimizer, hidden_dim, is_af_device=is_af)
            
            if is_af:
                print(f"UAV {i} designated as Aggregator-Forward device")
            
            self.users.append(user)
            self.total_train_samples += len(user_train_data)
            
        # Count AF devices
        af_count = sum(1 for user in self.users if user.is_AF_device)
        print(f"Number of UAVs / total UAVs: {num_users} / {total_users}")
        print(f"Number of Aggregator-Forward UAVs: {af_count}")
        print("Finished creating FEDL server with UAV-GNN support.")

    def train(self):
        """Train the federated UAV-GNN model."""
        import time
        for glob_iter in range(self.num_glob_iters):
            print(f"-------------Round number: {glob_iter} -------------")
            self.round_start_time = time.time()

            # Distribute global model parameters to users
            self.send_parameters()
            
            # Evaluate current model performance
            self.evaluate()
            
            # Select subset of users for this round
            self.selected_users = self.select_users(glob_iter, self.num_users)
            
            # Train selected users locally
            for user in self.selected_users:
                user.train(self.local_epochs)
            
            # Aggregate user parameters with validation
            self.aggregate_parameters()
            
            # Log metrics for this round
            self.log_metrics(glob_iter)

        # Save final results and model
        self.save_results()
        self.save_model()
        
    def save_results(self):
        """Save final results and metrics."""
        # Save metadata and close logger
        self.metrics_logger.save_metadata()
        
        # Print final statistics
        print("\nFinal Results:")
        print(f"Average Training Loss: {np.mean(self.rs_train_loss):.4f}")
        print(f"Average Training Accuracy: {np.mean(self.rs_train_acc):.4f}")
        print(f"Average Test Loss: {np.mean(self.rs_test_loss):.4f}")
        print(f"Average Test Accuracy: {np.mean(self.rs_test_acc):.4f}")
        print(f"Total Communication Overhead: {sum(self.communication_overhead)/1024/1024:.2f} MB")
        print(f"Total Training Time: {sum(self.training_times):.2f} seconds")
        
    def aggregate_parameters(self):
        """Aggregate parameters from users with support for aggregator-forward UAVs."""
        assert (len(self.users) > 0)
        
        # Initialize parameter dictionary
        param_state = OrderedDict()
        for name, param in self.global_model.state_dict().items():
            param_state[name] = torch.zeros_like(param)
            
        # First pass: Aggregate parameters at AF (Aggregator-Forward) UAVs
        af_aggregated = OrderedDict()
        af_samples = {}
        
        for user in self.selected_users:
            if user.is_AF_device:
                # Initialize aggregation state for this AF UAV
                af_aggregated[user.id] = OrderedDict()
                af_samples[user.id] = 0
                for name, param in self.global_model.state_dict().items():
                    af_aggregated[user.id][name] = torch.zeros_like(param)
        
        # Collect gradients at AF UAVs from nearby users
        for user in self.selected_users:
            if not user.is_AF_device:
                # Find nearest AF UAV if relay is enabled
                if self.use_relay:
                    nearest_af = self._find_nearest_af_uav(user)
                    if nearest_af is not None:
                        af_samples[nearest_af.id] += user.train_samples
                        for name, param in user.model.state_dict().items():
                            af_aggregated[nearest_af.id][name] += param.data * user.train_samples
                            
                            # Apply resource optimization if enabled
                            if self.use_resource_opt:
                                # Scale parameters based on compute speed and bandwidth
                                compute_factor = self.model_config.uav_compute_speed
                                bandwidth_factor = self.model_config.uav_bandwidth / 10.0  # Normalize to baseline
                                af_aggregated[nearest_af.id][name] *= (compute_factor * bandwidth_factor)
        
        # Second pass: Aggregate from AF UAVs to global model
        total_train = 0
        for user in self.selected_users:
            if user.is_AF_device:
                # Add AF's own parameters
                total_train += user.train_samples + af_samples.get(user.id, 0)
                for name, param in user.model.state_dict().items():
                    # Combine AF's parameters with aggregated ones
                    if user.id in af_aggregated:
                        combined_param = (param.data * user.train_samples + af_aggregated[user.id][name])
                        param_state[name] += combined_param
            elif not self._has_nearby_af(user):
                # Direct aggregation for users not near any AF UAV
                total_train += user.train_samples
                for name, param in user.model.state_dict().items():
                    param_state[name] += param.data * user.train_samples
                
        # Average parameters
        for name, param in self.global_model.state_dict().items():
            param_state[name] = torch.div(param_state[name], total_train)
            
        # Update global model
        self.global_model.load_state_dict(param_state, strict=True)
        
    def _find_nearest_af_uav(self, user):
        """Find the nearest aggregator-forward UAV for a given user."""
        nearest_af = None
        min_distance = float('inf')
        
        for potential_af in self.selected_users:
            if potential_af.is_AF_device:
                distance = self._calculate_distance(user, potential_af)
                if distance < min_distance:
                    min_distance = distance
                    nearest_af = potential_af
        
        return nearest_af if min_distance <= self.model_config.max_relay_distance else None
    
    def _has_nearby_af(self, user):
        """Check if user has a nearby aggregator-forward UAV."""
        return self._find_nearest_af_uav(user) is not None
    
    def _calculate_distance(self, user1, user2):
        """Calculate distance between two UAVs using their positions."""
        # For now, using a simple Euclidean distance
        # This should be updated to use actual UAV positions when available
        return abs(user1.id - user2.id)  # Placeholder distance calculation
        
    def log_metrics(self, round_num):
        """Log training metrics for the current round."""
        # Calculate average training loss and accuracy
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0
        total_samples = 0
        
        for user in self.selected_users:
            # Handle GNNDataset objects
            if hasattr(user, 'train_samples'):
                user_samples = user.train_samples
            else:
                # For GNNDataset, use length of dataset
                user_samples = len(user.trainloader.dataset)
                user.train_samples = user_samples  # Cache for future use
            
            total_samples += user_samples
            
            # Get user metrics
            train_loss += user_samples * user.train_loss
            train_acc += user_samples * user.train_accuracy
            test_loss += user_samples * user.test_loss
            test_acc += user_samples * user.test_accuracy
        
        # Compute weighted averages
        train_loss /= total_samples
        train_acc /= total_samples
        test_loss /= total_samples
        test_acc /= total_samples
        
        # Store metrics
        self.rs_train_loss.append(train_loss)
        self.rs_train_acc.append(train_acc)
        self.rs_test_loss.append(test_loss)
        self.rs_test_acc.append(test_acc)
        
        # Calculate communication overhead considering bandwidth
        param_size = sum(p.numel() for p in self.global_model.parameters()) * 4  # 4 bytes per float32
        total_data = param_size * len(self.selected_users) * 2  # *2 for upload and download
        
        # Convert bandwidth from Mbps to bytes per second and calculate overhead
        bandwidth_bytes_per_sec = self.model_config.uav_bandwidth * 1024 * 1024 / 8  # Convert Mbps to bytes/sec
        communication_time = total_data / bandwidth_bytes_per_sec
        self.communication_overhead.append(total_data)
        
        # Store training time for this round
        if hasattr(self, 'round_start_time'):
            round_time = time.time() - self.round_start_time
            self.training_times.append(round_time)
        
        print(f"Round {round_num} Metrics:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Communication Overhead: {self.communication_overhead[-1]/1024/1024:.2f} MB")
        
        # Log metrics to CSV files
        self.metrics_logger.log_training_metrics(
            round_num, train_loss, train_acc, test_loss, test_acc
        )
        # Log communication metrics with time based on bandwidth
        comm_time = self.communication_overhead[-1] / (self.model_config.uav_bandwidth * 1024 * 1024 / 8)
        self.metrics_logger.log_communication_metrics(
            round_num, self.communication_overhead[-1] / 1024 / 1024,  # Convert to MB
            comm_time  # Use calculated communication time
        )
        self.metrics_logger.log_resource_metrics(
            round_num,
            round_time if hasattr(self, 'round_start_time') else 0,
            psutil.Process().memory_info().rss / 1024 / 1024  # Convert to MB
        )
