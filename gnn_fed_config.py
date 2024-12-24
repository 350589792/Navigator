import argparse

def get_args():
    args = argparse.ArgumentParser()
    
    # UAV Network Configuration
    args.add_argument('--n_users_small', type=int, default=10, help='Number of users for small network')
    args.add_argument('--n_uavs_small', type=int, default=2, help='Number of UAVs for small network')
    args.add_argument('--n_users_medium', type=int, default=20, help='Number of users for medium network')
    args.add_argument('--n_uavs_medium', type=int, default=5, help='Number of UAVs for medium network')
    args.add_argument('--n_users_large', type=int, default=50, help='Number of users for large network')
    args.add_argument('--n_uavs_large', type=int, default=10, help='Number of UAVs for large network')
    
    # GNN Model Parameters (from LGNN-RGNN/configs.py)
    args.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    args.add_argument('--alpha', type=float, default=0.2, help='Alpha parameter for attention')
    args.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    args.add_argument('--batch_size', type=int, default=512, help='Batch size')
    
    # Federated Learning Parameters
    args.add_argument('--num_rounds', type=int, default=100, help='Number of federated learning rounds')
    args.add_argument('--local_epochs', type=int, default=5, help='Number of local training epochs')
    args.add_argument('--client_sample_ratio', type=float, default=1.0, help='Ratio of clients to sample per round')
    
    # Training Parameters
    args.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    args.add_argument('--seed', type=int, default=42, help='Random seed')
    args.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval in rounds')
    args.add_argument('--train_num', type=int, default=4096, help='Number of training samples')
    args.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    args.add_argument('--path_model', default='./model/rgnn_10.pt', help='Path to save model')
    
    # Model Checkpointing
    args.add_argument('--checkpoint_dir', default='./checkpoints', help='Directory to save model checkpoints')
    args.add_argument('--log_dir', default='./logs', help='Directory to save training logs')
    
    return args.parse_args()

# Create singleton instance
args = get_args()

if __name__ == '__main__':
    print(args)
