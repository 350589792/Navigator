import argparse

def get_default_args():
    """Get default arguments for GNN Federated Learning."""
    parser = argparse.ArgumentParser()
    
    # UAV Network Configuration
    parser.add_argument('--n_users_small', type=int, default=10, help='Number of users for small network')
    parser.add_argument('--n_uavs_small', type=int, default=2, help='Number of UAVs for small network')
    parser.add_argument('--n_users_medium', type=int, default=20, help='Number of users for medium network')
    parser.add_argument('--n_uavs_medium', type=int, default=5, help='Number of UAVs for medium network')
    parser.add_argument('--n_users_large', type=int, default=50, help='Number of users for large network')
    parser.add_argument('--n_uavs_large', type=int, default=10, help='Number of UAVs for large network')
    
    # GNN Model Parameters (from LGNN-RGNN/configs.py)
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha parameter for attention')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    
    # Federated Learning Parameters
    parser.add_argument('--num_rounds', type=int, default=100, help='Number of federated learning rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local training epochs')
    parser.add_argument('--client_sample_ratio', type=float, default=1.0, help='Ratio of clients to sample per round')
    
    # Training Parameters
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval in rounds')
    parser.add_argument('--train_num', type=int, default=4096, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--path_model', default='./model/rgnn_10.pt', help='Path to save model')
    
    # Model Checkpointing
    parser.add_argument('--checkpoint_dir', default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', default='./logs', help='Directory to save training logs')
    
    return parser

# Default configuration as a dictionary
default_config = {
    'n_users_small': 10,
    'n_uavs_small': 2,
    'n_users_medium': 20,
    'n_uavs_medium': 5,
    'n_users_large': 50,
    'n_uavs_large': 10,
    'hidden_dim': 128,
    'alpha': 0.2,
    'learning_rate': 1e-4,
    'batch_size': 512,
    'num_rounds': 100,
    'local_epochs': 5,
    'client_sample_ratio': 1.0,
    'device': 'cpu',
    'seed': 42,
    'eval_interval': 5,
    'train_num': 4096,
    'epochs': 500,
    'path_model': './model/rgnn_10.pt',
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs'
}

if __name__ == '__main__':
    parser = get_default_args()
    args = parser.parse_args()
    print(args)
