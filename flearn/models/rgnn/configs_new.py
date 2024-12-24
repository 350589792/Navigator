import  argparse

def get_default_args():
    """Get default arguments for RGNN model."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--train_num', default=4096)
    parser.add_argument('--device', default='cpu')  # cuda:0
    parser.add_argument('--hidden_dim', default=128)
    parser.add_argument('--alpha', default=0.2)
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--path_model', default='./model/rgnn_10.pt')
    
    return parser

# Default configuration as a dictionary
default_config = {
    'batch_size': 512,
    'train_num': 4096,
    'device': 'cpu',
    'hidden_dim': 128,
    'alpha': 0.2,
    'learning_rate': 1e-4,
    'epochs': 500,
    'path_model': './model/rgnn_10.pt'
}
