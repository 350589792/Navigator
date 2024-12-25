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
class Config:
    def __init__(self):
        self.batch_size = 512
        self.train_num = 4096
        self.device = 'cpu'
        self.hidden_dim = 128
        self.alpha = 0.2
        self.learning_rate = 1e-4
        self.epochs = 500
        self.path_model = './model/rgnn_10.pt'

default_config = Config()
