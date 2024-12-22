import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_cifar10(batch_size=64, root='./data'):
    """
    Load CIFAR-10 dataset with standard preprocessing.
    Optimized for cross-platform compatibility and low memory usage.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return trainloader, testloader

class VWWDataset(Dataset):
    """
    Visual Wake Words Dataset
    Simplified implementation for basic functionality verification
    """
    def __init__(self, root='./data/vww', train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((96, 96)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Placeholder for actual data loading
        # In real implementation, load data from root directory
        self.data = torch.randn(1000 if train else 100, 3, 96, 96)
        self.targets = torch.randint(0, 2, (1000 if train else 100,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_vww(batch_size=32, root='./data/vww'):
    """
    Load Visual Wake Words dataset
    """
    trainset = VWWDataset(root=root, train=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = VWWDataset(root=root, train=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

class KWSDataset(Dataset):
    """
    Keyword Spotting Dataset
    Simplified implementation for basic functionality verification
    """
    def __init__(self, root='./data/kws', train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Placeholder for actual data loading
        # In real implementation, load audio features from root directory
        self.data = torch.randn(1000 if train else 100, 1, 40, 98)  # MFCC features
        self.targets = torch.randint(0, 12, (1000 if train else 100,))  # 12 keywords

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_kws(batch_size=32, root='./data/kws'):
    """
    Load Keyword Spotting dataset
    """
    trainset = KWSDataset(root=root, train=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = KWSDataset(root=root, train=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def create_data_directory(root='./data'):
    """
    Create data directories if they don't exist
    Ensures cross-platform compatibility
    """
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, 'vww'), exist_ok=True)
    os.makedirs(os.path.join(root, 'kws'), exist_ok=True)
