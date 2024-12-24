import sys
import torch
import torch_geometric
import psutil
import matplotlib.pyplot as plt
import numpy as np

def verify_environment():
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyG version: {torch_geometric.__version__}")
    print(f"Psutil version: {psutil.__version__}")
    print("All required packages are installed!")

if __name__ == "__main__":
    verify_environment()
