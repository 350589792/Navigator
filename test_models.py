"""Test script for verifying model and dataset compatibility across platforms."""
import os
import argparse
import torch
from analognas.models.resnet32 import ResNet32
from analognas.models.analog_t500 import T500Network
from analognas.datasets import get_cifar10, get_vww, get_kws

def get_model(model_name, num_classes):
    """Get model by name"""
    if model_name.lower() == 'resnet32':
        return ResNet32(num_classes=num_classes)
    elif model_name.lower() == 't500':
        return T500Network(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_dataset(dataset_name, batch_size=32):
    """Get dataset by name"""
    if dataset_name.lower() == 'cifar10':
        return get_cifar10(batch_size=batch_size)
    elif dataset_name.lower() == 'vww':
        return get_vww(batch_size=batch_size)
    elif dataset_name.lower() == 'kws':
        return get_kws(batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    """Run model testing with specified configuration."""
    parser = argparse.ArgumentParser(description='Test AnalogNAS models')
    parser.add_argument('--model', type=str, default='resnet32',
                      choices=['resnet32', 't500'],
                      help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                      choices=['cifar10', 'vww', 'kws'],
                      help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for testing')
    args = parser.parse_args()

    # Print platform information
    print(f"Platform: {os.name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    # Get dataset and determine number of classes
    print(f"Loading {args.dataset} dataset...")
    trainloader, testloader = get_dataset(args.dataset, args.batch_size)
    num_classes = 10 if args.dataset == 'cifar10' else (2 if args.dataset == 'vww' else 12)

    # Create model
    print(f"Creating {args.model} model...")
    model = get_model(args.model, num_classes)
    print(f"Estimated ACU consumption: {model.get_acu_consumption():.2f}")

    # Test forward pass
    print("\nTesting forward pass...")
    try:
        for images, labels in trainloader:
            outputs = model(images)
            print(f"Forward pass successful!")
            print(f"Input shape: {images.shape}")
            print(f"Output shape: {outputs.shape}")
            break
        
        print("\nTest complete! The model runs successfully on your system.")
        print("\nUsage examples:")
        print("1. Test ResNet32 on CIFAR-10:")
        print("   python test_models.py --model resnet32 --dataset cifar10")
        print("2. Test T500 on VWW:")
        print("   python test_models.py --model t500 --dataset vww")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
