import sys
import importlib

def get_package_version(package_name):
    try:
        package = importlib.import_module(package_name)
        return package.__version__
    except AttributeError:
        return "Version not found"

def verify_dependencies():
    required_packages = {
        'torch': 'PyTorch',
        'torch_geometric': 'torch-geometric',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn'
    }
    
    success = True
    for package, display_name in required_packages.items():
        try:
            importlib.import_module(package)
            version = get_package_version(package)
            print(f"{display_name} import successful (version: {version})")
        except ImportError as e:
            print(f"{display_name} import failed: {e}")
            success = False
    return success

if __name__ == "__main__":
    success = verify_dependencies()
    sys.exit(0 if success else 1)
