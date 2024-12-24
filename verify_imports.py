import os
import sys
print("Starting import verification...")

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print(f"Python path: {sys.path}")

try:
    print("\nTrying imports...")
    from flearn.servers.serverfedl import FEDL
    print("✓ Successfully imported serverfedl.FEDL")
    from flearn.models.models import FedUAVGNN
    print("✓ Successfully imported models.FedUAVGNN")
    from flearn.data.data_load_new import create_federated_data
    print("✓ Successfully imported data_load_new.create_federated_data")
    from flearn.utils.metrics_logger import MetricsLogger
    print("✓ Successfully imported metrics_logger.MetricsLogger")
    print("\nAll imports successful!")
except Exception as e:
    print(f"\nError during imports: {str(e)}")
    print("\nCurrent directory structure:")
    os.system('ls -R .')
    sys.exit(1)
