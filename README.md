# AnalogNAS

Neural Architecture Search framework optimized for IMC (In-Memory Computing) inference accelerators.

## Cross-Platform Compatibility

This code is designed to run on both Ubuntu and Windows systems. All dependencies used are cross-platform compatible:
- Python 3.8+
- NumPy
- PyTorch
- NetworkX
- PyYAML

## Installation

```bash
# Create and activate virtual environment (Ubuntu/Linux)
python -m venv venv
source venv/bin/activate

# Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Usage

Run the example search:
```bash
python example_run.py
```

The framework is designed for minimal hardware resource usage:
- Efficient architecture representation using NetworkX
- Fast performance estimation using proxy model
- Configurable population size and generations
- No training required during search

## Components

1. Search Space
   - ResNet-like architecture space
   - Configurable depth and width
   - Flexible connection patterns

2. Proxy Model
   - Hardware-aware performance estimation
   - Fast architecture evaluation
   - Hardware resource monitoring

3. Evolution Search
   - Population-based optimization
   - Score-weighted selection
   - Hardware-aware constraints

## Configuration

Modify `config.py` parameters to adjust:
- Search space constraints
- Evolution parameters
- Hardware constraints (resource limits)

## Testing

Run the test suite:
```bash
python -m unittest tests/test_basic.py -v
```
