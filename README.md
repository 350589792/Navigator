# UAV Network Federated Learning Framework

This framework implements a federated learning system for UAV networks using Graph Neural Networks (GNN) for dynamic observation and relay decisions.

## Configuration Parameters

### Network Size Parameters
- `n_users_small`: Number of users for small network (default: 10)
- `n_uavs_small`: Number of UAVs for small network (default: 2)
- `n_users_medium`: Number of users for medium network (default: 20)
- `n_uavs_medium`: Number of UAVs for medium network (default: 5)
- `n_users_large`: Number of users for large network (default: 50)
- `n_uavs_large`: Number of UAVs for large network (default: 10)

### UAV Resource Configuration
- `uav_compute_speed`: UAV computation speed factor (default: 1.0)
- `uav_bandwidth`: UAV bandwidth capacity in Mbps (default: 10.0)
- `max_relay_distance`: Maximum distance for UAV relay communication (default: 100.0)

## Running Different Deployment Scenarios

### 1. Direct Gradient Transmission
Users directly transmit gradients to the server without relay:
```bash
python simulate_uav_fed.py --n_users_small=10 --n_uavs_small=2 --uav_bandwidth=10
```

### 2. No Relay with Resource Optimization
Optimize computation and bandwidth without relay:
```bash
python simulate_uav_fed.py --n_users_small=10 --n_uavs_small=2 --uav_compute_speed=2.0 --uav_bandwidth=20
```

### 3. FedUAVGNN with Dynamic Relay
Enable GNN-based relay decisions:
```bash
python simulate_uav_fed.py --n_users_medium=20 --n_uavs_medium=5 --uav_compute_speed=1.0 --uav_bandwidth=10 --max_relay_distance=100
```

### 4. Bellman-Ford Relay Path Selection
Use Bellman-Ford algorithm for relay path optimization:
```bash
python simulate_uav_fed.py --n_users_large=50 --n_uavs_large=10 --uav_bandwidth=15 --max_relay_distance=150
```

## Comparing Different Scenarios

To compare all scenarios, run:
```bash
# Direct transmission baseline
python simulate_uav_fed.py --n_users_medium=20 --n_uavs_medium=5 --uav_bandwidth=10

# Resource optimization without relay
python simulate_uav_fed.py --n_users_medium=20 --n_uavs_medium=5 --uav_compute_speed=2.0 --uav_bandwidth=20

# FedUAVGNN with dynamic relay
python simulate_uav_fed.py --n_users_medium=20 --n_uavs_medium=5 --uav_compute_speed=1.0 --uav_bandwidth=10 --max_relay_distance=100

# Bellman-Ford path optimization
python simulate_uav_fed.py --n_users_medium=20 --n_uavs_medium=5 --uav_bandwidth=15 --max_relay_distance=150
```

The framework will automatically log metrics for each scenario in the `logs` directory, including:
- Training time
- Communication overhead
- Model accuracy
- Convergence speed

Results can be visualized using:
```bash
python analyze_metrics.py
```
