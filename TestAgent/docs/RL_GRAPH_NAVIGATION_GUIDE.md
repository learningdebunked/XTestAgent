# RL-Based Graph Navigation Guide

## Overview

This module implements **Deep Q-Network (DQN)** based reinforcement learning for intelligent graph traversal in the TestAgentX knowledge graph, as mentioned in the paper.

The RL agent learns optimal paths through the knowledge graph to find relevant nodes, tests, and relationships efficiently.

## Key Components

### 1. DQN Architecture

**Deep Q-Network (DQN)** with Graph Attention Networks (GAT):

```
Input: Node Embedding (128-dim)
  ↓
GAT Layer 1 (Multi-head Attention, 4 heads)
  ↓
ReLU + Dropout (0.2)
  ↓
GAT Layer 2 (Single-head Attention)
  ↓
Fully Connected Layer
  ↓
Output: Q-value (scalar)
```

### 2. RL Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **State** | Node embedding (features) | 12-dim vector |
| **Action** | Navigate to neighbor node | Node ID |
| **Reward** | +10 for target, -0.1 per step | Sparse rewards |
| **Policy** | Epsilon-greedy | ε-greedy exploration |
| **Learning** | Experience replay + target network | DQN algorithm |

### 3. Training Process

1. **Exploration**: Random action with probability ε
2. **Exploitation**: Choose action with highest Q-value
3. **Experience Replay**: Store (state, action, reward, next_state, done)
4. **Batch Learning**: Sample mini-batch and update Q-network
5. **Target Network**: Soft update for stability

## Quick Start

### Basic Navigation

```python
from layer5_knowledge_graph import GraphNavigator, GraphConstructor

# Initialize
constructor = GraphConstructor(neo4j_uri="bolt://localhost:7687")
navigator = GraphNavigator(constructor)

# Navigate to find a test node
result = navigator.navigate_to_target(
    start_node_id=123,
    target_node_type=NodeType.TEST,
    max_steps=20,
    explore=True
)

print(f"Success: {result['success']}")
print(f"Path length: {result['steps']}")
print(f"Total reward: {result['total_reward']}")
```

### Training the RL Agent

```python
# Train over multiple episodes
num_episodes = 1000

for episode in range(num_episodes):
    # Random start node
    start_node = random.choice(all_nodes)
    
    # Navigate
    result = navigator.navigate_to_target(
        start_node_id=start_node,
        target_node_type=NodeType.TEST,
        explore=True  # Enable exploration during training
    )
    
    if episode % 100 == 0:
        print(f"Episode {episode}: Success rate = {success_rate:.2f}")
        print(f"Epsilon: {navigator.epsilon:.3f}")

# Save trained model
navigator.save_model('models/graph_navigator.pth')
```

### Using Trained Model

```python
# Load pre-trained model
navigator.load_model('models/graph_navigator.pth')

# Use for inference (no exploration)
result = navigator.navigate_to_target(
    start_node_id=456,
    target_node_type=NodeType.BUG,
    explore=False  # Pure exploitation
)
```

## DQN Algorithm Details

### State Representation

Node features (12-dimensional):
- **Node type** (8-dim one-hot): Method, Class, Test, Bug, File, Package, TestCase, Other
- **Complexity** (1-dim): Cyclomatic complexity
- **Name length** (1-dim): Normalized length
- **Is abstract** (1-dim): Boolean flag
- **Is static** (1-dim): Boolean flag

```python
def _get_node_embedding(self, node_id: int) -> torch.Tensor:
    node_data = self.get_node(node_id)
    
    # Extract features
    features = []
    
    # Node type (one-hot)
    type_features = [0.0] * 8
    type_features[type_idx] = 1.0
    features.extend(type_features)
    
    # Properties
    features.extend([
        float(node_data.get("complexity", 0.0)),
        float(len(node_data.get("name", "")) / 100.0),
        float(node_data.get("is_abstract", 0.0)),
        float(node_data.get("is_static", 0.0))
    ])
    
    return torch.FloatTensor(features)
```

### Action Selection

**Epsilon-Greedy Policy**:

```python
if random.random() < epsilon:
    # Explore: random action
    action = random.choice(neighbors)
else:
    # Exploit: best Q-value
    action = argmax(Q(state, action))
```

### Reward Function

```python
def _get_reward(self, node_id: int, target_type: NodeType) -> Tuple[float, bool]:
    node = self.get_node(node_id)
    
    if target_type in node.labels:
        return +10.0, True  # Found target!
    else:
        return -0.1, False  # Step penalty
```

### Q-Learning Update

**Bellman Equation**:
```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

**Implementation**:
```python
# Current Q-values
current_q = q_network(states)

# Target Q-values
with torch.no_grad():
    next_q = target_network(next_states)
    target_q = rewards + gamma * next_q * (1 - dones)

# Loss and update
loss = MSE(current_q, target_q)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Target Network Update

**Soft Update** (Polyak averaging):
```python
θ_target ← τ * θ + (1 - τ) * θ_target
```

```python
def _update_target_network(self, tau=0.001):
    for target_param, param in zip(target_network.parameters(), 
                                   q_network.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )
```

## Hyperparameters

### Default Values

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_dim` | 128 | Node embedding dimension |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Decay rate per episode |
| `memory_size` | 10000 | Experience replay buffer size |
| `batch_size` | 64 | Training batch size |
| `tau` | 0.001 | Target network update rate |

### Tuning Guidelines

**For faster convergence**:
- Increase `learning_rate` to 0.01
- Decrease `epsilon_decay` to 0.99
- Increase `batch_size` to 128

**For more stable learning**:
- Decrease `learning_rate` to 0.0001
- Increase `memory_size` to 50000
- Decrease `tau` to 0.0001

**For better exploration**:
- Increase `epsilon_min` to 0.1
- Decrease `epsilon_decay` to 0.999

## Advanced Usage

### Custom Reward Function

```python
class CustomNavigator(GraphNavigator):
    def _get_reward(self, node_id: int, target_type: NodeType) -> Tuple[float, bool]:
        node = self.get_node(node_id)
        
        # Custom rewards
        if target_type in node.labels:
            return +100.0, True  # Higher reward for target
        
        # Intermediate rewards for getting closer
        if "Test" in node.labels:
            return +1.0, False  # Bonus for test nodes
        
        # Penalty for wrong direction
        if "File" in node.labels:
            return -1.0, False  # Avoid file nodes
        
        return -0.1, False  # Default step penalty
```

### Multi-Target Navigation

```python
def navigate_to_multiple_targets(navigator, start_node, target_types):
    """Navigate to find any of multiple target types"""
    results = []
    
    for target_type in target_types:
        result = navigator.navigate_to_target(
            start_node_id=start_node,
            target_node_type=target_type,
            max_steps=15
        )
        
        if result['success']:
            results.append(result)
    
    # Return best result (shortest path)
    return min(results, key=lambda r: r['steps'])
```

### Curriculum Learning

```python
# Start with easy tasks, gradually increase difficulty
difficulties = [
    (NodeType.FILE, 5),      # Easy: nearby files
    (NodeType.CLASS, 10),    # Medium: classes
    (NodeType.TEST, 15),     # Hard: tests
    (NodeType.BUG, 20)       # Very hard: bugs
]

for target_type, max_steps in difficulties:
    for episode in range(200):
        result = navigator.navigate_to_target(
            start_node_id=random_node(),
            target_node_type=target_type,
            max_steps=max_steps
        )
```

## Integration with TestAgentX

### Find Related Tests

```python
# Use RL to find tests related to a method
method_node_id = 123

result = navigator.navigate_to_target(
    start_node_id=method_node_id,
    target_node_type=NodeType.TEST,
    max_steps=10
)

if result['success']:
    test_node = result['target_node']
    print(f"Found test: {test_node['name']}")
```

### Bug Impact Analysis

```python
# Navigate from bug to impacted code
bug_node_id = 456

# Find all methods affected by bug
impacted_methods = []

for _ in range(10):  # Multiple navigation attempts
    result = navigator.navigate_to_target(
        start_node_id=bug_node_id,
        target_node_type=NodeType.METHOD,
        max_steps=15
    )
    
    if result['success']:
        impacted_methods.append(result['target_node'])
```

### Test Prioritization

```python
# Find shortest path from change to tests
changed_method_id = 789

# Navigate to find relevant tests
test_distances = {}

for test_id in all_test_ids:
    path = navigator.find_shortest_path(
        source_id=changed_method_id,
        target_id=test_id,
        max_depth=5
    )
    
    if path:
        test_distances[test_id] = len(path)

# Prioritize tests by distance
prioritized_tests = sorted(test_distances.items(), key=lambda x: x[1])
```

## Performance Metrics

### Training Metrics

Track during training:

```python
metrics = {
    'episode': [],
    'success_rate': [],
    'avg_steps': [],
    'avg_reward': [],
    'epsilon': [],
    'loss': []
}

# After each episode
metrics['episode'].append(episode)
metrics['success_rate'].append(success_count / total_episodes)
metrics['avg_steps'].append(np.mean(steps_per_episode))
metrics['avg_reward'].append(np.mean(rewards_per_episode))
metrics['epsilon'].append(navigator.epsilon)
```

### Evaluation Metrics

```python
def evaluate_navigator(navigator, test_cases):
    """Evaluate trained navigator"""
    results = {
        'success_rate': 0,
        'avg_path_length': 0,
        'avg_time': 0
    }
    
    for start, target_type in test_cases:
        result = navigator.navigate_to_target(
            start_node_id=start,
            target_node_type=target_type,
            explore=False
        )
        
        if result['success']:
            results['success_rate'] += 1
            results['avg_path_length'] += result['steps']
    
    results['success_rate'] /= len(test_cases)
    results['avg_path_length'] /= len(test_cases)
    
    return results
```

## Troubleshooting

### Low Success Rate

**Problem**: Agent rarely finds target nodes

**Solutions**:
1. Increase `max_steps`
2. Adjust reward function (higher target reward)
3. Add intermediate rewards
4. Increase exploration (`epsilon_min`)

### Slow Convergence

**Problem**: Training takes too long

**Solutions**:
1. Increase `learning_rate`
2. Increase `batch_size`
3. Use curriculum learning
4. Pre-train on simpler tasks

### Unstable Training

**Problem**: Performance fluctuates wildly

**Solutions**:
1. Decrease `learning_rate`
2. Increase `memory_size`
3. Decrease `tau` (slower target updates)
4. Add gradient clipping

### Memory Issues

**Problem**: Out of memory during training

**Solutions**:
1. Decrease `batch_size`
2. Decrease `memory_size`
3. Use smaller `embedding_dim`
4. Clear cache periodically

## Example: Complete Training Pipeline

```python
#!/usr/bin/env python3
"""
Complete RL-based graph navigation training pipeline
"""

from layer5_knowledge_graph import GraphNavigator, GraphConstructor
import random
import numpy as np
import matplotlib.pyplot as plt

def train_navigator():
    # Initialize
    constructor = GraphConstructor(neo4j_uri="bolt://localhost:7687")
    navigator = GraphNavigator(
        constructor,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    # Get all nodes
    all_nodes = constructor.get_all_nodes()
    
    # Training
    num_episodes = 1000
    success_history = []
    reward_history = []
    
    for episode in range(num_episodes):
        # Random start
        start_node = random.choice(all_nodes)
        
        # Navigate
        result = navigator.navigate_to_target(
            start_node_id=start_node['id'],
            target_node_type=NodeType.TEST,
            max_steps=20,
            explore=True
        )
        
        # Track metrics
        success_history.append(1 if result['success'] else 0)
        reward_history.append(result['total_reward'])
        
        # Log progress
        if episode % 100 == 0:
            recent_success = np.mean(success_history[-100:])
            recent_reward = np.mean(reward_history[-100:])
            print(f"Episode {episode}:")
            print(f"  Success Rate: {recent_success:.2%}")
            print(f"  Avg Reward: {recent_reward:.2f}")
            print(f"  Epsilon: {navigator.epsilon:.3f}")
    
    # Save model
    navigator.save_model('models/trained_navigator.pth')
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(success_history)
    plt.title('Success Rate Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Success')
    
    plt.subplot(1, 2, 2)
    plt.plot(reward_history)
    plt.title('Reward Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    
    print("\nTraining complete!")
    print(f"Final success rate: {np.mean(success_history[-100:]):.2%}")

if __name__ == "__main__":
    train_navigator()
```

## References

- DQN Paper: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- GAT Paper: "Graph Attention Networks" (Veličković et al., 2018)
- TestAgentX Paper: Section on RL-based graph navigation
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

## Support

For questions about RL-based navigation:
1. Review this guide
2. Check example scripts in `examples/rl_navigation/`
3. See training logs in `logs/`
4. Open an issue on GitHub
