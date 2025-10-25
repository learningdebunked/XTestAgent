"""
Knowledge Graph Navigator for TestAgentX.

Handles graph traversal, querying, and RL-based navigation.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import numpy as np
import networkx as nx
from neo4j import GraphDatabase, Driver, Session, Transaction, Result
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
import random
from collections import deque
import json

from .schema import (
    NodeType, RelationshipType, NodeProperties, MethodNode, ClassNode, 
    TestNode, BugNode, SchemaManager
)

class GraphNavigator:
    """Handles graph traversal, querying, and RL-based navigation."""
    
    def __init__(
        self, 
        graph_constructor: 'GraphConstructor',
        embedding_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64
    ):
        """Initialize the graph navigator.
        
        Args:
            graph_constructor: Instance of GraphConstructor
            embedding_dim: Dimension of node embeddings
            learning_rate: Learning rate for the RL agent
            gamma: Discount factor for future rewards
            epsilon: Exploration rate (initial)
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            memory_size: Size of the experience replay buffer
            batch_size: Batch size for training
        """
        self.graph_constructor = graph_constructor
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize RL components
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Q-network and target network
        self.q_network = GATNetwork(embedding_dim).to(self.device)
        self.target_network = GATNetwork(embedding_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # For graph traversal caching
        self._graph_cache = {}
        self._node_embeddings = {}
    
    # Graph Query Methods
    
    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get a node by its ID."""
        if not self.graph_constructor.driver:
            self.logger.error("Database connection not established")
            return None
            
        with self.graph_constructor.driver.session(database=self.graph_constructor.database) as session:
            try:
                result = session.run(
                    "MATCH (n) WHERE id(n) = $node_id RETURN n, labels(n) as labels",
                    node_id=node_id
                ).single()
                
                if not result:
                    return None
                    
                node_data = dict(result["n"].items())
                node_data["id"] = node_id
                node_data["labels"] = result["labels"]
                return node_data
                
            except Exception as e:
                self.logger.error(f"Error getting node {node_id}: {e}")
                return None
    
    def get_neighbors(
        self, 
        node_id: int, 
        rel_types: Optional[List[RelationshipType]] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get neighbors of a node with optional relationship filtering.
        
        Args:
            node_id: ID of the source node
            rel_types: Optional list of relationship types to filter by
            direction: "incoming", "outgoing", or "both"
            
        Returns:
            List of neighbor nodes with relationship information
        """
        if not self.graph_constructor.driver:
            self.logger.error("Database connection not established")
            return []
            
        with self.graph_constructor.driver.session(database=self.graph_constructor.database) as session:
            try:
                # Build the relationship pattern
                rel_pattern = ""
                if rel_types:
                    rel_types_str = "|".join([rt.value for rt in rel_types])
                    rel_pattern = f"[:{rel_types_str}]"
                
                # Build the direction part of the query
                direction_map = {
                    "incoming": "<-",
                    "outgoing": "->",
                    "both": "-"
                }
                direction_str = direction_map.get(direction, "-")
                
                query = f"""
                MATCH (n)-{direction_str}{rel_pattern}-(m)
                WHERE id(n) = $node_id
                RETURN id(m) as id, m as node, type(r) as rel_type, 
                       startNode(r) = n as is_outgoing,
                       labels(m) as labels
                """
                
                results = session.run(query, node_id=node_id)
                neighbors = []
                
                for record in results:
                    node_data = dict(record["node"].items())
                    node_data["id"] = record["id"]
                    node_data["labels"] = record["labels"]
                    
                    neighbors.append({
                        "node": node_data,
                        "relationship": {
                            "type": record["rel_type"],
                            "is_outgoing": record["is_outgoing"]
                        }
                    })
                
                return neighbors
                
            except Exception as e:
                self.logger.error(f"Error getting neighbors for node {node_id}: {e}")
                return []
    
    def find_shortest_path(
        self, 
        source_id: int, 
        target_id: int,
        max_depth: int = 5,
        rel_types: Optional[List[RelationshipType]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Find the shortest path between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            max_depth: Maximum path length to consider
            rel_types: Optional list of relationship types to allow
            
        Returns:
            List of nodes and relationships representing the path, or None if no path exists
        """
        if not self.graph_constructor.driver:
            self.logger.error("Database connection not established")
            return None
            
        with self.graph_constructor.driver.session(database=self.graph_constructor.database) as session:
            try:
                # Build the relationship pattern
                rel_pattern = ""
                if rel_types:
                    rel_types_str = "|%".join([rt.value for rt in rel_types])
                    rel_pattern = f"[:`{rel_types_str}`*1..{max_depth}]"
                else:
                    rel_pattern = f"[*1..{max_depth}]"
                
                query = f"""
                MATCH path = shortestPath((a)-{rel_pattern}->(b))
                WHERE id(a) = $source_id AND id(b) = $target_id
                RETURN [n IN nodes(path) | [id(n), labels(n)[0]]] as nodes,
                       [r IN relationships(path) | [id(startNode(r)), id(endNode(r)), type(r)]] as rels
                """
                
                result = session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id
                ).single()
                
                if not result:
                    return None
                
                # Reconstruct the path
                nodes_data = {}
                for node_id, node_label in result["nodes"]:
                    nodes_data[node_id] = {"id": node_id, "label": node_label}
                
                path = []
                for start_id, end_id, rel_type in result["rels"]:
                    path.append({
                        "from": nodes_data[start_id],
                        "to": nodes_data[end_id],
                        "relationship": rel_type
                    })
                
                return path
                
            except Exception as e:
                self.logger.error(f"Error finding shortest path: {e}")
                return None
    
    def get_related_tests(
        self, 
        node_id: int, 
        max_hops: int = 3,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find tests related to a given node.
        
        Args:
            node_id: ID of the source node
            max_hops: Maximum number of hops to traverse
            min_confidence: Minimum confidence score to include
            
        Returns:
            List of related tests with confidence scores
        """
        if not self.graph_constructor.driver:
            self.logger.error("Database connection not established")
            return []
            
        with self.graph_constructor.driver.session(database=self.graph_constructor.database) as session:
            try:
                query = """
                MATCH (start) WHERE id(start) = $node_id
                CALL apoc.path.expandConfig(start, {
                    relationshipFilter: "CALLS|CONTAINS|TESTS|COVERS|VERIFIES|AFFECTS|IMPACTED_BY",
                    minLevel: 1,
                    maxLevel: $max_hops,
                    uniqueness: "NODE_GLOBAL"
                }) YIELD path
                WITH last(nodes(path)) as test_node
                WHERE any(label IN labels(test_node) WHERE label IN ['Test', 'TestCase'])
                RETURN DISTINCT id(test_node) as test_id, test_node.name as test_name,
                       test_node.test_framework as framework,
                       length(path) as distance,
                       apoc.algo.pageRank([test_node]) as pagerank
                ORDER BY distance ASC, pagerank DESC
                """
                
                results = session.run(
                    query,
                    node_id=node_id,
                    max_hops=max_hops
                )
                
                tests = []
                for record in results:
                    # Calculate confidence score based on distance and pagerank
                    distance = record["distance"]
                    pagerank = record["pagerank"] or 0.0
                    confidence = (1.0 / distance) * (1.0 + pagerank) / 2.0
                    
                    if confidence >= min_confidence:
                        tests.append({
                            "test_id": record["test_id"],
                            "test_name": record["test_name"],
                            "framework": record["framework"],
                            "distance": distance,
                            "pagerank": pagerank,
                            "confidence": min(confidence, 1.0)  # Cap at 1.0
                        })
                
                return sorted(tests, key=lambda x: -x["confidence"])
                
            except Exception as e:
                self.logger.error(f"Error finding related tests: {e}")
                return []
    
    # RL-based Navigation
    
    def navigate_to_target(
        self, 
        start_node_id: int, 
        target_node_type: NodeType,
        max_steps: int = 20,
        explore: bool = True
    ) -> Dict[str, Any]:
        """Navigate from start node to target node type using RL.
        
        Args:
            start_node_id: ID of the starting node
            target_node_type: Type of node to find
            max_steps: Maximum number of steps to take
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Dictionary containing the result of the navigation
        """
        current_node_id = start_node_id
        path = [{"node_id": start_node_id, "action": "start"}]
        total_reward = 0
        found = False
        
        for step in range(max_steps):
            # Get current state (node embedding)
            state = self._get_node_embedding(current_node_id)
            if state is None:
                break
                
            # Choose action (next node to visit)
            neighbors = self.get_neighbors(current_node_id)
            if not neighbors:
                break
                
            # Epsilon-greedy action selection
            if explore and random.random() < self.epsilon:
                # Random exploration
                next_neighbor = random.choice(neighbors)
                next_node_id = next_neighbor["node"]["id"]
            else:
                # Exploitation: choose best action based on Q-values
                next_node_id = self._choose_best_action(current_node_id, neighbors)
            
            # Take action and observe reward
            reward, done = self._get_reward(next_node_id, target_node_type)
            total_reward += reward
            
            # Store experience in replay memory
            next_state = self._get_node_embedding(next_node_id)
            if next_state is not None:
                self.memory.append((state, next_node_id, reward, next_state, done))
            
            # Update current node
            current_node_id = next_node_id
            path.append({
                "node_id": current_node_id,
                "action": "navigate",
                "reward": reward
            })
            
            # Train the network
            self._train_network()
            
            # Check if we've reached the target
            if done:
                found = True
                break
        
        # Decay epsilon
        if explore:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Get the target node if found
        target_node = None
        if found:
            target_node = self.get_node(current_node_id)
        
        return {
            "success": found,
            "path": path,
            "total_reward": total_reward,
            "target_node": target_node,
            "steps": len(path) - 1,  # Exclude the start node
            "exploration_rate": self.epsilon
        }
    
    def _get_node_embedding(self, node_id: int) -> Optional[torch.Tensor]:
        """Get or compute the embedding for a node."""
        if node_id in self._node_embeddings:
            return self._node_embeddings[node_id]
        
        # If not in cache, compute it (simplified example)
        node_data = self.get_node(node_id)
        if not node_data:
            return None
        
        # Simple feature extraction (in a real implementation, use GNN or other methods)
        features = []
        
        # Add node type feature (one-hot encoded)
        node_type = node_data.get("labels", ["Unknown"])[0]
        type_mapping = {
            "Method": 0, "Class": 1, "Test": 2, 
            "Bug": 3, "File": 4, "Package": 5, "TestCase": 6
        }
        type_idx = type_mapping.get(node_type, 7)
        type_features = [0.0] * 8
        type_features[type_idx] = 1.0
        features.extend(type_features)
        
        # Add some basic properties as features
        features.extend([
            float(node_data.get("complexity", 0.0)),
            float(len(node_data.get("name", "")) / 100.0),  # Normalized name length
            float(node_data.get("is_abstract", 0.0) if "is_abstract" in node_data else 0.0),
            float(node_data.get("is_static", 0.0) if "is_static" in node_data else 0.0)
        ])
        
        # Convert to tensor and store in cache
        embedding = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        self._node_embeddings[node_id] = embedding
        return embedding
    
    def _choose_best_action(self, current_node_id: int, neighbors: List[Dict[str, Any]]) -> int:
        """Choose the best action (next node) based on Q-values."""
        if not neighbors:
            return current_node_id
            
        # Get Q-values for all neighbors
        neighbor_ids = [n["node"]["id"] for n in neighbors]
        neighbor_embeddings = []
        
        for node_id in neighbor_ids:
            embedding = self._get_node_embedding(node_id)
            if embedding is not None:
                neighbor_embeddings.append(embedding)
        
        if not neighbor_embeddings:
            return current_node_id
            
        # Stack embeddings and get Q-values
        neighbor_embeddings = torch.cat(neighbor_embeddings, dim=0)
        with torch.no_grad():
            q_values = self.q_network(neighbor_embeddings.unsqueeze(0))
        
        # Choose neighbor with highest Q-value
        best_idx = torch.argmax(q_values).item()
        return neighbor_ids[best_idx % len(neighbor_ids)]
    
    def _get_reward(self, node_id: int, target_node_type: NodeType) -> Tuple[float, bool]:
        """Calculate reward for reaching a node."""
        node = self.get_node(node_id)
        if not node:
            return -1.0, False
        
        # Check if we've reached a node of the target type
        if target_node_type.value in node.get("labels", []):
            return 10.0, True
        
        # Small negative reward for each step to encourage shorter paths
        return -0.1, False
    
    def _train_network(self) -> None:
        """Train the Q-network using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.cat([s for s in states if s is not None])
        next_states = torch.cat([s for s in next_states if s is not None])
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).squeeze()
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_network(next_states).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q.detach())
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._update_target_network()
    
    def _update_target_network(self, tau: float = 0.001) -> None:
        """Soft update of target network parameters.
        
        Args:
            tau: Interpolation parameter for soft update
        """
        for target_param, param in zip(self.target_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def analyze_impact(self, node_id: int, max_depth: int = 3) -> Dict[str, Any]:
        """Analyze the impact of a node on the graph.
        
        Args:
            node_id: ID of the node to analyze
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary containing impact analysis
        """
        if not self.graph_constructor.driver:
            self.logger.error("Database connection not established")
            return {
                "node_id": node_id,
                "error": "No database connection",
                "impacted_nodes": {},
                "total_impacted": 0
            }
            
        with self.graph_constructor.driver.session(database=self.graph_constructor.database) as session:
            try:
                query = """
                MATCH (start) WHERE id(start) = $node_id
                CALL apoc.path.subgraphNodes(start, {
                    relationshipFilter: ">",
                    minLevel: 1,
                    maxLevel: $max_depth
                }) YIELD node
                WITH collect(DISTINCT node) as nodes
                UNWIND nodes as n
                WITH DISTINCT n, labels(n)[0] as type
                RETURN type, count(*) as count, collect(id(n)) as node_ids
                """
                
                results = session.run(query, node_id=node_id, max_depth=max_depth)
                
                impact = {
                    "node_id": node_id,
                    "max_depth": max_depth,
                    "impacted_nodes": {},
                    "total_impacted": 0
                }
                
                for record in results:
                    node_type = record["type"]
                    count = record["count"]
                    node_ids = record["node_ids"]
                    
                    impact["impacted_nodes"][node_type] = {
                        "count": count,
                        "node_ids": node_ids
                    }
                    impact["total_impacted"] += count
                
                return impact
                
            except Exception as e:
                self.logger.error(f"Error analyzing impact: {e}")
                return {
                    "node_id": node_id,
                    "error": str(e),
                    "impacted_nodes": {},
                    "total_impacted": 0
                }
    
    def save_model(self, path: str) -> None:
        """Save the Q-network model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load the Q-network model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.logger.info(f"Model loaded from {path}")


class GATNetwork(nn.Module):
    """Graph Attention Network for node classification and Q-value estimation."""
    
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # Q-value output
    
    def forward(self, x, edge_index=None):
        # If edge_index is not provided, assume fully connected graph
        if edge_index is None:
            batch_size = x.size(0)
            edge_index = self._create_fully_connected(batch_size).to(x.device)
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x
    
    def _create_fully_connected(self, num_nodes: int) -> torch.Tensor:
        """Create a fully connected graph edge index."""
        rows = []
        cols = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        return edge_index
