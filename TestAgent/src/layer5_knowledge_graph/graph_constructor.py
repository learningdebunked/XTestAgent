"""
Knowledge Graph Constructor for TestAgentX.

Handles the creation, updating, and management of the Neo4j knowledge graph.
"""
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import uuid
import numpy as np
from neo4j import GraphDatabase, Driver, Session, Transaction, Result

from .schema import (
    NodeType, RelationshipType, NodeProperties, MethodNode, ClassNode, 
    TestNode, BugNode, FileNode, PackageNode, TestCaseNode, SchemaManager
)

class GraphConstructor:
    """Handles construction and management of the knowledge graph."""
    
    def __init__(
        self, 
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "testagentx",
        clear_existing: bool = False
    ):
        """Initialize the graph constructor with database connection details.
        
        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
            database: Database name
            clear_existing: If True, clears existing data before initializing
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection
        self.connect()
        
        # Initialize schema if needed or clear existing data
        with self.driver.session(database=self.database) as session:
            if clear_existing:
                self.clear_database(session)
            self.initialize_schema(session)
    
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Verify connection
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j database at {self.uri}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close the database connection."""
        if self.driver is not None:
            self.driver.close()
            self.logger.info("Closed Neo4j connection")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def clear_database(self, session: Session) -> None:
        """Clear all data from the database."""
        try:
            self.logger.warning("Clearing all data from the database")
            session.run("MATCH (n) DETACH DELETE n")
            session.run("CALL apoc.schema.assert({}, {}, true) YIELD label, key")
            self.logger.info("Database cleared")
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            raise
    
    def initialize_schema(self, session: Session) -> None:
        """Initialize the database schema with constraints and indexes."""
        try:
            self.logger.info("Initializing database schema")
            schema_manager = SchemaManager()
            
            # Execute all schema creation queries
            for query in schema_manager.get_schema_queries():
                try:
                    session.run(query)
                except Exception as e:
                    self.logger.warning(f"Error executing schema query: {e}")
                    # Continue with other queries even if some fail
                    
            self.logger.info("Database schema initialized")
        except Exception as e:
            self.logger.error(f"Error initializing schema: {e}")
            raise
    
    # Node creation methods
    
    def create_node(self, node: NodeProperties) -> Tuple[bool, int]:
        """Create a node in the graph.
        
        Args:
            node: Node properties object
            
        Returns:
            Tuple of (success: bool, node_id: int)
        """
        if not self.driver:
            self.logger.error("Database connection not established")
            return False, -1
            
        with self.driver.session(database=self.database) as session:
            try:
                # Convert node to dictionary
                node_dict = node.to_dict()
                labels = ":".join(node_dict["labels"])
                props = self._prepare_properties(node_dict["properties"])
                
                # Create the node
                query = f"""
                MERGE (n:{labels} {{ {', '.join([f'{k}: ${k}' for k in props.keys()])} }})
                RETURN id(n) as node_id
                """
                
                result = session.run(query, **props).single()
                node_id = result["node_id"] if result else -1
                
                if node_id == -1:
                    self.logger.error(f"Failed to create node: {node}")
                    return False, -1
                    
                self.logger.debug(f"Created node {node_id} with labels {labels}")
                return True, node_id
                
            except Exception as e:
                self.logger.error(f"Error creating node: {e}")
                return False, -1
    
    def create_method_node(self, method: MethodNode) -> Tuple[bool, int]:
        """Create a method node in the graph."""
        return self.create_node(method)
    
    def create_class_node(self, class_node: ClassNode) -> Tuple[bool, int]:
        """Create a class node in the graph."""
        return self.create_node(class_node)
    
    def create_test_node(self, test: TestNode) -> Tuple[bool, int]:
        """Create a test node in the graph."""
        return self.create_node(test)
    
    def create_bug_node(self, bug: BugNode) -> Tuple[bool, int]:
        """Create a bug node in the graph."""
        return self.create_node(bug)
    
    def create_file_node(self, file_node: FileNode) -> Tuple[bool, int]:
        """Create a file node in the graph."""
        return self.create_node(file_node)
    
    def create_package_node(self, package: PackageNode) -> Tuple[bool, int]:
        """Create a package node in the graph."""
        return self.create_node(package)
    
    def create_test_case_node(self, test_case: TestCaseNode) -> Tuple[bool, int]:
        """Create a test case node in the graph."""
        return self.create_node(test_case)
    
    # Relationship creation methods
    
    def create_relationship(
        self, 
        from_node_id: int, 
        to_node_id: int, 
        rel_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between two nodes.
        
        Args:
            from_node_id: ID of the source node
            to_node_id: ID of the target node
            rel_type: Type of relationship
            properties: Optional properties for the relationship
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            self.logger.error("Database connection not established")
            return False
            
        with self.driver.session(database=self.database) as session:
            try:
                props = self._prepare_properties(properties or {})
                props["from_id"] = from_node_id
                props["to_id"] = to_node_id
                
                # Create the relationship
                query = f"""
                MATCH (from) WHERE id(from) = $from_id
                MATCH (to) WHERE id(to) = $to_id
                MERGE (from)-[r:{rel_type.value}]->(to)
                """
                
                if props:
                    query += " SET r += $props"
                    props["props"] = {k: v for k, v in props.items() if k not in ["from_id", "to_id"]}
                
                session.run(query, **props)
                self.logger.debug(f"Created relationship {rel_type.value} between {from_node_id} and {to_node_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error creating relationship: {e}")
                return False
    
    # Batch operations
    
    def batch_create_nodes(self, nodes: List[NodeProperties]) -> List[Tuple[bool, int]]:
        """Create multiple nodes in a single transaction."""
        if not self.driver:
            self.logger.error("Database connection not established")
            return [(False, -1)] * len(nodes)
            
        results = []
        with self.driver.session(database=self.database) as session:
            tx = session.begin_transaction()
            try:
                for node in nodes:
                    success, node_id = self._create_node_in_tx(tx, node)
                    results.append((success, node_id))
                tx.commit()
                return results
            except Exception as e:
                self.logger.error(f"Error in batch create nodes: {e}")
                tx.rollback()
                return [(False, -1)] * len(nodes)
    
    def _create_node_in_tx(self, tx: Transaction, node: NodeProperties) -> Tuple[bool, int]:
        """Helper method to create a node within a transaction."""
        try:
            node_dict = node.to_dict()
            labels = ":".join(node_dict["labels"])
            props = self._prepare_properties(node_dict["properties"])
            
            query = f"""
            CREATE (n:{labels} $props)
            RETURN id(n) as node_id
            """
            
            result = tx.run(query, props=props).single()
            node_id = result["node_id"] if result else -1
            
            if node_id == -1:
                self.logger.error(f"Failed to create node in transaction: {node}")
                return False, -1
                
            return True, node_id
            
        except Exception as e:
            self.logger.error(f"Error in _create_node_in_tx: {e}")
            return False, -1
    
    # Query methods
    
    def find_node(self, node_type: NodeType, properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a node by its properties."""
        if not self.driver:
            self.logger.error("Database connection not established")
            return None
            
        with self.driver.session(database=self.database) as session:
            try:
                props = self._prepare_properties(properties)
                query = f"""
                MATCH (n:{node_type.value} {{ {', '.join([f'{k}: ${k}' for k in props.keys()])} }})
                RETURN n, id(n) as id
                """
                
                result = session.run(query, **props).single()
                if result:
                    node = dict(result["n"].items())
                    node["id"] = result["id"]
                    return node
                return None
                
            except Exception as e:
                self.logger.error(f"Error finding node: {e}")
                return None
    
    def find_nodes_by_type(self, node_type: NodeType, limit: int = 100) -> List[Dict[str, Any]]:
        """Find all nodes of a given type."""
        if not self.driver:
            self.logger.error("Database connection not established")
            return []
            
        with self.driver.session(database=self.database) as session:
            try:
                query = f"""
                MATCH (n:{node_type.value})
                RETURN n, id(n) as id
                LIMIT $limit
                """
                
                results = session.run(query, limit=limit)
                nodes = []
                for record in results:
                    node = dict(record["n"].items())
                    node["id"] = record["id"]
                    nodes.append(node)
                return nodes
                
            except Exception as e:
                self.logger.error(f"Error finding nodes by type: {e}")
                return []
    
    # Helper methods
    
    def _prepare_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare properties for Neo4j query."""
        if properties is None:
            return {}
            
        # Convert non-serializable types to strings
        prepared = {}
        for key, value in properties.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, (list, dict)):
                prepared[key] = str(value)
            elif hasattr(value, 'isoformat'):  # Handle datetime objects
                prepared[key] = value.isoformat()
            else:
                prepared[key] = str(value)
        return prepared
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Result:
        """Execute a custom Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Optional query parameters
            
        Returns:
            Neo4j Result object
        """
        if not self.driver:
            self.logger.error("Database connection not established")
            raise RuntimeError("Database connection not established")
            
        with self.driver.session(database=self.database) as session:
            try:
                return session.run(query, parameters or {})
            except Exception as e:
                self.logger.error(f"Error executing query: {e}")
                raise
