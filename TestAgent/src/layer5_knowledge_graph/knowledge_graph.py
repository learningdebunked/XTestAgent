"""
Knowledge Graph Core Module for TestAgentX.

This module provides the main interface for interacting with the knowledge graph,
combining graph construction, navigation, and querying capabilities.

Implementation of the Knowledge Graph as described in Section 3.7.1 of the paper.
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
import logging
import os
from pathlib import Path
import json
import uuid
from datetime import datetime

from .graph_constructor import GraphConstructor
from .graph_navigator import GraphNavigator
from .schema import (
    NodeType, RelationshipType, NodeProperties, 
    MethodNode, ClassNode, TestNode, BugNode, 
    FileNode, PackageNode, TestCaseNode, SchemaManager
)

# Type aliases
Node = Dict[str, Any]
Edge = Dict[str, Any]

@dataclass
class KnowledgeGraphConfig:
    """Configuration for the Knowledge Graph."""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    database: str = "testagentx"
    clear_existing: bool = False
    embedding_dim: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64

class KnowledgeGraph:
    """Main class for interacting with the knowledge graph."""
    
    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        """Initialize the knowledge graph with the given configuration."""
        self.config = config or KnowledgeGraphConfig()
        self.logger = logging.getLogger(f"{__name__}.KnowledgeGraph")
        
        # Initialize graph constructor and navigator
        self.constructor = GraphConstructor(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            database=self.config.database,
            clear_existing=self.config.clear_existing
        )
        
        self.navigator = GraphNavigator(
            graph_constructor=self.constructor,
            embedding_dim=self.config.embedding_dim,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            epsilon=self.config.epsilon,
            epsilon_min=self.config.epsilon_min,
            epsilon_decay=self.config.epsilon_decay,
            memory_size=self.config.memory_size,
            batch_size=self.config.batch_size
        )
        
        self.logger.info("Knowledge Graph initialized")
    
    def initialize_schema(self) -> None:
        """Initialize the database schema with required constraints and indexes.
        
        This ensures that all required constraints and indexes exist in the database
        for optimal query performance and data integrity.
        """
        schema_manager = SchemaManager(self.constructor.driver)
        
        # Create node key constraints
        schema_manager.create_constraint(NodeType.METHOD, 'id')
        schema_manager.create_constraint(NodeType.CLASS, 'id')
        schema_manager.create_constraint(NodeType.TEST, 'id')
        schema_manager.create_constraint(NodeType.BUG, 'id')
        schema_manager.create_constraint(NodeType.FILE, 'path')
        schema_manager.create_constraint(NodeType.PACKAGE, 'name')
        
        # Create indexes for faster lookups
        schema_manager.create_index(NodeType.METHOD, 'name')
        schema_manager.create_index(NodeType.CLASS, 'name')
        schema_manager.create_index(NodeType.TEST, 'name')
        schema_manager.create_index(NodeType.BUG, 'summary')
        schema_manager.create_index(NodeType.FILE, 'name')
        
        self.logger.info("Knowledge Graph schema initialized")
    
    def build_graph(self, project_root: str) -> None:
        """Build the knowledge graph from a project root directory.
        
        This is the main entry point for constructing the knowledge graph from source code.
        It discovers and processes all source files in the project.
        
        Args:
            project_root: Path to the root directory of the project
        """
        project_root = Path(project_root).resolve()
        if not project_root.exists() or not project_root.is_dir():
            raise ValueError(f"Project root not found: {project_root}")
        
        self.logger.info(f"Building knowledge graph from: {project_root}")
        
        # Clear existing graph if configured
        if self.config.clear_existing:
            self.constructor.clear_graph()
            self.initialize_schema()
        
        # Process source files
        source_files = self._discover_source_files(project_root)
        
        for file_path in source_files:
            try:
                self._process_file(file_path, project_root)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        
        # Build relationships between nodes
        self._build_relationships()
        
        self.logger.info(f"Knowledge graph construction complete. Processed {len(source_files)} files.")
    
    def _discover_source_files(self, project_root: Path) -> List[Path]:
        """Discover all source files in the project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            List of Path objects for all source files
        """
        source_extensions = {
            '.java', '.py', '.js', '.ts', '.go', '.rb', 
            '.c', '.cpp', '.h', '.hpp', '.cs', '.php'
        }
        
        source_files = []
        for ext in source_extensions:
            source_files.extend(project_root.glob(f'**/*{ext}'))
        
        return source_files
    
    def _process_file(self, file_path: Path, project_root: Path) -> None:
        """Process a single source file and add its contents to the graph.
        
        Args:
            file_path: Path to the source file
            project_root: Root directory of the project
        """
        # Skip test files for now (they'll be processed separately)
        if 'test' in str(file_path).lower() or 'spec' in str(file_path).lower():
            return self._process_test_file(file_path, project_root)
        
        # Determine file type and process accordingly
        ext = file_path.suffix.lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create file node
            file_node = self._create_file_node(file_path, project_root, content)
            
            # Parse file based on extension
            if ext == '.java':
                self._process_java_file(content, file_node)
            elif ext == '.py':
                self._process_python_file(content, file_node)
            # Add more language processors as needed
                
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
    
    def _process_test_file(self, file_path: Path, project_root: Path) -> None:
        """Process a test file and add test cases to the graph.
        
        Args:
            file_path: Path to the test file
            project_root: Root directory of the project
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create file node
            file_node = self._create_file_node(file_path, project_root, content)
            
            # Parse test file based on framework
            if file_path.suffix == '.java':
                self._process_java_test_file(content, file_node)
            elif file_path.suffix == '.py':
                self._process_python_test_file(content, file_node)
            
        except Exception as e:
            self.logger.error(f"Error processing test file {file_path}: {str(e)}", exc_info=True)
    
    def _build_relationships(self) -> None:
        """Build relationships between nodes in the graph."""
        # This would be implemented based on the specific relationships needed
        # For example, method calls, class inheritance, test coverage, etc.
        pass
    
    def _create_file_node(self, file_path: Path, project_root: Path, content: str) -> Dict[str, Any]:
        """Create a file node in the graph.
        
        Args:
            file_path: Path to the file
            project_root: Root directory of the project
            content: File content
            
        Returns:
            Created file node
        """
        rel_path = str(file_path.relative_to(project_root))
        
        file_node = FileNode(
            path=str(file_path),
            name=file_path.name,
            relative_path=rel_path,
            size=len(content),
            extension=file_path.suffix,
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            language=self._get_language(file_path.suffix)
        )
        
        return self.constructor.create_node(NodeType.FILE, asdict(file_node))
    
    def _get_language(self, extension: str) -> str:
        """Get the programming language from a file extension."""
        language_map = {
            '.java': 'java',
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.go': 'go',
            '.rb': 'ruby',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.php': 'php'
        }
        return language_map.get(extension.lower(), 'unknown')
    
    def _process_java_file(self, content: str, file_node: Dict[str, Any]) -> None:
        """Process a Java source file and add its contents to the graph."""
        # This would use a Java parser to extract classes, methods, etc.
        # For now, we'll just create a placeholder
        pass
    
    def _process_python_file(self, content: str, file_node: Dict[str, Any]) -> None:
        """Process a Python source file and add its contents to the graph."""
        # This would use the ast module to parse the Python file
        pass
    
    def _process_java_test_file(self, content: str, file_node: Dict[str, Any]) -> None:
        """Process a Java test file and add test cases to the graph."""
        # This would parse JUnit or TestNG test files
        pass
    
    def _process_python_test_file(self, content: str, file_node: Dict[str, Any]) -> None:
        """Process a Python test file and add test cases to the graph."""
        # This would parse unittest or pytest test files
        pass
    
    # Node creation methods
    
    def _create_method_node(self, method_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a method node in the graph.
        
        Args:
            method_data: Method data including name, parameters, return type, etc.
            
        Returns:
            Created method node
        """
        method_id = method_data.get('id', f"method_{uuid.uuid4().hex}")
        
        method_node = MethodNode(
            id=method_id,
            name=method_data['name'],
            parameters=method_data.get('parameters', []),
            return_type=method_data.get('return_type', 'void'),
            visibility=method_data.get('visibility', 'public'),
            is_static=method_data.get('is_static', False),
            is_abstract=method_data.get('is_abstract', False),
            is_constructor=method_data.get('is_constructor', False),
            start_line=method_data.get('start_line', 0),
            end_line=method_data.get('end_line', 0),
            docstring=method_data.get('docstring', ''),
            source_code=method_data.get('source_code', '')
        )
        
        return self.constructor.create_node(NodeType.METHOD, asdict(method_node))
    
    def _create_class_node(self, class_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a class node in the graph.
        
        Args:
            class_data: Class data including name, methods, fields, etc.
            
        Returns:
            Created class node
        """
        class_id = class_data.get('id', f"class_{uuid.uuid4().hex}")
        
        class_node = ClassNode(
            id=class_id,
            name=class_data['name'],
            package=class_data.get('package', ''),
            is_abstract=class_data.get('is_abstract', False),
            is_interface=class_data.get('is_interface', False),
            is_enum=class_data.get('is_enum', False),
            superclass=class_data.get('superclass'),
            interfaces=class_data.get('interfaces', []),
            start_line=class_data.get('start_line', 0),
            end_line=class_data.get('end_line', 0),
            docstring=class_data.get('docstring', '')
        )
        
        return self.constructor.create_node(NodeType.CLASS, asdict(class_node))
    
    def _create_test_node(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a test node in the graph.
        
        Args:
            test_data: Test data including name, file, framework, etc.
            
        Returns:
            Created test node
        """
        test_id = test_data.get('id', f"test_{uuid.uuid4().hex}")
        
        test_node = TestNode(
            id=test_id,
            name=test_data['name'],
            file_path=test_data['file_path'],
            framework=test_data.get('framework', 'unknown'),
            is_parameterized=test_data.get('is_parameterized', False),
            parameters=test_data.get('parameters', []),
            start_line=test_data.get('start_line', 0),
            end_line=test_data.get('end_line', 0),
            docstring=test_data.get('docstring', '')
        )
        
        return self.constructor.create_node(NodeType.TEST, asdict(test_node))
    
    def _create_bug_node(self, bug_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a bug node in the graph.
        
        Args:
            bug_data: Bug data including summary, description, status, etc.
            
        Returns:
            Created bug node
        """
        bug_id = bug_data.get('id', f"bug_{uuid.uuid4().hex}")
        
        bug_node = BugNode(
            id=bug_id,
            summary=bug_data['summary'],
            description=bug_data.get('description', ''),
            status=bug_data.get('status', 'open'),
            priority=bug_data.get('priority', 'medium'),
            severity=bug_data.get('severity', 'normal'),
            created_at=bug_data.get('created_at', datetime.now().isoformat()),
            updated_at=bug_data.get('updated_at', datetime.now().isoformat()),
            resolution=bug_data.get('resolution'),
            affected_versions=bug_data.get('affected_versions', []),
            fixed_versions=bug_data.get('fixed_versions', []),
            labels=bug_data.get('labels', []),
            components=bug_data.get('components', [])
        )
        
        return self.constructor.create_node(NodeType.BUG, asdict(bug_node))
    
    # Relationship creation methods
    
    def _create_calls_edge(self, source_id: str, target_id: str, **properties) -> Dict[str, Any]:
        """Create a CALLS relationship between two method nodes.
        
        Args:
            source_id: ID of the source method node
            target_id: ID of the target method node
            **properties: Additional properties for the relationship
            
        Returns:
            Created relationship
        """
        return self.constructor.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type=RelationshipType.CALLS,
            properties=properties
        )
    
    def _create_tested_by_edge(self, source_id: str, target_id: str, **properties) -> Dict[str, Any]:
        """Create a TESTED_BY relationship between a method/class and a test.
        
        Args:
            source_id: ID of the source node (method or class)
            target_id: ID of the test node
            **properties: Additional properties for the relationship
            
        Returns:
            Created relationship
        """
        return self.constructor.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type=RelationshipType.TESTED_BY,
            properties=properties
        )
    
    def _create_fixes_edge(self, source_id: str, target_id: str, **properties) -> Dict[str, Any]:
        """Create a FIXES relationship between a commit/change and a bug.
        
        Args:
            source_id: ID of the source node (commit or change)
            target_id: ID of the bug node
            **properties: Additional properties for the relationship
            
        Returns:
            Created relationship
        """
        return self.constructor.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type=RelationshipType.FIXES,
            properties=properties
        )
    
    def _create_fails_edge(self, source_id: str, target_id: str, **properties) -> Dict[str, Any]:
        """Create a FAILS relationship between a test and a bug.
        
        Args:
            source_id: ID of the test node
            target_id: ID of the bug node
            **properties: Additional properties for the relationship
            
        Returns:
            Created relationship
        """
        return self.constructor.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type=RelationshipType.FAILS,
            properties=properties
        )
    
    def add_codebase(self, codebase_path: str) -> None:
        """Add an entire codebase to the knowledge graph.
        
        Args:
            codebase_path: Path to the root of the codebase
        """
        self.build_graph(codebase_path)
    
    def add_test_suite(self, test_path: str) -> None:
        """Add a test suite to the knowledge graph.
        
        Args:
            test_path: Path to the test directory or file
        """
        # TODO: Implement test suite import
        pass
    
    def find_impacted_tests(self, file_path: str) -> List[Dict[str, Any]]:
        """Find tests that might be impacted by changes to a file.
        
        Args:
            file_path: Path to the changed file
            
        Returns:
            List of impacted tests with relevance scores
        """
        # TODO: Implement test impact analysis
        return []
    
    def find_similar_bugs(self, bug_description: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar bugs based on description.
        
        Args:
            bug_description: Description of the new bug
            top_k: Number of similar bugs to return
            
        Returns:
            List of similar bugs with similarity scores
        """
        # TODO: Implement bug similarity search
        return []
    
    def get_test_coverage(self, test_id: str) -> Dict[str, Any]:
        """Get code coverage information for a test.
        
        Args:
            test_id: ID of the test
            
        Returns:
            Coverage information including covered lines and methods
        """
        # TODO: Implement test coverage retrieval
        return {}
    
    def get_code_context(self, file_path: str, line_number: int) -> Dict[str, Any]:
        """Get context information for a specific code location.
        
        Args:
            file_path: Path to the source file
            line_number: Line number in the file
            
        Returns:
            Context information including method, class, and related tests
        """
        # TODO: Implement code context retrieval
        return {}
    
    def export_graph(self, output_path: str, format: str = "json") -> None:
        """Export the knowledge graph to a file.
        
        Args:
            output_path: Path to the output file
            format: Export format (json, graphml, etc.)
        """
        # TODO: Implement graph export
        pass
    
    def import_graph(self, input_path: str, format: str = "json") -> None:
        """Import a knowledge graph from a file.
        
        Args:
            input_path: Path to the input file
            format: Import format (json, graphml, etc.)
        """
        # TODO: Implement graph import
        pass
    
    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'constructor') and hasattr(self.constructor, 'close'):
            self.constructor.close()
        self.logger.info("Knowledge Graph connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the knowledge graph
    config = KnowledgeGraphConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        database="testagentx"
    )
    
    with KnowledgeGraph(config) as kg:
        # Example: Add a codebase
        kg.add_codebase("/path/to/your/codebase")
        
        # Example: Find impacted tests
        impacted_tests = kg.find_impacted_tests("/path/to/changed/file.java")
        print(f"Impacted tests: {impacted_tests}")
        
        # Example: Find similar bugs
        similar_bugs = kg.find_similar_bugs("Null pointer exception in user authentication")
        print(f"Similar bugs: {similar_bugs}")
