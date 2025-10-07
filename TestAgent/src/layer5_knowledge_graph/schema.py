"""
Knowledge Graph Schema Definition for TestAgentX.

Defines the node types, relationships, and constraints for the Neo4j knowledge graph.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    METHOD = "Method"
    CLASS = "Class"
    TEST = "Test"
    BUG = "Bug"
    FILE = "File"
    PACKAGE = "Package"
    TEST_CASE = "TestCase"


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    # Structural relationships
    CONTAINS = "CONTAINS"  # Package -> Class, Class -> Method, File -> Class, etc.
    CALLS = "CALLS"  # Method -> Method
    IMPLEMENTS = "IMPLEMENTS"  # Class -> Interface
    EXTENDS = "EXTENDS"  # Class -> Class
    
    # Test relationships
    TESTS = "TESTS"  # Test -> Method/Class
    COVERS = "COVERS"  # TestCase -> CodeElement
    VERIFIES = "VERIFIES"  # Test -> Requirement/Assertion
    
    # Bug relationships
    FIXES = "FIXES"  # Commit/PR -> Bug
    INTRODUCES = "INTRODUCES"  # Commit/PR -> Bug
    AFFECTS = "AFFECTS"  # Bug -> Method/Class
    
    # Change impact
    DEPENDS_ON = "DEPENDS_ON"  # Method/Class -> Method/Class
    IMPACTED_BY = "IMPACTED_BY"  # Method/Class -> Method/Class
    
    # Similarity
    SIMILAR_TO = "SIMILAR_TO"  # Any node -> Any node (with similarity score)


@dataclass
class NodeProperties:
    """Base class for node properties."""
    node_type: NodeType
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j query parameters."""
        return {
            "labels": [self.node_type.value] + self.labels,
            "properties": self.properties
        }


@dataclass
class MethodNode(NodeProperties):
    """Properties for a method node."""
    def __post_init__(self):
        self.node_type = NodeType.METHOD
        self.properties.setdefault("type", "method")
        
    name: str
    signature: str
    return_type: str
    parameters: List[Dict[str, str]]
    visibility: str = "public"
    is_static: bool = False
    is_abstract: bool = False
    is_constructor: bool = False
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    file_path: Optional[str] = None
    source_code: Optional[str] = None
    documentation: Optional[str] = None
    complexity: Optional[float] = None
    vector_embedding: Optional[List[float]] = None


@dataclass
class ClassNode(NodeProperties):
    """Properties for a class node."""
    def __post_init__(self):
        self.node_type = NodeType.CLASS
        self.properties.setdefault("type", "class")
    
    name: str
    package: str
    is_interface: bool = False
    is_abstract: bool = False
    file_path: Optional[str] = None
    documentation: Optional[str] = None
    vector_embedding: Optional[List[float]] = None


@dataclass
class TestNode(NodeProperties):
    """Properties for a test node."""
    def __post_init__(self):
        self.node_type = NodeType.TEST
        self.properties.setdefault("type", "test")
    
    name: str
    class_name: str
    package: str
    file_path: str
    test_framework: str  # e.g., JUnit, pytest, etc.
    is_parameterized: bool = False
    execution_time: Optional[float] = None
    last_run: Optional[datetime] = None
    last_status: Optional[str] = None  # PASS, FAIL, SKIPPED, etc.
    flakiness_score: Optional[float] = None
    vector_embedding: Optional[List[float]] = None


@dataclass
class BugNode(NodeProperties):
    """Properties for a bug node."""
    def __post_init__(self):
        self.node_type = NodeType.BUG
        self.properties.setdefault("type", "bug")
    
    bug_id: str  # e.g., JIRA-1234 or GitHub#123
    title: str
    description: str
    status: str  # OPEN, IN_PROGRESS, RESOLVED, etc.
    priority: Optional[str] = None
    severity: Optional[str] = None
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    vector_embedding: Optional[List[float]] = None


@dataclass
class FileNode(NodeProperties):
    """Properties for a file node."""
    def __post_init__(self):
        self.node_type = NodeType.FILE
        self.properties.setdefault("type", "file")
    
    path: str
    file_name: str
    extension: str
    language: str
    size_bytes: int
    last_modified: datetime
    checksum: str
    vector_embedding: Optional[List[float]] = None


@dataclass
class PackageNode(NodeProperties):
    """Properties for a package node."""
    def __post_init__(self):
        self.node_type = NodeType.PACKAGE
        self.properties.setdefault("type", "package")
    
    name: str
    path: str
    language: str
    vector_embedding: Optional[List[float]] = None


@dataclass
class TestCaseNode(NodeProperties):
    """Properties for a test case node."""
    def __post_init__(self):
        self.node_type = NodeType.TEST_CASE
        self.properties.setdefault("type", "test_case")
    
    name: str
    test_id: str
    file_path: str
    test_suite: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    vector_embedding: Optional[List[float]] = None


class SchemaManager:
    """Manages the Neo4j database schema and constraints."""
    
    CONSTRAINT_QUERIES = [
        # Unique constraints
        "CREATE CONSTRAINT method_signature IF NOT EXISTS 
         FOR (m:Method) REQUIRE (m.signature) IS NODE KEY",
         
        "CREATE CONSTRAINT class_name IF NOT EXISTS 
         FOR (c:Class) REQUIRE (c.package, c.name) IS NODE KEY",
         
        "CREATE CONSTRAINT test_identity IF NOT EXISTS 
         FOR (t:Test) REQUIRE (t.package, t.class_name, t.name) IS NODE KEY",
         
        "CREATE CONSTRAINT bug_id IF NOT EXISTS 
         FOR (b:Bug) REQUIRE b.bug_id IS NODE KEY",
         
        "CREATE CONSTRAINT file_path IF NOT EXISTS 
         FOR (f:File) REQUIRE f.path IS NODE KEY",
         
        # Indexes for faster lookups
        "CREATE INDEX method_name IF NOT EXISTS 
         FOR (m:Method) ON (m.name)",
         
        "CREATE INDEX class_name_simple IF NOT EXISTS 
         FOR (c:Class) ON (c.name)",
         
        # Index for vector similarity search (if using Neo4j's vector search)
        "CREATE VECTOR INDEX method_embeddings IF NOT EXISTS 
         FOR (m:Method) ON m.vector_embedding 
         OPTIONS {indexConfig: {
           `vector.dimensions`: 768,
           `vector.similarity_function`: 'cosine'
         }}",
    ]
    
    @classmethod
    def get_schema_queries(cls) -> List[str]:
        """Get all schema creation queries."""
        return cls.CONSTRAINT_QUERIES
    
    @classmethod
    def drop_all_constraints(cls, session) -> None:
        """Drop all constraints and indexes (for testing/reset)."""
        # Get all constraints and indexes
        result = session.run("""
        CALL db.constraints() YIELD name
        CALL {
          WITH name
          CALL apoc.schema.assert({}, {}, true) YIELD label, key
          RETURN collect(name) AS dropped
        }
        RETURN dropped
        """)
        
        # Also drop any vector indexes
        session.run("""
        CALL db.index.vector.list() YIELD name
        CALL db.index.drop(name)
        RETURN count(*) AS indexesDropped
        """)
