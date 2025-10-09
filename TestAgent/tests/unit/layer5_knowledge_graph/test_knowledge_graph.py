"""
Unit tests for the KnowledgeGraph class.

These tests verify the core functionality of the KnowledgeGraph class,
including initialization, codebase import, and query operations.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.layer5_knowledge_graph.knowledge_graph import KnowledgeGraph, KnowledgeGraphConfig
from src.layer5_knowledge_graph.schema import NodeType, RelationshipType

class TestKnowledgeGraph:
    """Test cases for the KnowledgeGraph class."""
    
    def test_initialization(self, test_config, mock_neo4j_driver):
        """Test that the KnowledgeGraph initializes correctly."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            # Create a mock session
            mock_session = MagicMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # Initialize the knowledge graph
            kg = KnowledgeGraph(test_config)
            
            # Verify the driver was created with the correct URI and auth
            mock_driver.assert_called_once_with(
                test_config.neo4j_uri,
                auth=(test_config.neo4j_user, test_config.neo4j_password)
            )
            
            # Verify the session was created with the correct database
            mock_driver.return_value.session.assert_called_once_with(database=test_config.database)
            
            # Verify the context manager works
            with kg:
                assert kg.session is not None
            
            # Verify the session is closed when exiting the context
            mock_session.close.assert_called_once()
    
    def test_add_codebase_java(self, test_config, sample_java_project, mock_neo4j_driver):
        """Test adding a Java codebase to the knowledge graph."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver, \
             patch('src.layer5_knowledge_graph.data_importer.JavaImporter') as mock_java_importer:
            
            # Set up mocks
            mock_session = MagicMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # Create a mock importer
            mock_importer_instance = MagicMock()
            mock_java_importer.return_value = mock_importer_instance
            
            # Initialize the knowledge graph
            with KnowledgeGraph(test_config) as kg:
                # Add the codebase
                kg.add_codebase(sample_java_project)
                
                # Verify the importer was called with the correct path
                mock_java_importer.assert_called_once_with(kg)
                mock_importer_instance.import_codebase.assert_called_once_with(sample_java_project)
    
    def test_add_codebase_python(self, test_config, sample_python_project, mock_neo4j_driver):
        """Test adding a Python codebase to the knowledge graph."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver, \
             patch('src.layer5_knowledge_graph.data_importer.PythonImporter') as mock_python_importer:
            
            # Set up mocks
            mock_session = MagicMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # Create a mock importer
            mock_importer_instance = MagicMock()
            mock_python_importer.return_value = mock_importer_instance
            
            # Initialize the knowledge graph
            with KnowledgeGraph(test_config) as kg:
                # Add the codebase
                kg.add_codebase(sample_python_project)
                
                # Verify the importer was called with the correct path
                mock_python_importer.assert_called_once_with(kg)
                mock_importer_instance.import_codebase.assert_called_once_with(sample_python_project)
    
    def test_find_impacted_tests(self, test_config, mock_neo4j_driver):
        """Test finding tests impacted by code changes."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            # Set up mocks
            mock_session = MagicMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # Mock query result
            mock_result = MagicMock()
            mock_session.run.return_value = mock_result
            mock_result.data.return_value = [
                {"test_path": "test/CalculatorTest.java", "test_name": "testAdd"},
                {"test_path": "test/CalculatorTest.java", "test_name": "testSubtract"}
            ]
            
            # Initialize the knowledge graph
            with KnowledgeGraph(test_config) as kg:
                # Find impacted tests
                changed_files = ["src/main/java/com/example/Calculator.java"]
                impacted_tests = kg.find_impacted_tests(changed_files)
                
                # Verify the query was executed
                assert mock_session.run.called
                
                # Verify the results
                assert len(impacted_tests) == 2
                assert any(t["test_name"] == "testAdd" for t in impacted_tests)
                assert any(t["test_name"] == "testSubtract" for t in impacted_tests)
    
    def test_find_similar_bugs(self, test_config, mock_neo4j_driver):
        """Test finding similar bugs using semantic search."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            # Set up mocks
            mock_session = MagicMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # Mock query result
            mock_result = MagicMock()
            mock_session.run.return_value = mock_result
            mock_result.data.return_value = [
                {"bug_id": "BUG-123", "similarity": 0.95, "title": "NullPointerException in Calculator"},
                {"bug_id": "BUG-456", "similarity": 0.85, "title": "Arithmetic error in calculation"}
            ]
            
            # Initialize the knowledge graph
            with KnowledgeGraph(test_config) as kg:
                # Find similar bugs
                bug_description = "Getting NullPointer when calling add method"
                similar_bugs = kg.find_similar_bugs(bug_description, limit=2)
                
                # Verify the query was executed
                assert mock_session.run.called
                
                # Verify the results
                assert len(similar_bugs) == 2
                assert similar_bugs[0]["bug_id"] == "BUG-123"
                assert similar_bugs[0]["similarity"] == 0.95
                assert "NullPointer" in similar_bugs[0]["title"]
    
    def test_get_code_context(self, test_config, mock_neo4j_driver):
        """Test getting context for a code location."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            # Set up mocks
            mock_session = MagicMock()
            mock_driver.return_value.session.return_value = mock_session
            
            # Mock query result
            mock_result = MagicMock()
            mock_session.run.return_value = mock_result
            mock_result.single.return_value = {
                "file_path": "src/main/java/com/example/Calculator.java",
                "node_type": "Method",
                "name": "add",
                "signature": "add(int a, int b)",
                "docstring": "Adds two numbers.",
                "source_code": "public int add(int a, int b) {\n    return a + b;\n}"
            }
            
            # Initialize the knowledge graph
            with KnowledgeGraph(test_config) as kg:
                # Get code context
                context = kg.get_code_context(
                    file_path="src/main/java/com/example/Calculator.java",
                    line_number=10
                )
                
                # Verify the query was executed
                assert mock_session.run.called
                
                # Verify the results
                assert context["name"] == "add"
                assert context["node_type"] == "Method"
                assert "Adds two numbers" in context["docstring"]
                assert "return a + b" in context["source_code"]
