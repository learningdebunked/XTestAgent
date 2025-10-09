"""
Test configuration for Knowledge Graph tests.

This file contains fixtures and configuration for testing the Knowledge Graph layer.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.layer5_knowledge_graph.knowledge_graph import KnowledgeGraph, KnowledgeGraphConfig

# Test configuration
TEST_DB_NAME = "test_knowledge_graph"
TEST_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
TEST_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration for the Knowledge Graph."""
    return KnowledgeGraphConfig(
        neo4j_uri=TEST_NEO4J_URI,
        neo4j_user=TEST_NEO4J_USER,
        neo4j_password=TEST_NEO4J_PASSWORD,
        database=TEST_DB_NAME,
        clear_existing=True
    )

@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver for testing."""
    with patch('neo4j.GraphDatabase.driver') as mock_driver:
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value = mock_session
        yield mock_driver

@pytest.fixture
def sample_java_project(tmp_path):
    """Create a sample Java project structure for testing."""
    # Create source files
    src_dir = tmp_path / "src" / "main" / "java" / "com" / "example"
    src_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample Java class
    (src_dir / "Calculator.java").write_text(
        """
        package com.example;
        
        /**
         * A simple calculator class.
         */
        public class Calculator {
            /**
             * Adds two numbers.
             * @param a First number
             * @param b Second number
             * @return Sum of a and b
             */
            public int add(int a, int b) {
                return a + b;
            }
            
            /**
             * Subtracts b from a.
             * @param a First number
             * @param b Number to subtract
             * @return Result of a - b
             */
            public int subtract(int a, int b) {
                return a - b;
            }
        }
        """
    )
    
    # Test directory
    test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample test file
    (test_dir / "CalculatorTest.java").write_text(
        """
        package com.example;
        
        import org.junit.Test;
        import static org.junit.Assert.*;
        
        public class CalculatorTest {
            @Test
            public void testAdd() {
                Calculator calc = new Calculator();
                assertEquals(5, calc.add(2, 3));
            }
            
            @Test
            public void testSubtract() {
                Calculator calc = new Calculator();
                assertEquals(1, calc.subtract(3, 2));
            }
        }
        """
    )
    
    return tmp_path

@pytest.fixture
def sample_python_project(tmp_path):
    """Create a sample Python project structure for testing."""
    # Create source files
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    
    # Sample Python module
    (src_dir / "calculator.py").write_text(
        """
        """"A simple calculator module."""
        
        def add(a: int, b: int) -> int:
            """Add two numbers.
            
            Args:
                a: First number
                b: Second number
                
            Returns:
                Sum of a and b
            """
            return a + b
            
        def subtract(a: int, b: int) -> int:
            """Subtract b from a.
            
            Args:
                a: First number
                b: Number to subtract
                
            Returns:
                Result of a - b
            """
            return a - b
        """
    )
    
    # Test directory
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    
    # Sample test file
    (test_dir / "test_calculator.py").write_text(
        """
        import unittest
        from src.calculator import add, subtract
        
        class TestCalculator(unittest.TestCase):
            def test_add(self):
                self.assertEqual(add(2, 3), 5)
                
            def test_subtract(self):
                self.assertEqual(subtract(3, 2), 1)
                
        if __name__ == '__main__':
            unittest.main()
        """
    )
    
    return tmp_path
