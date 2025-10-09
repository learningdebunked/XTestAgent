"""
Unit tests for the QueryEngine class.

These tests verify the functionality of the query engine for the Knowledge Graph.
"""

import pytest
from unittest.mock import MagicMock, patch, ANY

from src.layer5_knowledge_graph.query_engine import QueryEngine, QueryType

class TestQueryEngine:
    """Test cases for the QueryEngine class."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock Neo4j session."""
        return MagicMock()
    
    def test_init(self, mock_session):
        """Test that the QueryEngine initializes correctly."""
        query_engine = QueryEngine(mock_session)
        assert query_engine.session == mock_session
    
    def test_execute_query_find_tests_for_code(self, mock_session):
        """Test finding tests that cover specific code."""
        # Set up the mock result
        mock_result = MagicMock()
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = [
            {"test_path": "test/CalculatorTest.java", "test_name": "testAdd"}
        ]
        
        # Initialize the query engine
        query_engine = QueryEngine(mock_session)
        
        # Execute the query
        result = query_engine.execute_query(
            query_type=QueryType.FIND_TESTS_FOR_CODE,
            params={"file_path": "src/main/java/com/example/Calculator.java", "line_number": 10}
        )
        
        # Verify the result
        assert len(result) == 1
        assert result[0]["test_name"] == "testAdd"
        assert "test/CalculatorTest.java" in result[0]["test_path"]
        
        # Verify the query was executed with the correct parameters
        mock_session.run.assert_called_once_with(ANY, {
            "file_path": "src/main/java/com/example/Calculator.java",
            "line_number": 10
        })
    
    def test_execute_query_find_code_covered_by_test(self, mock_session):
        """Test finding code covered by a specific test."""
        # Set up the mock result
        mock_result = MagicMock()
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = [
            {"file_path": "src/main/java/com/example/Calculator.java", "line_numbers": [8, 9, 10]}
        ]
        
        # Initialize the query engine
        query_engine = QueryEngine(mock_session)
        
        # Execute the query
        result = query_engine.execute_query(
            query_type=QueryType.FIND_CODE_COVERED_BY_TEST,
            params={"test_path": "test/CalculatorTest.java", "test_name": "testAdd"}
        )
        
        # Verify the result
        assert len(result) == 1
        assert "Calculator.java" in result[0]["file_path"]
        assert result[0]["line_numbers"] == [8, 9, 10]
    
    def test_execute_query_find_impacted_tests(self, mock_session):
        """Test finding tests impacted by code changes."""
        # Set up the mock result
        mock_result = MagicMock()
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = [
            {"test_path": "test/CalculatorTest.java", "test_name": "testAdd"},
            {"test_path": "test/CalculatorTest.java", "test_name": "testSubtract"}
        ]
        
        # Initialize the query engine
        query_engine = QueryEngine(mock_session)
        
        # Execute the query
        result = query_engine.execute_query(
            query_type=QueryType.FIND_IMPACTED_TESTS,
            params={"changed_files": ["src/main/java/com/example/Calculator.java"]}
        )
        
        # Verify the result
        assert len(result) == 2
        test_names = {t["test_name"] for t in result}
        assert "testAdd" in test_names
        assert "testSubtract" in test_names
    
    def test_execute_query_find_similar_bugs(self, mock_session):
        """Test finding similar bugs using semantic search."""
        # Set up the mock result
        mock_result = MagicMock()
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = [
            {"bug_id": "BUG-123", "similarity": 0.95, "title": "NullPointerException in Calculator"}
        ]
        
        # Initialize the query engine
        query_engine = QueryEngine(mock_session)
        
        # Execute the query
        result = query_engine.execute_query(
            query_type=QueryType.FIND_SIMILAR_BUGS,
            params={
                "description": "Getting NullPointer when calling add method",
                "limit": 1
            }
        )
        
        # Verify the result
        assert len(result) == 1
        assert result[0]["bug_id"] == "BUG-123"
        assert result[0]["similarity"] == 0.95
        assert "NullPointer" in result[0]["title"]
    
    def test_execute_query_get_code_coverage(self, mock_session):
        """Test getting code coverage information."""
        # Set up the mock result
        mock_result = MagicMock()
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = [
            {
                "file_path": "src/main/java/com/example/Calculator.java",
                "total_lines": 100,
                "covered_lines": 85,
                "coverage_percentage": 85.0
            }
        ]
        
        # Initialize the query engine
        query_engine = QueryEngine(mock_session)
        
        # Execute the query
        result = query_engine.execute_query(
            query_type=QueryType.GET_CODE_COVERAGE,
            params={"file_path": "src/main/java/com/example/Calculator.java"}
        )
        
        # Verify the result
        assert len(result) == 1
        assert result[0]["file_path"] == "src/main/java/com/example/Calculator.java"
        assert result[0]["total_lines"] == 100
        assert result[0]["covered_lines"] == 85
        assert result[0]["coverage_percentage"] == 85.0
    
    def test_execute_query_find_duplicate_tests(self, mock_session):
        """Test finding duplicate tests."""
        # Set up the mock result
        mock_result = MagicMock()
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = [
            {
                "test1_path": "test/CalculatorTest.java",
                "test1_name": "testAdd",
                "test2_path": "test/AnotherTest.java",
                "test2_name": "testAddition",
                "similarity": 0.95
            }
        ]
        
        # Initialize the query engine
        query_engine = QueryEngine(mock_session)
        
        # Execute the query
        result = query_engine.execute_query(
            query_type=QueryType.FIND_DUPLICATE_TESTS,
            params={"min_similarity": 0.9}
        )
        
        # Verify the result
        assert len(result) == 1
        assert result[0]["test1_name"] == "testAdd"
        assert result[0]["test2_name"] == "testAddition"
        assert result[0]["similarity"] == 0.95
    
    def test_execute_raw_cypher_query(self, mock_session):
        """Test executing a raw Cypher query."""
        # Set up the mock result
        mock_result = MagicMock()
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = [{"count": 42}]
        
        # Initialize the query engine
        query_engine = QueryEngine(mock_session)
        
        # Execute the query
        result = query_engine.execute_raw_query(
            "MATCH (n) RETURN count(n) AS count",
            {}
        )
        
        # Verify the result
        assert result == [{"count": 42}]
        mock_session.run.assert_called_once_with("MATCH (n) RETURN count(n) AS count", {})
    
    def test_execute_query_invalid_type(self, mock_session):
        """Test that an invalid query type raises a ValueError."""
        query_engine = QueryEngine(mock_session)
        
        with pytest.raises(ValueError, match="Unknown query type: INVALID_TYPE"):
            query_engine.execute_query("INVALID_TYPE", {})
