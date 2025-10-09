"""
Query Engine for Knowledge Graph.

Provides high-level querying capabilities for the knowledge graph.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
from neo4j import GraphDatabase

class QueryType(str, Enum):
    """Types of supported queries."""
    FIND_TESTS_FOR_CODE = "find_tests_for_code"
    FIND_CODE_FOR_TESTS = "find_code_for_tests"
    FIND_IMPACTED_TESTS = "find_impacted_tests"
    FIND_SIMILAR_BUGS = "find_similar_bugs"
    GET_CODE_COVERAGE = "get_code_coverage"
    GET_CODE_CONTEXT = "get_code_context"
    FIND_IMPACTED_CODE = "find_impacted_code"
    FIND_TEST_DUPLICATES = "find_test_duplicates"
    
@dataclass
class QueryResult:
    """Result of a knowledge graph query."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms
        }

class QueryEngine:
    """Handles complex queries against the knowledge graph."""
    
    def __init__(self, driver: GraphDatabase.driver, database: str = "testagentx"):
        """Initialize the query engine.
        
        Args:
            driver: Neo4j driver instance
            database: Database name
        """
        self.driver = driver
        self.database = database
        self.logger = logging.getLogger(f"{__name__}.QueryEngine")
        
        # Pre-compiled queries
        self._queries = {
            QueryType.FIND_TESTS_FOR_CODE: self._find_tests_for_code,
            QueryType.FIND_CODE_FOR_TESTS: self._find_code_for_tests,
            QueryType.FIND_IMPACTED_TESTS: self._find_impacted_tests,
            QueryType.FIND_SIMILAR_BUGS: self._find_similar_bugs,
            QueryType.GET_CODE_COVERAGE: self._get_code_coverage,
            QueryType.GET_CODE_CONTEXT: self._get_code_context,
            QueryType.FIND_IMPACTED_CODE: self._find_impacted_code,
            QueryType.FIND_TEST_DUPLICATES: self._find_test_duplicates
        }
    
    def execute_query(
        self, 
        query_type: QueryType, 
        params: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute a pre-defined query.
        
        Args:
            query_type: Type of query to execute
            params: Query parameters
            
        Returns:
            Query result
        """
        if query_type not in self._queries:
            return QueryResult(
                success=False,
                error=f"Unknown query type: {query_type}"
            )
        
        try:
            import time
            start_time = time.time()
            
            # Execute the query
            result = self._queries[query_type](params or {})
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            return QueryResult(
                success=True,
                data=result,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error executing query {query_type}: {e}", exc_info=True)
            return QueryResult(
                success=False,
                error=str(e)
            )
    
    def execute_cypher(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute a raw Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query result
        """
        try:
            import time
            start_time = time.time()
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                data = [dict(record) for record in result]
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                return QueryResult(
                    success=True,
                    data=data,
                    execution_time_ms=execution_time_ms
                )
                
        except Exception as e:
            self.logger.error(f"Error executing Cypher query: {e}", exc_info=True)
            return QueryResult(
                success=False,
                error=str(e)
            )
    
    # Query implementations
    def _find_tests_for_code(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find tests that cover specific code elements."""
        query = """
        MATCH (c:Class {name: $class_name})<-[:CONTAINS]-(m:Method {name: $method_name})
        OPTIONAL MATCH (m)<-[:COVERS]-(t:Test)
        RETURN m as method, collect(DISTINCT t) as tests
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def _find_code_for_tests(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find code elements covered by specific tests."""
        test_name = params.get('test_name')
        query = """
        MATCH (t:Test {name: $test_name})-[:COVERS]->(m:Method)-[:CONTAINS]->(c:Class)
        RETURN t as test, collect(DISTINCT {method: m, class: c}) as covered_code
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, {'test_name': test_name})
            return [dict(record) for record in result]
    
    def _find_impacted_tests(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find tests that might be impacted by code changes."""
        file_path = params.get('file_path')
        query = """
        MATCH (f:File {path: $file_path})<-[:CONTAINS]-(p:Package)
        OPTIONAL MATCH (p)-[:CONTAINS]->(c:Class)<-[:CONTAINS]-(m:Method)
        OPTIONAL MATCH (m)<-[:COVERS]-(t:Test)
        RETURN DISTINCT t as test, count(m) as impacted_methods
        ORDER BY impacted_methods DESC
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, {'file_path': file_path})
            return [dict(record) for record in result]
    
    def _find_similar_bugs(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find bugs similar to the given description."""
        description = params.get('description', '')
        limit = params.get('limit', 5)
        
        # This is a simplified example. In practice, you would use:
        # 1. Text embedding similarity
        # 2. Graph-based similarity
        # 3. Or a combination of both
        
        query = """
        MATCH (b:Bug)
        WHERE toLower(b.description) CONTAINS toLower($keyword)
        RETURN b as bug, 1.0 as similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        # Simple keyword matching as fallback
        keywords = set(description.lower().split())
        keywords = {w for w in keywords if len(w) > 3}  # Filter out short words
        
        results = []
        for keyword in keywords:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {'keyword': keyword, 'limit': limit})
                results.extend([dict(record) for record in result])
        
        # Deduplicate and sort results
        seen = set()
        unique_results = []
        for r in results:
            bug_id = r['bug']['bug_id']
            if bug_id not in seen:
                seen.add(bug_id)
                unique_results.append(r)
        
        return unique_results[:limit]
    
    def _get_code_coverage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get code coverage information for a test."""
        test_id = params.get('test_id')
        query = """
        MATCH (t:Test {test_id: $test_id})-[:COVERS]->(m:Method)-[:CONTAINS]->(c:Class)
        RETURN t as test, 
               collect(DISTINCT {method: m, class: c}) as covered_code,
               count(DISTINCT m) as method_count
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, {'test_id': test_id})
            records = [dict(record) for record in result]
            return records[0] if records else {}
    
    def _get_code_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get context information for a code location."""
        file_path = params.get('file_path')
        line_number = params.get('line_number')
        
        query = """
        MATCH (f:File {path: $file_path})<-[:CONTAINS]-(p:Package)
        OPTIONAL MATCH (p)-[:CONTAINS]->(c:Class)
        WHERE c.start_line <= $line_number AND c.end_line >= $line_number
        WITH p, c
        OPTIONAL MATCH (c)-[:CONTAINS]->(m:Method)
        WHERE m.start_line <= $line_number AND m.end_line >= $line_number
        RETURN {
            package: p,
            class: c,
            method: m,
            containing_file: f
        } as context
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, {
                'file_path': file_path,
                'line_number': line_number
            })
            records = [dict(record) for record in result]
            return records[0] if records else {}
    
    def _find_impacted_code(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find code that might be impacted by changes to a file."""
        file_path = params.get('file_path')
        query = """
        // Find methods in the changed file
        MATCH (f:File {path: $file_path})<-[:CONTAINS]-(m:Method)
        // Find methods that call these methods
        OPTIONAL MATCH (m)<-[:CALLS]-(caller:Method)
        // Find classes that contain these methods
        OPTIONAL MATCH (caller)-[:CONTAINS]->(c:Class)
        RETURN DISTINCT {
            method: m,
            impacted_methods: collect(DISTINCT caller),
            impacted_classes: collect(DISTINCT c)
        } as impact
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, {'file_path': file_path})
            return [dict(record) for record in result]
    
    def _find_test_duplicates(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find potentially duplicate tests based on code coverage."""
        similarity_threshold = params.get('similarity_threshold', 0.8)
        
        # This is a simplified example. In practice, you would:
        # 1. Compare test coverage vectors
        # 2. Use graph-based similarity
        # 3. Or use machine learning to detect duplicates
        
        query = """
        MATCH (t1:Test)-[:COVERS]->(m:Method)<-[:COVERS]-(t2:Test)
        WHERE id(t1) < id(t2)  // Avoid duplicate pairs
        WITH t1, t2, count(m) as common_methods
        MATCH (t1)-[:COVERS]->(m1:Method)
        WITH t1, t2, common_methods, count(DISTINCT m1) as t1_methods
        MATCH (t2)-[:COVERS]->(m2:Method)
        WITH t1, t2, common_methods, t1_methods, count(DISTINCT m2) as t2_methods
        WITH t1, t2, common_methods, t1_methods, t2_methods,
             2.0 * common_methods / (t1_methods + t2_methods) as jaccard_similarity
        WHERE jaccard_similarity > $threshold
        RETURN t1.test_id as test1, 
               t2.test_id as test2, 
               jaccard_similarity as similarity,
               common_methods,
               t1_methods + t2_methods - common_methods as total_unique_methods
        ORDER BY similarity DESC
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, {'threshold': similarity_threshold})
            return [dict(record) for record in result]
