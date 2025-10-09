# TestAgentX API Documentation

## Table of Contents
- [Layer 1: Preprocessing](#layer-1-preprocessing)
- [Layer 2: Test Generation](#layer-2-test-generation)
- [Layer 3: Fuzzy Validation](#layer-3-fuzzy-validation)
- [Layer 4: Patch Regression](#layer-4-patch-regression)
- [Layer 5: Knowledge Graph](#layer-5-knowledge-graph)
- [Core Modules](#core-modules)

## Layer 1: Preprocessing

### `class Defects4JLoader`
Handles loading and processing of Defects4J bug data.

```python
class Defects4JLoader:
    def __init__(self, d4j_home: str):
        """Initialize with path to Defects4J installation."""
        
    def load_bug(self, project: str, bug_id: int) -> Dict[str, Any]:
        """Load bug data for a specific project and bug ID."""
        
    def get_all_bugs(self) -> List[Dict[str, Any]]:
        """Get metadata for all available bugs."""
```

### `class ASTCFGGenerator`
Generates Abstract Syntax Tree (AST) and Control Flow Graph (CFG) from source code.

```python
class ASTCFGGenerator:
    def generate_ast(self, source_code: str, language: str) -> Dict:
        """Generate AST from source code.
        
        Args:
            source_code: Source code as string
            language: Programming language ('java', 'python', etc.)
            
        Returns:
            Dictionary representing the AST
        """
        
    def generate_cfg(self, ast: Dict) -> Dict:
        """Generate CFG from AST.
        
        Args:
            ast: Abstract Syntax Tree
            
        Returns:
            Dictionary representing the CFG
        """
```

### `class CodeEncoder`
Encodes source code into numerical representations.

```python
class CodeEncoder:
    def __init__(self, model_name: str = 'codebert'):
        """Initialize with specified pre-trained model."""
        
    def encode(self, code: str) -> np.ndarray:
        """Encode source code into embedding vector.
        
        Args:
            code: Source code to encode
            
        Returns:
            numpy.ndarray: Numerical representation of the code
        """
```

### `class SemanticDiffAnalyzer`
Performs semantic diffing between code versions.

```python
class SemanticDiffAnalyzer:
    def diff(self, old_code: str, new_code: str, language: str) -> Dict:
        """Compute semantic differences between two code versions.
        
        Args:
            old_code: Original code version
            new_code: New code version
            language: Programming language
            
        Returns:
            Dictionary containing diff information
        """
```

## Layer 2: Test Generation

### `class LLMTestGenerationAgent`
Generates test cases using Large Language Models.

```python
class LLMTestGenerationAgent:
    def __init__(self, model_name: str = 'gpt-4', temperature: float = 0.7):
        """Initialize with LLM configuration."""
        
    def generate_tests(self, code: str, test_framework: str = 'pytest') -> str:
        """Generate test cases for the given code.
        
        Args:
            code: Source code to generate tests for
            test_framework: Testing framework to use
            
        Returns:
            Generated test code as string
        """
```

### `class RLPrioritizationAgent`
Uses Reinforcement Learning to prioritize test cases.

```python
class RLPrioritizationAgent:
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        """Initialize RL agent with state and action dimensions."""
        
    def prioritize_tests(self, tests: List[Dict], state: np.ndarray) -> List[Dict]:
        """Prioritize test cases based on current state.
        
        Args:
            tests: List of test cases
            state: Current system state
            
        Returns:
            Prioritized list of test cases
        """
```

## Layer 3: Fuzzy Validation

### `class FuzzyAssertionAgent`
Performs fuzzy matching and validation of test outputs.

```python
class FuzzyAssertionAgent:
    def __init__(self, threshold: float = 0.8):
        """Initialize with similarity threshold."""
        
    def validate_output(self, expected: Any, actual: Any) -> bool:
        """Validate if actual output matches expected with fuzzy matching.
        
        Args:
            expected: Expected output
            actual: Actual output
            
        Returns:
            bool: True if outputs match according to fuzzy criteria
        """
```

## Layer 4: Patch Regression

### `class PatchVerificationAgent`
Verifies patches and ensures they don't introduce regressions.

```python
class PatchVerificationAgent:
    def __init__(self, epsilon: float = 0.1):
        """Initialize with epsilon threshold for trace comparison."""
        
    def verify_patch(
        self,
        original_code: str,
        patched_code: str,
        test_cases: List[Dict]
    ) -> Dict[str, Any]:
        """Verify that a patch doesn't introduce regressions.
        
        Args:
            original_code: Original code version
            patched_code: Patched code version
            test_cases: List of test cases to verify
            
        Returns:
            Dictionary with verification results
        """
```

### `class RegressionSentinelAgent`
Monitors for and detects regressions in the codebase.

```python
class RegressionSentinelAgent:
    def detect_regressions(self, commit_range: str) -> List[Dict]:
        """Detect regressions in the specified commit range.
        
        Args:
            commit_range: Git commit range (e.g., 'HEAD~3..HEAD')
            
        Returns:
            List of detected regressions with details
        """
```

## Layer 5: Knowledge Graph

The Knowledge Graph layer provides a comprehensive representation of the codebase, tests, and their relationships, enabling powerful code analysis and test impact analysis.

### `class KnowledgeGraph`
Main entry point for interacting with the Knowledge Graph.

```python
class KnowledgeGraph:
    def __init__(self, config: KnowledgeGraphConfig):
        """Initialize the Knowledge Graph with configuration.
        
        Args:
            config: Configuration for the Knowledge Graph
        """
        
    def add_codebase(self, project_path: str, language: str = None) -> None:
        """Import a codebase into the knowledge graph.
        
        Args:
            project_path: Path to the project root directory
            language: Programming language (auto-detected if None)
        """
        
    def find_impacted_tests(self, changed_files: List[str]) -> List[Dict]:
        """Find tests impacted by changes to the specified files.
        
        Args:
            changed_files: List of file paths that were modified
            
        Returns:
            List of impacted tests with metadata
        """
        
    def find_similar_bugs(self, description: str, limit: int = 5) -> List[Dict]:
        """Find similar bugs using semantic search.
        
        Args:
            description: Description of the bug or issue
            limit: Maximum number of results to return
            
        Returns:
            List of similar bugs with similarity scores
        """
        
    def get_code_context(self, file_path: str, line_number: int) -> Dict:
        """Get context for a specific code location.
        
        Args:
            file_path: Path to the source file
            line_number: Line number in the file
            
        Returns:
            Dictionary containing code context information
        """
        
    def get_test_coverage(self, test_path: str = None) -> Dict:
        """Get code coverage information for tests.
        
        Args:
            test_path: Optional path to a specific test file
            
        Returns:
            Coverage information including line coverage and branch coverage
        """
```

### `class QueryEngine`
Executes complex queries against the knowledge graph.

```python
class QueryEngine:
    def __init__(self, session):
        """Initialize with a Neo4j session."""
        
    def execute_query(self, query_type: QueryType, params: Dict = None) -> List[Dict]:
        """Execute a predefined query.
        
        Args:
            query_type: Type of query to execute
            params: Query parameters
            
        Returns:
            Query results
        """
        
    def execute_raw_query(self, cypher_query: str, params: Dict = None) -> List[Dict]:
        """Execute a raw Cypher query.
        
        Args:
            cypher_query: Cypher query string
            params: Query parameters
            
        Returns:
            Query results
        """
```

### `class DataImporter`
Base class for importing code from different programming languages.

```python
class DataImporter(ABC):
    @abstractmethod
    def import_codebase(self, project_path: str) -> None:
        """Import a codebase into the knowledge graph."""
        
    @classmethod
    def get_importer(cls, language: str, kg) -> 'DataImporter':
        """Get an importer for the specified language."""
```

### `class JavaImporter`, `class PythonImporter`, `class JavaScriptImporter`
Language-specific importers that extend `DataImporter`.

### `class KnowledgeGraphConfig`
Configuration for the Knowledge Graph.

```python
@dataclass
class KnowledgeGraphConfig:
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
```

### Example Usage

```python
# Initialize the knowledge graph
config = KnowledgeGraphConfig(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

with KnowledgeGraph(config) as kg:
    # Import a codebase
    kg.add_codebase("/path/to/your/project")
    
    # Find tests impacted by changes
    impacted_tests = kg.find_impacted_tests(["src/main/java/com/example/Calculator.java"])
    print(f"Impacted tests: {impacted_tests}")
    
    # Find similar bugs
    similar_bugs = kg.find_similar_bugs("NullPointerException in user authentication")
    print(f"Similar bugs: {similar_bugs}")
    
    # Get code coverage
    coverage = kg.get_test_coverage()
    print(f"Code coverage: {coverage['line_coverage']}%")
```

### Query Types

The following query types are supported by the `QueryEngine`:

- `FIND_TESTS_FOR_CODE`: Find tests that cover specific code
- `FIND_CODE_COVERED_BY_TEST`: Find code covered by a specific test
- `FIND_IMPACTED_TESTS`: Find tests impacted by code changes
- `FIND_SIMILAR_BUGS`: Find similar bugs using semantic search
- `GET_CODE_COVERAGE`: Get code coverage information
- `FIND_DUPLICATE_TESTS`: Find potentially duplicate tests

## Core Modules

### `class BaseAgent`
Base class for all agents in the framework.

```python
class BaseAgent:
    def __init__(self, config: Dict = None):
        """Initialize with optional configuration."""
        
    def validate_config(self) -> bool:
        """Validate agent configuration."""
```

### `class AgentOrchestrator`
Orchestrates the execution of multiple agents.

```python
class AgentOrchestrator:
    def __init__(self, agents: List[BaseAgent]):
        """Initialize with list of agents to orchestrate."""
        
    def run_pipeline(self, input_data: Any) -> Dict:
        """Execute the agent pipeline on input data."""
```

## Usage Examples

### Example 1: Generating Tests with LLM
```python
agent = LLMTestGenerationAgent()
tests = agent.generate_tests("""
def add(a, b):
    return a + b
""")
print(tests)
```

### Example 2: Verifying a Patch
```python
verifier = PatchVerificationAgent()
result = verifier.verify_patch(
    original_code="def add(a, b): return a + b",
    patched_code="def add(a, b): return a + b  # Fixed edge case",
    test_cases=[{"input": (1, 2), "expected": 3}]
)
print(f"Patch verified: {result['is_effective']}")
```

## Error Handling

All API methods may raise the following exceptions:
- `ValueError`: For invalid input parameters
- `RuntimeError`: For execution errors
- `ImportError`: For missing dependencies
- `TimeoutError`: For operations that exceed time limits
