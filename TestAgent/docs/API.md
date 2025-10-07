# TestAgentX API Documentation

This document provides detailed API documentation for the TestAgentX framework.

## Core Modules

### Layer 1: Preprocessing

#### `Defects4JLoader`
```python
class Defects4JLoader:
    def get_all_bugs(project: str) -> List[str]: ...
    def load_bug(project: str, bug_id: str) -> BugInstance: ...
```

#### `ASTCFGGenerator`
```python
class ASTCFGGenerator:
    def generate_ast(self, code: str) -> Dict: ...
    def generate_cfg(self, ast: Dict) -> Dict: ...
```

### Layer 2: Test Generation

#### `LLMTestGenerationAgent`
```python
class LLMTestGenerationAgent:
    def generate_tests(self, bug: BugInstance, num_tests: int = 5) -> List[Test]: ...
```

#### `RLPrioritizationAgent`
```python
class RLPrioritizationAgent:
    def prioritize_tests(self, tests: List[Test]) -> List[Test]: ...
```

### Layer 3: Fuzzy Validation

#### `FuzzyAssertionAgent`
```python
class FuzzyAssertionAgent:
    def validate_output(self, output_buggy: str, output_fixed: str) -> ValidationResult: ...
```

### Layer 4: Patch & Regression

#### `PatchVerificationAgent`
```python
class PatchVerificationAgent:
    def verify_patch(self, bug: BugInstance, patch: Patch) -> PatchVerificationResult: ...
```

### Layer 5: Knowledge Graph

#### `KnowledgeGraphConstructor`
```python
class KnowledgeGraphConstructor:
    def build_graph(self, project_data: Dict) -> KnowledgeGraph: ...
```

## Data Structures

### `BugInstance`
```python
@dataclass
class BugInstance:
    project: str
    bug_id: str
    buggy_code: str
    fixed_code: str
    test_cases: List[TestCase]
```

### `Test`
```python
@dataclass
class Test:
    code: str
    metadata: Dict[str, Any]
```

### `ValidationResult`
```python
@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float
    details: Dict[str, Any]
```

## Utilities

### Configuration
Configuration is handled through YAML files in the `configs/` directory.

### Logging
Use the standard Python `logging` module with the logger name `testagentx`.

## Error Handling
All custom exceptions inherit from `TestAgentXError`.
