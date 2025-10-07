# TestAgentX: Complete Usage Guide

This document provides detailed instructions for using the TestAgentX framework, including the complete pipeline execution and individual component usage.

## Table of Contents
1. [Complete Pipeline Execution](#complete-pipeline-execution)
2. [Individual Component Usage](#individual-component-usage)
3. [Configuration Guide](#configuration-guide)
4. [Troubleshooting](#troubleshooting)
5. [Advanced Usage](#advanced-usage)

## Complete Pipeline Execution

The main entry point for running the complete TestAgentX pipeline is `run_complete_pipeline.py`. This script coordinates all components and reproduces the results from the paper.

### Basic Usage

```bash
# Run with default settings (processes first 10 bugs from all projects)
python run_complete_pipeline.py

# Run in debug mode (more verbose output)
python run_complete_pipeline.py --debug

# Process specific projects with a limit on number of bugs
python run_complete_pipeline.py --projects Lang Chart --max-bugs 5

# Specify custom output directories
python run_complete_pipeline.py --output-dir ./my_results --models-dir ./my_models
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--projects` | Comma-separated list of projects to process | All projects |
| `--max-bugs` | Maximum number of bugs to process per project | 10 |
| `--output-dir` | Directory to save results | `./results` |
| `--figures-dir` | Directory to save figures | `./results/figures` |
| `--models-dir` | Directory to save trained models | `./models` |
| `--debug` | Enable debug logging | False |
| `--skip-existing` | Skip processing bugs with existing results | True |

## Individual Component Usage

### 1. Bug Ingestion

```python
from layer1_preprocessing.bug_ingestion import Defects4JLoader

# Initialize the loader
loader = Defects4JLoader()

# Get all bugs for a project
bug_ids = loader.get_all_bugs("Lang")

# Load a specific bug
bug = loader.load_bug("Lang", 1)
```

### 2. Test Generation

```python
from layer2_test_generation.llm_test_agent import LLMTestGenerationAgent

# Initialize the test generator
test_generator = LLMTestGenerationAgent(model_name="gpt-4")

# Generate tests for a bug
tests = test_generator.generate_tests(
    bug=bug,
    num_tests=5,
    temperature=0.7
)
```

### 3. Test Prioritization

```python
from layer2_test_generation.rl_prioritization_agent import RLPrioritizationAgent

# Initialize the prioritization agent
prioritizer = RLPrioritizationAgent(alpha=0.7, beta=0.3)

# Prioritize tests
priority_order = prioritizer.prioritize_tests(tests)
```

### 4. Fuzzy Validation

```python
from layer3_fuzzy_validation.fuzzy_assertion_agent import FuzzyAssertionAgent

# Initialize the fuzzy validator
validator = FuzzyAssertionAgent(threshold=0.8)

# Validate a test output
result = validator.validate_output(
    output_buggy="NullPointerException",
    output_fixed="42"
)

if result.is_valid:
    print(f"Test passed with confidence {result.confidence:.2f}")
```

## Configuration Guide

### Environment Variables

Create a `.env` file in the project root with the following variables:

```ini
# OpenAI API (for LLM test generation)
OPENAI_API_KEY=your_openai_api_key

# Defects4J path (if not in default location)
DEFECTS4J_HOME=/path/to/defects4j

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```

### Configuration Files

Configuration files are stored in the `configs/` directory:

- `experiment_config.yaml`: Main configuration for experiments
- `model_config.yaml`: Model parameters and hyperparameters
- `paths.yaml`: File system paths and directories

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Defects4J Not Found**
   - Set the `DEFECTS4J_HOME` environment variable
   - Or specify the path in `.env`

3. **Out of Memory**
   - Reduce batch sizes in the configuration
   - Use smaller models
   - Enable gradient checkpointing

4. **API Rate Limits**
   - Reduce the number of concurrent API calls
   - Implement retry logic with exponential backoff

## Advanced Usage

### Custom Models

You can use custom models by implementing the appropriate interface:

```python
class MyCustomTestGenerator:
    def generate_tests(self, bug, num_tests, **kwargs):
        # Your implementation here
        return tests
```

### Adding New Components

1. Create a new Python module in the appropriate layer directory
2. Implement the required interface
3. Register the component in the orchestrator
4. Update the configuration files

### Extending the Knowledge Graph

To add new node or relationship types to the knowledge graph:

```python
from py2neo import Node, Relationship

def add_custom_node(tx, node_type, properties):
    node = Node(node_type, **properties)
    tx.create(node)
    return node
```

## Support

For questions or issues, please open an issue on the GitHub repository.
