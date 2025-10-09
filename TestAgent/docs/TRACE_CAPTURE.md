# Trace Capture Infrastructure

## Overview
The Trace Capture infrastructure provides a unified way to capture execution traces across multiple programming languages. It's a core component of the patch verification system, enabling the comparison of execution behavior between buggy and patched code versions.

## Features

- **Multi-language Support**: Java (JaCoCo), Python (coverage.py), and JavaScript (Jest)
- **Flexible Configuration**: Customizable timeouts, memory limits, and environment variables
- **Detailed Reporting**: Comprehensive trace data including coverage information
- **Error Handling**: Robust error handling and reporting

## Components

### 1. TraceCollector

The main class for capturing execution traces.

#### Key Methods

- `capture_trace(context, output_dir=None, test_filter=None, extra_args=None)`: Captures execution traces
- `_capture_java_trace()`: Java-specific trace capture
- `_capture_python_trace()`: Python-specific trace capture
- `_capture_javascript_trace()`: JavaScript-specific trace capture
- `_execute_command()`: Internal method for running shell commands

### 2. ExecutionContext

Holds the context for test execution.

#### Attributes
- `project_path`: Path to the project root
- `language`: Programming language (from Language enum)
- `test_command`: Command to run tests
- `env_vars`: Environment variables for test execution
- `timeout_seconds`: Maximum execution time
- `memory_limit_mb`: Maximum memory usage

### 3. TraceResult

Stores the results of a trace capture operation.

#### Attributes
- `success`: Boolean indicating success/failure
- `coverage_data`: Dictionary with coverage information
- `execution_time`: Time taken for execution (seconds)
- `memory_usage`: Memory used (MB)
- `stdout`: Standard output from the test run
- `stderr`: Standard error from the test run
- `error`: Error message if execution failed

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from trace_capture import TraceCollector, ExecutionContext, Language

# Initialize the collector
collector = TraceCollector({
    'java_home': '/path/to/java',
    'jacoco_agent': '/path/to/jacocoagent.jar'
})

# Set up execution context
context = ExecutionContext(
    project_path=Path("/path/to/project"),
    language=Language.JAVA,
    test_command="./gradlew test",
    timeout_seconds=300,
    memory_limit_mb=2048
)

# Capture traces
result = collector.capture_trace(
    context=context,
    test_filter="com.example.MyTest",
    extra_args=["--tests", "com.example.MyTest.testMethod"]
)

# Process results
if result.success:
    print(f"Tests passed in {result.execution_time:.2f} seconds")
    print(f"Coverage: {result.coverage_data}")
else:
    print(f"Test failed: {result.error}")
```

### Python Project Example

```python
context = ExecutionContext(
    project_path=Path("/path/to/python/project"),
    language=Language.PYTHON,
    test_command="pytest",
    env_vars={"PYTHONPATH": "/path/to/python/project"}
)

result = collector.capture_trace(context)
```

### JavaScript Project Example

```python
context = ExecutionContext(
    project_path=Path("/path/to/js/project"),
    language=Language.JAVASCRIPT,
    test_command="npm test",
    env_vars={"NODE_ENV": "test"}
)

result = collector.capture_trace(context)
```

## Configuration Options

### TraceCollector Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `java_home` | str | Path to Java home directory | System JAVA_HOME |
| `jacoco_agent` | str | Path to JaCoCo agent JAR | 'lib/jacocoagent.jar' |
| `python_coverage_module` | str | Python coverage module to use | 'coverage' |
| `node_options` | str | Additional Node.js options | '--trace-warnings' |

### ExecutionContext Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `project_path` | Path | Path to project root | Required |
| `language` | Language | Programming language | Required |
| `test_command` | str | Command to run tests | Required |
| `env_vars` | Dict[str, str] | Environment variables | {} |
| `timeout_seconds` | int | Maximum execution time | 300 |
| `memory_limit_mb` | int | Maximum memory usage (MB) | 4096 |

## Best Practices

1. **Resource Management**: Always set appropriate timeouts and memory limits
2. **Error Handling**: Check the `success` flag and handle errors appropriately
3. **Caching**: Cache trace results when possible to improve performance
4. **Cleanup**: Ensure temporary files are cleaned up after use

## Troubleshooting

### Common Issues

1. **Java/JaCoCo Not Found**
   - Ensure `java_home` points to a valid JDK installation
   - Verify the JaCoCo agent JAR exists at the specified path

2. **Test Timeouts**
   - Increase the `timeout_seconds` parameter if tests are timing out
   - Check for long-running tests that might need optimization

3. **Memory Issues**
   - Increase `memory_limit_mb` if tests are running out of memory
   - Consider splitting large test suites into smaller chunks

## Performance Considerations

- **Parallel Execution**: The collector supports parallel test execution
- **Caching**: Implement caching for repeated test executions
- **Incremental Analysis**: Only analyze changed files when possible

## Extending for New Languages

To add support for a new language:

1. Add a new enum value to the `Language` class
2. Implement a new `_capture_<language>_trace()` method in `TraceCollector`
3. Add corresponding test cases
4. Update documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
