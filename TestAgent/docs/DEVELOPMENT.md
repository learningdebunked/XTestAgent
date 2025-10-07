# Development Guide

This guide provides instructions for setting up the development environment and contributing to TestAgentX.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git
- (Optional) Docker and Docker Compose

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/testagentx.git
   cd testagentx
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Project Structure

```
testagentx/
├── src/                    # Source code
│   ├── layer1_preprocessing/
│   ├── layer2_test_generation/
│   ├── layer3_fuzzy_validation/
│   ├── layer4_patch_regression/
│   └── layer5_knowledge_graph/
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── data/                  # Data files (gitignored)
```

## Development Workflow

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style (PEP 8)
   - Write tests for new features
   - Update documentation as needed

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Run linting and type checking**
   ```bash
   pre-commit run --all-files
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add your commit message"
   ```

6. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src tests/

# Run a specific test file
pytest tests/test_module.py

# Run a specific test function
pytest tests/test_module.py::test_function_name
```

### Writing Tests

- Put test files in the `tests/` directory
- Test files should be named `test_*.py`
- Use fixtures for common test data
- Follow the Arrange-Act-Assert pattern

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for all function signatures
- Document public APIs with docstrings
- Keep lines under 100 characters

## Documentation

### Building Documentation

```bash
cd docs
make html
```

### Writing Documentation

- Update docstrings for all public APIs
- Add examples where helpful
- Keep the README up to date
- Document any configuration options

## Release Process

1. Update the version number in `setup.py`
2. Update the changelog in `CHANGELOG.md`
3. Create a release tag
   ```bash
   git tag -a v0.1.0 -m "Version 0.1.0"
   git push origin v0.1.0
   ```
4. Create a GitHub release with release notes

## Getting Help

- Check the [issues](https://github.com/yourusername/testagentx/issues) page
- Join our [Discord/Slack channel]
- Email: your.email@example.com
