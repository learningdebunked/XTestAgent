# TestAgentX Development Guide

## Table of Contents
- [Environment Setup](#environment-setup)
- [Running Tests](#running-tests)
- [Adding New Components](#adding-new-components)
- [Code Style Guidelines](#code-style-guidelines)
- [Git Workflow](#git-workflow)
- [Troubleshooting](#troubleshooting-common-issues)
- [Release Process](#release-process)

## Environment Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git 2.25+
- Java 11+ (for Java code analysis)
- Graphviz (for visualization)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/testagentx.git
   cd testagentx
   ```

2. **Set up virtual environment**:
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   
   # Upgrade pip and install build tools
   pip install --upgrade pip setuptools wheel
   ```

3. **Install dependencies**:
   ```bash
   # Install core dependencies
   pip install -e .
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install test dependencies
   pip install -e ".[test]"
   ```

4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Verify installation**:
   ```bash
   pytest tests/ -v  # Run test suite
   ```

## Running Tests

### Test Structure
```
tests/
├── unit/                  # Unit tests
├── integration/          # Integration tests
├── e2e/                  # End-to-end tests
└── fixtures/             # Test fixtures and data
```

### Running Tests

1. **Run all tests**:
   ```bash
   # Run all tests with coverage
   pytest --cov=testagentx tests/ -v
   
   # Run tests in parallel
   pytest -n auto tests/
   ```

2. **Run specific test types**:
   ```bash
   # Run unit tests only
   pytest tests/unit/
   
   # Run a specific test file
   pytest tests/unit/test_feature.py -v
   
   # Run tests matching a pattern
   pytest -k "test_something" -v
   ```

3. **Generate coverage report**:
   ```bash
   pytest --cov=testagentx --cov-report=html tests/
   open htmlcov/index.html  # View coverage report
   ```

## Adding New Components

### 1. Creating a New Agent

1. **Create agent file** in the appropriate layer:
   ```python
   # src/layerX_feature/awesome_agent.py
   from ..base_agent import BaseAgent
   
   class AwesomeAgent(BaseAgent):
       """Documentation for AwesomeAgent."""
       
       def __init__(self, config=None):
           super().__init__(config)
           # Initialization code
   ```

2. **Add tests** in the corresponding test directory:
   ```python
   # tests/unit/test_awesome_agent.py
   def test_awesome_agent_initialization():
       agent = AwesomeAgent()
       assert agent is not None
   ```

3. **Update documentation** in `docs/API.md`

### 2. Adding Dependencies

1. Add to `setup.py` under the appropriate section:
   ```python
   install_requires=[
       # Existing dependencies
       'new_dependency>=1.0.0',
   ]
   ```

2. For development dependencies, add to `extras_require`:
   ```python
   extras_require={
       'dev': [
           'pytest>=6.0',
           'black>=21.0',
           # Other dev dependencies
       ],
   }
   ```

## Code Style Guidelines

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Maximum line length: 88 characters (Black default)
- Use type hints for all function signatures
- Use Google-style docstrings

### Code Formatting
```bash
# Auto-format code
black .

# Sort imports
isort .

# Check for style issues
flake8
```

### Type Checking
```bash
mypy src/
```

## Git Workflow

### Branch Naming
- `feature/`: New features
- `bugfix/`: Bug fixes
- `hotfix/`: Critical production fixes
- `docs/`: Documentation updates
- `refactor/`: Code refactoring

### Commit Message Format
```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

Example:
```
feat(agent): add new validation logic

- Implement input validation
- Add test cases
- Update documentation

Fixes #123
```

### Pull Request Process
1. Create a feature branch from `develop`
2. Make your changes with atomic commits
3. Run tests and ensure they pass
4. Update documentation if needed
5. Create a pull request to `develop`
6. Address review comments
7. Squash commits before merging

## Troubleshooting Common Issues

### 1. Import Errors
**Issue**: `ModuleNotFoundError` when running tests
**Solution**:
```bash
# Ensure package is installed in development mode
pip install -e .

# Check PYTHONPATH
PYTHONPATH=$PYTHONPATH:. pytest tests/
```

### 2. Test Failures
**Issue**: Tests pass locally but fail in CI
**Solution**:
```bash
# Run with the same Python version as CI
pyenv local 3.9.0
pip install -r requirements-test.txt
pytest
```

### 3. Dependency Conflicts
**Issue**: Version conflicts when installing dependencies
**Solution**:
```bash
# Create a fresh virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 4. Performance Issues
**Issue**: Tests are running slowly
**Solution**:
```bash
# Run tests without coverage for faster feedback
pytest -k "test_name" -v --no-cov

# Run with xdist for parallel execution
pytest -n auto tests/
```

## Release Process

### 1. Prepare Release
```bash
git checkout develop
git pull
# Update version in src/testagentx/__init__.py
# Update CHANGELOG.md
git commit -m "chore: prepare release vX.Y.Z"
```

### 2. Create Release Branch
```bash
git checkout -b release/vX.Y.Z
git push -u origin release/vX.Y.Z
# Create PR to main
```

### 3. Publish Release
```bash
# After PR is merged
git checkout main
git pull
# Create tag
git tag -a vX.Y.Z -m "Version X.Y.Z"
git push --tags

# Build and upload to PyPI
rm -rf dist/*
python -m build
python -m twine upload dist/*
```

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for all function signatures
- Write docstrings following Google style
- Keep functions small and focused on a single responsibility
- Use meaningful variable and function names
- Write unit tests for all new features

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=testagentx tests/

# Run a specific test file
pytest tests/test_specific_feature.py
```

### Writing Tests
- Place test files in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names starting with `test_`
- Use fixtures for common test data
- Aim for high test coverage (minimum 80%)

## Documentation

### Updating Documentation
- Update docstrings when modifying code
- Keep API documentation in `docs/API.md` up to date
- Add usage examples for new features
- Document any breaking changes

### Building Documentation
```bash
# Install documentation dependencies
pip install -e .[docs]

# Build documentation
cd docs
make html
```

## Version Control

### Branching Strategy
- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/`: Feature branches (e.g., `feature/add-new-agent`)
- `bugfix/`: Bug fix branches
- `release/`: Release preparation branches

### Commit Messages
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat:` A new feature
- `fix:` A bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code changes that neither fix bugs nor add features
- `test:` Adding tests or correcting existing tests
- `chore:` Changes to the build process or auxiliary tools

## Pull Request Process

1. Fork the repository and create your feature branch
2. Commit your changes following the commit message conventions
3. Push your branch and create a Pull Request
4. Ensure all CI checks pass
5. Address any code review feedback
6. Get at least one approval before merging

## Release Process

1. Create a release branch from `develop`
   ```bash
   git checkout develop
   git pull
   git checkout -b release/vX.Y.Z
   ```

2. Update version in `src/testagentx/__init__.py`
3. Update `CHANGELOG.md` with release notes
4. Create a pull request to merge into `main`
5. After merging, create a GitHub release with version tag (vX.Y.Z)
6. Merge `main` back into `develop`
