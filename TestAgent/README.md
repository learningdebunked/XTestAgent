# TestAgent: Automated Test Generation and Validation Framework

TestAgent is a comprehensive framework for automated test generation and validation, designed to work with real-world software projects. It implements a multi-layered approach to test generation, validation, and analysis, with a focus on producing high-quality test cases.

## Features

- **Multi-layered Architecture**: Five distinct layers for comprehensive test generation and validation
- **LLM Integration**: Leverages large language models for intelligent test generation
- **Fuzzy Validation**: Advanced validation techniques to ensure test quality
- **Knowledge Graph**: Maintains context and relationships between test cases and code
- **Explainability**: Provides insights into test generation decisions
- **Reproducible Experiments**: Built-in support for running and comparing experiments

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd TestAgent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Quick Start

### Option 1: Simple Test (Fastest - Recommended for First Run)

```bash
# Run simple functionality test (no complex dependencies)
python3 tests/simple_test.py
```

### Option 2: Quick Validation

```bash
# Make script executable
chmod +x run_validation.sh

# Run quick validation (integration tests)
./run_validation.sh --quick
```

### Option 2: Full Validation

```bash
# Validate all paper claims
./run_validation.sh /path/to/your/project

# Or use Python directly
python evaluation/run_full_evaluation.py \
  --project /path/to/project \
  --output evaluation_results/
```

### Option 3: Step-by-Step

1. **Setup JaCoCo** (for coverage measurement):
   ```bash
   bash scripts/setup_jacoco.sh
   ```

2. **Run integration tests**:
   ```bash
   python tests/integration/test_end_to_end_pipeline.py
   ```

3. **Generate tests for a class**:
   ```bash
   python -m layer2_test_generation.llm_test_agent \
     --source Calculator.java \
     --output CalculatorTest.java
   ```

4. **Validate paper claims**:
   ```bash
   bash scripts/validate_claims.sh /path/to/project
   ```

For detailed instructions, see [QUICKSTART.md](QUICKSTART.md)

## Project Structure

```
TestAgent/
├── configs/             # Configuration files
├── data/                # Data storage
├── docker/              # Docker configuration
├── docs/                # Documentation
├── evaluation/          # Evaluation scripts
├── notebooks/           # Example notebooks
├── scripts/             # Utility scripts
├── src/                 # Source code
│   ├── agents/          # Agent implementations
│   ├── explainability/  # Explainability components
│   ├── layer*_*/        # Implementation layers
│   └── utils/           # Utility functions
└── tests/               # Test files
```

## Documentation

For detailed documentation, please see the [docs](docs/) directory:

- [Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Reproduction Guide](docs/reproduction_guide.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Cite

If you use TestAgent in your research, please cite our paper (to be added).
