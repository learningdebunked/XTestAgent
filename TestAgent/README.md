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

1. Run the setup script to download dependencies:
   ```bash
   bash scripts/setup_defects4j.sh
   ```

2. Run the test generation pipeline:
   ```bash
   python -m evaluation.run_experiments --config configs/experiment_config.yaml
   ```

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
