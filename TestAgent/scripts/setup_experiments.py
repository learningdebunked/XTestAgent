"""
Experimental Setup for TestAgentX

This script sets up the directory structure and initial files needed for running
the TestAgentX experiments as described in the paper.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = DATA_DIR / 'results'
FIGURES_DIR = DATA_DIR / 'figures'
BASELINES_DIR = DATA_DIR / 'baselines'
SCRIPTS_DIR = BASE_DIR / 'scripts'

# Required subdirectories
REQUIRED_DIRS = [
    DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    BASELINES_DIR,
    SCRIPTS_DIR,
    RESULTS_DIR / 'raw',
    RESULTS_DIR / 'processed',
    FIGURES_DIR / 'paper',
    FIGURES_DIR / 'exploratory',
    BASELINES_DIR / 'evosuite',
    BASELINES_DIR / 'randoop',
    BASELINES_DIR / 'manual_qa',
    BASELINES_DIR / 'llm_baselines',
]

def setup_directories() -> None:
    """Create all required directories if they don't exist."""
    logger.info("Setting up experiment directories...")
    
    for directory in REQUIRED_DIRS:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise
    
    logger.info("Directory structure created successfully.")

def create_experiment_config() -> None:
    """Create a default experiment configuration file."""
    config = {
        "experiment": {
            "name": "TestAgentX Evaluation",
            "description": "Evaluation of TestAgentX on Defects4J benchmark",
            "num_bugs": 835,
            "random_seed": 42,
            "max_workers": 4,
        },
        "paths": {
            "defects4j_root": "/path/to/defects4j",
            "output_dir": str(RESULTS_DIR),
            "figures_dir": str(FIGURES_DIR),
        },
        "baselines": {
            "evosuite": {
                "enabled": True,
                "version": "1.2.0",
                "timeout_seconds": 300,
            },
            "randoop": {
                "enabled": True,
                "version": "4.3.0",
                "timeout_seconds": 300,
            },
            "manual_qa": {
                "enabled": False,
                "data_path": "data/manual_qa_results.json"
            },
            "llm_baselines": {
                "enabled": True,
                "models": ["codex", "t5", "codet5"],
                "timeout_seconds": 600,
            }
        },
        "metrics": {
            "coverage_metrics": ["line", "branch", "mutation"],
            "fault_detection_metrics": ["fault_revelation", "test_effectiveness"],
            "execution_metrics": ["execution_time", "memory_usage"],
        }
    }
    
    config_path = DATA_DIR / 'experiment_config.json'
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created experiment config at {config_path}")
    except Exception as e:
        logger.error(f"Failed to create experiment config: {e}")
        raise

def create_runner_script() -> None:
    """Create a script to run the experiments."""
    script_content = """#!/usr/bin/env python3
"""
    # Rest of the script content will be added in the next step
    
    script_path = SCRIPTS_DIR / 'run_experiments.py'
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)  # Make it executable
        logger.info(f"Created experiment runner script at {script_path}")
    except Exception as e:
        logger.error(f"Failed to create runner script: {e}")
        raise

def main():
    """Main function to set up the experiment environment."""
    try:
        setup_directories()
        create_experiment_config()
        create_runner_script()
        logger.info("Experiment setup completed successfully!")
    except Exception as e:
        logger.error(f"Failed to set up experiments: {e}")
        raise

if __name__ == "__main__":
    main()
