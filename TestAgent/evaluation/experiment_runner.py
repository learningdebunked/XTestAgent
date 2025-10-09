""
Experiment Runner for comparing TestAgentX with baseline test generation tools.
"""
import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    project_name: str
    target_classes: List[str]
    classpath: str
    time_budget: int = 300  # seconds
    seed: int = 42
    output_dir: str = "results"
    tools: List[str] = field(default_factory=lambda: ["testagentx", "evosuite", "randoop"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))

@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    tool: str
    target_class: str
    time_elapsed: float
    test_cases_generated: int = 0
    test_classes_generated: int = 0
    coverage_line: float = 0.0
    coverage_branch: float = 0.0
    coverage_method: float = 0.0
    mutation_score: float = 0.0
    tests_executed: int = 0
    tests_failed: int = 0
    tests_error: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create result from dictionary."""
        return cls(**data)

class ExperimentRunner:
    """Runs experiments comparing TestAgentX with baseline tools."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment runner."""
        self.config = config
        self.results: List[ExperimentResult] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.experiment_dir = os.path.join(
            config.output_dir, 
            f"{config.project_name}_{self.timestamp}"
        )
        self.raw_dir = os.path.join(self.experiment_dir, "raw")
        self.figures_dir = os.path.join(self.experiment_dir, "figures")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(self.experiment_dir, "config.json"))
        
        logger.info(f"Experiment output directory: {self.experiment_dir}")
    
    def run_testagentx(self, target_class: str) -> ExperimentResult:
        """Run TestAgentX test generation."""
        logger.info(f"Running TestAgentX for {target_class}")
        result = ExperimentResult(
            tool="testagentx",
            target_class=target_class,
            time_elapsed=0.0
        )
        
        # TODO: Implement actual TestAgentX integration
        # This is a placeholder implementation
        try:
            start_time = time.time()
            
            # Simulate test generation
            time.sleep(min(5, self.config.time_budget / 2))
            
            result.time_elapsed = time.time() - start_time
            result.test_cases_generated = 15  # Example value
            result.coverage_line = 75.5  # Example value
            result.coverage_branch = 65.2  # Example value
            result.mutation_score = 60.0  # Example value
            result.tests_executed = 15  # Example value
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"TestAgentX failed for {target_class}: {e}")
        
        return result
    
    def run_evosuite(self, target_class: str) -> ExperimentResult:
        """Run EvoSuite test generation."""
        logger.info(f"Running EvoSuite for {target_class}")
        
        try:
            from baselines.evosuite_runner import EvoSuiteRunner, EvoSuiteResult
            
            runner = EvoSuiteRunner()
            evo_result = runner.run_evosuite(
                target_class=target_class,
                classpath=self.config.classpath,
                time_budget=self.config.time_budget,
                seed=self.config.seed
            )
            
            result = ExperimentResult(
                tool="evosuite",
                target_class=target_class,
                time_elapsed=evo_result.time_elapsed,
                test_cases_generated=evo_result.test_cases_generated,
                coverage_line=evo_result.coverage_line,
                coverage_branch=evo_result.coverage_branch,
                coverage_method=evo_result.coverage_method,
                mutation_score=evo_result.mutation_score,
                tests_executed=evo_result.tests_executed,
                tests_failed=evo_result.tests_failed,
                error_message=evo_result.error_message
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run EvoSuite: {e}")
            return ExperimentResult(
                tool="evosuite",
                target_class=target_class,
                time_elapsed=0,
                error_message=str(e)
            )
    
    def run_randoop(self, target_class: str) -> ExperimentResult:
        """Run Randoop test generation."""
        logger.info(f"Running Randoop for {target_class}")
        
        try:
            from baselines.randoop_runner import RandoopRunner, RandoopResult
            
            runner = RandoopRunner()
            randoop_result = runner.run_randoop(
                target_class=target_class,
                classpath=self.config.classpath,
                time_budget=self.config.time_budget,
                seed=self.config.seed
            )
            
            result = ExperimentResult(
                tool="randoop",
                target_class=target_class,
                time_elapsed=randoop_result.time_elapsed,
                test_cases_generated=randoop_result.test_cases_generated,
                test_classes_generated=randoop_result.test_classes_generated,
                coverage_line=randoop_result.coverage_line,
                coverage_branch=randoop_result.coverage_branch,
                tests_executed=randoop_result.tests_executed,
                tests_failed=randoop_result.tests_failed,
                tests_error=randoop_result.tests_error,
                error_message=randoop_result.error_message
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run Randoop: {e}")
            return ExperimentResult(
                tool="randoop",
                target_class=target_class,
                time_elapsed=0,
                error_message=str(e)
            )
    
    def run_single_experiment(self, target_class: str) -> List[ExperimentResult]:
        """Run all configured tools on a single target class."""
        results = []
        
        for tool in self.config.tools:
            if tool == "testagentx":
                result = self.run_testagentx(target_class)
            elif tool == "evosuite":
                result = self.run_evosuite(target_class)
            elif tool == "randoop":
                result = self.run_randoop(target_class)
            else:
                logger.warning(f"Unknown tool: {tool}")
                continue
                
            results.append(result)
            self.results.append(result)
            
            # Save result to file
            result_file = os.path.join(
                self.raw_dir,
                f"{target_class}_{tool}_{int(time.time())}.json"
            )
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Completed {tool} for {target_class} in {result.time_elapsed:.2f}s")
        
        return results
    
    def run_all_experiments(self) -> None:
        """Run experiments for all target classes."""
        logger.info(f"Starting experiments for {len(self.config.target_classes)} classes")
        
        for target_class in self.config.target_classes:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Running experiments for class: {target_class}")
                logger.info(f"{'='*50}")
                
                self.run_single_experiment(target_class)
                
            except Exception as e:
                logger.error(f"Error running experiments for {target_class}: {e}")
                continue
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self) -> None:
        """Generate summary report and visualizations."""
        if not self.results:
            logger.warning("No results to generate report")
            return
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Save raw results
        df.to_csv(os.path.join(self.experiment_dir, "results.csv"), index=False)
        
        # Generate summary statistics
        summary = df.groupby('tool').agg({
            'test_cases_generated': ['mean', 'std', 'min', 'max'],
            'coverage_line': ['mean', 'std', 'min', 'max'],
            'coverage_branch': ['mean', 'std', 'min', 'max'],
            'mutation_score': ['mean', 'std', 'min', 'max'],
            'time_elapsed': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Save summary
        summary.to_csv(os.path.join(self.experiment_dir, "summary.csv"))
        
        # Generate visualizations
        self._generate_plots(df)
        
        logger.info(f"\nExperiment completed. Results saved to: {self.experiment_dir}")
    
    def _generate_plots(self, df: pd.DataFrame) -> None:
        """Generate visualization plots."""
        # Set plot style
        sns.set_theme(style="whitegrid")
        
        # Coverage comparison
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x='tool', y='coverage_line')
        ax.set_title('Line Coverage by Tool')
        ax.set_ylabel('Line Coverage (%)')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'coverage_comparison.png'))
        plt.close()
        
        # Test generation speed
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x='tool', y='test_cases_generated')
        ax.set_title('Test Cases Generated by Tool')
        ax.set_ylabel('Number of Test Cases')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'test_generation.png'))
        plt.close()
        
        # Mutation score comparison
        if 'mutation_score' in df.columns:
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=df, x='tool', y='mutation_score')
            ax.set_title('Mutation Score by Tool')
            ax.set_ylabel('Mutation Score (%)')
            ax.set_xlabel('')
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, 'mutation_score.png'))
            plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run test generation experiments.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config JSON file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory for results')
    return parser.parse_args()

def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return ExperimentConfig.from_dict(config_data)

def main():
    """Main entry point for the experiment runner."""
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config.output_dir = args.output_dir
        
        # Create and run experiment
        runner = ExperimentRunner(config)
        runner.run_all_experiments()
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
