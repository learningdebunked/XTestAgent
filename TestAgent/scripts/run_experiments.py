#!/usr/bin/env python3
"""
TestAgentX Experiment Runner

This script runs experiments to evaluate TestAgentX against various baselines
on the Defects4J benchmark.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Manages the execution of experiments."""
    
    def __init__(self, config_path: Path):
        """Initialize the experiment runner with configuration."""
        self.config = self._load_config(config_path)
        self.results_dir = Path(self.config['paths']['output_dir'])
        self.figures_dir = Path(self.config['paths']['figures_dir'])
        self.defects4j_root = Path(self.config['paths']['defects4j_root'])
        
        # Validate paths
        self._validate_paths()
        
        # Initialize results storage
        self.results: Dict[str, Any] = {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiments': {},
            'metrics': {}
        }
    
    def _load_config(self, config_path: Path) -> Dict:
        """Load experiment configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _validate_paths(self) -> None:
        """Validate that all required paths exist."""
        required_paths = {
            'Defects4J root': self.defects4j_root,
            'Results directory': self.results_dir,
            'Figures directory': self.figures_dir,
        }
        
        for name, path in required_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} not found at {path}")
    
    def _run_command(self, command: List[str], timeout: int = 3600) -> str:
        """Run a shell command with timeout."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode != 0:
                logger.error(f"Command failed: {' '.join(command)}")
                logger.error(f"Error: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode, command, result.stdout, result.stderr
                )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            raise
        except Exception as e:
            logger.error(f"Error running command: {e}")
            raise
    
    def _run_evosuite(self, bug_id: str, output_dir: Path) -> Dict:
        """Run EvoSuite for the given bug ID."""
        logger.info(f"Running EvoSuite for bug {bug_id}")
        
        # Create output directory
        bug_dir = output_dir / bug_id
        bug_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the EvoSuite command
        cmd = [
            'defects4j', 'test',
            '-w', str(bug_dir),
            '-t', bug_id,
            '-b',
            '-v',
            '-a', 'evosuite',
            '-o', f"-Dsearch_budget={self.config['baselines']['evosuite']['timeout_seconds']}"
        ]
        
        # Run the command
        try:
            self._run_command(cmd, timeout=self.config['baselines']['evosuite']['timeout_seconds'] * 2)
            
            # Parse and return results
            return {
                'status': 'success',
                'coverage': self._parse_coverage(bug_dir),
                'tests_generated': self._count_tests(bug_dir / 'evosuite-tests')
            }
        except Exception as e:
            logger.error(f"EvoSuite failed for {bug_id}: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_randoop(self, bug_id: str, output_dir: Path) -> Dict:
        """Run Randoop for the given bug ID."""
        logger.info(f"Running Randoop for bug {bug_id}")
        # Implementation similar to _run_evosuite
        return {'status': 'not_implemented'}
    
    def _run_testagentx(self, bug_id: str, output_dir: Path) -> Dict:
        """Run TestAgentX for the given bug ID."""
        logger.info(f"Running TestAgentX for bug {bug_id}")
        # Implementation will be added later
        return {'status': 'not_implemented'}
    
    def _parse_coverage(self, output_dir: Path) -> Dict:
        """Parse coverage results from the output directory."""
        # Implementation to parse coverage reports
        return {}
    
    def _count_tests(self, test_dir: Path) -> int:
        """Count the number of test files in a directory."""
        if not test_dir.exists():
            return 0
        return len(list(test_dir.glob('**/*Test*.java')))
    
    def run_single_experiment(self, bug_id: str) -> Dict:
        """Run all configured experiments for a single bug."""
        result = {'bug_id': bug_id, 'start_time': time.time()}
        
        try:
            # Create output directory
            output_dir = self.results_dir / 'raw' / bug_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run baselines
            if self.config['baselines']['evosuite']['enabled']:
                result['evosuite'] = self._run_evosuite(bug_id, output_dir)
                
            if self.config['baselines']['randoop']['enabled']:
                result['randoop'] = self._run_randoop(bug_id, output_dir)
                
            # Run TestAgentX
            result['testagentx'] = self._run_testagentx(bug_id, output_dir)
            
            result['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Experiment failed for {bug_id}: {e}")
            result.update({
                'status': 'failed',
                'error': str(e)
            })
        
        result['end_time'] = time.time()
        result['duration'] = result['end_time'] - result['start_time']
        
        # Save individual result
        self._save_result(bug_id, result)
        
        return result
    
    def _save_result(self, bug_id: str, result: Dict) -> None:
        """Save experiment results to a JSON file."""
        output_file = self.results_dir / 'raw' / f"{bug_id}.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.debug(f"Saved results for {bug_id} to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results for {bug_id}: {e}")
    
    def aggregate_results(self) -> None:
        """Aggregate results from all experiments."""
        logger.info("Aggregating results...")
        # Implementation to aggregate results
        pass
    
    def generate_figures(self) -> None:
        """Generate figures from the aggregated results."""
        logger.info("Generating figures...")
        # Implementation to generate figures
        pass
    
    def run_all_experiments(self, bug_ids: List[str]) -> None:
        """Run experiments for all specified bug IDs."""
        logger.info(f"Starting experiments for {len(bug_ids)} bugs")
        
        # Create a process pool
        with ProcessPoolExecutor(max_workers=self.config['experiment']['max_workers']) as executor:
            # Submit all experiments
            futures = {
                executor.submit(self.run_single_experiment, bug_id): bug_id
                for bug_id in bug_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                bug_id = futures[future]
                try:
                    result = future.result()
                    logger.info(f"Completed {bug_id}: {result['status']} "
                              f"(took {result.get('duration', 0):.2f}s)")
                except Exception as e:
                    logger.error(f"Error processing {bug_id}: {e}")
        
        # After all experiments complete
        self.aggregate_results()
        self.generate_figures()
        
        logger.info("All experiments completed!")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run TestAgentX experiments')
    parser.add_argument('--config', type=Path, default='data/experiment_config.json',
                       help='Path to experiment configuration file')
    parser.add_argument('--bug-ids', type=str, nargs='+',
                       help='List of bug IDs to run (default: all)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers')
    return parser.parse_args()

def main():
    """Main function to run experiments."""
    args = parse_args()
    
    try:
        # Initialize experiment runner
        runner = ExperimentRunner(args.config)
        
        # Override config if needed
        if args.max_workers is not None:
            runner.config['experiment']['max_workers'] = args.max_workers
        
        # Get bug IDs to run
        if args.bug_ids:
            bug_ids = args.bug_ids
        else:
            # Load all bug IDs from Defects4J
            # This is a placeholder - you'll need to implement this
            bug_ids = []
        
        # Run experiments
        runner.run_all_experiments(bug_ids)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
