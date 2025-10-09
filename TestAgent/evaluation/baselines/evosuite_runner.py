"""
EvoSuite test generation and execution runner for TestAgentX evaluation.
"""
import os
import json
import subprocess
import tempfile
import shutil
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EvoSuite versions and download URLs
EVOSUITE_VERSION = "1.2.0"
EVOSUITE_JAR = f"evosuite-{EVOSUITE_VERSION}.jar"
EVOSUITE_URL = f"https://github.com/EvoSuite/evosuite/releases/download/v{EVOSUITE_VERSION}/{EVOSUITE_JAR}"

@dataclass
class EvoSuiteResult:
    """Container for EvoSuite test generation and execution results."""
    test_cases_generated: int = 0
    coverage_branch: float = 0.0
    coverage_line: float = 0.0
    coverage_method: float = 0.0
    mutation_score: float = 0.0
    tests_executed: int = 0
    tests_failed: int = 0
    time_elapsed: float = 0.0
    error_message: Optional[str] = None

class EvoSuiteRunner:
    """Runs EvoSuite test generation and collects metrics."""
    
    def __init__(self, java_home: Optional[str] = None):
        """Initialize the EvoSuite runner.
        
        Args:
            java_home: Path to Java home directory. If None, uses JAVA_HOME environment variable.
        """
        self.java_home = java_home or os.environ.get('JAVA_HOME')
        if not self.java_home:
            raise ValueError("JAVA_HOME environment variable must be set or provided as an argument")
            
        self.evosuite_jar = self._ensure_evosuite_jar()
        self.temp_dir = tempfile.mkdtemp(prefix="evosuite_")
        logger.info(f"Using temporary directory: {self.temp_dir}")
    
    def _ensure_evosuite_jar(self) -> str:
        """Ensure EvoSuite JAR is downloaded and return its path."""
        home = str(Path.home())
        evosuite_dir = os.path.join(home, ".testagent", "evosuite")
        os.makedirs(evosuite_dir, exist_ok=True)
        
        jar_path = os.path.join(evosuite_dir, EVOSUITE_JAR)
        
        if not os.path.exists(jar_path):
            logger.info(f"Downloading EvoSuite {EVOSUITE_VERSION}...")
            response = requests.get(EVOSUITE_URL, stream=True)
            response.raise_for_status()
            
            with open(jar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded EvoSuite to {jar_path}")
            
        return jar_path
    
    def _run_command(self, command: List[str], cwd: Optional[str] = None, timeout: int = 600) -> Tuple[int, str, str]:
        """Run a shell command and return (returncode, stdout, stderr)."""
        env = os.environ.copy()
        if self.java_home:
            env['JAVA_HOME'] = self.java_home
            
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd or os.getcwd(),
                env=env,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr
            
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", f"Command timed out after {timeout} seconds"
    
    def _parse_coverage_report(self, report_path: str) -> Dict[str, float]:
        """Parse JaCoCo coverage report."""
        if not os.path.exists(report_path):
            return {}
            
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            # Extract coverage metrics
            coverage = {}
            for counter in root.findall(".//counter"):
                counter_type = counter.get('type')
                covered = float(counter.get('covered', 0))
                missed = float(counter.get('missed', 0))
                total = covered + missed
                
                if total > 0:
                    coverage[f'coverage_{counter_type.lower()}'] = (covered / total) * 100
                    
            return coverage
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse coverage report: {e}")
            return {}
    
    def run_evosuite(
        self,
        target_class: str,
        classpath: str,
        time_budget: int = 60,
        seed: int = 42,
        base_dir: Optional[str] = None
    ) -> EvoSuiteResult:
        """Run EvoSuite test generation and execution.
        
        Args:
            target_class: Fully qualified name of the target class
            classpath: Classpath for the target project and its dependencies
            time_budget: Time budget in seconds (default: 60)
            seed: Random seed for reproducibility (default: 42)
            base_dir: Base directory for the project (default: current directory)
            
        Returns:
            EvoSuiteResult containing test generation and execution metrics
        """
        result = EvoSuiteResult()
        base_dir = base_dir or os.getcwd()
        
        # Prepare output directories
        test_dir = os.path.join(self.temp_dir, "tests")
        report_dir = os.path.join(self.temp_dir, "reports")
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Build EvoSuite command
        cmd = [
            "java",
            "-jar", self.evosuite_jar,
            "-class", target_class,
            "-projectCP", classpath,
            "-Dtest_dir", test_dir,
            "-Dreport_dir", report_dir,
            "-Dsearch_budget", str(time_budget),
            "-seed", str(seed),
            "-Dassertion_strategy=all",
            "-Dminimize=false",
            "-Dp_reflection_on_private=true",
            "-Dp_functional_mocking=0.5",
            "-Dfunctional_mocking_upper=5",
            "-Dinline_mocks=false",
            "-Danalysis_criteria=LINE:BRANCH:EXCEPTION:WEAKMUTATION:OUTPUT:WEAKMUTATION:ONLYMATH:METHOD:ONLYTEST:ONLYREGRESSION:CBRANCH",
            "-Dassertion_strategy=all",
            "-Dminimize=false"
        ]
        
        # Run EvoSuite
        logger.info(f"Running EvoSuite for {target_class} with {time_budget}s time budget...")
        start_time = time.time()
        returncode, stdout, stderr = self._run_command(cmd, cwd=base_dir, timeout=time_budget + 30)
        result.time_elapsed = time.time() - start_time
        
        if returncode != 0:
            result.error_message = f"EvoSuite failed with return code {returncode}\n{stderr}"
            logger.error(result.error_message)
            return result
        
        # Parse test results
        stats_file = os.path.join(report_dir, "statistics.csv")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                for line in f:
                    if line.startswith("Coverage,"):
                        parts = line.strip().split(',')
                        if len(parts) >= 4:
                            result.coverage_line = float(parts[1]) * 100  # Convert to percentage
                            result.coverage_branch = float(parts[2]) * 100
                            result.coverage_method = float(parts[3]) * 100
                    elif line.startswith("Mutation,"):
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            result.mutation_score = float(parts[1]) * 100  # Convert to percentage
                    elif line.startswith("TestsRun,"):
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            result.tests_executed = int(parts[1])
                            result.tests_failed = int(parts[2])
        
        # Count generated test cases
        if os.path.exists(test_dir):
            result.test_cases_generated = sum(1 for f in os.listdir(test_dir) 
                                           if f.endswith("Test.java") or f.endswith("Test_scaffolding.java"))
        
        logger.info(f"EvoSuite completed in {result.time_elapsed:.2f}s")
        logger.info(f"Generated {result.test_cases_generated} test cases")
        logger.info(f"Coverage: Line={result.coverage_line:.2f}%, Branch={result.coverage_branch:.2f}%")
        logger.info(f"Mutation score: {result.mutation_score:.2f}%")
        
        return result
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def __del__(self):
        self.cleanup()

# Example usage
if __name__ == "__main__":
    # Example usage
    runner = EvoSuiteRunner()
    try:
        result = runner.run_evosuite(
            target_class="com.example.MyClass",
            classpath="/path/to/classpath",
            time_budget=60
        )
        print(f"Test cases generated: {result.test_cases_generated}")
        print(f"Line coverage: {result.coverage_line:.2f}%")
        print(f"Branch coverage: {result.coverage_branch:.2f}%")
        print(f"Mutation score: {result.mutation_score:.2f}%")
    finally:
        runner.cleanup()
