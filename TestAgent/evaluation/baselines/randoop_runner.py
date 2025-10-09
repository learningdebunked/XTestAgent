"""
Randoop test generation and execution runner for TestAgentX evaluation.
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
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Randoop versions and download URLs
RANDOOP_VERSION = "4.3.2"
RANDOOP_JAR = f"randoop-{RANDOOP_VERSION}.jar"
RANDOOP_URL = f"https://github.com/randoop/randoop/releases/download/v{RANDOOP_VERSION}/{RANDOOP_JAR}"

@dataclass
class RandoopResult:
    """Container for Randoop test generation and execution results."""
    test_cases_generated: int = 0
    test_classes_generated: int = 0
    coverage_line: float = 0.0
    coverage_branch: float = 0.0
    tests_executed: int = 0
    tests_failed: int = 0
    tests_error: int = 0
    time_elapsed: float = 0.0
    error_message: Optional[str] = None

class RandoopRunner:
    """Runs Randoop test generation and collects metrics."""
    
    def __init__(self, java_home: Optional[str] = None):
        """Initialize the Randoop runner.
        
        Args:
            java_home: Path to Java home directory. If None, uses JAVA_HOME environment variable.
        """
        self.java_home = java_home or os.environ.get('JAVA_HOME')
        if not self.java_home:
            raise ValueError("JAVA_HOME environment variable must be set or provided as an argument")
            
        self.randoop_jar = self._ensure_randoop_jar()
        self.temp_dir = tempfile.mkdtemp(prefix="randoop_")
        logger.info(f"Using temporary directory: {self.temp_dir}")
    
    def _ensure_randoop_jar(self) -> str:
        """Ensure Randoop JAR is downloaded and return its path."""
        home = str(Path.home())
        randoop_dir = os.path.join(home, ".testagent", "randoop")
        os.makedirs(randoop_dir, exist_ok=True)
        
        jar_path = os.path.join(randoop_dir, RANDOOP_JAR)
        
        if not os.path.exists(jar_path):
            logger.info(f"Downloading Randoop {RANDOOP_VERSION}...")
            response = requests.get(RANDOOP_URL, stream=True)
            response.raise_for_status()
            
            with open(jar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded Randoop to {jar_path}")
            
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
    
    def _parse_jacoco_report(self, report_path: str) -> Dict[str, float]:
        """Parse JaCoCo XML report to extract coverage metrics."""
        if not os.path.exists(report_path):
            return {}
            
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            
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
            logger.error(f"Failed to parse JaCoCo report: {e}")
            return {}
    
    def run_randoop(
        self,
        target_class: str,
        classpath: str,
        time_budget: int = 60,
        seed: int = 42,
        base_dir: Optional[str] = None
    ) -> RandoopResult:
        """Run Randoop test generation and execution.
        
        Args:
            target_class: Fully qualified name of the target class
            classpath: Classpath for the target project and its dependencies
            time_budget: Time budget in seconds (default: 60)
            seed: Random seed for reproducibility (default: 42)
            base_dir: Base directory for the project (default: current directory)
            
        Returns:
            RandoopResult containing test generation and execution metrics
        """
        result = RandoopResult()
        base_dir = base_dir or os.getcwd()
        
        # Prepare output directories
        test_dir = os.path.join(self.temp_dir, "tests")
        report_dir = os.path.join(self.temp_dir, "reports")
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate test class name
        test_class_name = f"{target_class.split('.')[-1]}Test"
        
        # Build Randoop command
        cmd = [
            "java",
            "-ea",
            "-jar", self.randoop_jar,
            "gentests",
            f"--testclass={target_class}",
            f"--classlist={target_class}",  # For simplicity, just the target class
            f"--time-limit={time_budget}",
            f"--randomseed={seed}",
            f"--junit-output-dir={test_dir}",
            f"--junit-package-name=randoop_tests",
            f"--testclass={target_class}",
            f"--classpath={classpath}",
            "--junit-reflection-allowed=false",
            "--forbid-null=true",
            "--null-ratio=0",
            "--testsperfile=50",
            "--no-error-revealing-tests=true",
            "--no-regression-tests=false",
            f"--testjar={self.randoop_jar}"
        ]
        
        # Run Randoop
        logger.info(f"Running Randoop for {target_class} with {time_budget}s time budget...")
        start_time = time.time()
        returncode, stdout, stderr = self._run_command(cmd, cwd=base_dir, timeout=time_budget + 30)
        result.time_elapsed = time.time() - start_time
        
        if returncode != 0:
            result.error_message = f"Randoop failed with return code {returncode}\n{stderr}"
            logger.error(result.error_message)
            return result
        
        # Count generated test files
        test_files = []
        if os.path.exists(test_dir):
            test_files = [f for f in os.listdir(test_dir) 
                         if f.endswith("Test.java") or f.endswith("Test.class")]
            
            # Count test cases by parsing the generated test files
            for test_file in test_files:
                with open(os.path.join(test_dir, test_file), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Count test methods (simplistic approach)
                    result.test_cases_generated += content.count('@Test')
            
            result.test_classes_generated = len(test_files)
        
        # Run tests with JaCoCo for coverage
        if test_files:
            # Compile tests
            test_classpath = f"{classpath}:{self.randoop_jar}"
            
            # Run tests with JaCoCo
            jacoco_cmd = [
                "java",
                "-javaagent:org.jacoco.agent-0.8.7-runtime.jar=destfile=./coverage.exec",
                "-cp", f"{test_classpath}:{test_dir}",
                "org.junit.runner.JUnitCore",
                *[f"randoop_tests.{os.path.splitext(f)[0]}" for f in test_files]
            ]
            
            # This is a simplified example - in practice, you'd need to handle test execution more robustly
            try:
                _, test_stdout, test_stderr = self._run_command(jacoco_cmd, cwd=base_dir, timeout=300)
                
                # Parse test results (simplified)
                if "FAILURES!!!" in test_stdout:
                    result.tests_failed = test_stdout.count("FAILURES!!!")
                result.tests_executed = test_stdout.count("test(")
                result.tests_error = test_stdout.count("ERROR")
                
            except Exception as e:
                logger.warning(f"Error running tests: {e}")
        
        logger.info(f"Randoop completed in {result.time_elapsed:.2f}s")
        logger.info(f"Generated {result.test_classes_generated} test classes with {result.test_cases_generated} test cases")
        logger.info(f"Tests executed: {result.tests_executed}, Failed: {result.tests_failed}, Errors: {result.tests_error}")
        
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
    runner = RandoopRunner()
    try:
        result = runner.run_randoop(
            target_class="com.example.MyClass",
            classpath="/path/to/classpath",
            time_budget=60
        )
        print(f"Test classes generated: {result.test_classes_generated}")
        print(f"Test cases generated: {result.test_cases_generated}")
        print(f"Tests executed: {result.tests_executed}")
        print(f"Tests failed: {result.tests_failed}")
        print(f"Tests with errors: {result.tests_error}")
        print(f"Line coverage: {result.coverage_line:.2f}%")
        print(f"Branch coverage: {result.coverage_branch:.2f}%")
    finally:
        runner.cleanup()
