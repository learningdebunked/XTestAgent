"""
Patch Verification Agent for TestAgentX.

This module implements the Patch Verification component that analyzes and verifies
patches by comparing execution traces between buggy and patched code versions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import subprocess
import json
import tempfile
import shutil
import xml.etree.ElementTree as ET
import time
import re
import psutil
import os

from testagentx.layer1_preprocessing.ast_cfg_generator import ASTCFGGenerator

@dataclass
class ExecutionTrace:
    """Represents execution trace of a test case.
    
    Attributes:
        method_calls: List of method calls made during execution
        line_coverage: List of line numbers that were executed
        branch_decisions: List of tuples (line_number, decision) for branch coverage
        exceptions: List of exceptions that occurred during execution
    """
    method_calls: List[str]
    line_coverage: List[int]
    branch_decisions: List[Tuple[int, bool]]
    exceptions: List[str]
    
    def to_vector(self) -> np.ndarray:
        """Convert the execution trace to a numerical vector.
        
        Returns:
            A numpy array representing the execution trace
        """
        # Convert method calls to one-hot encoding
        method_vector = np.zeros(100)  # Assuming max 100 unique methods
        for method in self.method_calls:
            method_id = hash(method) % 100
            method_vector[method_id] += 1
            
        # Convert line coverage to binary vector
        max_line = max(self.line_coverage) if self.line_coverage else 0
        line_vector = np.zeros(max_line + 1)
        for line in self.line_coverage:
            line_vector[line] = 1
            
        # Convert branch decisions to binary vector
        branch_vector = np.zeros(2)  # [taken, not_taken]
        for _, decision in self.branch_decisions:
            if decision:
                branch_vector[0] += 1
            else:
                branch_vector[1] += 1
                
        # Combine all features
        return np.concatenate([
            method_vector,
            line_vector,
            branch_vector,
            [len(self.exceptions)]  # Number of exceptions
        ])

@dataclass
class PatchVerificationResult:
    """Results of patch verification."""
    is_effective: bool
    effectiveness_score: float  # 0.0 to 1.0
    trace_differences: Dict[str, Any]
    execution_time: float
    memory_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'is_effective': self.is_effective,
            'effectiveness_score': self.effectiveness_score,
            'trace_differences': self.trace_differences,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage
        }

class PatchVerificationAgent:
    """
    Agent responsible for verifying the effectiveness of patches by comparing
    execution traces between buggy and patched code versions.
    
    Implements the verification process using execution trace comparison:
    Δ_trace = Trace(patched_code, test) - Trace(original_code, test)
    """
    
    def __init__(self, epsilon: float = 0.1, config: Optional[Dict[str, Any]] = None):
        """Initialize the PatchVerificationAgent.
        
        Args:
            epsilon: Threshold for considering traces as different (default: 0.1)
            config: Configuration dictionary with optional parameters:
                - jacoco_path: Path to JaCoCo agent JAR
                - java_home: Path to Java home directory
                - timeout_seconds: Test execution timeout in seconds
                - memory_limit_mb: Memory limit in MB for test execution
        """
        self.epsilon = epsilon
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.ast_cfg_generator = ASTCFGGenerator()
        
        # Set default config
        self.jacoco_agent = self.config.get('jacoco_agent', 'lib/jacocoagent.jar')
        self.java_home = self.config.get('java_home', '/usr/lib/jvm/default-java')
        self.timeout_seconds = self.config.get('timeout_seconds', 300)
        self.memory_limit_mb = self.config.get('memory_limit_mb', 4096)
        
        # Cache for execution traces
        self._trace_cache = {}
        
        # Ensure JaCoCo agent exists
        if not Path(self.jacoco_agent).exists():
            self.logger.warning(f"JaCoCo agent not found at {self.jacoco_agent}")
    
    def verify_patch(
        self,
        project_path: str,
        test_cases: List[Dict[str, Any]],
        patch_file: Optional[str] = None,
        patch_content: Optional[str] = None,
        buggy_version_path: Optional[str] = None
    ) -> PatchVerificationResult:
        """Verify the effectiveness of a patch.
        
        Args:
            project_path: Path to the project
            test_cases: List of test cases to run
            patch_file: Path to the patch file (unified diff format)
            patch_content: Patch content as string (alternative to patch_file)
            buggy_version_path: Path to the buggy version of the code
            
        Returns:
            PatchVerificationResult with verification details
        """
        if not buggy_version_path:
            buggy_version_path = project_path
            
        # Create a temporary directory for the patched version
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the buggy version to the temp directory
            self._copy_directory(buggy_version_path, temp_dir)
            
            # Apply the patch
            if patch_file:
                self._apply_patch(temp_dir, patch_file)
            elif patch_content:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.patch') as f:
                    f.write(patch_content)
                    f.flush()
                    self._apply_patch(temp_dir, f.name)
            
            # Collect traces for both versions
            buggy_traces = self._collect_traces(buggy_version_path, test_cases)
            patched_traces = self._collect_traces(temp_dir, test_cases)
            
            # Compare traces and calculate effectiveness
            return self._compare_traces(buggy_traces, patched_traces)
    
    def _collect_traces(
        self, 
        project_path: str, 
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, ExecutionTrace]:
        """Collect execution traces for test cases.
        
        Args:
            project_path: Path to the project
            test_cases: List of test cases to run
            
        Returns:
            Dictionary mapping test IDs to their execution traces
        """
        traces = {}
        
        # Build the project
        if not self._build_project(project_path):
            self.logger.error(f"Failed to build project at {project_path}")
            return {}
        
        # Run tests with JaCoCo agent
        for test_case in test_cases:
            test_id = test_case['id']
            test_class = test_case['class_name']
            test_method = test_case['method_name']
            
            # Run test with JaCoCo agent
            jacoco_output = Path(project_path) / f"jacoco-{test_id}.exec"
            cmd = [
                f"{self.java_home}/bin/java",
                f"-javaagent:{self.jacoco_agent}=destfile={jacoco_output}",
                f"-Xmx{self.memory_limit_mb}m",
                "-cp", 
                f"{project_path}/target/classes:{project_path}/target/test-classes:lib/*",
                "org.junit.runner.JUnitCore",
                f"{test_class}#{test_method}"
            ]
            
            try:
                # Run test and capture output
                result = subprocess.run(
                    cmd,
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                
                # Measure execution time
                start_time = time.time()
                
                # Measure memory usage before test
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Parse JaCoCo report
                coverage = self._parse_jacoco_report(jacoco_output)
                
                # Measure memory usage after test
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_used = max(0, mem_after - mem_before)
                
                # Calculate execution time
                exec_time = time.time() - start_time
                
                # Extract method calls from output
                method_calls = self._extract_method_calls(result.stdout + result.stderr)
                
                # Create execution trace
                trace = ExecutionTrace(
                    method_calls=method_calls,
                    line_coverage=coverage.get('line_coverage', []),
                    branch_decisions=self._extract_branch_decisions(coverage.get('branch_coverage', {})),
                    exceptions=self._extract_exceptions(result.stdout + result.stderr)
                )
                
                traces[test_id] = trace
                
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Test {test_id} timed out")
            except Exception as e:
                self.logger.error(f"Error running test {test_id}: {e}")
        
        return traces
    
    def _compare_traces(
        self,
        buggy_traces: Dict[str, ExecutionTrace],
        patched_traces: Dict[str, ExecutionTrace]
    ) -> PatchVerificationResult:
        """Compare execution traces between buggy and patched versions.
        
        Implements Equation (8): Δ_trace = Trace(P_f, t_j) - Trace(P_b, t_j)
        
        Args:
            buggy_traces: Execution traces from buggy version
            patched_traces: Execution traces from patched version
            
        Returns:
            PatchVerificationResult with comparison results
        """
        common_test_ids = set(buggy_traces.keys()) & set(patched_traces.keys())
        if not common_test_ids:
            return PatchVerificationResult(
                is_effective=False,
                effectiveness_score=0.0,
                trace_differences={"error": "No common test cases between versions"},
                execution_time=0.0,
                memory_usage=0.0
            )
        
        # Compare traces for each test case
        differences = {}
        total_score = 0.0
        
        for test_id in common_test_ids:
            buggy = buggy_traces[test_id]
            patched = patched_traces[test_id]
            
            # Calculate line coverage differences
            buggy_lines = set(buggy.covered_lines)
            patched_lines = set(patched.covered_lines)
            
            # Calculate branch coverage differences
            buggy_branches = set(buggy.branch_coverage.items())
            patched_branches = set(patched.branch_coverage.items())
            
            # Calculate method call differences
            buggy_methods = set(buggy.method_calls)
            patched_methods = set(patched.method_calls)
            
            # Store differences
            diff = {
                'line_coverage': {
                    'added': list(patched_lines - buggy_lines),
                    'removed': list(buggy_lines - patched_lines),
                    'common': list(buggy_lines & patched_lines)
                },
                'branch_coverage': {
                    'added': [f"{src}-{dst}" for (src, dst), _ in (patched_branches - buggy_branches)],
                    'removed': [f"{src}-{dst}" for (src, dst), _ in (buggy_branches - patched_branches)],
                    'changed': [
                        f"{src}-{dst}: {buggy.branch_coverage.get((src, dst))} -> {patched.branch_coverage.get((src, dst))}"
                        for (src, dst), _ in (buggy_branches & patched_branches)
                        if buggy.branch_coverage.get((src, dst)) != patched.branch_coverage.get((src, dst))
                    ]
                },
                'method_calls': {
                    'added': list(patched_methods - buggy_methods),
                    'removed': list(buggy_methods - patched_methods)
                },
                'execution_time_diff': patched.execution_time - buggy.execution_time,
                'memory_usage_diff': patched.memory_usage - buggy.memory_usage
            }
            
            # Calculate a simple effectiveness score (0.0 to 1.0)
            # This is a simplified version - can be enhanced based on specific requirements
            score = 0.0
            
            # Reward for covering new lines/branches
            if diff['line_coverage']['added']:
                score += 0.3
            if diff['branch_coverage']['added']:
                score += 0.3
                
            # Penalize for performance regression
            if diff['execution_time_diff'] > 1.0:  # More than 1 second slower
                score -= 0.1
            if diff['memory_usage_diff'] > 10.0:  # More than 10MB increase
                score -= 0.1
                
            # Cap the score between 0 and 1
            score = max(0.0, min(1.0, score))
            
            differences[test_id] = {
                'differences': diff,
                'effectiveness_score': score
            }
            total_score += score
        
        # Calculate average effectiveness score
        avg_score = total_score / len(common_test_ids) if common_test_ids else 0.0
        
        # Calculate max memory usage with a proper generator expression
        max_mem = max((t.memory_usage for t in patched_traces.values()), default=0.0)
        
        return PatchVerificationResult(
            is_effective=avg_score > 0.5,  # Threshold can be adjusted
            effectiveness_score=avg_score,
            trace_differences=differences,
            execution_time=sum(t.execution_time for t in patched_traces.values()),
            memory_usage=max_mem
        )
    
    def _apply_patch(self, target_dir: str, patch_file: str) -> bool:
        """Apply a patch file to the target directory."""
        try:
            subprocess.run(
                ["patch", "-p1", "--directory", target_dir, "--input", patch_file],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to apply patch: {e.stderr}")
            return False
    
    def _build_project(self, project_path: str) -> bool:
        """Build the project using Maven."""
        try:
            subprocess.run(
                ["mvn", "clean", "compile", "test-compile"],
                cwd=project_path,
                check=True,
                capture_output=True
            )
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Build failed: {e.stderr}")
            return False
    
    def _parse_jacoco_report(self, jacoco_exec: Path) -> Dict[str, Any]:
        """Parse JaCoCo execution data file.
        
        Converts the binary .exec file to XML format and parses it to extract
        line coverage and branch coverage information.
        
        Args:
            jacoco_exec: Path to the JaCoCo .exec file
            
        Returns:
            Dictionary containing line_coverage and branch_coverage data
        """
        try:
            # Generate XML report from .exec file
            xml_report = jacoco_exec.parent / f"{jacoco_exec.stem}.xml"
            
            # Use JaCoCo CLI to generate XML report
            # This requires jacococli.jar to be available
            jacoco_cli = self.config.get('jacoco_cli', 'lib/jacococli.jar')
            
            if Path(jacoco_cli).exists() and jacoco_exec.exists():
                # Generate XML report
                cmd = [
                    'java', '-jar', jacoco_cli, 'report', str(jacoco_exec),
                    '--classfiles', str(jacoco_exec.parent.parent / 'target' / 'classes'),
                    '--sourcefiles', str(jacoco_exec.parent.parent / 'src' / 'main' / 'java'),
                    '--xml', str(xml_report)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    self.logger.warning(f"JaCoCo report generation failed: {result.stderr}")
                    return self._parse_jacoco_fallback(jacoco_exec)
                
                # Parse the XML report
                return self._parse_jacoco_xml(xml_report)
            else:
                self.logger.warning(f"JaCoCo CLI not found or .exec file missing")
                return self._parse_jacoco_fallback(jacoco_exec)
                
        except Exception as e:
            self.logger.error(f"Error parsing JaCoCo report: {e}")
            return self._parse_jacoco_fallback(jacoco_exec)
    
    def _parse_jacoco_xml(self, xml_report: Path) -> Dict[str, Any]:
        """Parse JaCoCo XML report to extract coverage data.
        
        Args:
            xml_report: Path to the XML report file
            
        Returns:
            Dictionary with line_coverage and branch_coverage
        """
        try:
            tree = ET.parse(xml_report)
            root = tree.getroot()
            
            line_coverage = []
            branch_coverage = {}
            
            # Parse packages, classes, and methods
            for package in root.findall('.//package'):
                for sourcefile in package.findall('.//sourcefile'):
                    filename = sourcefile.get('name')
                    
                    # Extract line coverage
                    for line in sourcefile.findall('.//line'):
                        line_num = int(line.get('nr'))
                        covered = int(line.get('ci', 0)) > 0  # ci = covered instructions
                        
                        if covered:
                            line_coverage.append(line_num)
                        
                        # Extract branch coverage
                        mb = int(line.get('mb', 0))  # missed branches
                        cb = int(line.get('cb', 0))  # covered branches
                        
                        if mb + cb > 0:
                            branch_coverage[line_num] = {
                                'covered': cb,
                                'missed': mb,
                                'total': mb + cb,
                                'coverage_ratio': cb / (mb + cb) if (mb + cb) > 0 else 0.0
                            }
            
            return {
                'line_coverage': sorted(line_coverage),
                'branch_coverage': branch_coverage
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing XML report: {e}")
            return {'line_coverage': [], 'branch_coverage': {}}
    
    def _parse_jacoco_fallback(self, jacoco_exec: Path) -> Dict[str, Any]:
        """Fallback method when JaCoCo CLI is not available.
        
        Uses basic heuristics to estimate coverage from test output.
        
        Args:
            jacoco_exec: Path to the .exec file
            
        Returns:
            Dictionary with estimated coverage data
        """
        self.logger.info("Using fallback coverage estimation")
        
        # Return empty coverage data as fallback
        # In production, you might want to use alternative coverage tools
        return {
            'line_coverage': [],
            'branch_coverage': {}
        }
    
    def _extract_method_calls(self, test_output: str) -> List[str]:
        """Extract method calls from test output.
        
        Parses stack traces and test output to identify method calls made during
        test execution. This is useful for understanding the execution flow.
        
        Args:
            test_output: Standard output from test execution
            
        Returns:
            List of method signatures that were called
        """
        method_calls = []
        
        try:
            # Pattern to match Java method calls in stack traces
            # Format: at package.Class.method(Class.java:line)
            stack_trace_pattern = r'at\s+([\w\.]+)\(([\w\.]+):(\d+)\)'
            
            # Find all stack trace entries
            matches = re.finditer(stack_trace_pattern, test_output)
            
            for match in matches:
                method_sig = match.group(1)
                source_file = match.group(2)
                line_num = match.group(3)
                
                # Filter out JUnit and test framework methods
                if not any(framework in method_sig for framework in 
                          ['org.junit', 'java.lang.reflect', 'sun.reflect']):
                    method_calls.append(method_sig)
            
            # Also look for explicit method call logging if present
            # Pattern: "Calling method: ClassName.methodName"
            log_pattern = r'Calling method:\s+([\w\.]+)'
            log_matches = re.finditer(log_pattern, test_output)
            
            for match in log_matches:
                method_calls.append(match.group(1))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_calls = []
            for call in method_calls:
                if call not in seen:
                    seen.add(call)
                    unique_calls.append(call)
            
            return unique_calls
            
        except Exception as e:
            self.logger.error(f"Error extracting method calls: {e}")
            return []
    
    def _extract_branch_decisions(self, branch_coverage: Dict[int, Dict[str, Any]]) -> List[Tuple[int, bool]]:
        """Extract branch decisions from coverage data.
        
        Args:
            branch_coverage: Dictionary mapping line numbers to branch coverage info
            
        Returns:
            List of tuples (line_number, decision) where decision is True if taken
        """
        decisions = []
        
        for line_num, branch_info in branch_coverage.items():
            covered = branch_info.get('covered', 0)
            total = branch_info.get('total', 0)
            
            if total > 0:
                # Add decisions for each branch
                for i in range(covered):
                    decisions.append((line_num, True))
                for i in range(total - covered):
                    decisions.append((line_num, False))
        
        return decisions
    
    def _extract_exceptions(self, output: str) -> List[str]:
        """Extract exceptions from test output.
        
        Args:
            output: Test output containing potential exceptions
            
        Returns:
            List of exception types and messages
        """
        exceptions = []
        
        try:
            # Pattern to match Java exceptions
            # Format: ExceptionType: message
            exception_pattern = r'([\w\.]+Exception|[\w\.]+Error):\s*(.+?)(?=\n|$)'
            
            matches = re.finditer(exception_pattern, output)
            
            for match in matches:
                exception_type = match.group(1)
                exception_msg = match.group(2).strip()
                exceptions.append(f"{exception_type}: {exception_msg}")
            
            # Remove duplicates
            return list(set(exceptions))
            
        except Exception as e:
            self.logger.error(f"Error extracting exceptions: {e}")
            return []
    
    def _copy_directory(self, src: str, dst: str) -> None:
        """Copy directory contents from src to dst."""
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to copy directory: {e}")
            raise
