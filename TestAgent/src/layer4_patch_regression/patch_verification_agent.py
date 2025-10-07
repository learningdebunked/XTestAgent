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

from ..layer1_preprocessing.ast_cfg_generator import ASTCFGGenerator

@dataclass
class ExecutionTrace:
    """Represents execution trace of a test case."""
    test_id: str
    covered_lines: List[int]
    branch_coverage: Dict[Tuple[int, int], bool]  # (src_line, dst_line) -> covered
    method_calls: List[str]
    execution_time: float  # in seconds
    memory_usage: float    # in MB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            'test_id': self.test_id,
            'covered_lines': self.covered_lines,
            'branch_coverage': {f"{src}-{dst}": covered 
                              for (src, dst), covered in self.branch_coverage.items()},
            'method_calls': self.method_calls,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionTrace':
        """Create ExecutionTrace from dictionary."""
        return cls(
            test_id=data['test_id'],
            covered_lines=data['covered_lines'],
            branch_coverage={tuple(map(int, k.split('-'))): v 
                           for k, v in data['branch_coverage'].items()},
            method_calls=data['method_calls'],
            execution_time=data['execution_time'],
            memory_usage=data['memory_usage']
        )

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
    
    Implements Equation (8) from the paper:
    Δ_trace = Trace(P_f, t_j) - Trace(P_b, t_j)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PatchVerificationAgent.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - jacoco_path: Path to JaCoCo agent JAR
                - java_home: Path to Java home directory
                - timeout_seconds: Test execution timeout in seconds
                - memory_limit_mb: Memory limit in MB for test execution
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.ast_cfg_generator = ASTCFGGenerator()
        
        # Set default config
        self.jacoco_agent = self.config.get('jacoco_agent', 'lib/jacocoagent.jar')
        self.java_home = self.config.get('java_home', '/usr/lib/jvm/default-java')
        self.timeout_seconds = self.config.get('timeout_seconds', 300)
        self.memory_limit_mb = self.config.get('memory_limit_mb', 4096)
        
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
                
                # Parse JaCoCo report
                coverage = self._parse_jacoco_report(jacoco_output)
                
                # Create execution trace
                trace = ExecutionTrace(
                    test_id=test_id,
                    covered_lines=coverage.get('line_coverage', []),
                    branch_coverage=coverage.get('branch_coverage', {}),
                    method_calls=self._extract_method_calls(result.stdout),
                    execution_time=0.0,  # Would be extracted from test output
                    memory_usage=0.0     # Would be measured during execution
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
        """Parse JaCoCo execution data file."""
        # This is a simplified version - in practice, you would use the JaCoCo API
        # or a library like pyjacoco to parse the .exec file
        return {
            'line_coverage': [],  # Would be populated with actual line numbers
            'branch_coverage': {}  # Would be populated with branch coverage data
        }
    
    def _extract_method_calls(self, test_output: str) -> List[str]:
        """Extract method calls from test output (simplified)."""
        # In a real implementation, this would parse the test output or use
        # a profiler to capture method calls
        return []
    
    def _copy_directory(self, src: str, dst: str) -> None:
        """Copy directory contents from src to dst."""
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to copy directory: {e}")
            raise
