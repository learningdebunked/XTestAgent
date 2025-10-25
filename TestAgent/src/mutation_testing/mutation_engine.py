"""
Mutation Engine for TestAgentX

Executes mutation testing by:
1. Generating mutants
2. Running tests against mutants
3. Determining if mutants are killed
4. Calculating mutation score
"""

import subprocess
import tempfile
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from .mutation_operators import (
    MutationOperator, 
    get_operators_for_language,
    MutationOperatorType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Mutant:
    """Represents a single mutant"""
    id: str
    operator_type: MutationOperatorType
    operator_name: str
    file_path: str
    line_number: int
    original_code: str
    mutated_code: str
    status: str = "PENDING"  # PENDING, KILLED, SURVIVED, TIMEOUT, ERROR
    killed_by: Optional[str] = None  # Test that killed the mutant
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'operator_type': self.operator_type.value,
            'operator_name': self.operator_name,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'original_code': self.original_code,
            'mutated_code': self.mutated_code,
            'status': self.status,
            'killed_by': self.killed_by
        }


@dataclass
class MutationResult:
    """Results of mutation testing"""
    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    timeout_mutants: int
    error_mutants: int
    mutation_score: float
    mutants: List[Mutant] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_mutants': self.total_mutants,
            'killed_mutants': self.killed_mutants,
            'survived_mutants': self.survived_mutants,
            'timeout_mutants': self.timeout_mutants,
            'error_mutants': self.error_mutants,
            'mutation_score': self.mutation_score,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp,
            'mutants': [m.to_dict() for m in self.mutants]
        }


class MutationEngine:
    """
    Mutation testing engine.
    
    Generates mutants, runs tests, and calculates mutation score.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mutation engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.timeout_seconds = self.config.get('timeout_seconds', 60)
        self.max_mutants_per_file = self.config.get('max_mutants_per_file', 100)
        self.results_dir = Path(self.config.get('results_dir', 'mutation_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("MutationEngine initialized")
    
    def generate_mutants(self, source_file: str, language: str) -> List[Mutant]:
        """
        Generate mutants for a source file.
        
        Args:
            source_file: Path to source file
            language: Programming language ('java' or 'python')
            
        Returns:
            List of generated mutants
        """
        logger.info(f"Generating mutants for {source_file}")
        
        # Read source code
        with open(source_file, 'r') as f:
            source_code = f.read()
        
        lines = source_code.split('\n')
        operators = get_operators_for_language(language)
        mutants = []
        mutant_id = 0
        
        # Apply each operator to each line
        for line_num, line in enumerate(lines):
            if mutant_id >= self.max_mutants_per_file:
                logger.warning(f"Reached max mutants limit: {self.max_mutants_per_file}")
                break
            
            for operator in operators:
                # Check if operator pattern matches the line
                import re
                if re.search(operator.pattern, line):
                    # Apply mutation
                    mutated_code = operator.apply(source_code, line_num)
                    
                    if mutated_code and mutated_code != source_code:
                        mutant = Mutant(
                            id=f"mutant_{mutant_id}",
                            operator_type=operator.operator_type,
                            operator_name=operator.name,
                            file_path=source_file,
                            line_number=line_num + 1,  # 1-indexed
                            original_code=line,
                            mutated_code=mutated_code.split('\n')[line_num]
                        )
                        mutants.append(mutant)
                        mutant_id += 1
                        
                        if mutant_id >= self.max_mutants_per_file:
                            break
        
        logger.info(f"Generated {len(mutants)} mutants")
        return mutants
    
    def run_mutation_testing(self, 
                            source_file: str,
                            test_command: str,
                            language: str,
                            project_path: Optional[str] = None) -> MutationResult:
        """
        Run mutation testing on a source file.
        
        Args:
            source_file: Path to source file to mutate
            test_command: Command to run tests (e.g., "mvn test")
            language: Programming language
            project_path: Project root path
            
        Returns:
            MutationResult with testing results
        """
        logger.info(f"Running mutation testing on {source_file}")
        
        import time
        start_time = time.time()
        
        # Generate mutants
        mutants = self.generate_mutants(source_file, language)
        
        if not mutants:
            logger.warning("No mutants generated")
            return MutationResult(
                total_mutants=0,
                killed_mutants=0,
                survived_mutants=0,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=0.0
            )
        
        # Backup original file
        with open(source_file, 'r') as f:
            original_code = f.read()
        
        # Test each mutant
        killed = 0
        survived = 0
        timeout = 0
        errors = 0
        
        for i, mutant in enumerate(mutants):
            logger.info(f"Testing mutant {i+1}/{len(mutants)}: {mutant.id}")
            
            try:
                # Write mutated code
                mutated_full_code = self._apply_mutant(original_code, mutant)
                with open(source_file, 'w') as f:
                    f.write(mutated_full_code)
                
                # Run tests
                result = subprocess.run(
                    test_command.split(),
                    cwd=project_path or os.path.dirname(source_file),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                
                # Check if mutant was killed
                if result.returncode != 0:
                    # Test failed, mutant was killed
                    mutant.status = "KILLED"
                    mutant.killed_by = self._extract_failing_test(result.stdout + result.stderr)
                    killed += 1
                else:
                    # Test passed, mutant survived
                    mutant.status = "SURVIVED"
                    survived += 1
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Mutant {mutant.id} timed out")
                mutant.status = "TIMEOUT"
                timeout += 1
                
            except Exception as e:
                logger.error(f"Error testing mutant {mutant.id}: {e}")
                mutant.status = "ERROR"
                errors += 1
            
            finally:
                # Restore original code
                with open(source_file, 'w') as f:
                    f.write(original_code)
        
        # Calculate mutation score
        total = len(mutants)
        # Mutation score = killed / (total - timeout - error)
        valid_mutants = total - timeout - errors
        mutation_score = (killed / valid_mutants * 100) if valid_mutants > 0 else 0.0
        
        execution_time = time.time() - start_time
        
        result = MutationResult(
            total_mutants=total,
            killed_mutants=killed,
            survived_mutants=survived,
            timeout_mutants=timeout,
            error_mutants=errors,
            mutation_score=mutation_score,
            mutants=mutants,
            execution_time=execution_time
        )
        
        # Save results
        self._save_results(result, source_file)
        
        logger.info(f"Mutation testing complete: {mutation_score:.2f}% score")
        logger.info(f"Killed: {killed}, Survived: {survived}, Timeout: {timeout}, Errors: {errors}")
        
        return result
    
    def run_pitest(self, project_path: str, 
                   target_classes: List[str],
                   test_classes: Optional[List[str]] = None) -> MutationResult:
        """
        Run PITest (mutation testing tool for Java).
        
        Args:
            project_path: Path to Maven project
            target_classes: Classes to mutate
            test_classes: Test classes to run
            
        Returns:
            MutationResult with PITest results
        """
        logger.info("Running PITest mutation testing")
        
        import time
        start_time = time.time()
        
        # Build PITest command
        cmd = [
            'mvn',
            'org.pitest:pitest-maven:mutationCoverage',
            f'-DtargetClasses={",".join(target_classes)}',
            '-DoutputFormats=XML,HTML'
        ]
        
        if test_classes:
            cmd.append(f'-DtargetTests={",".join(test_classes)}')
        
        try:
            # Run PITest
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds * 10  # PITest takes longer
            )
            
            # Parse PITest XML report
            report_path = Path(project_path) / 'target' / 'pit-reports' / 'mutations.xml'
            mutation_result = self._parse_pitest_report(report_path)
            
            mutation_result.execution_time = time.time() - start_time
            
            # Save results
            self._save_results(mutation_result, project_path)
            
            logger.info(f"PITest complete: {mutation_result.mutation_score:.2f}% score")
            
            return mutation_result
            
        except subprocess.TimeoutExpired:
            logger.error("PITest timed out")
            raise
        except Exception as e:
            logger.error(f"Error running PITest: {e}", exc_info=True)
            raise
    
    def _apply_mutant(self, original_code: str, mutant: Mutant) -> str:
        """Apply a mutant to the original code"""
        lines = original_code.split('\n')
        if 0 <= mutant.line_number - 1 < len(lines):
            lines[mutant.line_number - 1] = mutant.mutated_code
        return '\n'.join(lines)
    
    def _extract_failing_test(self, output: str) -> Optional[str]:
        """Extract the name of the failing test from output"""
        # Try to find test name in output
        import re
        
        # JUnit pattern
        match = re.search(r'(\w+Test)\.(\w+)', output)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        
        # pytest pattern
        match = re.search(r'test_\w+', output)
        if match:
            return match.group(0)
        
        return None
    
    def _parse_pitest_report(self, report_path: Path) -> MutationResult:
        """Parse PITest XML report"""
        import xml.etree.ElementTree as ET
        
        if not report_path.exists():
            logger.warning(f"PITest report not found: {report_path}")
            return MutationResult(
                total_mutants=0,
                killed_mutants=0,
                survived_mutants=0,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=0.0
            )
        
        tree = ET.parse(report_path)
        root = tree.getroot()
        
        mutants = []
        killed = 0
        survived = 0
        timeout = 0
        errors = 0
        
        for mutation in root.findall('.//mutation'):
            status = mutation.get('status', 'UNKNOWN')
            
            mutant = Mutant(
                id=mutation.get('id', ''),
                operator_type=MutationOperatorType.AOR,  # Default
                operator_name=mutation.find('mutator').text if mutation.find('mutator') is not None else '',
                file_path=mutation.find('sourceFile').text if mutation.find('sourceFile') is not None else '',
                line_number=int(mutation.find('lineNumber').text) if mutation.find('lineNumber') is not None else 0,
                original_code='',
                mutated_code='',
                status=status,
                killed_by=mutation.find('killingTest').text if mutation.find('killingTest') is not None else None
            )
            
            mutants.append(mutant)
            
            if status == 'KILLED':
                killed += 1
            elif status == 'SURVIVED':
                survived += 1
            elif status == 'TIMED_OUT':
                timeout += 1
            else:
                errors += 1
        
        total = len(mutants)
        valid = total - timeout - errors
        score = (killed / valid * 100) if valid > 0 else 0.0
        
        return MutationResult(
            total_mutants=total,
            killed_mutants=killed,
            survived_mutants=survived,
            timeout_mutants=timeout,
            error_mutants=errors,
            mutation_score=score,
            mutants=mutants
        )
    
    def _save_results(self, result: MutationResult, source_identifier: str) -> None:
        """Save mutation testing results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mutation_result_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    engine = MutationEngine()
    
    # Example: Generate mutants
    mutants = engine.generate_mutants('Calculator.java', 'java')
    print(f"Generated {len(mutants)} mutants")
    
    # Example: Run mutation testing
    # result = engine.run_mutation_testing(
    #     source_file='Calculator.java',
    #     test_command='mvn test',
    #     language='java',
    #     project_path='/path/to/project'
    # )
    # print(f"Mutation score: {result.mutation_score:.2f}%")
