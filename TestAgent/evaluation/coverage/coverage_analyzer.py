""
Coverage Analysis Module for TestAgentX

Provides tools to measure and analyze code coverage for different programming languages.
Supports JaCoCo, Cobertura, and other coverage report formats.
"""
import os
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import subprocess
import tempfile
import shutil
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoverageFormat(Enum):
    """Supported coverage report formats."""
    JACOCO = auto()
    COBERTURA = auto()
    LCOV = auto()
    PYCOV = auto()  # Python coverage.py
    CLANG = auto()  # Clang/LLVM coverage
    
    @classmethod
    def from_extension(cls, filepath: str) -> 'CoverageFormat':
        """Determine coverage format from file extension."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.xml':
            # Need to check the root element
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()
                if root.tag == 'coverage':
                    if 'cobertura' in root.tag or 'cobertura' in root.attrib.get('generator', '').lower():
                        return cls.COBERTURA
                    elif 'jacoco' in root.tag or 'jacoco' in root.attrib.get('generator', '').lower():
                        return cls.JACOCO
            except ET.ParseError:
                pass
        elif ext == '.info':
            return cls.LCOV
        elif ext == '.json':
            return cls.PYCOV
        elif ext in ('.profraw', '.profdata'):
            return cls.CLANG
            
        raise ValueError(f"Could not determine coverage format for {filepath}")

@dataclass
class CoverageResult:
    """Container for coverage measurement results."""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    method_coverage: float = 0.0
    class_coverage: float = 0.0
    total_lines: int = 0
    covered_lines: int = 0
    total_branches: int = 0
    covered_branches: int = 0
    total_methods: int = 0
    covered_methods: int = 0
    total_classes: int = 0
    covered_classes: int = 0
    report_format: Optional[CoverageFormat] = None
    report_file: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'line_coverage': self.line_coverage,
            'branch_coverage': self.branch_coverage,
            'method_coverage': self.method_coverage,
            'class_coverage': self.class_coverage,
            'total_lines': self.total_lines,
            'covered_lines': self.covered_lines,
            'total_branches': self.total_branches,
            'covered_branches': self.covered_branches,
            'total_methods': self.total_methods,
            'covered_methods': self.covered_methods,
            'total_classes': self.total_classes,
            'covered_classes': self.covered_classes,
            'report_format': self.report_format.name if self.report_format else None,
            'report_file': self.report_file,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoverageResult':
        """Create from dictionary."""
        result = cls()
        for key, value in data.items():
            if key == 'report_format' and value:
                setattr(result, key, CoverageFormat[value])
            elif hasattr(result, key):
                setattr(result, key, value)
        return result

class CoverageAnalyzer:
    ""
    Analyzes code coverage reports in various formats.
    
    Supports:
    - JaCoCo (Java)
    - Cobertura (Java, Python, JavaScript, etc.)
    - LCOV (C/C++, JavaScript, etc.)
    - Python coverage.py
    - Clang/LLVM (C/C++)
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the coverage analyzer.
        
        Args:
            base_dir: Base directory for resolving relative paths in coverage reports
        """
        self.base_dir = os.path.abspath(base_dir) if base_dir else os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix="coverage_")
        logger.debug(f"Initialized CoverageAnalyzer with base_dir={self.base_dir}")
    
    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def analyze(self, report_path: str) -> CoverageResult:
        """Analyze a coverage report file.
        
        Args:
            report_path: Path to the coverage report file
            
        Returns:
            CoverageResult with the analysis
        """
        report_path = os.path.abspath(report_path)
        if not os.path.exists(report_path):
            return self._error_result(f"Coverage report not found: {report_path}")
        
        try:
            # Determine report format
            try:
                report_format = CoverageFormat.from_extension(report_path)
            except ValueError as e:
                return self._error_result(str(e))
            
            # Parse the report
            if report_format == CoverageFormat.JACOCO:
                return self._parse_jacoco_report(report_path)
            elif report_format == CoverageFormat.COBERTURA:
                return self._parse_cobertura_report(report_path)
            elif report_format == CoverageFormat.LCOV:
                return self._parse_lcov_report(report_path)
            elif report_format == CoverageFormat.PYCOV:
                return self._parse_pycov_report(report_path)
            elif report_format == CoverageFormat.CLANG:
                return self._parse_clang_report(report_path)
            else:
                return self._error_result(f"Unsupported coverage format: {report_format}")
                
        except Exception as e:
            logger.exception(f"Error analyzing coverage report: {e}")
            return self._error_result(f"Failed to analyze coverage: {str(e)}")
    
    def _error_result(self, message: str) -> CoverageResult:
        """Create an error result."""
        logger.error(message)
        return CoverageResult(error=message)
    
    def _parse_jacoco_report(self, report_path: str) -> CoverageResult:
        """Parse a JaCoCo XML report."""
        result = CoverageResult(report_format=CoverageFormat.JACOCO, report_file=report_path)
        
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            # Get counters from the report
            counters = {}
            for counter in root.findall('.//counter'):
                counter_type = counter.get('type')
                covered = float(counter.get('covered', 0))
                missed = float(counter.get('missed', 0))
                total = covered + missed
                
                if total > 0:
                    coverage = (covered / total) * 100
                    
                    if counter_type == 'LINE':
                        result.line_coverage = coverage
                        result.covered_lines = int(covered)
                        result.total_lines = int(total)
                    elif counter_type == 'BRANCH':
                        result.branch_coverage = coverage
                        result.covered_branches = int(covered)
                        result.total_branches = int(total)
                    elif counter_type == 'METHOD':
                        result.method_coverage = coverage
                        result.covered_methods = int(covered)
                        result.total_methods = int(total)
                    elif counter_type == 'CLASS':
                        result.class_coverage = coverage
                        result.covered_classes = int(covered)
                        result.total_classes = int(total)
            
            return result
            
        except ET.ParseError as e:
            return self._error_result(f"Failed to parse JaCoCo report: {e}")
    
    def _parse_cobertura_report(self, report_path: str) -> CoverageResult:
        """Parse a Cobertura XML report."""
        result = CoverageResult(report_format=CoverageFormat.COBERTURA, report_file=report_path)
        
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            # Get line and branch coverage from the root
            line_rate = float(root.get('line-rate', 0)) * 100
            branch_rate = float(root.get('branch-rate', 0)) * 100
            
            # Get package/class level metrics
            total_lines = 0
            covered_lines = 0
            total_branches = 0
            covered_branches = 0
            total_methods = 0
            covered_methods = 0
            
            # Process all classes in the report
            for cls in root.findall('.//class'):
                # Line coverage
                cls_lines = int(cls.get('number', 0))
                cls_covered = int(round(float(cls.get('line-rate', 0)) * cls_lines))
                total_lines += cls_lines
                covered_lines += cls_covered
                
                # Method coverage
                methods = cls.findall('.//method')
                total_methods += len(methods)
                for method in methods:
                    if float(method.get('line-rate', 0)) > 0:
                        covered_methods += 1
                
                # Branch coverage (if available)
                for line in cls.findall('.//line'):
                    branch = line.get('branch')
                    condition_coverage = line.get('condition-coverage')
                    
                    if branch == 'true' and condition_coverage:
                        # Parse condition coverage like "50% (1/2)"
                        match = re.match(r'(\d+)%\s*\((\d+)/(\d+)\)', condition_coverage)
                        if match:
                            total = int(match.group(3))
                            covered = int(match.group(2))
                            total_branches += total
                            covered_branches += covered
            
            # Update result
            result.line_coverage = line_rate
            result.branch_coverage = branch_rate
            result.method_coverage = (covered_methods / total_methods * 100) if total_methods > 0 else 0
            result.covered_lines = covered_lines
            result.total_lines = total_lines
            result.covered_branches = covered_branches
            result.total_branches = total_branches
            result.covered_methods = covered_methods
            result.total_methods = total_methods
            
            return result
            
        except ET.ParseError as e:
            return self._error_result(f"Failed to parse Cobertura report: {e}")
    
    def _parse_lcov_report(self, report_path: str) -> CoverageResult:
        """Parse an LCOV report."""
        result = CoverageResult(report_format=CoverageFormat.LCOV, report_file=report_path)
        
        try:
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Parse LCOV format
            files = content.split('SF:')
            total_lines = 0
            covered_lines = 0
            total_functions = 0
            covered_functions = 0
            total_branches = 0
            covered_branches = 0
            
            for file_section in files[1:]:  # Skip first empty section
                # Count lines
                lines = file_section.split('\n')
                for line in lines:
                    if line.startswith('DA:'):
                        # DA:line_number,execution_count
                        parts = line[3:].split(',')
                        if len(parts) >= 2 and int(parts[1]) > 0:
                            covered_lines += 1
                        total_lines += 1
                    elif line.startswith('FNDA:'):
                        # FNDA:execution_count,function_name
                        parts = line[5:].split(',')
                        if len(parts) >= 2 and int(parts[0]) > 0:
                            covered_functions += 1
                        total_functions += 1
                    elif line.startswith('BRDA:'):
                        # BRDA:line,block,branch,taken
                        parts = line[5:].split(',')
                        if len(parts) >= 4:
                            if parts[3] != '-':  # Not an exception branch
                                if int(parts[3]) > 0:
                                    covered_branches += 1
                                total_branches += 1
            
            # Update result
            result.line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
            result.branch_coverage = (covered_branches / total_branches * 100) if total_branches > 0 else 0
            result.method_coverage = (covered_functions / total_functions * 100) if total_functions > 0 else 0
            result.covered_lines = covered_lines
            result.total_lines = total_lines
            result.covered_branches = covered_branches
            result.total_branches = total_branches
            result.covered_methods = covered_functions
            result.total_methods = total_functions
            
            return result
            
        except Exception as e:
            return self._error_result(f"Failed to parse LCOV report: {e}")
    
    def _parse_pycov_report(self, report_path: str) -> CoverageResult:
        """Parse a Python coverage.py JSON report."""
        result = CoverageResult(report_format=CoverageFormat.PYCOV, report_file=report_path)
        
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)
            
            # Python coverage.py JSON format
            if 'files' in data:
                total_lines = 0
                covered_lines = 0
                total_branches = 0
                covered_branches = 0
                
                for file_data in data['files'].values():
                    # Line coverage
                    if 'executed_lines' in file_data and 'summary' in file_data:
                        covered_lines += len(file_data['executed_lines'])
                        total_lines += file_data['summary']['num_statements']
                    
                    # Branch coverage
                    if 'summary' in file_data and 'covered_branches' in file_data['summary']:
                        covered_branches += file_data['summary']['covered_branches']
                        total_branches += file_data['summary']['num_branches']
                
                # Update result
                result.line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
                result.branch_coverage = (covered_branches / total_branches * 100) if total_branches > 0 else 0
                result.covered_lines = covered_lines
                result.total_lines = total_lines
                result.covered_branches = covered_branches
                result.total_branches = total_branches
                
            return result
            
        except Exception as e:
            return self._error_result(f"Failed to parse Python coverage report: {e}")
    
    def _parse_clang_report(self, report_path: str) -> CoverageResult:
        """Parse a Clang/LLVM coverage report."""
        result = CoverageResult(report_format=CoverageFormat.CLANG, report_file=report_path)
        
        try:
            # This is a simplified implementation - in practice, you'd use llvm-cov
            # to process .profdata files and extract coverage information
            # For now, we'll just return a placeholder result
            return self._error_result("Clang/LLVM coverage analysis not yet implemented")
            
        except Exception as e:
            return self._error_result(f"Failed to parse Clang/LLVM coverage report: {e}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze code coverage reports.')
    parser.add_argument('report', help='Path to the coverage report file')
    parser.add_argument('--base-dir', help='Base directory for resolving relative paths')
    args = parser.parse_args()
    
    analyzer = CoverageAnalyzer(base_dir=args.base_dir)
    result = analyzer.analyze(args.report)
    
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Coverage Report: {args.report}")
        print(f"Format: {result.report_format.name if result.report_format else 'Unknown'}")
        print(f"Line Coverage: {result.line_coverage:.2f}% ({result.covered_lines}/{result.total_lines})")
        print(f"Branch Coverage: {result.branch_coverage:.2f}% ({result.covered_branches}/{result.total_branches})")
        print(f"Method Coverage: {result.method_coverage:.2f}% ({result.covered_methods}/{result.total_methods})")
        if result.total_classes > 0:
            print(f"Class Coverage: {result.class_coverage:.2f}% ({result.covered_classes}/{result.total_classes})")
