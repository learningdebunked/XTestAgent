"""Unit tests for the Patch Verification module."""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules to test
from src.layer4_patch_regression.patch_verification_agent import (
    PatchVerificationAgent,
    ExecutionTrace,
    PatchVerificationResult
)
from src.layer4_patch_regression.trace_analyzer import TraceAnalyzer, TraceDifference

class TestPatchVerification(unittest.TestCase):
    """Test cases for the Patch Verification module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.agent = PatchVerificationAgent({
            'jacoco_agent': 'mock_jacoco_agent.jar',
            'java_home': '/mock/java/home'
        })
        
        # Create a simple project structure
        self.project_path = os.path.join(self.test_dir, 'test_project')
        os.makedirs(os.path.join(self.project_path, 'src', 'main', 'java', 'com', 'example'), exist_ok=True)
        os.makedirs(os.path.join(self.project_path, 'src', 'test', 'java', 'com', 'example'), exist_ok=True)
        
        # Create a simple Java class
        with open(os.path.join(self.project_path, 'src', 'main', 'java', 'com', 'example', 'Calculator.java'), 'w') as f:
            f.write("""
            package com.example;
            
            public class Calculator {
                public int add(int a, int b) {
                    return a + b;
                }
                
                public int subtract(int a, int b) {
                    return a - b;
                }
            }
            """)
        
        # Create a simple test class
        with open(os.path.join(self.project_path, 'src', 'test', 'java', 'com', 'example', 'CalculatorTest.java'), 'w') as f:
            f.write("""
            package com.example;
            
            import org.junit.Test;
            import static org.junit.Assert.*;
            
            public class CalculatorTest {
                @Test
                public void testAdd() {
                    Calculator calc = new Calculator();
                    assertEquals(5, calc.add(2, 3));
                }
                
                @Test
                public void testSubtract() {
                    Calculator calc = new Calculator();
                    assertEquals(1, calc.subtract(3, 2));
                }
            }
            """)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_execution_trace_serialization(self):
        """Test serialization and deserialization of ExecutionTrace."""
        trace = ExecutionTrace(
            test_id='test1',
            covered_lines=[1, 2, 3, 4, 5],
            branch_coverage={(1, 2): True, (2, 3): False},
            method_calls=['com.example.Calculator.add', 'com.example.Calculator.subtract'],
            execution_time=0.123,
            memory_usage=42.5
        )
        
        # Convert to dict and back
        trace_dict = trace.to_dict()
        new_trace = ExecutionTrace.from_dict(trace_dict)
        
        # Verify all fields are preserved
        self.assertEqual(trace.test_id, new_trace.test_id)
        self.assertEqual(trace.covered_lines, new_trace.covered_lines)
        self.assertEqual(trace.branch_coverage, new_trace.branch_coverage)
        self.assertEqual(trace.method_calls, new_trace.method_calls)
        self.assertAlmostEqual(trace.execution_time, new_trace.execution_time)
        self.assertAlmostEqual(trace.memory_usage, new_trace.memory_usage)
    
    @patch('subprocess.run')
    def test_apply_patch_success(self, mock_run):
        """Test successful patch application."""
        # Mock subprocess.run to simulate successful patch application
        mock_run.return_value.returncode = 0
        
        # Create a simple patch file
        patch_file = os.path.join(self.test_dir, 'test.patch')
        with open(patch_file, 'w') as f:
            f.write("""
            diff --git a/src/main/java/com/example/Calculator.java b/src/main/java/com/example/Calculator.java
            index 1234567..89abcde 100644
            --- a/src/main/java/com/example/Calculator.java
            +++ b/src/main/java/com/example/Calculator.java
            @@ -1,5 +1,5 @@
             package com.example;
            -public class Calculator {
            +public class Calculator implements Serializable {
                 public int add(int a, int b) {
                     return a + b;
                 }
            """)
        
        result = self.agent._apply_patch(self.project_path, patch_file)
        self.assertTrue(result)
        
        # Verify the patch command was called correctly
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        self.assertEqual(args[0][0], 'patch')
        self.assertIn('--directory', args[0])
        self.assertIn('--input', args[0])
    
    def test_trace_analyzer_compare_traces(self):
        """Test trace comparison in TraceAnalyzer."""
        analyzer = TraceAnalyzer()
        
        # Create test traces
        trace1 = ExecutionTrace(
            test_id='test1',
            covered_lines=[1, 2, 3, 4, 5],
            branch_coverage={(1, 2): True, (2, 3): False},
            method_calls=['methodA', 'methodB'],
            execution_time=1.0,
            memory_usage=100.0
        )
        
        trace2 = ExecutionTrace(
            test_id='test1',
            covered_lines=[1, 2, 3, 6, 7],  # Lines 4,5 removed; 6,7 added
            branch_coverage={(1, 2): True, (2, 4): True},  # (2,3) removed; (2,4) added
            method_calls=['methodA', 'methodC'],  # methodB removed; methodC added
            execution_time=1.2,
            memory_usage=110.0
        )
        
        # Compare traces
        diff = analyzer.compare_traces(trace1, trace2, 'test1')
        
        # Verify differences
        self.assertEqual(diff.test_id, 'test1')
        self.assertEqual(diff.line_coverage_diff['added'], [6, 7])
        self.assertEqual(diff.line_coverage_diff['removed'], [4, 5])
        self.assertEqual(diff.line_coverage_diff['common'], [1, 2, 3])
        self.assertIn((2, 4), diff.branch_coverage_diff['added'])
        self.assertIn((2, 3), diff.branch_coverage_diff['removed'])
        self.assertEqual(diff.method_call_diff['added'], ['methodC'])
        self.assertEqual(diff.method_call_diff['removed'], ['methodB'])
        self.assertAlmostEqual(diff.execution_time_diff, 0.2)
        self.assertAlmostEqual(diff.memory_usage_diff, 10.0)
    
    def test_calculate_patch_effectiveness(self):
        """Test patch effectiveness calculation."""
        analyzer = TraceAnalyzer()
        
        # Create test differences
        differences = {
            'test1': TraceDifference(
                test_id='test1',
                line_coverage_diff={'added': [6, 7], 'removed': [4], 'common': [1, 2, 3]},
                branch_coverage_diff={'added': [(2, 4)], 'removed': [(2, 3)], 'changed': []},
                method_call_diff={'added': ['methodC'], 'removed': ['methodB'], 'common': ['methodA']},
                execution_time_diff=0.1,
                memory_usage_diff=5.0
            ),
            'test2': TraceDifference(
                test_id='test2',
                line_coverage_diff={'added': [8, 9], 'removed': [5], 'common': [1, 2, 3, 4]},
                branch_coverage_diff={'added': [(3, 5)], 'removed': [], 'changed': [(1, 2)]},
                method_call_diff={'added': ['methodD'], 'removed': [], 'common': ['methodA']},
                execution_time_diff=0.2,
                memory_usage_diff=10.0
            )
        }
        
        # Calculate effectiveness
        metrics = analyzer.calculate_patch_effectiveness(differences)
        
        # Verify metrics
        self.assertIn('overall_score', metrics)
        self.assertIn('line_coverage_improvement', metrics)
        self.assertIn('branch_coverage_improvement', metrics)
        self.assertIn('method_coverage_improvement', metrics)
        self.assertIn('performance_impact', metrics)
        self.assertIn('memory_impact', metrics)
        
        # All scores should be between 0 and 1
        for metric, value in metrics.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

if __name__ == '__main__':
    unittest.main()
