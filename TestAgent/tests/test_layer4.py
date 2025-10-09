"""
Tests for Layer 4: Patch Regression
"""
import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json

from src.layer4_patch_regression.patch_verification_agent import (
    PatchVerificationAgent,
    ExecutionTrace,
    PatchVerificationResult
)

class TestLayer4PatchRegression(unittest.TestCase):
    """Test cases for the patch regression layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.agent = PatchVerificationAgent()
        
        # Sample test cases
        self.test_cases = [
            {"test_id": "Test1", "class": "com.example.TestClass", "method": "testMethod1"},
            {"test_id": "Test2", "class": "com.example.TestClass", "method": "testMethod2"}
        ]
        
        # Sample patch content
        self.patch_content = """--- src/main/java/com/example/Example.java
+++ src/main/java/com/example/Example.java
@@ -10,7 +10,7 @@
     }
 
     public int getValue() {
-        return this.value;
+        return this.value * 2; // Fix: Return double the value
     }
 """
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary directory
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    @patch('builtins.open', new_callable=mock_open, read_data='mocked file content')
    @patch('os.path.exists')
    def test_verify_patch_success(self, mock_exists, mock_file, mock_which, mock_subprocess):
        """Test successful patch verification."""
        # Mock dependencies
        mock_exists.return_value = True
        mock_which.return_value = '/usr/bin/java'
        
        # Mock subprocess output
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='Test execution successful',
            stderr=''
        )
        
        # Call the method under test
        result = self.agent.verify_patch(
            project_path=self.test_dir,
            test_cases=self.test_cases,
            patch_content=self.patch_content
        )
        
        # Verify results
        self.assertIsInstance(result, PatchVerificationResult)
        self.assertTrue(result.is_effective)
        self.assertGreaterEqual(result.effectiveness_score, 0.0)
        self.assertLessEqual(result.effectiveness_score, 1.0)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    @patch('builtins.open', new_callable=mock_open, read_data='mocked file content')
    @patch('os.path.exists')
    def test_verify_patch_failure(self, mock_exists, mock_file, mock_which, mock_subprocess):
        """Test patch verification with test failures."""
        # Mock dependencies
        mock_exists.return_value = True
        mock_which.return_value = '/usr/bin/java'
        
        # Mock subprocess output with failure
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout='Test execution failed',
            stderr='AssertionError: Expected 2 but was 1'
        )
        
        # Call the method under test
        result = self.agent.verify_patch(
            project_path=self.test_dir,
            test_cases=self.test_cases,
            patch_content=self.patch_content
        )
        
        # Verify results
        self.assertIsInstance(result, PatchVerificationResult)
        self.assertFalse(result.is_effective)
        self.assertEqual(result.effectiveness_score, 0.0)
    
    def test_compare_traces(self):
        ""Test trace comparison logic."""
        # Create sample traces
        buggy_trace = ExecutionTrace(
            test_id="Test1",
            covered_lines=[1, 2, 3, 4],
            branch_coverage={(1, 2): True, (2, 3): True},
            method_calls=["method1", "method2"],
            execution_time=1.2,
            memory_usage=100.5
        )
        
        patched_trace = ExecutionTrace(
            test_id="Test1",
            covered_lines=[1, 2, 3, 5],  # Different line 4 -> 5
            branch_coverage={(1, 2): True, (2, 4): True},  # Different branch
            method_calls=["method1", "method3"],  # Different method call
            execution_time=1.5,
            memory_usage=105.0
        )
        
        # Call the method under test
        result = self.agent._compare_traces(
            {"Test1": buggy_trace},
            {"Test1": patched_trace}
        )
        
        # Verify results
        self.assertIsInstance(result, PatchVerificationResult)
        self.assertTrue(result.is_effective)
        self.assertGreater(result.effectiveness_score, 0.0)
        self.assertIn('line_coverage_diff', result.trace_differences)
        self.assertIn('branch_coverage_diff', result.trace_differences)
        self.assertIn('method_calls_diff', result.trace_differences)

if __name__ == "__main__":
    unittest.main()
