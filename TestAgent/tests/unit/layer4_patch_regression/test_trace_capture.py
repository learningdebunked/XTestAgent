"""
Unit tests for the TraceCapture module.

These tests verify the functionality of the trace capture infrastructure,
including execution context handling, trace collection, and result processing.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.layer4_patch_regression.trace_capture import (
    TraceCollector,
    ExecutionContext,
    TraceResult,
    Language
)

# Test data
SAMPLE_JAVA_PROJECT = """
public class Sample {
    public static int add(int a, int b) {
        return a + b;
    }
}
"""

SAMPLE_JAVA_TEST = """
import org.junit.Test;
import static org.junit.Assert.*;

public class SampleTest {
    @Test
    public void testAdd() {
        assertEquals(5, Sample.add(2, 3));
    }
}
"""

class MockCompletedProcess:
    """Mock subprocess.CompletedProcess for testing."""
    
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

@pytest.fixture
def mock_java_project(tmp_path):
    """Create a mock Java project directory structure."""
    # Create source directory
    src_dir = tmp_path / "src" / "main" / "java"
    src_dir.mkdir(parents=True)
    
    # Create test directory
    test_dir = tmp_path / "src" / "test" / "java"
    test_dir.mkdir(parents=True)
    
    # Create sample files
    (src_dir / "Sample.java").write_text(SAMPLE_JAVA_PROJECT)
    (test_dir / "SampleTest.java").write_text(SAMPLE_JAVA_TEST)
    
    # Create a simple build.gradle
    build_gradle = tmp_path / "build.gradle"
    build_gradle.write_text("""
    plugins {
        id 'java'
        id 'jacoco'
    }
    
    repositories {
        mavenCentral()
    }
    
    dependencies {
        testImplementation 'junit:junit:4.13.2'
    }
    
    test {
        useJUnit()
    }
    """)
    
    return tmp_path

@pytest.fixture
def mock_trace_collector():
    """Create a TraceCollector instance with mocked dependencies."""
    with patch('subprocess.Popen') as mock_popen:
        # Configure the mock to return a successful process
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b'Test output', b'')
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        collector = TraceCollector({
            'java_home': '/fake/java/home',
            'jacoco_agent': '/fake/jacoco/agent.jar'
        })
        
        # Mock the _execute_command method
        collector._execute_command = MagicMock(return_value=type('CommandResult', (), {
            'returncode': 0,
            'stdout': 'Test output',
            'stderr': '',
            'execution_time': 1.23,
            'memory_usage': 1024.0,
            'error': None
        }))
        
        yield collector

class TestTraceCollector:
    """Test cases for the TraceCollector class."""
    
    def test_initialization(self):
        """Test that the collector initializes with default values."""
        collector = TraceCollector()
        assert collector is not None
        assert hasattr(collector, 'java_home')
        assert hasattr(collector, 'jacoco_agent')
    
    def test_java_trace_capture(self, mock_trace_collector, mock_java_project, tmp_path):
        """Test capturing Java execution traces."""
        context = ExecutionContext(
            project_path=mock_java_project,
            language=Language.JAVA,
            test_command="./gradlew test",
            timeout_seconds=300,
            memory_limit_mb=2048
        )
        
        # Configure the mock to return a successful result
        mock_trace_collector._execute_command.return_value = type('CommandResult', (), {
            'returncode': 0,
            'stdout': 'Tests passed',
            'stderr': '',
            'execution_time': 1.23,
            'memory_usage': 1024.0,
            'error': None
        })
        
        # Call the method under test
        result = mock_trace_collector.capture_trace(
            context=context,
            output_dir=tmp_path,
            test_filter="SampleTest",
            extra_args=["--tests", "SampleTest.testAdd"]
        )
        
        # Verify the results
        assert result.success is True
        assert "coverage_data" in result.coverage_data
        assert result.execution_time > 0
        assert result.memory_usage > 0
    
    @pytest.mark.parametrize("language,test_command", [
        (Language.JAVA, "./gradlew test"),
        (Language.PYTHON, "pytest"),
        (Language.JAVASCRIPT, "npm test")
    ])
    def test_language_support(self, mock_trace_collector, tmp_path, language, test_command):
        """Test support for different programming languages."""
        context = ExecutionContext(
            project_path=tmp_path,
            language=language,
            test_command=test_command,
            timeout_seconds=60,
            memory_limit_mb=1024
        )
        
        # Call the method under test
        result = mock_trace_collector.capture_trace(context=context)
        
        # Verify basic success
        assert isinstance(result, TraceResult)
    
    def test_error_handling(self, mock_trace_collector, tmp_path):
        """Test error handling during trace capture."""
        # Configure the mock to simulate a command failure
        mock_trace_collector._execute_command.return_value = type('CommandResult', (), {
            'returncode': 1,
            'stdout': '',
            'stderr': 'Test error',
            'execution_time': 0.0,
            'memory_usage': 0.0,
            'error': 'Command failed'
        })
        
        context = ExecutionContext(
            project_path=tmp_path,
            language=Language.JAVA,
            test_command="./gradlew test"
        )
        
        # Call the method under test
        result = mock_trace_collector.capture_trace(context=context)
        
        # Verify error handling
        assert result.success is False
        assert result.error is not None
        assert "error" in result.stderr

class TestExecutionContext:
    """Test cases for the ExecutionContext class."""
    
    def test_initialization(self, tmp_path):
        """Test that the execution context initializes correctly."""
        context = ExecutionContext(
            project_path=tmp_path,
            language=Language.JAVA,
            test_command="./gradlew test"
        )
        
        assert context.project_path == tmp_path
        assert context.language == Language.JAVA
        assert context.test_command == "./gradlew test"
        assert context.timeout_seconds == 300  # Default value
        assert context.memory_limit_mb == 4096  # Default value
        assert isinstance(context.env_vars, dict)
    
    def test_custom_values(self, tmp_path):
        """Test initialization with custom values."""
        env_vars = {"CUSTOM_ENV": "value"}
        context = ExecutionContext(
            project_path=tmp_path,
            language=Language.PYTHON,
            test_command="pytest",
            env_vars=env_vars,
            timeout_seconds=120,
            memory_limit_mb=2048
        )
        
        assert context.timeout_seconds == 120
        assert context.memory_limit_mb == 2048
        assert context.env_vars == env_vars

class TestTraceResult:
    """Test cases for the TraceResult class."""
    
    def test_initialization(self):
        """Test that the trace result initializes correctly."""
        result = TraceResult(
            success=True,
            coverage_data={"lines_covered": 10, "lines_total": 20},
            execution_time=1.5,
            memory_usage=1024.0,
            stdout="Test output",
            stderr="",
            error=None
        )
        
        assert result.success is True
        assert result.coverage_data == {"lines_covered": 10, "lines_total": 20}
        assert result.execution_time == 1.5
        assert result.memory_usage == 1024.0
        assert result.stdout == "Test output"
        assert result.stderr == ""
        assert result.error is None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TraceResult(
            success=True,
            coverage_data={"lines_covered": 5, "lines_total": 10},
            execution_time=1.0,
            memory_usage=512.0,
            stdout="Test",
            stderr="Error"
        )
        
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["coverage_data"] == {"lines_covered": 5, "lines_total": 10}
        assert result_dict["execution_time"] == 1.0
        assert result_dict["memory_usage"] == 512.0
        assert result_dict["stdout"] == "Test"
        assert result_dict["stderr"] == "Error"
        assert "error" not in result_dict  # Should be omitted when None
