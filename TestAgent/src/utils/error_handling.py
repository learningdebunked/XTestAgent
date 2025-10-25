"""
Enhanced Error Handling and Recovery for TestAgentX

Provides comprehensive error handling, logging, and recovery mechanisms.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Optional, Type, Tuple
from dataclasses import dataclass
from enum import Enum
import sys


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors"""
    component: str
    operation: str
    severity: ErrorSeverity
    recoverable: bool
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict = None


class TestAgentXError(Exception):
    """Base exception for TestAgentX"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        self.message = message
        self.context = context
        super().__init__(self.message)


class ConfigurationError(TestAgentXError):
    """Configuration-related errors"""
    pass


class ModelError(TestAgentXError):
    """ML model-related errors"""
    pass


class DatabaseError(TestAgentXError):
    """Database connection/query errors"""
    pass


class TestGenerationError(TestAgentXError):
    """Test generation errors"""
    pass


class ValidationError(TestAgentXError):
    """Validation errors"""
    pass


class PatchVerificationError(TestAgentXError):
    """Patch verification errors"""
    pass


def setup_logging(log_file: str = "logs/testagentx.log",
                 level: str = "INFO",
                 format_string: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging.
    
    Args:
        log_file: Path to log file
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    import os
    from pathlib import Path
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[%(filename)s:%(lineno)d] - %(message)s'
        )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('TestAgentX')
    logger.setLevel(getattr(logging, level.upper()))
    
    return logger


def retry_on_error(max_retries: int = 3,
                  delay: float = 1.0,
                  backoff: float = 2.0,
                  exceptions: Tuple[Type[Exception], ...] = (Exception,),
                  logger: Optional[logging.Logger] = None) -> Callable:
    """Decorator for automatic retry on errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
        logger: Logger instance
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        if logger:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        if logger:
                            logger.error(
                                f"All {max_retries} retry attempts failed for {func.__name__}: {e}"
                            )
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


def handle_errors(component: str,
                 operation: str,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recoverable: bool = True,
                 default_return: Any = None,
                 logger: Optional[logging.Logger] = None) -> Callable:
    """Decorator for comprehensive error handling.
    
    Args:
        component: Component name (e.g., 'CodeEncoder')
        operation: Operation name (e.g., 'encode_method')
        severity: Error severity level
        recoverable: Whether error is recoverable
        default_return: Default value to return on error
        logger: Logger instance
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            context = ErrorContext(
                component=component,
                operation=operation,
                severity=severity,
                recoverable=recoverable
            )
            
            try:
                return func(*args, **kwargs)
            
            except TestAgentXError as e:
                # Our custom errors
                if logger:
                    logger.error(
                        f"[{component}] {operation} failed: {e.message}",
                        exc_info=True
                    )
                
                if not recoverable:
                    raise
                
                return default_return
            
            except Exception as e:
                # Unexpected errors
                if logger:
                    logger.error(
                        f"[{component}] Unexpected error in {operation}: {e}",
                        exc_info=True
                    )
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if severity == ErrorSeverity.CRITICAL:
                    raise TestAgentXError(
                        f"Critical error in {component}.{operation}: {e}",
                        context=context
                    )
                
                if not recoverable:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None) -> Callable:
    """Decorator to log function execution time.
    
    Args:
        logger: Logger instance
        
    Returns:
        Decorated function with timing
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if logger:
                    logger.info(
                        f"{func.__name__} completed in {elapsed:.2f}s"
                    )
                
                return result
            
            except Exception as e:
                elapsed = time.time() - start_time
                
                if logger:
                    logger.error(
                        f"{func.__name__} failed after {elapsed:.2f}s: {e}"
                    )
                
                raise
        
        return wrapper
    return decorator


def safe_execute(func: Callable,
                *args,
                default: Any = None,
                logger: Optional[logging.Logger] = None,
                **kwargs) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        logger: Logger instance
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default


class ErrorRecovery:
    """Error recovery strategies"""
    
    @staticmethod
    def retry_with_backoff(func: Callable,
                          max_retries: int = 3,
                          initial_delay: float = 1.0,
                          logger: Optional[logging.Logger] = None) -> Any:
        """Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum retry attempts
            initial_delay: Initial delay in seconds
            logger: Logger instance
            
        Returns:
            Function result
        """
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt < max_retries - 1:
                    if logger:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                    time.sleep(delay)
                    delay *= 2
                else:
                    if logger:
                        logger.error(f"All retries exhausted: {e}")
                    raise
    
    @staticmethod
    def fallback_chain(*funcs: Callable,
                      logger: Optional[logging.Logger] = None) -> Any:
        """Try functions in sequence until one succeeds.
        
        Args:
            *funcs: Functions to try
            logger: Logger instance
            
        Returns:
            First successful result
        """
        last_error = None
        
        for i, func in enumerate(funcs):
            try:
                result = func()
                if logger and i > 0:
                    logger.info(f"Fallback {i} succeeded")
                return result
            except Exception as e:
                last_error = e
                if logger:
                    logger.warning(f"Fallback {i} failed: {e}")
        
        raise last_error or Exception("All fallbacks failed")
    
    @staticmethod
    def circuit_breaker(func: Callable,
                       failure_threshold: int = 5,
                       timeout: float = 60.0,
                       logger: Optional[logging.Logger] = None) -> Callable:
        """Circuit breaker pattern for fault tolerance.
        
        Args:
            func: Function to protect
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before trying again
            logger: Logger instance
            
        Returns:
            Protected function
        """
        state = {'failures': 0, 'last_failure_time': 0, 'is_open': False}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Check if circuit is open
            if state['is_open']:
                if current_time - state['last_failure_time'] < timeout:
                    raise TestAgentXError("Circuit breaker is open")
                else:
                    # Try to close circuit
                    state['is_open'] = False
                    state['failures'] = 0
                    if logger:
                        logger.info("Circuit breaker attempting to close")
            
            try:
                result = func(*args, **kwargs)
                # Success - reset failures
                state['failures'] = 0
                return result
            
            except Exception as e:
                state['failures'] += 1
                state['last_failure_time'] = current_time
                
                if state['failures'] >= failure_threshold:
                    state['is_open'] = True
                    if logger:
                        logger.error(
                            f"Circuit breaker opened after {failure_threshold} failures"
                        )
                
                raise
        
        return wrapper


# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    
    # Example 1: Retry decorator
    @retry_on_error(max_retries=3, delay=1.0, logger=logger)
    def unreliable_function():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success!"
    
    # Example 2: Error handling decorator
    @handle_errors(
        component="TestGenerator",
        operation="generate_test",
        severity=ErrorSeverity.HIGH,
        recoverable=True,
        default_return=[],
        logger=logger
    )
    def generate_test():
        # Simulate test generation
        raise TestGenerationError("Failed to generate test")
    
    # Example 3: Execution time logging
    @log_execution_time(logger=logger)
    def slow_operation():
        time.sleep(2)
        return "Done"
    
    print("Testing error handling...")
    try:
        result = unreliable_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
