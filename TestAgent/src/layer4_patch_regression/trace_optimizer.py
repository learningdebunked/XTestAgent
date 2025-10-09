"""
Trace Optimization Module for TestAgentX.

This module provides performance optimizations for trace capture,
including caching and parallel execution.
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, TypeVar, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
import pickle
import gzip

from .trace_capture import TraceCollector, ExecutionContext, TraceResult, Language

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])

class TraceCache:
    """Disk-based cache for trace results."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size_mb: int = 1024):
        """Initialize the trace cache.
        
        Args:
            cache_dir: Directory to store cache files (default: ~/.testagentx/cache)
            max_size_mb: Maximum cache size in MB (default: 1024)
        """
        self.logger = logging.getLogger(f"{__name__}.TraceCache")
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".testagentx" / "cache"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self._enabled = True
        
        # Clean up old cache entries if needed
        self._cleanup()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a given key."""
        # Create a hash of the key for the filename
        key_hash = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"trace_{key_hash}.pkl.gz"
    
    def _get_cache_size(self) -> int:
        """Calculate the total size of the cache directory."""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl.gz"))
    
    def _cleanup(self) -> None:
        """Clean up old cache entries if the cache is too large."""
        try:
            # Get all cache files with their modification times
            cache_files = []
            for f in self.cache_dir.glob("*.pkl.gz"):
                cache_files.append((f, f.stat().st_mtime))
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Remove oldest files until we're under the size limit
            total_size = self._get_cache_size()
            while total_size > self.max_size and cache_files:
                file_path, _ = cache_files.pop(0)
                file_size = file_path.stat().st_size
                try:
                    file_path.unlink()
                    total_size -= file_size
                    self.logger.debug(f"Removed old cache file: {file_path}")
                except OSError as e:
                    self.logger.warning(f"Failed to remove cache file {file_path}: {e}")
        except Exception as e:
            self.logger.warning(f"Error during cache cleanup: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self._enabled:
            return None
            
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return None
            
        try:
            with gzip.open(cache_file, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, OSError) as e:
            self.logger.warning(f"Error reading from cache {cache_file}: {e}")
            try:
                cache_file.unlink()  # Remove corrupted cache file
            except OSError:
                pass
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be picklable)
        """
        if not self._enabled:
            return
            
        cache_file = self._get_cache_path(key)
        try:
            # Write to a temporary file first, then atomically rename
            temp_file = cache_file.with_suffix('.tmp')
            with gzip.open(temp_file, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_file.replace(cache_file)  # Atomic rename
        except (pickle.PickleError, OSError) as e:
            self.logger.warning(f"Error writing to cache {cache_file}: {e}")
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass
        
        # Check if we need to clean up old cache entries
        if self._get_cache_size() > self.max_size:
            self._cleanup()
    
    def clear(self) -> None:
        """Clear all cached values."""
        try:
            for f in self.cache_dir.glob("*.pkl.gz"):
                try:
                    f.unlink()
                except OSError as e:
                    self.logger.warning(f"Error removing cache file {f}: {e}")
        except Exception as e:
            self.logger.warning(f"Error clearing cache: {e}")
    
    def disable(self) -> None:
        """Disable the cache."""
        self._enabled = False
    
    def enable(self) -> None:
        """Enable the cache."""
        self._enabled = True


def cached_trace(cache: Optional[TraceCache] = None):
    """Decorator to cache trace results.
    
    Args:
        cache: Optional TraceCache instance. If None, a default cache will be used.
    """
    if cache is None:
        cache = TraceCache()
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(collector, context, *args, **kwargs):
            # Create a cache key from the function name and arguments
            cache_key_parts = [
                func.__name__,
                str(context.project_path.absolute()),
                context.language.name,
                context.test_command,
                json.dumps(context.env_vars, sort_keys=True),
                str(args),
                json.dumps(kwargs, sort_keys=True)
            ]
            
            # Add file modification times for input files
            if hasattr(context, 'input_files'):
                for file_path in context.input_files:
                    try:
                        mtime = os.path.getmtime(file_path)
                        cache_key_parts.append(f"{file_path}:{mtime}")
                    except OSError:
                        pass
            
            cache_key = hashlib.sha256("|".join(cache_key_parts).encode('utf-8')).hexdigest()
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Not in cache, compute the result
            result = func(collector, context, *args, **kwargs)
            
            # Store in cache if successful
            if result.success:
                cache.set(cache_key, result)
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator


class ParallelTraceCollector(TraceCollector):
    """Extension of TraceCollector with parallel execution support."""
    
    def __init__(self, max_workers: Optional[int] = None, **kwargs):
        """Initialize the parallel trace collector.
        
        Args:
            max_workers: Maximum number of worker threads (default: os.cpu_count())
            **kwargs: Additional arguments for TraceCollector
        """
        super().__init__(**kwargs)
        self.max_workers = max_workers or os.cpu_count() or 4
        self.logger = logging.getLogger(f"{__name__}.ParallelTraceCollector")
    
    def capture_traces(
        self,
        contexts: List[ExecutionContext],
        output_dirs: Optional[List[Optional[Path]]] = None,
        test_filters: Optional[List[Optional[str]]] = None,
        extra_args_list: Optional[List[Optional[List[str]]]] = None
    ) -> List[TraceResult]:
        """Capture traces for multiple test contexts in parallel.
        
        Args:
            contexts: List of execution contexts
            output_dirs: Optional list of output directories (one per context)
            test_filters: Optional list of test filters (one per context)
            extra_args_list: Optional list of extra arguments (one per context)
            
        Returns:
            List of TraceResult objects in the same order as the input contexts
        """
        # Set default values for optional parameters
        if output_dirs is None:
            output_dirs = [None] * len(contexts)
        if test_filters is None:
            test_filters = [None] * len(contexts)
        if extra_args_list is None:
            extra_args_list = [None] * len(contexts)
        
        results = [None] * len(contexts)
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self.capture_trace,
                    context=contexts[i],
                    output_dir=output_dirs[i],
                    test_filter=test_filters[i],
                    extra_args=extra_args_list[i]
                ): i for i in range(len(contexts))
            }
            
            # Process results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    self.logger.error(f"Error in trace capture for context {idx}: {e}")
                    results[idx] = TraceResult(
                        success=False,
                        error=str(e),
                        stderr=str(e)
                    )
        
        return results


def trace_capture_timing(func: F) -> F:
    """Decorator to measure and log execution time of trace capture functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.monotonic()
        result = func(*args, **kwargs)
        end_time = time.monotonic()
        
        # Get the logger from the instance if it exists
        logger = None
        if len(args) > 0 and hasattr(args[0], 'logger'):
            logger = args[0].logger
        
        if logger:
            logger.info(
                f"{func.__name__} executed in {end_time - start_time:.2f} seconds"
            )
        
        return result
    
    return wrapper  # type: ignore


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a test context
    context = ExecutionContext(
        project_path=Path("/path/to/project"),
        language=Language.JAVA,
        test_command="./gradlew test"
    )
    
    # Create a parallel collector with caching
    cache = TraceCache()
    collector = ParallelTraceCollector(cache=cache)
    
    # Capture traces in parallel
    contexts = [context] * 5  # Example: 5 identical contexts
    results = collector.capture_traces(contexts)
    
    # Print results
    for i, result in enumerate(results):
        print(f"Test {i+1}: {'Success' if result.success else 'Failed'}")
        if not result.success and result.error:
            print(f"  Error: {result.error}")
