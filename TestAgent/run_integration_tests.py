#!/usr/bin/env python3
"""
Wrapper script to run integration tests with proper environment setup
Fixes macOS PyTorch multiprocessing issues
"""

import os
import sys

# CRITICAL: Set these BEFORE importing any torch-related modules
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Disable torch multiprocessing
os.environ['TORCH_NUM_THREADS'] = '1'

# Set multiprocessing start method BEFORE any imports
import multiprocessing
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Now it's safe to import and run tests
    import unittest
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    # Import test module
    test_dir = Path(__file__).parent / 'tests' / 'integration'
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
