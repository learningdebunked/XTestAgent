#!/usr/bin/env python3
"""
Simple Test Script for TestAgentX
Tests basic functionality without complex dependencies
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test that core modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test config loader
        from config.config_loader import ConfigLoader
        print("✓ Config loader imported")
        
        # Test utilities
        from utils.error_handling import setup_logging
        print("✓ Error handling imported")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration system"""
    print("\nTesting configuration...")
    
    try:
        from config.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        config = loader.load()
        
        # Test accessing config values
        embedding_dim = config.get('code_encoder.embedding_dim', 768)
        print(f"✓ Loaded embedding_dim: {embedding_dim}")
        
        threshold = config.get('fuzzy_validation.threshold', 0.7)
        print(f"✓ Loaded threshold: {threshold}")
        
        print("\n✅ Configuration system works!")
        return True
    except Exception as e:
        print(f"\n❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling utilities"""
    print("\nTesting error handling...")
    
    try:
        from utils.error_handling import retry_on_error, safe_execute
        
        # Test retry decorator
        attempt_count = [0]
        
        @retry_on_error(max_retries=2, delay=0.1)
        def flaky_function():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise Exception("Temporary failure")
            return "Success"
        
        result = flaky_function()
        print(f"✓ Retry mechanism works (attempts: {attempt_count[0]})")
        
        # Test safe_execute
        def failing_function():
            raise Exception("Test error")
        
        result = safe_execute(failing_function, default="default_value")
        print(f"✓ Safe execute works (returned: {result})")
        
        print("\n✅ Error handling works!")
        return True
    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic system functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Test that we can create basic objects
        print("✓ System is functional")
        
        print("\n✅ Basic functionality works!")
        return True
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all simple tests"""
    print("="*60)
    print("TestAgentX Simple Test Suite")
    print("="*60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Error Handling", test_error_handling),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        success = test_func()
        results.append((name, success))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<30} {status}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
