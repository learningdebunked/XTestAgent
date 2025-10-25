# Installation Guide - Step by Step

## Quick Install (Recommended)

```bash
# 1. Upgrade pip
python3 -m pip install --upgrade pip

# 2. Run quick install script
chmod +x quick_install.sh
bash quick_install.sh

# 3. Test
python3 tests/simple_test.py
```

---

## Manual Installation

### Step 1: Upgrade Pip

```bash
python3 -m pip install --upgrade pip
```

### Step 2: Install Core Dependencies

```bash
# Minimal (for testing only)
pip3 install pyyaml numpy tqdm pytest coverage

# Or full (for all features)
pip3 install -r requirements.txt
```

### Step 3: Handle Version Conflicts

If you see version errors like:
```
ERROR: Could not find a version that satisfies the requirement X==Y.Z
```

**Solution**: The requirements.txt has been updated to use flexible versions (>=).

```bash
# Install with updated requirements
pip3 install -r requirements.txt
```

---

## Common Installation Issues

### Issue 1: tree-sitter-java version not found

**Error**:
```
ERROR: Could not find a version that satisfies the requirement tree-sitter-java==0.20.2
```

**Solution**: ✅ **FIXED** - requirements.txt now uses `tree-sitter-java>=0.21.0`

```bash
pip3 install 'tree-sitter-java>=0.21.0'
```

### Issue 2: Old pip version

**Error**:
```
WARNING: You are using pip version 21.2.4; however, version 25.3 is available.
```

**Solution**:
```bash
python3 -m pip install --upgrade pip
```

### Issue 3: Permission denied

**Error**:
```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied
```

**Solution**:
```bash
# Option 1: Use --user flag
pip3 install --user -r requirements.txt

# Option 2: Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue 4: No matching distribution for python>=3.10

**Error**:
```
ERROR: No matching distribution found for python>=3.10
```

**Solution**: ✅ **FIXED** - removed from requirements.txt

Python version is not a pip package. Just ensure you have Python 3.8+:
```bash
python3 --version
```

---

## Installation Options

### Option 1: Minimal Install (Fastest)

For basic testing and validation:

```bash
pip3 install pyyaml numpy tqdm pytest coverage
```

**What works**:
- ✅ Configuration system
- ✅ Error handling
- ✅ Simple tests
- ✅ Basic validation

**What doesn't work**:
- ❌ Code encoding (needs torch/transformers)
- ❌ LLM test generation (needs openai)
- ❌ Full integration tests

### Option 2: Core Install (Recommended)

For most features:

```bash
pip3 install torch transformers pyyaml numpy tqdm pytest coverage javalang
```

**What works**:
- ✅ Everything in minimal
- ✅ Code encoding
- ✅ AST/CFG generation
- ✅ Most validation features

**What doesn't work**:
- ❌ LLM test generation (needs API key)
- ❌ Knowledge graph (needs Neo4j)

### Option 3: Full Install

For all features:

```bash
pip3 install -r requirements.txt
```

**What works**:
- ✅ Everything!

**Additional setup needed**:
- OpenAI API key for LLM generation
- Neo4j for knowledge graph

---

## Verification

After installation, verify it works:

### Quick Verification

```bash
python3 tests/simple_test.py
```

**Expected output**:
```
🎉 All tests passed!
```

### Detailed Verification

```bash
# Test imports
python3 -c "from config.config_loader import get_config; print('✓ Config works')"
python3 -c "from utils.error_handling import setup_logging; print('✓ Error handling works')"
python3 -c "import yaml; print('✓ YAML works')"
python3 -c "import numpy; print('✓ NumPy works')"

# Optional: Test torch
python3 -c "import torch; print('✓ PyTorch works')" || echo "⚠️  PyTorch not installed (optional)"

# Optional: Test transformers
python3 -c "import transformers; print('✓ Transformers works')" || echo "⚠️  Transformers not installed (optional)"
```

---

## Platform-Specific Instructions

### macOS

```bash
# Install Xcode command line tools (if needed)
xcode-select --install

# Use Homebrew for Python (recommended)
brew install python@3.11

# Then install dependencies
python3 -m pip install --upgrade pip
bash quick_install.sh
```

### Linux (Ubuntu/Debian)

```bash
# Install Python and pip
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Install dependencies
python3 -m pip install --upgrade pip
bash quick_install.sh
```

### Windows

```bash
# Use Git Bash or WSL

# Install Python from python.org
# Then:
python -m pip install --upgrade pip
bash quick_install.sh
```

---

## Virtual Environment (Recommended)

Using a virtual environment avoids conflicts:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python tests/simple_test.py

# Deactivate when done
deactivate
```

---

## Troubleshooting Installation

### Check Python Version

```bash
python3 --version
# Should be 3.8 or higher
```

### Check Pip Version

```bash
pip3 --version
# Should be recent (20.0+)
```

### Check Installed Packages

```bash
pip3 list | grep -E "torch|transformers|yaml|numpy"
```

### Clear Cache and Reinstall

```bash
# Clear pip cache
pip3 cache purge

# Reinstall
pip3 install --no-cache-dir -r requirements.txt
```

### Use Conda (Alternative)

```bash
# Create conda environment
conda create -n testagentx python=3.9
conda activate testagentx

# Install dependencies
pip install -r requirements.txt
```

---

## What to Install for Different Use Cases

### Just Want to Test the System

```bash
pip3 install pyyaml numpy tqdm pytest
python3 tests/simple_test.py
```

### Want to Run Validation

```bash
pip3 install pyyaml numpy tqdm pytest coverage
bash run_validation.sh --quick
```

### Want to Generate Tests

```bash
pip3 install torch transformers openai pyyaml numpy
# Set API key
export OPENAI_API_KEY=your_key_here
python3 -m layer2_test_generation.llm_test_agent
```

### Want Full Functionality

```bash
pip3 install -r requirements.txt
# Setup Neo4j for knowledge graph
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```

---

## Success Checklist

After installation, you should be able to:

- [ ] Run `python3 tests/simple_test.py` successfully
- [ ] Import config: `python3 -c "from config.config_loader import get_config"`
- [ ] Import error handling: `python3 -c "from utils.error_handling import setup_logging"`
- [ ] Run validation script: `bash run_validation.sh --quick`

If all checked, you're ready to go! 🚀

---

## Getting Help

If you're still having issues:

1. Check `TROUBLESHOOTING.md`
2. Run diagnostics: `bash check_setup.sh`
3. Open an issue with:
   - Python version (`python3 --version`)
   - Pip version (`pip3 --version`)
   - OS (`uname -a`)
   - Error message

---

## Quick Reference

```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Quick install
bash quick_install.sh

# Test
python3 tests/simple_test.py

# Validate
bash run_validation.sh --quick
```
