# Troubleshooting Guide

## Quick Fixes

### Error: "Could not find a version that satisfies the requirement python>=3.10"

**Problem**: `requirements.txt` had `python>=3.10` which is not a pip package.

**Solution**: This has been fixed! Now run:

```bash
# Upgrade pip first
python3 -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Alternative**: Use minimal requirements for quick testing:
```bash
pip install -r requirements-minimal.txt
```

---

### Error: "sh: run_validation.sh: No such file or directory"

**Problem**: Wrong directory or using `sh` instead of `bash`.

**Solution**:
```bash
# Make sure you're in the TestAgent directory
cd TestAgent

# Use bash instead of sh
bash run_validation.sh --quick

# Or make it executable and run directly
chmod +x run_validation.sh
./run_validation.sh --quick
```

---

### Error: "⚠ Maven not found"

**Problem**: Maven is not installed (only needed for Java projects).

**Solution**:

**On macOS**:
```bash
brew install maven
```

**On Ubuntu/Debian**:
```bash
sudo apt-get install maven
```

**On Windows**:
```bash
# Download from https://maven.apache.org/download.cgi
# Or use chocolatey:
choco install maven
```

**Skip Maven** (if testing Python only):
```bash
# Maven is optional for Python-only testing
python tests/integration/test_end_to_end_pipeline.py
```

---

### Error: "mutex lock failed: Invalid argument" or "libc++abi: terminating"

**Problem**: PyTorch multiprocessing issue on macOS.

**Solution**: This has been fixed in the code! But if you still see it:

```bash
# Option 1: Use the simple test instead
python3 tests/simple_test.py

# Option 2: Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 tests/integration/test_end_to_end_pipeline.py

# Option 3: Run with spawn method
python3 -c "import multiprocessing; multiprocessing.set_start_method('spawn'); exec(open('tests/integration/test_end_to_end_pipeline.py').read())"
```

---

### Error: "ModuleNotFoundError: No module named 'torch'"

**Problem**: Dependencies not installed.

**Solution**:
```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or minimal for testing
pip install -r requirements-minimal.txt
```

---

### Error: "ImportError: cannot import name 'X' from 'Y'"

**Problem**: Incompatible package versions.

**Solution**:
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or use minimal requirements
pip install -r requirements-minimal.txt
```

---

### Error: "Neo4j connection failed"

**Problem**: Neo4j is not running (optional component).

**Solution**:

**Option 1**: Skip Neo4j (for quick testing):
```bash
# Edit config to disable knowledge graph
vim config/default_config.yaml
# Set: features.enable_knowledge_graph: false
```

**Option 2**: Start Neo4j with Docker:
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

**Option 3**: Install Neo4j locally:
```bash
# Download from https://neo4j.com/download/
# Start Neo4j and set password to 'password'
```

---

### Error: "CUDA out of memory"

**Problem**: GPU memory exhausted.

**Solution**:
```bash
# Use CPU instead
export TESTAGENTX_FEATURES__USE_GPU=false

# Or reduce batch size
export TESTAGENTX_CODE_ENCODER__BATCH_SIZE=8
```

---

### Error: "OpenAI API key not found"

**Problem**: API key not set (only needed for LLM test generation).

**Solution**:
```bash
# Set API key
export OPENAI_API_KEY=your_api_key_here

# Or add to config
vim config/default_config.yaml
# Add your API key in the appropriate section
```

**Skip LLM** (for quick testing):
```bash
# Run tests without LLM generation
python tests/integration/test_end_to_end_pipeline.py \
  --skip-llm
```

---

## Step-by-Step Recovery

If you're having multiple issues, follow these steps:

### Step 1: Check Python Version

```bash
python3 --version
# Should be 3.8 or higher
```

If too old:
```bash
# On macOS
brew install python@3.11

# On Ubuntu
sudo apt-get install python3.11
```

### Step 2: Clean Install

```bash
# Remove old virtual environment
rm -rf venv

# Create new one
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install minimal requirements first
pip install -r requirements-minimal.txt

# Test
python -c "import torch; print('Success!')"
```

### Step 3: Run Quick Test

```bash
# Run integration tests
python tests/integration/test_end_to_end_pipeline.py

# If it works, install full requirements
pip install -r requirements.txt
```

### Step 4: Run Validation

```bash
# Quick validation
bash run_validation.sh --quick

# Or full validation
bash run_validation.sh sample_project
```

---

## Common Issues by Platform

### macOS

**Issue**: "xcrun: error: invalid active developer path"
```bash
# Solution: Install Xcode command line tools
xcode-select --install
```

**Issue**: "zsh: command not found: python"
```bash
# Solution: Use python3
alias python=python3
# Or add to ~/.zshrc
```

### Linux

**Issue**: "python3-venv not found"
```bash
# Solution: Install venv
sudo apt-get install python3-venv
```

**Issue**: "Permission denied"
```bash
# Solution: Use sudo or fix permissions
sudo chown -R $USER:$USER .
```

### Windows

**Issue**: "Scripts\activate: cannot be loaded"
```bash
# Solution: Change execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
venv\Scripts\activate
```

**Issue**: "bash: command not found"
```bash
# Solution: Use Git Bash or WSL
# Or run Python directly
python evaluation/run_full_evaluation.py
```

---

## Quick Diagnostics

Run this to check your setup:

```bash
#!/bin/bash
echo "=== TestAgentX Diagnostics ==="
echo ""

echo "Python version:"
python3 --version || echo "❌ Python not found"

echo ""
echo "Pip version:"
pip --version || echo "❌ Pip not found"

echo ""
echo "Java version:"
java -version 2>&1 | head -1 || echo "⚠️  Java not found (optional)"

echo ""
echo "Maven version:"
mvn -version 2>&1 | head -1 || echo "⚠️  Maven not found (optional)"

echo ""
echo "Virtual environment:"
if [ -d "venv" ]; then
    echo "✓ venv exists"
else
    echo "❌ venv not found"
fi

echo ""
echo "Key packages:"
python3 -c "import torch; print('✓ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch not installed"
python3 -c "import transformers; print('✓ transformers:', transformers.__version__)" 2>/dev/null || echo "❌ transformers not installed"
python3 -c "import pytest; print('✓ pytest:', pytest.__version__)" 2>/dev/null || echo "❌ pytest not installed"

echo ""
echo "=== End Diagnostics ==="
```

Save as `check_setup.sh` and run:
```bash
bash check_setup.sh
```

---

## Still Having Issues?

1. **Check Python version**: Must be 3.8+
   ```bash
   python3 --version
   ```

2. **Try minimal install**:
   ```bash
   pip install -r requirements-minimal.txt
   ```

3. **Run simple test**:
   ```bash
   python -c "import torch; print('OK')"
   ```

4. **Check logs**:
   ```bash
   tail -f logs/testagentx.log
   ```

5. **Ask for help**:
   - Open an issue on GitHub
   - Include output of `bash check_setup.sh`
   - Include error messages

---

## Quick Reference

```bash
# Fix requirements error
pip install --upgrade pip
pip install -r requirements.txt

# Fix script error
bash run_validation.sh --quick  # Use bash, not sh

# Fix import errors
source venv/bin/activate
pip install -r requirements-minimal.txt

# Fix Maven warning
brew install maven  # macOS
sudo apt-get install maven  # Linux

# Skip optional components
export TESTAGENTX_FEATURES__ENABLE_KNOWLEDGE_GRAPH=false
export TESTAGENTX_FEATURES__USE_GPU=false
```

---

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Pip upgraded
- [ ] Dependencies installed
- [ ] Integration tests pass
- [ ] Validation script runs

Once all checked, you're ready to go! 🚀
