#!/usr/bin/env bash
# Quick installation script for TestAgentX
# Handles common dependency issues

set -e

echo "========================================="
echo "TestAgentX Quick Installation"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "❌ Python 3 not found"; exit 1; }
echo "✓ Python 3 found"
echo ""

# Upgrade pip first
echo "Upgrading pip..."
python3 -m pip install --upgrade pip
echo "✓ Pip upgraded"
echo ""

# Install minimal dependencies first
echo "Installing minimal dependencies..."
pip3 install pyyaml numpy tqdm pytest coverage
echo "✓ Minimal dependencies installed"
echo ""

# Try to install torch (may take a while)
echo "Installing PyTorch (this may take a few minutes)..."
pip3 install torch>=2.0.0 || echo "⚠️  PyTorch installation failed (optional for basic testing)"
echo ""

# Install transformers
echo "Installing transformers..."
pip3 install transformers>=4.30.0 || echo "⚠️  Transformers installation failed (optional for basic testing)"
echo ""

# Install javalang for Java parsing
echo "Installing Java parsing tools..."
pip3 install javalang>=0.13.0 || echo "⚠️  javalang installation failed (optional)"
echo ""

# Try tree-sitter with flexible version
echo "Installing tree-sitter..."
pip3 install 'tree-sitter>=0.20.0' 'tree-sitter-java>=0.21.0' || echo "⚠️  tree-sitter installation failed (optional)"
echo ""

echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Run tests:"
echo "  python3 tests/simple_test.py"
echo ""
echo "Or run validation:"
echo "  bash run_validation.sh --quick"
echo ""
