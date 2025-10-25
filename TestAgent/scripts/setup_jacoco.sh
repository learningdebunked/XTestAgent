#!/bin/bash
# Setup script for JaCoCo coverage tool
# This script downloads JaCoCo agent and CLI tools needed for patch verification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROJECT_ROOT/lib"

# JaCoCo version
JACOCO_VERSION="0.8.11"
JACOCO_URL="https://repo1.maven.org/maven2/org/jacoco/jacoco/${JACOCO_VERSION}/jacoco-${JACOCO_VERSION}.zip"

echo "=========================================="
echo "Setting up JaCoCo for TestAgent"
echo "=========================================="
echo ""

# Create lib directory if it doesn't exist
mkdir -p "$LIB_DIR"

# Download JaCoCo if not already present
if [ ! -f "$LIB_DIR/jacocoagent.jar" ] || [ ! -f "$LIB_DIR/jacococli.jar" ]; then
    echo "Downloading JaCoCo version ${JACOCO_VERSION}..."
    
    # Download to temp directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    curl -L -o jacoco.zip "$JACOCO_URL"
    
    echo "Extracting JaCoCo..."
    unzip -q jacoco.zip
    
    # Copy required files
    cp lib/jacocoagent.jar "$LIB_DIR/"
    cp lib/jacococli.jar "$LIB_DIR/"
    
    # Cleanup
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_DIR"
    
    echo "✓ JaCoCo installed successfully"
else
    echo "✓ JaCoCo already installed"
fi

# Verify installation
if [ -f "$LIB_DIR/jacocoagent.jar" ] && [ -f "$LIB_DIR/jacococli.jar" ]; then
    echo ""
    echo "=========================================="
    echo "JaCoCo Setup Complete!"
    echo "=========================================="
    echo "Agent JAR: $LIB_DIR/jacocoagent.jar"
    echo "CLI JAR:   $LIB_DIR/jacococli.jar"
    echo ""
    echo "You can now use the PatchVerificationAgent for trace-based verification."
else
    echo ""
    echo "ERROR: JaCoCo installation failed"
    exit 1
fi
