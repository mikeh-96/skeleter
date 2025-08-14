#!/bin/bash
# Development installation script for Skeleter package

echo "Installing Skeleter package in development mode..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using:"
    echo "   python -m venv venv && source venv/bin/activate"
    echo ""
fi

# Install in development mode
echo "Installing package in editable mode..."
pip install -e .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Installation successful!"
    echo ""
    echo "You can now use:"
    echo "  skeleter --help"
    echo "  skeleter --var service=my-service --var environment=production"
    echo ""
    echo "Or import in Python:"
    echo "  from skeleter import Skeleter"
else
    echo "✗ Installation failed!"
    exit 1
fi
