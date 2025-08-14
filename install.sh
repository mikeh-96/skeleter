#!/bin/bash
# Installation script for Skeleter

echo "Installing Skeleter dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install dependencies
pip install -r requirements.txt

echo "Dependencies installed successfully!"
echo ""
echo "To use Skeleter:"
echo "1. Configure your AWS credentials"
echo "2. Edit config.yaml to match your Parameter Store paths"
echo "3. Create your templates in the templates/ directory"
echo "4. Run: python skeleter.py --var key1=value1 --var key2=value2"
echo ""
echo "For more information, see README.md"
