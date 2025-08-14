#!/bin/bash
# Build script for Skeleter package

echo "🔨 Building Skeleter Package..."
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Install build dependencies
echo "Installing build dependencies..."
pip install --upgrade build twine

# Build the package
echo "Building package..."
python -m build

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "📦 Generated files:"
    ls -la dist/
    echo ""
    echo "🚀 To install locally:"
    echo "   pip install dist/skeleter-1.0.0-py3-none-any.whl"
    echo ""
    echo "📤 To upload to PyPI:"
    echo "   twine upload dist/*"
    echo ""
    echo "🧪 To test upload to Test PyPI:"
    echo "   twine upload --repository testpypi dist/*"
else
    echo "❌ Build failed!"
    exit 1
fi
