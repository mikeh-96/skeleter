# Skeleter Python Package

## 📦 **Installation Options**

### **Option 1: Install from Source (Development)**
```bash
cd skeleter_package
pip install -e .
```

### **Option 2: Install from PyPI (Once Published)**
```bash
pip install skeleter
```

### **Option 3: Install from Git Repository**
```bash
pip install git+https://github.com/yourusername/skeleter.git
```

## 🚀 **Usage After Installation**

### **Command Line Usage**
```bash
# After installation, 'skeleter' command is available globally
skeleter --help
skeleter --var service=my-service --var environment=production
skeleter --config /path/to/config.yaml --var version=1.2.3 --verbose
skeleter --var service=user-service --force-merge
```

### **Python API Usage**
```python
from skeleter import Skeleter

# Create instance
skeleter = Skeleter(config_path="config.yaml")

# Run with variables
cli_vars = {
    "service": "user-service", 
    "environment": "production",
    "version": "1.2.3"
}
skeleter.run(cli_vars, cli_force_merge=False)
```

### **Advanced Python Usage**
```python
from skeleter import Skeleter

# Custom configuration
skeleter = Skeleter()
skeleter.config = {
    'input_variables': {
        'service': 'payment-service',
        'environment': 'staging'
    },
    'parameter_store_map': {
        'api_key': '/services/{{ service }}/{{ environment }}/api-key'
    },
    'templates_map': {
        'https://github.com/org/repo': {
            'templates/{{ service }}/config.j2': 'config/{{ service }}/{{ environment }}.yaml'
        }
    }
}

# Run with CLI overrides
skeleter.run({'version': '2.1.0'})
```

## 📁 **Package Structure**
```
skeleter_package/
├── skeleter/                 # Main package
│   ├── __init__.py          # Package exports
│   └── main.py              # Core functionality
├── pyproject.toml           # Modern Python packaging
├── MANIFEST.in              # Additional files to include
├── README.md                # Documentation
├── LICENSE                  # MIT License
├── requirements.txt         # Dependencies
└── install_dev.sh          # Development installation script
```

## 🔧 **Development Setup**

### **1. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **2. Install in Development Mode**
```bash
cd skeleter_package
./install_dev.sh
```

### **3. Install Development Dependencies**
```bash
pip install -e ".[dev]"
```

## 📋 **Package Features**

✅ **Modern Python Packaging** (pyproject.toml)  
✅ **Console Script Entry Point** (`skeleter` command)  
✅ **Proper Package Structure**  
✅ **Type Hints Support**  
✅ **Development Dependencies**  
✅ **Editable Installation**  
✅ **MIT License**  
✅ **Comprehensive Dependencies**  

## 🚢 **Publishing to PyPI**

### **1. Build the Package**
```bash
pip install build twine
python -m build
```

### **2. Upload to PyPI**
```bash
twine upload dist/*
```

### **3. Test Installation**
```bash
pip install skeleter
skeleter --help
```

## 🐍 **Python Version Support**
- Python 3.8+
- Cross-platform (Windows, macOS, Linux)

## 📚 **Import Examples**

```python
# Import main class
from skeleter import Skeleter

# Import specific exceptions  
from skeleter.main import (
    SkeletorError,
    ConfigurationError, 
    ParameterStoreError,
    TemplateError,
    GitError,
    GitHubError
)

# Import main function
from skeleter import main
```

## 🔗 **Benefits of Package Installation**

1. **Global CLI Access**: Use `skeleter` command from anywhere
2. **Python API**: Import and use programmatically  
3. **Dependency Management**: Automatic handling of all dependencies
4. **Virtual Environment Friendly**: Works with venv, conda, etc.
5. **Easy Updates**: `pip install --upgrade skeleter`
6. **Distribution**: Share via PyPI, Git, or wheel files
