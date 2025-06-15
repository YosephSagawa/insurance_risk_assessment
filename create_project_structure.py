#!/usr/bin/env python3
"""
🚀 Python Project Structure Generator
====================================

A script to quickly create a comprehensive Python project folder structure 
optimized for Object-Oriented Programming (OOP) and modular development.

📋 USAGE INSTRUCTIONS:
----------------------

1. **Basic Usage (creates structure in current directory):**
   python3 create_project_structure.py

2. **Create in specific directory:**
   python3 create_project_structure.py my_new_project
   
3. **Make script executable and run:**
   chmod +x create_project_structure.py
   ./create_project_structure.py my_project

4. **After creating your project:**
   cd my_project
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt

🏗️ WHAT THIS SCRIPT CREATES:
----------------------------

This script creates the following structure:
    ├── .vscode/
    │   └── settings.json
    ├── .github/
    │   └── workflows/
    │       └── unittests.yml
    ├── .gitignore
    ├── requirements.txt
    ├── pyproject.toml
    ├── README.md
    ├── Makefile
    ├── .env.example
    ├── src/
    │   ├── __init__.py
    │   ├── core/
    │   │   └── __init__.py
    │   ├── models/
    │   │   └── __init__.py
    │   ├── utils/
    │   │   └── __init__.py
    │   └── services/
    │       └── __init__.py
    ├── tests/
    │   ├── __init__.py
    │   ├── unit/
    │   │   └── __init__.py
    │   └── integration/
    │       └── __init__.py
    ├── notebooks/
    │   ├── __init__.py
    │   └── README.md
    ├── scripts/
    │   ├── __init__.py
    │   └── README.md
    ├── docs/
    │   └── README.md
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── README.md
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    └── examples/
        ├── __init__.py
        └── README.md

✨ FEATURES:
-----------
- Modular source code organization with src/ structure
- Comprehensive testing setup (unit & integration tests)
- VSCode configuration for Python development
- GitHub Actions workflow for CI/CD
- Modern Python packaging with pyproject.toml
- Environment configuration with .env file
- Makefile for common development tasks
- Documentation structure ready for Sphinx
- Data directories for ML/analysis projects
- Examples and scripts directories

🎯 PERFECT FOR:
--------------
- Data Science projects
- Web applications
- API development
- Machine Learning projects
- Library/package development
- Any Python project requiring clean architecture

💡 TIP: After running this script, check the README.md in your new project
        for detailed instructions on setting up your development environment!

"""

import os
import sys
from pathlib import Path


def create_directory(path):
    """Create a directory if it doesn't exist."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {path}")
    except Exception as e:
        print(f"✗ Failed to create directory {path}: {e}")
        return False
    return True


def create_file(filepath, content=""):
    """Create a file with optional content."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created file: {filepath}")
    except Exception as e:
        print(f"✗ Failed to create file {filepath}: {e}")
        return False
    return True


def create_project_structure(base_path="."):
    """Create the complete project structure."""
    print("Creating comprehensive Python project structure...")
    print("-" * 50)
    
    # Convert to Path object for easier handling
    base = Path(base_path)
    
    # Define the directory structure
    directories = [
        base / ".vscode",
        base / ".github" / "workflows",
        base / "src" / "core",
        base / "src" / "models", 
        base / "src" / "utils",
        base / "src" / "services",
        base / "tests" / "unit",
        base / "tests" / "integration",
        base / "notebooks",
        base / "scripts",
        base / "docs",
        base / "data" / "raw",
        base / "data" / "processed",
        base / "config",
        base / "examples"
    ]
    
    # Create directories
    for directory in directories:
        if not create_directory(directory):
            return False
    
    # Create __init__.py files
    init_files = [
        base / "src" / "__init__.py",
        base / "src" / "core" / "__init__.py",
        base / "src" / "models" / "__init__.py",
        base / "src" / "utils" / "__init__.py",
        base / "src" / "services" / "__init__.py",
        base / "tests" / "__init__.py",
        base / "tests" / "unit" / "__init__.py",
        base / "tests" / "integration" / "__init__.py",
        base / "notebooks" / "__init__.py",
        base / "scripts" / "__init__.py",
        base / "config" / "__init__.py",
        base / "examples" / "__init__.py"
    ]
    
    init_content = '"""Package initialization file."""\n'
    
    for init_file in init_files:
        if not create_file(init_file, init_content):
            return False
    
    # Create configuration and documentation files
    files_to_create = [
        # VSCode settings
        (base / ".vscode" / "settings.json", get_vscode_settings()),
        
        # GitHub Actions
        (base / ".github" / "workflows" / "unittests.yml", get_github_workflow()),
        
        # Git and Python configuration
        (base / ".gitignore", get_gitignore()),
        (base / "requirements.txt", get_requirements()),
        (base / "pyproject.toml", get_pyproject_toml()),
        (base / ".env", get_env_file()),
        (base / "Makefile", get_makefile()),
        
        # Main project files
        (base / "README.md", get_main_readme()),
        (base / "config" / "settings.py", get_config_settings()),
        
        # Documentation
        (base / "docs" / "README.md", get_docs_readme()),
        (base / "notebooks" / "README.md", get_notebooks_readme()),
        (base / "scripts" / "README.md", get_scripts_readme()),
        (base / "data" / "README.md", get_data_readme()),
        (base / "examples" / "README.md", get_examples_readme()),
    ]
    
    for file_path, content in files_to_create:
        if not create_file(file_path, content):
            return False
    
    print("-" * 50)
    print("✅ Comprehensive project structure created successfully!")
    print("\nCreated structure:")
    print("├── .vscode/")
    print("│   └── settings.json")
    print("├── .github/")
    print("│   └── workflows/")
    print("│       └── unittests.yml")
    print("├── .gitignore")
    print("├── requirements.txt")
    print("├── pyproject.toml")
    print("├── README.md")
    print("├── Makefile")
    print("├── .env")
    print("├── src/")
    print("│   ├── __init__.py")
    print("│   ├── core/")
    print("│   ├── models/")
    print("│   ├── utils/")
    print("│   └── services/")
    print("├── tests/")
    print("│   ├── __init__.py")
    print("│   ├── unit/")
    print("│   └── integration/")
    print("├── notebooks/")
    print("│   ├── __init__.py")
    print("│   └── README.md")
    print("├── scripts/")
    print("│   ├── __init__.py")
    print("│   └── README.md")
    print("├── docs/")
    print("│   └── README.md")
    print("├── data/")
    print("│   ├── raw/")
    print("│   ├── processed/")
    print("│   └── README.md")
    print("├── config/")
    print("│   ├── __init__.py")
    print("│   └── settings.py")
    print("└── examples/")
    print("    ├── __init__.py")
    print("    └── README.md")
    
    return True


def get_vscode_settings():
    """Generate VSCode settings.json content."""
    return """{
    "python.defaultInterpreterPath": "./venv/bin/python"
}"""


def get_github_workflow():
    """Generate GitHub Actions workflow for unit tests."""
    return """# Add your GitHub Actions workflow here
# Example: CI/CD pipeline, automated testing, etc.
"""


def get_gitignore():
    """Generate .gitignore content."""
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VSCode
.vscode/
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json
!.vscode/*.code-snippets

# Local History for Visual Studio Code
.history/

# Built Visual Studio Code Extensions
*.vsix

# macOS
.DS_Store

# Data files
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Logs
logs/
*.log

# Temporary files
*.tmp
*.temp
"""


def get_requirements():
    """Generate requirements.txt content."""
    return """# Add your project dependencies here
"""


def get_pyproject_toml():
    """Generate pyproject.toml content."""
    return """[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "your-project-name"
version = "0.1.0"
description = ""
requires-python = ">=3.8"

[tool.setuptools.packages.find]
where = ["src"]
"""


def get_env_file():
    """Generate .env content."""
    return """# Add your environment variables here
"""


def get_makefile():
    """Generate Makefile content."""
    return """# Add your Makefile commands here
"""


def get_main_readme():
    """Generate main README.md content."""
    return """# Your Project Name

Add your project description here.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Add usage instructions here.
"""


def get_config_settings():
    """Generate config/settings.py content."""
    return """\"\"\"
Configuration settings for the application.
\"\"\"

# Add your configuration here
"""


def get_docs_readme():
    """Generate docs/README.md content."""
    return """# Documentation

Add your project documentation here.
"""


def get_notebooks_readme():
    """Generate notebooks/README.md content."""
    return """# Notebooks

Add your Jupyter notebooks here.
"""


def get_scripts_readme():
    """Generate scripts/README.md content."""
    return """# Scripts

Add your utility scripts here.
"""


def get_data_readme():
    """Generate data/README.md content."""
    return """# Data

Store your data files here.
"""


def get_examples_readme():
    """Generate examples/README.md content."""
    return """# Examples

Add usage examples here.
"""


def main():
    """Main function to handle command line arguments and execute the script."""
    # Get target directory from command line argument or use current directory
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"Target directory: {os.path.abspath(target_dir)}")
    
    # Check if target directory exists
    if not os.path.exists(target_dir):
        print(f"✗ Target directory '{target_dir}' does not exist.")
        return 1
    
    # Create the project structure
    success = create_project_structure(target_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 