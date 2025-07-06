# UV Overview
Fast python package manager that replaces pip

# Features
- much faster than pip
- parallel downloads and installations

## Prerequisites
1. **Python**: Ensure Python 3.8 or higher is installed.
2. **uv**: To install uv check this https://docs.astral.sh/uv/getting-started/installation/ 

# How to Build and Run
1. **Initialize the project**: This will create a **pyproject.toml** file
``` uv init --no-readme```
2. **Adding dependencies**:
This will create a **uv.lock** file if one does not already exist
- With a requirements file:  ``` uv add -r requirements.txt ``` 
- Adding dependencies without requirements file: ``` uv add $package-name-1 $package-name-2 ... ```  

- **uv.lock**: a lock file the holds the versions of all your dependencies and sub-dependencies

## Common Commands

Install packages
```uv add $package```

Create a .venv virtual environment.
```uv venv``` 

Ensure all project dependencies are installed and up-to-date with the lockfile.
```uv sync```

Run scripts
```uv run $script.py```

Install specific Python version
```uv python install 3.10```