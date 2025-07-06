# UV Overview
Fast python package manager that replaces pip. 

# Features
- much faster than pip.
- parallel downloads and installations.

## Prerequisites
1. **Python**: Ensure Python 3.8 or higher is installed.
2. **uv**: To install uv check this https://docs.astral.sh/uv/getting-started/installation/ 

# How to Build and Run
**NOTE**: If you **pull** from the project repository, skip the steps below. Simply run ```uv sync ``` or ```uv run main.py $FLAG ``` and all dependencies will be installed. 

1. **Initialize the project**: ``` uv init $PROJECT_NAME```
This will create the following files inside the project:
```
payload-ai-suite/
├── .gitignore
├── .python-version
├── README.md
├── main.py
└── pyproject.toml
```
2. **Adding dependencies**:
This will create a **uv.lock** file if one does not already exist.
- With a requirements file:  ``` uv add -r requirements.txt ``` 
- Adding dependencies without requirements file: ``` uv add $PACKAGE-NAME-1 $PACKAGE-NAME-2 ... ```  

- **uv.lock**: a lock file the holds the versions of all your dependencies and sub-dependencies.


## Common Commands

Install packages
```uv add $PACKAGE```

Remove packages
``` uv remove $PACKAGE ```

Create a .venv virtual environment.
```uv venv``` 

Ensure all project dependencies are installed and up-to-date with the environment.
```uv sync```

Run scripts
```uv run $SCRIPT.py```

Run commands
``` uv run main.py $FLAG ``` 

Changing python versions
``` uv python pin $VERSION ```

Install specific Python version
```uv python install 3.10.5```

Will give a tree of dependencies being used in the project
``` uv tree ```