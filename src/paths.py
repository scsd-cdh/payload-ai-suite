"""Path utilities to handle running from different directories."""
import os
import sys


def get_project_root():
    """Get the absolute path to the project root directory.
    
    Works whether the script is run from project root, src/, or any subdirectory.
    """
    # Get the directory containing this file (paths.py)
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    
    # Project root is one level up from src/
    project_root = os.path.dirname(src_dir)
    
    return project_root


def get_data_dir():
    """Get the absolute path to the data directory."""
    return os.path.join(get_project_root(), "data")


def get_src_dir():
    """Get the absolute path to the src directory."""
    return os.path.join(get_project_root(), "src")


def resolve_path(relative_path):
    """Convert a relative path to absolute, handling different execution contexts.
    
    Args:
        relative_path: Path relative to project root (e.g., "data/labeled/yes")
    
    Returns:
        Absolute path
    """
    # Remove any leading ../ or ./ 
    cleaned_path = relative_path
    while cleaned_path.startswith('../') or cleaned_path.startswith('./'):
        if cleaned_path.startswith('../'):
            cleaned_path = cleaned_path[3:]
        else:
            cleaned_path = cleaned_path[2:]
    
    # If it starts with 'data/', 'src/', etc., it's relative to project root
    if cleaned_path.startswith(('data/', 'src/', 'models/', 'events/')):
        return os.path.join(get_project_root(), cleaned_path)
    
    # Otherwise, assume it's a full relative path from project root
    return os.path.join(get_project_root(), cleaned_path)


# Convenience functions for common paths
def get_labeled_dir():
    """Get path to labeled data directory."""
    return os.path.join(get_data_dir(), "labeled")


def get_eonet_dir():
    """Get path to EONET fire events directory."""
    return os.path.join(get_data_dir(), "eonet_fire_events")


def get_sen2fire_dir():
    """Get path to Sen2Fire data directory."""
    return os.path.join(get_data_dir(), "sen2fire")


def get_gcs_credentials_path():
    """Get path to GCS credentials file."""
    return os.path.join(get_project_root(), ".gcs", "zetane-api-a76252006fa7.json")


def get_model_path(model_name="zetane.onnx"):
    """Get path to model file."""
    return os.path.join(get_project_root(), model_name)


# Add src to Python path if needed
def ensure_imports_work():
    """Ensure imports work regardless of where script is run from."""
    src_dir = get_src_dir()
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)