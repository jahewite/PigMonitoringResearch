import os
import sys

def add_project_root_to_path(relative_path_to_root='../'):
    """
    Adds the project root directory to the Python path to enable imports.

    Args:
        relative_path_to_root (str): The relative path from the current working directory to the project root.
    """
    project_root = os.path.abspath(os.path.join(os.getcwd(), relative_path_to_root))
    if project_root not in sys.path:
        sys.path.append(project_root)

    print(f"Project root added to sys.path: {project_root}")
