"""General helpers shared across the project."""
from pathlib import Path

def ensure_parent(path):
    """Create parent directories before writing a file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
