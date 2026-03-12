"""
pytest conftest — shared fixtures available to all test modules.
Adds project root to sys.path so imports work without installation.
"""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
