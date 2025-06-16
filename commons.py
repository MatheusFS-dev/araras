"""
This module contains common imports and configs for the Araras project.
"""

try:
    import pretty_errors
except ImportError:
    print("\033[33mWARNING: Pretty Errors Module not found. Install it with 'pip install pretty_errors' for better error formatting.\033[0m")