"""
This module contains common imports and configs for the Araras project.
"""

try:
    import pretty_errors
except ImportError:
    print("[33mWARNING: Pretty Errors Module not found. Install it with 'pip install pretty_errors' for better error formatting.[0m")

try:
    from matplotlib import font_manager
    if not any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist):
        print("[33mWARNING: 'Times New Roman' font not found. Install the font (e.g. sudo apt-get install ttf-mscorefonts-installer) to avoid matplotlib warnings.[0m")
except Exception:
    pass
