"""
This module contains common imports and configs for the Araras project.
"""

try:
    import pretty_errors
except ImportError:
    print(
        "\033[33mWARNING: Pretty Errors Module not found. Install it with '\033[34mpip install pretty_errors\033[33m' for better error formatting.\033[0m"
    )

try:
    from matplotlib import font_manager

    if not any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist):
        print("\033[33mWARNING: 'Times New Roman' font not found.\033[0m")
        print(
            "\033[33mInstall the font by running '\033[34msudo apt install msttcorefonts -qq && rm ~/.cache/matplotlib -rf\033[33m'.\033[0m"
        )
except Exception:
    pass
