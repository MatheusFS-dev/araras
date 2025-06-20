import os
import platform
import subprocess
import tempfile
from typing import *


# ————————————————————————————— Terminal Launcher ———————————————————————————— #
class SimpleTerminalLauncher:
    """Minimal terminal launcher for cross-platform execution."""

    __slots__ = ("system",)

    def __init__(self):
        """Initialize launcher with OS detection."""
        self.system = platform.system().lower()

    def launch(self, command: List[str], working_dir: str) -> subprocess.Popen:
        """Launch command in new terminal with PID capture.

        Args:
            command: Command array to execute
            working_dir: Working directory

        Returns:
            Terminal process object with pid_file attribute

        Raises:
            OSError: If unsupported OS or launch fails
        """
        pid_file = tempfile.mktemp(suffix=".pid")
        cmd_str = " ".join(f'"{arg}"' for arg in command)
        cmd_str = f"{cmd_str} 2> >(awk '!/ptxas/')"

        # Simple PID capture without exit code complexity
        full_cmd = f"({cmd_str}) & echo $! > '{pid_file}'; wait"

        # OS-specific terminal commands optimized for common terminals
        if self.system == "linux":
            terminal_cmd = ["gnome-terminal", "--", "bash", "-c", full_cmd]
        elif self.system == "darwin":
            terminal_cmd = ["osascript", "-e", f'tell application "Terminal" to do script "{full_cmd}"']
        elif self.system == "windows":
            terminal_cmd = ["cmd", "/c", full_cmd]
        else:
            raise OSError(f"Unsupported OS: {self.system}")

        process = subprocess.Popen(
            terminal_cmd,
            cwd=working_dir,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True if os.name != "nt" else False,
        )

        process.pid_file = pid_file
        return process
