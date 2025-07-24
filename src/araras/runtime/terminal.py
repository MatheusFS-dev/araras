from araras.core import *

import platform
import subprocess
import tempfile


# ————————————————————————————— Terminal Launcher ———————————————————————————— #
class SimpleTerminalLauncher:
    """Minimal terminal launcher for cross-platform execution."""

    __slots__ = ("system", "supress_tf_warnings")

    def __init__(self, supress_tf_warnings: bool = False):
        """Initialize launcher with OS detection."""
        self.system = platform.system().lower()
        self.supress_tf_warnings = supress_tf_warnings

    def set_supress_tf_warnings(self, value: bool) -> None:
        """Set the supress_tf_warnings attribute.

        Args:
            value: Boolean indicating whether to suppress TensorFlow warnings.
        """
        self.supress_tf_warnings = value

    def launch(self, command: List[str], working_dir: str) -> subprocess.Popen:
        """Launch a command in a new terminal.

        The command is executed inside a platform specific terminal window and
        a temporary file is created to capture the spawned process ID. When
        ``supress_tf_warnings`` is enabled, only TensorFlow register spill
        warnings produced by the ``ptxas`` compiler are filtered from stderr so
        that all other log messages remain visible.

        Args:
            command: Sequence of command arguments to execute.
            working_dir: Directory where ``command`` should run.

        Returns:
            The ``subprocess.Popen`` instance representing the terminal process.
            The object includes a ``pid_file`` attribute with the path to the
            temporary PID file.

        Raises:
            OSError: If the current operating system is unsupported or the
                terminal fails to launch.
        """
        pid_file = tempfile.mktemp(suffix=".pid")
        cmd_str = " ".join(f'"{arg}"' for arg in command)

        if self.supress_tf_warnings:
            # Filter only XLA ``ptxas`` register spill warnings while keeping
            # all other TensorFlow logs visible.
            cmd_str = f"{cmd_str} 2>&1 | grep -v 'ptxas warning' | grep -v '^$'"

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
            start_new_session=False,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=False,
        )

        process.pid_file = pid_file
        return process
