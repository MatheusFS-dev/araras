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

        if self.supress_tf_warnings:
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
            start_new_session=False,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=False,
        )

        process.pid_file = pid_file
        return process


class TmuxTerminalLauncher(SimpleTerminalLauncher):
    """Launch commands inside a tmux split pane using ``libtmux``.

    The class behaves similarly to :class:`SimpleTerminalLauncher` but
    executes the given command in a new tmux split.  It falls back to the
    first available session when ``session_name`` is ``None``.

    Args:
        session_name: Optional tmux session to target. If the session does not
            exist it will be created.
        supress_tf_warnings: Whether to filter TensorFlow warnings from the
            command output.
    """

    __slots__ = ("session_name", "pane_id")

    def __init__(
        self, session_name: Optional[str] = None, supress_tf_warnings: bool = False
    ) -> None:
        super().__init__(supress_tf_warnings=supress_tf_warnings)
        self.session_name = session_name
        self.pane_id = None

    def launch(self, command: List[str], working_dir: str) -> subprocess.Popen:
        """Launch the command in a new tmux split pane.

        Args:
            command: Command array to execute.
            working_dir: Working directory in which the command should run.

        Returns:
            ``subprocess.Popen`` object for the ``tmux`` invocation.

        Raises:
            OSError: If ``tmux`` is not available or the command fails.
        """

        import libtmux

        cmd_str = " ".join(f'"{arg}"' for arg in command)
        if self.supress_tf_warnings:
            cmd_str = f"{cmd_str} 2> >(awk '!/ptxas/')"

        try:
            server = libtmux.Server()
        except Exception as exc:  # pragma: no cover - system dependent
            raise OSError(f"Failed to connect to tmux server: {exc}") from exc

        session = (
            server.find_where({"session_name": self.session_name})
            if self.session_name
            else server.list_sessions()[0]
            if server.list_sessions()
            else server.new_session(session_name="araras", attach=False)
        )

        window = session.attached_window
        pane = window.split_window(attach=False)
        pane.send_keys(cmd_str, enter=True)
        self.pane_id = pane.id

        # Return a short-lived process just to satisfy the interface.
        process = subprocess.Popen(["true"])
        process.pid_file = tempfile.mktemp(suffix=".pid")
        return process
