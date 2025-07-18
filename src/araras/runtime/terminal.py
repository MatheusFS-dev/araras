from araras.core import *

import platform
import subprocess
import tempfile
import shlex


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
    """Terminal launcher that executes commands in tmux panes.

    This behaves similarly to :class:`SimpleTerminalLauncher` but splits the
    target tmux window and sends the command to that new pane. If
    ``session_name`` is ``None`` the first available session is used, or a new
    session is created when none exist.

    Args:
        session_name: Name of the tmux session to target. When the session does
            not exist, it will be created automatically.
        supress_tf_warnings: Whether to filter TensorFlow warnings from the
            command output.
    """

    __slots__ = ("session_name", "pane_id")

    def __init__(
        self, session_name: Optional[str] = None, supress_tf_warnings: bool = False
    ) -> None:
        """Initialize the tmux launcher.

        Args:
            session_name: Target tmux session name. A new session is created if
                it does not already exist.
            supress_tf_warnings: Whether to filter TensorFlow warnings from the
                command output.
        """
        super().__init__(supress_tf_warnings=supress_tf_warnings)
        self.session_name = session_name
        self.pane_id = None

    def launch(self, command: List[str], working_dir: str) -> subprocess.Popen:
        """Launch a command in a new tmux split pane.

        Args:
            command: Sequence of command arguments.
            working_dir: Directory where the command should run.

        Returns:
            ``subprocess.Popen`` instance used solely for PID file bookkeeping.

        Raises:
            OSError: If ``tmux`` cannot be contacted or the split fails.
        """

        import libtmux

        pid_file = tempfile.mktemp(suffix=".pid")
        cmd_str = " ".join(shlex.quote(arg) for arg in command)
        if self.supress_tf_warnings:
            cmd_str = f"{cmd_str} 2> >(awk '!/ptxas/')"

        full_cmd = (
            f"cd {shlex.quote(working_dir)} && ( {cmd_str} )"
            f" & echo $! > {shlex.quote(pid_file)}; wait"
        )

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
        pane.send_keys(full_cmd, enter=True)
        self.pane_id = pane.id

        process = subprocess.Popen(["sleep", "infinity"])
        process.pid_file = pid_file
        process.pane_id = pane.id
        process.session_name = session.get("session_name")
        return process
