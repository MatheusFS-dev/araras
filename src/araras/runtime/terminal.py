from typing import List

import os
import platform
import shlex
import shutil
import subprocess
import tempfile


# ————————————————————————————— Terminal Launcher ———————————————————————————— #
class SimpleTerminalLauncher:
    """Launch monitored processes either in a GUI terminal or inline.

    The launcher keeps the existing desktop-terminal behavior for graphical
    Linux, macOS, and Windows sessions, but falls back to inline execution on
    headless Linux environments such as SSH shells. Inline execution is
    intentionally limited to the Linux headless case requested by the monitor
    workflow so the rest of the package keeps its existing behavior.

    Attributes:
        system (str): Lowercase platform name from ``platform.system()``.
        supress_tf_warnings (bool): If ``True``, only ``ptxas warning`` lines
            are filtered from stderr. If ``False``, stderr is passed through
            unchanged. Filtering is implemented only for POSIX shell launches
            because the current monitor flow uses Linux shells.
    """

    __slots__ = ("system", "supress_tf_warnings")

    def __init__(self, supress_tf_warnings: bool = False):
        """Initialize launcher state for the active operating system.

        Args:
            supress_tf_warnings (bool): If ``True``, launch commands with a
                shell redirection that removes only ``ptxas warning`` lines
                from stderr. This keeps most TensorFlow output visible but adds
                a shell wrapper around POSIX launches. If ``False``, commands
                are launched without any stderr filtering and no extra shell
                redirection is added.

        Returns:
            None: The constructor only stores runtime configuration.
        """
        self.system = platform.system().lower()
        self.supress_tf_warnings = supress_tf_warnings

    def set_supress_tf_warnings(self, value: bool) -> None:
        """Update whether TensorFlow ``ptxas`` warnings are filtered.

        Args:
            value (bool): If ``True``, subsequent launches filter only
                ``ptxas warning`` lines from stderr on POSIX shells. If
                ``False``, launches keep stderr unchanged. This switch affects
                output visibility only; it does not change the executed command
                or restart behavior.

        Returns:
            None: The method mutates launcher state in place.

        Raises:
            TypeError: If ``value`` is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError("value must be a bool")
        self.supress_tf_warnings = value

    def _has_linux_gui_terminal(self) -> bool:
        """Return whether Linux can open a graphical terminal window.

        The monitor should only depend on ``gnome-terminal`` when the current
        session is graphical and the executable is available. Headless shells
        should use inline execution instead of failing during launch.

        Returns:
            bool: ``True`` when both a GUI display server and
            ``gnome-terminal`` are available. ``False`` means the launcher
            should stay in the current shell session.
        """
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        return has_display and shutil.which("gnome-terminal") is not None

    def _build_posix_command(self, command: List[str]) -> str:
        """Build a shell-safe POSIX command string for monitor launches.

        Args:
            command (List[str]): Command arguments in the order they should be
                executed. Each argument is quoted with ``shlex.quote`` so file
                paths with spaces remain valid in shell mode.

        Returns:
            str: A shell command string ready for ``bash -lc``.
        """
        command_text = shlex.join(command)

        # The filter must live inside the shell command because it relies on
        # Bash process substitution to keep stderr visible while removing only
        # the noisy ``ptxas warning`` lines.
        if self.supress_tf_warnings:
            command_text = f"{command_text} 2> >(grep -v 'ptxas warning' >&2)"

        return command_text

    def launch(self, command: List[str], working_dir: str) -> subprocess.Popen:
        """Launch a command in a GUI terminal or inline shell session.

        On graphical Linux sessions, the command is launched inside
        ``gnome-terminal`` and the target PID is written to a temporary file so
        the restart manager can discover the child Python process. On headless
        Linux sessions, the command runs inline in the current shell and the
        returned ``Popen`` PID is already the target PID because the shell uses
        ``exec`` to replace itself with the monitored process.

        Args:
            command (List[str]): Sequence of command arguments to execute. The
                list is preserved exactly; quoting is applied only when the
                command must be embedded in a POSIX shell string.
            working_dir (str): Directory where ``command`` should run. The
                command executes relative to this path in both GUI and inline
                modes.

        Returns:
            subprocess.Popen: Process handle representing the launched shell or
            target process. The object includes ``launch_mode`` set to either
            ``"gui"`` or ``"inline"``. In GUI mode it also includes a
            ``pid_file`` attribute pointing to the temporary PID file used for
            target PID discovery. In inline mode ``pid_file`` is ``None``.

        Raises:
            OSError: If the current operating system is unsupported or the
                terminal process fails to launch.

        Examples:
            >>> launcher = SimpleTerminalLauncher()
            >>> process = launcher.launch(["python", "train.py"], "/tmp/project")
            >>> process.launch_mode in {"gui", "inline"}
            True
        """
        if self.system == "linux":
            command_text = self._build_posix_command(command)

            # ``exec`` is critical in inline mode because it makes the shell
            # PID become the Python process PID, which lets the restart manager
            # use ``process.pid`` directly without a temporary PID file.
            if self._has_linux_gui_terminal():
                fd, pid_file = tempfile.mkstemp(suffix=".pid")
                os.close(fd)
                full_cmd = (
                    f"({command_text}) & echo $! > {shlex.quote(pid_file)}; wait"
                )
                terminal_cmd = ["gnome-terminal", "--", "bash", "-lc", full_cmd]
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
                process.launch_mode = "gui"
                return process

            process = subprocess.Popen(
                ["bash", "-lc", f"exec {command_text}"],
                cwd=working_dir,
                start_new_session=False,
                stdin=None,
                stdout=None,
                stderr=None,
                close_fds=False,
            )
            process.pid_file = None
            process.launch_mode = "inline"
            return process

        if self.system == "darwin":
            command_text = self._build_posix_command(command)
            terminal_cmd = [
                "osascript",
                "-e",
                f'tell application "Terminal" to do script "{command_text}"',
            ]
            process = subprocess.Popen(
                terminal_cmd,
                cwd=working_dir,
                start_new_session=False,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=False,
            )
            process.pid_file = None
            process.launch_mode = "gui"
            return process

        if self.system == "windows":
            terminal_cmd = ["cmd", "/c", subprocess.list2cmdline(command)]
            process = subprocess.Popen(
                terminal_cmd,
                cwd=working_dir,
                start_new_session=False,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=False,
            )
            process.pid_file = None
            process.launch_mode = "gui"
            return process

        raise OSError(f"Unsupported OS: {self.system}")
