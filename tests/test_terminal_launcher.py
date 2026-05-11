"""Tests for monitor terminal launch selection."""

import sys
import unittest
import importlib.util
from pathlib import Path
from unittest import mock


def _ensure_src_on_path() -> None:
    """Ensure the local ``src`` directory is importable.

    Returns:
        None: The function mutates ``sys.path`` only when the repository
        ``src`` directory is not already present.

    Raises:
        RuntimeError: Not raised by this helper.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


class _FakeProcess:
    """Minimal process double that accepts monitor launcher attributes."""

    def __init__(self) -> None:
        """Create a process double with no predefined process metadata.

        Returns:
            None: The constructor only creates an object that can receive
            ``pid_file`` and ``launch_mode`` attributes from the launcher.

        Raises:
            RuntimeError: Not raised by this constructor.
        """


class TestSimpleTerminalLauncher(unittest.TestCase):
    """Validate Linux terminal selection for the monitor launcher."""

    def setUp(self) -> None:
        """Import the launcher after preparing the local source path.

        Returns:
            None: The method stores imported modules on the test instance.

        Raises:
            ImportError: If the local source tree cannot provide the launcher.
        """
        _ensure_src_on_path()
        terminal_path = Path(__file__).resolve().parents[1] / "src" / "araras" / "runtime" / "terminal.py"
        spec = importlib.util.spec_from_file_location("araras_runtime_terminal_test", terminal_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load terminal module from {terminal_path}")

        terminal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(terminal)
        self.terminal = terminal

    def test_linux_zellij_session_launches_in_zellij_pane(self) -> None:
        """Linux zellij sessions should launch targets in a new zellij pane.

        Returns:
            None: Assertions validate the subprocess command and process
            metadata generated for a zellij launch.

        Raises:
            AssertionError: If zellij is not preferred over GUI terminals.
        """
        process = _FakeProcess()

        with (
            mock.patch.dict(self.terminal.os.environ, {"ZELLIJ": "0", "DISPLAY": ":1"}, clear=True),
            mock.patch.object(self.terminal.shutil, "which", return_value="/usr/bin/zellij"),
            mock.patch.object(self.terminal.tempfile, "mkstemp", return_value=(9, "/tmp/araras-target.pid")),
            mock.patch.object(self.terminal.os, "close") as close_mock,
            mock.patch.object(self.terminal.subprocess, "Popen", return_value=process) as popen_mock,
            mock.patch.object(
                self.terminal.SimpleTerminalLauncher,
                "_has_linux_gui_terminal",
                side_effect=AssertionError("zellij should be checked before gnome-terminal"),
            ),
        ):
            launcher = self.terminal.SimpleTerminalLauncher()
            launcher.system = "linux"

            result = launcher.launch(["python", "-u", "train.py"], "/tmp/project")

        popen_mock.assert_called_once()
        command = popen_mock.call_args.args[0]
        self.assertEqual(command[:7], ["zellij", "run", "--cwd", "/tmp/project", "--name", "araras monitor", "--"])
        self.assertEqual(command[7:9], ["bash", "-lc"])
        self.assertEqual(command[9], "echo $$ > /tmp/araras-target.pid; exec python -u train.py")
        close_mock.assert_called_once_with(9)
        self.assertIs(result, process)
        self.assertEqual(process.pid_file, "/tmp/araras-target.pid")
        self.assertEqual(process.launch_mode, "zellij")

    def test_linux_zellij_launch_preserves_tensorflow_warning_filter(self) -> None:
        """Zellij launches should keep the existing TensorFlow warning filter.

        Returns:
            None: Assertions validate that the zellij shell command contains
            the same ``ptxas warning`` filter used by other POSIX launches.

        Raises:
            AssertionError: If warning suppression is dropped in zellij mode.
        """
        process = _FakeProcess()

        with (
            mock.patch.dict(self.terminal.os.environ, {"ZELLIJ": "0"}, clear=True),
            mock.patch.object(self.terminal.shutil, "which", return_value="/usr/bin/zellij"),
            mock.patch.object(self.terminal.tempfile, "mkstemp", return_value=(9, "/tmp/araras-target.pid")),
            mock.patch.object(self.terminal.os, "close"),
            mock.patch.object(self.terminal.subprocess, "Popen", return_value=process) as popen_mock,
        ):
            launcher = self.terminal.SimpleTerminalLauncher(supress_tf_warnings=True)
            launcher.system = "linux"

            launcher.launch(["python", "-u", "train.py"], "/tmp/project")

        command = popen_mock.call_args.args[0]
        self.assertIn("ptxas warning", command[9])

    def test_linux_gui_outside_zellij_keeps_gnome_terminal_launch(self) -> None:
        """Linux GUI sessions outside zellij should keep gnome-terminal behavior.

        Returns:
            None: Assertions validate the existing graphical terminal command.

        Raises:
            AssertionError: If non-zellij GUI launches stop using gnome-terminal.
        """
        process = _FakeProcess()

        with (
            mock.patch.dict(self.terminal.os.environ, {"DISPLAY": ":1"}, clear=True),
            mock.patch.object(
                self.terminal.shutil,
                "which",
                side_effect=lambda name: "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None,
            ),
            mock.patch.object(self.terminal.tempfile, "mkstemp", return_value=(9, "/tmp/araras-target.pid")),
            mock.patch.object(self.terminal.os, "close"),
            mock.patch.object(self.terminal.subprocess, "Popen", return_value=process) as popen_mock,
        ):
            launcher = self.terminal.SimpleTerminalLauncher()
            launcher.system = "linux"

            result = launcher.launch(["python", "-u", "train.py"], "/tmp/project")

        command = popen_mock.call_args.args[0]
        self.assertEqual(command[:3], ["gnome-terminal", "--", "bash"])
        self.assertIs(result, process)
        self.assertEqual(process.pid_file, "/tmp/araras-target.pid")
        self.assertEqual(process.launch_mode, "gui")

    def test_linux_headless_outside_zellij_keeps_inline_launch(self) -> None:
        """Linux headless sessions outside zellij should keep inline execution.

        Returns:
            None: Assertions validate the existing inline ``bash -lc exec``
            launch command and process metadata.

        Raises:
            AssertionError: If headless fallback behavior changes.
        """
        process = _FakeProcess()

        with (
            mock.patch.dict(self.terminal.os.environ, {}, clear=True),
            mock.patch.object(self.terminal.shutil, "which", return_value=None),
            mock.patch.object(self.terminal.subprocess, "Popen", return_value=process) as popen_mock,
        ):
            launcher = self.terminal.SimpleTerminalLauncher()
            launcher.system = "linux"

            result = launcher.launch(["python", "-u", "train.py"], "/tmp/project")

        popen_mock.assert_called_once_with(
            ["bash", "-lc", "exec python -u train.py"],
            cwd="/tmp/project",
            start_new_session=False,
            stdin=None,
            stdout=None,
            stderr=None,
            close_fds=False,
        )
        self.assertIs(result, process)
        self.assertIsNone(process.pid_file)
        self.assertEqual(process.launch_mode, "inline")
