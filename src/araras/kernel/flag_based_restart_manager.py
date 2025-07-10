from __future__ import annotations

import os
import psutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from araras.utils.cleanup import ChildProcessCleanup
from araras.utils.terminal import SimpleTerminalLauncher
from araras.utils.misc import NotebookConverter
from .consolidated_email_manager import ConsolidatedEmailManager
from .file_type_handler import FileTypeHandler
from . import monitoring as _mon
from .monitoring import (
    print_completion_summary,
    print_success_message,
)


class FlagBasedRestartManager:
    """Enhanced restart manager with consolidated email notifications and retry logic."""

    __slots__ = (
        "max_restarts",
        "restart_delay",
        "restart_count",
        "running",
        "current_terminal_process",
        "current_target_pid",
        "monitor_info",
        "email_manager",
        "process_title",
        "recipients_file",
        "credentials_file",
        "child_cleanup",
        "converted_python_file",
        "original_was_notebook",
        "start_time",
        "last_process_start_time",
        "_last_restart_file",
        "pid_history",
    )

    def __init__(
        self,
        max_restarts: int = 10,
        restart_delay: float = 3.0,
        recipients_file: Optional[str] = None,
        credentials_file: Optional[str] = None,
        retry_attempts: int = 2,
    ):
        """Initialize restart manager with consolidated email notification support.

        Args:
            max_restarts: Maximum restart attempts
            restart_delay: Delay between restarts in seconds
            recipients_file: Path to recipients JSON file
            credentials_file: Path to credentials JSON file
            retry_attempts: Number of retry attempts before sending failure email
        """
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.restart_count = 0
        self.running = False
        self.start_time = None
        self.last_process_start_time = None

        # Process tracking with minimal state
        self.current_terminal_process: Optional[subprocess.Popen] = None
        self.current_target_pid: Optional[int] = None
        self.monitor_info: Optional[Dict[str, Any]] = None
        self._last_restart_file: Optional[str] = None

        # File cleanup tracking
        self.converted_python_file: Optional[Path] = None
        self.original_was_notebook: bool = False

        # Email configuration with consolidated manager
        self.recipients_file = recipients_file or "./json/recipients.json"
        self.credentials_file = credentials_file or "./json/credentials.json"
        self.email_manager = ConsolidatedEmailManager(
            self.recipients_file, self.credentials_file, retry_attempts
        )
        self.process_title: str = ""

        # Child process cleanup manager
        self.child_cleanup = ChildProcessCleanup()

        # Track all target PIDs that have been launched. This allows us to
        # verify old instances are truly gone and to forcibly terminate them if
        # they linger after a forced restart.
        self.pid_history: List[int] = []

    def run_file_with_restart(
        self,
        file_path: str,
        success_flag_file: str,
        title: Optional[str] = None,
        restart_after_delay: Optional[float] = None,
        supress_tf_warnings: bool = False,
    ) -> None:
        """Run file with flag-based restart logic and consolidated email notifications.

        Args:
            file_path: Path to Python or Jupyter notebook file
            success_flag_file: Path where target process writes completion flag
            title: Custom title for monitoring
            restart_after_delay: Optional delay after which the run will be restarted
            supress_tf_warnings: Suppress TensorFlow warnings (default: False)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is unsupported
        """
        self.start_time = time.time()

        # Validate file with early exit pattern for performance
        validated_path = FileTypeHandler.validate_file(file_path)
        file_type = FileTypeHandler.get_file_type(validated_path)

        # Convert notebook to Python if needed
        if file_type == "notebook":
            # _mon.print_process_status(f"Converting notebook to Python: {validated_path.name}")
            try:
                self.converted_python_file = NotebookConverter.convert_notebook_to_python(validated_path)
                self.original_was_notebook = True
                validated_path = self.converted_python_file
                file_type = "python"
            except Exception as e:
                _mon.print_error_message("CONVERSION", f"Notebook conversion failed: {e}")
                raise

        working_dir = str(validated_path.parent)
        self.process_title = title or validated_path.stem
        flag_path = Path(success_flag_file).resolve()

        # Print configuration summary
        _mon.print_monitoring_config_summary(
            file_path=str(validated_path),
            file_type=file_type,
            success_flag_file=str(flag_path),
            max_restarts=self.max_restarts,
            email_enabled=self.email_manager.email_enabled,
            title=self.process_title,
            restart_after_delay=restart_after_delay,
        )

        self.running = True
        previous_pid = None

        try:
            # before launching a new run
            if self.current_target_pid and psutil.pid_exists(self.current_target_pid):
                raise RuntimeError("Previous target process still running, aborting duplicate start")

            # Clean up any lingering processes from earlier runs
            self._cleanup_stale_pids()

            # Main restart loop with consolidated email notifications
            while self.running and self.restart_count < self.max_restarts:
                # Ensure any leftover processes from previous iteration are gone
                self._cleanup_stale_pids()

                # Remove old success flag (atomic operation)
                if flag_path.exists():
                    flag_path.unlink()

                self.last_process_start_time = time.time()

                try:
                    if self.monitor_info:
                        _mon.stop_monitor(self.monitor_info)
                        self.monitor_info = None

                    # Launch process
                    target_pid = self._launch_process(validated_path, working_dir, success_flag_file)
                    _mon.print_process_status("Process started", target_pid)

                    # Send successful restart email (only for actual restarts, not first start)
                    if self.restart_count > 0:
                        runtime = time.time() - self.last_process_start_time
                        self.email_manager.report_successful_restart(
                            self.process_title,
                            previous_pid,
                            target_pid,
                            self.restart_count,
                            runtime,
                        )

                    # Start crash monitor with simplified monitoring
                    self.monitor_info = _mon.start_monitor(
                        target_pid,
                        self.process_title,
                        supress_tf_warnings=supress_tf_warnings,
                    )
                    self._last_restart_file = self.monitor_info["restart_file"]

                    # Wait for completion or crash with optimized polling
                    completion_reason = self._wait_for_completion(flag_path)
                    runtime = time.time() - self.last_process_start_time

                    _mon.print_process_status(f"Process finished: {completion_reason}", target_pid, runtime)

                    # Store PID for next restart notification
                    previous_pid = target_pid

                    # Immediate cleanup for memory efficiency
                    self._cleanup_all()

                    # Smart decision logic based on completion reason
                    if completion_reason == "success_flag":
                        print_success_message("Process completed successfully")
                        total_runtime = time.time() - self.start_time
                        self.email_manager.report_task_completion(
                            self.process_title, self.restart_count, total_runtime
                        )
                        break
                    elif completion_reason == "crashed":
                        _mon.print_process_status("Process crashed, checking restart policy")
                        if not self._handle_restart_with_retry():
                            break
                    elif completion_reason == "interrupted":
                        # User pressed CTRL+C, clean up and exit
                        _mon.print_process_status("Process interrupted by user")
                        break
                    elif completion_reason == "stopped":
                        # External request to stop without treating as failure
                        _mon.print_process_status("Process stopped by external request")
                        break
                    else:
                        _mon.print_process_status("Process ended without success flag, treating as failure")
                        if not self._handle_restart_with_retry():
                            break

                except Exception as e:
                    _mon.print_error_message("LAUNCH", str(e))
                    self._cleanup_all()
                    if not self._handle_restart_with_retry():
                        break

            # Handle maximum restarts reached
            if self.restart_count >= self.max_restarts:
                _mon.print_error_message("MAX_RESTARTS", f"Maximum restarts reached: {self.max_restarts}")
                self.email_manager.report_final_failure(
                    self.process_title,
                    self.restart_count,
                    f"Maximum restart attempts ({self.max_restarts}) reached",
                )

        except KeyboardInterrupt:
            _mon.print_process_status("Interrupted by user, cleaning up resources")
            self.running = False
        except Exception as e:
            _mon.print_error_message("FATAL", str(e))
            self.email_manager.report_final_failure(
                self.process_title, self.restart_count, f"Fatal error: {str(e)}"
            )
        finally:
            # Ensure all cleanup operations are performed
            # Explicitly mark the manager as no longer running.  This prevents
            # any background wait/sleep loops from continuing if the restart
            # loop wraps the call in a thread and forces a shutdown.
            self.running = False
            self._cleanup_all()
            self._cleanup_converted_file()
            total_runtime = time.time() - self.start_time if self.start_time else None
            print_completion_summary(self.restart_count, total_runtime)

    def _handle_restart_with_retry(self) -> bool:
        """Handle restart with retry logic and consolidated email notifications.

        Returns:
            True if should continue restart attempts, False if should stop
        """
        self.restart_count += 1

        # Check if should attempt restart using consolidated email manager
        if not self.email_manager.should_attempt_restart(
            self.process_title, self.restart_count, self.max_restarts
        ):
            return False

        if self.restart_count < self.max_restarts:
            # Protect current target process if still running
            exclude_pids = []
            if self.current_target_pid and psutil.pid_exists(self.current_target_pid):
                exclude_pids.append(self.current_target_pid)

            # Perform child process cleanup before restart
            try:
                terminated, killed = self.child_cleanup.cleanup_children(exclude_pids)
                _mon.print_cleanup_info(terminated, killed)
            except psutil.NoSuchProcess:
                _mon.print_warning_message("Current process not found during cleanup")
            except Exception as e:
                _mon.print_error_message("CLEANUP", f"Child cleanup failed (non-fatal): {e}")

            # Exponential backoff with cap at 30 seconds
            delay = min(self.restart_delay * (1.2 ** (self.restart_count - 1)), 30.0)
            _mon.print_restart_info(self.restart_count, self.max_restarts, delay)
            self._sleep(delay)
            return True

        return False

    def _launch_process(self, file_path: Path, working_dir: str, success_flag_file: str) -> int:
        """Launch target process.

        Args:
            file_path: Validated path to Python file
            working_dir: Working directory
            success_flag_file: Success flag file path

        Returns:
            Target process PID

        Raises:
            OSError: If PID discovery fails
        """
        # Build command for Python file
        command, execution_type = FileTypeHandler.build_execution_command(file_path, success_flag_file)

        launcher = SimpleTerminalLauncher()
        self.current_terminal_process = launcher.launch(command, working_dir)

        # Efficient PID discovery with timeout
        pid_file = self.current_terminal_process.pid_file
        target_pid = self._discover_target_pid(pid_file, timeout=5.0)

        if not target_pid:
            self._cleanup_terminal()
            raise OSError("Failed to get target process PID")

        self.current_target_pid = target_pid
        # Record the pid so we can later ensure it has terminated
        self.pid_history.append(target_pid)

        # Cleanup PID file immediately (no longer needed)
        try:
            os.unlink(pid_file)
        except:
            pass

        return target_pid

    def _discover_target_pid(self, pid_file: str, timeout: float) -> Optional[int]:
        """Discover target PID with optimized polling strategy.

        Args:
            pid_file: Path to PID file
            timeout: Discovery timeout in seconds

        Returns:
            Target PID if found, None otherwise
        """
        end_time = time.time() + timeout
        check_count = 0

        # Adaptive polling: start fast, slow down for efficiency
        while time.time() < end_time:
            check_count += 1

            try:
                if os.path.exists(pid_file):
                    with open(pid_file) as f:
                        pid_str = f.read().strip()
                        if pid_str.isdigit():
                            pid = int(pid_str)
                            if psutil.pid_exists(pid):
                                return pid
            except:
                pass

            # Progressive delay for efficiency optimization
            if check_count < 10:
                time.sleep(0.05)  # Fast initial checks
            elif check_count < 30:
                time.sleep(0.1)  # Medium frequency
            else:
                time.sleep(0.2)  # Stable frequency

        return None

    def _wait_for_completion(self, flag_path: Path) -> str:
        """Wait for process completion with optimized polling strategy.

        Args:
            flag_path: Path to success flag file

        Returns:
            Completion reason string
        """
        check_count = 0

        while self.running:
            check_count += 1

            # Check for keyboard interrupt (CTRL+C)
            try:
                # Check for success flag (highest priority, O(1) operation)
                if flag_path.exists():
                    return "success_flag"

                # Check crash signal every other iteration to reduce I/O
                if check_count % 2 == 0 and self.monitor_info:
                    crash_info = _mon.check_crash_signal(self.monitor_info)
                    if crash_info:
                        return "crashed"

                # Check process existence every 4th iteration for efficiency
                if check_count % 4 == 0 and self.current_target_pid:
                    if not psutil.pid_exists(self.current_target_pid):
                        return "process_died"

                time.sleep(0.5)
            except KeyboardInterrupt:
                # Handle CTRL+C by cleaning up the current monitored process
                _mon.print_process_status("CTRL+C detected, shutting down monitored process")
                self.running = False
                self._cleanup_all()
                return "interrupted"

        return "stopped"

    def _cleanup_all(self) -> None:
        """Cleanup all resources with optimized order for reliability."""
        # Stop monitor first (most critical for clean shutdown)
        if self.monitor_info:
            _mon.stop_monitor(self.monitor_info)
            self.monitor_info = None

        # now delete its restart_file if it still exists
        if self._last_restart_file and os.path.exists(self._last_restart_file):
            try:
                os.unlink(self._last_restart_file)
            except OSError:
                pass
        self._last_restart_file = None

        time.sleep(0.1)

        # Terminate target process
        if self.current_target_pid:
            try:
                proc = psutil.Process(self.current_target_pid)
                proc.terminate()
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait()
            except psutil.NoSuchProcess:
                pass
            finally:
                self.current_target_pid = None

        time.sleep(0.1)

        # Cleanup terminal last
        self._cleanup_terminal()

        time.sleep(0.1)

        # Ensure any historical PIDs are truly dead
        self._cleanup_stale_pids()

    def force_stop(self) -> None:
        """Request the currently running loop to stop and cleanup."""
        self.running = False
        self._cleanup_all()

    def _cleanup_terminal(self) -> None:
        """Cleanup terminal process with minimal overhead."""
        if self.current_terminal_process:
            try:
                self.current_terminal_process.terminate()
                self.current_terminal_process.wait(timeout=2)
            except:
                pass

            # Cleanup PID file if exists
            try:
                if hasattr(self.current_terminal_process, "pid_file"):
                    pid_file = self.current_terminal_process.pid_file
                    if os.path.exists(pid_file):
                        os.unlink(pid_file)
            except:
                pass

            self.current_terminal_process = None

    # ------------------------------------------------------------------
    # PID tracking helpers
    # ------------------------------------------------------------------
    def _kill_pid(self, pid: int) -> None:
        """Terminate a specific PID if it is still running."""
        try:
            proc = psutil.Process(pid)
            if proc.is_running():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            _mon.print_warning_message(f"Failed to kill pid {pid}: {e}")

    def _cleanup_stale_pids(self) -> None:
        """Ensure any previously launched target PIDs are fully terminated."""
        stale_pids = []
        for pid in list(self.pid_history):
            if pid == self.current_target_pid:
                continue
            if psutil.pid_exists(pid):
                _mon.print_process_status("Cleaning up stale process", pid)
                self._kill_pid(pid)
            if not psutil.pid_exists(pid):
                stale_pids.append(pid)

        # Remove PIDs that are confirmed dead from history
        for pid in stale_pids:
            if pid in self.pid_history:
                self.pid_history.remove(pid)

    def _cleanup_converted_file(self) -> None:
        """Delete converted Python file if original was a notebook.

        Only deletes the file if it was converted from a notebook during this session.
        Direct .py files are never deleted.
        """
        if self.original_was_notebook and self.converted_python_file:
            try:
                if self.converted_python_file.exists():
                    self.converted_python_file.unlink()
                    _mon.print_process_status(f"Cleaned up converted file: {self.converted_python_file}")
            except Exception as e:
                _mon.print_warning_message(
                    f"Failed to cleanup converted file {self.converted_python_file}: {e}"
                )
            finally:
                self.converted_python_file = None
                self.original_was_notebook = False

    def _sleep(self, duration: float) -> None:
        """Interruptible sleep with minimal CPU usage.

        Args:
            duration: Sleep duration in seconds
        """
        end_time = time.time() + duration
        while self.running and time.time() < end_time:
            try:
                time.sleep(min(0.1, end_time - time.time()))
            except KeyboardInterrupt:
                # Handle CTRL+C during sleep
                self.running = False
                _mon.print_process_status("CTRL+C detected during restart delay, aborting restart")
                break
