from typing import Optional, List, Dict, Any

import math
import os
import psutil
import subprocess
import time
import traceback
from pathlib import Path

from .cleanup import ChildProcessCleanup
from .terminal import SimpleTerminalLauncher
from araras.utils.misc import NotebookConverter
from .email_manager import ConsolidatedEmailManager
from .file_handler import FileTypeHandler
from .memory_leak import (
    MemoryLeakEvidence,
    ProcessMemoryLeakDetector,
    _get_process_tree_rss_bytes,
    render_memory_leak_graph,
    render_scheduled_restart_graph,
)
from .reporting import MonitorReportSession
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
        "scheduled_restart_count",
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
        "monitored_file",
        "original_was_notebook",
        "start_time",
        "last_process_start_time",
        "_last_restart_file",
        "pid_history",
        "report_logs",
        "report_session",
        "scheduled_restart_memory_threshold_bytes",
        "scheduled_restart_poll_interval_seconds",
        "detect_memory_leaks",
        "memory_leak_warmup_seconds",
        "memory_leak_warning_attempted",
        "memory_leak_detector",
        "last_failure_error",
        "pending_restart_old_pid",
        "pending_restart_runtime",
        "pending_restart_error",
        "pending_scheduled_restart_old_pid",
        "pending_scheduled_restart_runtime",
        "pending_scheduled_restart_graph_png",
        "pending_scheduled_restart_current_rss_bytes",
        "last_completed_pid",
        "last_completed_runtime",
    )

    def __init__(
        self,
        max_restarts: int = 10,
        restart_delay: float = 3.0,
        recipients_file: Optional[str] = None,
        credentials_file: Optional[str] = None,
        retry_attempts: int = 2,
        restart_email_warning: bool = True,
        report_logs: bool = False,
        scheduled_restart_memory_threshold_bytes: Optional[float] = None,
        scheduled_restart_poll_interval_seconds: Optional[float] = None,
        detect_memory_leaks: bool = False,
        memory_leak_warmup_seconds: float = 300.0,
    ) -> None:
        """Initialize restart manager with consolidated email notification support.

        Args:
            max_restarts (int): Maximum restart attempts
            restart_delay (float): Delay between restarts in seconds
            recipients_file (Optional[str]): Path to recipients JSON file
            credentials_file (Optional[str]): Path to credentials JSON file
            retry_attempts (int): Number of retry attempts before sending failure email
            restart_email_warning (bool): Enable restart success and failure email messages
            report_logs (bool): When ``True``, write monitor report artifacts
                under ``runs/monitor_logs`` for each monitored source file. If
                ``False``, the runtime keeps the previous behavior and does not
                create report files.
            scheduled_restart_memory_threshold_bytes (Optional[float]):
                Positive target-process-tree RSS threshold in bytes. When set
                with ``scheduled_restart_poll_interval_seconds``, RSS is
                checked at the polling interval and restarts when it reaches
                this value. ``None`` preserves fixed scheduled restarts.
            scheduled_restart_poll_interval_seconds (Optional[float]):
                Positive delay in seconds between smart-restart RSS checks. It
                must be set together with
                ``scheduled_restart_memory_threshold_bytes``.
            detect_memory_leaks (bool): When ``True``, sample process-tree RSS
                and warn once when the conservative growth rule qualifies.
                Defaults to ``False`` and does not affect report logging.
            memory_leak_warmup_seconds (float): Positive finite no-check period
                in seconds applied separately to each launched PID. Defaults
                to ``300.0``.

        Returns:
            None: The constructor initializes manager state without launching
            a target process.

        Raises:
            ValueError: If ``memory_leak_warmup_seconds`` is not positive and
                finite, or smart scheduled-restart values are incomplete or
                not positive and finite.
        """
        if not math.isfinite(memory_leak_warmup_seconds) or memory_leak_warmup_seconds <= 0:
            raise ValueError("memory_leak_warmup_seconds must be a positive finite number")
        if (
            scheduled_restart_memory_threshold_bytes is None
        ) != (
            scheduled_restart_poll_interval_seconds is None
        ):
            raise ValueError(
                "scheduled restart memory threshold and polling interval must be provided together"
            )
        if scheduled_restart_memory_threshold_bytes is not None and (
            not math.isfinite(scheduled_restart_memory_threshold_bytes)
            or scheduled_restart_memory_threshold_bytes <= 0
        ):
            raise ValueError(
                "scheduled_restart_memory_threshold_bytes must be a positive finite number"
            )
        if scheduled_restart_poll_interval_seconds is not None and (
            not math.isfinite(scheduled_restart_poll_interval_seconds)
            or scheduled_restart_poll_interval_seconds <= 0
        ):
            raise ValueError(
                "scheduled_restart_poll_interval_seconds must be a positive finite number"
            )

        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.restart_count = 0
        self.scheduled_restart_count = 0
        self.running = False
        self.start_time = None
        self.last_process_start_time = None
        self.report_logs = report_logs
        self.report_session: Optional[MonitorReportSession] = None
        self.scheduled_restart_memory_threshold_bytes = (
            scheduled_restart_memory_threshold_bytes
        )
        self.scheduled_restart_poll_interval_seconds = (
            scheduled_restart_poll_interval_seconds
        )
        self.detect_memory_leaks = detect_memory_leaks
        self.memory_leak_warmup_seconds = memory_leak_warmup_seconds
        self.memory_leak_warning_attempted = False
        self.memory_leak_detector: Optional[ProcessMemoryLeakDetector] = None
        self.last_failure_error: Optional[str] = None
        self.pending_restart_old_pid: Optional[int] = None
        self.pending_restart_runtime: Optional[float] = None
        self.pending_restart_error: Optional[str] = None
        self.pending_scheduled_restart_old_pid: Optional[int] = None
        self.pending_scheduled_restart_runtime: Optional[float] = None
        self.pending_scheduled_restart_graph_png: Optional[bytes] = None
        self.pending_scheduled_restart_current_rss_bytes: Optional[float] = None
        self.last_completed_pid: Optional[int] = None
        self.last_completed_runtime: Optional[float] = None

        # Process tracking with minimal state
        self.current_terminal_process: Optional[subprocess.Popen] = None
        self.current_target_pid: Optional[int] = None
        self.monitor_info: Optional[Dict[str, Any]] = None
        self._last_restart_file: Optional[str] = None

        # File cleanup tracking
        self.converted_python_file: Optional[Path] = None
        self.monitored_file: Optional[Path] = None
        self.original_was_notebook: bool = False

        # Email configuration with consolidated manager
        self.recipients_file = recipients_file or "./json/recipients.json"
        self.credentials_file = credentials_file or "./json/credentials.json"
        self.email_manager = ConsolidatedEmailManager(
            self.recipients_file,
            self.credentials_file,
            retry_attempts,
            restart_email_warning,
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
        """Execute a Python file with automatic restart logic.

        The target file is copied or converted into a monitored version and
        executed. Restarts occur automatically when the process crashes or
        fails to create the expected success flag until the maximum restart
        count is reached. Temporary files created during monitoring are removed
        after execution completes.

        Args:
            file_path (str): Path to the Python script or Jupyter notebook to run.
            success_flag_file (str): Path where the monitored file writes the
                ``SUCCESS`` flag upon completion.
            title (Optional[str]): Optional custom title displayed in status messages. If not
                provided, the original file name is used for display purposes.
            restart_after_delay (Optional[float]): Optional delay in seconds that forces a restart
                even if the process is still running. The value must be finite
                and strictly positive. Scheduled restarts do not consume or
                reset the genuine-crash restart budget. ``None`` disables the
                fixed restart policy; it does not disable an independently
                configured smart scheduled-restart policy.
            supress_tf_warnings (bool): When ``True``, TensorFlow warnings emitted by
                the target script are filtered out in the terminal.

        Notes:
            Only the information printed to the user changes; the monitored
            process itself continues to run using the converted or copied file.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            ValueError: If ``file_path`` has an unsupported extension.
            ValueError: If ``restart_after_delay`` is not finite and strictly
                positive when provided.
        """
        if restart_after_delay is not None and (
            not math.isfinite(restart_after_delay) or restart_after_delay <= 0
        ):
            raise ValueError("restart_after_delay must be a positive finite number of seconds")
        self.start_time = time.time()
        self.scheduled_restart_count = 0
        self.memory_leak_warning_attempted = False
        self._stop_memory_leak_detector()
        self.last_failure_error = None
        self.pending_restart_old_pid = None
        self.pending_restart_runtime = None
        self.pending_restart_error = None
        self.pending_scheduled_restart_old_pid = None
        self.pending_scheduled_restart_runtime = None
        self.pending_scheduled_restart_graph_png = None
        self.pending_scheduled_restart_current_rss_bytes = None
        self.last_completed_pid = None
        self.last_completed_runtime = None

        # Validate file with early exit pattern for performance
        validated_path = FileTypeHandler.validate_file(file_path)
        file_type = FileTypeHandler.get_file_type(validated_path)

        original_path = validated_path

        if file_type == "notebook":
            try:
                self.monitored_file = NotebookConverter.convert_notebook_to_monitored_python(
                    validated_path, success_flag_file
                )
                self.original_was_notebook = True
                validated_path = self.monitored_file
                file_type = "python"
            except Exception as e:
                _mon.print_error_message("CONVERSION", f"Notebook conversion failed: {e}")
                raise
        else:
            self.monitored_file = self._create_monitored_copy(validated_path, success_flag_file)
            validated_path = self.monitored_file

        working_dir = str(validated_path.parent)
        self.process_title = title or original_path.stem
        flag_path = Path(success_flag_file).resolve()

        display_title = title or original_path.stem

        if self.report_logs:
            self.report_session = MonitorReportSession(original_path, display_title)

        # Print configuration summary
        _mon.print_monitoring_config_summary(
            file_path=str(original_path),
            file_type=file_type,
            success_flag_file=str(flag_path),
            max_restarts=self.max_restarts,
            email_enabled=self.email_manager.email_enabled,
            title=display_title,
            restart_after_delay=restart_after_delay,
            scheduled_restart_memory_threshold_bytes=(
                self.scheduled_restart_memory_threshold_bytes
            ),
            scheduled_restart_poll_interval_seconds=(
                self.scheduled_restart_poll_interval_seconds
            ),
            detect_memory_leaks=self.detect_memory_leaks,
            memory_leak_warmup_seconds=self.memory_leak_warmup_seconds,
        )

        self.running = True
        previous_pid = None
        final_status = "failed"

        try:
            # before launching a new run
            if self.current_target_pid and psutil.pid_exists(self.current_target_pid):
                raise RuntimeError("Previous target process still running, aborting duplicate start")

            # Clean up any lingering processes from earlier runs
            self._cleanup_stale_pids()

            # Main restart loop with consolidated email notifications
            while self.running and self._should_start_attempt():
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
                    target_pid = self._launch_process(
                        validated_path,
                        working_dir,
                        success_flag_file,
                        supress_tf_warnings=supress_tf_warnings,
                    )
                    if self.report_session:
                        self.report_session.start_attempt(target_pid)
                    _mon.print_process_status("\033[92mProcess started\033[0m", target_pid)

                    # Send successful restart email (only for actual restarts, not first start)
                    if self.restart_count > 0 and self.pending_restart_old_pid is not None:
                        runtime = 0.0 if self.pending_restart_runtime is None else self.pending_restart_runtime
                        self.email_manager.report_successful_restart(
                            self.process_title,
                            self.pending_restart_old_pid,
                            target_pid,
                            self.restart_count,
                            runtime,
                        )
                        if self.report_session:
                            self.report_session.record_restart(
                                old_pid=self.pending_restart_old_pid,
                                new_pid=target_pid,
                                restart_count=self.restart_count,
                                runtime_seconds=self.pending_restart_runtime,
                                restart_type="crash_recovery",
                                scheduled_restart_count=self.scheduled_restart_count,
                                error=self.pending_restart_error,
                            )
                        self.pending_restart_old_pid = None
                        self.pending_restart_runtime = None
                        self.pending_restart_error = None

                    # Start crash monitor with simplified monitoring
                    self.monitor_info = _mon.start_monitor(
                        target_pid,
                        self.process_title,
                        supress_tf_warnings=supress_tf_warnings,
                    )
                    self._last_restart_file = self.monitor_info["restart_file"]

                    # A scheduled restart is confirmed only after the new PID
                    # is both running and covered by the crash monitor.
                    if self.pending_scheduled_restart_old_pid is not None:
                        if self.pending_scheduled_restart_runtime is None:
                            raise RuntimeError(
                                "Scheduled restart runtime is missing for the replacement PID"
                            )
                        scheduled_restart_interval = (
                            restart_after_delay
                            if restart_after_delay is not None
                            else self.scheduled_restart_poll_interval_seconds
                        )
                        if scheduled_restart_interval is None:
                            raise RuntimeError(
                                "Scheduled restart interval is missing for the replacement PID"
                            )
                        self.scheduled_restart_count += 1
                        if self.pending_scheduled_restart_current_rss_bytes is not None:
                            if self.scheduled_restart_memory_threshold_bytes is None:
                                raise RuntimeError(
                                    "memory-aware scheduled restart threshold is missing"
                                )
                            if self.scheduled_restart_poll_interval_seconds is None:
                                raise RuntimeError(
                                    "memory-aware scheduled restart polling interval is missing"
                                )
                            self.email_manager.report_successful_memory_aware_scheduled_restart(
                                self.process_title,
                                self.pending_scheduled_restart_old_pid,
                                target_pid,
                                self.scheduled_restart_count,
                                self.pending_scheduled_restart_runtime,
                                self.pending_scheduled_restart_current_rss_bytes,
                                self.scheduled_restart_memory_threshold_bytes,
                                self.scheduled_restart_poll_interval_seconds,
                                self.pending_scheduled_restart_graph_png,
                            )
                        elif self.pending_scheduled_restart_graph_png is not None:
                            self.email_manager.report_successful_scheduled_restart(
                                self.process_title,
                                self.pending_scheduled_restart_old_pid,
                                target_pid,
                                self.scheduled_restart_count,
                                self.pending_scheduled_restart_runtime,
                                scheduled_restart_interval,
                                self.pending_scheduled_restart_graph_png,
                            )
                        if self.report_session:
                            self.report_session.record_restart(
                                old_pid=self.pending_scheduled_restart_old_pid,
                                new_pid=target_pid,
                                restart_count=self.restart_count,
                                runtime_seconds=self.pending_scheduled_restart_runtime,
                                restart_type="scheduled",
                                scheduled_restart_count=self.scheduled_restart_count,
                            )
                        _mon.print_process_status(
                            f"Scheduled restart successful ({self.scheduled_restart_count})",
                            target_pid,
                        )
                        self.pending_scheduled_restart_old_pid = None
                        self.pending_scheduled_restart_runtime = None
                        self.pending_scheduled_restart_graph_png = None
                        self.pending_scheduled_restart_current_rss_bytes = None

                    self._start_memory_leak_detector(
                        target_pid,
                        collect_for_scheduled_restart=(
                            (
                                restart_after_delay is not None
                                or self.scheduled_restart_memory_threshold_bytes is not None
                            )
                            and self.email_manager.email_enabled
                        ),
                    )

                    # Wait for completion or crash with optimized polling
                    completion_reason = self._wait_for_completion(
                        flag_path,
                        restart_after_delay=restart_after_delay,
                    )
                    runtime = time.time() - self.last_process_start_time

                    _mon.print_process_status(f"Process finished: {completion_reason}", target_pid, runtime)
                    if self.report_session:
                        self.report_session.stop_attempt()
                        self.report_session.record_attempt_end(completion_reason, target_pid, runtime)
                    self.last_completed_pid = target_pid
                    self.last_completed_runtime = runtime

                    # Store PID for next restart notification
                    previous_pid = target_pid

                    scheduled_restart_graph_png = None
                    if (
                        completion_reason == "scheduled_restart"
                        and self.email_manager.email_enabled
                    ):
                        scheduled_restart_graph_png = (
                            self._capture_scheduled_restart_graph()
                        )

                    # Immediate cleanup for memory efficiency
                    self._cleanup_all()

                    # Smart decision logic based on completion reason
                    if completion_reason == "success_flag":
                        print_success_message("Process completed successfully")
                        total_runtime = time.time() - self.start_time
                        self.email_manager.report_task_completion(
                            self.process_title,
                            self.restart_count,
                            total_runtime,
                            scheduled_restart_count=self.scheduled_restart_count,
                        )
                        final_status = "success"
                        break
                    elif completion_reason == "scheduled_restart":
                        _mon.print_process_status(
                            "Scheduled restart condition met; restarting without recording a crash"
                        )
                        self.pending_scheduled_restart_old_pid = target_pid
                        self.pending_scheduled_restart_runtime = runtime
                        self.pending_scheduled_restart_graph_png = (
                            scheduled_restart_graph_png
                        )
                    elif completion_reason == "crashed":
                        _mon.print_process_status("Process crashed, checking restart policy")
                        self.last_failure_error = "Process crashed"
                        final_status = "failed"
                        if not self._handle_restart_with_retry():
                            break
                    elif completion_reason == "interrupted":
                        # User pressed CTRL+C, clean up and exit
                        _mon.print_process_status("Process interrupted by user")
                        final_status = "interrupted"
                        break
                    elif completion_reason == "stopped":
                        # External request to stop without treating as failure
                        _mon.print_process_status("Process stopped by external request")
                        final_status = "stopped"
                        break
                    else:
                        _mon.print_process_status("Process ended without success flag, treating as failure")
                        self.last_failure_error = f"Process ended with reason: {completion_reason}"
                        final_status = "failed"
                        if not self._handle_restart_with_retry():
                            break

                except Exception as e:
                    _mon.print_error_message("LAUNCH", str(e))
                    traceback.print_exc()
                    self.last_failure_error = str(e)
                    # A replacement that did not become monitored is not a
                    # successful scheduled restart and must not be confirmed.
                    self.pending_scheduled_restart_old_pid = None
                    self.pending_scheduled_restart_runtime = None
                    self.pending_scheduled_restart_graph_png = None
                    self.pending_scheduled_restart_current_rss_bytes = None
                    if self.report_session:
                        self.report_session.stop_attempt()
                        self.report_session.record_attempt_end("launch_error", previous_pid, None, error=str(e))
                    self._cleanup_all()
                    if not self._handle_restart_with_retry():
                        break

            # Handle maximum restarts reached
            if final_status != "success" and self._has_exhausted_restarts():
                _mon.print_error_message("MAX_RESTARTS", f"Maximum restarts reached: {self.max_restarts}")
                self.email_manager.report_final_failure(
                    self.process_title,
                    self.restart_count,
                    f"Maximum restart attempts ({self.max_restarts}) reached",
                )
                self.last_failure_error = f"Maximum restart attempts ({self.max_restarts}) reached"

        except KeyboardInterrupt:
            _mon.print_process_status("Interrupted by user, cleaning up resources")
            self.running = False
            final_status = "interrupted"
        except Exception as e:
            _mon.print_error_message("FATAL", str(e))
            self.last_failure_error = str(e)
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
            self._cleanup_monitored_file()
            total_runtime = time.time() - self.start_time if self.start_time else None
            if self.report_session:
                self.report_session.finalize(
                    final_status=final_status,
                    total_runtime_seconds=total_runtime,
                    total_restarts=self.restart_count + self.scheduled_restart_count,
                    final_error=self.last_failure_error,
                )
            print_completion_summary(
                self.restart_count,
                total_runtime,
                scheduled_restart_count=self.scheduled_restart_count,
            )

    def _handle_restart_with_retry(self) -> bool:
        """Handle restart with retry logic and consolidated email notifications.

        Returns:
            bool: True if should continue restart attempts, False if should stop
        """
        previous_pid = self.last_completed_pid
        previous_runtime = self.last_completed_runtime
        if previous_runtime is None and self.last_process_start_time is not None:
            previous_runtime = time.time() - self.last_process_start_time
        self.restart_count += 1

        # Check if should attempt restart using consolidated email manager
        if not self.email_manager.should_attempt_restart(
            self.process_title, self.restart_count, self.max_restarts
        ):
            self.pending_restart_old_pid = None
            self.pending_restart_runtime = None
            self.pending_restart_error = None
            if self.report_session:
                self.report_session.record_restart(
                    old_pid=previous_pid,
                    new_pid=None,
                    restart_count=self.restart_count,
                    runtime_seconds=previous_runtime,
                    restart_type="crash_recovery",
                    scheduled_restart_count=self.scheduled_restart_count,
                    error=self.last_failure_error,
                )
            return False

        if self.restart_count <= self.max_restarts:
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
            self.pending_restart_old_pid = previous_pid
            self.pending_restart_runtime = previous_runtime
            self.pending_restart_error = self.last_failure_error
            self._sleep(delay)
            return True

        return False

    def _should_start_attempt(self) -> bool:
        """Return whether another launch attempt is allowed.

        Returns:
            bool: ``True`` for the initial launch regardless of
                ``max_restarts`` and for later attempts only while the number
                of already-consumed restarts does not exceed
                ``max_restarts``. A value of ``0`` therefore allows one launch
                attempt and no automatic retries.

        Raises:
            RuntimeError: Not raised by this helper.
        """
        if self.current_target_pid is None and self.restart_count == 0:
            return True
        return self.restart_count <= self.max_restarts

    def _has_exhausted_restarts(self) -> bool:
        """Return whether the configured restart budget has been exhausted.

        Returns:
            bool: ``True`` when at least one restart was attempted and the
                count reached or exceeded ``max_restarts``. ``False`` for the
                initial launch with ``max_restarts=0`` so the monitor does not
                incorrectly report "maximum restarts reached" before any retry
                happened.

        Raises:
            RuntimeError: Not raised by this helper.
        """
        return self.restart_count > 0 and self.restart_count >= self.max_restarts

    def _launch_process(
        self,
        file_path: Path,
        working_dir: str,
        success_flag_file: str,
        supress_tf_warnings: bool = False,
    ) -> int:
        """Launch the monitored process in a terminal or inline shell session.

        Args:
            file_path (Path): Validated path to the Python file.
            working_dir (str): Directory where the command should be executed.
            success_flag_file (str): Path to the success flag written by the process.
            supress_tf_warnings (bool): When ``True``, filter out TensorFlow warnings
                printed to the terminal. If ``False``, the launched process keeps
                its original stderr output.

        Returns:
            int: The PID of the launched process.

        Raises:
            OSError: If the process fails to start or the PID cannot be
                discovered.
        """
        # Build command for Python file
        command, execution_type = FileTypeHandler.build_execution_command(file_path, success_flag_file)

        launcher = SimpleTerminalLauncher(supress_tf_warnings=supress_tf_warnings)
        self.current_terminal_process = launcher.launch(command, working_dir)

        pid_file = getattr(self.current_terminal_process, "pid_file", None)

        # GUI launches still need the temporary PID file because the terminal
        # process is different from the Python child. Inline launches use
        # ``bash -lc 'exec ...'`` so the returned process PID is already the
        # monitored target PID.
        if pid_file:
            target_pid = self._discover_target_pid(pid_file, timeout=5.0)
        else:
            target_pid = self.current_terminal_process.pid if self.current_terminal_process.poll() is None else None

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
            pid_file (str): Path to PID file
            timeout (float): Discovery timeout in seconds

        Returns:
            Optional[int]: Target PID if found, None otherwise
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

    def _wait_for_completion(
        self,
        flag_path: Path,
        restart_after_delay: Optional[float] = None,
    ) -> str:
        """Wait for process completion, failure, or a scheduled restart.

        Args:
            flag_path (Path): Path to the success flag file. Its presence takes
                priority over crash detection and the scheduled deadline.
            restart_after_delay (Optional[float]): Positive fixed-restart
                interval in seconds. ``None`` disables only the fixed policy.
                When smart restart settings are configured, their polling
                interval schedules the RSS checks instead.

        Returns:
            str: One of ``"success_flag"``, ``"crashed"``,
            ``"process_died"``, ``"scheduled_restart"``, ``"interrupted"``,
            or ``"stopped"``.

        Raises:
            RuntimeError: Not raised by this helper.
        """
        check_count = 0
        if restart_after_delay is not None:
            restart_deadline = time.monotonic() + restart_after_delay
        elif self.scheduled_restart_poll_interval_seconds is not None:
            restart_deadline = time.monotonic() + self.scheduled_restart_poll_interval_seconds
        else:
            restart_deadline = None

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
                    _mon.print_process_resource_usage(self.current_target_pid)

                if restart_deadline is not None:
                    remaining = restart_deadline - time.monotonic()
                    if remaining <= 0:
                        # Recheck failure signals at the boundary so a genuine
                        # crash is never mislabeled as an intentional restart.
                        if self.monitor_info and _mon.check_crash_signal(self.monitor_info):
                            return "crashed"
                        if self.current_target_pid and not psutil.pid_exists(
                            self.current_target_pid
                        ):
                            return "process_died"
                        if self.scheduled_restart_memory_threshold_bytes is not None:
                            if self.current_target_pid is None:
                                return "process_died"
                            try:
                                current_rss_bytes = _get_process_tree_rss_bytes(
                                    self.current_target_pid
                                )
                            except psutil.NoSuchProcess:
                                return "process_died"
                            if current_rss_bytes < self.scheduled_restart_memory_threshold_bytes:
                                poll_interval_seconds = (
                                    self.scheduled_restart_poll_interval_seconds
                                )
                                if poll_interval_seconds is None:
                                    raise RuntimeError(
                                        "smart scheduled restart polling interval is missing"
                                    )
                                current_rss_gib = current_rss_bytes / float(1024**3)
                                threshold_gib = (
                                    self.scheduled_restart_memory_threshold_bytes
                                    / float(1024**3)
                                )
                                _mon.print_process_status(
                                    "Scheduled restart deferred: process-tree RSS "
                                    f"{current_rss_gib:.2f} GiB is below threshold "
                                    f"{threshold_gib:.2f} GiB; rechecking in "
                                    f"{poll_interval_seconds / 60.0:g} minutes",
                                    self.current_target_pid,
                                )
                                restart_deadline = (
                                    time.monotonic() + poll_interval_seconds
                                )
                                continue
                            self.pending_scheduled_restart_current_rss_bytes = current_rss_bytes
                        return "scheduled_restart"
                    time.sleep(min(0.5, remaining))
                else:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                # Handle CTRL+C by cleaning up the current monitored process
                _mon.print_process_status("CTRL+C detected, shutting down monitored process")
                self.running = False
                self._cleanup_all()
                return "interrupted"

        return "stopped"

    def _start_memory_leak_detector(
        self,
        pid: int,
        collect_for_scheduled_restart: bool = False,
    ) -> None:
        """Start RSS collection for leak checks or a scheduled-restart graph.

        Args:
            pid (int): Positive root PID whose process-tree RSS should be
                sampled.
            collect_for_scheduled_restart (bool): When ``True``, retain RSS
                history for the scheduled-restart email even if leak detection
                is disabled or already warned. Defaults to ``False``. This
                adds one process-tree RSS query per sampling interval.

        Returns:
            None: A sampler is started when either requested use is active.
            Leak analysis is disabled when its warning was already attempted.

        Raises:
            RuntimeError: If a detector is unexpectedly still active when a
                replacement PID starts.
            ValueError: If detector configuration or ``pid`` is invalid.
        """
        should_detect_leak = (
            self.detect_memory_leaks and not self.memory_leak_warning_attempted
        )
        if not should_detect_leak and not collect_for_scheduled_restart:
            return
        if self.memory_leak_detector is not None:
            raise RuntimeError("previous memory leak detector is still active")

        self.memory_leak_detector = ProcessMemoryLeakDetector(
            pid=pid,
            warmup_seconds=self.memory_leak_warmup_seconds,
            warning_callback=(
                self._handle_memory_leak_warning if should_detect_leak else None
            ),
        )
        self.memory_leak_detector.start()

    def _capture_scheduled_restart_graph(self) -> Optional[bytes]:
        """Stop RSS collection and render the stopped PID's usage graph.

        Returns:
            Optional[bytes]: PNG bytes for the successful scheduled-restart
            email, or ``None`` after an explicitly reported capture or render
            failure. A failed graph suppresses that email instead of sending a
            confirmation without the requested attachment.

        Raises:
            RuntimeError: Not raised. Missing samples and Matplotlib failures
                are reported through the monitor error output.
        """
        detector = self.memory_leak_detector
        if detector is None:
            _mon.print_error_message(
                "SCHEDULED_RESTART_GRAPH",
                "Process memory sampler is not active",
            )
            return None

        detector.stop()
        self.memory_leak_detector = None
        try:
            # The restart deadline can coincide with the detector's first
            # eligible window. Take one final live sample and bypass only the
            # minute cadence before the old PID is terminated.
            detector.evaluate_current_sample()
            return render_scheduled_restart_graph(detector.get_graph_points())
        except (
            ImportError,
            OSError,
            psutil.NoSuchProcess,
            RuntimeError,
            ValueError,
        ) as error:
            _mon.print_error_message("SCHEDULED_RESTART_GRAPH", str(error))
            return None

    def _stop_memory_leak_detector(self) -> None:
        """Stop and clear the detector for the current target PID.

        Returns:
            None: The active detector is stopped and its reference is cleared.

        Raises:
            RuntimeError: Not raised when no detector is active.
        """
        if self.memory_leak_detector is None:
            return
        self.memory_leak_detector.stop()
        self.memory_leak_detector = None

    def _handle_memory_leak_warning(self, evidence: MemoryLeakEvidence) -> None:
        """Print and attempt the single possible-leak warning for this target.

        Args:
            evidence (MemoryLeakEvidence): Qualified RSS trend evidence from
                the active target detector.

        Returns:
            None: Terminal output is emitted and, when configured, one email
            attempt is made with an inline graph.

        Raises:
            RuntimeError: Not raised. Graph-generation errors are reported
                explicitly and suppress the required graph email.
        """
        if self.memory_leak_warning_attempted:
            return
        self.memory_leak_warning_attempted = True
        _mon.print_warning_message(
            "Possible memory leak detected for "
            f"{self.process_title} (PID {evidence.pid}): "
            f"RSS grew {evidence.net_growth_mib:.1f} MiB at "
            f"{evidence.slope_mib_per_minute:.1f} MiB/min "
            f"(R-squared {evidence.r_squared:.3f})"
        )

        if not self.email_manager.email_enabled:
            _mon.print_warning_message(
                "Memory leak warning email was not sent because email alerts are disabled"
            )
            return

        try:
            graph_png = render_memory_leak_graph(evidence)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            _mon.print_error_message("MEMORY_LEAK_GRAPH", str(error))
            return

        self.email_manager.report_possible_memory_leak(
            title=self.process_title,
            pid=evidence.pid,
            current_rss_mib=evidence.current_rss_mib,
            net_growth_mib=evidence.net_growth_mib,
            slope_mib_per_minute=evidence.slope_mib_per_minute,
            r_squared=evidence.r_squared,
            warmup_seconds=evidence.warmup_seconds,
            window_seconds=evidence.window_seconds,
            graph_png=graph_png,
        )

    def _cleanup_all(self) -> None:
        """Cleanup all resources with optimized order for reliability."""
        self._stop_memory_leak_detector()

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

    def _create_monitored_copy(self, file_path: Path, success_flag: str) -> Path:
        """Create a temporary monitored copy of a Python file.

        The copy is placed alongside ``file_path`` with the name
        ``temp_monitor_<original>.py`` and includes a small code snippet that
        writes the provided success flag when execution completes.

        Args:
            file_path (Path): Source Python file to be monitored.
            success_flag (str): Location where the success flag should be written.

        Returns:
            Path: Path to the generated monitored file.

        Raises:
            OSError: If the file cannot be created or written.
        """
        monitored_name = f"temp_monitor_{file_path.name}"
        monitored_path = file_path.parent / monitored_name

        try:
            content = file_path.read_text()
            success_flag_path = Path(success_flag).resolve()
            append_lines = (
                "\n\n# Write success flag for the auto restart script\n"
                "from pathlib import Path\n"
                f"Path({repr(str(success_flag_path))}).write_text('SUCCESS')\n"
            )
            monitored_path.write_text(content + append_lines)
        except Exception as e:
            raise OSError(f"Failed to create monitored file {monitored_path}: {e}")

        return monitored_path

    def _cleanup_monitored_file(self) -> None:
        """Remove the temporary monitored script.

        This method is safe to call multiple times. Any errors encountered
        during deletion are reported but do not raise exceptions.
        """
        if self.monitored_file and self.monitored_file.exists():
            try:
                self.monitored_file.unlink()
                _mon.print_process_status(f"Cleaned up monitored file: {self.monitored_file}")
            except Exception as e:
                _mon.print_warning_message(f"Failed to cleanup monitored file {self.monitored_file}: {e}")
            finally:
                self.monitored_file = None

    def _sleep(self, duration: float) -> None:
        """Interruptible sleep with minimal CPU usage.

        Args:
            duration (float): Sleep duration in seconds
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
