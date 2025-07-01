"""
This module provides a restarting monitoring system for processes with email alert capabilities.
It monitors a process for crashes and restarts it if necessary.

Usage example:
    run_auto_restart(
        file_path="my_script.py",
        title="My Critical Process",
    )
"""

import os
import sys
import time
import json
import glob
import psutil
import tempfile
import subprocess
from typing import *
from pathlib import Path

from threading import Event, Thread

# Local imports
from araras.email.utils import send_email
from araras.utils.cleanup import ChildProcessCleanup
from araras.utils.terminal import SimpleTerminalLauncher
from araras.utils.misc import NotebookConverter, clear


# Enhanced HTML template for consolidated status reports
CONSOLIDATED_STATUS_TEMPLATE = """<html><body style="font-family:Arial,sans-serif;color:#333;padding:20px"><div style="max-width:600px;margin:auto;background:#fff;padding:20px;border:1px solid #ddd"><h2 style="color:{color}">{status_title}</h2><div style="background:#f9f9f9;padding:15px;margin:15px 0;border-left:4px solid {color}"><h3>Process Information</h3><p><strong>Process:</strong> {title}</p><p><strong>Status:</strong> {status_description}</p><p><strong>Timestamp:</strong> {timestamp}</p></div>{details_section}<div style="background:#f0f0f0;padding:10px;margin-top:20px;font-size:12px;color:#666"><p>This is an automated status report from the process monitoring system.</p></div></div></body></html>"""

RESTART_DETAILS_TEMPLATE = """<div style="background:#fff3cd;padding:15px;margin:15px 0;border-left:4px solid #ffc107"><h3>Restart Information</h3><p><strong>Previous PID:</strong> {old_pid}</p><p><strong>New PID:</strong> {new_pid}</p><p><strong>Total Restarts:</strong> {restart_count}</p><p><strong>Runtime Before Restart:</strong> {runtime:.1f}s</p></div>"""

FAILURE_DETAILS_TEMPLATE = """<div style="background:#f8d7da;padding:15px;margin:15px 0;border-left:4px solid #dc3545"><h3>Failure Details</h3><p><strong>Failed Attempts:</strong> {failed_attempts}</p><p><strong>Remaining Attempts:</strong> {remaining_attempts}</p><p><strong>Total Restart Count:</strong> {restart_count}</p><p><strong>Error:</strong> {error}</p></div>"""

COMPLETION_DETAILS_TEMPLATE = """<div style="background:#d4edda;padding:15px;margin:15px 0;border-left:4px solid #28a745"><h3>Completion Summary</h3><p><strong>Total Restarts:</strong> {restart_count}</p><p><strong>Total Runtime:</strong> {total_runtime:.1f}s</p><p><strong>Final Status:</strong> Successfully completed</p></div>"""

# Updated monitoring script with consolidated email capabilities
MONITOR_SCRIPT = """import os,sys,time,psutil,json
sys.path.insert(0,r"{cwd}")

with open(r"{pid_file}", "w") as f:
    f.write(str(os.getpid()))

def send_crash_signal(pid, title, restart_count=0):
    \"\"\"Send crash signal for restart manager to handle.\"\"\"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    print(f"CRASH DETECTED: {{title}} (PID {{pid}}) at {{timestamp}}")
    
    with open(r"{restart_file}", "w") as f:
        json.dump({{"crashed": True, "timestamp": timestamp, "restart_count": restart_count, "pid": pid}}, f)
    
    try: os.unlink(r"{pid_file}")
    except: pass
    sys.exit(0)

try:
    proc = psutil.Process({pid})
    print(f"Monitoring PID {{pid}} for crashes")
except psutil.NoSuchProcess:
    send_crash_signal({pid}, {title})

count = 0
while True:
    # Check stop signal every 10 iterations to reduce I/O overhead
    if count % 10 == 0 and os.path.exists(r"{stop_file}"):
        try: os.unlink(r"{pid_file}")
        except: pass
        break
    
    count += 1
    
    try:
        if not proc.is_running():
            restart_count = 0
            try:
                if os.path.exists(r"{restart_file}"):
                    with open(r"{restart_file}") as f:
                        data = json.load(f)
                        restart_count = data.get("restart_count", 0)
            except:
                pass
            send_crash_signal({pid}, {title}, restart_count)
        
        # Check for zombie/stopped states that indicate crashes
        status = proc.status()
        if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_STOPPED, psutil.STATUS_DEAD]:
            restart_count = 0
            try:
                if os.path.exists(r"{restart_file}"):
                    with open(r"{restart_file}") as f:
                        data = json.load(f)
                        restart_count = data.get("restart_count", 0)
            except:
                pass
            send_crash_signal({pid}, {title}, restart_count)
            
    except psutil.NoSuchProcess:
        restart_count = 0
        try:
            if os.path.exists(r"{restart_file}"):
                with open(r"{restart_file}") as f:
                    data = json.load(f)
                    restart_count = data.get("restart_count", 0)
        except:
            pass
        send_crash_signal({pid}, {title}, restart_count)
    except Exception:
        restart_count = 0
        try:
            if os.path.exists(r"{restart_file}"):
                with open(r"{restart_file}") as f:
                    data = json.load(f)
                    restart_count = data.get("restart_count", 0)
        except:
            pass
        send_crash_signal({pid}, {title}, restart_count)
    
    time.sleep({interval})

print("Monitor completed")"""


# ——————————————————————————— Print Functions ——————————————————————————————— #
def print_monitoring_config_summary(
    file_path: str,
    file_type: str,
    success_flag_file: str,
    max_restarts: int,
    email_enabled: bool,
    title: str,
    restart_after_delay: Optional[float] = None,
) -> None:
    """Print a summary of monitoring configuration."""
    print()
    print("=" * 70)
    print("MONITORING CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Target File: {file_path}")
    print(f"File Type: {file_type}")
    print(f"Process Title: \033[33m{title}\033[0m")
    print(f"Success Flag: {success_flag_file}")
    print(f"Max Restarts: {max_restarts}")
    if restart_after_delay is not None:
        print(f"Run will force restart after: {restart_after_delay} seconds")

    if email_enabled:
        print(f"Email Alerts: \033[92mEnabled\033[0m")
    else:
        print(f"Email Alerts: \033[91mDisabled\033[0m")
    print("=" * 70)
    print()


def print_process_status(message: str, pid: Optional[int] = None, runtime: Optional[float] = None) -> None:
    """Print process status messages with consistent formatting."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    if pid and runtime is not None:
        print(f"[{timestamp}] {message} (PID {pid}, runtime: {runtime:.1f}s)")
    elif pid:
        print(f"[{timestamp}] {message} (PID {pid})")
    else:
        print(f"[{timestamp}] {message}")


def print_restart_info(restart_count: int, max_restarts: int, delay: float) -> None:
    """Print restart information with formatting."""
    print(f"Restarting in {delay:.1f}s ({restart_count}/{max_restarts})")


def print_completion_summary(restart_count: int, total_runtime: Optional[float] = None) -> None:
    """Print final completion summary."""
    print("=" * 50)
    print("MONITORING COMPLETED")
    print("=" * 50)
    print(f"Total Restarts: {restart_count}")
    if total_runtime is not None:
        print(f"Total Runtime:  {total_runtime:.1f}s")
    print("=" * 50)


def print_error_message(error_type: str, message: str) -> None:
    """Print error messages with consistent formatting."""
    print(f"ERROR [{error_type}]: {message}")


def print_warning_message(message: str) -> None:
    """Print warning messages with consistent formatting."""
    print(f"Warning: {message}")


def print_success_message(message: str) -> None:
    """Print success messages with consistent formatting."""
    print(f"SUCCESS: {message}")


def print_cleanup_info(terminated: int, killed: int) -> None:
    """Print child process cleanup information."""
    if terminated > 0 or killed > 0:
        print(f"Child cleanup: {terminated} terminated, {killed} killed")


# —————————————————————————————————— Utility ————————————————————————————————— #
def _cleanup_stale_monitor_files():
    tmpdir = tempfile.gettempdir()
    for path in glob.glob(os.path.join(tmpdir, "*_monitor.*")):
        try:
            os.unlink(path)
        except OSError:
            pass


# ——————————————————————————— Consolidated Email Manager —————————————————————————————— #
class ConsolidatedEmailManager:
    """Handles consolidated email notifications for restart events with configurable paths and retry logic."""

    __slots__ = (
        "recipients_file",
        "credentials_file",
        "email_enabled",
        "retry_attempts",
        "retry_count",
        "last_notification_time",
    )

    def __init__(
        self,
        recipients_file: Optional[str] = None,
        credentials_file: Optional[str] = None,
        retry_attempts: int = 2,
    ):
        """Initialize consolidated email manager with retry logic.

        Args:
            recipients_file: Path to recipients JSON file
            credentials_file: Path to credentials JSON file
            retry_attempts: Number of retry attempts before sending failure email
        """
        self.recipients_file = recipients_file or "./json/recipients.json"
        self.credentials_file = credentials_file or "./json/credentials.json"
        self.retry_attempts = retry_attempts
        self.email_enabled = self._validate_email_config()
        self.retry_count = 0
        self.last_notification_time = 0

    def _validate_email_config(self) -> bool:
        """Validate email configuration files exist.

        Returns:
            True if email config is valid, False otherwise
        """
        recipients_exists = Path(self.recipients_file).exists()
        credentials_exists = Path(self.credentials_file).exists()

        if not (recipients_exists and credentials_exists):
            print_warning_message("Email config files not found, email alerts disabled")
            print(f"Expected files: {self.recipients_file}, {self.credentials_file}")
            return False

        return True

    def send_consolidated_status_email(self, status_type: str, process_data: Dict[str, Any]) -> None:
        """Send consolidated status email with unified reporting.

        Args:
            status_type: Type of status ('restart_success', 'restart_failed', 'task_complete')
            process_data: Dictionary containing process information and metrics
        """
        if not self.email_enabled:
            return

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            title = process_data.get("title", "Unknown Process")

            # Generate subject and content based on status type
            if status_type == "restart_success":
                subject = f"{title} crashed - Restart Successful"
                color = "#28a745"
                status_title = "Process Restart Successful"
                status_description = "Process crashed but was successfully restarted"
                details_section = RESTART_DETAILS_TEMPLATE.format(
                    old_pid=process_data.get("old_pid", "N/A"),
                    new_pid=process_data.get("new_pid", "N/A"),
                    restart_count=process_data.get("restart_count", 0),
                    runtime=process_data.get("runtime", 0.0),
                )

            elif status_type == "restart_failed":
                failed_attempts = process_data.get("failed_attempts", 0)
                remaining = process_data.get("remaining_attempts", 0)

                if remaining > 0:
                    subject = f"{title} crashed - Restart Failed ({failed_attempts} attempts, {remaining} remaining)"
                    status_description = (
                        f"Restart failed after {failed_attempts} attempts, {remaining} attempts remaining"
                    )
                else:
                    subject = f"{title} crashed - Maximum Restarts Reached"
                    status_description = "All restart attempts have been exhausted"

                color = "#dc3545"
                status_title = "Process Restart Failed"
                details_section = FAILURE_DETAILS_TEMPLATE.format(
                    failed_attempts=failed_attempts,
                    remaining_attempts=remaining,
                    restart_count=process_data.get("restart_count", 0),
                    error=process_data.get("error", "Unknown error"),
                )

            elif status_type == "task_complete":
                subject = f"{title} - Task Completed Successfully"
                color = "#28a745"
                status_title = "Task Completed Successfully"
                status_description = "Process completed all tasks successfully"
                details_section = COMPLETION_DETAILS_TEMPLATE.format(
                    restart_count=process_data.get("restart_count", 0),
                    total_runtime=process_data.get("total_runtime", 0.0),
                )
            else:
                return  # Unknown status type

            # Generate consolidated HTML email
            html_content = CONSOLIDATED_STATUS_TEMPLATE.format(
                color=color,
                status_title=status_title,
                title=title,
                status_description=status_description,
                timestamp=timestamp,
                details_section=details_section,
            )

            send_email(subject, html_content, self.recipients_file, self.credentials_file, "html")
            self.last_notification_time = time.time()

        except Exception as e:
            print_error_message("EMAIL", f"Failed to send consolidated status email: {e}")

    def should_attempt_restart(self, title: str, restart_count: int, max_restarts: int) -> bool:
        """Determine if should attempt restart with retry logic.

        Args:
            title: Process title
            restart_count: Current restart count
            max_restarts: Maximum allowed restarts

        Returns:
            True if should attempt restart, False if should send failure email
        """
        if self.retry_count < self.retry_attempts:
            self.retry_count += 1
            return True

        # Send failure email after exhausting retries
        remaining_attempts = max(0, max_restarts - restart_count)
        self.send_consolidated_status_email(
            "restart_failed",
            {
                "title": title,
                "failed_attempts": self.retry_attempts,
                "remaining_attempts": remaining_attempts,
                "restart_count": restart_count,
                "error": f"Process failed to restart after {self.retry_attempts} retry attempts",
            },
        )

        # Reset retry count for next failure cycle
        self.retry_count = 0
        return remaining_attempts > 0

    def report_successful_restart(
        self, title: str, old_pid: Optional[int], new_pid: int, restart_count: int, runtime: float
    ) -> None:
        """Report successful restart with consolidated information.

        Args:
            title: Process title
            old_pid: Previous process PID
            new_pid: New process PID
            restart_count: Current restart count
            runtime: Runtime before restart
        """
        # Reset retry count on successful restart
        self.retry_count = 0

        self.send_consolidated_status_email(
            "restart_success",
            {
                "title": title,
                "old_pid": old_pid,
                "new_pid": new_pid,
                "restart_count": restart_count,
                "runtime": runtime,
            },
        )

    def report_task_completion(self, title: str, restart_count: int, total_runtime: float) -> None:
        """Report successful task completion.

        Args:
            title: Process title
            restart_count: Total number of restarts during execution
            total_runtime: Total execution time
        """
        self.send_consolidated_status_email(
            "task_complete", {"title": title, "restart_count": restart_count, "total_runtime": total_runtime}
        )

    def report_final_failure(self, title: str, restart_count: int, error: str) -> None:
        """Report final failure after all attempts exhausted.

        Args:
            title: Process title
            restart_count: Final restart count
            error: Error description
        """
        self.send_consolidated_status_email(
            "restart_failed",
            {
                "title": title,
                "failed_attempts": restart_count,
                "remaining_attempts": 0,
                "restart_count": restart_count,
                "error": error,
            },
        )


# ————————————————————————————— File Type Handler ———————————————————————————— #
class FileTypeHandler:
    """File type detection and command generation with caching."""

    # Class-level cache for performance optimization (bounded to prevent memory growth)
    _file_type_cache: Dict[str, str] = {}
    _command_cache: Dict[str, List[str]] = {}
    _CACHE_LIMIT = 100

    @classmethod
    def get_file_type(cls, file_path: Path) -> str:
        """Determine file type with O(1) cached lookup.

        Args:
            file_path: Path to the file

        Returns:
            File type ('python', 'notebook', 'unknown')
        """
        path_str = str(file_path)

        # O(1) cache lookup for performance
        if path_str in cls._file_type_cache:
            return cls._file_type_cache[path_str]

        # Single suffix check (most efficient approach)
        suffix = file_path.suffix.lower()
        if suffix == ".py":
            file_type = "python"
        elif suffix == ".ipynb":
            file_type = "notebook"
        else:
            file_type = "unknown"

        # Bounded cache to prevent memory growth
        if len(cls._file_type_cache) < cls._CACHE_LIMIT:
            cls._file_type_cache[path_str] = file_type

        return file_type

    @classmethod
    def build_execution_command(cls, file_path: Path, success_flag_file: str) -> Tuple[List[str], str]:
        """Build optimized execution command based on file type.

        Args:
            file_path: Path to file to execute
            success_flag_file: Path to success flag file

        Returns:
            Tuple of (command_list, execution_type)

        Raises:
            ValueError: If file type is unsupported
        """
        path_str = str(file_path.resolve())
        cache_key = f"{path_str}:{success_flag_file}"

        # Check command cache for performance optimization
        if cache_key in cls._command_cache:
            cached_cmd = cls._command_cache[cache_key]
            return cached_cmd.copy(), cls.get_file_type(file_path)

        file_type = cls.get_file_type(file_path)

        if file_type == "python":
            # Direct Python execution (most efficient)
            command = [sys.executable, "-u", str(file_path), success_flag_file]

        elif file_type == "notebook":
            # This should never be reached after conversion, but keeping for safety
            raise ValueError(f"Notebook files should be converted to Python first: {file_path}")

        else:
            raise ValueError(f"Unsupported file type: {file_type} for {file_path}")

        # Cache command with bounded size to prevent memory growth
        if len(cls._command_cache) < cls._CACHE_LIMIT:
            cls._command_cache[cache_key] = command.copy()

        return command, file_type

    @classmethod
    def validate_file(cls, file_path: str) -> Path:
        """Validate file existence and type with early exit pattern.

        Args:
            file_path: Path string to validate

        Returns:
            Resolved Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is unsupported
        """
        path_obj = Path(file_path)

        # Early exit validation for performance
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        file_type = cls.get_file_type(path_obj)
        if file_type == "unknown":
            raise ValueError(f"Unsupported file type: {path_obj.suffix}. Supported: .py, .ipynb")

        return path_obj.resolve()


# —————————————————————————————— Enhanced Restart Manager ————————————————————————————— #
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

    def run_file_with_restart(
        self,
        file_path: str,
        success_flag_file: str,
        title: Optional[str] = None,
        restart_after_delay: Optional[float] = None,
        supress_tf_warnings: bool = False
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
            print_process_status(f"Converting notebook to Python: {validated_path.name}")
            try:
                self.converted_python_file = NotebookConverter.convert_notebook_to_python(validated_path)
                self.original_was_notebook = True
                validated_path = self.converted_python_file
                file_type = "python"
            except Exception as e:
                print_error_message("CONVERSION", f"Notebook conversion failed: {e}")
                raise

        working_dir = str(validated_path.parent)
        self.process_title = title or validated_path.stem
        flag_path = Path(success_flag_file).resolve()

        # Print configuration summary
        print_monitoring_config_summary(
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

            # Main restart loop with consolidated email notifications
            while self.running and self.restart_count < self.max_restarts:
                # Remove old success flag (atomic operation)
                if flag_path.exists():
                    flag_path.unlink()

                self.last_process_start_time = time.time()

                try:
                    if self.monitor_info:
                        stop_monitor(self.monitor_info)
                        self.monitor_info = None

                    # Launch process
                    target_pid = self._launch_process(validated_path, working_dir, success_flag_file)
                    print_process_status("Process started", target_pid)

                    # Send successful restart email (only for actual restarts, not first start)
                    if self.restart_count > 0:
                        runtime = time.time() - self.last_process_start_time
                        self.email_manager.report_successful_restart(
                            self.process_title, previous_pid, target_pid, self.restart_count, runtime
                        )

                    # Start crash monitor with simplified monitoring
                    self.monitor_info = start_monitor(target_pid, self.process_title, supress_tf_warnings=supress_tf_warnings)
                    self._last_restart_file = self.monitor_info["restart_file"]

                    # Wait for completion or crash with optimized polling
                    completion_reason = self._wait_for_completion(flag_path)
                    runtime = time.time() - self.last_process_start_time

                    print_process_status(f"Process finished: {completion_reason}", target_pid, runtime)

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
                        print_process_status("Process crashed, checking restart policy")
                        if not self._handle_restart_with_retry():
                            break
                    elif completion_reason == "interrupted":
                        # User pressed CTRL+C, clean up and exit
                        print_process_status("Process interrupted by user")
                        break
                    else:
                        print_process_status("Process ended without success flag, treating as failure")
                        if not self._handle_restart_with_retry():
                            break

                except Exception as e:
                    print_error_message("LAUNCH", str(e))
                    self._cleanup_all()
                    if not self._handle_restart_with_retry():
                        break

            # Handle maximum restarts reached
            if self.restart_count >= self.max_restarts:
                print_error_message("MAX_RESTARTS", f"Maximum restarts reached: {self.max_restarts}")
                self.email_manager.report_final_failure(
                    self.process_title,
                    self.restart_count,
                    f"Maximum restart attempts ({self.max_restarts}) reached",
                )

        except KeyboardInterrupt:
            print_process_status("Interrupted by user, cleaning up resources")
            self.running = False
        except Exception as e:
            print_error_message("FATAL", str(e))
            self.email_manager.report_final_failure(
                self.process_title, self.restart_count, f"Fatal error: {str(e)}"
            )
        finally:
            # Ensure all cleanup operations are performed
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
                print_cleanup_info(terminated, killed)
            except psutil.NoSuchProcess:
                print_warning_message("Current process not found during cleanup")
            except Exception as e:
                print_error_message("CLEANUP", f"Child cleanup failed (non-fatal): {e}")

            # Exponential backoff with cap at 30 seconds
            delay = min(self.restart_delay * (1.2 ** (self.restart_count - 1)), 30.0)
            print_restart_info(self.restart_count, self.max_restarts, delay)
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
                    crash_info = check_crash_signal(self.monitor_info)
                    if crash_info:
                        return "crashed"

                # Check process existence every 4th iteration for efficiency
                if check_count % 4 == 0 and self.current_target_pid:
                    if not psutil.pid_exists(self.current_target_pid):
                        return "process_died"

                time.sleep(0.5)
            except KeyboardInterrupt:
                # Handle CTRL+C by cleaning up the current monitored process
                print_process_status("CTRL+C detected, shutting down monitored process")
                self.running = False
                self._cleanup_all()
                return "interrupted"

        return "stopped"

    def _cleanup_all(self) -> None:
        """Cleanup all resources with optimized order for reliability."""
        # Stop monitor first (most critical for clean shutdown)
        if self.monitor_info:
            stop_monitor(self.monitor_info)
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

    def _cleanup_converted_file(self) -> None:
        """Delete converted Python file if original was a notebook.

        Only deletes the file if it was converted from a notebook during this session.
        Direct .py files are never deleted.
        """
        if self.original_was_notebook and self.converted_python_file:
            try:
                if self.converted_python_file.exists():
                    self.converted_python_file.unlink()
                    print_process_status(f"Cleaned up converted file: {self.converted_python_file}")
            except Exception as e:
                print_warning_message(f"Failed to cleanup converted file {self.converted_python_file}: {e}")
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
                print_process_status("CTRL+C detected during restart delay, aborting restart")
                break


def start_monitor(pid: int, title: str, supress_tf_warnings: bool = False) -> Dict[str, Any]:
    """Start simplified crash monitor without email capabilities.

    Args:
        pid: Process ID to monitor
        title: Process title for alerts
        supress_tf_warnings: Suppress TensorFlow warnings (default: False)

    Returns:
        Monitor control info dictionary

    Raises:
        ValueError: If PID doesn't exist
        OSError: If monitor startup fails
    """
    _cleanup_stale_monitor_files()
    time.sleep(0.1)  # Allow time for process to stabilize

    if not psutil.pid_exists(pid):
        raise ValueError(f"Process PID {pid} not found")

    # Create minimal control files
    fd, script_path = tempfile.mkstemp(suffix="_monitor.py")
    base_path = script_path.replace(".py", "")

    control_files = {
        "script_path": script_path,
        "pid_file": f"{base_path}.pid",
        "stop_file": f"{base_path}.stop",
        "restart_file": f"{base_path}.restart",
    }

    # Generate simplified monitoring script
    script_content = MONITOR_SCRIPT.format(
        cwd=os.getcwd(),
        pid=pid,
        interval=2,
        title=repr(title),
        **control_files,
    )

    with os.fdopen(fd, "w") as f:
        f.write(script_content)

    if os.name != "nt":
        os.chmod(script_path, 0o755)

    # Launch monitor in terminal
    launcher = SimpleTerminalLauncher()
    launcher.set_supress_tf_warnings(supress_tf_warnings)
    process = launcher.launch([sys.executable, script_path], os.getcwd())

    time.sleep(0.1)
    if process.poll() is not None:  # Check if it died
        exit_code = process.returncode
        error_msg = f"Monitor failed to start (exit code: {exit_code})"

        # Try to get stderr output if available
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stderr:
                error_msg += f". Error output: {stderr.decode().strip()}"
            elif stdout:
                error_msg += f". Output: {stdout.decode().strip()}"
        except:
            pass

        # Cleanup the failed script file
        try:
            os.unlink(script_path)
        except:
            pass

        raise OSError(error_msg)

    return {"process": process, **control_files}


def stop_monitor(monitor_info: Dict[str, Any]) -> None:
    """Stop monitor and cleanup files with optimized batch operations.

    Args:
        monitor_info: Monitor control info from start_monitor()
    """
    if not monitor_info:
        return

    # Signal stop (single I/O operation)
    try:
        with open(monitor_info["stop_file"], "w") as f:
            f.write("STOP")
    except:
        pass

    # Wait for graceful shutdown with optimized timeout
    for _ in range(20):  # 2 second timeout
        if not os.path.exists(monitor_info["pid_file"]):
            break
        time.sleep(0.1)

    # Force terminate if needed
    process = monitor_info.get("process")
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            pass

    # Batch file cleanup (single loop for efficiency)
    cleanup_files = ["script_path", "pid_file", "stop_file", "restart_file"]
    for file_key in cleanup_files:
        try:
            file_path = monitor_info.get(file_key)
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass


def check_crash_signal(monitor_info: Dict[str, Any]) -> Dict[str, Any]:
    """Check if process crashed with minimal I/O operations.

    Args:
        monitor_info: Monitor control info

    Returns:
        Dictionary with crash info or empty dict if no crash
    """
    restart_file = monitor_info.get("restart_file")
    if not restart_file or not os.path.exists(restart_file):
        return {}

    try:
        with open(restart_file) as f:
            data = json.load(f)
            if data.get("crashed", False):
                return data
    except:
        pass

    return {}


def run_auto_restart(
    file_path: str,
    success_flag_file: str = "/tmp/success.flag",
    title: Optional[str] = None,
    max_restarts: int = 10,
    restart_delay: float = 3.0,
    recipients_file: Optional[str] = None,
    credentials_file: Optional[str] = None,
    restart_after_delay: Optional[float] = None,
    retry_attempts: int = None,
    supress_tf_warnings: bool = False,
) -> None:
    """Main function with notebook conversion, file cleanup, and consolidated email notification support.

    Args:
        file_path: Path to .py or .ipynb file to execute
        success_flag_file: Path to success flag file
        title: Custom title for monitoring and email alerts
        max_restarts: Maximum restart attempts
        restart_delay: Delay between restarts in seconds
        recipients_file: Path to recipients JSON file (defaults to ./json/recipients.json)
        credentials_file: Path to credentials JSON file (defaults to ./json/credentials.json)
        restart_after_delay: restart the run after a delay in seconds
        retry_attempts: Number of retry attempts before sending failure email
        supress_tf_warnings: Suppress TensorFlow warnings (default: False)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
        ImportError: If notebook dependencies missing for .ipynb files
    """

    try:
        # Clean up any existing success flag file before starting
        Path(success_flag_file).unlink(missing_ok=True)

        manager = FlagBasedRestartManager(
            max_restarts=max_restarts,
            restart_delay=restart_delay,
            recipients_file=recipients_file,
            credentials_file=credentials_file,
            retry_attempts=max_restarts if retry_attempts is None else retry_attempts,
        )

        if restart_after_delay is not None and restart_after_delay > 0:
            # Wrapping logic for forced restart not counting as crash/max_restarts
            # This will run in a loop, restarting after each interval, until success_flag is found.

            stop_event = Event()

            def restart_loop():
                try:
                    while not stop_event.is_set():
                        manager.restart_count = 0  # Never increment max_restarts for forced restart
                        finished = [False]

                        def run_and_flag():
                            try:
                                manager.run_file_with_restart(
                                    file_path=file_path,
                                    success_flag_file=success_flag_file,
                                    title=title,
                                    restart_after_delay=restart_after_delay,
                                    supress_tf_warnings=supress_tf_warnings,
                                )
                                finished[0] = True
                            except Exception:
                                finished[0] = True  # On error, still allow restart

                        thread = Thread(target=run_and_flag)
                        thread.start()
                        thread.join(timeout=restart_after_delay)
                        if thread.is_alive():
                            print_process_status(
                                f"Forcing restart after {restart_after_delay} seconds (not a crash)"
                            )

                            # First stop the monitor before restarting the process
                            if manager.monitor_info:
                                print_process_status("Stopping monitor before restart")
                                stop_monitor(manager.monitor_info)
                                manager.monitor_info = None
                                time.sleep(0.1)

                            manager._cleanup_all()
                            # Intentionally NOT incrementing restart_count
                            # Signal process to stop, then continue
                            # The completion reason will be 'stopped', and the outer loop will restart
                            # Wait for thread to finish cleanup
                            thread.join(2)
                            clear()

                        else:
                            # If finished (success or crash), check if success
                            if Path(success_flag_file).exists():
                                stop_event.set()
                            else:
                                print_process_status(
                                    "Process ended before restart_after_delay, restarting..."
                                )     
                except KeyboardInterrupt:
                    # Handle CTRL+C in the restart loop
                    stop_event.set()
                    print_process_status("Restart loop interrupted by user, cleaning up")
                    manager._cleanup_all()
                    manager._cleanup_converted_file()
                print_process_status("Restart-after-delay loop done")

            restart_loop()

        else:
            # Regular auto-restart logic
            manager.run_file_with_restart(
                file_path=file_path,
                success_flag_file=success_flag_file,
                title=title,
                supress_tf_warnings=supress_tf_warnings,
            )

    except (FileNotFoundError, ValueError, ImportError) as e:
        print_error_message("CONFIG", str(e))
        raise
    except KeyboardInterrupt:
        print_process_status("Main process interrupted by user, performing final cleanup")
    except Exception as e:
        print_error_message("FATAL", str(e))
        raise
