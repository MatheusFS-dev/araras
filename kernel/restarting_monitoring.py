"""
This module provides a restarting monitoring system for processes with email alert capabilities.
It monitors a process for crashes, restarts it if necessary, and sends email notifications.

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
import tempfile
import subprocess
import psutil
import platform
from pathlib import Path
from typing import *
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from araras.email.utils import send_email, notify_training_success


# —————————————————————————————— HTML templates —————————————————————————————— #
CRASH_ALERT_TEMPLATE = """<html><body style="font-family:Arial,sans-serif;color:#333;padding:20px"><div style="max-width:600px;margin:auto;background:#fff;padding:20px;border:1px solid #ddd"><h2 style="color:#d9534f">Process Crash Alert</h2><p>Process "<strong>{title}</strong>" (PID <strong>{pid}</strong>) has crashed.</p><p>Time: {timestamp}</p><p>Restart Count: {restart_count}</p></div></body></html>"""

SUCCESS_RESTART_TEMPLATE = """<html><body style="font-family:Arial,sans-serif;color:#333;padding:20px"><div style="max-width:600px;margin:auto;background:#fff;padding:20px;border:1px solid #ddd"><h2 style="color:#5cb85c">Successful Restart</h2><p>Process "<strong>{title}</strong>" has been successfully restarted.</p><p>Time: {timestamp}</p><p>Previous PID: {old_pid}</p><p>New PID: {new_pid}</p><p>Restart Count: {restart_count}</p></div></body></html>"""

FAILED_RESTART_TEMPLATE = """<html><body style="font-family:Arial,sans-serif;color:#333;padding:20px"><div style="max-width:600px;margin:auto;background:#fff;padding:20px;border:1px solid #ddd"><h2 style="color:#f0ad4e">Restart Failed</h2><p>Process "<strong>{title}</strong>" failed to restart after maximum attempts.</p><p>Time: {timestamp}</p><p>Final Restart Count: {restart_count}</p><p>Error: {error}</p></div></body></html>"""

# —————————————— Monitoring script with email alert capabilities ————————————— #
MONITOR_SCRIPT = """import os,sys,time,psutil,json
sys.path.insert(0,r"{cwd}")

with open(r"{pid_file}", "w") as f:
    f.write(str(os.getpid()))

def send_email_alert(alert_type, data, recipients, credentials):
    \"\"\"Send email alert based on type.\"\"\"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    
    try:
        from araras.email.utils import send_email
        
        if alert_type == "crash":
            subject = f"CRASH ALERT: {{data['title']}} (PID {{data['pid']}})"
            html_content = "{crash_template}".format(
                title=data['title'], 
                pid=data['pid'], 
                timestamp=timestamp,
                restart_count=data.get('restart_count', 0)
            )
        elif alert_type == "success_restart":
            subject = f"RESTART SUCCESS: {{data['title']}}"
            html_content = "{success_template}".format(
                title=data['title'],
                timestamp=timestamp,
                old_pid=data.get('old_pid', 'N/A'),
                new_pid=data.get('new_pid', 'N/A'),
                restart_count=data.get('restart_count', 0)
            )
        elif alert_type == "failed_restart":
            subject = f"RESTART FAILED: {{data['title']}}"
            html_content = "{failed_template}".format(
                title=data['title'],
                timestamp=timestamp,
                restart_count=data.get('restart_count', 0),
                error=data.get('error', 'Unknown error')
            )
        else:
            return
            
        send_email(subject, html_content, recipients, credentials, "html")
        print(f"{{alert_type.upper()}} email alert sent")
        
    except Exception as e:
        print(f"Email alert failed: {{e}}")

def send_crash_alert(pid, title, recipients, credentials, restart_count=0):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    print(f"CRASH ALERT: {{title}} (PID {{pid}}) at {{timestamp}}")
    
    # Send crash email
    send_email_alert("crash", {{
        'title': title,
        'pid': pid,
        'restart_count': restart_count
    }}, recipients, credentials)
    
    with open(r"{restart_file}", "w") as f:
        json.dump({{"crashed": True, "timestamp": timestamp, "restart_count": restart_count}}, f)
    
    try: os.unlink(r"{pid_file}")
    except: pass
    sys.exit(0)

try:
    proc = psutil.Process({pid})
    print(f"Monitoring PID {{pid}} for crashes")
except psutil.NoSuchProcess:
    send_crash_alert({pid}, {title}, r"{recipients}", r"{credentials}")

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
            # Try to get restart count from restart file for context
            restart_count = 0
            try:
                if os.path.exists(r"{restart_file}"):
                    with open(r"{restart_file}") as f:
                        data = json.load(f)
                        restart_count = data.get("restart_count", 0)
            except:
                pass
            send_crash_alert({pid}, {title}, r"{recipients}", r"{credentials}", restart_count)
        
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
            send_crash_alert({pid}, {title}, r"{recipients}", r"{credentials}", restart_count)
            
    except psutil.NoSuchProcess:
        restart_count = 0
        try:
            if os.path.exists(r"{restart_file}"):
                with open(r"{restart_file}") as f:
                    data = json.load(f)
                    restart_count = data.get("restart_count", 0)
        except:
            pass
        send_crash_alert({pid}, {title}, r"{recipients}", r"{credentials}", restart_count)
    except Exception:
        restart_count = 0
        try:
            if os.path.exists(r"{restart_file}"):
                with open(r"{restart_file}") as f:
                    data = json.load(f)
                    restart_count = data.get("restart_count", 0)
        except:
            pass
        send_crash_alert({pid}, {title}, r"{recipients}", r"{credentials}", restart_count)
    
    time.sleep({interval})

print("Monitor completed")"""


# ——————————————————————————— Child Process Cleanup —————————————————————————— #
class ChildProcessCleanup:
    """Efficient child process cleanup with graceful termination and force kill fallback."""

    __slots__ = ("_current_pid", "_exclude_pids", "_termination_timeout", "_kill_timeout")

    def __init__(self, termination_timeout: float = 2.0, kill_timeout: float = 1.0):
        """Initialize cleanup manager with configurable timeouts.

        Args:
            termination_timeout: Seconds to wait for graceful termination
            kill_timeout: Seconds to wait after force kill
        """
        self._current_pid = os.getpid()
        self._exclude_pids: Set[int] = {self._current_pid}
        self._termination_timeout = termination_timeout
        self._kill_timeout = kill_timeout

    def cleanup_children(self, exclude_pids: Optional[List[int]] = None) -> Tuple[int, int]:
        """Clean up all child processes with optimized batch operations.

        Args:
            exclude_pids: Additional PIDs to exclude from cleanup

        Returns:
            Tuple of (terminated_count, killed_count)

        Raises:
            psutil.NoSuchProcess: If current process doesn't exist
        """
        # Combine exclude PIDs for efficient lookup
        exclude_set = self._exclude_pids.copy()
        if exclude_pids:
            exclude_set.update(exclude_pids)

        try:
            current_process = psutil.Process(self._current_pid)
            children = current_process.children(recursive=True)

            if not children:
                return 0, 0

            # Filter children to cleanup (exclude protected PIDs)
            targets = [child for child in children if child.pid not in exclude_set]

            if not targets:
                return 0, 0

            print(f"Cleaning up {len(targets)} child processes")

            # Phase 1: Graceful termination with parallel execution
            terminated_count = self._terminate_processes(targets)

            # Phase 2: Force kill remaining processes
            killed_count = self._kill_remaining_processes(targets, exclude_set)

            return terminated_count, killed_count

        except psutil.NoSuchProcess:
            raise psutil.NoSuchProcess(f"Current process {self._current_pid} not found")

    def _terminate_processes(self, processes: List[psutil.Process]) -> int:
        """Terminate processes gracefully with parallel execution.

        Args:
            processes: List of processes to terminate

        Returns:
            Number of processes successfully terminated
        """
        terminated_count = 0

        # Parallel termination for efficiency
        with ThreadPoolExecutor(max_workers=min(len(processes), 8)) as executor:
            # Submit termination tasks
            future_to_process = {executor.submit(self._safe_terminate, proc): proc for proc in processes}

            # Collect results
            for future in as_completed(future_to_process, timeout=self._termination_timeout + 1):
                if future.result():
                    terminated_count += 1

        # Wait for graceful shutdown
        if terminated_count > 0:
            time.sleep(self._termination_timeout)

        return terminated_count

    def _safe_terminate(self, process: psutil.Process) -> bool:
        """Safely terminate a single process with error handling.

        Args:
            process: Process to terminate

        Returns:
            True if termination signal sent successfully, False otherwise
        """
        try:
            if process.is_running():
                process.terminate()
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return False

    def _kill_remaining_processes(
        self, original_processes: List[psutil.Process], exclude_set: Set[int]
    ) -> int:
        """Force kill processes that didn't terminate gracefully.

        Args:
            original_processes: Original list of processes to check
            exclude_set: PIDs to exclude from force kill

        Returns:
            Number of processes force killed
        """
        killed_count = 0

        # Re-check which processes are still running
        for process in original_processes:
            if process.pid in exclude_set:
                continue

            try:
                if process.is_running():
                    process.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process already gone or no permission
                pass

        # Brief wait after force kill
        if killed_count > 0:
            time.sleep(self._kill_timeout)

        return killed_count

    def add_protected_pid(self, pid: int) -> None:
        """Add a PID to the protected (exclude) list.

        Args:
            pid: Process ID to protect from cleanup
        """
        self._exclude_pids.add(pid)

    def remove_protected_pid(self, pid: int) -> None:
        """Remove a PID from the protected list.

        Args:
            pid: Process ID to remove from protection
        """
        self._exclude_pids.discard(pid)  # discard won't raise if not present

    def get_child_count(self) -> int:
        """Get current number of child processes.

        Returns:
            Number of child processes (including nested children)
        """
        try:
            current_process = psutil.Process(self._current_pid)
            return len(current_process.children(recursive=True))
        except psutil.NoSuchProcess:
            return 0


# ——————————————————————————— Notifications Manager —————————————————————————— #
class EmailNotificationManager:
    """Handles email notifications for restart events with configurable paths."""

    __slots__ = ("recipients_file", "credentials_file", "email_enabled")

    def __init__(self, recipients_file: Optional[str] = None, credentials_file: Optional[str] = None):
        """Initialize email manager with configurable paths.

        Args:
            recipients_file: Path to recipients JSON file
            credentials_file: Path to credentials JSON file
        """
        self.recipients_file = recipients_file or "./json/recipients.json"
        self.credentials_file = credentials_file or "./json/credentials.json"
        self.email_enabled = self._validate_email_config()

    def _validate_email_config(self) -> bool:
        """Validate email configuration files exist.

        Returns:
            True if email config is valid, False otherwise
        """
        recipients_exists = Path(self.recipients_file).exists()
        credentials_exists = Path(self.credentials_file).exists()

        if not (recipients_exists and credentials_exists):
            print("\033[91mWarning: Email config files not found, email alerts disabled\033[0m")
            print(f"Expected files: {self.recipients_file}, {self.credentials_file}")
            return False

        return True

    def send_restart_success_alert(
        self, title: str, old_pid: Optional[int], new_pid: int, restart_count: int
    ) -> None:
        """Send successful restart notification.

        Args:
            title: Process title
            old_pid: Previous process PID
            new_pid: New process PID
            restart_count: Current restart count
        """
        if not self.email_enabled:
            return

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            subject = f"RESTART SUCCESS: {title}"
            html_content = SUCCESS_RESTART_TEMPLATE.format(
                title=title,
                timestamp=timestamp,
                old_pid=old_pid or "N/A",
                new_pid=new_pid,
                restart_count=restart_count,
            )

            send_email(subject, html_content, self.recipients_file, self.credentials_file, "html")
            print("Successful restart email alert sent")

        except Exception as e:
            print(f"Failed to send restart success email: {e}")

    def send_restart_failed_alert(
        self, title: str, restart_count: int, error: str = "Maximum restart attempts reached"
    ) -> None:
        """Send restart failure notification.

        Args:
            title: Process title
            restart_count: Final restart count
            error: Error description
        """
        if not self.email_enabled:
            return

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            subject = f"RESTART FAILED: {title}"
            html_content = FAILED_RESTART_TEMPLATE.format(
                title=title, timestamp=timestamp, restart_count=restart_count, error=error
            )

            send_email(subject, html_content, self.recipients_file, self.credentials_file, "html")
            print("Restart failure email alert sent")

        except Exception as e:
            print(f"Failed to send restart failure email: {e}")

    def send_task_completion_alert(self, title: str) -> None:
        """Send task completion notification.

        Args:
            title: Process title
        """
        if not self.email_enabled:
            return

        try:
            notify_training_success(
                recipients_file=self.recipients_file,
                credentials_file=self.credentials_file,
                subject=f"Training Complete: {title}",
            )
            print("Task completion email alert sent")

        except Exception as e:
            print(f"Failed to send task completion email: {e}")


# ——————————————————————————— Notebook Converter ———————————————————————————— #
class NotebookConverter:
    """Efficient notebook to Python conversion with minimal memory overhead."""

    @staticmethod
    def convert_notebook_to_python(notebook_path: Path) -> Path:
        """Convert Jupyter notebook to Python file with same name.

        Args:
            notebook_path: Path to .ipynb file

        Returns:
            Path to generated .py file

        Raises:
            ImportError: If notebook dependencies missing
            ValueError: If notebook conversion fails
        """
        try:
            import nbformat
        except ImportError as e:
            raise ImportError("Missing notebook dependencies. " "Please install: pip install nbformat") from e

        # Target Python file path (same directory, same name, .py extension)
        python_path = notebook_path.with_suffix(".py")

        try:
            # Load notebook with minimal memory footprint
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            # Extract code cells efficiently
            python_lines = [
                "#!/usr/bin/env python",
                "# -*- coding: utf-8 -*-",
                f"# Converted from: {notebook_path.name}",
                f'# Generated on: {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}',
                "",
            ]

            # Process cells with memory-efficient iteration
            for cell_idx, cell in enumerate(notebook.cells):
                if cell.cell_type == "code" and cell.source.strip():
                    # Add cell separator for debugging
                    python_lines.append(f"# Cell {cell_idx + 1}")

                    # Clean and add source code
                    source_lines = cell.source.strip().split("\n")
                    python_lines.extend(source_lines)
                    python_lines.append("")  # Empty line between cells

            # Write to Python file atomically
            temp_path = python_path.with_suffix(".py.tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(python_lines))

                # Atomic rename for consistency
                temp_path.replace(python_path)

            except Exception:
                # Cleanup temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise

            print(f"Converted notebook to: {python_path}")
            return python_path

        except Exception as e:
            raise ValueError(f"Failed to convert notebook {notebook_path}: {e}") from e


# ————————————————————————————— File Type Handler ———————————————————————————— #
class FileTypeHandler:
    """Efficient file type detection and command generation with caching."""

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


# —————————————————————————————— Restart Manager ————————————————————————————— #
class FlagBasedRestartManager:
    """Enhanced restart manager with comprehensive email notifications and file cleanup."""

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
    )

    def __init__(
        self,
        max_restarts: int = 10,
        restart_delay: float = 3.0,
        recipients_file: Optional[str] = None,
        credentials_file: Optional[str] = None,
    ):
        """Initialize restart manager with email notification support.

        Args:
            max_restarts: Maximum restart attempts
            restart_delay: Delay between restarts in seconds
            recipients_file: Path to recipients JSON file
            credentials_file: Path to credentials JSON file
        """
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.restart_count = 0
        self.running = False

        # Process tracking with minimal state
        self.current_terminal_process: Optional[subprocess.Popen] = None
        self.current_target_pid: Optional[int] = None
        self.monitor_info: Optional[Dict[str, Any]] = None

        # File cleanup tracking
        self.converted_python_file: Optional[Path] = None
        self.original_was_notebook: bool = False

        # Email configuration
        self.recipients_file = recipients_file or "./json/recipients.json"
        self.credentials_file = credentials_file or "./json/credentials.json"
        self.email_manager = EmailNotificationManager(self.recipients_file, self.credentials_file)
        self.process_title: str = ""

        # Child process cleanup manager
        self.child_cleanup = ChildProcessCleanup()

    def run_file_with_restart(
        self, file_path: str, success_flag_file: str, title: Optional[str] = None
    ) -> None:
        """Run file with flag-based restart logic and email notifications.

        Args:
            file_path: Path to Python or Jupyter notebook file
            success_flag_file: Path where target process writes completion flag
            title: Custom title for monitoring

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is unsupported
        """
        # Validate file with early exit pattern for performance
        validated_path = FileTypeHandler.validate_file(file_path)
        file_type = FileTypeHandler.get_file_type(validated_path)

        # Convert notebook to Python if needed
        if file_type == "notebook":
            print(f"Converting notebook to Python: {validated_path.name}")
            try:
                self.converted_python_file = NotebookConverter.convert_notebook_to_python(validated_path)
                self.original_was_notebook = True
                validated_path = self.converted_python_file
                file_type = "python"
            except Exception as e:
                print(f"Notebook conversion failed: {e}")
                raise

        working_dir = str(validated_path.parent)
        self.process_title = title or validated_path.stem
        flag_path = Path(success_flag_file).resolve()

        print(f"Starting flag-based auto-restart for: {validated_path}")
        print(f"File type: {file_type}")
        print(f"Success flag file: {flag_path}")
        print(f"Max restarts: {self.max_restarts}")
        print(f"Email alerts: {'Enabled' if self.email_manager.email_enabled else 'Disabled'}")

        self.running = True
        previous_pid = None

        try:
            # Main restart loop with enhanced email notifications
            while self.running and self.restart_count < self.max_restarts:
                # Remove old success flag (atomic operation)
                if flag_path.exists():
                    flag_path.unlink()

                start_time = time.time()

                try:
                    # Launch process
                    target_pid = self._launch_process(validated_path, working_dir, success_flag_file)
                    print(f"Process started: PID {target_pid}")

                    # Send successful restart email (except for first start)
                    if self.restart_count > 0:
                        self.email_manager.send_restart_success_alert(
                            self.process_title, previous_pid, target_pid, self.restart_count
                        )

                    # Start crash monitor with email configuration
                    self.monitor_info = start_monitor(
                        target_pid, self.process_title, self.recipients_file, self.credentials_file
                    )

                    # Wait for completion or crash with optimized polling
                    completion_reason = self._wait_for_completion(flag_path)
                    runtime = time.time() - start_time

                    print(f"Process finished: {completion_reason}, runtime: {runtime:.1f}s")

                    # Store PID for next restart notification
                    previous_pid = target_pid

                    # Immediate cleanup for memory efficiency
                    self._cleanup_all()

                    # Smart decision logic based on completion reason
                    if completion_reason == "success_flag":
                        print("Process completed successfully")
                        self.email_manager.send_task_completion_alert(self.process_title)
                        break
                    elif completion_reason == "crashed":
                        print("Process crashed, will restart")
                        self._handle_restart()
                    else:
                        print("Process ended without success flag, treating as failure")
                        self._handle_restart()

                except Exception as e:
                    print(f"Launch error: {e}")
                    self._cleanup_all()
                    self._handle_restart()

            # Handle maximum restarts reached
            if self.restart_count >= self.max_restarts:
                print(f"Maximum restarts reached: {self.max_restarts}")
                self.email_manager.send_restart_failed_alert(
                    self.process_title,
                    self.restart_count,
                    f"Maximum restart attempts ({self.max_restarts}) reached",
                )

        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Fatal error: {e}")
            self.email_manager.send_restart_failed_alert(
                self.process_title, self.restart_count, f"Fatal error: {str(e)}"
            )
        finally:
            self._cleanup_all()
            self._cleanup_converted_file()
            print(f"Total restarts: {self.restart_count}")

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

        return "stopped"

    def _handle_restart(self) -> None:
        """Handle restart logic with exponential backoff for stability."""
        self.restart_count += 1
        if self.restart_count < self.max_restarts:
            # Protect current target process if still running
            exclude_pids = []
            if self.current_target_pid and psutil.pid_exists(self.current_target_pid):
                exclude_pids.append(self.current_target_pid)

            # Perform child process cleanup before restart
            try:
                terminated, killed = self.child_cleanup.cleanup_children(exclude_pids)
                if terminated > 0 or killed > 0:
                    print(f"Child cleanup: {terminated} terminated, {killed} killed")
            except psutil.NoSuchProcess:
                print("Warning: Current process not found during cleanup")
            except Exception as e:
                print(f"Child cleanup failed (non-fatal): {e}")

            # Exponential backoff with cap at 30 seconds
            delay = min(self.restart_delay * (1.2 ** (self.restart_count - 1)), 30.0)
            print(f"Restarting in {delay:.1f}s ({self.restart_count}/{self.max_restarts})")
            self._sleep(delay)

    def _cleanup_all(self) -> None:
        """Cleanup all resources with optimized order for reliability."""
        # Stop monitor first (most critical for clean shutdown)
        if self.monitor_info:
            stop_monitor(self.monitor_info)
            self.monitor_info = None

        # Terminate target process
        if self.current_target_pid:
            try:
                proc = psutil.Process(self.current_target_pid)
                proc.terminate()
                proc.wait(timeout=3)
            except:
                pass
            self.current_target_pid = None

        # Cleanup terminal last
        self._cleanup_terminal()

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
                    print(f"Cleaned up converted file: {self.converted_python_file}")
            except Exception as e:
                print(f"Warning: Failed to cleanup converted file {self.converted_python_file}: {e}")
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
            time.sleep(min(0.1, end_time - time.time()))


def start_monitor(pid: int, title: str, recipients_file: str, credentials_file: str) -> Dict[str, Any]:
    """Start enhanced crash monitor with email alert capabilities.

    Args:
        pid: Process ID to monitor
        title: Process title for alerts
        recipients_file: Path to recipients configuration file
        credentials_file: Path to credentials configuration file

    Returns:
        Monitor control info dictionary

    Raises:
        ValueError: If PID doesn't exist
        OSError: If monitor startup fails
    """
    if not psutil.pid_exists(pid):
        raise ValueError(f"Process PID {pid} not found")

    # Use provided email config files or disable if missing
    if not (Path(recipients_file).exists() and Path(credentials_file).exists()):
        print("Warning: Email config files not found, email alerts disabled")
        recipients_file = credentials_file = "/dev/null"

    # Create minimal control files
    fd, script_path = tempfile.mkstemp(suffix="_monitor.py")
    base_path = script_path.replace(".py", "")

    control_files = {
        "script_path": script_path,
        "pid_file": f"{base_path}.pid",
        "stop_file": f"{base_path}.stop",
        "restart_file": f"{base_path}.restart",
    }

    # Generate enhanced monitoring script with email templates
    script_content = MONITOR_SCRIPT.format(
        cwd=os.getcwd(),
        pid=pid,
        interval=2,
        title=repr(title),
        recipients=recipients_file,
        credentials=credentials_file,
        crash_template=CRASH_ALERT_TEMPLATE.replace('"', '\\"').replace("\n", "\\n"),
        success_template=SUCCESS_RESTART_TEMPLATE.replace('"', '\\"').replace("\n", "\\n"),
        failed_template=FAILED_RESTART_TEMPLATE.replace('"', '\\"').replace("\n", "\\n"),
        **control_files,
    )

    with os.fdopen(fd, "w") as f:
        f.write(script_content)

    if os.name != "nt":
        os.chmod(script_path, 0o755)

    # Launch monitor in terminal
    launcher = SimpleTerminalLauncher()
    process = launcher.launch([sys.executable, script_path], os.getcwd())

    time.sleep(0.1)
    if process.poll() is not None:
        raise OSError("Monitor failed to start")

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
) -> None:
    """Main function with notebook conversion, file cleanup, and enhanced email notification support.

    Args:
        file_path: Path to .py or .ipynb file to execute
        success_flag_file: Path to success flag file
        title: Custom title for monitoring and email alerts
        max_restarts: Maximum restart attempts
        restart_delay: Delay between restarts in seconds
        recipients_file: Path to recipients JSON file (defaults to ./json/recipients.json)
        credentials_file: Path to credentials JSON file (defaults to ./json/credentials.json)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
        ImportError: If notebook dependencies missing for .ipynb files
    """
    try:
        # Clean up any existing success flag file before starting
        Path(success_flag_file).unlink(missing_ok=True)

        # Validate file early with detailed error messages
        validated_path = FileTypeHandler.validate_file(file_path)
        file_type = FileTypeHandler.get_file_type(validated_path)

        # Check notebook dependencies if needed
        if file_type == "notebook":
            try:
                import nbformat
            except ImportError as e:
                raise ImportError(
                    "Missing notebook dependencies. " "Please install: pip install nbformat"
                ) from e

        # Create enhanced restart manager with configurable email paths
        manager = FlagBasedRestartManager(
            max_restarts=max_restarts,
            restart_delay=restart_delay,
            recipients_file=recipients_file,
            credentials_file=credentials_file,
        )

        manager.run_file_with_restart(file_path=file_path, success_flag_file=success_flag_file, title=title)

    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
