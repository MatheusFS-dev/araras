"""
Flag-based auto-restart launcher with minimal overhead.
Supports both .py and .ipynb files with direct Jupyter execution.
Monitors completion flags instead of exit codes for reliable restart decisions.
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
from typing import Optional, Dict, Any, List, Tuple


# Minimal HTML template for crash alerts only
HTML_TEMPLATE = """<html><body style="font-family:Arial,sans-serif;color:#333;padding:20px"><div style="max-width:600px;margin:auto;background:#fff;padding:20px;border:1px solid #ddd"><h2 style="color:#d9534f">Process Crash Alert</h2><p>Process "<strong>{title}</strong>" (PID <strong>{pid}</strong>) has crashed.</p><p>Time: {timestamp}</p></div></body></html>"""

# Minimal monitoring script focused on crash detection only
MONITOR_SCRIPT = """import os,sys,time,psutil,json
sys.path.insert(0,r"{cwd}")

with open(r"{pid_file}", "w") as f:
    f.write(str(os.getpid()))

def send_crash_alert(pid, title, recipients, credentials):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    print(f"CRASH ALERT: {{title}} (PID {{pid}}) at {{timestamp}}")
    
    try:
        from araras.email.utils import send_email
        send_email(
            f"{{title}} (PID {{pid}}) crashed", 
            "{html}".format(title=title, pid=pid, timestamp=timestamp),
            recipients, credentials, "html"
        )
        print("Crash alert email sent")
    except Exception as e:
        print(f"Email failed: {{e}}")
    
    with open(r"{restart_file}", "w") as f:
        json.dump({{"crashed": True, "timestamp": timestamp}}, f)
    
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
    if count % 10 == 0 and os.path.exists(r"{stop_file}"):
        try: os.unlink(r"{pid_file}")
        except: pass
        break
    
    count += 1
    
    try:
        if not proc.is_running():
            send_crash_alert({pid}, {title}, r"{recipients}", r"{credentials}")
        
        # Only check for zombie/stopped states that indicate crashes
        status = proc.status()
        if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_STOPPED, psutil.STATUS_DEAD]:
            send_crash_alert({pid}, {title}, r"{recipients}", r"{credentials}")
            
    except psutil.NoSuchProcess:
        send_crash_alert({pid}, {title}, r"{recipients}", r"{credentials}")
    except Exception:
        send_crash_alert({pid}, {title}, r"{recipients}", r"{credentials}")
    
    time.sleep({interval})

print("Monitor completed")"""


class FileTypeHandler:
    """Efficient file type detection and command generation with caching."""

    # Class-level cache for performance optimization
    _file_type_cache: Dict[str, str] = {}
    _command_cache: Dict[str, List[str]] = {}

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
        if len(cls._file_type_cache) < 100:
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
            # Direct Jupyter execution with runipy/papermill alternative
            # Create a Python wrapper script that executes the notebook
            wrapper_script = cls._create_notebook_wrapper(file_path, success_flag_file)
            command = [sys.executable, "-u", wrapper_script]

        else:
            raise ValueError(f"Unsupported file type: {file_type} for {file_path}")

        # Cache command with bounded size to prevent memory growth
        if len(cls._command_cache) < 100:
            cls._command_cache[cache_key] = command.copy()

        return command, file_type

    @classmethod
    def _create_notebook_wrapper(cls, notebook_path: Path, success_flag_file: str) -> str:
        """Create a Python wrapper script for notebook execution.

        Args:
            notebook_path: Path to notebook file
            success_flag_file: Path to success flag file

        Returns:
            Path to created wrapper script
        """
        # Create temporary wrapper script
        wrapper_fd, wrapper_path = tempfile.mkstemp(suffix="_notebook_wrapper.py")

        # Efficient notebook execution wrapper using nbformat and exec
        wrapper_content = f'''
import json
import sys
from pathlib import Path

def execute_notebook():
    """Execute notebook cells and handle success flag."""
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        
        # Load notebook
        with open(r"{notebook_path}", 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Configure execution preprocessor
        ep = ExecutePreprocessor(
            timeout=None,  # No timeout for long-running cells
            kernel_name='python3',
            allow_errors=False  # Stop on first error
        )
        
        # Execute notebook in current working directory
        ep.preprocess(nb, {{'metadata': {{'path': r"{notebook_path.parent}"}}}})
        
        # Write success flag on completion
        Path(r"{success_flag_file}").write_text("SUCCESS")
        print("Notebook executed successfully")
        
    except ImportError as e:
        print(f"Missing dependencies: {{e}}")
        print("Please install: pip install nbformat nbconvert jupyter")
        sys.exit(1)
    except Exception as e:
        print(f"Notebook execution failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    execute_notebook()
'''

        with os.fdopen(wrapper_fd, "w") as f:
            f.write(wrapper_content)

        # Set executable permissions on Unix systems
        if os.name != "nt":
            os.chmod(wrapper_path, 0o755)

        return wrapper_path

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

    @classmethod
    def cleanup_wrapper_scripts(cls) -> None:
        """Clean up temporary wrapper scripts created for notebook execution."""
        # Find and remove temporary wrapper scripts
        temp_dir = Path(tempfile.gettempdir())
        for wrapper_file in temp_dir.glob("*_notebook_wrapper.py"):
            try:
                wrapper_file.unlink()
            except OSError:
                pass  # Ignore cleanup errors


class SimpleTerminalLauncher:
    """Minimal terminal launcher for cross-platform execution."""

    __slots__ = ("system",)  # Memory optimization

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


def start_monitor(pid: int, title: str) -> Dict[str, Any]:
    """Start minimal crash monitor with optimized resource usage.

    Args:
        pid: Process ID to monitor
        title: Process title for alerts

    Returns:
        Monitor control info dictionary

    Raises:
        ValueError: If PID doesn't exist
        OSError: If monitor startup fails
    """
    if not psutil.pid_exists(pid):
        raise ValueError(f"Process PID {pid} not found")

    # Check for email config files (single batch check)
    recipients_file = "./json/recipients.json"
    credentials_file = "./json/credentials.json"

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

    # Generate minimal monitoring script using string formatting for performance
    script_content = MONITOR_SCRIPT.format(
        cwd=os.getcwd(),
        pid=pid,
        interval=2,
        title=repr(title),
        recipients=recipients_file,
        credentials=credentials_file,
        html=HTML_TEMPLATE.replace('"', '\\"').replace("\n", "\\n"),
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


def check_crash_signal(monitor_info: Dict[str, Any]) -> bool:
    """Check if process crashed with minimal I/O operations.

    Args:
        monitor_info: Monitor control info

    Returns:
        True if process crashed, False otherwise
    """
    restart_file = monitor_info.get("restart_file")
    if not restart_file or not os.path.exists(restart_file):
        return False

    try:
        with open(restart_file) as f:
            data = json.load(f)
            return data.get("crashed", False)
    except:
        return False


class FlagBasedRestartManager:
    """Efficient restart manager using completion flags with multi-file support."""

    __slots__ = (
        "max_restarts",
        "restart_delay",
        "restart_count",
        "running",
        "current_terminal_process",
        "current_target_pid",
        "monitor_info",
        "wrapper_scripts",  # Track wrapper scripts for cleanup
    )  # Memory optimization using slots

    def __init__(self, max_restarts: int = 10, restart_delay: float = 3.0):
        """Initialize restart manager with optimized defaults.

        Args:
            max_restarts: Maximum restart attempts
            restart_delay: Delay between restarts in seconds
        """
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.restart_count = 0
        self.running = False

        # Process tracking with minimal state
        self.current_terminal_process: Optional[subprocess.Popen] = None
        self.current_target_pid: Optional[int] = None
        self.monitor_info: Optional[Dict[str, Any]] = None
        self.wrapper_scripts: List[str] = []  # Track for cleanup

    def run_file_with_restart(
        self, file_path: str, success_flag_file: str, title: Optional[str] = None
    ) -> None:
        """Run file with flag-based restart logic supporting .py and .ipynb files.

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

        working_dir = str(validated_path.parent)
        monitor_title = title or validated_path.stem
        flag_path = Path(success_flag_file).resolve()

        print(f"Starting flag-based auto-restart for: {validated_path}")
        print(f"File type: {file_type}")
        print(f"Success flag file: {flag_path}")
        print(f"Max restarts: {self.max_restarts}")

        self.running = True

        try:
            # Main restart loop with optimized control flow
            while self.running and self.restart_count < self.max_restarts:
                # Remove old success flag (atomic operation)
                if flag_path.exists():
                    flag_path.unlink()

                start_time = time.time()

                try:
                    # Launch process with file type awareness
                    target_pid = self._launch_process(validated_path, working_dir, success_flag_file)
                    print(f"Process started: PID {target_pid}")

                    # Start crash monitor
                    self.monitor_info = start_monitor(target_pid, monitor_title)

                    # Wait for completion or crash with optimized polling
                    completion_reason = self._wait_for_completion(flag_path)
                    runtime = time.time() - start_time

                    print(f"Process finished: {completion_reason}, runtime: {runtime:.1f}s")

                    # Immediate cleanup for memory efficiency
                    self._cleanup_all()

                    # Smart decision logic based on completion reason
                    if completion_reason == "success_flag":
                        print("Process completed successfully")
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

            # Final status report
            if self.restart_count >= self.max_restarts:
                print(f"Maximum restarts reached: {self.max_restarts}")

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self._cleanup_all()
            self._cleanup_wrapper_scripts()  # Clean up notebook wrappers
            print(f"Total restarts: {self.restart_count}")

    def _launch_process(self, file_path: Path, working_dir: str, success_flag_file: str) -> int:
        """Launch target process with file type awareness.

        Args:
            file_path: Validated path to file
            working_dir: Working directory
            success_flag_file: Success flag file path

        Returns:
            Target process PID

        Raises:
            OSError: If PID discovery fails
        """
        # Build command based on file type
        command, execution_type = FileTypeHandler.build_execution_command(file_path, success_flag_file)

        # Track wrapper scripts for cleanup
        if execution_type == "notebook" and len(command) > 2:
            wrapper_script = command[2]  # Third element is wrapper script path
            self.wrapper_scripts.append(wrapper_script)
            print(f"Executing notebook via wrapper: {Path(wrapper_script).name}")

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
                if check_crash_signal(self.monitor_info):
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
            # Exponential backoff with reasonable upper limit
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

    def _cleanup_wrapper_scripts(self) -> None:
        """Clean up notebook wrapper scripts to prevent disk clutter."""
        for wrapper_script in self.wrapper_scripts:
            try:
                if os.path.exists(wrapper_script):
                    os.unlink(wrapper_script)
            except:
                pass
        self.wrapper_scripts.clear()

    def _sleep(self, duration: float) -> None:
        """Interruptible sleep with minimal CPU usage.

        Args:
            duration: Sleep duration in seconds
        """
        end_time = time.time() + duration
        while self.running and time.time() < end_time:
            time.sleep(min(0.1, end_time - time.time()))


def run_auto_restart(
    file_path: str,
    success_flag_file: str = "/tmp/success.flag",
    title: Optional[str] = None,
    max_restarts: int = 10,
    restart_delay: float = 3.0,
) -> None:
    """Main function accepting parameters directly for programmatic use.

    Args:
        file_path: Path to .py or .ipynb file to execute
        success_flag_file: Path to success flag file
        title: Custom title for monitoring
        max_restarts: Maximum restart attempts
        restart_delay: Delay between restarts in seconds

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
        ImportError: If notebook dependencies missing for .ipynb files
    """
    try:
        # Validate file early with detailed error messages
        validated_path = FileTypeHandler.validate_file(file_path)
        file_type = FileTypeHandler.get_file_type(validated_path)

        print(f"Validated file: {validated_path} (type: {file_type})")

        # Check notebook dependencies if needed
        if file_type == "notebook":
            try:
                import nbformat, nbconvert

                print("Notebook dependencies found")
            except ImportError as e:
                raise ImportError(
                    "Missing notebook dependencies. " "Please install: pip install nbformat nbconvert jupyter"
                ) from e

        # Create optimized restart manager
        manager = FlagBasedRestartManager(max_restarts=max_restarts, restart_delay=restart_delay)

        manager.run_file_with_restart(file_path=file_path, success_flag_file=success_flag_file, title=title)

    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
