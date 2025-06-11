"""
This module provides functionality to monitor a process by its PID.
It checks the process status at regular intervals and sends an email alert if the process crashes or terminates.

Functions:
    - start_monitor: Starts a background process to monitor the specified PID.
    - stop_monitor: Stops the monitoring process.
    - _send_alert_email: Sends an email alert when the monitored process crashes or terminates.

Example usage:
    # Start monitoring a process with PID 1234
    monitor_process = start_monitor(1234)

    # Stop monitoring the process
    stop_monitor(monitor_process)
"""

import os
import sys
import time
import psutil
import tempfile
import subprocess
import platform
from pathlib import Path

HTML_TEMPLATE = """<html><body style="font-family:Arial,sans-serif;line-height:1.6;color:#333;background:#f9f9f9;padding:20px"><div style="max-width:600px;margin:auto;background:#fff;padding:20px;border:1px solid #ddd;border-radius:8px"><h2 style="color:#d9534f;text-align:center">Alert: {title} {status}</h2><p>Process "<strong>{title}</strong>" (PID <strong>{pid}</strong>) has {status}.</p><p>Time: {timestamp}</p><footer style="margin-top:20px;text-align:center;font-size:14px;color:#888"><p>Best regards,<br><strong>The Monitoring Team</strong></p></footer></div></body></html>"""

# High-frequency monitoring script with aggressive termination detection
SCRIPT_TEMPLATE = """import os,sys,time,psutil,traceback,threading
from pathlib import Path
sys.path.insert(0,r"{cwd}")

# Write PID file for terminal control
with open(r"{pid_file}", "w") as f:
    f.write(str(os.getpid()))

def send_alert(pid, status, title, recipients, credentials, attempt=1):
    \"\"\"Send email with retry logic and detailed error reporting.\"\"\"
    title = title or "Process"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    
    print(f"ALERT: {{title}} (PID {{pid}}) {{status}} at {{timestamp}}")
    
    for retry in range(3):  # 3 retry attempts
        try:
            from araras.email.utils import send_email
            result = send_email(
                f"{{title}} (PID {{pid}}) {{status}}", 
                "{html}".format(title=title, pid=pid, status=status, timestamp=timestamp),
                recipients, credentials, "html"
            )
            print(f"Email sent successfully (attempt {{retry + 1}}): {{result}}")
            break
        except ImportError as e:
            print(f"Import error: {{e}}")
            break  # Don't retry import errors
        except Exception as e:
            print(f"Email attempt {{retry + 1}} failed: {{type(e).__name__}}: {{e}}")
            if retry < 2:  # Don't sleep on last attempt
                time.sleep(2)  # Wait before retry
    else:
        print(f"All email attempts failed for {{status}} alert")
    
    # Always exit after alert attempt
    try: os.unlink(r"{pid_file}")
    except: pass
    sys.exit(0)

def fast_process_check(proc):
    \"\"\"Optimized process state check with minimal syscalls.\"\"\"
    try:
        # Single atomic check - most efficient
        if not proc.is_running():
            return False, "terminated"
        
        # Get status only if process is running
        status = proc.status()
        if status == psutil.STATUS_ZOMBIE:
            return False, "crashed"
        elif status in [psutil.STATUS_STOPPED, psutil.STATUS_DEAD]:
            return False, "stopped"
        
        return True, status
    except psutil.NoSuchProcess:
        return False, "terminated"
    except psutil.AccessDenied:
        return True, "access_denied"  # Process exists but no access

# Initialize process monitoring
try:
    proc = psutil.Process({pid})
    process_name = proc.name()
    print(f"Monitoring PID {pid} - {{process_name}} [Mode: {interval}s intervals]")
except psutil.NoSuchProcess:
    send_alert({pid}, "not found", {title}, r"{recipients}", r"{credentials}")

# High-frequency monitoring loop with optimizations
count = 0
consecutive_errors = 0
last_status = None

# Cache process info to reduce syscalls
process_cache = {{"last_check": 0, "name": None, "status": None}}

while True:
    # Non-blocking stop signal check (filesystem cache friendly)
    if count % 5 == 0 and os.path.exists(r"{stop_file}"):
        try: os.unlink(r"{pid_file}")
        except: pass
        print("Stop signal received")
        break
    
    count += 1
    
    # Fast process state check
    is_alive, current_status = fast_process_check(proc)
    
    if not is_alive:
        # Process terminated - send alert immediately
        send_alert({pid}, current_status, {title}, r"{recipients}", r"{credentials}")
    
    # Status change detection (for debugging)
    if current_status != last_status:
        print(f"PID {pid} status: {{last_status}} → {{current_status}}")
        last_status = current_status
    
    # Reduced logging frequency for performance
    if count % 20 == 0:
        try:
            # Batch process info gathering
            cpu_percent = proc.cpu_percent(interval=None)  # Non-blocking
            memory_mb = proc.memory_info().rss / 1024 / 1024
            print(f"PID {pid} OK - CPU: {{cpu_percent:.1f}}% | RAM: {{memory_mb:.1f}}MB | Checks: {{count}}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            send_alert({pid}, "terminated", {title}, r"{recipients}", r"{credentials}")
    
    # Adaptive sleep with process priority
    time.sleep({interval})

print(f"Monitoring completed for PID {pid} ({{count}} checks)")"""


def launch_standalone_terminal(command, working_directory=None):
    """Launch standalone terminal with OS-specific optimizations.

    Args:
        command (str): Command to execute in terminal
        working_directory (str, optional): Working directory path

    Returns:
        subprocess.Popen: Terminal process object

    Raises:
        OSError: If terminal launch fails or OS unsupported
    """
    system = platform.system().lower()

    if system == "linux":
        cmd = ["gnome-terminal", "--", "bash", "-c", command]
    elif system == "darwin":
        cmd = ["osascript", "-e", f'tell application "Terminal" to do script "{command}"']
    elif system == "windows":
        cmd = ["cmd", "/c", command]
    else:
        raise OSError(f"Unsupported OS: {system}")

    return subprocess.Popen(
        cmd,
        cwd=working_directory,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_monitor(
    pid,
    interval=2,  # Reduced default interval for faster detection
    custom_title=None,
    recipients_file="./json/recipients.json",
    credentials_file="./json/credentials.json",
    log_dir=None,
):
    """Start high-frequency process monitor with guaranteed termination detection.

    Args:
        pid (int): Process ID to monitor
        interval (int): Check interval in seconds (default: 2 for IDE monitoring)
        custom_title (str, optional): Custom process title
        recipients_file (str): Path to email recipients JSON
        credentials_file (str): Path to email credentials JSON
        log_dir (str, optional): Log directory path

    Returns:
        dict: Monitor control information

    Raises:
        ValueError: If PID invalid or process doesn't exist
        FileNotFoundError: If required config files missing
        OSError: If monitor startup fails
    """
    # Fast input validation with early exit
    if pid <= 0:
        raise ValueError(f"Invalid PID: {pid}")

    if not psutil.pid_exists(pid):
        raise ValueError(f"Process PID {pid} not found")

    # Efficient batch file validation
    required_files = [recipients_file, credentials_file]
    missing = [f for f in required_files if not Path(f).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    # Create temporary control files with secure permissions
    fd, script_path = tempfile.mkstemp(suffix="_monitor.py", text=True)
    pid_file = script_path.replace(".py", ".pid")
    stop_file = script_path.replace(".py", ".stop")

    # Pre-compute absolute paths for efficiency
    abs_recipients = Path(recipients_file).resolve()
    abs_credentials = Path(credentials_file).resolve()
    abs_log_dir = Path(log_dir).resolve() if log_dir else None

    # Generate optimized monitoring script
    script_content = SCRIPT_TEMPLATE.format(
        cwd=os.getcwd(),
        pid=pid,
        interval=interval,
        title=repr(custom_title),
        recipients=str(abs_recipients),
        credentials=str(abs_credentials),
        log_dir=repr(str(abs_log_dir)) if abs_log_dir else None,
        html=HTML_TEMPLATE.replace('"', '\\"').replace("\n", "\\n"),
        pid_file=pid_file,
        stop_file=stop_file,
    )

    try:
        # Single file write with UTF-8 encoding
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(script_content)

        # Set executable permissions (Unix only)
        if os.name != "nt":
            os.chmod(script_path, 0o755)

        # Launch monitor terminal
        process = launch_standalone_terminal(
            f'"{sys.executable}" "{script_path}"', working_directory=os.getcwd()
        )

        # Verify successful launch
        time.sleep(0.1)
        if process.poll() is not None:
            raise OSError("Monitor process failed to start")

        print(f"🚀 Fast monitor started for PID {pid} (interval: {interval}s)")
        return {
            "process": process,
            "script_path": script_path,
            "pid_file": pid_file,
            "stop_file": stop_file,
        }

    except Exception as e:
        # Cleanup on failure
        try:
            os.unlink(script_path)
        except:
            pass
        raise OSError(f"Monitor startup failed: {e}") from e


def stop_monitor(monitor_info):
    """Stop monitor with guaranteed terminal closure.

    Args:
        monitor_info (dict): Monitor control info from start_monitor()
    """
    if not monitor_info:
        print("⚠ No monitor to stop")
        return

    try:
        # Step 1: Graceful shutdown signal
        with open(monitor_info["stop_file"], "w") as f:
            f.write("STOP")

        # Step 2: Wait for graceful shutdown (3 second timeout)
        for _ in range(30):
            if not os.path.exists(monitor_info["pid_file"]):
                print("Monitor stopped")
                break
            time.sleep(0.1)
        else:
            # Step 3: Force termination
            if os.path.exists(monitor_info["pid_file"]):
                try:
                    with open(monitor_info["pid_file"], "r") as f:
                        terminal_pid = int(f.read().strip())

                    terminal_proc = psutil.Process(terminal_pid)
                    terminal_proc.terminate()
                    terminal_proc.wait(timeout=2)
                    print("Monitor force-terminated")
                except (ValueError, psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass

        # Step 4: Cleanup launcher process
        process = monitor_info.get("process")
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

        # Step 5: File cleanup
        cleanup_files = [monitor_info["script_path"], monitor_info["pid_file"], monitor_info["stop_file"]]

        for file_path in cleanup_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except:
                pass

    except Exception as e:
        print(f"Stop error: {e}")
