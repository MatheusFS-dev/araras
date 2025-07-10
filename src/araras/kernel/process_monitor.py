from __future__ import annotations

import json
import os
import sys
import glob
import tempfile
import time
from typing import Any, Dict

import psutil

from araras.utils.terminal import SimpleTerminalLauncher
from .constants import MONITOR_SCRIPT


def _cleanup_stale_monitor_files() -> None:
    tmpdir = tempfile.gettempdir()
    for path in glob.glob(os.path.join(tmpdir, "*_monitor.*")):
        try:
            os.unlink(path)
        except OSError:
            pass


def start_monitor(pid: int, title: str, supress_tf_warnings: bool = False) -> Dict[str, Any]:
    """Start simplified crash monitor."""
    _cleanup_stale_monitor_files()
    time.sleep(0.1)

    if not psutil.pid_exists(pid):
        raise ValueError(f"Process PID {pid} not found")

    fd, script_path = tempfile.mkstemp(suffix="_monitor.py")
    base_path = script_path.replace(".py", "")

    control_files = {
        "script_path": script_path,
        "pid_file": f"{base_path}.pid",
        "stop_file": f"{base_path}.stop",
        "restart_file": f"{base_path}.restart",
    }

    script_content = MONITOR_SCRIPT.format(
        cwd=os.getcwd(),
        pid=pid,
        interval=2,
        title=repr(title),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        **control_files,
    )

    with os.fdopen(fd, "w") as f:
        f.write(script_content)

    if os.name != "nt":
        os.chmod(script_path, 0o755)

    launcher = SimpleTerminalLauncher()
    launcher.set_supress_tf_warnings(supress_tf_warnings)
    process = launcher.launch([sys.executable, script_path], os.getcwd())

    time.sleep(0.1)
    if process.poll() is not None:
        exit_code = process.returncode
        error_msg = f"Monitor failed to start (exit code: {exit_code})"
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stderr:
                error_msg += f". Error output: {stderr.decode().strip()}"
            elif stdout:
                error_msg += f". Output: {stdout.decode().strip()}"
        except Exception:
            pass
        try:
            os.unlink(script_path)
        except Exception:
            pass
        raise OSError(error_msg)

    return {"process": process, **control_files}


def stop_monitor(monitor_info: Dict[str, Any]) -> None:
    """Stop monitor and cleanup files."""
    if not monitor_info:
        return

    try:
        with open(monitor_info["stop_file"], "w") as f:
            f.write("STOP")
    except Exception:
        pass

    for _ in range(20):
        if not os.path.exists(monitor_info["pid_file"]):
            break
        time.sleep(0.1)

    process = monitor_info.get("process")
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            pass

    cleanup_files = ["script_path", "pid_file", "stop_file", "restart_file"]
    for file_key in cleanup_files:
        try:
            file_path = monitor_info.get(file_key)
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass


def check_crash_signal(monitor_info: Dict[str, Any]) -> Dict[str, Any]:
    """Check if process crashed."""
    restart_file = monitor_info.get("restart_file")
    if not restart_file or not os.path.exists(restart_file):
        return {}

    try:
        with open(restart_file) as f:
            data = json.load(f)
            if data.get("crashed", False):
                return data
    except Exception:
        pass

    return {}
