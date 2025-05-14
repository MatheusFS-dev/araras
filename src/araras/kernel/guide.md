
# Process Monitoring with `monitoring.py`

This document explains how to import, configure, and use the `monitoring.py` module to watch a running process by PID, send an email alert if it crashes or terminates, and cleanly stop the watcher. It also shows how to simulate a crash on Windows using a test script.

---

## 1. Requirements

- Python 3.7+  
- `psutil`  
- Your project’s `araras.email.utils.send_email` configured to read:
  - `recipients_file` (JSON list of recipient addresses)
  - `credentials_file` (JSON with SMTP credentials)

Install psutil if needed:

```bash
pip install psutil
````

---

## 2. `monitoring.py` API

Place `monitoring.py` in your project (e.g. under `araras/kernel/monitoring.py`).

```python
from multiprocessing import Process
from typing import Optional

def start_monitor(
    pid: int,
    interval: int = 10,
    custom_title: Optional[str] = None,
    recipients_file: str = "./json/recipients.json",
    credentials_file: str = "./json/credentials.json",
) -> Process:
    """
    Start a background watcher on `pid`, polling every `interval` seconds.
    Returns the `multiprocessing.Process` instance.
    """

def stop_monitor(process: Process) -> None:
    """
    Stop the monitoring process started by `start_monitor`.
    """
```

* **`pid`**: the process ID to watch.
* **`interval`**: seconds between checks.
* **`custom_title`**: optional name for email subject/body.
* **`recipients_file`** & **`credentials_file`**: paths to your email configuration JSON files.

---

## 3. Example in `main.py`

```python
import os
import time
from araras.kernel.monitoring import start_monitor, stop_monitor

if __name__ == "__main__":
    RUN_DIR = os.getcwd()
    pid = os.getpid()

    # 1. Start the monitor (non-daemon → survives parent crash)
    watcher = start_monitor(
        pid=pid,
        interval=5,
        custom_title="MyApp Main Process",
        recipients_file="./json/recipients.json",
        credentials_file="./json/credentials.json",
    )
    print(f"[INFO] Monitoring started for PID={pid} (watcher PID={watcher.pid})")

    # 2. Do your work
    time.sleep(30)  # replace with your real workload

    # 3. Clean up on normal exit
    if watcher.is_alive():
        stop_monitor(watcher)
        print("[INFO] Monitoring stopped cleanly.")
```

---

## 4. Windows “safe import” guard

On Windows, `multiprocessing` uses the **spawn** start method. You **must** protect your top-level script with:

```python
if __name__ == "__main__":
    multiprocessing.freeze_support()   # only needed if you build an .exe; safe to include
    # … your import/start_monitor logic …
```

Without this guard, you’ll see errors like:

```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

---

## 5. Complete `test_monitor.py` Example

Use this script to verify that:

1. The watcher survives a crash of your main script.
2. You receive exactly one “terminated” email when the main process dies.

```python
# test_monitor.py

import multiprocessing
import os
import time

if __name__ == "__main__":
    # Required on Windows to avoid multiprocessing spawn errors
    multiprocessing.freeze_support()

    from araras.kernel.monitoring import start_monitor

    print("▶️  Starting monitor…")
    pid = os.getpid()

    watcher = start_monitor(
        pid=pid,
        interval=1,
        custom_title="Test Process",
        recipients_file="./json/recipients.json",
        credentials_file="./json/credentials.json",
    )
    print(f"✔️  Monitor launched (watching PID={pid}, watcher PID={watcher.pid})")

    # Let the watcher poll a few times
    time.sleep(5)

    # Simulate an unexpected crash
    raise RuntimeError("Simulated crash for testing purposes.")
```

**Run the test**:

```bash
python test_monitor.py
```

* You should see the “▶️ Starting monitor…” and then a traceback.
* The orphaned watcher process will detect the missing PID, send one “terminated” email, and then exit.

---