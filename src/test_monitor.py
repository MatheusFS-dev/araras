# test_monitor.py

import multiprocessing
import os
import time

# 1. mp.spawn on Windows requires this guard:
if __name__ == "__main__":
    multiprocessing.freeze_support()

    from araras.kernel.monitoring import start_monitor

    print("▶️  Starting monitor…")
    pid = os.getpid()

    monitor_proc = start_monitor(
        pid=pid,
        interval=1,
        custom_title="Test Process",
        recipients_file="./json/recipients.json",
        credentials_file="./json/credentials.json",
    )
    print(f"✔️  Monitor launched (watching PID={pid}, watcher PID={monitor_proc.pid})")

    # 2. let the monitor run alongside your “work”
    time.sleep(5)

    # Kill the process to simulate a crash
    print("❌  Simulating crash…")
    os.kill(pid, 9)  # SIGKILL
