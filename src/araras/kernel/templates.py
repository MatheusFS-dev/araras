"""Templates used by the monitoring utilities."""

# HTML templates
CONSOLIDATED_STATUS_TEMPLATE = """<html><body style="font-family:Arial,sans-serif;color:#333;padding:20px"><div style="max-width:600px;margin:auto;background:#fff;padding:20px;border:1px solid #ddd"><h2 style="color:{color}">{status_title}</h2><div style="background:#f9f9f9;padding:15px;margin:15px 0;border-left:4px solid {color}"><h3>Process Information</h3><p><strong>Process:</strong> {title}</p><p><strong>Status:</strong> {status_description}</p><p><strong>Timestamp:</strong> {timestamp}</p></div>{details_section}<div style="background:#f0f0f0;padding:10px;margin-top:20px;font-size:12px;color:#666"><p>This is an automated status report from the process monitoring system.</p></div></div></body></html>"""

RESTART_DETAILS_TEMPLATE = """<div style="background:#fff3cd;padding:15px;margin:15px 0;border-left:4px solid #ffc107"><h3>Restart Information</h3><p><strong>Previous PID:</strong> {old_pid}</p><p><strong>New PID:</strong> {new_pid}</p><p><strong>Total Restarts:</strong> {restart_count}</p><p><strong>Runtime Before Restart:</strong> {runtime:.1f}s</p></div>"""

FAILURE_DETAILS_TEMPLATE = """<div style="background:#f8d7da;padding:15px;margin:15px 0;border-left:4px solid #dc3545"><h3>Failure Details</h3><p><strong>Failed Attempts:</strong> {failed_attempts}</p><p><strong>Remaining Attempts:</strong> {remaining_attempts}</p><p><strong>Total Restart Count:</strong> {restart_count}</p><p><strong>Error:</strong> {error}</p></div>"""

COMPLETION_DETAILS_TEMPLATE = """<div style="background:#d4edda;padding:15px;margin:15px 0;border-left:4px solid #28a745"><h3>Completion Summary</h3><p><strong>Total Restarts:</strong> {restart_count}</p><p><strong>Total Runtime:</strong> {total_runtime:.1f}s</p><p><strong>Final Status:</strong> Successfully completed</p></div>"""

# Python monitor script template
MONITOR_SCRIPT = """import os,sys,time,psutil,json
sys.path.insert(0,r"{cwd}")

with open(r"{pid_file}", "w") as f:
    f.write(str(os.getpid()))

def send_crash_signal(pid, title, restart_count=0):
    \"\"\"Send crash signal for restart manager to handle.\"\"\"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())
    print(f"CRASH DETECTED: {title} (PID {pid}) at {timestamp}")

    with open(r"{restart_file}", "w") as f:
        json.dump({{"crashed": True, "timestamp": timestamp, "restart_count": restart_count, "pid": pid}}, f)

    try: os.unlink(r"{pid_file}")
    except: pass
    sys.exit(0)

try:
    proc = psutil.Process({pid})
    print(f"Monitoring PID {pid} for crashes")
except psutil.NoSuchProcess:
    send_crash_signal({pid}, {title})

count = 0
while True:
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

__all__ = [
    "CONSOLIDATED_STATUS_TEMPLATE",
    "RESTART_DETAILS_TEMPLATE",
    "FAILURE_DETAILS_TEMPLATE",
    "COMPLETION_DETAILS_TEMPLATE",
    "MONITOR_SCRIPT",
]
