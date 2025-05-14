"""
This script reads a PID and monitors whether the corresponding
process is still running. If the process dies, an alert is logged and an
email notification is sent. You can optionally supply a custom title
to identify the monitored process.

Functions:
    - send_alert_email: Sends an email notification about the process status.
    - monitor_process: Monitors a running process by checking if the PID exists.
    - parse_args: Parses command-line arguments.

Example usage:
    python _monitor_kernel_life.py --pid 12345 --interval 10

#! Move this script to the base directory where the code you want to monitor is located.
"""

import time
import psutil
import argparse
from typing import Optional
from araras.email.utils import send_email

def send_alert_email(pid: int, status: str, custom_title: Optional[str] = None) -> None:
    """
    Sends a styled email notification reporting a process crash or termination.

    Logic:
        -> Import `send_email` utility (with error fallback)
        -> Construct email subject and body with process details
        -> Send the email using configured sender and recipient files

    Args:
        pid (int): The process ID that was being monitored.
        status (str): Status of the process. Expected values are "crashed" or "terminated".
        custom_title (Optional[str]): A descriptive name for the process; defaults to "Kernel process".

    Returns:
        None

    Example:
        send_alert_email(pid=1234, status="crashed", custom_title="Data Ingestion Job")
    """
    # Determine fallback title if no custom title is provided
    process_title = custom_title or "Kernel process"

    # Construct email subject line
    email_subject = f"{process_title} (PID {pid}) {status.capitalize()}"

    # Construct rich HTML-formatted body
    email_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;
                     background-color: #f9f9f9; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background: #fff; padding: 20px;
                    border: 1px solid #ddd; border-radius: 8px;">
            <h2 style="color: #d9534f; text-align: center;">
                ⚠️ {process_title} Alert: {status.capitalize()}
            </h2>
            <p style="font-size: 16px; color: #444;">
                <strong>Dear User,</strong>
            </p>
            <p style="font-size: 18px; color: #333;">
                The process "<strong>{process_title}</strong>" with PID
                <strong style="color: #d9534f;">{pid}</strong> has {status}.
            </p>
            <p style="font-size: 16px; color: #555;">
                Please check the system logs and take necessary actions to resolve
                the issue.
            </p>
            <p style="text-align: center; font-size: 16px;">
                <strong style="color: #d9534f;">⚠️ Status:</strong>
                <span style="color: #d9534f;">{status.capitalize()}</span>
            </p>
            <footer style="margin-top: 20px; text-align: center; font-size: 14px;
                           color: #888;">
                <p>Best regards,</p>
                <p><strong>The Monitoring Team</strong></p>
            </footer>
        </div>
        </body>
    </html>
    """

    try:
        # Attempt to send the email using utility function
        send_email(
            subject=email_subject,
            body=email_body,
            recipients_file="./json/recipients.json",
            credentials_file="./json/credentials.json",
            text_type="html",
        )
        print(f"[INFO] Email sent: {process_title} (PID {pid}) {status}.")
    except Exception as e:
        # Print error if sending fails
        print(f"[ERROR] Failed to send email for {process_title} (PID {pid}): {e}")


def monitor_process(pid: int, check_interval: int = 10, custom_title: Optional[str] = None) -> None:
    """
    Continuously monitors a process by PID and triggers an alert if it crashes or terminates.

    Logic:
        -> Print monitoring startup message
        -> Loop:
            -> Attempt to locate process by PID
            -> If not running or zombie → alert (crash)
            -> If not found → alert (terminated)
            -> Else → sleep and recheck

    Args:
        pid (int): The process ID to monitor.
        check_interval (int): Interval (in seconds) between each health check.
        custom_title (Optional[str]): Optional descriptive label for the process.

    Returns:
        None

    Example:
        monitor_process(pid=4567, check_interval=5, custom_title="Data ETL Worker")
    """
    process_title = custom_title or "Kernel process"
    print(f"[INFO] Monitoring '{process_title}' with PID {pid}...")

    while True:
        try:
            # Attempt to access the process
            proc = psutil.Process(pid)

            # Check if process is running or has become a zombie
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                print(f"[ALERT] {process_title} (PID {pid}) has crashed!")
                send_alert_email(pid, "crashed", custom_title)
                break

        except psutil.NoSuchProcess:
            # If process no longer exists, trigger alert for termination
            print(f"[ALERT] {process_title} (PID {pid}) no longer exists!")
            send_alert_email(pid, "terminated", custom_title)
            break

        # Wait before checking again
        time.sleep(check_interval)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for PID monitoring.

    Logic:
        -> Define expected arguments
        -> Parse them using argparse
        -> Return the parsed object

    Returns:
        argparse.Namespace: Object containing parsed command-line arguments.

    Example:
        args = parse_args()
        monitor_process(pid=args.pid, ...)
    """
    parser = argparse.ArgumentParser(description="Monitor a process by PID and send alerts if it stops.")
    parser.add_argument("--pid", type=int, required=True, help="Process ID to monitor.")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in seconds (default: 10).")
    parser.add_argument(
        "--custom-title", type=str, default=None, help="Optional custom name for the monitored process."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Entry point: parse arguments and start monitoring
    args = parse_args()
    monitor_process(pid=args.pid, check_interval=args.interval, custom_title=args.custom_title)
