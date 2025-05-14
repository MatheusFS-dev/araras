"""
This module provides functionality to monitor a process by its PID.
It checks the process status at regular intervals and sends an email alert if the process crashes or terminates.

Functions:
    - start_monitor: Starts a background process to monitor the specified PID.
    - stop_monitor: Stops the monitoring process.
    - _send_alert_email: Sends an email alert when the monitored process crashes or terminates.

Example usage:
    from araras.kernel.monitoring import start_monitor, stop_monitor

    # Start monitoring a process with PID 1234
    monitor_process = start_monitor(1234)

    # Stop monitoring the process
    stop_monitor(monitor_process)
"""

import time
import psutil
from multiprocessing import Process
from typing import *
from araras.email.utils import send_email


def _send_alert_email(
    pid: int,
    status: str,
    custom_title: Optional[str],
    recipients_file: str,
    credentials_file: str,
) -> None:
    """
    Send an email alert about the status of a monitored process.

    The email provides a styled HTML message indicating the process ID and status.

    Args:
        pid (int): Process ID of the monitored process.
        status (str): Status string, either 'crashed' or 'terminated'.
        custom_title (Optional[str]): Optional title for the process; used in the subject and body.
        recipients_file (str): Path to the JSON file containing recipient email addresses.
        credentials_file (str): Path to the JSON file with email credentials.

    Returns:
        None

    Flow:
        -> Build subject and body using `pid`, `status`, and `custom_title`
        -> Use `send_email()` utility to send HTML email
        -> If email sending fails, print error to stdout
    """
    # Fallback to a default title if no custom title is provided
    title = custom_title or "Kernel process"

    # Construct the subject line of the email
    subject = f"{title} (PID {pid}) {status.capitalize()}"

    # Construct the body of the email in HTML format
    body = f"""\
<html>
  <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;
               background-color: #f9f9f9; padding: 20px;">
    <div style="max-width: 600px; margin: auto; background: #fff; padding: 20px;
                border: 1px solid #ddd; border-radius: 8px;">
      <h2 style="color: #d9534f; text-align: center;">
        ⚠️ {title} Alert: {status.capitalize()}
      </h2>
      <p>The process "<strong>{title}</strong>" (PID <strong>{pid}</strong>) has {status}.</p>
      <p>Please check the system logs and take necessary actions.</p>
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
        # Attempt to send the email using external utility
        send_email(
            subject=subject,
            body=body,
            recipients_file=recipients_file,
            credentials_file=credentials_file,
            text_type="html",  # Specify that the email is in HTML format
        )
    except Exception as e:
        # Print error message if email fails to send
        print(f"[ERROR] Email send failed: {e}")


def _monitor_worker(
    pid: int,
    interval: int,
    custom_title: Optional[str],
    recipients_file: str,
    credentials_file: str,
) -> None:
    """
    Internal worker function that continuously checks the status of a given process.

    If the process is not running or has become a zombie, an alert email is sent and the loop terminates.

    Args:
        pid (int): PID of the process to monitor.
        interval (int): Time interval (in seconds) between checks.
        custom_title (Optional[str]): Optional title for the process in email.
        recipients_file (str): Path to the JSON file with email recipients.
        credentials_file (str): Path to the JSON file with email credentials.

    Returns:
        None

    Flow:
        -> Loop indefinitely
            -> Try to get process by PID
            -> If process is not running or is a zombie:
                -> Send "crashed" alert email
                -> Break loop
            -> If process no longer exists:
                -> Send "terminated" alert email
                -> Break loop
            -> Sleep for given interval
    """
    while True:
        try:
            # Try to create a Process object from the PID
            proc = psutil.Process(pid)

            # Check if process is not running or has become a zombie
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                # Send alert that the process has crashed
                _send_alert_email(pid, "crashed", custom_title, recipients_file, credentials_file)
                break  # Exit the loop after alert

        except psutil.NoSuchProcess:
            # Handle case where the process no longer exists
            _send_alert_email(pid, "terminated", custom_title, recipients_file, credentials_file)
            break  # Exit the loop after alert

        # Sleep for the specified interval before next check
        time.sleep(interval)


def start_monitor(
    pid: int,
    interval: int = 10,
    custom_title: Optional[str] = None,
    recipients_file: str = "./json/recipients.json",
    credentials_file: str = "./json/credentials.json",
) -> Process:
    """
    Start a background process to monitor the status of a given PID.

    Args:
        pid (int): Process ID to monitor.
        interval (int): Interval (in seconds) between status checks. Default is 10.
        custom_title (Optional[str]): Optional title for the monitored process.
        recipients_file (str): Path to recipients JSON file.
        credentials_file (str): Path to credentials JSON file.

    Returns:
        Process: A multiprocessing.Process object that can be used to control the monitor.

    Flow:
        -> Create a new Process with `_monitor_worker` as target
        -> Start the Process
        -> Return the Process object to the caller
    """
    # Initialize a background process for monitoring
    p = Process(
        target=_monitor_worker,
        args=(pid, interval, custom_title, recipients_file, credentials_file),
        daemon=False,  # Ensures the watcher continues even if parent exits
    )

    # Start the process
    p.start()

    # Return the process object to allow external control
    return p


def stop_monitor(process: Process) -> None:
    """
    Stop a previously started monitoring process.

    Args:
        process (Process): A multiprocessing.Process object returned from `start_monitor`.

    Returns:
        None

    Flow:
        -> Call terminate() to stop the process
        -> Call join() to wait for its termination
    """
    # Request the process to terminate
    process.terminate()

    # Wait for the process to finish termination
    process.join()
