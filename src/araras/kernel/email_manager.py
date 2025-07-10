from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from araras.email.utils import send_email
from .constants import (
    COMPLETION_DETAILS_TEMPLATE,
    CONSOLIDATED_STATUS_TEMPLATE,
    FAILURE_DETAILS_TEMPLATE,
    RESTART_DETAILS_TEMPLATE,
)
from .prints import print_error_message, print_warning_message

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
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
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

            send_email(
                subject,
                html_content,
                self.recipients_file,
                self.credentials_file,
                "html",
            )
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
        self,
        title: str,
        old_pid: Optional[int],
        new_pid: int,
        restart_count: int,
        runtime: float,
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
            "task_complete",
            {
                "title": title,
                "restart_count": restart_count,
                "total_runtime": total_runtime,
            },
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


