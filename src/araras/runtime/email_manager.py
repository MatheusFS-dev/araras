"""
Last Edited: 14 July 2025
Description:
    This script demonstrates a detailed 
    header format with additional metadata.
"""
from araras.core import *

import time
from pathlib import Path

from araras.notifications.email import send_email
from . import monitoring as _mon


class ConsolidatedEmailManager:
    """Handles consolidated email notifications with retry logic."""

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
    ) -> None:
        self.recipients_file = recipients_file or "./json/recipients.json"
        self.credentials_file = credentials_file or "./json/credentials.json"
        self.retry_attempts = retry_attempts
        self.email_enabled = self._validate_email_config()
        self.retry_count = 0
        self.last_notification_time = 0

    def _validate_email_config(self) -> bool:
        recipients_exists = Path(self.recipients_file).exists()
        credentials_exists = Path(self.credentials_file).exists()

        if not (recipients_exists and credentials_exists):
            _mon.print_warning_message("Email config files not found, email alerts disabled")
            logger.warning(f"{YELLOW}Expected files: {self.recipients_file}, {self.credentials_file}{RESET}")
            return False

        return True

    def send_consolidated_status_email(self, status_type: str, process_data: Dict[str, Any]) -> None:
        if not self.email_enabled:
            return

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
            title = process_data.get("title", "Unknown Process")

            if status_type == "restart_success":
                subject = f"{title} crashed - Restart Successful"
                color = "#28a745"
                status_title = "Process Restart Successful"
                status_description = "Process crashed but was successfully restarted"
                details_section = _mon.RESTART_DETAILS_TEMPLATE.format(
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
                details_section = _mon.FAILURE_DETAILS_TEMPLATE.format(
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
                details_section = _mon.COMPLETION_DETAILS_TEMPLATE.format(
                    restart_count=process_data.get("restart_count", 0),
                    total_runtime=process_data.get("total_runtime", 0.0),
                )
            else:
                return

            html_content = _mon.CONSOLIDATED_STATUS_TEMPLATE.format(
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
            _mon.print_error_message("EMAIL", f"Failed to send consolidated status email: {e}")

    def should_attempt_restart(self, title: str, restart_count: int, max_restarts: int) -> bool:
        if self.retry_count < self.retry_attempts:
            self.retry_count += 1
            return True

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
        self.send_consolidated_status_email(
            "task_complete",
            {
                "title": title,
                "restart_count": restart_count,
                "total_runtime": total_runtime,
            },
        )

    def report_final_failure(self, title: str, restart_count: int, error: str) -> None:
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
