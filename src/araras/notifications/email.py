from araras.core import *

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def get_credentials(file_path: str) -> tuple[str, str]:
    """Load sender credentials from a JSON file.

    Args:
        file_path: Path to a JSON document containing ``"email"`` and
            ``"password"`` keys.

    Returns:
        tuple[str, str]: Sender email address and password extracted from the
        file.

    Raises:
        ValueError: If the file cannot be read or does not contain the expected
        keys.

    Examples:
        >>> get_credentials("credentials.json")
        ("your_email@gmail.com", "your_password")
    """
    try:
        # Open and read the JSON file containing credentials
        with open(file_path, "r") as file:
            credentials = json.load(file)  # Load file content as a Python dict
            return credentials["email"], credentials["password"]  # Extract and return credentials
    except Exception as e:
        raise ValueError(f"Failed to read credentials: {e}")


def get_recipient_emails(file_path: str) -> list[str]:
    """Load recipient addresses from a JSON file.

    Args:
        file_path: Path to a JSON document containing an ``"emails"`` list.

    Returns:
        list[str]: Email addresses that should receive the notification.

    Raises:
        ValueError: If the file cannot be read or does not include an
        ``"emails"`` key with an iterable value.

    Examples:
        >>> get_recipient_emails("recipients.json")
        ["recipient1@example.com", "recipient2@example.com"]
    """
    try:
        # Open and read the JSON file containing recipient email addresses
        with open(file_path, "r") as file:
            recipient_data = json.load(file)  # Load file content as a Python dict
            return recipient_data["emails"]  # Return the list of emails
    except Exception as e:
        raise ValueError(f"Failed to read recipient emails: {e}")


def send_email(
    subject: str,
    body: str,
    recipients_file: str,
    credentials_file: str,
    text_type: str = "plain",
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> None:
    """Send an email notification to every configured recipient.

    Args:
        subject: Subject line for the message.
        body: Plain-text or HTML content to send.
        recipients_file: Path to a JSON file consumed by
            :func:`get_recipient_emails`.
        credentials_file: Path to a JSON file understood by
            :func:`get_credentials`.
        text_type: MIME subtype to use for the message body (``"plain"`` or
            ``"html"``).
        smtp_server: Hostname or IP address of the SMTP server.
        smtp_port: Port used to connect to the SMTP server.

    Notes:
        Exceptions raised while loading files or sending the message are
        caught and logged. No exception propagates to the caller.

    Examples:
        >>> send_email(
        ...     "Hello",
        ...     "<p>This is a test</p>",
        ...     "recipients.json",
        ...     "credentials.json",
        ...     text_type="html",
        ... )
    """

    try:
        sender_email, sender_password = get_credentials(credentials_file)
        recipient_emails = get_recipient_emails(recipients_file)

        # Create a multipart email message object
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = ", ".join(recipient_emails)
        message["Subject"] = subject
        message.attach(MIMEText(body, text_type))
    except Exception as e:
        logger_error.error(f"{RED}[ERROR] {e}{RESET}")
        return

    try:
        # Establish connection to SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Start TLS encryption
            server.login(sender_email, sender_password)  # Login using credentials
            server.sendmail(sender_email, recipient_emails, message.as_string())  # Send email
        logger.info("Email sent successfully.")
    except Exception as e:
        logger_error.error(f"{RED}[ERROR] Failed to send email: {e}{RESET}")
