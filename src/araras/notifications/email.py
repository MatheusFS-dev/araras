"""
This module provides functions to send emails using SMTP with Gmail.

Functions:
    - get_credentials: Reads the sender's email and password from a JSON file.
    - get_recipient_emails: Reads a list of recipient email addresses from a JSON file.
    - send_email: Sends an email notification with the specified subject and body content to multiple recipients.
"""

from araras.core import *  # Common imports and configs for the Araras lib

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def get_credentials(file_path: str) -> tuple[str, str]:
    """
    Reads the sender's email and password from a JSON file.
    The json file format should be:
    {
        "email": "your_email@gmail.com",
        "password": "your_password"
    }

    Args:
        file_path (str): Path to the credentials JSON file.

    Returns:
        tuple[str, str]: A tuple containing the sender email and password.

    Raises:
        ValueError: If the credentials cannot be read or parsed.
    """
    try:
        # Open and read the JSON file containing credentials
        with open(file_path, "r") as file:
            credentials = json.load(file)  # Load file content as a Python dict
            return credentials["email"], credentials["password"]  # Extract and return credentials
    except Exception as e:
        raise ValueError(f"Failed to read credentials: {e}")


def get_recipient_emails(file_path: str) -> list[str]:
    """
    Reads a list of recipient email addresses from a JSON file.
    The json file format should be:
    {
        "emails": ["recipient1@example.com", "recipient2@example.com"]
    }

    Args:
        file_path (str): Path to the recipient JSON file.

    Returns:
        list[str]: A list of recipient email addresses.

    Raises:
        ValueError: If the file or its contents cannot be read.
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
    """
    Sends an email notification with the specified subject and body content to multiple recipients.

    Example:
        send_email("Hi", "This is a test", "recipients.json", "credentials.json", text_type="html")

    Args:
        subject (str): The subject of the email.
        body (str): The main content of the email.
        recipients_file (str): Path to the recipients JSON file.
        credentials_file (str): Path to the credentials JSON file.
        text_type (str): The type of text content (e.g., "plain" or "html").
        smtp_server (str): The SMTP server address (default is Gmail's SMTP server).
        smtp_port (int): The port number for the SMTP server (default is 587 for TLS).

    Returns:
        None
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
