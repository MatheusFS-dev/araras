from araras.commons import *  # Common imports and configs for the Araras lib

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def get_credentials(file_path: str) -> tuple[str, str]:
    """
    Reads the sender's email and password from a JSON file.

    Logic:
        file_path -> open JSON file -> load contents -> extract 'email' and 'password' -> return as tuple

    Args:
        file_path (str): Path to the credentials JSON file.

    Returns:
        tuple[str, str]: A tuple containing the sender email and password.

    Raises:
        ValueError: If the credentials cannot be read or parsed.

    Example:
        email, password = get_credentials("credentials.json")
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

    Logic:
        file_path -> open JSON file -> load contents -> extract 'emails' list -> return

    Args:
        file_path (str): Path to the recipient JSON file.

    Returns:
        list[str]: A list of recipient email addresses.

    Raises:
        ValueError: If the file or its contents cannot be read.

    Example:
        recipients = get_recipient_emails("recipients.json")
    """
    try:
        # Open and read the JSON file containing recipient email addresses
        with open(file_path, "r") as file:
            recipient_data = json.load(file)  # Load file content as a Python dict
            return recipient_data["emails"]  # Return the list of emails
    except Exception as e:
        raise ValueError(f"Failed to read recipient emails: {e}")


def send_email(
    subject: str, body: str, recipients_file: str, credentials_file: str, text_type: str = "plain"
) -> None:
    """
    Sends an email notification with the specified subject and body content to multiple recipients.

    Logic:
        get sender credentials -> get recipients list -> create MIME message ->
        connect to SMTP -> authenticate -> send email

    Args:
        subject (str): The subject of the email.
        body (str): The main content of the email.
        recipients_file (str): Path to the recipients JSON file.
        credentials_file (str): Path to the credentials JSON file.
        text_type (str): The type of text content (e.g., "plain" or "html").

    Returns:
        None

    Example:
        send_email("Hi", "This is a test", "recipients.json", "credentials.json", text_type="html")
    """
    smtp_server = "smtp.gmail.com"  # SMTP server for Gmail
    smtp_port = 587  # TLS port

    try:
        # Get sender email and password
        sender_email, sender_password = get_credentials(credentials_file)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    try:
        # Get list of recipient emails
        recipient_emails = get_recipient_emails(recipients_file)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    # Create a multipart email message object
    message = MIMEMultipart()
    message["From"] = sender_email  # Set sender
    message["To"] = ", ".join(recipient_emails)  # Join recipients into string
    message["Subject"] = subject  # Set subject
    message.attach(MIMEText(body, text_type))  # Attach message body with specified format

    try:
        # Establish connection to SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Start TLS encryption
            server.login(sender_email, sender_password)  # Login using credentials
            server.sendmail(sender_email, recipient_emails, message.as_string())  # Send email
        print("[INFO] Email sent successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
