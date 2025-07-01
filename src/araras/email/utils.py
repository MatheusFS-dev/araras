"""
This module provides utility functions for sending email notifications using SMTP.
It includes functionality to read sender credentials and recipient email addresses
from JSON files, send emails with customizable content, and handle success or failure
notifications for tasks. The module is designed to simplify the process of integrating
email notifications into Python applications.

Functions:
    - get_credentials: Reads sender email and password from a JSON file.
    - get_recipient_emails: Reads recipient email addresses from a JSON file.
    - send_email: Sends an email to multiple recipients with a specified subject and body.
    - run_with_notification: Executes a function and sends success or failure email notifications.
    - notify_training_success: Sends a standardized "training succeeded" email notification.
    - notify_warning: Sends a detailed warning email when an exception is caught.
Example Usage:
    - Sending a custom email:
        send_email("Subject", "Body", "recipients.json", "credentials.json", text_type="html")
    - Running a task with notifications:
        run_with_notification(func=my_task, func_args=(1, 2), func_kwargs={}, ...)
"""

import json
import smtplib
import traceback
from typing import *
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


def run_with_notification(
    func: Callable,
    func_args: tuple,
    func_kwargs: dict,
    recipients_file: str,
    credentials_file: str,
    subject_success: str = "🎉 Task Completed Successfully",
    body_success: str = """
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <h2 style="color: #28a745;">✔️ Task Completed</h2>
                <p>The task you ran has completed successfully without any errors.</p>
                <p>Thank you for using our system.</p>
                <footer style="margin-top: 20px; text-align: center; font-size: 14px; color: #888;">
                    <p>Best regards,</p>
                    <p><strong>The Bot Mailman</strong></p>
                </footer>
            </body>
        </html>
        """,
    text_type: str = "plain",
) -> None:
    """
    Executes a function and sends an email indicating success or failure.

    Logic:
        try -> run the function ->
            if success -> send success email
            if exception -> capture traceback -> send error email

    Args:
        func (Callable): Function to execute.
        func_args (tuple): Positional arguments for the function.
        func_kwargs (dict): Keyword arguments for the function.
        recipients_file (str): Path to the recipients JSON file.
        credentials_file (str): Path to the credentials JSON file.
        subject_success (str): Subject for success email.
        body_success (str): Body content for success email.
        text_type (str): MIME type (e.g., "plain" or "html").

    Returns:
        None

    Example:
        run_with_notification(func=my_task, func_args=(1,2), func_kwargs={}, ...)
    """
    try:
        func(*func_args, **func_kwargs)  # Run the target function with arguments
        send_email(subject_success, body_success, recipients_file, credentials_file, text_type)
        print("[INFO] Success email sent.")
    except Exception:
        # Format full traceback from the exception
        error_message = traceback.format_exc()
        subject = "❌ Task Failed with an Error"

        # HTML-formatted error email
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <h2 style="color: #dc3545;">❌ Task Failed</h2>
                <p>The task you ran encountered an error:</p>
                <pre style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; font-size: 14px;">
                {error_message}
                </pre>
                <p>Please review the error above and try again.</p>
                <footer style="margin-top: 20px; text-align: center; font-size: 14px; color: #888;">
                    <p>Best regards,</p>
                    <p><strong>The Bot Mailman</strong></p>
                </footer>
            </body>
        </html>
        """
        send_email(subject, body, recipients_file, credentials_file, text_type)
        print("[INFO] Error email sent.")


def notify_training_success(
    recipients_file: str,
    credentials_file: str,
    *,
    subject: str = "🎉 Task Completed Successfully",
    body: Optional[str] = None,
    text_type: str = "html",
) -> None:
    """
    Send a standardized “training succeeded” email notification.

    Logic:
        check for custom body -> if not provided use default HTML template ->
        send email

    Args:
        recipients_file (str): Path to JSON file containing recipient emails.
        credentials_file (str): Path to JSON file containing sender credentials.
        subject (str): Email subject line.
        body (Optional[str]): Full HTML or plain-text body. Uses default if None.
        text_type (str): MIME text type, e.g. "html" or "plain".

    Returns:
        None

    Example:
        notify_training_success("recipients.json", "credentials.json")
    """
    # Define default HTML body if none is provided
    default_body = """
    <html>
      <body style="font-family:Arial,sans-serif;line-height:1.6;color:#333;">
        <h2 style="color:#28a745;">✔️ Task Completed</h2>
        <p>Your training job has finished successfully.</p>
        <p>Thank you for using our system.</p>
        <footer style="margin-top:20px;text-align:center;font-size:14px;color:#888;">
          <p>Best regards,</p>
          <p><strong>The Bot Mailman</strong></p>
        </footer>
      </body>
    </html>
    """.strip()

    # Use provided body or fallback to default
    final_body = body if body is not None else default_body

    # Send success email
    send_email(subject, final_body, recipients_file, credentials_file, text_type)


def notify_warning(
    recipients_file: str,
    credentials_file: str,
    *,
    error: Optional[Exception] = None,
    subject: str = "❌ Task Failed with Error",
    text_type: str = "html",
) -> None:
    """
    Send a detailed warning email when an exception is caught.

    This function is designed to be used in an `except:` block to inform recipients
    about a runtime error or failure in a task.

    Logic:
        capture traceback -> format as HTML -> embed into styled email body ->
        send via email

    Args:
        recipients_file (str): Path to JSON file containing recipient emails.
        credentials_file (str): Path to JSON file containing sender credentials.
        error (Optional[Exception]): Exception instance. If None, full traceback is used.
        subject (str): Email subject line.
        text_type (str): MIME text type, e.g. "html" or "plain".

    Returns:
        None

    Example:
        try:
            run_critical_process()
        except Exception as e:
            notify_warning("recipients.json", "credentials.json", error=e)
    """
    # Retrieve error traceback as HTML-formatted string
    error_details = (
        traceback.format_exc() if error is None else traceback.format_exception_only(type(error), error)
    )
    error_html = "<br>".join(line.replace(" ", "&nbsp;") for line in error_details)

    # Compose error email with highlighted traceback
    body = f"""
    <html>
      <body style="font-family:Arial,sans-serif;line-height:1.6;color:#333;">
        <h2 style="color:#dc3545;">❌ Task Failed</h2>
        <p>The task encountered an error. See details below:</p>
        <div style="background-color:#f8d7da;color:#721c24;padding:10px;border-radius:5px;
                    font-family:monospace;font-size:14px;white-space:pre-wrap;">
          {error_html}
        </div>
        <p>Please investigate and retry.</p>
        <footer style="margin-top:20px;text-align:center;font-size:14px;color:#888;">
          <p>Best regards,</p>
          <p><strong>The Bot Mailman</strong></p>
        </footer>
      </body>
    </html>
    """.strip()

    # Send the error notification email
    send_email(subject, body, recipients_file, credentials_file, text_type)


# Example usage section
if __name__ == "__main__":
    credentials_file_path = "credentials.json"
    recipient_file_path = "recipients.json"

    # --------------------------- Custom Email Example --------------------------- #
    email_subject = "Model Training Complete"
    email_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f9f9f9; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: #fff; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                <h2 style="color: #0056b3; text-align: center;">🎉 Training Complete</h2>
                <p style="font-size: 16px; color: #444;"><strong>Dear User,</strong></p>
                <p style="font-size: 18px; color: #333;">Your model, <strong style="color: #0056b3;">LSTM_DNN</strong>, has successfully completed training!</p>
                <p style="font-size: 16px; color: #555;">I'm excited to see how you'll utilize your new model!</p>
                <p style="text-align: center; font-size: 16px;"><strong style="color: #28a745;">✔️ Training Status:</strong> <span style="color: #0056b3;">Completed</span></p>
                <footer style="margin-top: 20px; text-align: center; font-size: 14px; color: #888;">
                    <p>Best regards,</p>
                    <p><strong>The Bot Mailman</strong></p>
                </footer>
            </div>
        </body>
    </html>
    """

    # Send standalone notification email
    send_email(
        subject=email_subject,
        body=email_body,
        recipients_file=recipient_file_path,
        credentials_file=credentials_file_path,
        text_type="html",
    )

    def example_task(x: int, y: int) -> int:
        """
        A sample function that adds two numbers and raises an exception if either is negative.

        Args:
            x (int): First number.
            y (int): Second number.

        Returns:
            int: Sum of x and y.

        Raises:
            ValueError: If either x or y is negative.
        """
        if x < 0 or y < 0:
            raise ValueError("Inputs must be non-negative.")
        return x + y

    # Run successful task with notification
    run_with_notification(
        func=example_task,
        func_args=(10, 20),
        func_kwargs={},
        recipients_file=recipient_file_path,
        credentials_file=credentials_file_path,
        text_type="html",
    )

    # Run failing task with notification
    run_with_notification(
        func=example_task,
        func_args=(-10, 20),
        func_kwargs={},
        recipients_file=recipient_file_path,
        credentials_file=credentials_file_path,
        text_type="html",
    )
