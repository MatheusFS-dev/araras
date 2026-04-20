"""Interactive launcher for the runtime monitor entrypoint.

This module keeps the public ``monitor`` console script but removes the
previous ``argparse`` surface. The command now accepts only positional target
file paths and prompts the user for the configurable monitor settings that are
still needed at launch time.
"""

from typing import List, Optional, Tuple

import json
import sys
from pathlib import Path

from araras.runtime.monitoring import run_auto_restart
from araras.utils.verbose_printer import VerbosePrinter

vp = VerbosePrinter()

_DEFAULT_MAX_RESTARTS = 10
_DEFAULT_JSON_CHOICE = "1"
_RECIPIENTS_FILE_NAME = "recipients.json"
_CREDENTIALS_FILE_NAME = "credentials.json"
_RECIPIENTS_TEMPLATE = {
    "emails": [
        "first-recipient@example.com",
        "second-recipient@example.com",
        "third-recipient@example.com",
    ]
}
_CREDENTIALS_TEMPLATE = {
    "email": "THE BOT EMAIL GOES HERE (YOU CAN USE app password FROM GMAIL TO CREATE A BOT ACCOUNT)",
    "password": "THE BOT PASSWORD GOES HERE",
}


def _print_monitor_banner() -> None:
    """Print the interactive monitor launcher banner.

    Returns:
        None: The function prints the banner to stdout and does not return a
        value.
    """
    vp.printf("=" * 80, color="yellow")
    vp.printf("MONITOR SCRIPT".center(80), color="yellow")
    vp.printf("=" * 80, color="yellow")
    vp.printf(
        vp.color("Run this script with ", "yellow")
        + vp.color("sudo", "red")
        + vp.color(" if the other process requires it!", "yellow")
    )
    vp.printf(
        "If using a Conda Venv, ensure you are using the correct python interpreter.",
        color="yellow",
    )
    vp.printf(f">>> {sys.executable} is running this script", color="orange")
    vp.printf("=" * 80, color="yellow")


def _parse_file_paths(argv: Optional[List[str]] = None) -> List[str]:
    """Return positional target file paths for the monitor entrypoint.

    The ``monitor`` command now accepts only positional paths. Any argument
    starting with ``-`` is rejected so users receive an explicit failure instead
    of silently mixing the new prompt-driven flow with the removed CLI flags.

    Args:
        argv (Optional[List[str]]): Optional argument list to parse instead of
            ``sys.argv[1:]``. If ``None``, the function reads the live command
            line. If a list is provided, every element is treated as a raw
            command-line token and no shell parsing is performed.

    Returns:
        List[str]: Ordered list of target file paths to monitor sequentially.

    Raises:
        SystemExit: If no target paths are supplied or if any removed option
        flag is present.

    Examples:
        >>> _parse_file_paths(["train.py", "evaluate.py"])
        ['train.py', 'evaluate.py']
    """
    file_paths = list(sys.argv[1:] if argv is None else argv)
    if not file_paths:
        raise SystemExit("Usage: monitor <file.py|file.ipynb> [more files...]")

    invalid_flags = [argument for argument in file_paths if argument.startswith("-")]
    if invalid_flags:
        raise SystemExit(
            "monitor now accepts only target file paths. "
            "Title, JSON location, and max restarts are configured through prompts."
        )

    return file_paths


def _prompt_title() -> Optional[str]:
    """Prompt the user for a shared process title.

    A blank response preserves the previous behavior where each monitored file
    uses its own stem as the process title.

    Returns:
        Optional[str]: The trimmed custom title when the user provides one, or
        ``None`` when Enter is pressed to keep the per-file default title.
    """
    response = input("Monitor title [default: file stem]: ").strip()
    return response or None


def _prompt_json_choice() -> str:
    """Prompt the user for the email JSON folder selection.

    Returns:
        str: One of ``"1"``, ``"2"``, or ``"3"``. Blank input selects
        ``"1"`` which resolves to ``$HOME/.araras/json``.
    """
    while True:
        print()
        print("Email JSON folder options:")
        print("1. [default] $HOME/.araras/json")
        print("2. ./json (the directory where monitor was launched)")
        print("3. Custom folder")
        response = input("Choose JSON folder [default: 1]: ").strip() or _DEFAULT_JSON_CHOICE
        if response in {"1", "2", "3"}:
            return response
        vp.printf("Please choose 1, 2, or 3.", color="yellow", tag="[ARARAS WARNING] ")


def _prompt_custom_json_directory() -> str:
    """Prompt the user for a custom JSON directory path.

    Returns:
        str: Non-empty path string entered by the user. Relative paths are kept
            relative here and resolved later against the launch directory.
    """
    while True:
        response = input("Custom JSON folder path: ").strip()
        if response:
            return response
        vp.printf("Custom JSON folder cannot be empty.", color="yellow", tag="[ARARAS WARNING] ")


def _prompt_max_restarts(default_max_restarts: int = _DEFAULT_MAX_RESTARTS) -> int:
    """Prompt the user for the restart limit.

    Args:
        default_max_restarts (int): Default restart count used when the user
            presses Enter. This value must be a positive integer. If a different
            default is supplied in the future, blank input will adopt that
            value, while any typed value must still be a positive integer.

    Returns:
        int: Positive maximum restart count chosen by the user or the provided
        default when Enter is pressed.

    Raises:
        ValueError: If ``default_max_restarts`` is not strictly positive.
    """
    if default_max_restarts <= 0:
        raise ValueError("default_max_restarts must be > 0")

    while True:
        response = input(f"Max restarts [default: {default_max_restarts}]: ").strip()
        if not response:
            return default_max_restarts
        if response.isdigit() and int(response) > 0:
            return int(response)
        vp.printf(
            "Max restarts must be a positive integer.",
            color="yellow",
            tag="[ARARAS WARNING] ",
        )


def _create_json_templates(destination_directory: Path) -> bool:
    """Create default JSON template files directly in the destination directory.

    Existing files are preserved so repeated launches do not overwrite user
    edits. Missing template files are written from the in-code template content
    so installed packages do not depend on repository-only files.

    Args:
        destination_directory (Path): Directory that should contain
            ``recipients.json`` and ``credentials.json``. The directory must
            already exist. Existing destination files are left unchanged while
            missing files are created from the built-in template dictionaries.

    Returns:
        bool: ``True`` if at least one template file was created, or ``False``
        if both destination files already existed and were left untouched.

    Raises:
        OSError: If the destination files cannot be created.
    """
    template_files = {
        _RECIPIENTS_FILE_NAME: _RECIPIENTS_TEMPLATE,
        _CREDENTIALS_FILE_NAME: _CREDENTIALS_TEMPLATE,
    }
    copied_any_template = False

    for file_name, template_data in template_files.items():
        destination_path = destination_directory / file_name
        if not destination_path.exists():
            with open(destination_path, "w") as file_pointer:
                json.dump(template_data, file_pointer, indent=4)
                file_pointer.write("\n")
            copied_any_template = True

    return copied_any_template


def _get_default_json_directory() -> Path:
    """Return the default JSON directory under the current user home.

    Returns:
        Path: ``$HOME/.araras/json`` resolved from the current process
        environment at call time so tests and SSH sessions can override
        ``HOME`` safely.
    """
    return Path.home() / ".araras" / "json"


def _print_json_setup_instructions(json_directory: Path) -> None:
    """Print detailed instructions for the generated default JSON templates.

    Args:
        json_directory (Path): Directory where the template JSON files were
            created. The path is displayed so the user knows exactly which
            files must be edited before email alerts can work.

    Returns:
        None: The function prints instructions to stdout.
    """
    vp.printf(
        f"Default email configuration folder was missing. Created templates in {json_directory}.",
        color="yellow",
        tag="[ARARAS WARNING] ",
    )
    vp.printf(
        "Email alerts will stay disabled until you replace the placeholder values in these files:",
        color="yellow",
        tag="[ARARAS WARNING] ",
    )
    vp.printf(
        f"1. {json_directory / _RECIPIENTS_FILE_NAME} -> set the 'emails' list with the recipients who should receive monitor alerts.",
        color="yellow",
        tag="[ARARAS WARNING] ",
    )
    vp.printf(
        f"2. {json_directory / _CREDENTIALS_FILE_NAME} -> set the sender email and password used by the alert bot account.",
        color="yellow",
        tag="[ARARAS WARNING] ",
    )
    vp.printf(
        "After editing the files, rerun monitor and email delivery will be enabled automatically.",
        color="yellow",
        tag="[ARARAS WARNING] ",
    )


def _resolve_json_file_paths(
    choice: str,
    launch_directory: Path,
    custom_directory: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve recipients and credentials JSON file paths from a folder choice.

    Args:
        choice (str): JSON directory selection. ``"1"`` resolves to
            ``$HOME/.araras/json`` and creates template files when the folder is
            missing. ``"2"`` resolves to ``./json`` relative to
            ``launch_directory``. ``"3"`` resolves to ``custom_directory`` and
            requires that argument to be non-empty.
        launch_directory (Path): Directory where the ``monitor`` command was
            invoked. This is used only for option ``"2"`` and for resolving a
            relative custom path for option ``"3"``.
        custom_directory (Optional[str]): User-entered custom directory for
            option ``"3"``. If ``choice`` is ``"3"``, a blank or ``None`` value
            is invalid because the launcher must not guess a fallback path.

    Returns:
        Tuple[str, str]: Absolute paths to ``recipients.json`` and
        ``credentials.json`` in the resolved directory.

    Raises:
        ValueError: If ``choice`` is invalid or option ``"3"`` is selected
            without a custom directory.
        FileNotFoundError: If the repository template JSON files are missing
            when option ``"1"`` needs to bootstrap the default directory.
        OSError: If the default directory or template files cannot be created.

    Examples:
        >>> recipients, credentials = _resolve_json_file_paths("2", Path.cwd())
        >>> recipients.endswith("json/recipients.json")
        True
    """
    if choice == "1":
        json_directory = _get_default_json_directory()

        # The default home directory needs an explicit bootstrap path because
        # the monitor should guide the user instead of failing later when email
        # support is first exercised.
        if not json_directory.exists():
            json_directory.mkdir(parents=True, exist_ok=True)
            _create_json_templates(json_directory)
            _print_json_setup_instructions(json_directory)
        else:
            copied_any_template = _create_json_templates(json_directory)
            if copied_any_template:
                _print_json_setup_instructions(json_directory)
    elif choice == "2":
        json_directory = (launch_directory / "json").resolve()
    elif choice == "3":
        if not custom_directory:
            raise ValueError("custom_directory is required when choice is '3'")

        custom_path = Path(custom_directory).expanduser()
        if not custom_path.is_absolute():
            custom_path = (launch_directory / custom_path).resolve()
        else:
            custom_path = custom_path.resolve()
        json_directory = custom_path
    else:
        raise ValueError(f"Unsupported JSON choice: {choice}")

    return (
        str((json_directory / _RECIPIENTS_FILE_NAME).resolve()),
        str((json_directory / _CREDENTIALS_FILE_NAME).resolve()),
    )


def _collect_launch_configuration(launch_directory: Path) -> Tuple[Optional[str], int, str, str]:
    """Collect the shared prompt-driven monitor configuration once per launch.

    Args:
        launch_directory (Path): Directory where ``monitor`` was invoked. This
            path defines the base location for JSON option ``"2"`` and relative
            custom paths for option ``"3"``.

    Returns:
        Tuple[Optional[str], int, str, str]: Shared launch configuration in the
            order ``(title, max_restarts, recipients_file, credentials_file)``.
            ``title`` is ``None`` when the user keeps the per-file default
            behavior.
    """
    print()
    title = _prompt_title()
    json_choice = _prompt_json_choice()
    custom_directory = _prompt_custom_json_directory() if json_choice == "3" else None
    recipients_file, credentials_file = _resolve_json_file_paths(
        json_choice,
        launch_directory,
        custom_directory=custom_directory,
    )
    max_restarts = _prompt_max_restarts()
    print()
    return title, max_restarts, recipients_file, credentials_file


def main(argv: Optional[List[str]] = None) -> None:
    """Run the interactive ``monitor`` launcher.

    The entrypoint accepts only positional target file paths. All remaining
    launch settings are collected through prompts once and reused for each
    monitored file in sequence. Blank prompt responses keep the previous
    defaults requested by the user.

    Args:
        argv (Optional[List[str]]): Optional target path list. If ``None``,
            ``sys.argv[1:]`` is used. Every provided token must be a positional
            file path because optional CLI flags were intentionally removed from
            the launcher.

    Returns:
        None: The function invokes ``run_auto_restart`` for each target and has
        no return value.

    Raises:
        SystemExit: If no file paths are provided or removed CLI flags are
            supplied.
        FileNotFoundError: Propagated when the template JSON files cannot be
            found during default-directory bootstrap.
        OSError: Propagated when the launcher cannot create the default JSON
            directory or copy the template files.

    Examples:
        >>> main(["train.py"])
        # Prompts are shown and the file is monitored with the chosen settings.
    """
    file_paths = _parse_file_paths(argv)
    launch_directory = Path.cwd()

    _print_monitor_banner()
    title, max_restarts, recipients_file, credentials_file = _collect_launch_configuration(
        launch_directory
    )

    for target in file_paths:
        run_auto_restart(
            file_path=target,
            title=title,
            max_restarts=max_restarts,
            recipients_file=recipients_file,
            credentials_file=credentials_file,
        )


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
