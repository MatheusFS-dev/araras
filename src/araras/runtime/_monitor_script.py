"""Interactive launcher for the runtime monitor entrypoint.

This module keeps the public ``monitor`` console script but removes the
previous ``argparse`` surface. The command now accepts only positional target
file paths and prompts the user for the configurable monitor settings that are
still needed at launch time.
"""

from typing import List, Optional, Tuple

import json
import math
import sys
from pathlib import Path

import psutil

from araras.runtime.monitoring import run_auto_restart
from araras.utils.verbose_printer import VerbosePrinter

vp = VerbosePrinter()

_DEFAULT_MAX_RESTARTS = 10
_DEFAULT_MEMORY_LEAK_WARMUP_MINUTES = 5.0
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
            presses Enter. This value must be a non-negative integer. If a
            different default is supplied in the future, blank input will
            adopt that value, while any typed value must still be a
            non-negative integer. A value of ``0`` disables automatic
            restarts after the initial launch attempt.

    Returns:
        int: Non-negative maximum restart count chosen by the user or the
            provided default when Enter is pressed. ``0`` means "do not
            restart".

    Raises:
        ValueError: If ``default_max_restarts`` is negative.
    """
    if default_max_restarts < 0:
        raise ValueError("default_max_restarts must be >= 0")

    while True:
        response = input(f"Max restarts [default: {default_max_restarts}]: ").strip()
        if not response:
            return default_max_restarts
        if response.isdigit():
            return int(response)
        vp.printf(
            "Max restarts must be a non-negative integer.",
            color="yellow",
            tag="[ARARAS WARNING] ",
        )


def _prompt_report_logs() -> bool:
    """Prompt the user whether detailed monitor artifacts should be written.

    Returns:
        bool: ``True`` when the user presses Enter or explicitly enables
            report logging with a yes-style answer such as ``"y"`` or
            ``"yes"``. ``False`` when the user enters a no-style answer such
            as ``"n"`` or ``"no"``.

    Raises:
        RuntimeError: Not raised by this helper.
    """
    while True:
        response = input("Report logs (Y/n) [default: Y]: ").strip().lower()
        if not response:
            return True
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        vp.printf(
            "Please answer y or n.",
            color="yellow",
            tag="[ARARAS WARNING] ",
        )


def _prompt_force_periodic_restart() -> bool:
    """Prompt whether either scheduled-restart policy should be enabled.

    Returns:
        bool: ``True`` when the user presses Enter or explicitly enables
        scheduled restarts with ``"y"`` or ``"yes"``. ``False`` when the
        user enters ``"n"`` or ``"no"``.

    Raises:
        RuntimeError: Not raised by this helper. Invalid responses are reported
        and requested again.
    """
    while True:
        response = input("Force periodic restart? (Y/n) [default: Y]: ").strip().lower()
        if not response or response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        vp.printf(
            "Please answer y or n.",
            color="yellow",
            tag="[ARARAS WARNING] ",
        )


def _prompt_force_restart_interval() -> float:
    """Prompt for the fixed scheduled-restart interval.

    This prompt is used only when scheduled restarts are enabled and the user
    declines the memory-aware policy. The user-facing value is expressed in
    minutes, while the runtime monitor consumes seconds.

    Returns:
        float: Positive fixed restart interval in seconds.

    Raises:
        RuntimeError: Not raised by this helper. Invalid responses are reported
        and requested again.

    Examples:
        Entering ``120`` returns ``7200.0`` seconds.
    """
    while True:
        response = input("Forced restart interval in minutes: ").strip()
        try:
            interval_minutes = float(response)
        except ValueError:
            vp.printf(
                "Forced restart interval must be a positive finite number of minutes.",
                color="yellow",
                tag="[ARARAS WARNING] ",
            )
            continue

        if math.isfinite(interval_minutes) and interval_minutes > 0:
            return interval_minutes * 60.0

        vp.printf(
            "Forced restart interval must be a positive finite number of minutes.",
            color="yellow",
            tag="[ARARAS WARNING] ",
        )


def _prompt_smart_scheduled_restart_configuration() -> Tuple[Optional[float], Optional[float]]:
    """Prompt for optional memory-aware scheduled-restart settings.

    Smart mode is an alternative to fixed periodic restarts. It checks current
    target-process-tree RSS at the selected polling interval and restarts only
    when RSS reaches the selected limit.

    Returns:
        Tuple[Optional[float], Optional[float]]: Process-tree RSS threshold in
        bytes and polling interval in seconds when smart mode is enabled. Both
        values are ``None`` when the user declines smart mode.

    Raises:
        RuntimeError: Not raised by this helper. Invalid responses are reported
        and requested again.

    Examples:
        Pressing Enter, then entering ``2.5`` and ``15``, returns a 2.5 GiB byte threshold
        and a 900-second polling interval.
    """
    while True:
        response = input("Use memory-aware scheduled restarts? (Y/n) [default: Y]: ").strip().lower()
        if response in {"n", "no"}:
            return None, None
        if not response or response in {"y", "yes"}:
            break
        vp.printf(
            "Please answer y or n.",
            color="yellow",
            tag="[ARARAS WARNING] ",
        )

    total_memory_gib = psutil.virtual_memory().total / float(1024**3)
    print(f"Current computer RAM: {total_memory_gib:g} GiB")

    while True:
        response = input("Restart when process-tree RSS reaches GiB: ").strip()
        try:
            threshold_gib = float(response)
        except ValueError:
            vp.printf(
                "Process-tree RSS threshold must be a positive finite number of GiB.",
                color="yellow",
                tag="[ARARAS WARNING] ",
            )
            continue

        if math.isfinite(threshold_gib) and threshold_gib > 0:
            threshold_bytes = threshold_gib * (1024**3)
            break

        vp.printf(
            "Process-tree RSS threshold must be a positive finite number of GiB.",
            color="yellow",
            tag="[ARARAS WARNING] ",
        )

    while True:
        response = input("Retry memory check every how many minutes: ").strip()
        try:
            poll_minutes = float(response)
        except ValueError:
            vp.printf(
                "Memory check interval must be a positive finite number of minutes.",
                color="yellow",
                tag="[ARARAS WARNING] ",
            )
            continue

        if math.isfinite(poll_minutes) and poll_minutes > 0:
            return threshold_bytes, poll_minutes * 60.0

        vp.printf(
            "Memory check interval must be a positive finite number of minutes.",
            color="yellow",
            tag="[ARARAS WARNING] ",
        )


def _prompt_memory_leak_configuration() -> Tuple[bool, float]:
    """Prompt for optional memory leak detection and its warm-up period.

    The user enters warm-up time in minutes while the runtime detector consumes
    seconds. Blank input enables detection and continues to the warm-up prompt.

    Returns:
        Tuple[bool, float]: Detection-enabled flag and positive warm-up duration
        in seconds. Disabled detection returns the default warm-up value because
        the runtime interface always carries an explicit duration.

    Raises:
        RuntimeError: Not raised by this helper. Invalid responses are reported
        and requested again.

    Examples:
        Entering ``y`` followed by ``10`` returns ``(True, 600.0)``.
    """
    while True:
        response = input("Detect possible memory leak? (Y/n) [default: Y]: ").strip().lower()
        if response in {"n", "no"}:
            return False, _DEFAULT_MEMORY_LEAK_WARMUP_MINUTES * 60.0
        if not response or response in {"y", "yes"}:
            break
        vp.printf(
            "Please answer y or n.",
            color="yellow",
            tag="[ARARAS WARNING] ",
        )

    while True:
        response = input(
            "Memory leak detection warm-up in minutes "
            f"[default: {_DEFAULT_MEMORY_LEAK_WARMUP_MINUTES:g}]: "
        ).strip()
        if not response:
            return True, _DEFAULT_MEMORY_LEAK_WARMUP_MINUTES * 60.0
        try:
            warmup_minutes = float(response)
        except ValueError:
            vp.printf(
                "Memory leak warm-up must be a positive finite number of minutes.",
                color="yellow",
                tag="[ARARAS WARNING] ",
            )
            continue
        if math.isfinite(warmup_minutes) and warmup_minutes > 0:
            return True, warmup_minutes * 60.0
        vp.printf(
            "Memory leak warm-up must be a positive finite number of minutes.",
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


def _collect_launch_configuration(
    launch_directory: Path,
) -> Tuple[
    Optional[str],
    int,
    str,
    str,
    Optional[float],
    Optional[float],
    Optional[float],
    bool,
    float,
    bool,
]:
    """Collect the shared prompt-driven monitor configuration once per launch.

    Args:
        launch_directory (Path): Directory where ``monitor`` was invoked. This
            path defines the base location for JSON option ``"2"`` and relative
            custom paths for option ``"3"``.

    Returns:
        Tuple[Optional[str], int, str, str, Optional[float], Optional[float],
        Optional[float], bool, float, bool]: Shared launch configuration in the
            order ``(title, max_restarts, recipients_file, credentials_file,
            force_restart, scheduled_restart_memory_threshold_bytes,
            scheduled_restart_poll_interval_seconds, detect_memory_leaks,
            memory_leak_warmup_seconds, report_logs)``. ``title`` is ``None``
            when the user keeps the per-file default. ``force_restart`` is a
            positive fixed interval in seconds or ``None``. Smart
            scheduled-restart values are both ``None`` when the user selects
            fixed restarts or disables scheduled restarts.
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
    force_restart = None
    scheduled_restart_memory_threshold_bytes = None
    scheduled_restart_poll_interval_seconds = None
    if _prompt_force_periodic_restart():
        (
            scheduled_restart_memory_threshold_bytes,
            scheduled_restart_poll_interval_seconds,
        ) = _prompt_smart_scheduled_restart_configuration()
        if scheduled_restart_memory_threshold_bytes is None:
            force_restart = _prompt_force_restart_interval()
    detect_memory_leaks, memory_leak_warmup_seconds = _prompt_memory_leak_configuration()
    report_logs = _prompt_report_logs()
    print()
    return (
        title,
        max_restarts,
        recipients_file,
        credentials_file,
        force_restart,
        scheduled_restart_memory_threshold_bytes,
        scheduled_restart_poll_interval_seconds,
        detect_memory_leaks,
        memory_leak_warmup_seconds,
        report_logs,
    )


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
    (
        title,
        max_restarts,
        recipients_file,
        credentials_file,
        force_restart,
        scheduled_restart_memory_threshold_bytes,
        scheduled_restart_poll_interval_seconds,
        detect_memory_leaks,
        memory_leak_warmup_seconds,
        report_logs,
    ) = _collect_launch_configuration(launch_directory)

    for target in file_paths:
        run_auto_restart(
            file_path=target,
            title=title,
            max_restarts=max_restarts,
            recipients_file=recipients_file,
            credentials_file=credentials_file,
            force_restart=force_restart,
            scheduled_restart_memory_threshold_bytes=scheduled_restart_memory_threshold_bytes,
            scheduled_restart_poll_interval_seconds=scheduled_restart_poll_interval_seconds,
            detect_memory_leaks=detect_memory_leaks,
            memory_leak_warmup_seconds=memory_leak_warmup_seconds,
            report_logs=report_logs,
        )


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
