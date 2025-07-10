"""
This module provides a restarting monitoring system for processes with email alert capabilities.
It monitors a process for crashes and restarts it if necessary.

Usage example:
    run_auto_restart(
        file_path="my_script.py",
        title="My Critical Process",
    )
"""


from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import Optional

from araras.utils.misc import clear
from .restart_manager import FlagBasedRestartManager
from .prints import print_error_message, print_process_status

def run_auto_restart(
    file_path: str,
    success_flag_file: str = "/tmp/success.flag",
    title: Optional[str] = None,
    max_restarts: int = 10,
    restart_delay: float = 3.0,
    recipients_file: Optional[str] = None,
    credentials_file: Optional[str] = None,
    restart_after_delay: Optional[float] = None,
    retry_attempts: int = None,
    supress_tf_warnings: bool = False,
) -> None:
    """Main function with notebook conversion, file cleanup, and consolidated email notification support.

    Args:
        file_path: Path to .py or .ipynb file to execute
        success_flag_file: Path to success flag file
        title: Custom title for monitoring and email alerts
        max_restarts: Maximum restart attempts
        restart_delay: Delay between restarts in seconds
        recipients_file: Path to recipients JSON file (defaults to ./json/recipients.json)
        credentials_file: Path to credentials JSON file (defaults to ./json/credentials.json)
        restart_after_delay: restart the run after a delay in seconds
        retry_attempts: Number of retry attempts before sending failure email
        supress_tf_warnings: Suppress TensorFlow warnings (default: False)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
        ImportError: If notebook dependencies missing for .ipynb files
    """

    try:
        # Clean up any existing success flag file before starting
        Path(success_flag_file).unlink(missing_ok=True)

        manager = FlagBasedRestartManager(
            max_restarts=max_restarts,
            restart_delay=restart_delay,
            recipients_file=recipients_file,
            credentials_file=credentials_file,
            retry_attempts=max_restarts if retry_attempts is None else retry_attempts,
        )

        if restart_after_delay is not None and restart_after_delay > 0:
            # Wrapping logic for forced restart not counting as crash/max_restarts
            # This will run in a loop, restarting after each interval, until success_flag is found.

            stop_event = Event()

            def restart_loop():
                try:
                    while not stop_event.is_set():
                        manager.restart_count = 0  # Never increment max_restarts for forced restart
                        # Ensure no leftover processes remain running
                        manager._cleanup_stale_pids()
                        finished = [False]

                        def run_and_flag():
                            try:
                                manager.run_file_with_restart(
                                    file_path=file_path,
                                    success_flag_file=success_flag_file,
                                    title=title,
                                    restart_after_delay=restart_after_delay,
                                    supress_tf_warnings=supress_tf_warnings,
                                )
                                finished[0] = True
                            except Exception:
                                finished[0] = True  # On error, still allow restart

                        thread = Thread(target=run_and_flag)
                        thread.start()
                        thread.join(timeout=restart_after_delay)
                        if thread.is_alive():
                            print_process_status(
                                f"Forcing restart after {restart_after_delay} seconds (not a crash)"
                            )
                            manager.force_stop()
                            # Ensure the worker thread finishes cleanly before continuing
                            thread.join(5)
                            clear()

                        else:
                            # If finished (success or crash), check if success
                            if Path(success_flag_file).exists():
                                stop_event.set()
                            else:
                                print_process_status(
                                    "Process ended before restart_after_delay, restarting..."
                                )
                except KeyboardInterrupt:
                    # Handle CTRL+C in the restart loop
                    stop_event.set()
                    print_process_status("Restart loop interrupted by user, cleaning up")
                    manager.force_stop()
                    manager._cleanup_converted_file()
                except Exception as e:
                    print_error_message("FATAL", str(e))
                    # Show traceback if needed
                    import traceback
                    traceback.print_exc()
                    stop_event.set()
                    manager.force_stop()
                    manager._cleanup_converted_file()
                # Ensure the worker thread has completely finished before returning
                thread.join()
                print_process_status("Restart-after-delay loop done")

            restart_loop()

        else:
            # Regular auto-restart logic
            manager.run_file_with_restart(
                file_path=file_path,
                success_flag_file=success_flag_file,
                title=title,
                supress_tf_warnings=supress_tf_warnings,
            )

    except (FileNotFoundError, ValueError, ImportError) as e:
        print_error_message("CONFIG", str(e))
        raise
    except KeyboardInterrupt:
        print_process_status("Main process interrupted by user, performing final cleanup")
    except Exception as e:
        print_error_message("FATAL", str(e))
        raise

