"""
This module provides a utility for cleaning up child processes efficiently
by gracefully terminating them first and force killing any that do not respond.
"""

import os
import psutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import *


# ——————————————————————————— Child Process Cleanup —————————————————————————— #
class ChildProcessCleanup:
    """Efficient child process cleanup with graceful termination and force kill fallback."""

    __slots__ = ("_current_pid", "_exclude_pids", "_termination_timeout", "_kill_timeout")

    def __init__(self, termination_timeout: float = 2.0, kill_timeout: float = 1.0):
        """Initialize cleanup manager with configurable timeouts.

        Args:
            termination_timeout: Seconds to wait for graceful termination
            kill_timeout: Seconds to wait after force kill
        """
        self._current_pid = os.getpid()
        self._exclude_pids: Set[int] = {self._current_pid}
        self._termination_timeout = termination_timeout
        self._kill_timeout = kill_timeout

    def cleanup_children(self, exclude_pids: Optional[List[int]] = None) -> Tuple[int, int]:
        """Clean up all child processes with optimized batch operations.

        Args:
            exclude_pids: Additional PIDs to exclude from cleanup

        Returns:
            Tuple of (terminated_count, killed_count)

        Raises:
            psutil.NoSuchProcess: If current process doesn't exist
        """
        # Combine exclude PIDs for efficient lookup
        exclude_set = self._exclude_pids.copy()
        if exclude_pids:
            exclude_set.update(exclude_pids)

        try:
            current_process = psutil.Process(self._current_pid)
            children = current_process.children(recursive=True)

            if not children:
                return 0, 0

            # Filter children to cleanup (exclude protected PIDs)
            targets = [child for child in children if child.pid not in exclude_set]

            if not targets:
                return 0, 0

            print(f"Cleaning up {len(targets)} child processes")

            # Phase 1: Graceful termination with parallel execution
            terminated_count = self._terminate_processes(targets)

            # Phase 2: Force kill remaining processes
            killed_count = self._kill_remaining_processes(targets, exclude_set)

            return terminated_count, killed_count

        except psutil.NoSuchProcess:
            raise psutil.NoSuchProcess(f"Current process {self._current_pid} not found")

    def _terminate_processes(self, processes: List[psutil.Process]) -> int:
        """Terminate processes gracefully with parallel execution.

        Args:
            processes: List of processes to terminate

        Returns:
            Number of processes successfully terminated
        """
        terminated_count = 0

        # Parallel termination for efficiency
        with ThreadPoolExecutor(max_workers=min(len(processes), 8)) as executor:
            # Submit termination tasks
            future_to_process = {executor.submit(self._safe_terminate, proc): proc for proc in processes}

            # Collect results
            for future in as_completed(future_to_process, timeout=self._termination_timeout + 1):
                if future.result():
                    terminated_count += 1

        # Wait for graceful shutdown
        if terminated_count > 0:
            time.sleep(self._termination_timeout)

        return terminated_count

    def _safe_terminate(self, process: psutil.Process) -> bool:
        """Safely terminate a single process with error handling.

        Args:
            process: Process to terminate

        Returns:
            True if termination signal sent successfully, False otherwise
        """
        try:
            if process.is_running():
                process.terminate()
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return False

    def _kill_remaining_processes(
        self, original_processes: List[psutil.Process], exclude_set: Set[int]
    ) -> int:
        """Force kill processes that didn't terminate gracefully.

        Args:
            original_processes: Original list of processes to check
            exclude_set: PIDs to exclude from force kill

        Returns:
            Number of processes force killed
        """
        killed_count = 0

        # Re-check which processes are still running
        for process in original_processes:
            if process.pid in exclude_set:
                continue

            try:
                if process.is_running():
                    process.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process already gone or no permission
                pass

        # Brief wait after force kill
        if killed_count > 0:
            time.sleep(self._kill_timeout)

        return killed_count

    def add_protected_pid(self, pid: int) -> None:
        """Add a PID to the protected (exclude) list.

        Args:
            pid: Process ID to protect from cleanup
        """
        self._exclude_pids.add(pid)

    def remove_protected_pid(self, pid: int) -> None:
        """Remove a PID from the protected list.

        Args:
            pid: Process ID to remove from protection
        """
        self._exclude_pids.discard(pid)  # discard won't raise if not present

    def get_child_count(self) -> int:
        """Get current number of child processes.

        Returns:
            Number of child processes (including nested children)
        """
        try:
            current_process = psutil.Process(self._current_pid)
            return len(current_process.children(recursive=True))
        except psutil.NoSuchProcess:
            return 0
