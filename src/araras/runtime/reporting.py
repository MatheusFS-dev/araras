"""Helpers for writing monitor runtime reports and resource samples."""

from typing import Any, Dict, List, Optional, Tuple

import json
import math
import platform
import subprocess
import threading
import time
from pathlib import Path

import psutil

PLOT_PALETTE = {
    "background": "#ffffff",
    "grid": "#d7dbe2",
    "axis": "#20242b",
    "primary": "#000000",
    "highlight": "#b54708",
    "restart": "#7a1f3d",
}

TIME_AXIS_SCALES = (
    ("s", 1.0, 120.0),
    ("min", 60.0, 7200.0),
    ("h", 3600.0, 172800.0),
    ("d", 86400.0, float("inf")),
)


class MonitorReportSession:
    """Collect process and GPU samples for one monitored file run.

    The session owns one output directory under ``runs/monitor_logs`` and keeps
    appending samples and lifecycle events across all restart attempts for the
    same monitored source file. CPU and memory metrics are tied to the launched
    process tree, while GPU metrics reflect host GPU values reported by
    ``nvidia-smi`` when available. When GPU telemetry is unavailable the
    session records explicit ``None`` values instead of inventing zeros.

    Args:
        source_path (Path): Original monitored source file path. The stem of
            this path is used in the output directory name and the human
            readable log filename.
        title (str): Display title chosen for the monitored process. This
            value is written into summary files and event logs.
        base_directory (Optional[Path]): Root directory where report
            directories should be created. If ``None``, reports are written
            under ``Path.cwd() / "runs" / "monitor_logs"``. Supplying an
            explicit path is useful for tests because it avoids coupling to
            the live working directory.
        sample_interval_seconds (float): Delay between resource samples while
            an attempt is active. Smaller values produce denser charts but more
            filesystem writes and more ``psutil`` overhead. Larger values
            reduce overhead at the cost of lower temporal resolution.

    Raises:
        ValueError: If ``sample_interval_seconds`` is not strictly positive.

    Examples:
        >>> session = MonitorReportSession(Path("train.py"), "train")
        >>> session.output_directory.name.startswith("train.py-")
        True
    """

    def __init__(
        self,
        source_path: Path,
        title: str,
        base_directory: Optional[Path] = None,
        sample_interval_seconds: float = 1.0,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be > 0")

        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        report_root = base_directory or (Path.cwd() / "runs" / "monitor_logs")
        report_root.mkdir(parents=True, exist_ok=True)

        self.source_path = source_path.resolve()
        self.title = title
        self.sample_interval_seconds = sample_interval_seconds
        self.output_directory = report_root / f"{self.source_path.name}-{timestamp}"
        self.output_directory.mkdir(parents=True, exist_ok=False)

        self.log_file = self.output_directory / f"{self.source_path.name}.log"
        self.samples_file = self.output_directory / "samples.jsonl"
        self.restarts_file = self.output_directory / "restarts.json"
        self.summary_json_file = self.output_directory / "summary.json"
        self.summary_text_file = self.output_directory / "summary.txt"
        self.gpu_temperature_plot_file = self.output_directory / "gpu_temperature.png"

        self._samples: List[Dict[str, Any]] = []
        self._restart_events: List[Dict[str, Any]] = []
        self._attempt_events: List[Dict[str, Any]] = []
        self._pid_history: List[int] = []
        self._attempt_index = 0
        self._current_pid: Optional[int] = None
        self._current_attempt_started_at: Optional[float] = None
        self._run_started_at = time.time()
        self._sample_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cpu_snapshots: Dict[int, Dict[str, float]] = {}
        self.samples_file.write_text("")
        self.restarts_file.write_text("[]\n")
        self._write_event(
            f"report session created for {self.source_path} with title '{self.title}'"
        )

    def start_attempt(self, pid: int) -> None:
        """Begin sampling for a newly launched attempt.

        Args:
            pid (int): Root PID of the launched monitored process. The session
                treats this PID as the root of the process tree for CPU and
                memory sampling.

        Returns:
            None: The method updates internal state and starts a background
            sampling thread.

        Raises:
            ValueError: If ``pid`` is not a positive integer.
        """
        if pid <= 0:
            raise ValueError("pid must be > 0")

        self.stop_attempt()
        self._attempt_index += 1
        self._current_pid = pid
        self._current_attempt_started_at = time.time()
        self._pid_history.append(pid)
        self._prime_cpu_counters(pid)
        self._stop_event.clear()
        self._sample_thread = threading.Thread(
            target=self._sampling_loop,
            name=f"monitor-report-{self.source_path.stem}",
            daemon=True,
        )
        self._sample_thread.start()
        self._write_event(f"attempt {self._attempt_index} started with pid {pid}")

    def stop_attempt(self) -> None:
        """Stop background sampling for the current attempt, if active.

        Returns:
            None: The method is idempotent and safe to call when no attempt is
            currently active.

        Raises:
            RuntimeError: Not raised by this helper.
        """
        self._stop_event.set()
        if self._sample_thread is not None:
            self._sample_thread.join(timeout=max(2.0, self.sample_interval_seconds * 2.0))
        self._sample_thread = None
        self._current_pid = None
        self._current_attempt_started_at = None

    def record_attempt_end(
        self,
        reason: str,
        pid: Optional[int],
        runtime_seconds: Optional[float],
        error: Optional[str] = None,
    ) -> None:
        """Record how one launch attempt ended.

        Args:
            reason (str): Completion reason such as ``"success_flag"``,
                ``"crashed"``, ``"process_died"``, ``"interrupted"``, or
                ``"stopped"``. The value is persisted exactly so summaries can
                reflect the runtime decision path.
            pid (Optional[int]): Root PID associated with the completed
                attempt. ``None`` is allowed for early launch failures where a
                target PID never became available.
            runtime_seconds (Optional[float]): Attempt runtime in seconds. If
                ``None``, the summary stores no per-attempt duration.
            error (Optional[str]): Optional human-readable error associated
                with the attempt. When provided, the text is included in event
                logs and summary files.

        Returns:
            None: The method appends attempt metadata to the in-memory report.
        """
        event = {
            "attempt_index": self._attempt_index,
            "pid": pid,
            "reason": reason,
            "runtime_seconds": runtime_seconds,
            "error": error,
            "elapsed_seconds": time.time() - self._run_started_at,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        }
        self._attempt_events.append(event)
        if error:
            self._write_event(
                f"attempt {self._attempt_index} ended with reason '{reason}' and error: {error}"
            )
        else:
            self._write_event(f"attempt {self._attempt_index} ended with reason '{reason}'")

    def record_restart(
        self,
        old_pid: Optional[int],
        new_pid: Optional[int],
        restart_count: int,
        runtime_seconds: Optional[float],
        error: Optional[str] = None,
        restart_type: str = "crash_recovery",
        scheduled_restart_count: int = 0,
    ) -> None:
        """Record one restart transition.

        Args:
            old_pid (Optional[int]): PID of the attempt that just ended. This
                may be ``None`` for launch failures that occur before a target
                PID is known.
            new_pid (Optional[int]): PID of the newly launched attempt. This
                may be ``None`` when a restart was considered but the next
                launch never started successfully.
            restart_count (int): Total restart count after the transition.
            runtime_seconds (Optional[float]): Runtime of the previous attempt
                in seconds. ``None`` is allowed when no attempt runtime is
                available.
            restart_type (str): Explicit transition classification. Use
                ``"crash_recovery"`` for genuine failure recovery and
                ``"scheduled"`` for an intentional timed restart. Defaults
                to ``"crash_recovery"`` for existing failure-restart callers.
            scheduled_restart_count (int): Number of successful scheduled
                restarts after this transition. This count is independent of
                ``restart_count`` and defaults to ``0``.
            error (Optional[str]): Optional failure description that triggered
                the restart decision. When omitted, the session records only
                the PID transition and restart count.

        Returns:
            None: The method appends restart metadata to the in-memory report.
        """
        event = {
            "restart_count": restart_count,
            "scheduled_restart_count": scheduled_restart_count,
            "restart_type": restart_type,
            "old_pid": old_pid,
            "new_pid": new_pid,
            "runtime_seconds": runtime_seconds,
            "error": error,
            "old_attempt_index": self._attempt_index - 1 if new_pid is not None else self._attempt_index,
            "new_attempt_index": self._attempt_index if new_pid is not None else None,
            "restart_elapsed_seconds": time.time() - self._run_started_at if new_pid is not None else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        }
        self._restart_events.append(event)
        if error:
            self._write_event(
                f"{restart_type} restart: pid {old_pid} -> {new_pid}, error: {error}"
            )
        else:
            self._write_event(f"{restart_type} restart: pid {old_pid} -> {new_pid}")

    def finalize(
        self,
        final_status: str,
        total_runtime_seconds: Optional[float],
        total_restarts: int,
        final_error: Optional[str] = None,
    ) -> None:
        """Write all report artifacts for the monitored run.

        Args:
            final_status (str): Final run outcome such as ``"success"``,
                ``"failed"``, ``"interrupted"``, or ``"stopped"``.
            total_runtime_seconds (Optional[float]): Total runtime across all
                attempts in seconds. ``None`` is allowed when the caller cannot
                determine a total duration.
            total_restarts (int): Number of restart attempts performed across
                the full run.
            final_error (Optional[str]): Optional final failure description.
                When provided, the summary files include it explicitly.

        Returns:
            None: The method writes report files into ``output_directory``.
        """
        self.stop_attempt()
        self._write_event(f"final status: {final_status}")
        scheduled_restarts = sum(
            1 for event in self._restart_events if event.get("restart_type") == "scheduled"
        )

        summary = {
            "source_path": str(self.source_path),
            "title": self.title,
            "output_directory": str(self.output_directory),
            "cpu_name": self._get_cpu_name(),
            "system_memory_total_bytes": self._get_total_memory_bytes(),
            "system_memory_total_gb": self._get_total_memory_gb(),
            "final_status": final_status,
            "final_error": final_error,
            "total_runtime_seconds": total_runtime_seconds,
            "total_restarts": total_restarts,
            "crash_restarts": total_restarts - scheduled_restarts,
            "scheduled_restarts": scheduled_restarts,
            "pid_history": self._pid_history,
            "attempt_events": self._attempt_events,
            "restart_events": self._restart_events,
            "aggregate_metrics": self._build_aggregate_metrics(),
            "gpu_temperature_summary": self._build_gpu_temperature_summary(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        }

        self._enrich_restart_events()
        self._write_restarts_file()
        self._write_summary_files(summary)
        self._write_plots()

    def _sampling_loop(self) -> None:
        """Collect and persist resource samples while one attempt is active.

        Returns:
            None: The method runs in a background thread until the current
            attempt stops or the session is finalized.
        """
        while not self._stop_event.wait(self.sample_interval_seconds):
            pid = self._current_pid
            started_at = self._current_attempt_started_at
            if pid is None or started_at is None:
                continue

            sample = self._collect_sample(pid, started_at)
            self._samples.append(sample)
            with open(self.samples_file, "a") as file_pointer:
                file_pointer.write(json.dumps(sample) + "\n")

            self._write_event(
                "sample "
                f"attempt={sample['attempt_index']} pid={sample['root_pid']} "
                f"cpu={sample['cpu_percent']} mem_gb={sample['memory_gb']} "
                f"gpu={sample['gpu_utilization_percent']}"
            )

    def _collect_sample(self, pid: int, started_at: float) -> Dict[str, Any]:
        """Return one timestamped resource sample for ``pid``.

        Args:
            pid (int): Root PID of the monitored process tree.
            started_at (float): POSIX timestamp when the current attempt
                started. The value is used to compute attempt-relative elapsed
                time for charts.

        Returns:
            Dict[str, Any]: Sample dictionary ready for JSON serialization.
        """
        tree_processes = self._get_process_tree(pid)
        cpu_percent = self._get_tree_cpu_percent(tree_processes)
        memory_bytes = self._get_tree_memory_bytes(tree_processes)
        memory_total_bytes = self._get_total_memory_bytes()
        gpu_metrics = self._query_gpu_metrics()

        if memory_total_bytes:
            memory_percent = (memory_bytes / memory_total_bytes) * 100.0
        else:
            memory_percent = None

        return self._build_sample_payload(
            pid=pid,
            started_at=started_at,
            tree_processes=tree_processes,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_bytes=memory_bytes,
            gpu_metrics=gpu_metrics,
        )

    def _empty_gpu_metrics(self) -> Dict[str, Any]:
        """Return the explicit unavailable-payload for GPU telemetry.

        Returns:
            Dict[str, Any]: Dictionary with ``None`` aggregate GPU values and
                an empty temperature mapping. The caller uses this instead of
                inventing zeroes when ``nvidia-smi`` is unavailable or returns
                no parseable metrics.
        """
        return {
            "utilization_percent": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "device_names": {},
            "temperatures_celsius": {},
            "utilization_by_device_percent": {},
            "memory_used_by_device_mb": {},
            "memory_total_by_device_mb": {},
        }

    def _sample_epoch_seconds(self) -> float:
        """Return a single wall-clock timestamp for one sample payload.

        Returns:
            float: POSIX epoch seconds captured once so every field in the
                sample uses the same time reference.
        """
        return time.time()

    def _build_sample_payload(
        self,
        pid: int,
        started_at: float,
        tree_processes: List[psutil.Process],
        cpu_percent: Optional[float],
        memory_percent: Optional[float],
        memory_bytes: int,
        gpu_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble a JSON-serializable resource sample dictionary.

        Args:
            pid (int): Root PID of the monitored process tree.
            started_at (float): POSIX timestamp when the current attempt
                began.
            tree_processes (List[psutil.Process]): Live process tree used for
                the current sample. The length is stored so reports can show
                how many descendants were active.
            cpu_percent (Optional[float]): Summed process-tree CPU percentage
                on the total-cores scale. ``None`` means the process tree
                could not be sampled.
            memory_percent (Optional[float]): Process-tree resident memory as
                a percentage of total system RAM. ``None`` means total memory
                was unavailable.
            memory_bytes (int): Process-tree resident memory in bytes.
            gpu_metrics (Dict[str, Any]): Aggregate GPU telemetry payload. It
                must contain host-level utilization and memory fields plus the
                per-GPU temperature mapping produced by
                :meth:`_query_gpu_metrics`.

        Returns:
            Dict[str, Any]: Sample payload ready to append to ``samples.jsonl``.
        """
        sample_epoch_seconds = self._sample_epoch_seconds()
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
            "epoch_seconds": sample_epoch_seconds,
            "elapsed_seconds": sample_epoch_seconds - self._run_started_at,
            "attempt_elapsed_seconds": sample_epoch_seconds - started_at,
            "attempt_index": self._attempt_index,
            "root_pid": pid,
            "tree_process_count": len(tree_processes),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_bytes": memory_bytes,
            "memory_gb": memory_bytes / float(1024**3),
            "gpu_utilization_percent": gpu_metrics["utilization_percent"],
            "gpu_memory_used_mb": gpu_metrics["memory_used_mb"],
            "gpu_memory_total_mb": gpu_metrics["memory_total_mb"],
            "gpu_device_names": gpu_metrics["device_names"],
            "gpu_temperatures_celsius": gpu_metrics["temperatures_celsius"],
            "gpu_utilization_by_device_percent": gpu_metrics["utilization_by_device_percent"],
            "gpu_memory_used_by_device_mb": gpu_metrics["memory_used_by_device_mb"],
            "gpu_memory_total_by_device_mb": gpu_metrics["memory_total_by_device_mb"],
        }

    def _prime_cpu_counters(self, pid: int) -> None:
        """Prime ``psutil`` CPU counters for the process tree rooted at ``pid``.

        Args:
            pid (int): Root PID of the monitored process tree.

        Returns:
            None: The method initializes per-process CPU counters in place.
        """
        snapshot_time = time.time()
        for process in self._get_process_tree(pid):
            try:
                cpu_times = process.cpu_times()
                self._cpu_snapshots[process.pid] = {
                    "timestamp": snapshot_time,
                    "cpu_time_seconds": float(cpu_times.user + cpu_times.system),
                }
            except (psutil.Error, OSError):
                continue

    def _get_process_tree(self, pid: int) -> List[psutil.Process]:
        """Return the live process tree rooted at ``pid``.

        Args:
            pid (int): Root PID whose descendants should be included.

        Returns:
            List[psutil.Process]: Root process followed by currently live
                descendants. Missing or dead descendants are skipped so one
                exiting child does not invalidate the whole sample.
        """
        try:
            root_process = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.Error, OSError):
            return []

        try:
            return [root_process] + root_process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.Error, OSError):
            return [root_process]

    def _get_tree_cpu_percent(self, processes: List[psutil.Process]) -> Optional[float]:
        """Sum CPU percent across the provided live process list.

        Args:
            processes (List[psutil.Process]): Root process and descendants to
                query. An empty list means the monitored process tree no longer
                exists for this sample.

        Returns:
            Optional[float]: Summed CPU percentage, or ``None`` when no live
                processes remain.
        """
        if not processes:
            return None

        total = 0.0
        found_live_process = False
        sample_time = time.time()
        seen_pids = set()
        for process in processes:
            try:
                cpu_times = process.cpu_times()
                current_cpu_time = float(cpu_times.user + cpu_times.system)
                snapshot = self._cpu_snapshots.get(process.pid)
                seen_pids.add(process.pid)
                if snapshot is None:
                    self._cpu_snapshots[process.pid] = {
                        "timestamp": sample_time,
                        "cpu_time_seconds": current_cpu_time,
                    }
                    continue
                elapsed_time = sample_time - snapshot["timestamp"]
                cpu_delta = current_cpu_time - snapshot["cpu_time_seconds"]
                self._cpu_snapshots[process.pid] = {
                    "timestamp": sample_time,
                    "cpu_time_seconds": current_cpu_time,
                }
                if elapsed_time > 0 and cpu_delta >= 0:
                    total += (cpu_delta / elapsed_time) * 100.0
                found_live_process = True
            except (psutil.NoSuchProcess, psutil.Error, OSError):
                continue
        self._cpu_snapshots = {
            pid: snapshot for pid, snapshot in self._cpu_snapshots.items() if pid in seen_pids
        }
        if not found_live_process:
            return None
        return total

    def _get_tree_memory_bytes(self, processes: List[psutil.Process]) -> int:
        """Sum RSS memory across the provided live process list.

        Args:
            processes (List[psutil.Process]): Root process and descendants to
                query. Missing or dead processes are skipped explicitly so one
                exiting child does not invalidate the sample.

        Returns:
            int: Total resident memory in bytes across the live process tree.
        """
        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.Error, OSError):
                continue
        return total

    def _get_total_memory_bytes(self) -> Optional[int]:
        """Return total system memory in bytes when available.

        Returns:
            Optional[int]: Total physical memory in bytes, or ``None`` when the
                host memory size cannot be determined through ``psutil``.
        """
        try:
            return int(psutil.virtual_memory().total)
        except (AttributeError, psutil.Error, OSError):
            return None

    def _get_total_memory_gb(self) -> Optional[float]:
        """Return total system memory in gibibytes when available.

        Returns:
            Optional[float]: Total physical memory in GiB, or ``None`` when
                the host memory size cannot be determined through ``psutil``.
        """
        total_memory_bytes = self._get_total_memory_bytes()
        if total_memory_bytes is None:
            return None
        return total_memory_bytes / float(1024**3)

    def _get_cpu_name(self) -> str:
        """Return a human-readable CPU model name for the current host.

        Returns:
            str: CPU model string gathered from the operating system when
                possible. On Linux, ``/proc/cpuinfo`` is preferred because it
                usually exposes the full model name. If that source is
                unavailable, the method falls back to ``platform.processor()``,
                then ``platform.machine()``.
        """
        cpu_info_path = Path("/proc/cpuinfo")
        if cpu_info_path.exists():
            try:
                for raw_line in cpu_info_path.read_text().splitlines():
                    if raw_line.lower().startswith("model name"):
                        parts = raw_line.split(":", 1)
                        if len(parts) == 2:
                            cpu_name = parts[1].strip()
                            if cpu_name:
                                return cpu_name
            except OSError:
                pass

        cpu_name = platform.processor().strip()
        if cpu_name:
            return cpu_name

        machine_name = platform.machine().strip()
        if machine_name:
            return machine_name

        return "Unknown CPU"

    def _query_gpu_metrics(self) -> Dict[str, Any]:
        """Return aggregate host GPU metrics using ``nvidia-smi``.

        The session reports the highest utilization percentage across visible
        GPUs and the total used and total available GPU memory across the same
        devices. This keeps the generated charts compact and still reflects the
        host GPU load over time.

        Returns:
            Dict[str, Any]: Dictionary containing ``utilization_percent``,
                ``memory_used_mb``, ``memory_total_mb``, ``device_names``,
                and per-device metric mappings. Aggregate values are ``None``
                and nested mappings are empty when GPU telemetry is
                unavailable. Individual GPU temperatures may be ``None`` when
                a specific device omits temperature data for a sample.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return self._empty_gpu_metrics()

        if result.returncode != 0:
            return self._empty_gpu_metrics()

        utilization_values: List[float] = []
        memory_used_values: List[float] = []
        memory_total_values: List[float] = []
        device_names: Dict[str, str] = {}
        temperatures_celsius: Dict[str, Optional[float]] = {}
        utilization_by_device_percent: Dict[str, float] = {}
        memory_used_by_device_mb: Dict[str, float] = {}
        memory_total_by_device_mb: Dict[str, float] = {}

        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                continue

            try:
                gpu_index = int(parts[0])
                gpu_display_name = parts[1]
                utilization_value = float(parts[2])
                memory_used_value = float(parts[3])
                memory_total_value = float(parts[4])
            except ValueError:
                continue

            try:
                gpu_temperature = float(parts[5])
            except ValueError:
                gpu_temperature = None

            gpu_name = f"gpu_{gpu_index}"
            utilization_values.append(utilization_value)
            memory_used_values.append(memory_used_value)
            memory_total_values.append(memory_total_value)
            device_names[gpu_name] = gpu_display_name
            temperatures_celsius[gpu_name] = gpu_temperature
            utilization_by_device_percent[gpu_name] = utilization_value
            memory_used_by_device_mb[gpu_name] = memory_used_value
            memory_total_by_device_mb[gpu_name] = memory_total_value

        if not utilization_values:
            return self._empty_gpu_metrics()

        return {
            "utilization_percent": max(utilization_values),
            "memory_used_mb": sum(memory_used_values),
            "memory_total_mb": sum(memory_total_values),
            "device_names": device_names,
            "temperatures_celsius": temperatures_celsius,
            "utilization_by_device_percent": utilization_by_device_percent,
            "memory_used_by_device_mb": memory_used_by_device_mb,
            "memory_total_by_device_mb": memory_total_by_device_mb,
        }

    def _write_event(self, message: str) -> None:
        """Append a timestamped event line to the human-readable log file.

        Args:
            message (str): Event text to append.

        Returns:
            None: The method writes one line to ``log_file``.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
        with open(self.log_file, "a") as file_pointer:
            file_pointer.write(f"[{timestamp}] {message}\n")

    def _write_restarts_file(self) -> None:
        """Persist restart metadata as JSON.

        Returns:
            None: The method writes ``restarts.json`` under the report
            directory.
        """
        with open(self.restarts_file, "w") as file_pointer:
            json.dump(self._restart_events, file_pointer, indent=2)
            file_pointer.write("\n")

    def _enrich_restart_events(self) -> None:
        """Attach boundary samples to restart events for graph reconstruction.

        Returns:
            None: The method mutates restart-event dictionaries in place.
        """
        for event in self._restart_events:
            old_attempt_index = event.get("old_attempt_index")
            new_attempt_index = event.get("new_attempt_index")
            crash_sample = self._build_attempt_end_boundary_sample(old_attempt_index)
            restart_sample = self._build_attempt_start_boundary_sample(
                new_attempt_index,
                fallback_sample=crash_sample,
                restart_elapsed_seconds=event.get("restart_elapsed_seconds"),
                new_pid=event.get("new_pid"),
            )
            event["crash_sample"] = crash_sample
            event["restart_sample"] = restart_sample

    def _write_summary_files(self, summary: Dict[str, Any]) -> None:
        """Persist machine-readable and text summaries for the full run.

        Args:
            summary (Dict[str, Any]): Summary payload prepared by
                :meth:`finalize`.

        Returns:
            None: The method writes ``summary.json`` and ``summary.txt``.
        """
        with open(self.summary_json_file, "w") as file_pointer:
            json.dump(summary, file_pointer, indent=2)
            file_pointer.write("\n")

        aggregate_metrics = summary["aggregate_metrics"]
        lines = [
            f"Source file: {summary['source_path']}",
            f"Title: {summary['title']}",
            f"CPU: {summary['cpu_name']}",
            f"System RAM total (GB): {summary['system_memory_total_gb']}",
            f"Final status: {summary['final_status']}",
            f"Final error: {summary['final_error']}",
            f"Total runtime (s): {summary['total_runtime_seconds']}",
            f"Total restarts: {summary['total_restarts']}",
            f"Crash/failure restarts: {summary['crash_restarts']}",
            f"Scheduled restarts: {summary['scheduled_restarts']}",
            f"PID history: {summary['pid_history']}",
            f"Attempt events: {len(summary['attempt_events'])}",
            f"Restart events: {len(summary['restart_events'])}",
            "Aggregate metrics:",
            f"  CPU percent avg/max: {aggregate_metrics['cpu_percent']['average']} / {aggregate_metrics['cpu_percent']['max']}",
            f"  Memory GB avg/max: {aggregate_metrics['memory_gb']['average']} / {aggregate_metrics['memory_gb']['max']}",
            f"  GPU util avg/max: {aggregate_metrics['gpu_utilization_percent']['average']} / {aggregate_metrics['gpu_utilization_percent']['max']}",
            f"  GPU memory MB avg/max: {aggregate_metrics['gpu_memory_used_mb']['average']} / {aggregate_metrics['gpu_memory_used_mb']['max']}",
        ]
        gpu_temperature_summary = summary["gpu_temperature_summary"]
        if gpu_temperature_summary:
            lines.append("  GPU temperatures C avg/max by device:")
            for gpu_name in sorted(gpu_temperature_summary):
                device_summary = gpu_temperature_summary[gpu_name]
                device_label = self._format_gpu_display_name(gpu_name)
                lines.append(
                    f"    {device_label}: {device_summary['average']} / {device_summary['max']}"
                )
        with open(self.summary_text_file, "w") as file_pointer:
            file_pointer.write("\n".join(lines) + "\n")

    def _build_aggregate_metrics(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Compute summary statistics for sampled metrics.

        Returns:
            Dict[str, Dict[str, Optional[float]]]: Mapping from metric name to
                simple summary statistics. Each metric stores ``average`` and
                ``max`` values. Metrics with no numeric samples use ``None``.
        """
        return {
            "cpu_percent": self._summarize_numeric_metric("cpu_percent"),
            "memory_gb": self._summarize_numeric_metric("memory_gb"),
            "gpu_utilization_percent": self._summarize_numeric_metric("gpu_utilization_percent"),
            "gpu_memory_used_mb": self._summarize_numeric_metric("gpu_memory_used_mb"),
        }

    def _build_gpu_temperature_summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Compute per-GPU temperature summary statistics.

        Returns:
            Dict[str, Dict[str, Optional[float]]]: Mapping from GPU name such
                as ``"gpu_0"`` to ``average`` and ``max`` values in degrees
                Celsius. GPUs with only missing samples are omitted so the
                summary does not pretend a device produced temperature data.
        """
        gpu_names = self._collect_gpu_device_names("gpu_temperatures_celsius")
        summary: Dict[str, Dict[str, Optional[float]]] = {}
        for gpu_name in gpu_names:
            values: List[float] = []
            for sample in self._samples:
                temperatures = sample.get("gpu_temperatures_celsius")
                if not isinstance(temperatures, dict):
                    continue
                temperature_value = temperatures.get(gpu_name)
                if isinstance(temperature_value, (int, float)) and not math.isnan(
                    temperature_value
                ):
                    values.append(float(temperature_value))
            if values:
                summary[gpu_name] = {
                    "average": sum(values) / len(values),
                    "max": max(values),
                }
        return summary

    def _summarize_numeric_metric(self, key: str) -> Dict[str, Optional[float]]:
        """Return average and max statistics for one metric key.

        Args:
            key (str): Sample dictionary key to summarize.

        Returns:
            Dict[str, Optional[float]]: Two-field dictionary containing
                ``average`` and ``max``. Both are ``None`` when no numeric
                values are present.
        """
        values = [sample[key] for sample in self._samples if isinstance(sample.get(key), (int, float))]
        if not values:
            return {"average": None, "max": None}
        return {
            "average": sum(values) / len(values),
            "max": max(values),
        }

    def _write_plots(self) -> None:
        """Render time-series plots for sampled metrics.

        Returns:
            None: The method writes PNG charts into ``output_directory`` when
                at least one sample exists for the corresponding metric.
        """
        if not self._samples:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._apply_plot_style(plt)
        self._write_single_plot(
            plt,
            "cpu.png",
            "CPU Usage Over Time",
            "CPU Percent",
            "cpu_percent",
        )
        self._write_single_plot(
            plt,
            "memory.png",
            "Memory Usage Over Time",
            "Memory (GB)",
            "memory_gb",
        )
        self._write_gpu_device_plot_series(
            plt,
            value_key="gpu_utilization_by_device_percent",
            filename_suffix="utilization",
            title_suffix="Utilization Over Time",
            ylabel="GPU Utilization (%)",
        )
        self._write_gpu_device_plot_series(
            plt,
            value_key="gpu_memory_used_by_device_mb",
            filename_suffix="memory",
            title_suffix="Memory Usage Over Time",
            ylabel="GPU Memory (MB)",
        )
        self._write_gpu_device_plot_series(
            plt,
            value_key="gpu_temperatures_celsius",
            filename_suffix="temperature",
            title_suffix="Temperature Over Time",
            ylabel="Temperature (C)",
        )

    def _write_single_plot(
        self,
        plt_module: Any,
        filename: str,
        title: str,
        ylabel: str,
        value_key: str,
    ) -> None:
        """Render one PNG chart for a single sampled metric.

        Args:
            plt_module (Any): Imported ``matplotlib.pyplot`` module used for
                chart creation. Passing the module keeps imports localized to
                chart generation only.
            filename (str): Output PNG filename inside ``output_directory``.
            title (str): Chart title shown above the plot.
            ylabel (str): Axis label describing the metric units.
            value_key (str): Sample dictionary key whose values should be
                charted over elapsed time.

        Returns:
            None: The method writes one PNG file when numeric data exists for
                ``value_key``. If all values are missing the plot is skipped.
        """
        points_by_attempt = self._build_plot_points_by_attempt(value_key)
        if not points_by_attempt:
            return

        figure = plt_module.figure(figsize=(8, 4.5))
        axis = figure.add_subplot(1, 1, 1)
        legend_handles = []

        for attempt_index in sorted(points_by_attempt):
            x_values = [point[0] for point in points_by_attempt[attempt_index]]
            y_values = [point[1] for point in points_by_attempt[attempt_index]]
            (line_handle,) = axis.plot(
                x_values,
                y_values,
                linewidth=1.6,
                color=PLOT_PALETTE["primary"],
            )
            if attempt_index == 1:
                legend_handles.append((line_handle, "Metric"))

        crash_points = self._collect_boundary_points("crash_sample", value_key)
        if crash_points:
            crash_handle = axis.scatter(
                [point[0] for point in crash_points],
                [point[1] for point in crash_points],
                color=PLOT_PALETTE["highlight"],
                s=38,
                zorder=4,
            )
            legend_handles.append((crash_handle, "Crash"))

        self._style_plot_axes(axis, title, ylabel)
        if legend_handles:
            axis.legend(
                [handle for handle, _ in legend_handles],
                [label for _, label in legend_handles],
                loc="best",
            )
        figure.tight_layout()
        figure.savefig(self.output_directory / filename)
        plt_module.close(figure)

    def _write_gpu_device_plot_series(
        self,
        plt_module: Any,
        value_key: str,
        filename_suffix: str,
        title_suffix: str,
        ylabel: str,
    ) -> None:
        """Render one separate chart per GPU for a nested metric series.

        Args:
            plt_module (Any): Imported ``matplotlib.pyplot`` module used for
                figure creation and saving.
            value_key (str): Sample dictionary key containing a per-device
                metric mapping.
            filename_suffix (str): Suffix appended to the generated filename.
            title_suffix (str): Human-readable metric title suffix appended to
                the GPU label.
            ylabel (str): Metric label shown on the y-axis.

        Returns:
            None: Writes one PNG per GPU when numeric samples exist for that
                metric/device pair. Devices with only missing values are
                skipped so the report does not fabricate continuous series.
        """
        gpu_names = self._collect_gpu_device_names(value_key)
        if not gpu_names:
            return

        for gpu_name in gpu_names:
            points_by_attempt = self._build_nested_plot_points_by_attempt(value_key, gpu_name)
            if not points_by_attempt:
                continue

            figure = plt_module.figure(figsize=(8, 4.5))
            axis = figure.add_subplot(1, 1, 1)
            legend_handles = []

            for attempt_index in sorted(points_by_attempt):
                x_values = [point[0] for point in points_by_attempt[attempt_index]]
                y_values = [point[1] for point in points_by_attempt[attempt_index]]
                (line_handle,) = axis.plot(
                    x_values,
                    y_values,
                    linewidth=1.6,
                    color=PLOT_PALETTE["primary"],
                )
                if attempt_index == 1:
                    legend_handles.append((line_handle, "Metric"))

            crash_points = self._collect_boundary_points("crash_sample", value_key)
            gpu_crash_points = [
                (elapsed_seconds, value)
                for elapsed_seconds, metric_name, value in crash_points
                if metric_name == gpu_name
            ]
            if gpu_crash_points:
                crash_handle = axis.scatter(
                    [point[0] for point in gpu_crash_points],
                    [point[1] for point in gpu_crash_points],
                    color=PLOT_PALETTE["highlight"],
                    s=38,
                    zorder=4,
                )
                legend_handles.append((crash_handle, "Crash"))

            self._style_plot_axes(
                axis,
                f"{self._format_gpu_display_name(gpu_name)} {title_suffix}",
                ylabel,
            )
            if legend_handles:
                axis.legend(
                    [handle for handle, _ in legend_handles],
                    [label for _, label in legend_handles],
                    loc="best",
                )
            figure.tight_layout()
            figure.savefig(
                self.output_directory
                / f"{self._format_gpu_filename_prefix(gpu_name)}_{filename_suffix}.png"
            )
            plt_module.close(figure)

    def _apply_plot_style(self, plt_module: Any) -> None:
        """Apply the runtime-local NAS-inspired Matplotlib theme.

        Args:
            plt_module (Any): Imported ``matplotlib.pyplot`` module that owns
                the process-local ``rcParams`` to update.

        Returns:
            None: The function updates Matplotlib defaults in place.
        """
        plt_module.rcParams.update(
            {
                "figure.facecolor": PLOT_PALETTE["background"],
                "axes.facecolor": PLOT_PALETTE["background"],
                "axes.edgecolor": PLOT_PALETTE["axis"],
                "axes.linewidth": 0.8,
                "axes.labelcolor": PLOT_PALETTE["axis"],
                "axes.labelsize": 16,
                "axes.titlesize": 20,
                "axes.titleweight": "semibold",
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "font.size": 16,
                "grid.color": PLOT_PALETTE["grid"],
                "grid.linewidth": 0.6,
                "grid.alpha": 0.65,
                "legend.frameon": False,
                "legend.fontsize": 14,
                "xtick.color": PLOT_PALETTE["axis"],
                "ytick.color": PLOT_PALETTE["axis"],
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "savefig.facecolor": PLOT_PALETTE["background"],
            }
        )

    def _style_plot_axes(self, axis: Any, title: str, ylabel: str) -> None:
        """Apply consistent axis styling after plotting metric data.

        Args:
            axis (Any): Matplotlib axes object to style.
            title (str): Plot title shown above the metric chart.
            ylabel (str): Metric label shown on the y-axis.

        Returns:
            None: Styles the provided axes in place.
        """
        axis_unit_label = self._select_time_axis_scale_label()
        axis.set_title(title, pad=10.0)
        axis.set_xlabel(f"Elapsed Time ({axis_unit_label})")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="both", linestyle="--", dashes=(2.0, 2.4))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=4.0, width=0.8)

    def _build_plot_points_by_attempt(self, value_key: str) -> Dict[int, List[Tuple[float, float]]]:
        """Group numeric samples by attempt for gap-aware plotting.

        Args:
            value_key (str): Sample dictionary key whose numeric values should
                be plotted.

        Returns:
            Dict[int, List[Tuple[float, float]]]: Mapping from attempt index to
                elapsed-time/value points for that attempt only.
        """
        points_by_attempt: Dict[int, List[Tuple[float, float]]] = {}
        axis_scale_seconds = self._select_time_axis_scale_seconds()
        for sample in self._samples:
            value = sample.get(value_key)
            if not isinstance(value, (int, float)) or math.isnan(value):
                continue
            points_by_attempt.setdefault(sample["attempt_index"], []).append(
                (sample["elapsed_seconds"] / axis_scale_seconds, value)
            )
        self._append_scalar_crash_boundary_points(points_by_attempt, value_key, axis_scale_seconds)
        return points_by_attempt

    def _build_nested_plot_points_by_attempt(
        self,
        value_key: str,
        gpu_name: str,
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Group numeric nested GPU metric samples by attempt for one device.

        Args:
            value_key (str): Sample dictionary key containing a per-device
                metric mapping.
            gpu_name (str): GPU-series key such as ``"gpu_0"``.

        Returns:
            Dict[int, List[Tuple[float, float]]]: Mapping from attempt index to
                elapsed-time/value pairs for the selected GPU only. Missing
                samples are skipped so each unavailable interval produces a
                visible gap.
        """
        points_by_attempt: Dict[int, List[Tuple[float, float]]] = {}
        axis_scale_seconds = self._select_time_axis_scale_seconds()
        for sample in self._samples:
            metric_mapping = sample.get(value_key)
            if not isinstance(metric_mapping, dict):
                continue
            value = metric_mapping.get(gpu_name)
            if not isinstance(value, (int, float)) or math.isnan(value):
                continue
            points_by_attempt.setdefault(sample["attempt_index"], []).append(
                (sample["elapsed_seconds"] / axis_scale_seconds, float(value))
            )
        self._append_nested_crash_boundary_points(
            points_by_attempt,
            value_key,
            gpu_name,
            axis_scale_seconds,
        )
        return points_by_attempt

    def _append_scalar_crash_boundary_points(
        self,
        points_by_attempt: Dict[int, List[Tuple[float, float]]],
        value_key: str,
        axis_scale_seconds: float,
    ) -> None:
        """Append scalar crash-boundary points to the plotted attempt series.

        Args:
            points_by_attempt (Dict[int, List[Tuple[float, float]]]): Existing
                elapsed-time/value series grouped by attempt. This mapping is
                mutated in place so the black line reaches the crash marker.
            value_key (str): Scalar metric key such as ``"cpu_percent"`` whose
                crash-boundary value should be added to the attempt line.
            axis_scale_seconds (float): Seconds-per-unit divisor used by the
                current plot so appended crash points share the same x-axis
                scale as normal samples.

        Returns:
            None: The method mutates ``points_by_attempt`` in place.
        """
        for event in self._restart_events:
            crash_sample = event.get("crash_sample")
            if not isinstance(crash_sample, dict):
                continue
            attempt_index = crash_sample.get("attempt_index")
            metric_value = crash_sample.get(value_key)
            elapsed_seconds = crash_sample.get("elapsed_seconds")
            if not isinstance(attempt_index, int):
                continue
            if not isinstance(metric_value, (int, float)) or math.isnan(metric_value):
                continue
            if not isinstance(elapsed_seconds, (int, float)):
                continue
            points_by_attempt.setdefault(attempt_index, []).append(
                (float(elapsed_seconds) / axis_scale_seconds, float(metric_value))
            )
            points_by_attempt[attempt_index].sort(key=lambda point: point[0])

    def _append_nested_crash_boundary_points(
        self,
        points_by_attempt: Dict[int, List[Tuple[float, float]]],
        value_key: str,
        gpu_name: str,
        axis_scale_seconds: float,
    ) -> None:
        """Append nested GPU crash-boundary points to one device series.

        Args:
            points_by_attempt (Dict[int, List[Tuple[float, float]]]): Existing
                elapsed-time/value series for one GPU device, grouped by
                attempt. This mapping is mutated in place so the plotted line
                reaches the crash marker.
            value_key (str): Nested metric key containing per-device values.
            gpu_name (str): GPU-series key such as ``"gpu_0"`` whose crash
                point should be added to the corresponding device line.
            axis_scale_seconds (float): Seconds-per-unit divisor used by the
                current plot so appended crash points share the same x-axis
                scale as normal samples.

        Returns:
            None: The method mutates ``points_by_attempt`` in place.
        """
        for event in self._restart_events:
            crash_sample = event.get("crash_sample")
            if not isinstance(crash_sample, dict):
                continue
            attempt_index = crash_sample.get("attempt_index")
            metric_mapping = crash_sample.get(value_key)
            elapsed_seconds = crash_sample.get("elapsed_seconds")
            if not isinstance(attempt_index, int):
                continue
            if not isinstance(metric_mapping, dict):
                continue
            metric_value = metric_mapping.get(gpu_name)
            if not isinstance(metric_value, (int, float)) or math.isnan(metric_value):
                continue
            if not isinstance(elapsed_seconds, (int, float)):
                continue
            points_by_attempt.setdefault(attempt_index, []).append(
                (float(elapsed_seconds) / axis_scale_seconds, float(metric_value))
            )
            points_by_attempt[attempt_index].sort(key=lambda point: point[0])

    def _collect_boundary_points(
        self,
        event_key: str,
        value_key: str,
    ) -> List[Any]:
        """Collect plot-ready restart boundary markers for one metric.

        Args:
            event_key (str): Restart-event sample key to inspect. Supported
                values are ``"crash_sample"`` and ``"restart_sample"``.
            value_key (str): Metric value key to extract from each boundary
                sample.

        Returns:
            List[Any]: Elapsed-time/value marker points. Numeric metrics return
                ``(elapsed_seconds, value)`` tuples. The GPU temperature metric
                returns ``(elapsed_seconds, gpu_name, value)`` tuples so each
                GPU line can break independently when a temperature sample is
                missing.
        """
        points: List[Any] = []
        axis_scale_seconds = self._select_time_axis_scale_seconds()
        for event in self._restart_events:
            sample = event.get(event_key)
            if not isinstance(sample, dict):
                continue
            value = sample.get(value_key)
            if value_key in {
                "gpu_temperatures_celsius",
                "gpu_utilization_by_device_percent",
                "gpu_memory_used_by_device_mb",
            }:
                if not isinstance(value, dict):
                    continue
                elapsed_seconds = sample["elapsed_seconds"] / axis_scale_seconds
                for gpu_name, gpu_value in value.items():
                    if not isinstance(gpu_value, (int, float)) or math.isnan(gpu_value):
                        continue
                    points.append((elapsed_seconds, gpu_name, float(gpu_value)))
                continue
            if not isinstance(value, (int, float)) or math.isnan(value):
                continue
            points.append((sample["elapsed_seconds"] / axis_scale_seconds, value))
        return points

    def _build_attempt_end_boundary_sample(self, attempt_index: Optional[int]) -> Optional[Dict[str, Any]]:
        """Return a crash-end sample synthesized from the last known attempt data.

        Args:
            attempt_index (Optional[int]): Attempt index whose end boundary
                should be synthesized.

        Returns:
            Optional[Dict[str, Any]]: Boundary sample dictionary or ``None``
                when the attempt index is unavailable.
        """
        if attempt_index is None:
            return None
        attempt_event = self._find_attempt_event(attempt_index)
        if attempt_event is None:
            return None
        last_sample = self._find_last_sample_for_attempt(attempt_index)
        if last_sample is None:
            return {
                "elapsed_seconds": attempt_event["elapsed_seconds"],
                "attempt_index": attempt_index,
                "root_pid": attempt_event.get("pid"),
                "cpu_percent": None,
                "memory_gb": None,
                "gpu_utilization_percent": None,
                "gpu_memory_used_mb": None,
                "gpu_device_names": {},
                "gpu_temperatures_celsius": {},
                "gpu_utilization_by_device_percent": {},
                "gpu_memory_used_by_device_mb": {},
                "gpu_memory_total_by_device_mb": {},
            }
        boundary_sample = self._clone_sample_with_elapsed(last_sample, attempt_event["elapsed_seconds"])
        return self._backfill_boundary_sample(boundary_sample, attempt_index)

    def _build_attempt_start_boundary_sample(
        self,
        attempt_index: Optional[int],
        fallback_sample: Optional[Dict[str, Any]],
        restart_elapsed_seconds: Optional[float],
        new_pid: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Return a restart-start sample for the next attempt boundary.

        Args:
            attempt_index (Optional[int]): Attempt index of the restarted run.
            fallback_sample (Optional[Dict[str, Any]]): Crash-end sample to use
                when the restarted attempt has not yet produced a metric sample.
            restart_elapsed_seconds (Optional[float]): Total elapsed time when
                the restarted attempt began.
            new_pid (Optional[int]): Root PID of the restarted attempt.

        Returns:
            Optional[Dict[str, Any]]: Boundary sample dictionary or ``None``
                when the restart never reached a new attempt.
        """
        if attempt_index is None or restart_elapsed_seconds is None:
            return None
        first_sample = self._find_first_sample_for_attempt(attempt_index)
        if first_sample is not None:
            return self._clone_sample_with_elapsed(first_sample, restart_elapsed_seconds)
        if fallback_sample is None:
            return {
                "elapsed_seconds": restart_elapsed_seconds,
                "attempt_index": attempt_index,
                "root_pid": new_pid,
                "cpu_percent": None,
                "memory_gb": None,
                "gpu_utilization_percent": None,
                "gpu_memory_used_mb": None,
                "gpu_device_names": {},
                "gpu_temperatures_celsius": {},
                "gpu_utilization_by_device_percent": {},
                "gpu_memory_used_by_device_mb": {},
                "gpu_memory_total_by_device_mb": {},
            }
        synthesized_sample = dict(fallback_sample)
        synthesized_sample["elapsed_seconds"] = restart_elapsed_seconds
        synthesized_sample["attempt_index"] = attempt_index
        synthesized_sample["root_pid"] = new_pid
        return synthesized_sample

    def _find_attempt_event(self, attempt_index: int) -> Optional[Dict[str, Any]]:
        """Return the recorded attempt-end event for ``attempt_index``.

        Args:
            attempt_index (int): Attempt index whose completion event should be
                looked up.

        Returns:
            Optional[Dict[str, Any]]: Matching attempt event or ``None`` when
                no completion event has been recorded for that attempt.
        """
        for event in self._attempt_events:
            if event["attempt_index"] == attempt_index:
                return event
        return None

    def _find_last_sample_for_attempt(self, attempt_index: int) -> Optional[Dict[str, Any]]:
        """Return the final recorded metric sample for ``attempt_index``.

        Args:
            attempt_index (int): Attempt index to inspect.

        Returns:
            Optional[Dict[str, Any]]: Last sample for the attempt or ``None``
                when the attempt never produced a sample.
        """
        for sample in reversed(self._samples):
            if sample["attempt_index"] == attempt_index:
                return sample
        return None

    def _find_first_sample_for_attempt(self, attempt_index: int) -> Optional[Dict[str, Any]]:
        """Return the first recorded metric sample for ``attempt_index``.

        Args:
            attempt_index (int): Attempt index to inspect.

        Returns:
            Optional[Dict[str, Any]]: First sample for the attempt or ``None``
                when the attempt never produced a sample.
        """
        for sample in self._samples:
            if sample["attempt_index"] == attempt_index:
                return sample
        return None

    def _clone_sample_with_elapsed(
        self,
        sample: Dict[str, Any],
        elapsed_seconds: float,
    ) -> Dict[str, Any]:
        """Clone a sample while overriding only the elapsed timestamp.

        Args:
            sample (Dict[str, Any]): Original sample dictionary to copy.
            elapsed_seconds (float): Replacement run-relative elapsed time in
                seconds for the cloned boundary sample.

        Returns:
            Dict[str, Any]: Shallow copy of ``sample`` with the requested
                elapsed time and a copied temperature mapping so boundary
                metadata is insulated from later mutations.
        """
        cloned_sample = dict(sample)
        cloned_sample["elapsed_seconds"] = elapsed_seconds
        device_names = sample.get("gpu_device_names")
        if isinstance(device_names, dict):
            cloned_sample["gpu_device_names"] = dict(device_names)
        temperatures = sample.get("gpu_temperatures_celsius")
        if isinstance(temperatures, dict):
            cloned_sample["gpu_temperatures_celsius"] = dict(temperatures)
        for metric_key in (
            "gpu_utilization_by_device_percent",
            "gpu_memory_used_by_device_mb",
            "gpu_memory_total_by_device_mb",
        ):
            metric_mapping = sample.get(metric_key)
            if isinstance(metric_mapping, dict):
                cloned_sample[metric_key] = dict(metric_mapping)
        return cloned_sample

    def _backfill_boundary_sample(
        self,
        boundary_sample: Dict[str, Any],
        attempt_index: int,
    ) -> Dict[str, Any]:
        """Backfill missing crash-boundary metrics from earlier attempt data.

        Args:
            boundary_sample (Dict[str, Any]): Boundary sample derived from the
                final recorded sample of an attempt.
            attempt_index (int): Attempt whose earlier samples should be used
                to recover metrics that became unavailable only after the
                process tree had already disappeared.

        Returns:
            Dict[str, Any]: Boundary sample with missing scalar and mapping
                metrics replaced by the last non-null values recorded earlier
                in the same attempt.
        """
        for metric_key in (
            "cpu_percent",
            "memory_percent",
            "memory_bytes",
            "memory_gb",
            "gpu_utilization_percent",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
        ):
            if boundary_sample.get(metric_key) is not None:
                continue
            metric_value = self._find_last_non_null_metric_value_for_attempt(
                attempt_index,
                metric_key,
            )
            if metric_value is not None:
                boundary_sample[metric_key] = metric_value

        for metric_key in (
            "gpu_temperatures_celsius",
            "gpu_utilization_by_device_percent",
            "gpu_memory_used_by_device_mb",
            "gpu_memory_total_by_device_mb",
        ):
            metric_mapping = boundary_sample.get(metric_key)
            if isinstance(metric_mapping, dict) and metric_mapping:
                continue
            previous_mapping = self._find_last_non_empty_mapping_for_attempt(
                attempt_index,
                metric_key,
            )
            if previous_mapping:
                boundary_sample[metric_key] = previous_mapping
        device_names = boundary_sample.get("gpu_device_names")
        if not isinstance(device_names, dict) or not device_names:
            previous_device_names = self._find_last_non_empty_mapping_for_attempt(
                attempt_index,
                "gpu_device_names",
            )
            if previous_device_names:
                boundary_sample["gpu_device_names"] = previous_device_names
        return boundary_sample

    def _find_last_non_null_metric_value_for_attempt(
        self,
        attempt_index: int,
        metric_key: str,
    ) -> Optional[float]:
        """Return the last non-null scalar metric for one attempt.

        Args:
            attempt_index (int): Attempt index to inspect.
            metric_key (str): Scalar sample key whose value should be
                backfilled if the crash-boundary sample lost it.

        Returns:
            Optional[float]: The last numeric value recorded for the metric in
                that attempt, or ``None`` when no earlier numeric value exists.
        """
        for sample in reversed(self._samples):
            if sample["attempt_index"] != attempt_index:
                continue
            metric_value = sample.get(metric_key)
            if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                return float(metric_value)
        return None

    def _find_last_non_empty_mapping_for_attempt(
        self,
        attempt_index: int,
        metric_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the last non-empty mapping metric for one attempt.

        Args:
            attempt_index (int): Attempt index to inspect.
            metric_key (str): Sample key whose value should be a mapping.

        Returns:
            Optional[Dict[str, Any]]: Copy of the last non-empty mapping for
                the selected attempt, or ``None`` when none exists.
        """
        for sample in reversed(self._samples):
            if sample["attempt_index"] != attempt_index:
                continue
            metric_mapping = sample.get(metric_key)
            if isinstance(metric_mapping, dict) and metric_mapping:
                return dict(metric_mapping)
        return None

    def _collect_gpu_device_names(self, value_key: str) -> List[str]:
        """Return the stable sorted set of GPU device names for one metric.

        Returns:
            List[str]: Sorted GPU keys gathered from samples and restart
                boundary samples for the requested nested GPU metric.
        """
        gpu_names = set()
        for sample in self._samples:
            metric_mapping = sample.get(value_key)
            if isinstance(metric_mapping, dict):
                gpu_names.update(metric_mapping.keys())
        for event in self._restart_events:
            for boundary_key in ("crash_sample", "restart_sample"):
                sample = event.get(boundary_key)
                if not isinstance(sample, dict):
                    continue
                metric_mapping = sample.get(value_key)
                if isinstance(metric_mapping, dict):
                    gpu_names.update(metric_mapping.keys())
        return sorted(gpu_names)

    def _format_gpu_filename_prefix(self, gpu_name: str) -> str:
        """Return the compact filename prefix used for one GPU graph.

        Args:
            gpu_name (str): Internal GPU key such as ``"gpu_0"``.

        Returns:
            str: Filesystem-safe prefix such as ``"gpu0"``.
        """
        return gpu_name.replace("_", "")

    def _format_gpu_display_name(self, gpu_name: str) -> str:
        """Return the display label used in per-GPU graphs and summaries.

        Args:
            gpu_name (str): Internal GPU key such as ``"gpu_0"``.

        Returns:
            str: Sampled GPU device name when available. If the monitor has
                not yet recorded a device name for this GPU, the method falls
                back to a generic ``GPU 0`` style label.
        """
        for sample in reversed(self._samples):
            device_names = sample.get("gpu_device_names")
            if not isinstance(device_names, dict):
                continue
            device_name = device_names.get(gpu_name)
            if isinstance(device_name, str) and device_name:
                return device_name
        for event in reversed(self._restart_events):
            crash_sample = event.get("crash_sample")
            if not isinstance(crash_sample, dict):
                continue
            device_names = crash_sample.get("gpu_device_names")
            if not isinstance(device_names, dict):
                continue
            device_name = device_names.get(gpu_name)
            if isinstance(device_name, str) and device_name:
                return device_name
        return gpu_name.replace("gpu_", "GPU ")

    def _select_time_axis_scale(self) -> Tuple[str, float]:
        """Choose one time-axis unit for the full report based on run span.

        Returns:
            Tuple[str, float]: Axis-unit label and the number of seconds per
                unit. Short runs use seconds, medium runs use minutes, longer
                runs use hours, and very long runs use days.
        """
        total_elapsed_seconds = max(
            [0.0] + [float(sample.get("elapsed_seconds", 0.0)) for sample in self._samples]
        )
        for axis_label, axis_scale_seconds, upper_bound_seconds in TIME_AXIS_SCALES:
            if total_elapsed_seconds <= upper_bound_seconds:
                return axis_label, axis_scale_seconds
        return "d", 86400.0

    def _select_time_axis_scale_label(self) -> str:
        """Return the chosen x-axis unit label for the current report.

        Returns:
            str: Time-unit label selected by :meth:`_select_time_axis_scale`.
        """
        return self._select_time_axis_scale()[0]

    def _select_time_axis_scale_seconds(self) -> float:
        """Return the chosen x-axis scale factor in seconds per unit.

        Returns:
            float: Seconds-per-unit divisor selected by
                :meth:`_select_time_axis_scale`.
        """
        return self._select_time_axis_scale()[1]
