"""Monitor target that varies CPU and GPU load over a fixed runtime window.

This script is intended for manual or integration-style monitor validation.
It creates deliberate CPU pressure on the host and CUDA activity on the first
GPU so the monitor command can be checked against a workload whose resource
usage changes over time.
"""

from typing import Any, Optional

import math
import multiprocessing
import os
import time


MONITOR_TEST_DURATION_SECONDS = 30
"""int: Total runtime for the monitor target in seconds."""

LOAD_UPDATE_INTERVAL_SECONDS = 0.1
"""float: Delay between load-shape updates in seconds."""

CPU_MATRIX_SIZE = 256
"""int: Width and height of the CPU-side matrix used for repeated math work."""

GPU_MATRIX_SIZE = 2048
"""int: Width and height of the CUDA matrices used to vary GPU load."""

CPU_IDLE_ITERATIONS = 8
"""int: CPU inner-loop count during the low-load phase."""

CPU_BUSY_ITERATIONS = 9000
"""int: CPU inner-loop count during the high-load burst phase."""

CPU_PEAK_ITERATIONS = 22000
"""int: CPU inner-loop count during the most aggressive burst phase."""

CPU_WORKER_COUNT = max(2, min(8, (os.cpu_count() or 2) - 1))
"""int: Number of CPU worker processes that run in parallel."""

GPU_IDLE_REPETITIONS = 1
"""int: Number of CUDA matrix passes during the low-load phase."""

GPU_BUSY_REPETITIONS = 18
"""int: Number of CUDA matrix passes during the high-load burst phase."""

GPU_PEAK_REPETITIONS = 36
"""int: Number of CUDA matrix passes during the heaviest GPU burst phase."""

MEMORY_LOW_WATERMARK_MB = 64
"""int: Approximate retained CPU memory in megabytes during low-load phases."""

MEMORY_HIGH_WATERMARK_MB = 3072
"""int: Approximate retained CPU memory in megabytes during burst phases."""

MEMORY_PEAK_WATERMARK_MB = 6144
"""int: Approximate retained CPU memory in megabytes during burst peaks."""

MEMORY_CHUNK_MB = 128
"""int: Allocation unit used to grow and shrink the retained memory buffer."""


def _require_torch_cuda():
    """Import Torch and fail explicitly when CUDA execution is unavailable.

    Returns:
        Any: Imported ``torch`` module ready for CUDA tensor allocation.

    Raises:
        ImportError: If Torch is not installed in the current interpreter.
        RuntimeError: If CUDA is unavailable or no GPU device is visible.
    """
    try:
        import torch
    except ModuleNotFoundError as error:
        raise ImportError(
            "tests/cpu_gpu_varying_load.py requires torch to run."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "tests/cpu_gpu_varying_load.py requires a CUDA-capable GPU."
        )

    return torch


def _cpu_wave_step(angle_radians: float) -> float:
    """Execute one CPU-heavy step whose intensity varies with ``angle_radians``.

    Args:
        angle_radians (float): Phase angle in radians that shapes how many
            inner-loop operations will run. Lower angles reduce CPU work,
            while higher angles increase the amount of repeated trigonometric
            computation performed in this step.

    Returns:
        float: Accumulated floating-point result from the CPU work. The caller
            can feed this into later calculations to discourage aggressive
            dead-code elimination.

    Raises:
        RuntimeError: Not raised by this helper.
    """
    phase_strength = (math.sin(angle_radians) + 1.0) / 2.0

    # The monitor needs large visible swings, so the CPU work toggles between
    # a light phase and a burst phase rather than changing only slightly.
    burst_gate = (math.sin(angle_radians * 3.0) + math.cos(angle_radians * 5.0)) / 2.0
    if phase_strength < 0.2:
        iterations = CPU_IDLE_ITERATIONS
    elif phase_strength < 0.45:
        iterations = int((CPU_IDLE_ITERATIONS + CPU_BUSY_ITERATIONS) / 2)
    elif phase_strength < 0.75:
        iterations = CPU_BUSY_ITERATIONS
    elif burst_gate < 0.3:
        iterations = int((CPU_BUSY_ITERATIONS + CPU_PEAK_ITERATIONS) / 2)
    else:
        iterations = CPU_PEAK_ITERATIONS

    # The nested loops intentionally create measurable CPU work while still
    # keeping the code compact and dependency-free.
    accumulator = 0.0
    for row_index in range(CPU_MATRIX_SIZE):
        row_phase = angle_radians + (row_index / CPU_MATRIX_SIZE)
        for column_index in range(iterations):
            value = (row_phase * (column_index + 1)) % math.pi
            accumulator += math.sin(value) * math.cos(value / 2.0)

    return accumulator


def _cpu_worker_loop(
    stop_event: Any,
    phase_value: Any,
    result_value: Any,
    worker_index: int,
) -> None:
    """Run sustained CPU work in a dedicated process until stopped.

    Args:
        stop_event (Any): Shared event that
            stops all worker processes when set. If the event is unset, the
            worker continues generating CPU load. If it is set, the worker
            exits its loop and returns immediately.
        phase_value (Any): Shared
            floating-point phase written by the parent process. Lower phase
            values reduce the inner-loop count while higher values push the
            worker into a much heavier CPU burst.
        result_value (Any): Shared
            floating-point slot where the worker stores its latest accumulated
            value. This keeps the work observable and discourages the process
            from collapsing into trivial no-op behavior.
        worker_index (int): Zero-based worker identifier. Different workers use
            slightly different phase offsets so their CPU spikes do not line up
            perfectly on every loop iteration.

    Returns:
        None: The worker mutates shared state and runs until ``stop_event`` is
            set by the parent process.

    Raises:
        RuntimeError: Not raised by this helper.
    """
    phase_offset = worker_index * 0.37
    while not stop_event.is_set():
        result_value.value = _cpu_wave_step(phase_value.value + phase_offset)


def _start_cpu_workers(
    worker_count: int,
) -> tuple[
    Any,
    Any,
    list[Any],
    list[multiprocessing.Process],
]:
    """Start dedicated CPU worker processes for the monitor target.

    Args:
        worker_count (int): Number of worker processes to spawn. Larger values
            drive more CPU cores at once but also increase total system load.

    Returns:
        tuple[Any,
        Any,
        list[Any],
        list[multiprocessing.Process]]: Stop event, shared phase value, shared
        result slots, and process handles for all launched workers.

    Raises:
        ValueError: If ``worker_count`` is less than ``1``.
    """
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")

    stop_event = multiprocessing.Event()
    phase_value = multiprocessing.Value("d", 0.0)
    result_slots = [multiprocessing.Value("d", 0.0) for _ in range(worker_count)]
    processes: list[multiprocessing.Process] = []

    # Separate processes avoid the GIL bottleneck and keep CPU usage high even
    # while the main process is blocked on CUDA kernels or sleeping.
    for worker_index in range(worker_count):
        process = multiprocessing.Process(
            target=_cpu_worker_loop,
            args=(stop_event, phase_value, result_slots[worker_index], worker_index),
            daemon=True,
        )
        process.start()
        processes.append(process)

    return stop_event, phase_value, result_slots, processes


def _stop_cpu_workers(
    stop_event: Any,
    processes: list[multiprocessing.Process],
) -> None:
    """Stop and join all CPU worker processes.

    Args:
        stop_event (Any): Shared event that tells
            the workers to exit. If unset, workers keep running. If set, each
            worker should terminate cleanly before the fallback kill logic runs.
        processes (list[multiprocessing.Process]): Process handles returned by
            :func:`_start_cpu_workers`. Live workers are joined first and
            forcibly terminated only if they ignore the stop signal.

    Returns:
        None: The function mutates process state in place and ensures no worker
            is left running after the monitor target exits.

    Raises:
        RuntimeError: Not raised by this helper.
    """
    stop_event.set()
    for process in processes:
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def _memory_wave_step(
    retained_chunks: list[bytearray],
    angle_radians: float,
    seed_value: float,
) -> float:
    """Resize a retained memory buffer to create visible RAM swings.

    Args:
        retained_chunks (list[bytearray]): Mutable list of allocated memory
            chunks that should remain referenced between loop iterations. When
            the target watermark grows, this list is extended. When the target
            watermark shrinks, the list is truncated so memory can be released.
        angle_radians (float): Phase angle in radians used to switch between
            low, medium, and burst memory targets. Larger positive sine values
            map to larger retained memory footprints.
        seed_value (float): Floating-point value from the CPU phase. The
            helper uses this to stamp the most recent chunk so the allocation
            work depends on live runtime state.

    Returns:
        float: Approximate retained memory size in megabytes after resizing.

    Raises:
        RuntimeError: Not raised by this helper.
    """
    phase_strength = (math.sin(angle_radians) + 1.0) / 2.0
    burst_gate = math.sin(angle_radians * 4.0)
    if phase_strength < 0.2:
        target_megabytes = MEMORY_LOW_WATERMARK_MB
    elif phase_strength < 0.45:
        target_megabytes = int((MEMORY_LOW_WATERMARK_MB + MEMORY_HIGH_WATERMARK_MB) / 2)
    elif phase_strength < 0.75:
        target_megabytes = MEMORY_HIGH_WATERMARK_MB
    elif burst_gate < 0.2:
        target_megabytes = int((MEMORY_HIGH_WATERMARK_MB + MEMORY_PEAK_WATERMARK_MB) / 2)
    else:
        target_megabytes = MEMORY_PEAK_WATERMARK_MB

    chunk_count_target = max(1, target_megabytes // MEMORY_CHUNK_MB)

    # The monitor should observe real resident memory swings, so allocations are
    # retained across loop iterations instead of being created and discarded
    # immediately inside a single helper call.
    while len(retained_chunks) < chunk_count_target:
        retained_chunks.append(bytearray(MEMORY_CHUNK_MB * 1024 * 1024))

    while len(retained_chunks) > chunk_count_target:
        retained_chunks.pop()

    if retained_chunks:
        retained_chunks[-1][0] = int(abs(seed_value)) % 251

    return len(retained_chunks) * MEMORY_CHUNK_MB


def _gpu_wave_step(torch_module, angle_radians: float, seed_value: float) -> Optional[float]:
    """Execute one CUDA-heavy step whose intensity varies with ``angle_radians``.

    Args:
        torch_module (Any): Imported ``torch`` module. It must support CUDA
            tensor allocation because this helper always runs on
            ``device="cuda"``.
        angle_radians (float): Phase angle in radians that controls the number
            of GPU matrix multiplications. Lower angles reduce the number of
            kernels launched, while higher angles increase it.
        seed_value (float): CPU-side floating-point result from the same loop
            iteration. The helper mixes this value into the CUDA tensors so the
            operations depend on the live workload instead of a compile-time
            constant.

    Returns:
        Optional[float]: Scalar value read back from the GPU result. ``None``
            is not returned during normal operation; the optional return type is
            used only to match the explicit failure style of callers that may
            choose to ignore the numeric value.

    Raises:
        RuntimeError: Propagates CUDA execution errors raised by Torch.
    """
    phase_strength = (math.cos(angle_radians) + 1.0) / 2.0

    # GPU usage needs sharper transitions than the original smooth curve, so
    # the number of CUDA passes jumps between clearly separated bands.
    burst_gate = math.sin(angle_radians * 2.5) + math.cos(angle_radians * 6.0)
    if phase_strength < 0.2:
        repetitions = GPU_IDLE_REPETITIONS
    elif phase_strength < 0.45:
        repetitions = int((GPU_IDLE_REPETITIONS + GPU_BUSY_REPETITIONS) / 2)
    elif phase_strength < 0.75:
        repetitions = GPU_BUSY_REPETITIONS
    elif burst_gate < 0.1:
        repetitions = int((GPU_BUSY_REPETITIONS + GPU_PEAK_REPETITIONS) / 2)
    else:
        repetitions = GPU_PEAK_REPETITIONS

    # CUDA tensors are allocated once per step, then used for several matrix
    # multiplies so GPU utilization visibly rises and falls over time.
    if repetitions >= GPU_PEAK_REPETITIONS:
        matrix_size = GPU_MATRIX_SIZE + 1024
    elif repetitions >= GPU_BUSY_REPETITIONS:
        matrix_size = GPU_MATRIX_SIZE
    else:
        matrix_size = GPU_MATRIX_SIZE // 2

    left = torch_module.full(
        (matrix_size, matrix_size),
        float(seed_value % 1.0) + 1.0,
        device="cuda",
        dtype=torch_module.float32,
    )
    right = torch_module.full(
        (matrix_size, matrix_size),
        float(math.sin(angle_radians)) + 1.5,
        device="cuda",
        dtype=torch_module.float32,
    )

    result = left
    for _ in range(repetitions):
        result = torch_module.matmul(result, right)
        result = torch_module.relu(result)
        result = result + torch_module.sin(result * 0.000001)
        if repetitions >= GPU_BUSY_REPETITIONS:
            result = result + torch_module.matmul(right, left).mul_(0.000001)

    torch_module.cuda.synchronize()
    return float(result.mean().item())


def main() -> None:
    """Run the fixed-duration CPU and GPU monitor target workload.

    The workload runs for ``MONITOR_TEST_DURATION_SECONDS`` and reshapes its
    CPU and GPU pressure every ``LOAD_UPDATE_INTERVAL_SECONDS``. CPU and GPU
    phases intentionally differ so the resulting monitor charts are not flat.

    Returns:
        None: The function runs the workload and prints start and completion
            messages to stdout.

    Raises:
        ImportError: If Torch is not installed when the script is executed.
        RuntimeError: If no CUDA-capable GPU is available for the GPU phase.
    """
    torch_module = _require_torch_cuda()
    started_at = time.monotonic()
    cpu_result = 0.0
    gpu_result = 0.0
    retained_memory_chunks: list[bytearray] = []
    retained_memory_megabytes = 0.0
    cpu_stop_event, cpu_phase_value, cpu_result_slots, cpu_processes = _start_cpu_workers(
        CPU_WORKER_COUNT
    )

    print(
        "Starting CPU/GPU/memory varying monitor target for "
        f"{MONITOR_TEST_DURATION_SECONDS} seconds."
    )

    # The phase angles intentionally run at different speeds so CPU, memory,
    # and GPU spikes drift in and out of alignment instead of tracing a single
    # smooth curve.
    try:
        while True:
            elapsed_seconds = time.monotonic() - started_at
            if elapsed_seconds >= MONITOR_TEST_DURATION_SECONDS:
                break

            normalized_progress = elapsed_seconds / MONITOR_TEST_DURATION_SECONDS
            cpu_angle = normalized_progress * math.tau * 4.0
            memory_angle = normalized_progress * math.tau * 2.5 + math.pi / 3.0
            gpu_angle = normalized_progress * math.tau * 5.5 + math.pi / 2.0

            # The shared phase drives the external CPU workers so the monitor
            # sees sustained multi-core activity instead of a brief CPU blip
            # before the main process blocks on GPU work.
            cpu_phase_value.value = cpu_angle
            cpu_result = sum(result_slot.value for result_slot in cpu_result_slots)
            retained_memory_megabytes = _memory_wave_step(
                retained_memory_chunks,
                memory_angle,
                cpu_result,
            )
            gpu_result = _gpu_wave_step(torch_module, gpu_angle, cpu_result) or gpu_result
            time.sleep(LOAD_UPDATE_INTERVAL_SECONDS)
    finally:
        _stop_cpu_workers(cpu_stop_event, cpu_processes)

    print(
        "Completed CPU/GPU/memory varying monitor target. "
        f"cpu_result={cpu_result:.4f}, "
        f"gpu_result={gpu_result:.4f}, "
        f"retained_memory_mb={retained_memory_megabytes:.0f}"
    )


if __name__ == "__main__":
    main()
