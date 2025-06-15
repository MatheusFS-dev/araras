"""
This module provides utilities for logging system.

Functions:
    - log_resources: Logs selected system and ML resources at regular time intervals.

Example:
    log_resources("logs", interval=10, cpu=True, ram=True, gpu=True)
"""

import os
import time
import psutil
import subprocess
import tensorflow as tf
from threading import Thread


def log_resources(log_dir: str, interval: int = 5, **kwargs) -> None:
    """
    Logs selected system and ML resources (CPU, RAM, GPU, CUDA, TensorFlow) at regular time intervals.

    Logic:
        -> Create log directory
        -> Define logging functions for each resource type
        -> Start logging functions in parallel threads based on keyword arguments

    Args:
        log_dir (str): Directory where log files will be stored.
        interval (int): Time interval between consecutive logs in seconds. Defaults to 5.
        kwargs: Boolean flags to specify which resources to log.
                Supported flags: "cpu", "ram", "gpu", "cuda", "tensorflow".

    Returns:
        None

    Example:
        log_resources("logs", interval=10, cpu=True, ram=True, gpu=True)
    """
    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    def log_cpu():
        """Logs total and per-core CPU usage to a CSV file."""
        log_path = os.path.join(log_dir, "cpu_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,CPU_Usage(%),Per-Core_Usage(%)\n")
            while True:
                try:
                    # Get timestamp and CPU usage
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    cpu_usage = psutil.cpu_percent()
                    per_core_usage = psutil.cpu_percent(percpu=True)

                    # Write CPU usage stats to file
                    f.write(f"{timestamp},{cpu_usage},{','.join(map(str, per_core_usage))}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    # Write error and stop logging on failure
                    f.write(f"Error: {e}\n")
                    break

    def log_ram():
        """Logs RAM usage (total, used, free) to a CSV file."""
        log_path = os.path.join(log_dir, "ram_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,Total(MB),Used(MB),Free(MB)\n")
            while True:
                try:
                    # Get memory statistics and convert from bytes to megabytes
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    mem = psutil.virtual_memory()
                    total = mem.total / (1024**2)
                    used = mem.used / (1024**2)
                    free = mem.available / (1024**2)

                    # Log the memory stats
                    f.write(f"{timestamp},{total:.2f},{used:.2f},{free:.2f}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_gpu():
        """Logs NVIDIA GPU usage statistics using nvidia-smi."""
        log_path = os.path.join(log_dir, "gpu_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,GPU_ID,Memory_Used(MB),Memory_Total(MB),GPU_Utilization(%)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    # Execute nvidia-smi to get GPU stats
                    gpu_stats = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    ).stdout.strip()

                    # Parse and write each GPU's data
                    for line in gpu_stats.split("\n"):
                        gpu_id, mem_used, mem_total, util = map(int, line.split(","))
                        f.write(f"{timestamp},{gpu_id},{mem_used},{mem_total},{util}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_cuda():
        """Logs CUDA memory usage by compute applications."""
        log_path = os.path.join(log_dir, "cuda_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,Process_Memory_Used(MB)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    # Query used CUDA memory by compute apps
                    cuda_mem_stats = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                    ).stdout.strip()

                    # Log memory usage or 0 MB if none used
                    if cuda_mem_stats:
                        f.write(f"{timestamp},{cuda_mem_stats} MB\n")
                    else:
                        f.write(f"{timestamp},0 MB\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_tensorflow():
        """Logs TensorFlow GPU memory usage."""
        log_path = os.path.join(log_dir, "tensorflow_usage_log.csv")
        os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("Timestamp,Device,Memory_Allocated(MB),Memory_Peak(MB)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    gpus = tf.config.experimental.list_physical_devices("GPU")
                    for gpu in gpus:
                        device_name = gpu.name  # Extract device name
                        memory_info = tf.config.experimental.get_memory_info("GPU:0")

                        # Convert bytes to megabytes
                        allocated_memory = memory_info["current"] / (1024**2)
                        peak_memory = memory_info["peak"] / (1024**2)

                        # Write usage info to log
                        f.write(f"{timestamp},{device_name},{allocated_memory:.2f},{peak_memory:.2f}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    # Launch threads for selected logging targets
    if kwargs.get("cpu", False):
        Thread(target=log_cpu, daemon=True).start()
    if kwargs.get("ram", False):
        Thread(target=log_ram, daemon=True).start()
    if kwargs.get("gpu", True):
        Thread(target=log_gpu, daemon=True).start()
    if kwargs.get("cuda", True):
        Thread(target=log_cuda, daemon=True).start()
    if kwargs.get("tensorflow", True):
        Thread(target=log_tensorflow, daemon=True).start()
