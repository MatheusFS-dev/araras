"""
This module provides utility functions for inspecting and reporting GPU-related
information in a TensorFlow environment. It is designed to help developers
understand the GPU configuration and capabilities of their system when using
TensorFlow for machine learning or deep learning tasks.

Functions:
    - get_gpu_info: Prints detailed TensorFlow and GPU configuration information.
"""

import tensorflow as tf
import subprocess
from datetime import datetime


# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


def get_user_gpu_choice():
    """
    Prompts the user to select a GPU index and validates the input.

    Returns:
        str: Valid GPU index as string
    """
    available_gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(available_gpus)

    if num_gpus == 0:
        print(f"{RED}No GPUs available. Using CPU.{RESET}")
        return ""
    elif num_gpus == 1:
        # Get GPU data for single GPU
        gpu_data = _get_nvidia_smi_data()
        if gpu_data and len(gpu_data) > 0:
            gpu = gpu_data[0]
            free_gb = gpu["free_mb"] / 1024
            total_gb = gpu["total_mb"] / 1024
            print(
                f"Only one GPU available: {gpu['name']} ({GREEN}{free_gb:.1f}GB/{RED}{total_gb:.1f}GB free){RESET}"
            )
        else:
            print(f"Only one GPU available: {GREEN}{available_gpus[0].name}{RESET}")
        return "0"

    # Get GPU data for multiple GPUs
    gpu_data = _get_nvidia_smi_data()
    gpu_info_map = {gpu["index"]: gpu for gpu in gpu_data} if gpu_data else {}

    print(f"{BOLD}{BLUE}Available GPUs: {GREEN}{num_gpus}{RESET}")
    for i, gpu in enumerate(available_gpus):
        if i in gpu_info_map:
            gpu_info = gpu_info_map[i]
            free_gb = gpu_info["free_mb"] / 1024
            total_gb = gpu_info["total_mb"] / 1024
            print(f"  {CYAN}GPU {i}{RESET}: {gpu_info['name']} ({free_gb:.1f}GB/{total_gb:.1f}GB free)")
        else:
            print(f"  {CYAN}GPU {i}{RESET}: {gpu.name}")

    while True:
        try:
            user_input = input(f"\nEnter GPU index to use {YELLOW}(0-{num_gpus-1}): {RESET}").strip()
            gpu_index = int(user_input)

            if 0 <= gpu_index < num_gpus:
                print(f"{GREEN}Selected GPU {gpu_index}{RESET}")
                return str(gpu_index)
            else:
                print(f"{RED}Invalid index. Please enter a number between 0 and {num_gpus-1}.{RESET}")
        except ValueError:
            print(f"{RED}Invalid input. Please enter a valid number.{RESET}")
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Operation cancelled. Using GPU 0 as default.{RESET}")


def _get_nvidia_smi_data():
    """
    Retrieves GPU information using nvidia-smi command.

    Returns:
        list: List of GPU information dictionaries or empty list if failed
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            gpu_data = []
            lines = result.stdout.strip().split("\n")

            for line in lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 7:
                    index, name, total_mb, used_mb, free_mb, temp, util = parts[:7]

                    gpu_info = {
                        "index": int(index),
                        "name": name,
                        "total_mb": float(total_mb),
                        "used_mb": float(used_mb),
                        "free_mb": float(free_mb),
                        "temperature": temp if temp != "[Not Supported]" else "N/A",
                        "utilization": util if util != "[Not Supported]" else "N/A",
                    }
                    gpu_data.append(gpu_info)

            return gpu_data
    except Exception:
        return []


def _print_tensorflow_info():
    """Print TensorFlow configuration information."""
    print(f"{BOLD}TensorFlow Configuration{RESET}")
    print("=" * 80)
    print(f"Version        : {tf.__version__}")

    if tf.test.is_built_with_cuda():
        print(f"CUDA Support   : {GREEN}Yes{RESET}")
        build_info = tf.sysconfig.get_build_info()
        print(f"CUDA Version   : {build_info.get('cuda_version', 'Unknown')}")
        print(f"cuDNN Version  : {build_info.get('cudnn_version', 'Unknown')}")
    else:
        print(f"CUDA Support   : {RED}No (CPU only){RESET}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
       tf.test.gpu_device_name()


def _print_gpu_table(gpu_data):
    """Print GPU information in nvidia-smi style table format."""
    if not gpu_data:
        print(f"{RED}No GPU data available{RESET}")
        return

    print(f"\n{BOLD}GPU Information{RESET}")
    print("=" * 80)

    # Header
    header = f"{'GPU':<3} {'Name':<25} {'Memory Usage':<20} {'Temp':<6} {'Util':<6}"
    print(f"{BOLD}{header}{RESET}")
    print("-" * 80)

    # GPU rows
    for gpu in gpu_data:
        # Memory calculations
        total_gb = gpu["total_mb"] / 1024
        used_gb = gpu["used_mb"] / 1024
        utilization_pct = (used_gb / total_gb) * 100 if total_gb > 0 else 0

        # Format memory usage with color coding
        memory_str = f"{used_gb:6.1f}GB / {total_gb:6.1f}GB"
        if utilization_pct > 80:
            memory_color = RED
        elif utilization_pct > 50:
            memory_color = YELLOW
        else:
            memory_color = GREEN

        # Format temperature
        temp_str = f"{gpu['temperature']}C" if gpu["temperature"] != "N/A" else "N/A"

        # Format utilization
        util_str = f"{gpu['utilization']}%" if gpu["utilization"] != "N/A" else "N/A"

        # Print row
        row = (
            f"{gpu['index']:<3} "
            f"{gpu['name'][:24]:<25} "
            f"{memory_color}{memory_str:<20}{RESET} "
            f"{temp_str:<6} "
            f"{util_str:<6}"
        )
        print(row)


def _print_memory_summary(gpu_data):
    """Print memory summary similar to nvidia-smi bottom section."""
    if not gpu_data:
        return

    print(f"\n{BOLD}Memory Summary{RESET}")
    print("=" * 80)

    total_memory = sum(gpu["total_mb"] for gpu in gpu_data) / 1024
    used_memory = sum(gpu["used_mb"] for gpu in gpu_data) / 1024
    free_memory = total_memory - used_memory

    print(f"Total GPU Memory : {total_memory:8.1f} GB")
    print(f"Used Memory      : {used_memory:8.1f} GB ({used_memory/total_memory*100:5.1f}%)")
    print(f"Free Memory      : {free_memory:8.1f} GB ({free_memory/total_memory*100:5.1f}%)")


def get_gpu_info() -> None:
    """
    Prints detailed TensorFlow and GPU configuration information in nvidia-smi style format.

    This function reports:
      - TensorFlow version and CUDA configuration
      - GPU devices in tabular format similar to nvidia-smi
      - Memory usage summary
      - Temperature and utilization data (when available)

    Args:
        None

    Returns:
        None

    Example:
        get_gpu_info()
    """
    # Header with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{BOLD}{BLUE}TensorFlow GPU Monitor - {timestamp}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}")

    # TensorFlow configuration
    _print_tensorflow_info()

    # Get GPU data from nvidia-smi
    gpu_data = _get_nvidia_smi_data()

    # Print GPU table
    _print_gpu_table(gpu_data)

    # Print memory summary
    # _print_memory_summary(gpu_data)

    print(f"\n{BLUE}{'=' * 80}{RESET}")


def gpu_summary() -> None:
    """
    Prints a compact GPU summary similar to nvidia-smi output.
    """
    timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    print(f"{timestamp}")
    print("+" + "-" * 88 + "+")
    print(
        f"| NVIDIA-SMI 470.xx                Driver Version: 470.xx       CUDA Version: {tf.sysconfig.get_build_info().get('cuda_version', 'N/A'):<4} |"
    )
    print("|" + "-" * 30 + "+" + "-" * 22 + "+" + "-" * 35 + "|")
    print("| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |")
    print("| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |")
    print("|                               |                      |               MIG M. |")
    print("|" + "=" * 88 + "|")

    gpu_data = _get_nvidia_smi_data()

    for gpu in gpu_data:
        name_short = gpu["name"][:16]
        used_gb = gpu["used_mb"] / 1024
        total_gb = gpu["total_mb"] / 1024
        temp = gpu["temperature"] if gpu["temperature"] != "N/A" else "--"
        util = gpu["utilization"] if gpu["utilization"] != "N/A" else "--"

        print(f"|   {gpu['index']}  {name_short:<16} Off  | 00000000:01:00.0 Off |                  N/A |")
        print(
            f"| N/A   {temp}C    P0    N/A /  N/A |  {used_gb:5.0f}MiB / {total_gb:5.0f}MiB |     {util}%      Default |"
        )
        print("|                               |                      |                  N/A |")
        print("+" + "-" * 88 + "+")

    print()
    print("+" + "-" * 88 + "+")
    print("| Processes:                                                                   |")
    print("|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |")
    print("|        ID   ID                                                   Usage      |")
    print("|" + "=" * 88 + "|")
    print("|  No running processes found                                                 |")
    print("+" + "-" * 88 + "+")
