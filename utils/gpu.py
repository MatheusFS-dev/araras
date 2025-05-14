"""
This module provides utility functions for inspecting and reporting GPU-related
information in a TensorFlow environment. It is designed to help developers
understand the GPU configuration and capabilities of their system when using
TensorFlow for machine learning or deep learning tasks.

Functions:
    - get_gpu_info: Prints detailed TensorFlow and GPU configuration information.

Example Usage:
    get_gpu_info()
"""

import tensorflow as tf


def get_gpu_info() -> None:
    """
    Prints detailed TensorFlow and GPU configuration information.

    This function reports:
      - TensorFlow version
      - Whether TensorFlow was built with CUDA support
      - Detected CUDA and cuDNN versions (if applicable)
      - Availability and names of physical GPU devices

    Logic:
        -> Print TensorFlow version
        -> Check if CUDA is supported by the TensorFlow build
           -> If so, print CUDA and cuDNN version info
           -> If not, note CPU-only support
        -> List physical GPU devices
           -> If found, print count, names, and default GPU
           -> If none, indicate CPU-only usage

    Raises:
        ImportError: If TensorFlow is not installed (implicit by tf import)

    Returns:
        None

    Example:
        get_gpu_info()
    """
    # Print the installed version of TensorFlow
    print(f"TensorFlow Version: {tf.__version__}")

    # Check if TensorFlow was compiled with CUDA (GPU) support
    if tf.test.is_built_with_cuda():
        # Retrieve low-level build information about the TensorFlow installation
        build_info = tf.sysconfig.get_build_info()

        print("CUDA support detected")

        # Extract and print CUDA version; fallback to "Unknown" if not available
        print(f"  CUDA Version: {build_info.get('cuda_version', 'Unknown')}")

        # Extract and print cuDNN version; fallback to "Unknown" if not available
        print(f"  cuDNN Version: {build_info.get('cudnn_version', 'Unknown')}")
    else:
        # Indicate that only CPU execution is available
        print("No CUDA support (CPU only)")

    # Retrieve a list of physical GPU devices accessible by TensorFlow
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        # If GPUs are found, print their count and names
        print(f"\nGPUs Detected ({len(gpus)}): {[gpu.name for gpu in gpus]}")

        # Print the default GPU TensorFlow would use (if configured)
        print(f"Default GPU device: {tf.test.gpu_device_name()}")
    else:
        # Indicate that no GPUs are accessible and fallback is to CPU
        print("No GPUs found (CPU execution)")
