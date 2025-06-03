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
import subprocess


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


def check_memory_model(model: tf.keras.Model, gpu_index: int = 0, param_dtype: str = "float32") -> bool:
    """
    Checks whether the memory footprint of a model's parameters exceeds the available GPU memory.

    This function uses `nvidia-smi` to retrieve the free memory available on the specified GPU.
    It then calculates the total memory required by the model's parameters, based on the number
    of parameters and their data type. If the model's memory exceeds available GPU memory,
    pruning is recommended.

    Flow:
    model.count_params() -> calculate parameter memory -> query GPU memory via subprocess ->
    compare values -> print summary -> return True if pruning is needed

    Args:
        model (tf.keras.Model): The model to analyze.
        gpu_index (int): Index of the GPU to check (default is 0).
        param_dtype (str): Data type of model parameters. Supported: "float32", "float16",
                           "bfloat16", "float64", "int8".

    Returns:
        bool: True if model's parameter memory exceeds free GPU memory (pruning is advised).

    Raises:
        ValueError: If `param_dtype` or `gpu_index` is invalid.
    """
    # Map param_dtype to corresponding bytes per parameter
    if param_dtype == "float32":
        bytes_per_param = 4
    elif param_dtype == "float16" or param_dtype == "bfloat16":
        bytes_per_param = 2
    elif param_dtype == "float64":
        bytes_per_param = 8
    elif param_dtype == "int8":
        bytes_per_param = 1
    else:
        # Raise error if dtype is unsupported
        raise ValueError(f"Unsupported param_dtype: {param_dtype}")

    # Count total number of model parameters
    num_params = model.count_params()

    # Calculate memory usage in bytes and megabytes
    param_memory_bytes = num_params * bytes_per_param
    param_memory_mb = param_memory_bytes / (1024**2)

    # Use nvidia-smi to fetch GPU free memory in MiB
    try:
        output = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
            )
            .decode("utf-8")  # Convert byte output to string
            .strip()  # Remove trailing whitespace
            .splitlines()  # Split output into list by line (1 line per GPU)
        )
    except subprocess.CalledProcessError as e:
        # Handle error if subprocess fails
        print(f"Error querying nvidia-smi: {e}")
        return False

    # Ensure the requested GPU index is within the available range
    if gpu_index < 0 or gpu_index >= len(output):
        raise ValueError(f"Invalid GPU index: {gpu_index}")

    # Convert free memory from MiB to bytes and MB
    free_mib = int(output[gpu_index])  # Extract free memory value as int
    free_bytes = free_mib * 1024**2  # Convert MiB to bytes
    free_mb = free_bytes / (1024**2)  # Convert bytes to MB (for printing)

    # Determine whether model parameters exceed free GPU memory
    should_prune = param_memory_bytes > free_bytes

    # Output a summary of memory usage and decision
    print("=== GPU Memory Check Summary ===")
    print(f"Bytes per parameter: {bytes_per_param}")
    print(f"Number of parameters: {num_params}")
    print(f"Param memory: {param_memory_mb:.2f} MB")
    print(f"Free GPU memory: {free_mb:.2f} MB")
    print("================================")

    # Return recommendation on pruning
    return should_prune


def estimate_training_memory(model: tf.keras.Model, batch_size: int, param_dtype: str) -> int:
    """
    Estimates the total GPU memory required to train a model with a given batch size and parameter dtype.

    This includes:
    - Model parameters
    - Gradients (assumed to be same size as parameters)
    - Activations needed for backpropagation (sum of outputs from each layer)

    **Note**: This does not account for optimizer state (e.g., Adam moments).

    Flow:
    param_dtype -> bytes per param -> count model params ->
    iterate layers -> compute activation memory -> sum with params and grads -> return total

    Args:
        model (tf.keras.Model): The model to estimate memory for.
        batch_size (int): Batch size used during training.
        param_dtype (str): Data type of model weights and activations.

    Returns:
        int: Estimated memory in bytes required for training.

    Raises:
        ValueError: If `param_dtype` is unsupported.
    """
    # Determine bytes per parameter based on the dtype
    if param_dtype == "float32":
        bytes_per_param = 4
    elif param_dtype == "float16" or param_dtype == "bfloat16":
        bytes_per_param = 2
    elif param_dtype == "float64":
        bytes_per_param = 8
    elif param_dtype == "int8":
        bytes_per_param = 1
    else:
        raise ValueError(f"Unsupported param_dtype: {param_dtype}")

    # Total memory for model weights (parameters)
    num_params = model.count_params()
    memory_params = num_params * bytes_per_param

    # Initialize total memory for activations
    memory_activations = 0
    for layer in model.layers:
        # Skip layers that do not have an output shape
        if not hasattr(layer, "output_shape"):
            continue

        out_shape = layer.output_shape

        # If layer has multiple outputs, take the first one
        if isinstance(out_shape, list):
            out_shape = out_shape[0]

        # Skip layers with undefined output shape
        if out_shape is None:
            continue

        # Compute number of elements per example (exclude batch dimension)
        dims = [d for d in out_shape if d is not None]
        num_elements_per_example = 1
        for d in dims[1:]:  # Skip batch dimension
            num_elements_per_example *= d

        # Compute total memory for activations of this layer
        memory_layer = batch_size * num_elements_per_example * bytes_per_param
        memory_activations += memory_layer

    # Gradients have same memory footprint as parameters
    memory_grads = memory_params

    # Return total estimated memory required
    return memory_params + memory_activations + memory_grads
