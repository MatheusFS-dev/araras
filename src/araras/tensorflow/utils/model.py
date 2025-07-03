"""
Module for estimating average power and energy consumption
of a TensorFlow model (SavedModel or Keras model) on CPU or GPU using NVML (GPU)
or RAPL (CPU).
"""

import time
import tensorflow as tf
from typing import Tuple, Optional, Callable


def get_model_usage_stats(
    saved_model: str | tf.keras.Model,
    n_trials: int = 10000,
    device: str = "cpu",
    rapl_path: str = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
    verbose: bool = False
    
) -> Tuple[float, float, float]:
    """
    Estimate average power draw and energy usage.
    Careful with the RAPL path; it may vary by system.
    The RAPL interface is typically found at:
        $ ls /sys/class/powercap
        intel-rapl
        $ ls /sys/class/powercap/intel-rapl
        intel-rapl:0       intel-rapl:0:0    intel-rapl:1    …
        $ ls /sys/class/powercap/intel-rapl/intel-rapl:0
        energy_uj  max_energy_range_uj  name
    Also, you MUST run this on a linux system with Intel CPUs!!!!!
    And run the python script with SUDO to access RAPL files.

    Args:
        saved_model (str | tf.keras.Model): Path to the TensorFlow SavedModel directory,
            a .keras model file, or a Keras Model instance.
        n_trials (int): Number of inference trials to perform. Defaults to 100000.
        device (str): Device for power measurement; must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        rapl_path (str): Path to the RAPL energy counter file for CPU measurements.
        verbose (bool): If True, displays a progress bar during the trials.

    Raises:
        RuntimeError: If GPU NVML initialization fails when device='gpu'.
        ValueError: If `device` is not 'cpu' or 'gpu'.

    Returns:
        Tuple[float, float, float]:
            - per_run_time (float): 
                Average run time in seconds. Measures a mix of tracing, initialization, 
                asynchronous queuing, Python overhead, and power-reading delays,
                so its “average” can be dominated by non-inference costs.
            - avg_power (float): Average power draw in watts.
            - avg_energy (float): Average energy consumed per inference in joules.
    """

    # Decide how we will run inference
    is_keras_model = False
    keras_model: Optional[tf.keras.Model] = None
    if isinstance(saved_model, tf.keras.Model) or (
        isinstance(saved_model, str) and saved_model.endswith(".keras")
    ):
        # We were given a Keras model or a path to a `.keras` file
        keras_model = (
            saved_model
            if isinstance(saved_model, tf.keras.Model)
            else tf.keras.models.load_model(saved_model)
        )
        is_keras_model = True

    # Initialize NVML for GPU power measurement if requested
    if device == "gpu":
        try:
            import pynvml

            pynvml.nvmlInit()  # initialize NVML library
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # get first GPU handle
        except Exception as e:
            # propagate error if NVML cannot be set up
            raise RuntimeError("Unable to initialize NVML for GPU power monitoring: " + str(e))

    def read_cpu_power_rapl() -> Optional[float]:
        """Read CPU package energy via Intel RAPL interface.

        Opens the RAPL energy counter file and returns energy in joules.
        Returns None if the interface is unavailable or unreadable.
        """
        try:
            with open(rapl_path, "r") as f:
                # energy in microjoules; convert to joules
                return int(f.read()) / 1e6
        except Exception:
            # RAPL interface not present or unreadable
            raise RuntimeError(
                f"\033[91mUnable to read CPU power from {rapl_path}. "
                "Ensure the path is correct and accessible, or run with sudo.\033[0m"
            )

    dummy_inputs = None
    infer: Callable[[], tf.Tensor]

    if is_keras_model:
        # Build random tensors based on the keras model input shapes
        tensors = []
        for tensor in keras_model.inputs:
            shape = [d if d is not None else 1 for d in tensor.shape]
            tensors.append(tf.random.normal(shape, dtype=tensor.dtype))
        dummy_inputs = tensors[0] if len(tensors) == 1 else tensors

        def infer():
            return keras_model(dummy_inputs, training=False)

    else:
        # Load the SavedModel and obtain the serving_default signature for inference
        reloaded_model: tf.Module = tf.saved_model.load(saved_model)
        signature = reloaded_model.signatures["serving_default"]

        _, kwargs = signature.structured_input_signature
        inputs = {}
        for name, spec in kwargs.items():
            shape = [d if d is not None else 1 for d in spec.shape.as_list()]
            inputs[name] = tf.random.normal(shape, dtype=spec.dtype)
        dummy_inputs = inputs

        def infer():
            return signature(**dummy_inputs)

    # Print the shapes of the input tensors
    # print(f"Input tensor shapes: {[t.shape for t in dummy_inputs.values()]}")

    powers: list[float] = []  # store measured power values
    times: list[float] = []  # store inference durations

    print(f"Estimating energy for {n_trials} trials on {device.upper()}...")
    for i in range(n_trials):
        if verbose:
            progress = (i + 1) / n_trials
            bar_len = 30
            filled = int(progress * bar_len)
            bar = "=" * filled + ">" + "." * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
            print(f"\r[{bar}] {i + 1}/{n_trials}", end="", flush=True)
        
        start_time = time.time()  # mark start of trial

        # Begin energy measurement depending on device
        if device == "gpu":
            # GPU: measure instantaneous power (milliwatts → watts)
            start_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        elif device == "cpu":
            # CPU: read cumulative energy counter at start
            start_energy = read_cpu_power_rapl()
        else:
            # invalid device selection
            raise ValueError("Unsupported device: choose 'gpu' or 'cpu'")

        # Run inference
        _ = infer()

        # Compute how long the inference took
        elapsed = time.time() - start_time

        # Complete energy measurement and compute average power
        if device == "gpu":
            # GPU: measure end instantaneous power and average the two readings
            end_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            avg_instant_power = (start_power + end_power) / 2
            powers.append(avg_instant_power)
        elif device == "cpu" and start_energy is not None:
            # CPU: read end energy counter and compute power
            end_energy = read_cpu_power_rapl()
            if end_energy is not None:
                energy_used = end_energy - start_energy  # joules consumed
                avg_power = energy_used / elapsed if elapsed > 0 else 0
                powers.append(avg_power)

        # Record inference duration
        times.append(elapsed)

    if verbose:
        print()

    # Shutdown NVML if used
    if device == "gpu":
        pynvml.nvmlShutdown()

    per_run_time = sum(times) / len(times)

    # Compute overall averages
    avg_power = sum(powers) / len(powers) if powers else 0
    # Total energy = sum(power_i * time_i) / trials
    avg_energy = sum(p * t for p, t in zip(powers, times)) / len(powers) if powers else 0

    return per_run_time, avg_power, avg_energy
