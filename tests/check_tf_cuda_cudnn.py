import tensorflow as tf
import os

# Specify GPU to use (e.g., GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
