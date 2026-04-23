#!/usr/bin/env bash
set -euo pipefail

DEFAULT_VENV_NAME="cuda"
DEFAULT_VENV_ROOT="$HOME/.venvs"
PYTHON_BIN="python3.12"
MIN_NVIDIA_DRIVER_VERSION="525.60.13"
CALL_DIR="$(pwd -P)"

APT_PACKAGES=(
  "python3.12"
  "python3.12-venv"
  "python3.12-dev"
  "python3-pip"
  "build-essential"
  "pkg-config"
  "git"
  "graphviz"
)

EXTRA_PIP_PACKAGES=(
  "optuna"
  "optuna-integration"
  "optunahub"
  "numpy"
  "matplotlib"
  "cycler"
  "scikit-learn"
  "psutil"
  "scipy"
  "pandas"
  "ipython"
  "nbformat"
  "plotly"
  "cmaes"
  "tabulate"
  "ipynbname"
  "spektral"
  "pynvml"
  "jupyterlab"
  "ipykernel"
  "tensorboard"
  "h5py"
  "pydot"
  "graphviz"
  "tqdm"
  "araras[all]"
)

TORCH_PIP_PACKAGES=(
  "torch"
  "torchinfo"
  "torchviz"
)

VENV_NAME=""
VENV_ROOT=""
VENV_PATH=""

print_header() {
  echo
  echo "=============================================="
  echo "TensorFlow + Optuna GPU environment installer"
  echo "=============================================="
  echo
}

command_exists() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1
}

version_ge() {
  local current="$1"
  local required="$2"
  [[ "$(printf '%s\n%s\n' "$required" "$current" | sort -V | head -n1)" == "$required" ]]
}

expand_home_path() {
  local input_path="$1"

  if [[ "$input_path" == "~" ]]; then
    printf '%s\n' "$HOME"
    return
  fi

  if [[ "$input_path" == "~/"* ]]; then
    printf '%s\n' "$HOME/${input_path#~/}"
    return
  fi

  printf '%s\n' "$input_path"
}

prompt_venv_name() {
  local answer=""

  read -r -p "Venv name [${DEFAULT_VENV_NAME}]: " answer
  VENV_NAME="${answer:-$DEFAULT_VENV_NAME}"
}

prompt_venv_root() {
  local choice=""
  local custom_parent=""
  local candidate_root=""
  local confirm=""

  while true; do
    echo
    echo "Choose where to store the virtual environment:"
    echo "  1) Default: ${DEFAULT_VENV_ROOT}"
    echo "  2) Current folder: ${CALL_DIR}/.venvs"
    echo "  3) Custom parent directory, script will create .venvs inside it"
    read -r -p "Option [1]: " choice

    case "${choice:-1}" in
      1)
        candidate_root="$DEFAULT_VENV_ROOT"
        ;;
      2)
        candidate_root="${CALL_DIR}/.venvs"
        ;;
      3)
        read -r -p "Enter the custom parent directory: " custom_parent
        custom_parent="$(expand_home_path "$custom_parent")"

        if [[ -z "$custom_parent" ]]; then
          echo "Custom parent directory cannot be empty."
          continue
        fi

        candidate_root="${custom_parent}/.venvs"
        ;;
      *)
        echo "Invalid option."
        continue
        ;;
    esac

    VENV_ROOT="$candidate_root"
    VENV_PATH="${VENV_ROOT}/${VENV_NAME}"

    echo
    echo "Full virtual environment path:"
    echo "  ${VENV_PATH}"
    read -r -p "Confirm this location? [Y/n]: " confirm

    case "${confirm:-Y}" in
      Y|y|"")
        break
        ;;
      N|n)
        ;;
      *)
        echo "Invalid confirmation, restarting location prompt."
        ;;
    esac
  done
}

install_system_prerequisites() {
  echo
  echo "[1/8] Installing Ubuntu prerequisites..."
  sudo apt update
  sudo apt install -y "${APT_PACKAGES[@]}"

  if ! command_exists "$PYTHON_BIN"; then
    echo "ERROR: ${PYTHON_BIN} is not available after package installation."
    exit 1
  fi
}

check_nvidia_requirements() {
  local first_line=""
  local gpu_name=""
  local driver_version=""

  echo
  echo "[2/8] Checking NVIDIA requirements..."

  if ! command_exists nvidia-smi; then
    echo "ERROR: nvidia-smi was not found."
    echo "Install a working NVIDIA driver first."
    exit 1
  fi

  first_line="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"
  gpu_name="${first_line%%,*}"
  driver_version="${first_line##*, }"

  echo "Detected GPU: ${gpu_name}"
  echo "Driver version: ${driver_version}"

  if ! version_ge "$driver_version" "$MIN_NVIDIA_DRIVER_VERSION"; then
    echo "ERROR: NVIDIA driver is too old."
    echo "Required by the TensorFlow tutorial path: >= ${MIN_NVIDIA_DRIVER_VERSION}"
    exit 1
  fi

  echo
  nvidia-smi
}

create_and_activate_venv() {
  echo
  echo "[3/8] Creating virtual environment..."
  mkdir -p "$VENV_ROOT"
  "$PYTHON_BIN" -m venv "$VENV_PATH"

  activate_virtual_environment
}

activate_virtual_environment() {
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"

  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: Failed to activate the virtual environment."
    exit 1
  fi

  echo "Activated: ${VIRTUAL_ENV}"
}

install_tensorflow_tutorial_flow() {
  echo
  echo "[4/8] Installing TensorFlow exactly through the tutorial flow..."
  python -m pip install --upgrade pip
  python -m pip install "tensorflow[and-cuda]"
}

install_torch_packages() {
  echo
  echo "[5/8] Installing Torch packages..."

  if [[ "${#TORCH_PIP_PACKAGES[@]}" -eq 0 ]]; then
    echo "No Torch packages configured."
    return
  fi

  python -m pip install "${TORCH_PIP_PACKAGES[@]}"
}

verify_tensorflow_cpu() {
  echo
  echo "[7/8] Verifying TensorFlow CPU setup..."
  activate_virtual_environment
  python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
}

verify_tensorflow_gpu() {
  echo
  echo "[8/8] Verifying TensorFlow GPU setup..."
  activate_virtual_environment

  python - <<'PY'
"""Check whether TensorFlow sees at least one GPU."""
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(gpus)
sys.exit(0 if gpus else 1)
PY
}

patch_activate_ld_library_path() {
  # Summary:
  #   Root cause and fix for TensorFlow GPU library resolution inside the virtual environment.
  #
  # Description:
  #   TensorFlow bundles CUDA shared libraries such as libcudart.so.12,
  #   libcublas.so.12, and libcudnn.so.9 inside its own directory within the
  #   virtual environment. The system dynamic linker was not finding these
  #   libraries because that directory was not included in LD_LIBRARY_PATH.
  #
  # Root cause:
  #   TensorFlow loads CUDA libraries through ctypes.CDLL() in Python, which
  #   searches LD_LIBRARY_PATH but does not reliably use $ORIGIN or RPATH for
  #   this case. Because of that, symlinks alone are not a dependable solution.
  #
  # Solution:
  #   The virtual environment activation script at, for example,
  #   /home/USER/.venvs/cuda/bin/activate (or any path) was updated to automatically export
  #   LD_LIBRARY_PATH with the TensorFlow directory that already contains the
  #   bundled .so files. The deactivate logic was also updated to restore the
  #   original LD_LIBRARY_PATH value.
  #
  # Usage:
  #   source ~/.venvs/cuda/bin/activate
  #   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  #
  # JIT note:
  #   The first GPU execution may take several minutes because TensorFlow may need
  #   to JIT-compile CUDA kernels from PTX. Later executions should be much faster
  #   because the compiled artifacts are cached.
  #
  # What was added to the script:
  #   A new function, patch_activate_ld_library_path(), was introduced. It uses
  #   Python to modify the virtual environment activate script safely, avoiding
  #   fragile sed-based editing.
  #
  # Behavior of patch_activate_ld_library_path():
  #   - On activation, exports LD_LIBRARY_PATH pointing to the TensorFlow library directory
  #   - On deactivation, restores the original LD_LIBRARY_PATH
  #
  # Fallback:
  #   The symlink repair function, apply_tensorflow_gpu_fix(), is still kept as a
  #   backup for exceptional cases.
  echo "Patching activate script to export LD_LIBRARY_PATH for CUDA libraries..."

  local tf_lib_dir activate_script
  tf_lib_dir="$(python -c 'import os, tensorflow; print(os.path.dirname(tensorflow.__file__))')"
  activate_script="$VIRTUAL_ENV/bin/activate"

  if grep -q "_OLD_VIRTUAL_LD_LIBRARY_PATH" "$activate_script"; then
    echo "activate script already patched."
    return
  fi

  python3 - "$activate_script" "$tf_lib_dir" <<'PY'
import sys

activate_path, tf_lib_dir = sys.argv[1], sys.argv[2]

with open(activate_path) as f:
    text = f.read()

deactivate_block = """\
    if [ -n "${_OLD_VIRTUAL_LD_LIBRARY_PATH:-}" ] ; then
        LD_LIBRARY_PATH="${_OLD_VIRTUAL_LD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH
        unset _OLD_VIRTUAL_LD_LIBRARY_PATH
    else
        unset LD_LIBRARY_PATH
    fi

"""

activate_block = f"""
_OLD_VIRTUAL_LD_LIBRARY_PATH="${{LD_LIBRARY_PATH:-}}"
LD_LIBRARY_PATH="{tf_lib_dir}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH
"""

text = text.replace(
    "    unset VIRTUAL_ENV\n",
    deactivate_block + "    unset VIRTUAL_ENV\n",
)
text = text.replace("\nexport PATH\n", "\nexport PATH\n" + activate_block)

with open(activate_path, "w") as f:
    f.write(text)

print("activate script patched successfully.")
PY
}

create_tensorflow_shared_library_symlinks() {
  echo
  echo "Creating symbolic links to NVIDIA shared libraries..."

  pushd "$(dirname "$(python -c 'print(__import__("tensorflow").__file__)')")" >/dev/null
  ln -svf ../nvidia/*/lib/*.so* .
  popd >/dev/null
}

resolve_ptxas_path() {
  python - <<'PY'
"""Resolve the ptxas binary path from the installed NVIDIA NVCC Python package."""
import os

try:
    import nvidia.cuda_nvcc
except Exception:
    print("")
    raise SystemExit(0)

candidates = []

module_path = getattr(nvidia.cuda_nvcc, "__path__", None)
if module_path:
    for base in module_path:
        candidates.append(os.path.join(base, "bin", "ptxas"))

module_file = getattr(nvidia.cuda_nvcc, "__file__", None)
if module_file:
    candidates.append(os.path.join(os.path.dirname(module_file), "bin", "ptxas"))

for candidate in candidates:
    if candidate and os.path.exists(candidate):
        print(candidate)
        break
else:
    print("")
PY
}

create_ptxas_symlink() {
  local ptxas_path=""
  ptxas_path="$(resolve_ptxas_path)"

  if [[ -z "$ptxas_path" ]]; then
    echo "ERROR: Could not locate ptxas inside the installed NVIDIA NVCC package."
    return 1
  fi

  echo "Creating symbolic link to ptxas..."
  echo "Resolved ptxas: ${ptxas_path}"
  ln -sf "$ptxas_path" "$VIRTUAL_ENV/bin/ptxas"
}

apply_tensorflow_gpu_fix() {
  echo
  echo "Initial GPU verification failed."
  echo "Applying the TensorFlow virtual environment symbolic-link fix..."

  create_tensorflow_shared_library_symlinks
  create_ptxas_symlink

  echo
  echo "Re-running GPU verification after symbolic-link fix..."
  verify_tensorflow_gpu
}

install_other_pip_packages() {
  echo
  echo "[6/8] Installing extra pip packages..."

  if [[ "${#EXTRA_PIP_PACKAGES[@]}" -eq 0 ]]; then
    echo "No extra pip packages configured."
    return
  fi

  python -m pip install "${EXTRA_PIP_PACKAGES[@]}"
}

print_summary() {
  echo
  echo "Done."
  echo
  echo "Virtual environment:"
  echo "  ${VENV_PATH}"
  echo
  echo "Activate it later with:"
  echo "  source \"${VENV_PATH}/bin/activate\""
  echo
  echo "Configured extra pip packages:"
  for pkg in "${EXTRA_PIP_PACKAGES[@]}"; do
    echo "  - ${pkg}"
  done
}

main() {
  print_header
  prompt_venv_name
  prompt_venv_root
  install_system_prerequisites
  check_nvidia_requirements
  create_and_activate_venv
  install_tensorflow_tutorial_flow
  patch_activate_ld_library_path
  install_torch_packages
  install_other_pip_packages
  verify_tensorflow_cpu

  if ! verify_tensorflow_gpu; then
    apply_tensorflow_gpu_fix
  fi

  print_summary
}

main "$@"