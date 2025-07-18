from araras.core import *


import tempfile
import zipfile
from pathlib import Path
from araras.ml.model.stats import get_flops

import tensorflow as tf


def convert_to_saved_model(input_keras_path: str, output_zip_path: str) -> None:
    """Convert a Keras `.keras` model archive into a zipped TensorFlow SavedModel.

    This will load the model, export it in SavedModel directory format,
    then compress that directory into a .zip file.

    Args:
        input_keras_path (str): Path to the source `.keras` model file.
        output_zip_path (str): Desired path for the output zip (e.g. 'saved_model.zip').

    Returns:
        None

    Raises:
        Any exception raised by TensorFlow I/O (e.g. file not found, load/save errors).
    """
    # Create a temp workspace for the SavedModel directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = Path(tmp_dir) / "saved_model"

        # Load the single-file Keras archive (.keras)
        model = tf.keras.models.load_model(input_keras_path)

        # Export the model in SavedModel format (creates a folder with saved_model.pb, variables/, assets/)
        tf.saved_model.save(model, str(saved_model_dir))

        # Compress the entire SavedModel directory tree into a zip file
        with zipfile.ZipFile(output_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in saved_model_dir.rglob("*"):
                # Compute the archive name by stripping off the temp-dir prefix
                arcname = file_path.relative_to(saved_model_dir.parent)
                zf.write(file_path, arcname)


def punish_model_flops(
    target: Union[float, Sequence[float]],
    model: tf.keras.Model,
    penalty_factor: float = 1e-10,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> Union[float, Sequence[float]]:
    """Penalize an objective according to the model's FLOPs.

    Args:
        target: Base objective value (scalar or list of scalars).
        model: Model whose FLOPs will be used for the penalty.
        penalty_factor: Multiplicative factor applied to the FLOPs count.
        direction: Whether the objective should be minimised or maximised.

    Returns:
        The penalised objective value or list of values.
    """

    if direction not in ("minimize", "maximize"):
        raise ValueError("`direction` must be either 'minimize' or 'maximize'.")

    total_flops = get_flops(model)

    # Compute penalty
    penalty = penalty_factor * total_flops

    # Apply penalty to single value or list
    if isinstance(target, (list, tuple)):
        return [t + penalty if direction == "minimize" else t - penalty for t in target]
    return target + penalty if direction == "minimize" else target - penalty


def punish_model_params(
    target: Union[float, Sequence[float]],
    model: tf.keras.Model,
    penalty_factor: float = 1e-9,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> Union[float, Sequence[float]]:
    """Penalize an objective according to the model's parameter count.

    Args:
        target: Base objective value (scalar or list of scalars).
        model: Model whose parameters will be used for the penalty.
        penalty_factor: Multiplicative factor applied to the parameter count.
        direction: Whether the objective should be minimised or maximised.

    Returns:
        The penalised objective value or list of values.
    """

    if direction not in ("minimize", "maximize"):
        raise ValueError("`direction` must be either 'minimize' or 'maximize'.")

    # Count total trainable + non-trainable parameters
    total_params = model.count_params()

    # Compute penalty
    penalty = penalty_factor * total_params

    # Apply penalty to single value or list
    if isinstance(target, (list, tuple)):
        return [t + penalty if direction == "minimize" else t - penalty for t in target]
    return target + penalty if direction == "minimize" else target - penalty


# Function to facilitate the use of both penalties together
def punish_model(
    target: Union[float, Sequence[float]],
    model: tf.keras.Model,
    type: Literal["flops", "params", None] = None,
    flops_penalty_factor: float = 1e-10,
    params_penalty_factor: float = 1e-9,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> Union[float, Sequence[float]]:
    """Apply both FLOPs and parameter penalties to an objective.

    Args:
        target: Base objective value (scalar or list of scalars).
        model: Model whose complexity will be penalised.
        type: Type of penalty to apply, either "flops" or "params".
        flops_penalty_factor: Factor for FLOPs penalty.
        params_penalty_factor: Factor for parameters penalty.
        direction: Whether the objective should be minimised or maximised.

    Returns:
        The penalised objective value or list of values.
    """
    if type is None:
        # If no type is specified, return the target unchanged
        return target

    if type == "flops":
        target = punish_model_flops(target, model, flops_penalty_factor, direction)
    elif type == "params":
        target = punish_model_params(target, model, params_penalty_factor, direction)
    else:
        raise ValueError("`type` must be either 'flops', 'params' or None.")

    return target
