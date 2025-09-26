from araras.core import *

import tempfile
import zipfile
from pathlib import Path
from araras.ml.model.stats import get_flops

import tensorflow as tf


def convert_to_saved_model(input_keras_path: str, output_zip_path: str) -> None:
    """Convert a `.keras` archive into a zipped TensorFlow SavedModel.

    Args:
        input_keras_path (str): Path to the source ``.keras`` model file.
        output_zip_path (str): Destination path for the resulting ``.zip`` bundle.

    Returns:
        None: The SavedModel artefacts are written into ``output_zip_path``.

    Raises:
        Exception: Propagates loader or saver failures raised by TensorFlow.
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


def save_model_plot(
    model_or_path: Union[tf.keras.Model, str, Path],
    output_path: Union[str, Path],
    safe_mode: bool = True,
) -> None:
    """
    Save a visual representation of a Keras model architecture to an image file.

    This function can take either a Keras model instance or a path to a `.keras`
    model archive. It generates a plot of the model architecture and saves it
    to the specified output path.

    Notes:
        The resulting plot format is inferred from the file extension provided
        in ``output_path``. A common choice is ``.png``. Existing
        files at ``output_path`` will be overwritten.

    Args:
        model_or_path (Union[tf.keras.Model, str, Path]): Either a :class:`tf.keras.Model` instance to be plotted or
            a filesystem path pointing to a ``.keras`` model archive.
        output_path (Union[str, Path]): Destination path where the plot image will be saved.
        safe_mode (bool): a parameter from :func:`tf.keras.utils.plot_model` that
            controls whether to use safe mode for plotting. If set to `True`, the
            function will not plot layers that are not supported by the plotting
            utility, which can help avoid errors with custom layers.

    Returns:
        None: The plot is written to ``output_path``.

    Raises:
        TypeError: If ``model_or_path`` is neither a model instance nor a path.
        ValueError: If a path is provided that does not end with ``.keras``.
        OSError: If saving the generated plot fails.
    """

    if isinstance(model_or_path, tf.keras.Model):
        model = model_or_path
    elif isinstance(model_or_path, (str, Path)):
        model_path = Path(model_or_path)
        if model_path.suffix != ".keras":
            raise ValueError("Model path must point to a '.keras' file.")
        try:
            model = tf.keras.models.load_model(model_path, safe_mode=safe_mode)
        except Exception as exc:  # pragma: no cover - external load
            raise OSError("Failed to load Keras model from file") from exc
    else:
        raise TypeError("model_or_path must be a tf.keras.Model or path to a '.keras' file")

    try:
        tf.keras.utils.plot_model(
            model,
            to_file=str(output_path),
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
        )
    except Exception as exc:
        logger_error.error(f"{RED} Failed to save model plot: {exc}{RESET}")
        logger.warning(f"{YELLOW} Ensure 'graphviz' is installed and updated.{RESET}")
        logger.warning(f"{YELLOW} If using conda, try: {ORANGE}conda install graphviz python-graphviz{RESET}")

        traceback.print_exc()


def punish_model_flops(
    target: Union[float, Sequence[float]],
    model: tf.keras.Model,
    penalty_factor: float = 1e-10,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> Union[float, Sequence[float]]:
    """Penalize an objective according to the model's FLOPs.

    Args:
        target (Union[float, Sequence[float]]): Base objective value (scalar or list of scalars).
        model (tf.keras.Model): Model whose FLOPs will be used for the penalty.
        penalty_factor (float): Multiplicative factor applied to the FLOPs count.
        direction (Literal['minimize', 'maximize']): Whether the objective should be minimised or maximised.

    Returns:
        Union[float, Sequence[float]]: The penalised objective value or list of values.
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
        target (Union[float, Sequence[float]]): Base objective value (scalar or list of scalars).
        model (tf.keras.Model): Model whose parameters will be used for the penalty.
        penalty_factor (float): Multiplicative factor applied to the parameter count.
        direction (Literal['minimize', 'maximize']): Whether the objective should be minimised or maximised.

    Returns:
        Union[float, Sequence[float]]: The penalised objective value or list of values.
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
        target (Union[float, Sequence[float]]): Base objective value (scalar or list of scalars).
        model (tf.keras.Model): Model whose complexity will be penalised.
        type (Literal['flops', 'params', None]): Type of penalty to apply, either "flops" or "params".
        flops_penalty_factor (float): Factor for FLOPs penalty.
        params_penalty_factor (float): Factor for parameters penalty.
        direction (Literal['minimize', 'maximize']): Whether the objective should be minimised or maximised.

    Returns:
        Union[float, Sequence[float]]: The penalised objective value or list of values.
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
