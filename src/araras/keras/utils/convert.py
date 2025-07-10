"""
This module provides a utility to convert a Keras `.keras` model archive.

Functions:
    - convert_to_saved_model: Convert a Keras `.keras` model archive into a zipped TensorFlow SavedModel.

Example:
    >>> from araras.keras.utils.convert import convert_to_saved_model
    >>> convert_to_saved_model(...)
"""
from araras.commons import *


import tempfile
import zipfile
from pathlib import Path

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
