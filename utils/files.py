"""
Utility functions for saving results.

Functions:
    - save_trial_params_to_file: Saves trial parameters and metadata to a text file.

Example usage:
    save_trial_params_to_file("trial1.txt", {"lr": 0.01}, trial_id="1", loss="0.15")
"""

from typing import *


def save_trial_params_to_file(filepath: str, params: dict[str, float], **kwargs: str) -> None:
    """
    Save Optuna trial parameters and associated metadata to a text file.

    Logic:
        -> Open file for writing
        -> Write key-value metadata (from kwargs)
        -> Write trial parameters section with indentation

    Args:
        filepath (str): Path where the parameter file should be saved.
        params (dict[str, float]): Dictionary of trial hyperparameters.
        **kwargs (str): Additional information such as trial ID, rank, or loss.

    Returns:
        None

    Example:
        save_trial_params_to_file("trial1.txt", {"lr": 0.01}, trial_id="1", loss="0.15")
    """
    with open(filepath, "w") as file:
        # Write metadata key-value pairs first
        file.writelines(f"{k}: {v}\n" for k, v in kwargs.items())

        # Write trial hyperparameters
        if params:
            file.write("\n")
            file.write("Trial hyperparameters:\n")
            file.writelines(f"  {k}: {v}\n" for k, v in params.items())
