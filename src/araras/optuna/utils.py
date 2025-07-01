"""
This module contains utility functions for Optuna integration.

Functions:
    - get_remaining_trials: Returns the number of remaining trials in a study.
    - cleanup_non_top_trials: Removes files for trials not in the top-K set.
    - rename_top_k_files: Renames top-K trial files with ranking prefix.
    - save_trial_params_to_file: Saves trial parameters and metadata to a text file.
    - get_top_trials: Retrieves the top-K trials from a study based on ranking criteria.
    - save_top_k_trials: Saves top-K trials to text files.
    - init_study_dirs: Initializes study directory structure for experiments.
"""

import os
import math
import optuna
from typing import *
from araras.utils.misc import format_number, format_bytes, format_scientific, format_number_commas


def get_remaining_trials(study: optuna.Study, num_trials: int) -> list[optuna.trial.FrozenTrial]:
    """
    Returns a list of completed trials from the given Optuna study.

    Args:
        study (optuna.Study): The Optuna study to retrieve trials from.
        num_trials (int): The total number of trials to consider.

    Returns:
        list[optuna.trial.FrozenTrial]: A list of completed trials.
    """

    done_trials = len(
        study.get_trials(
            deepcopy=False,
            states=(
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL,
            ),
        )
    )
    n_remaining_trials = max(0, num_trials - done_trials)

    return n_remaining_trials


def cleanup_non_top_trials(
    all_trial_ids: Set[int], top_trial_ids: Set[int], cleanup_paths: List[Tuple[str, str]]
) -> None:
    """
    Remove files for trials not in the top-K set.

    Args:
        all_trial_ids (Set[int]): Set of all trial IDs in the study.
        top_trial_ids (Set[int]): Set of top-K trial IDs to preserve.
        cleanup_paths (List[Tuple[str, str]]): List of (base_directory, filename_template)
            tuples. The filename_template should contain '{trial_id}' placeholder.

    Raises:
        OSError: If file removal operations fail.
    """
    # Identify trials to clean up (non-top trials)
    trials_to_cleanup = all_trial_ids - top_trial_ids

    if not trials_to_cleanup:
        return  # Nothing to clean up

    # Remove files for non-top trials
    for trial_id in trials_to_cleanup:
        for base_dir, filename_template in cleanup_paths:
            try:
                file_path = os.path.join(base_dir, filename_template.format(trial_id=trial_id))
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError as e:
                # Log the error but continue with other files
                print(f"Warning: Failed to remove {file_path}: {e}")


def rename_top_k_files(top_trials: List[optuna.Trial], file_configs: List[Tuple[str, str]]) -> None:
    """
    Rename top-K trial files with ranking prefix.

    Args:
        top_trials (List[optuna.Trial]): List of top trials in ranked order.
        file_configs (List[Tuple[str, str]]): List of (base_directory, file_extension)
            tuples. Files are expected to follow pattern 'trial_{trial_id}{extension}'.

    Raises:
        OSError: If file rename operations fail.
    """
    for rank, trial in enumerate(top_trials, start=1):
        trial_id = trial.number

        for base_dir, extension in file_configs:
            try:
                old_filename = f"trial_{trial_id}{extension}"
                old_path = os.path.join(base_dir, old_filename)

                if os.path.exists(old_path):
                    new_filename = f"top_{rank}_{old_filename}"
                    new_path = os.path.join(base_dir, new_filename)
                    os.rename(old_path, new_path)
            except OSError as e:
                # Log the error but continue with other files
                print(f"Warning: Failed to rename {old_path}: {e}")


def save_trial_params_to_file(filepath: str, params: dict[str, float], **kwargs: str) -> None:
    """
    Save Optuna trial parameters and associated metadata to a text file.

    Args:
        filepath (str): Path where the parameter file should be saved.
        params (dict[str, float]): Dictionary of trial hyperparameters.
        **kwargs (str): Additional information such as trial ID, rank, or loss.

    Returns:
        None

    """
    with open(filepath, "w") as file:
        # Write metadata key-value pairs first
        file.writelines(f"{k}: {v}\n" for k, v in kwargs.items())

        # Write trial hyperparameters
        if params:
            file.write("\n")
            file.write("Trial hyperparameters:\n")
            file.writelines(f"  {k}: {v}\n" for k, v in params.items())


def get_top_trials(
    study: optuna.Study, top_k: int, rank_key: str = "value", rank_descending: bool = True
) -> List[optuna.Trial]:
    """
    Get the top-K trials from an Optuna study based on ranking criteria.

    Args:
        study (optuna.Study): The completed Optuna study.
        top_k (int): Number of top trials to retrieve.
        rank_key (str): Key to rank trials by ("value" for objective value,
                       or any user attribute key).
        rank_descending (bool): If True, rank in descending order (higher is better).
                               If False, rank in ascending order (lower is better).

    Returns:
        List[optuna.Trial]: List of top-K trials sorted by the ranking criteria.
    """
    # Define getter function based on rank_key
    if rank_key == "value":
        getter = lambda t: t.value
    else:
        getter = lambda t: t.user_attrs.get(rank_key, float("nan"))

    # Filter and sort trials
    top_trials = sorted(
        (t for t in study.trials if (v := getter(t)) is not None and not math.isnan(v)),
        key=getter,
        reverse=rank_descending,
    )[:top_k]

    return top_trials


def save_top_k_trials(
    top_trials: List[optuna.Trial],
    args_dir: str,
    study: optuna.Study,
    extra_attrs: Optional[List[str]] = None,
) -> None:
    """
    Save top-K trials to text files.

    Args:
        top_trials (List[optuna.Trial]): List of trials to save.
        args_dir (str): Directory to save trial parameter files.
        study (optuna.Study): The Optuna study (needed for sampler info).
        extra_attrs (Optional[List[str]]): List of additional user attributes to save.
                                          If None, defaults to common accuracy metrics.
    """

    # Save each top trial
    for rank, trial in enumerate(top_trials):
        # Always saved attributes
        trial_id = trial.number
        trial_params = trial.params
        trial_loss = trial.value
        trial_num_params = trial.user_attrs.get("num_params", None)
        trial_model_size = trial.user_attrs.get("model_size", None)
        trial_flops = trial.user_attrs.get("flops", None)
        trial_macs = trial.user_attrs.get("macs", None)
        trial_mem_usage = trial.user_attrs.get("peak_memory_usage", None)
        trial_inference_time = trial.user_attrs.get("inference_time", None)
        trial_avg_power = trial.user_attrs.get("avg_power", None)
        trial_avg_energy = trial.user_attrs.get("avg_energy", None)
        trial_summary = trial.user_attrs.get("model_summary", None)
        print(trial_summary)

        # Extract extra attributes
        extra_values = {}
        for attr in extra_attrs:
            extra_values[attr] = trial.user_attrs.get(attr, None)

        # Save trial parameters to file manually
        filepath = os.path.join(args_dir, f"top_{rank + 1}_trial.txt")
        with open(filepath, "w") as file:
            # Write metadata
            file.write(f"Rank: {rank + 1}\n")
            file.write(f"Trial ID: {trial_id}\n")
            file.write(f"Loss: {format_scientific(trial_loss, max_precision=12)}\n")
            file.write(f"Number of parameters: {format_number_commas(trial_num_params)}\n")
            file.write(f"Model size: {format_bytes(trial_model_size)}\n")
            file.write(f"FLOPs: {format_number(trial_flops)}FLOPs\n")
            file.write(f"MACs: {format_number(trial_macs)}MACs\n")
            file.write(f"Peak memory usage: {format_bytes(trial_mem_usage)}\n")
            file.write(f"Inference time: {format_scientific(trial_inference_time, max_precision=4)} s\n")
            file.write(f"Average power consumption: {format_scientific(trial_avg_power, max_precision=4)} W\n")
            file.write(f"Average energy consumption: {format_scientific(trial_avg_energy, max_precision=4)} J\n")
            file.write(f"Sampler: {study.sampler.__class__.__name__}\n")

            # Write extra attributes
            for attr, value in extra_values.items():
                file.write(f"{attr}: {value}\n")

            file.write(f"\nModel summary: {trial_summary}\n")

            # Write trial hyperparameters
            if trial_params:
                file.write("\n")
                file.write("Trial hyperparameters:\n")
            for k, v in trial_params.items():
                file.write(f"  {k}: {v}\n")


def init_study_dirs(run_dir, study_name="optuna_study", subdirs=None):
    """
    Create and return study directory structure for experiments.

    Args:
        run_dir (str): Base directory for the run
        study_name (str): Name of the study directory (default: "optuna_study")
        subdirs (list): List of subdirectory names to create
                       (default: ["args", "figures", "backup", "history", "models", "logs"])

    Returns:
        tuple: (study_dir, *subdirectory_paths) in the order specified by subdirs
    """
    if subdirs is None:
        subdirs = ["args", "figures", "backup", "history", "models", "logs"]

    study_dir = os.path.join(run_dir, study_name)
    dirs = {d: os.path.join(study_dir, d) for d in subdirs}

    # Create all directories
    for p in (study_dir, *dirs.values()):
        os.makedirs(p, exist_ok=True)

    # Return study_dir and all subdirectory paths in the specified order
    subdirectory_paths = [dirs[k] for k in subdirs]

    return study_dir, *subdirectory_paths
