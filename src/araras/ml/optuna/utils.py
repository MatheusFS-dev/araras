from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Sequence

import os, shutil
import traceback
import math
import optuna
import tensorflow as tf
from araras.ml.model.stats import render_model_stats_report
from araras.utils.misc import format_scientific
from araras.utils.system import _get_nvidia_smi_data

from araras.utils.verbose_printer import VerbosePrinter

vp = VerbosePrinter()

_CONSECUTIVE_OOM_ERRORS = 0


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
    Remove files or directories for trials not in the top-K set.

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

                # Check if it is a directory or file
                if os.path.isdir(file_path):
                    # Remove directory and all its contents
                    shutil.rmtree(file_path)
                else:
                    # Remove file
                    if os.path.isfile(file_path):
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
    """Persist trial parameters and metadata in a plain-text file.

    Args:
        filepath (str): Destination where the summary should be written.
        params (dict[str, float]): Dictionary containing the hyperparameters evaluated during the
            trial.
        **kwargs (str): Additional metadata such as trial identifier, rank or loss
            values. Each keyword is emitted as a ``key: value`` line before the
            parameter block.

    Raises:
        OSError: If the file cannot be opened or written.
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
    study: optuna.Study,
    top_k: int,
    rank_key: str = "value",
    order: str = "descending",
) -> List[optuna.Trial]:
    """
    Get the top-K trials from an Optuna study based on ranking criteria.

    Args:
        study (optuna.Study): The completed Optuna study.
        top_k (int): Number of top trials to retrieve.
        rank_key (str): Key to rank trials by ("value" for objective value,
                       or any user attribute key).
        order (str): "descending" for highest values first or "ascending" for
            lowest values first.

    Returns:
        List[optuna.Trial]: List of top-K trials sorted by the ranking criteria.
    """
    # Define getter function based on rank_key
    if rank_key == "value":
        getter = lambda t: t.value
    else:
        getter = lambda t: t.user_attrs.get(rank_key, float("nan"))

    if order not in ("descending", "ascending"):
        raise ValueError("order must be 'descending' or 'ascending'")

    # Filter and sort trials
    top_trials = sorted(
        (t for t in study.trials if (v := getter(t)) is not None and not math.isnan(v)),
        key=getter,
        reverse=(order == "descending"),
    )[:top_k]

    return top_trials



def save_top_k_trials(
    top_trials: List[optuna.Trial],
    args_dir: str,
    study: optuna.Study,
    extra_attrs: Optional[List[str]] = None,
) -> None:
    """Persist summaries of the top-K trials to disk.

    The helper reconstructs the statistics collected by
    :func:`set_user_attr_model_stats` and renders them using
    :func:`araras.ml.model.stats.render_model_stats_report`. Each trial
    summary includes the scalar loss value, structured CPU/GPU metrics (when
    available), and optionally any user-provided attributes.

    Args:
        top_trials (List[optuna.Trial]): Trials ordered from best to worst that
            should be saved.
        args_dir (str): Directory where the summary files are written.
        study (optuna.Study): Parent study used to resolve sampler
            information.
        extra_attrs (Optional[List[str]]): Additional user attribute names to
            include at the end of each report. Defaults to ``None``.

    Returns:
        None: The summaries are written directly to ``args_dir``.

    Raises:
        OSError: If the destination directory or files cannot be created.

    Notes:
        The output files follow the same layout produced by
        :func:`write_model_stats_to_file`, ensuring consistent formatting
        between ad-hoc exports and Optuna summaries.

    Warnings:
        Existing files named ``top_<rank>_trial.txt`` in ``args_dir`` are
        overwritten. Ensure the destination directory is dedicated to the
        current export to avoid losing unrelated files.
    """

    os.makedirs(args_dir, exist_ok=True)

    for rank, trial in enumerate(top_trials, start=1):
        stats_map: Dict[str, Dict[str, Any]] = {}
        raw_stats = trial.user_attrs.get("model_stats")
        if isinstance(raw_stats, dict):
            stats_map = {
                key: value for key, value in raw_stats.items() if isinstance(value, dict)
            }
        else:
            for device_label in ("gpu", "cpu"):
                device_stats = trial.user_attrs.get(f"model_stats_{device_label}")
                if isinstance(device_stats, dict):
                    stats_map[device_label] = device_stats

        gpu_stats = stats_map.get("gpu")
        cpu_stats = stats_map.get("cpu")

        structural_stats = next(
            (
                stats_map[label]
                for label in ("gpu", "cpu")
                if isinstance(stats_map.get(label), dict)
            ),
            None,
        )

        if structural_stats is None:
            structural_stats = {}
            for key, attr_name in (
                ("parameters", "num_params"),
                ("model_size", "model_size"),
                ("flops", "flops"),
                ("macs", "macs"),
            ):
                value = trial.user_attrs.get(attr_name)
                if value is not None:
                    structural_stats[key] = value
            summary_text = trial.user_attrs.get("model_summary")
            if summary_text is not None:
                structural_stats["summary"] = summary_text

        report_extra: Dict[str, Any] = {}
        stored_extra = trial.user_attrs.get("model_stats_extra_attrs")
        if isinstance(stored_extra, dict):
            report_extra.update(stored_extra)
        if extra_attrs:
            for attr_name in extra_attrs:
                report_extra[attr_name] = trial.user_attrs.get(attr_name)
        report_extra["Sampler"] = study.sampler.__class__.__name__

        report_text = render_model_stats_report(
            structural_stats,
            cpu_stats=cpu_stats if isinstance(cpu_stats, dict) else None,
            gpu_stats=gpu_stats if isinstance(gpu_stats, dict) else None,
            extra_attrs=report_extra,
        )

        filepath = os.path.join(args_dir, f"top_{rank}_trial.txt")
        loss_value = trial.value
        loss_text = "N/A"
        if loss_value is not None:
            loss_text = format_scientific(loss_value, max_precision=12)

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"Rank: {rank}\n")
            file.write(f"Trial ID: {trial.number}\n")
            file.write(f"Loss: {loss_text}\n\n")
            file.write(report_text + "\n")

            if trial.params:
                file.write("\nTrial hyperparameters:\n")
                for key, value in trial.params.items():
                    file.write(f"  {key}: {value}\n")


def init_study_dirs(run_dir, study_name="optuna_study", subdirs=None):
    """Create the directory layout used for Optuna studies.

    The helper mirrors the folder structure expected by :func:`run_study`,
    guaranteeing that every artifact directory exists before the optimization
    starts. Custom layouts can be provided through ``subdirs`` when needed.

    Notes:
        Providing an explicit ``subdirs`` sequence allows callers to reuse the
        helper for bespoke experiment layouts while still benefiting from the
        creation guarantees.

    Args:
        run_dir (str): Base directory for the optimization run.
        study_name (str): Name of the study directory created inside
            ``run_dir``. Defaults to ``"optuna_study"``.
        subdirs (list[str] | None): Explicit list of subdirectories to create
            under the study directory. When ``None`` the defaults are used,
            matching the keyword arguments consumed by
            :func:`run_study`:
            ``["args", "figures", "backup", "history", "scaler", "models", "logs", "tensorboard"]``.

    Returns:
        tuple[str, ...]: ``study_dir`` followed by the paths for each
        requested subdirectory in the order provided by ``subdirs``.

    Raises:
        OSError: Propagated if any directory cannot be created.
    """
    if subdirs is None:
        subdirs = [
            "args",
            "figures",
            "backup",
            "history",
            "scaler",
            "models",
            "logs",
            "tensorboard",
        ]

    study_dir = os.path.join(run_dir, study_name)
    dirs = {d: os.path.join(study_dir, d) for d in subdirs}

    # Create all directories
    for p in (study_dir, *dirs.values()):
        os.makedirs(p, exist_ok=True)

    # Return study_dir and all subdirectory paths in the specified order
    subdirectory_paths = [dirs[k] for k in subdirs]

    return study_dir, *subdirectory_paths


def log_trial_error(
    trial,
    exc,
    logs_dir,
    prune_on={
        tf.errors.ResourceExhaustedError: None,
        tf.errors.InternalError: None,
        tf.errors.UnavailableError: None,
        # tf.errors.UnknownError: "CUDNN failed to allocate the scratch space",
    },
    propagate={
        optuna.exceptions.TrialPruned: None,
    },
    force_crash_oom: int | None = 10,
):
    """Log and manage trial errors, optionally aborting after repeated OOMs.

    The function records information about a failed Optuna trial and decides
    whether to prune or propagate the exception. It can also terminate the
    process after a configurable number of consecutive
    ``tf.errors.ResourceExhaustedError`` occurrences, which typically indicate
    Out-Of-Memory (OOM) issues.

    Notes:
        Setting ``force_crash_oom`` to ``None`` disables the crash
        mechanism.

    Args:
        trial (Any): Trial that encountered the error.
        exc (Any): Exception raised during the trial execution.
        logs_dir (Any): Directory where the error log file should be saved.
        prune_on (Any): Mapping of exception types to substrings that trigger
            pruning. Defaults to ``{tf.errors.ResourceExhaustedError: None,
            tf.errors.InternalError: None, tf.errors.UnavailableError: None}``.
        propagate (Any): Mapping of exception types to substrings that trigger
            propagation. Defaults to ``{optuna.exceptions.TrialPruned: None}``.
        force_crash_oom (int | None): Minimum number of consecutive
            ``tf.errors.ResourceExhaustedError``, ``tf.errors.InternalError`` or
            ``tf.errors.UnavailableError`` exceptions before the process is
            aborted. Defaults to ``10``; ``None`` disables this behaviour.

    Raises:
        optuna.TrialPruned: If the error matches any pruning rules.
        Exception: If the error matches any propagation rules, if the number of
            consecutive OOM errors reaches ``force_crash_oom`` or if
            no rules match.
    """

    prune_on = {} if prune_on is None else prune_on
    propagate = {} if propagate is None else propagate

    # If exception should just propagate, re-raise now
    for exc_type, msg_substr in propagate.items():
        if isinstance(exc, exc_type) and (msg_substr is None or msg_substr in str(exc)):
            raise exc

    path = os.path.join(logs_dir, f"trial_{trial.number}.log")
    with open(str(path), "w", encoding="utf-8") as log_file:
        # Gather GPU statistics using nvidia-smi
        try:
            gpu_data = _get_nvidia_smi_data()
            log_file.write("GPU Stats:\n")
            if gpu_data:
                for gpu in gpu_data:
                    log_file.write(
                        f"  GPU {gpu['index']} - {gpu['name']}: "
                        f"used {gpu['used_mb']}MB / {gpu['total_mb']}MB, "
                        f"free {gpu['free_mb']}MB, "
                        f"temp {gpu['temperature']}C, "
                        f"util {gpu['utilization']}%\n"
                    )
            else:
                log_file.write("  No GPU information available\n")
        except Exception as e:  # noqa: BLE001
            log_file.write(f"Failed to collect GPU stats: {e}\n")
        log_file.write(f"Trial: {trial.number}\n")
        log_file.write(f"Params: {trial.params}\n")
        log_file.write(f"User Attributes: {trial.user_attrs}\n")
        log_file.write(f"Exception Type: {type(exc).__name__}\n")
        log_file.write(f"Exception Message: {str(exc)}\n")
        log_file.write("Traceback:\n")
        log_file.write(traceback.format_exc())

    global _CONSECUTIVE_OOM_ERRORS
    if isinstance(
        exc, (tf.errors.ResourceExhaustedError, tf.errors.InternalError, tf.errors.UnavailableError)
    ):
        _CONSECUTIVE_OOM_ERRORS += 1
    else:
        _CONSECUTIVE_OOM_ERRORS = 0

    if force_crash_oom is not None and _CONSECUTIVE_OOM_ERRORS >= force_crash_oom:
        vp.printf(
            f"Reached {_CONSECUTIVE_OOM_ERRORS} consecutive OOM errors. Aborting.",
            tag="[ARARAR ERROR] ",
            color="red",
        )
        os.abort()  # Crash the process

    # Check each prune rule
    for exc_type, msg_substr in prune_on.items():
        if isinstance(exc, exc_type) and (msg_substr is None or msg_substr in str(exc)):
            vp.printf(
                (
                    f"Trial {trial.number} failed with {type(exc).__name__}"
                    f"{', message contains ' + repr(msg_substr) if msg_substr else ''}. Pruning."
                ),
                tag="[ARARAS WARNING] ",
                color="yellow",
            )
            raise optuna.TrialPruned() from exc

    raise exc  # Otherwise re-raise


def run_study(
    objective: Callable[[optuna.Trial], float],
    run_dir: str,
    *,
    epochs: int,
    num_trials: int,
    sampler_seed: int,
    direction: str,
    top_k: int,
    rank_key: str,
    order: str,
    extra_attrs: Sequence[str] | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    patience: int = 100,
    prune_threshold: int = 50,
    variance_threshold: float = 1e-10,
    **kwargs: Any,
) -> optuna.study.Study:
    """Run an Optuna hyperparameter optimization study.

    This helper sets up the directory structure, executes the optimization and
    post-processes the results. It mirrors the logic used throughout the
    notebooks while exposing parameters for customization.

    Notes:
        - If an unexpected exception occurs the error is logged to
          ``training_error.log`` and the process aborts so an external
          monitor can restart it.
        - The temporary ``backup_dir`` is deleted regardless of success.
        - Setting ``variance_threshold``, ``prune_threshold`` or ``patience`` to
          ``None`` disables the corresponding callback.

    Args:
        objective (Callable[[optuna.Trial], float]): Function that trains a model and returns the metric to
            optimize.
        run_dir (str): Base directory for the study run.
        epochs (int): Number of training epochs per trial.
        num_trials (int): Total number of trials to evaluate.
        sampler_seed (int): Seed for the Optuna sampler.
        direction (str): Optimization direction, e.g. ``"minimize"`` or ``"maximize"``.
        top_k (int): Number of best trials to keep after optimization.
        rank_key (str): Study attribute used for ranking trials.
        order (str): Sorting order for ranking trials.
        extra_attrs (Sequence[str] | None): Additional user attributes copied when saving the top trials.
        pruner (optuna.pruners.BasePruner | None): Optional custom Optuna pruner. Defaults to :class:`HyperbandPruner`.
        sampler (optuna.samplers.BaseSampler | None): Optional custom Optuna sampler. Defaults to :class:`TPESampler`.
        patience (int): Number of completed trials allowed without improvement before
            stopping the study. ``None`` disables this check.
        prune_threshold (int): Number of consecutive pruned trials that triggers early
            stopping. ``None`` disables this check.
        variance_threshold (float): Improvement variance threshold used to detect
            stagnation. ``None`` disables this check.
        **kwargs (Any): Additional keyword arguments passed to the objective.
            This is merged with default args:
                - backup_dir: Directory for temporary trial backups.
                - model_dir: Directory to save trained models.
                - fig_dir: Directory to save training figures.
                - logs_dir: Directory to save training logs.
                - tensorboard_dir: Directory for TensorBoard logs.
                - history_dir: Directory to save training history CSV files.
                - scaler_dir: Directory to persist fitted scalers.

    Returns:
        optuna.study.Study: The completed study object.

    Raises:
        Exception: Propagates any exception raised during optimization.
    """

    from araras.ml.optuna.callbacks import (
        ImprovementStagnation,
        StopIfKeepBeingPruned,
        StopWhenNoValueImprovement,
    )
    from araras.ml.optuna.utils import (
        get_remaining_trials,
        cleanup_non_top_trials,
        rename_top_k_files,
        get_top_trials,
        save_top_k_trials,
        init_study_dirs,
    )
    from araras.ml.optuna.analyzer import analyze_study
    from araras.utils.misc import clear

    (
        study_dir,
        args_dir,
        fig_dir,
        backup_dir,
        history_dir,
        scaler_dir,
        model_dir,
        logs_dir,
        tensorboard_dir,
    ) = init_study_dirs(run_dir)

    try:
        study = optuna.create_study(
            study_name=os.path.basename(study_dir),
            storage=f"sqlite:///{study_dir}/optuna_study.db",
            pruner=pruner or optuna.pruners.HyperbandPruner(),
            sampler=sampler or optuna.samplers.TPESampler(seed=sampler_seed),
            load_if_exists=True,
            direction=direction,
        )

        callbacks_list: list[Callable[[optuna.Study, optuna.Trial], None]] = []
        if variance_threshold is not None:
            callbacks_list.append(ImprovementStagnation(variance_threshold=variance_threshold))
        if prune_threshold is not None:
            callbacks_list.append(StopIfKeepBeingPruned(threshold=prune_threshold))
        if patience is not None:
            callbacks_list.append(StopWhenNoValueImprovement(patience=patience))

        default_objective_kwargs = {
            "backup_dir": backup_dir,
            "model_dir": model_dir,
            "fig_dir": fig_dir,
            "logs_dir": logs_dir,
            "tensorboard_dir": tensorboard_dir,
            "history_dir": history_dir,
            "scaler_dir": scaler_dir,
        }
        objective_kwargs = {**default_objective_kwargs, **kwargs}

        study.optimize(
            lambda trial: objective(
                trial,
                epochs=epochs,
                size_penalizer=None,
                **objective_kwargs,
            ),
            n_trials=get_remaining_trials(study, num_trials),
            callbacks=callbacks_list,
            catch=(),
            gc_after_trial=True,
        )

        top_trials = get_top_trials(
            study,
            top_k=top_k,
            rank_key=rank_key,
            order=order,
        )

        cleanup_paths = [
            (model_dir, "trial_{trial_id}.keras"),
            (fig_dir, "trial_{trial_id}.png"),
            (history_dir, "trial_{trial_id}.csv"),
            (tensorboard_dir, "trial_{trial_id}"),
        ]

        rename_paths = [
            (model_dir, ".keras"),
            (fig_dir, ".png"),
            (history_dir, ".csv"),
        ]

        save_top_k_trials(
            top_trials,
            args_dir=args_dir,
            study=study,
            extra_attrs=extra_attrs or [],
        )
        cleanup_non_top_trials(
            {t.number for t in study.trials},
            {t.number for t in top_trials},
            cleanup_paths,
        )
        rename_top_k_files(top_trials, rename_paths)

        clear(), analyze_study(study, table_dir=os.path.join(study_dir, "analysis"))
        return study

    except Exception as e:  # pragma: no cover - runtime failure path
        print(f"\n An error occurred: {e}\n")
        traceback.print_exc()

        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, "training_error.log"), "a") as f:
            f.write(f"An error occurred during training:\n{e}\n{traceback.format_exc()}\n\n")

        os.abort()
        raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)
        if os.path.isdir(logs_dir) and not os.listdir(logs_dir):
            os.rmdir(logs_dir)
