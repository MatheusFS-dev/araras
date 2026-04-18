from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Sequence

import csv
import glob
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
_DEFAULT_STUDY_SUBDIRS: tuple[str, ...] = (
    "args",
    "fig",
    "backup",
    "history",
    "scaler",
    "model",
    "logs",
    "tensorboard",
)


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
                optuna.trial.TrialState.WAITING, # Might have crashed midway, lets consider it done
                optuna.trial.TrialState.RUNNING, # Might have crashed midway, lets consider it done
            ),
        )
    )
    
    n_remaining_trials = max(0, num_trials - done_trials)
    
    vp.printf(
        (
            f"Found {done_trials} completed/pruned/failed/waiting/running trials out of {num_trials} requested."
            f" {n_remaining_trials} trials remaining in the study."
        ),
        tag="[ARARAS WARNING] ",
        color="yellow",
    )

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


def _collect_rankable_trials(
    study: optuna.Study, rank_key: str
) -> list[tuple[optuna.Trial, float]]:
    if rank_key != "value":
        available_attrs = {key for trial in study.trials for key in trial.user_attrs}
        if rank_key not in available_attrs:
            available = ", ".join(sorted(available_attrs)) or "<none>"
            raise ValueError(
                f"rank_key '{rank_key}' not found in any trial.user_attrs; available keys: {available}."
            )

    rankable: list[tuple[optuna.Trial, float]] = []
    for trial in study.trials:
        if rank_key == "value":
            value = trial.value
        else:
            if rank_key not in trial.user_attrs:
                continue
            value = trial.user_attrs.get(rank_key)

        if value is None:
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"rank_key '{rank_key}' must map to numeric values; "
                f"trial {trial.number} has value of type {type(value).__name__}."
            ) from exc

        if math.isnan(numeric_value):
            continue

        rankable.append((trial, numeric_value))

    if not rankable:
        raise ValueError(
            f"No numeric values found for rank_key '{rank_key}' in study '{study.study_name}'."
        )

    return rankable


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

    Raises:
        ValueError: If ``order`` is invalid, ``rank_key`` is missing from all
            trials or no numeric values are available for ranking.
    """
    if order not in ("descending", "ascending"):
        raise ValueError("order must be 'descending' or 'ascending'")

    rankable_trials = _collect_rankable_trials(study, rank_key)

    top_trials = [
        trial
        for trial, _ in sorted(
            rankable_trials,
            key=lambda item: item[1],
            reverse=(order == "descending"),
        )[:top_k]
    ]

    return top_trials


def _load_loss_history(
    history_dir: str | None, trial_number: int, convergence_epoch_column: str = "train_loss"
) -> list[float]:
    """Load per-epoch loss values for a specific trial.

    The function reads a CSV history file named ``trial_<trial_number>.csv``
    located inside ``history_dir`` and extracts the values for
    ``convergence_epoch_column``.
    When the requested column is missing, the helper falls back to a generic
    ``"loss"`` column to preserve compatibility with older exports. The helper
    also emits verbose warnings describing why convergence cannot be determined
    when the history is missing or malformed, providing visibility into "N/A"
    convergence epoch outputs.

    Args:
        history_dir (str | None): Directory containing the history CSV files. If
            ``None`` or empty, no history is loaded.
        trial_number (int): Identifier for the target Optuna trial whose history
            should be retrieved.
        convergence_epoch_column (str): Column name used to read loss values.
            Defaults to ``"train_loss"`` to match the standard training export
            schema.

    Returns:
        list[float]: Sequence of loss values ordered by epoch. Returns an empty
        list when the history file cannot be found, the expected columns are
        missing, or parsing fails.

    Raises:
        None.

    Notes:
        The file is parsed with :class:`csv.DictReader`, so column headers must
        be present in the first row. Only numeric values are retained; malformed
        entries are skipped silently.

        When the canonical ``trial_<id>.csv`` is missing, the loader searches
        for renamed files using patterns ``trial_<id>_*.csv`` and
        ``*trial_<id>*.csv``, defaulting to the first match to preserve backward
        compatibility.

    Warnings:
        This helper does not validate whether the loss values correspond to a
        specific dataset split. Ensure ``convergence_epoch_column`` references the correct
        training or validation metric to prevent misleading convergence reports.
    """

    if not history_dir:
        vp.printf(
            (
                f"Convergence epoch unavailable for trial {trial_number}: "
                "no history directory was provided."
            ),
            tag="[ARARAS WARNING] ",
            color="yellow",
        )
        return []

    default_history_path = os.path.join(history_dir, f"trial_{trial_number}.csv")
    history_path = default_history_path
    if not os.path.exists(history_path):
        fallback_patterns = [
            os.path.join(history_dir, f"trial_{trial_number}_*.csv"),
            os.path.join(history_dir, f"*trial_{trial_number}*.csv"),
        ]
        candidate_matches: list[str] = []
        for pattern in fallback_patterns:
            candidate_matches.extend(glob.glob(pattern))

        if candidate_matches:
            candidate_matches = sorted({os.path.abspath(p) for p in candidate_matches if os.path.isfile(p)})
            history_path = candidate_matches[0]
            if len(candidate_matches) > 1:
                vp.printf(
                    (
                        f"Multiple history files match trial {trial_number}: {candidate_matches}. "
                        f"Using '{history_path}'."
                    ),
                    tag="[ARARAS WARNING] ",
                    color="yellow",
                )
            else:
                vp.printf(
                    (
                        f"History file for trial {trial_number} appears to have been renamed to "
                        f"'{history_path}'. Using the detected file."
                    ),
                    tag="[ARARAS WARNING] ",
                    color="yellow",
                )
        else:
            vp.printf(
                (
                    f"Convergence epoch unavailable for trial {trial_number}: "
                    f"history file '{default_history_path}' not found after checking alternative patterns "
                    f"{fallback_patterns}."
                ),
                tag="[ARARAS WARNING] ",
                color="yellow",
            )
            return []

    losses: list[float] = []
    with open(history_path, newline="", encoding="utf-8") as history_file:
        reader = csv.DictReader(history_file)
        if not reader.fieldnames:
            vp.printf(
                (
                    f"Convergence epoch unavailable for trial {trial_number}: "
                    f"history file '{history_path}' has no header row."
                ),
                tag="[ARARAS WARNING] ",
                color="yellow",
            )
            return []

        column = (
            convergence_epoch_column if convergence_epoch_column in reader.fieldnames else "loss"
        )
        if column not in reader.fieldnames:
            vp.printf(
                (
                    f"Convergence epoch unavailable for trial {trial_number}: "
                    f"columns '{convergence_epoch_column}' and 'loss' are both missing in "
                    f"'{history_path}'."
                ),
                tag="[ARARAS WARNING] ",
                color="yellow",
            )
            return []

        for row in reader:
            try:
                losses.append(float(row.get(column, "")))
            except (TypeError, ValueError):
                continue

    if not losses:
        vp.printf(
            (
                f"Convergence epoch unavailable for trial {trial_number}: "
                f"no numeric values found in column '{column}' of "
                f"'{history_path}'."
            ),
            tag="[ARARAS WARNING] ",
            color="yellow",
        )

    return losses


def _calculate_convergence_epoch(
    losses: list[float], direction: optuna.study.StudyDirection
) -> int | None:
    """Compute the convergence epoch where 95% of the target loss is reached.

    The function derives a threshold based on the observed loss range and the
    optimization direction. For minimization problems, convergence is defined as
    the first epoch where the loss is within 5% of the minimum observed loss. In
    maximization scenarios, the threshold is mirrored against the maximum loss.

    Args:
        losses (list[float]): Sequence of loss values ordered by epoch.
        direction (optuna.study.StudyDirection): Study direction indicating
            whether the objective should be minimized or maximized.

    Returns:
        int | None: The one-based epoch index where convergence occurs, or
        ``None`` if the threshold cannot be determined (e.g., empty loss
        history).

    Raises:
        None
    """

    if not losses:
        return None

    max_loss = max(losses)
    min_loss = min(losses)

    if math.isclose(max_loss, min_loss):
        return 1

    if direction == optuna.study.StudyDirection.MINIMIZE:
        threshold = min_loss + 0.05 * (max_loss - min_loss)
        comparator = lambda loss: loss <= threshold
    else:
        threshold = max_loss - 0.05 * (max_loss - min_loss)
        comparator = lambda loss: loss >= threshold

    for epoch_index, loss in enumerate(losses, start=1):
        if comparator(loss):
            return epoch_index

    return len(losses)


def _format_trial_objective_value(trial: optuna.Trial, study: optuna.Study) -> str:
    """Format objective values for trial reports across Optuna study modes.

    Args:
        trial (optuna.Trial): Trial whose objective value(s) should be
            rendered.
        study (optuna.Study): Study used to detect whether optimization is
            single-objective or multi-objective.

    Returns:
        str: Scientific-notation representation of the trial objective. For
        single-objective studies, returns one scalar value or ``"N/A"`` when
        missing. For multi-objective studies, returns a bracketed comma-
        separated list in objective order, e.g. ``"[1.23e-03, 9.87e-01]"``.

    Raises:
        None.
    """

    # Multi-objective trials expose values through ``trial.values`` and raise
    # when ``trial.value`` is accessed, so we branch by study configuration.
    if len(study.directions) > 1:
        values = trial.values
        if not values:
            return "N/A"

        formatted_values = [
            "N/A" if value is None else format_scientific(value, max_precision=12)
            for value in values
        ]
        return "[" + ", ".join(formatted_values) + "]"

    value = trial.value
    if value is None:
        return "N/A"

    return format_scientific(value, max_precision=12)


def save_top_k_trials(
    top_trials: List[optuna.Trial],
    args_dir: str,
    study: optuna.Study,
    extra_attrs: Optional[List[str]] = None,
    history_dir: str | None = None,
    convergence_epoch_column: str = "train_loss",
    convergence_epoch_direction: optuna.study.StudyDirection | str = "minimize",
) -> None:
    """Persist summaries of the top-K trials to disk.

    The helper reconstructs the statistics collected by
    :func:`set_user_attr_model_stats` and renders them using
    :func:`araras.ml.model.stats.render_model_stats_report`. Each trial summary
    includes the scalar loss value, structured CPU/GPU metrics (when available),
    the convergence epoch derived from the specified column and direction, and
    optionally any user-provided attributes.

    Args:
        top_trials (List[optuna.Trial]): Trials ordered from best to worst that
            should be saved.
        args_dir (str): Directory where the summary files are written.
        study (optuna.Study): Parent study used to resolve sampler
            information.
        extra_attrs (Optional[List[str]]): Additional user attribute names to
            include at the end of each report. Defaults to ``None``.
        history_dir (str | None): Directory containing training history CSV
            files named ``trial_<id>.csv``. When provided, each trial report
            includes the convergence epoch calculated from the loss curve.
        convergence_epoch_column (str): Column name to use when extracting loss
            values from the history files. Defaults to ``"train_loss"``.
        convergence_epoch_direction (optuna.study.StudyDirection | str):
            Direction to apply when deriving the convergence epoch from the
            specified column. Accepts ``"minimize"`` or ``"maximize"``, defaulting
            to minimization.

    Returns:
        None. The summaries are written directly to ``args_dir``.

    Raises:
        OSError: If the destination directory or files cannot be created.

    Notes:
        The output files follow the same layout produced by
        :func:`write_model_stats_to_file`, ensuring consistent formatting
        between ad-hoc exports and Optuna summaries. Changing
        ``convergence_epoch_column`` does not affect other recorded metrics.

    Warnings:
        Existing files named ``top_<rank>_trial.txt`` in ``args_dir`` are
        overwritten. Ensure the destination directory is dedicated to the
        current export to avoid losing unrelated files. Providing an invalid
        ``convergence_epoch_column`` results in "N/A" convergence epochs without
        raising an exception.
    """

    try:
        convergence_direction = (
            convergence_epoch_direction
            if isinstance(convergence_epoch_direction, optuna.study.StudyDirection)
            else optuna.study.StudyDirection[convergence_epoch_direction.upper()]
        )
    except (AttributeError, KeyError) as exc:
        raise ValueError("convergence_epoch_direction must be 'minimize' or 'maximize'.") from exc

    os.makedirs(args_dir, exist_ok=True)

    for rank, trial in enumerate(top_trials, start=1):
        stats_map: Dict[str, Dict[str, Any]] = {}
        raw_stats = trial.user_attrs.get("model_stats")
        if isinstance(raw_stats, dict):
            stats_map = {key: value for key, value in raw_stats.items() if isinstance(value, dict)}
        else:
            for device_label in ("gpu", "cpu"):
                device_stats = trial.user_attrs.get(f"model_stats_{device_label}")
                if isinstance(device_stats, dict):
                    stats_map[device_label] = device_stats

        gpu_stats = stats_map.get("gpu")
        cpu_stats = stats_map.get("cpu")

        structural_stats = next(
            (stats_map[label] for label in ("gpu", "cpu") if isinstance(stats_map.get(label), dict)),
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
        value_text = _format_trial_objective_value(trial, study)

        losses = _load_loss_history(history_dir, trial.number, convergence_epoch_column)
        convergence_epoch = _calculate_convergence_epoch(losses, convergence_direction)
        convergence_epoch_text = "N/A" if convergence_epoch is None else convergence_epoch

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"Rank: {rank}\n")
            file.write(f"Trial ID: {trial.number}\n")
            file.write(f"Value: {value_text}\n")
            file.write(f"Convergence Epoch: {convergence_epoch_text}\n\n")
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
            ``["args", "fig", "backup", "history", "scaler", "model", "logs", "tensorboard"]``.

    Returns:
        tuple[str, ...]: ``study_dir`` followed by the paths for each
        requested subdirectory in the order provided by ``subdirs``.

    Raises:
        OSError: Propagated if any directory cannot be created.
    """
    if subdirs is None:
        subdirs = list(_DEFAULT_STUDY_SUBDIRS)

    study_dir = os.path.join(run_dir, study_name)
    dirs = {d: os.path.join(study_dir, d) for d in subdirs}

    # Create all directories
    for p in (study_dir, *dirs.values()):
        os.makedirs(p, exist_ok=True)

    # Return study_dir and all subdirectory paths in the specified order
    subdirectory_paths = [dirs[k] for k in subdirs]

    return study_dir, *subdirectory_paths


_INIT_STUDY_DIRS_FUNC = init_study_dirs


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
    objective: Callable[[optuna.Trial], float | Sequence[float]],
    run_dir: str,
    *,
    num_trials: int,
    sampler_seed: int,
    direction: str | Sequence[str],
    top_k: int,
    rank_key: str,
    order: str,
    extra_attrs: Sequence[str] | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    patience: int = 100,
    prune_threshold: int = 50,
    variance_threshold: float = 1e-10,
    convergence_epoch_column: str = "train_loss",
    convergence_epoch_direction: str = "minimize",
    init_study_dirs: Sequence[str] | None = None,
    cleanup_paths: Sequence[tuple[str, str]] | None = None,
    rename_paths: Sequence[tuple[str, str]] | None = None,
    **kwargs: Any,
) -> optuna.study.Study:
    """Run an Optuna hyperparameter optimization study.

    This helper sets up the directory structure, executes the optimization and
    post-processes the results. It mirrors the logic used throughout the
    notebooks while exposing parameters for customization. Exported training
    histories use ``convergence_epoch_column`` and
    ``convergence_epoch_direction`` to derive convergence epochs for the final
    summary files.

    Args:
        objective (Callable[[optuna.Trial], float | Sequence[float]]): Function
            that trains a model and returns the metric to optimize. If
            ``direction`` is a string, the objective must return a scalar. If
            ``direction`` is a sequence of strings (multiobjective mode), the
            objective must return a sequence with the same number of objective
            values.
        run_dir (str): Base directory for the study run.
        num_trials (int): Total number of trials to evaluate.
        sampler_seed (int): Seed for the Optuna sampler.
                direction (str | Sequence[str]): Optimization direction(s).
                        - If ``str``: single-objective mode (backward compatible), e.g.
                            ``"minimize"`` or ``"maximize"``.
                        - If ``Sequence[str]``: multiobjective mode using Optuna
                            ``directions=[...]``. At least two directions are required.
        top_k (int): Number of best trials to keep after optimization.
        rank_key (str): Study attribute used for ranking trials.
        order (str): Sorting order for ranking trials.
        extra_attrs (Sequence[str] | None): Additional user attributes copied
            when saving the top trials.
        pruner (optuna.pruners.BasePruner | None): Optional custom Optuna
            pruner. Defaults to :class:`HyperbandPruner`.
        sampler (optuna.samplers.BaseSampler | None): Optional custom Optuna
            sampler. Defaults to :class:`TPESampler`.
        patience (int): Number of completed trials allowed without improvement
            before stopping the study. ``None`` disables this check.
        prune_threshold (int): Number of consecutive pruned trials that triggers
            early stopping. ``None`` disables this check.
        variance_threshold (float): Improvement variance threshold used to
            detect stagnation. ``None`` disables this check.
        convergence_epoch_column (str): Column name to use when reading loss
            values from training history files for convergence epoch
            calculations. Defaults to ``"train_loss"``.
        convergence_epoch_direction (str): Direction to use when deriving the
            convergence epoch from the selected column (``"minimize"`` or
            ``"maximize"``). Defaults to ``"minimize"``.
        init_study_dirs (Sequence[str] | None): Optional list of subdirectory
            names forwarded to :func:`init_study_dirs` to control the layout of
            study artifacts. ``None`` uses the defaults defined by
            :func:`init_study_dirs`.
        cleanup_paths (Sequence[tuple[str, str]] | None): Optional pairs of
            ``(subdir_name, filename_template)`` used to delete artifacts from
            non-top trials. ``subdir_name`` must match an entry in
            ``init_study_dirs`` (with or without the ``_dir`` suffix). Defaults
            target models, figures, histories, tensorboard logs and scalers.
        rename_paths (Sequence[tuple[str, str]] | None): Optional pairs of
            ``(subdir_name, file_extension)`` used to rename artifacts from the
            top trials. ``subdir_name`` must match an entry in
            ``init_study_dirs`` (with or without the ``_dir`` suffix). Defaults
            target models, figures, histories and scalers.
        **kwargs (Any): Additional keyword arguments passed to the objective.
            If not provided, the directory parameters are automatically added
            using the paths created by :func:`init_study_dirs` and exposed under
            both ``<name>`` and ``<name>_dir`` (for backward compatibility):
                - args
                - backup
                - model
                - fig
                - logs
                - tensorboard
                - history
                - scaler
            Any extra subdirectories provided through ``init_study_dirs`` are
            also injected under both naming conventions. If a ``kwargs``
            mapping is provided it is merged after these defaults so callers
            can append or override objective arguments.

    Returns:
        optuna.study.Study: The completed study object.

    Raises:
        ValueError: If the study directories cannot be resolved, ``rank_key`` is
            missing or no numeric ranking values are available.
        Exception: Propagates any exception raised during optimization.

    Notes:
        - If an unexpected exception occurs the error is logged to
          ``training_error.log`` and the process aborts so an external monitor
          can restart it.
        - The temporary ``backup_dir`` is deleted regardless of success.
        - Setting ``variance_threshold``, ``prune_threshold`` or ``patience`` to
          ``None`` disables the corresponding callback.
                - Multiobjective mode currently skips ``ImprovementStagnation``,
                    ``StopIfKeepBeingPruned`` and ``StopWhenNoValueImprovement``
                    callbacks.
                - Multiobjective mode currently skips ``analyze_study``.

    Warnings:
        Passing a ``convergence_epoch_column`` that does not exist in the
        history files will result in missing convergence epoch information in
        the exported trial summaries.
        In multiobjective mode, using ``rank_key="value"`` is not supported
        because ranking requires a scalar criterion. Use a numeric user
        attribute key in ``rank_key``.
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
    )
    from araras.ml.optuna.analyzer import analyze_study

    subdirs = tuple(init_study_dirs) if init_study_dirs is not None else _DEFAULT_STUDY_SUBDIRS
    study_dir, *subdir_paths = _INIT_STUDY_DIRS_FUNC(run_dir, subdirs=subdirs)
    dir_map = dict(zip(subdirs, subdir_paths))
    user_objective_kwargs = kwargs.pop("kwargs", None)
    if user_objective_kwargs is not None:
        if not isinstance(user_objective_kwargs, dict):
            raise TypeError("run_study 'kwargs' must be a mapping when provided.")
        user_objective_kwargs = dict(user_objective_kwargs)

    def _resolve_dir(name: str) -> str | None:
        """Resolve directory keys regardless of whether they use the `_dir` suffix."""
        if name in dir_map:
            return dir_map[name]

        alt_name = name[:-4] if name.endswith("_dir") else f"{name}_dir"
        return dir_map.get(alt_name)

    def _require_dir(name: str) -> str:
        resolved = _resolve_dir(name)
        if resolved is None:
            alt_name = name[:-4] if name.endswith("_dir") else f"{name}_dir"
            raise ValueError(
                f"init_study_dirs must include '{name}' or '{alt_name}' "
                f"(got {list(dir_map)}) to run the study."
            )
        return resolved

    args_dir = _require_dir("args_dir")
    fig_dir = _require_dir("fig_dir")
    backup_dir = _require_dir("backup_dir")
    history_dir = _require_dir("history_dir")
    scaler_dir = _require_dir("scaler_dir")
    model_dir = _require_dir("model_dir")
    logs_dir = _require_dir("logs_dir")
    tensorboard_dir = _require_dir("tensorboard_dir")

    # Normalize optimization direction(s) while preserving backward
    # compatibility for single-objective studies.
    if isinstance(direction, str):
        directions_normalized = [direction]
        is_multiobjective = False
    elif isinstance(direction, Sequence):
        directions_normalized = list(direction)
        if not directions_normalized:
            raise ValueError("direction sequence must contain at least two entries.")
        if not all(isinstance(direction_name, str) for direction_name in directions_normalized):
            raise TypeError("All entries in direction sequence must be strings.")
        if len(directions_normalized) < 2:
            raise ValueError(
                "Multiobjective mode requires at least two directions; "
                "pass a single string for single-objective studies."
            )
        is_multiobjective = True
    else:
        raise TypeError("direction must be a string or a sequence of strings.")

    if is_multiobjective and rank_key == "value":
        raise ValueError(
            "rank_key='value' is not supported in multiobjective mode. "
            "Use a numeric trial.user_attrs key in rank_key."
        )

    try:
        study_create_kwargs: dict[str, Any] = {
            "study_name": os.path.basename(study_dir),
            "storage": f"sqlite:///{study_dir}/optuna_study.db",
            "pruner": pruner or optuna.pruners.HyperbandPruner(),
            "sampler": sampler or optuna.samplers.TPESampler(seed=sampler_seed),
            "load_if_exists": True,
        }
        if is_multiobjective:
            study_create_kwargs["directions"] = directions_normalized
        else:
            study_create_kwargs["direction"] = directions_normalized[0]

        study = optuna.create_study(**study_create_kwargs)

        callbacks_list: list[Callable[[optuna.Study, optuna.Trial], None]] = []
        if is_multiobjective:
            vp.printf(
                (
                    "Multiobjective mode detected: skipping early-stop callbacks "
                    "(ImprovementStagnation, StopIfKeepBeingPruned, StopWhenNoValueImprovement)."
                ),
                tag="[ARARAS WARNING] ",
                color="yellow",
            )
        else:
            if variance_threshold is not None:
                callbacks_list.append(ImprovementStagnation(variance_threshold=variance_threshold))
            if prune_threshold is not None:
                callbacks_list.append(StopIfKeepBeingPruned(threshold=prune_threshold))
            if patience is not None:
                callbacks_list.append(StopWhenNoValueImprovement(patience=patience))

        for name, path in dir_map.items():
            kwargs.setdefault(name, path)
            if name.endswith("_dir"):
                kwargs.setdefault(name[:-4], path)
            else:
                kwargs.setdefault(f"{name}_dir", path)

        objective_call_kwargs = dict(kwargs)
        if user_objective_kwargs:
            objective_call_kwargs.update(user_objective_kwargs)

        study.optimize(
            lambda trial: objective(
                trial,
                **objective_call_kwargs,
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

        default_cleanup_paths = [
            ("model_dir", "trial_{trial_id}.keras"),
            ("fig_dir", "trial_{trial_id}.png"),
            ("history_dir", "trial_{trial_id}.csv"),
            ("tensorboard_dir", "trial_{trial_id}"),
            ("scaler_dir", "trial_{trial_id}.pkl"),
        ]
        cleanup_config = list(cleanup_paths) if cleanup_paths is not None else default_cleanup_paths
        resolved_cleanup_paths: list[tuple[str, str]] = []
        for dir_key, pattern in cleanup_config:
            base_dir = _resolve_dir(dir_key)
            if base_dir is None:
                raise ValueError(
                    f"cleanup_paths entry '{dir_key}' is not in init_study_dirs; available {list(dir_map)}"
                )
            resolved_cleanup_paths.append((base_dir, pattern))

        default_rename_paths = [
            ("model_dir", ".keras"),
            ("fig_dir", ".png"),
            ("history_dir", ".csv"),
            ("scaler_dir", ".pkl"),
        ]
        rename_config = list(rename_paths) if rename_paths is not None else default_rename_paths
        resolved_rename_paths: list[tuple[str, str]] = []
        for dir_key, extension in rename_config:
            base_dir = _resolve_dir(dir_key)
            if base_dir is None:
                raise ValueError(
                    f"rename_paths entry '{dir_key}' is not in init_study_dirs; available {list(dir_map)}"
                )
            resolved_rename_paths.append((base_dir, extension))

        save_top_k_trials(
            top_trials,
            args_dir=args_dir,
            study=study,
            extra_attrs=extra_attrs or [],
            history_dir=history_dir,
            convergence_epoch_column=convergence_epoch_column,
            convergence_epoch_direction=convergence_epoch_direction,
        )
        cleanup_non_top_trials(
            {t.number for t in study.trials},
            {t.number for t in top_trials},
            resolved_cleanup_paths,
        )
        rename_top_k_files(top_trials, resolved_rename_paths)

        print()
        if is_multiobjective:
            vp.printf(
                (
                    "Multiobjective mode detected: analyze_study is not supported yet and will be skipped."
                ),
                tag="[ARARAS WARNING] ",
                color="yellow",
            )
        else:
            analyze_study(study, table_dir=os.path.join(study_dir, "analysis"))
        return study

    except ValueError:
        raise
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
