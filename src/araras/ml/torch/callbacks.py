"""Reusable PyTorch training callbacks."""

from typing import Dict, Optional

import optuna
import torch

from araras.utils.verbose_printer import VerbosePrinter


class EarlyStopping:
    """Early stopping tracker to prevent overfitting by monitoring metric improvement.

    Tracks a monitored metric (e.g., validation loss) and halts training when the
    metric stops improving for a specified number of epochs (patience). Stores the
    best model weights encountered during training, which can be restored after
    training completes. This is critical for recovering the actual best model state,
    since training often continues slightly past the optimum due to finite patience.

    Attributes:
        patience: Epochs with no improvement before stopping training.
        min_delta: Minimum change in metric to qualify as improvement (threshold).
        mode: "min" for loss-like metrics, "max" for accuracy-like metrics.
        best_score: Best metric value recorded so far (None until first epoch).
        best_state: Deep copy of model state_dict at best_score epoch.
        best_epoch: Epoch number (1-indexed) when best_score was achieved.
        num_bad_epochs: Current count of consecutive epochs without improvement.
        epoch: Counter tracking the total number of epochs seen.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: int = 1,
    ) -> None:
        """Initialize early stopping with stopping criteria and improvement thresholds.

        Args:
            patience: Number of epochs tolerated without improvement before stopping.
                Defaults to 10. Must be non-negative.
            min_delta: Minimum magnitude of change in monitored metric to qualify as
                improvement. For "min" mode, metric must decrease by at least min_delta;
                for "max" mode, metric must increase by at least min_delta. Helps avoid
                stopping on tiny, noise-driven improvements. Defaults to 0.0.
            mode: Direction of improvement: "min" for metrics where lower is better
                (e.g., loss, validation error) or "max" for metrics where higher is
                better (e.g., accuracy, F1-score). Defaults to "min".
            verbose: Verbosity level passed to VerbosePrinter for logging output.
                Defaults to 1.

        Raises:
            ValueError: If patience < 0 or mode not in {"min", "max"}.
        """
        if patience < 0:
            raise ValueError("patience must be >= 0")
        if mode not in {"min", "max"}:
            raise ValueError('mode must be "min" or "max"')
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.vp = VerbosePrinter(verbose=verbose)
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch: Optional[int] = None
        self.num_bad_epochs = 0
        self.epoch = 0

    def _is_improvement(self, metric: float) -> bool:
        """Check whether the current metric represents improvement over best recorded.

        Implements the comparison logic accounting for the configured mode ("min" or "max")
        and the min_delta threshold. On the first call, any metric is considered an
        improvement to bootstrap the tracking. Subsequent calls check if the new metric
        beats the best recorded metric by at least min_delta.

        Args:
            metric: The current epoch's metric value for comparison.

        Returns:
            True if metric improves on best_score according to mode and min_delta;
            False if metric is worse or only marginally better (within min_delta).
        """
        # Bootstrap: first metric is always considered an improvement to start tracking.
        if self.best_score is None:
            return True
        # For minimization (loss): improvement means beating previous best minus tolerance.
        # This allows small noise-driven improvements to not count as progress.
        if self.mode == "min":
            return metric < self.best_score - self.min_delta
        # For maximization (accuracy): improvement means exceeding previous best plus tolerance.
        return metric > self.best_score + self.min_delta

    def __call__(self, metric: float, model: torch.nn.Module) -> bool:
        """Evaluate epoch metric and update early stopping state; return stopping decision.

        Called once per training epoch to report the monitored metric and update the
        stopping counter. Returns True immediately when patience is exhausted, signaling
        that training should terminate. The model's state is saved whenever improvement
        occurs, enabling recovery of the best weights via restore_best_weights().

        Args:
            metric: The monitored metric value for the current epoch (e.g., val_loss).
            model: The model instance to save state from if this epoch improves on best.

        Returns:
            True if training should stop (num_bad_epochs > patience); False to continue.
        """
        self.epoch += 1
        if self._is_improvement(metric):
            # Improvement detected: reset patience counter and save model state.
            self.best_score = metric
            self.num_bad_epochs = 0
            self.best_epoch = self.epoch
            # Deep clone state_dict to avoid holding references to live parameters,
            # which could prevent garbage collection or cause unintended updates during
            # continued training if parameters change after this epoch.
            self.best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            return False
        # No improvement: increment patience counter and check stopping criterion.
        self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        """Restore model to its best-performing state recorded during training.

        After training completes and early stopping has triggered, call this to
        roll back the model to the weights from the epoch with the best monitored
        metric. This recovers the actual best model state, which typically occurs
        before the final training step (since training continues for patience epochs
        after the peak).

        Args:
            model: The model instance into which best weights will be loaded via
                load_state_dict. The model is modified in place.

        Returns:
            None.
        """
        # Silently return if no weights have been recorded (e.g., if first epoch was never good).
        if self.best_state is None:
            return
        if self.best_epoch is not None:
            self.vp.printf(
                f"restoring weights from best epoch {self.best_epoch}",
                tag="[ARARAS INFO EarlyStopping] ",
            )
        model.load_state_dict(self.best_state)


class TorchPruningCallback:
    """Optuna trial pruning callback that integrates early stopping into PyTorch training.

    Reports a monitored metric to an Optuna trial at regular intervals during training,
    allowing Optuna's pruning algorithm to automatically stop unpromising trials early.
    This avoids wasting computational budget on trials that are unlikely to be competitive,
    based on intermediate performance indicators.

    Unlike EarlyStopping (which stops based on patience), pruning decisions are made by
    the Optuna trial's configured pruning algorithm (e.g., MedianPruner, PercentilePruner),
    enabling sophisticated pruning strategies that consider the study's history.

    Typical usage: instantiate with an Optuna trial, add to your training loop, and
    catch TrialPruned exceptions to clean up and return early. This integrates Optuna
    directly into raw PyTorch training without requiring a high-level training framework.

    Attributes:
        trial: Optuna Trial instance for reporting metrics and checking pruning status.
        monitor: Metric key to monitor (e.g., "val_loss", "val_accuracy").
        interval: Report to trial every N epochs (>= 1). Higher values reduce overhead.
    """

    def __init__(self, trial: optuna.Trial, monitor: str, interval: int = 1) -> None:
        """Initialize pruning callback with trial and reporting configuration.

        Args:
            trial: Optuna Trial object. Used to report intermediate metric values
                via trial.report() and check pruning status via trial.should_prune().
            monitor: Name of the metric to monitor in the logs dict passed to
                on_epoch_end. Common examples: "val_loss", "val_accuracy", "train_loss".
                The exact key must match what is passed to on_epoch_end's logs dict.
            interval: Report metric every N epochs. Default is 1 (every epoch).
                Use higher values (e.g., 5) to reduce overhead if metrics are reported
                frequently. Must be >= 1.

        Raises:
            TypeError: If trial is not an optuna.Trial instance.
            ValueError: If interval < 1.
        """
        if not isinstance(trial, optuna.Trial):
            raise TypeError("trial must be an optuna.Trial instance")
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self.trial = trial
        self.monitor = monitor
        self.interval = interval

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None:
        """Report metric to Optuna and check if trial should be pruned.

        Called once per epoch with the current metrics. Reports the monitored metric
        to the trial using trial.report(). Then checks if Optuna's pruning algorithm
        recommends pruning this trial; if so, raises TrialPruned to halt training.

        This enables Optuna to make pruning decisions based on intermediate metric
        values (e.g., stopping trials that fall below expected performance early in
        training before wasting time on full training runs).

        Args:
            epoch: Current epoch number (1-indexed or 0-indexed; used as step value
                for trial.report). Typically incremented by training loop, no assumptions
                on starting value.
            logs: Dictionary of metrics, keyed by name (e.g., {"val_loss": 0.42,
                "train_loss": 0.5}). The metric specified in self.monitor will be
                extracted and reported. Should contain float values.

        Raises:
            optuna.TrialPruned: If the trial should be pruned according to the
                pruning algorithm (checked via trial.should_prune()).
            KeyError: Not raised explicitly; silently skips reporting if self.monitor
                is not in logs (returns without pruning).

        Notes:
            - Only reports at epoch % self.interval == 0 to reduce overhead.
            - Silently returns if the metric is not in logs (allows flexible metric
              reporting where some metrics may be missing or computed conditionally).
            - Does not modify logs or the model; purely observational side effects.
        """
        # Skip reporting for epochs not matching the interval pattern.
        if epoch % self.interval != 0:
            return

        # Extract metric from logs; silently skip if unavailable.
        current = logs.get(self.monitor)
        if current is None:
            # Metric not yet computed for this epoch (e.g., validation not every epoch).
            return

        # Report intermediate metric value to Optuna's trial object with step number.
        self.trial.report(float(current), step=epoch)

        # Check if Optuna's pruning algorithm recommends pruning this trial.
        # If so, raise TrialPruned to signal training loop to halt cleanly.
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

