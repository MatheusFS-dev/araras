"""
Callback that stops an Optuna study after many consecutive pruned trials.

Classes:
    - StopIfKeepBeingPruned: Interrupts optimization if ``threshold`` consecutive
      trials are pruned.

Example:
    >>> from araras.optuna.callbacks.keep_being_pruned import StopIfKeepBeingPruned
    >>> StopIfKeepBeingPruned(threshold=3)
"""
from araras.commons import *
import optuna


class StopIfKeepBeingPruned:
    """
    A callback for Optuna studies that stops the optimization process
    when a specified number of consecutive trials are pruned.

    Args:
        threshold (int): The number of consecutive pruned trials required to stop the study.
    """

    def __init__(self, threshold: int):
        """
        Initializes the callback with the pruning threshold.

        Args:
            threshold (int): The number of consecutive pruned trials required to stop the study.
        """
        self.threshold = threshold
        self._consequtive_pruned_count = 0  # Tracks the count of consecutive pruned trials.

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Invoked after each trial to check its state and decide whether to stop the study.

        Args:
            study (optuna.study.Study): The Optuna study object.
            trial (optuna.trial.FrozenTrial): The trial object containing the state of the trial.
        """
        # Increment the count if the trial was pruned; reset otherwise.
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        # Stop the study if the threshold of consecutive pruned trials is reached.
        if self._consequtive_pruned_count >= self.threshold:
            study.stop()
