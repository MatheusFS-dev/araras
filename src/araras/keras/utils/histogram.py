"""Utilities for profiling model size distributions."""

from typing import Callable
import optuna
import matplotlib.pyplot as plt
import tensorflow as tf

from araras.plot.configs import config_plt

config_plt("single-column")  # Configure matplotlib for single-column figures

def model_param_distribution(
    build_model_fn: Callable[[optuna.Trial, object], tf.keras.Model],
    hparams,
    bits_per_param: int,
    n_trials: int = 1000,
) -> None:
    """Sample random models and plot parameter and size histograms.

    Args:
        build_model_fn: Function that builds a Keras model given an Optuna
            ``Trial``.
        hparams: Hyperparameter object consumed by ``build_model_fn``.
        bits_per_param: Number of bits used to store each parameter.
        n_trials: Number of random trials to run.

    """

    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(sampler=sampler, direction="minimize")

    param_counts = []
    model_sizes_mb = []

    bar_len = 30
    for i in range(n_trials):
        trial = study.ask()
        model = build_model_fn(trial, hparams)
        n_params = model.count_params()
        param_counts.append(n_params)
        size_mb = (n_params * bits_per_param) / (8 * 1024 * 1024)
        model_sizes_mb.append(size_mb)
        study.tell(trial, 0.0)

        progress = (i + 1) / n_trials
        filled = int(progress * bar_len)
        bar = "=" * filled + ">" + " " * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
        print(f"\r[{bar}] {i + 1}/{n_trials}", end="", flush=True)

    print()

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(param_counts, bins=100, color="black")
    axes[0].set_xlabel("Number of parameters")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Parameter count distribution")

    axes[1].hist(model_sizes_mb, bins=100, color="black")
    axes[1].set_xlabel("Model size (MB)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Model size distribution")

    plt.tight_layout()
    plt.show()
