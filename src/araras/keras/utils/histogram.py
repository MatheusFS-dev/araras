"""
This module provides utilities for analyzing and visualizing the distribution of model parameters

Functions:
    - model_param_distribution: Sample random models and plot parameter and size histograms.

Example:
    >>> from araras.keras.utils.histogram import model_param_distribution
    >>> model_param_distribution(...)
"""
from araras.commons import *

import optuna
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from araras.plot.configs import config_plt
from .memory_estimator import estimate_training_memory

config_plt("double-column")  # Configure matplotlib for double-column figures

def model_param_distribution(
    build_model_fn: Callable[[optuna.Trial], tf.keras.Model],
    bits_per_param: int,
    batch_size: int = 1,
    n_trials: int = 1000,
) -> None:
    """Sample random models and plot parameter and size histograms.

    Args:
        build_model_fn: Function that builds a Keras model given an Optuna
            ``Trial``.
        bits_per_param: Number of bits used to store each parameter.
        n_trials: Number of random trials to run.

    """

    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(sampler=sampler, direction="minimize")

    param_counts = []
    model_sizes_mb = []
    training_memory = []

    progress_iter = range(n_trials)
    if n_trials:
        progress_iter = tqdm(
            progress_iter,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )
    for _ in progress_iter:
        trial = study.ask()
        model = build_model_fn(trial)
        
        n_params = model.count_params()
        param_counts.append(n_params)
        
        size_mb = (n_params * bits_per_param) / (8 * 1024 * 1024)
        model_sizes_mb.append(size_mb)
        
        training_memory_mb = estimate_training_memory(model, batch_size=1) / (1024 * 1024)
        training_memory.append(training_memory_mb)

        study.tell(trial, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].hist(param_counts, bins=100, color="black")
    axes[0].set_xlabel("Number of parameters")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Parameter count distribution")

    axes[1].hist(model_sizes_mb, bins=100, color="black")
    axes[1].set_xlabel("Model size (MB)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Model size distribution")
    
    axes[2].hist(training_memory, bins=100, color="black")
    axes[2].set_xlabel("Training memory (MB)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Training memory distribution")

    plt.tight_layout()
    plt.show()
