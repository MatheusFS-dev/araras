"""
This module provides utilities for analyzing and visualizing the distribution of model parameters
and sizes in Keras models using Optuna for hyperparameter optimization.

How to use:
    1. Define a function `build_model_fn` that takes an Optuna `Trial` and returns a compiled Keras model.
    2. Call `model_param_distribution` with the `build_model_fn`, number of bits per parameter, and the 
       number of trials you want to run.
    3. The function will sample random models, compute their parameter counts and sizes, and
       plot histograms of these distributions.
       
Example usage:
    ```python
    import tensorflow as tf
    import optuna
    from araras.keras.utils.histogram import model_param_distribution
    
    def build_model_fn(trial, hparams):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(trial.suggest_int("units", 32, 512),
                                   activation="relu",
                                   input_shape=(hparams.input_dim,)),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
        
    # use lambda to pass trial and hparams
    model_param_distribution(
        build_model_fn=lambda trial: build_model_fn(trial, *Other parameters if needed),
        bits_per_param=4,
        n_trials=1000
    )
"""

from typing import Callable
import optuna
import matplotlib.pyplot as plt
import tensorflow as tf

from araras.plot.configs import config_plt

config_plt("double-column")  # Configure matplotlib for double-column figures

def model_param_distribution(
    build_model_fn: Callable[[optuna.Trial], tf.keras.Model],
    bits_per_param: int,
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

    bar_len = 30
    for i in range(n_trials):
        trial = study.ask()
        model = build_model_fn(trial)
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