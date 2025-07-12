"""
Module model_stats of keras

Functions:
    - get_model_stats: Extract and return model statistics from the given Optuna trial.

Example:
    >>> from araras.optuna.keras.model_stats import get_model_stats
    >>> get_model_stats(...)
"""
from araras.commons import *
import tensorflow as tf
import optuna
from araras.keras.analysis.profiler import get_flops, get_macs, get_memory_and_time
from araras.keras.utils import capture_model_summary
from araras.tensorflow.model import get_model_usage_stats


def get_model_stats(
    trial: optuna.Trial,
    model: tf.keras.Model,
    bits_per_param: int,
    batch_size: int,
    n_trials: int = 10000,
    device: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Extract and return model statistics from the given Optuna trial.

    Args:
        trial (optuna.Trial): The Optuna trial object
        model (tf.keras.Model): The Keras model to analyze.
        policy (tf.keras.DTypePolicy): The precision policy used for the model.
        batch_size (int): The batch size to simulate for input.
        n_trials (int): Number of trials for power and energy measurement.
        device (int): GPU index to run the model on. Use ``-1`` for CPU.
        verbose (bool): If True, print detailed information.

    Returns:
        Dict[str, float]: A dictionary containing model statistics
    """
    params = model.count_params()
    peak_mem_usage, inference_time = get_memory_and_time(
        model, batch_size=batch_size, device=device, verbose=verbose
    )
    _, avg_power, avg_energy = get_model_usage_stats(
        model, device=device, n_trials=n_trials, verbose=verbose
    )

    trial.set_user_attr("num_params", params)
    trial.set_user_attr("model_size", params * bits_per_param)
    trial.set_user_attr("flops", get_flops(model))
    trial.set_user_attr("macs", get_macs(model))
    trial.set_user_attr("model_summary", capture_model_summary(model))
    trial.set_user_attr("peak_memory_usage", peak_mem_usage)
    trial.set_user_attr("inference_time", inference_time)
    trial.set_user_attr("avg_power", avg_power)
    trial.set_user_attr("avg_energy", avg_energy)

    return {
        "num_params": params,
        "model_size": params * bits_per_param,
        "flops": get_flops(model),
        "macs": get_macs(model),
        "model_summary": capture_model_summary(model),
        "peak_memory_usage": peak_mem_usage,
        "inference_time": inference_time,
        "avg_power": avg_power,
        "avg_energy": avg_energy,
    }
