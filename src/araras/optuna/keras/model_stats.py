import tensorflow as tf
import optuna
from typing import Dict
from araras.keras.utils.profiler import get_flops, get_macs, get_memory_and_time
from araras.keras.utils.summary import capture_model_summary
from araras.tensorflow.utils.model import get_model_usage_stats


def get_model_stats(
    trial: optuna.Trial,
    model: tf.keras.Model,
    bits_per_param: int,
    batch_size: int,
    n_trials: int = 10000,
) -> Dict[str, float]:
    """
    Extract and return model statistics from the given Optuna trial.

    Args:
        trial (optuna.Trial): The Optuna trial object
        model (tf.keras.Model): The Keras model to analyze.
        policy (tf.keras.DTypePolicy): The precision policy used for the model.
        batch_size (int): The batch size to simulate for input.

    Returns:
        Dict[str, float]: A dictionary containing model statistics
    """
    params = model.count_params()
    peak_mem_usage, inference_time = get_memory_and_time(model, batch_size=batch_size, device="GPU:0")
    _, avg_power, avg_energy = get_model_usage_stats(model, device="gpu", n_trials=n_trials)

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
