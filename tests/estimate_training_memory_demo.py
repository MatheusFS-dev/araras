from __future__ import annotations

"""Run dummy trainings with multiple optimizers and compare GPU usage.

The script samples GPU metrics in parallel while ``model.fit`` executes by
leveraging :class:`ResourceMonitor`. For each optimizer configuration we show
the baseline allocation, observed peak, delta, and analytical memory
estimates.

Execute with ``python tests/estimate_training_memory_demo.py``.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:  # pragma: no cover - optional dependency
    import tensorflow as tf
    from tensorflow import keras
except Exception as exc:  # pragma: no cover - graceful message
    print("TensorFlow (with GPU support) is required to run this demo.")
    print(f"Import error: {exc}")
    sys.exit(0)

try:  # pragma: no cover - optional dependency
    from tensorflow.keras import mixed_precision
except Exception:  # pragma: no cover - fallback for TF versions without module
    mixed_precision = None

from araras.ml.optuna.model_tools import estimate_training_memory
from araras.utils.system import ResourceMonitor


@dataclass
class TrainingConfig:
    name: str
    optimizer: keras.optimizers.Optimizer
    batch_size: int
    mixed_policy: str | None = None  # e.g., "mixed_float16"
    epochs: int = 1
    steps_per_epoch: int = 30


def _ensure_gpu() -> bool:
    devices = tf.config.experimental.list_physical_devices("GPU")
    if not devices:
        print("No GPU detected. Skipping demo.")
        return False
    # Enable memory growth so small demos do not pre-allocate all VRAM.
    tf.config.experimental.set_memory_growth(devices[0], True)
    return True

def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} {units[-1]}"


def _iter_output_shapes(layer: keras.layers.Layer) -> Iterable[Sequence[int | None]]:
    raw_shape = getattr(layer, "output_shape", None)
    if raw_shape is None and hasattr(layer, "output"):
        raw_shape = tf.keras.backend.int_shape(layer.output)

    if raw_shape is None:
        return []

    if isinstance(raw_shape, (tuple, list)) and raw_shape and all(
        isinstance(dim, (int, type(None))) for dim in raw_shape
    ):
        return [raw_shape]

    if isinstance(raw_shape, (tuple, list)):
        shapes = []
        for candidate in raw_shape:
            if candidate is None:
                continue
            if isinstance(candidate, (tuple, list)) and all(
                isinstance(dim, (int, type(None))) for dim in candidate
            ):
                shapes.append(candidate)
        return shapes

    return [tf.TensorShape(raw_shape).as_list()]


def _manual_activation_elements(model: keras.Model, batch_size: int) -> int:
    total = 0
    for layer in model.layers:
        for shape in _iter_output_shapes(layer):
            if not shape:
                continue
            dims: list[int] = []
            for index, dim in enumerate(shape):
                if index == 0:
                    resolved = batch_size if dim is None else int(dim)
                else:
                    resolved = 1 if dim is None else int(dim)
                dims.append(resolved)
            if dims:
                total += int(np.prod(dims, dtype=np.int64))
    return total


def _slot_factor(optimizer: keras.optimizers.Optimizer) -> int:
    name = optimizer.__class__.__name__.lower()
    if "adam" in name or "nadam" in name or "adamax" in name:
        return 2
    if "adafactor" in name:
        return 2
    if "rmsprop" in name:
        return 2 if getattr(optimizer, "momentum", 0) > 0 else 1
    if "adagrad" in name:
        return 1
    if "adadelta" in name:
        return 2
    if "ftrl" in name:
        return 2
    if "sgd" in name:
        return 1 if getattr(optimizer, "momentum", 0) > 0 else 0
    return 2


def _manual_memory_estimate(model: keras.Model, batch_size: int) -> int:
    policy = getattr(model, "dtype_policy", None)
    variable_dtype = getattr(policy, "variable_dtype", tf.float32)
    compute_dtype = getattr(policy, "compute_dtype", variable_dtype)

    variable_bytes = tf.as_dtype(variable_dtype).size or 4
    compute_bytes = tf.as_dtype(compute_dtype).size or variable_bytes
    gradient_bytes = max(variable_bytes, compute_bytes)

    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    weight_bytes = trainable_params * variable_bytes
    non_trainable_bytes = non_trainable_params * variable_bytes
    gradient_param_bytes = trainable_params * gradient_bytes
    slot_bytes = trainable_params * variable_bytes * _slot_factor(model.optimizer)

    activation_elements = _manual_activation_elements(model, batch_size)
    activation_forward = activation_elements * compute_bytes
    activation_backward = activation_elements * gradient_bytes

    base_manual = (
        weight_bytes
        + non_trainable_bytes
        + gradient_param_bytes
        + slot_bytes
        + activation_forward
        + activation_backward
    )

    # Use the same overhead heuristic as the estimator for apples-to-apples comparison.
    from araras.ml.optuna.model_tools import _get_framework_overhead  # type: ignore

    return base_manual + _get_framework_overhead(base_manual)


def _build_model() -> keras.Model:
    inputs = keras.Input(shape=(128,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def _dummy_dataset(batch_size: int, steps: int) -> tf.data.Dataset:
    total_samples = batch_size * steps
    x = tf.random.uniform((total_samples, 128), dtype=tf.float32)
    y = tf.one_hot(tf.random.uniform((total_samples,), maxval=10, dtype=tf.int32), depth=10)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def _run_config(cfg: TrainingConfig, default_policy: str) -> None:
    if mixed_precision is not None:
        policy_name = cfg.mixed_policy or default_policy
        mixed_precision.set_global_policy(policy_name)

    tf.keras.backend.clear_session()
    model = _build_model()
    model.compile(optimizer=cfg.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    dataset = _dummy_dataset(cfg.batch_size, cfg.steps_per_epoch)

    monitor = ResourceMonitor(
        metrics=("gpu_ram", "gpu_usage"),
        target_gpu_index=0,
        sample_interval=0.05,
    )

    def _train() -> None:
        model.fit(dataset, epochs=cfg.epochs, steps_per_epoch=cfg.steps_per_epoch, verbose=0)

    metrics_summary, _ = monitor.measure_callable(_train)

    analytical_estimate = estimate_training_memory(model, batch_size=cfg.batch_size)
    manual_estimate = _manual_memory_estimate(model, cfg.batch_size)

    print("\n===", cfg.name, "===")
    print(f"Optimizer: {cfg.optimizer.__class__.__name__}")
    print(f"Batch size: {cfg.batch_size}")
    if cfg.mixed_policy:
        print(f"Mixed precision policy: {cfg.mixed_policy}")
    gpu_ram_metrics = metrics_summary.get("gpu_ram", "Not measured")
    if isinstance(gpu_ram_metrics, dict):
        before = int(round(gpu_ram_metrics.get("before", 0)))
        peak = int(round(gpu_ram_metrics.get("current", 0)))
        diff = int(round(gpu_ram_metrics.get("difference", 0)))
        final = int(round(gpu_ram_metrics.get("final", peak)))
        print("GPU RAM (bytes):")
        print(f"  before : {_format_bytes(before)}")
        print(f"  peak   : {_format_bytes(peak)}")
        print(f"  delta  : {_format_bytes(diff)}")
        print(f"  final  : {_format_bytes(final)}")
    else:
        print(f"GPU RAM metrics: {gpu_ram_metrics}")

    gpu_usage_metrics = metrics_summary.get("gpu_usage", "Not measured")
    if isinstance(gpu_usage_metrics, dict):
        before = gpu_usage_metrics.get("before")
        peak = gpu_usage_metrics.get("current")
        diff = gpu_usage_metrics.get("difference")
        final = gpu_usage_metrics.get("final", peak)

        def _fmt_percent(value: object) -> str:
            if isinstance(value, (int, float)):
                return f"{value:.2f}%"
            return str(value)

        print("GPU util (%):")
        print(f"  before : {_fmt_percent(before)}")
        print(f"  peak   : {_fmt_percent(peak)}")
        print(f"  delta  : {_fmt_percent(diff)}")
        print(f"  final  : {_fmt_percent(final)}")
    else:
        print(f"GPU util metrics: {gpu_usage_metrics}")
    print(f"Estimator output: {_format_bytes(analytical_estimate)}")
    print(f"Manual formula:   {_format_bytes(manual_estimate)}")


def main() -> None:
    if not _ensure_gpu():
        return

    default_policy = "float32"
    if mixed_precision is not None:
        mixed_precision.set_global_policy(default_policy)

    configs = [
        TrainingConfig(
            name="Adam bs=32",
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            batch_size=32,
        ),
        TrainingConfig(
            name="SGD momentum bs=64",
            optimizer=keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9),
            batch_size=64,
        ),
        TrainingConfig(
            name="RMSprop mixed_float16 bs=32",
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3, momentum=0.0),
            batch_size=32,
            mixed_policy="mixed_float16" if mixed_precision is not None else None,
        ),
    ]

    for cfg in configs:
        _run_config(cfg, default_policy)

    if mixed_precision is not None:
        mixed_precision.set_global_policy(default_policy)


if __name__ == "__main__":
    main()
