"""Validate GPU memory tracking with a tiny Keras workload."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("optuna", types.ModuleType("optuna"))


def _resolve_gpu_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",")[0].strip()
        if first:
            try:
                return int(first)
            except ValueError:
                pass
    return 0


def _build_toy_model(tf):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(32,), name="features")
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, name="prediction")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def _train_toy_model(tf, model) -> None:
    features = tf.random.uniform((128, 32), dtype=tf.float32)
    targets = tf.random.uniform((128, 1), dtype=tf.float32)
    model.fit(features, targets, epochs=1, batch_size=32, verbose=0)


def _bytes_to_mib(value: float) -> float:
    return float(value) / (1024 * 1024)


def test_measure_gpu_memory_during_inference() -> None:
    tf = pytest.importorskip("tensorflow")
    from araras.utils.system import measure_callable_resource_usage

    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        pytest.skip("TensorFlow GPU support not available")

    for device in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            continue

    np.random.seed(42)
    tf.random.set_seed(42)

    model = _build_toy_model(tf)
    _train_toy_model(tf, model)

    inference_batch = tf.random.uniform((64, 32), dtype=tf.float32)

    def run_inference() -> float:
        predictions = model.predict(inference_batch, batch_size=32, verbose=0)
        return float(np.mean(predictions))

    summary, _ = measure_callable_resource_usage(
        run_inference,
        metrics=("gpu_ram",),
        target_gpu_index=_resolve_gpu_index(),
        repeat=1000,
    )
    
    print(summary)

    gpu_metrics = summary.get("gpu_ram")
    if gpu_metrics == "Not measured":
        pytest.skip("GPU RAM metrics unavailable (nvidia-smi missing?)")

    assert isinstance(gpu_metrics, dict)
    for phase in ("before", "during", "delta"):
        assert phase in gpu_metrics
        phase_stats = gpu_metrics[phase]
        assert isinstance(phase_stats, dict)
        for stat_key in ("measurements", "min", "max", "avg", "std", "var"):
            assert stat_key in phase_stats
        measurements = phase_stats["measurements"]
        assert isinstance(measurements, list)
        if measurements:
            assert all(isinstance(item, (int, float)) for item in measurements)

    delta_max = gpu_metrics["delta"]["max"]
    if delta_max is not None:
        assert isinstance(delta_max, (int, float))
        assert delta_max >= 0

    gpu_metrics_mib = {
        phase: (
            round(_bytes_to_mib(stats["max"]), 3)
            if isinstance(stats, dict) and stats.get("max") is not None
            else None
        )
        for phase, stats in gpu_metrics.items()
    }
    print(f"GPU RAM metrics max (MiB): {gpu_metrics_mib}")


if __name__ == "__main__":
    try:
        test_measure_gpu_memory_during_inference()
        print("Measurement test completed successfully.")
    except pytest.SkipTest as exc:
        print(f"Skipped: {exc}")
