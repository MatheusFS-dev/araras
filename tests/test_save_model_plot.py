import sys
from pathlib import Path

import pytest

# Ensure package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

tf = pytest.importorskip("tensorflow")
hl = pytest.importorskip("hiddenlayer")

from araras.ml.model.tools import save_model_plot


def test_save_model_plot_from_instance(tmp_path):
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(3,)), tf.keras.layers.Dense(1)])
    output = tmp_path / "plot.png"
    save_model_plot(model, output)
    assert output.exists()


def test_save_model_plot_from_path(tmp_path):
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(3,)), tf.keras.layers.Dense(1)])
    model_file = tmp_path / "model.keras"
    model.save(model_file)
    output = tmp_path / "plot.png"
    save_model_plot(model_file, output)
    assert output.exists()
