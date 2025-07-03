from araras.keras.builders.tcnn import build_tcnn1d
from araras.keras.hparams import HParams
from tests.helpers import DummyTrial
import tensorflow as tf


def main():
    print("Testing build_tcnn1d from araras.keras.builders.tcnn")
    trial = DummyTrial()
    hparams = HParams(["relu"], ["none"], ["adam"], ["StandardScaler"])
    x = tf.keras.Input(shape=(10, 1))
    try:
        res = build_tcnn1d(trial, hparams, x, 8, 3)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
