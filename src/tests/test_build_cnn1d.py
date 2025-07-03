from araras.keras.builders.cnn import build_cnn1d
from araras.keras.hparams import HParams
from tests.helpers import DummyTrial
import tensorflow as tf


def main():
    print("Testing build_cnn1d from araras.keras.builders.cnn")
    trial = DummyTrial()
    hparams = HParams(["relu"], ["none"], ["adam"], ["StandardScaler"])
    x = tf.keras.Input(shape=(10, 1))
    try:
        res = build_cnn1d(trial, hparams, x, 8, 3)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
