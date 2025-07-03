from araras.keras.builders.cnn import build_dense_as_conv3d
from araras.keras.hparams import HParams
from tests.helpers import DummyTrial
import tensorflow as tf


def main():
    print("Testing build_dense_as_conv3d from araras.keras.builders.cnn")
    trial = DummyTrial()
    hparams = HParams(["relu"], ["none"], ["adam"], ["StandardScaler"])
    x = tf.keras.Input(shape=(1, 1, 1, 8))
    try:
        res = build_dense_as_conv3d(trial, hparams, x, 8)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
