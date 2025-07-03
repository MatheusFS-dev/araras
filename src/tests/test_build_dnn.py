from araras.keras.builders.dnn import build_dnn
from araras.keras.hparams import HParams
from tests.helpers import DummyTrial
import tensorflow as tf


def main():
    print("Testing build_dnn from araras.keras.builders.dnn")
    trial = DummyTrial()
    hparams = HParams([
        "relu"], ["none"], ["adam"], ["StandardScaler"])
    x = tf.keras.Input(shape=(4,))
    try:
        res = build_dnn(trial, hparams, x, 8, 0.1)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
