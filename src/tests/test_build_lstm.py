from araras.keras.builders.lstm import build_lstm
from araras.keras.hparams import HParams
from tests.helpers import DummyTrial
import tensorflow as tf


def main():
    print("Testing build_lstm from araras.keras.builders.lstm")
    trial = DummyTrial()
    hparams = HParams(["tanh"], ["none"], ["adam"], ["StandardScaler"])
    x = tf.keras.Input(shape=(5, 8))
    try:
        res = build_lstm(trial, hparams, x, 8)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
