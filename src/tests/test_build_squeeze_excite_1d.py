from araras.keras.builders.se import build_squeeze_excite_1d
from tests.helpers import DummyTrial
import tensorflow as tf


def main():
    print("Testing build_squeeze_excite_1d from araras.keras.builders.se")
    trial = DummyTrial()
    x = tf.keras.Input(shape=(10, 8))
    try:
        res = build_squeeze_excite_1d(trial, x, 8, 8)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
