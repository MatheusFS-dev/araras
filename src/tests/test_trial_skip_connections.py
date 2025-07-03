from araras.keras.skip_connections import trial_skip_connections
from tests.helpers import DummyTrial
import tensorflow as tf


def main():
    print("Testing trial_skip_connections from araras.keras.skip_connections")
    trial = DummyTrial()
    layers_list = [tf.keras.Input(shape=(4,)), tf.keras.layers.Dense(4)(tf.keras.Input(shape=(4,)))]
    try:
        res = trial_skip_connections(trial, layers_list)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
