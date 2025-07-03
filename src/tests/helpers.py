import random
from types import SimpleNamespace

class DummyTrial:
    def suggest_int(self, name, low, high, step=1):
        print(f"suggest_int {name}")
        return low
    def suggest_float(self, name, low, high, step=None):
        print(f"suggest_float {name}")
        return low
    def suggest_categorical(self, name, choices):
        print(f"suggest_categorical {name}")
        return choices[0]

class DummyStudy:
    def __init__(self):
        self.trials = []
    def get_trials(self, deepcopy=False, states=None):
        return self.trials

# simple tensorflow model creator
def make_model():
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='sgd', loss='mse')
    return model
