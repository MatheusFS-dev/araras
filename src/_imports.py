"""
This module imports all the necessary libraries and modules for the project.

Example usage:
    from _imports import *

"""

# ———————————————————————————— Standard Libraries ———————————————————————————— #
import gc, math, signal, shutil, traceback, subprocess
from IPython.display import clear_output, display, HTML
from tkinter import Image

# ———————————————————————————————— Annotations ——————————————————————————————— #
from typing import *

# ————————————————————————— Data Processing Libraries ———————————————————————— #
import numpy as np
import fireducks.pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt

# ——————————————————————— TensorFlow and Keras Modules ——————————————————————— #
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras import (
    layers,
    Model,
    callbacks,
    optimizers,
    regularizers,
    metrics,
    losses,
    mixed_precision,
)
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# Set TensorFlow logger level to ERROR
tf.get_logger().setLevel('ERROR')

# ——————————————————————————— Scikit-learn Modules ——————————————————————————— #
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# —————————————————————————————————— Optuna —————————————————————————————————— #
import optuna
from optuna.integration import KerasPruningCallback
# import optunahub  # For the AutoSampler

# —————————————————————————————————— Araras —————————————————————————————————— #
import araras as aa
from araras.callbacks.nan_loss_pruner import NanLossPrunerCallback
from araras.email.utils import send_email, notify_training_success
from araras.utils.gpu import get_gpu_info