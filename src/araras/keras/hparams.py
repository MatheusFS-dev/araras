"""
Hyperparameter utilities for Keras models.

Classes:
    - HParams: Dataclass with methods to sample activation functions,
      regularizers, optimizers, and scalers.

Example:
    >>> from araras.keras.hparams import HParams
    >>> hp = HParams()
    >>> hp.get_optimizer(optuna.trial.FixedTrial({}))
"""
from araras.commons import *

from dataclasses import dataclass
import optuna
import tensorflow as tf
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)


@dataclass
class HParams:
    """
    A container class for hyperparameter choices and their corresponding Optuna sampling logic.

    This class encapsulates lists of candidate values for several hyperparameters
    (activation functions, regularizers, optimizers, and scalers) and provides
    methods to sample from these lists using Optuna trials.

    Logic flow for each sampler:
        Trial input -> Suggest categorical/float -> Map to corresponding object/value -> Return

    Attributes:
        activation_choices (List[str]): List of activation function names.
        regularizer_choices (List[str]): List of regularizer choices.
        optimizer_choices (List[str]): List of optimizer names.
        scaler_choices (List[str]): List of scaler class names.
    """

    activation_choices: List[str]
    regularizer_choices: List[str]
    optimizer_choices: List[str]
    scaler_choices: List[str]

    l1_value: float = 1e-2
    l2_value: float = 1e-2
    orthogonal_factor: float = 0.01
    orthogonal_mode: str = "rows"

    min_lr: float = 1e-5
    max_lr: float = 1e-2

    lr_value: float = None # Can be set for fixed learning rate

    def get_activation(self, trial: optuna.Trial, name: str) -> Optional[str]:
        """
        Samples an activation function from the predefined list, including support for None.

        Args:
            trial (optuna.Trial): The Optuna trial object for sampling.
            name (str): The unique name for the activation parameter.

        Returns:
            Optional[str]: The selected activation function or None.
        """

        # If length of is 1, return the the only option (So, no trial)
        if len(self.activation_choices) == 1:
            return self.activation_choices[0]

        # Selects an activation function from the candidate list using Optuna's categorical sampler
        choice = trial.suggest_categorical(name, self.activation_choices)

        # Convert choice to lowercase and return None if the choice is "none"
        if choice.lower() == "none":
            return None

        return choice

    def get_regularizer(
        self,
        trial: optuna.Trial,
        name: str,
    ) -> Optional[tf.keras.regularizers.Regularizer]:
        """
        Samples and maps a string regularizer to a TensorFlow regularizer object.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            name (str): Unique identifier for the regularizer parameter.

        Returns:
            Optional[tf.keras.regularizers.Regularizer]: A TensorFlow regularizer or None.

        Raises:
            ValueError: If the sampled regularizer name is unknown.
        """

        # If length of is 1, return the the only option (So, no trial)
        if len(self.regularizer_choices) == 1:
            choice = self.regularizer_choices[0]
        else:
            # Suggests a regularizer choice from the candidate list
            choice = trial.suggest_categorical(name, self.regularizer_choices)

        # Maps choice string to corresponding TensorFlow regularizer object
        if choice == "none":
            return None
        elif choice == "l1":
            return tf.keras.regularizers.L1(l1_value=self.l1_value)
        elif choice == "l2":
            return tf.keras.regularizers.L2(l2_value=self.l2_value)
        elif choice == "l1l2":
            return tf.keras.regularizers.L1L2(l1_value=self.l1_value, l2_value=self.l2_value)
        elif choice == "orthogonal":
            #! Only works for rank-2 tensors
            return tf.keras.regularizers.OrthogonalRegularizer(factor=self.orthogonal_factor, mode=self.orthogonal_mode)

        # Raise error for unknown regularizer option
        raise ValueError(f"Unknown regularizer {choice}")

    def get_optimizer(
        self,
        trial: optuna.Trial,
    ) -> tf.keras.optimizers.Optimizer:
        """
        Samples an optimizer type and learning rate, returning a configured optimizer instance.

        Args:
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            tf.keras.optimizers.Optimizer: A configured TensorFlow optimizer instance.
        """
        # If length of is 1, return the the only option (So, no trial)
        if len(self.optimizer_choices) == 1:
            optim = self.optimizer_choices[0]
        else:
            # Suggests an optimizer type from the list
            optim = trial.suggest_categorical("optimizer", self.optimizer_choices)

        # If a fixed learning rate is set, use it; otherwise, sample a log-scaled learning rate
        if self.lr_value is not None:
            lr = self.lr_value
        else:
            lr = trial.suggest_float("lr", self.min_lr, self.max_lr, log=True)

        # Maps optimizer name to the corresponding TensorFlow class
        mapping = {
            "SGD": tf.keras.optimizers.SGD,
            "RMSprop": tf.keras.optimizers.RMSprop,
            "Adam": tf.keras.optimizers.Adam,
            "AdamW": tf.keras.optimizers.AdamW,
            "Adadelta": tf.keras.optimizers.Adadelta,
            "Adagrad": tf.keras.optimizers.Adagrad,
            "Adamax": tf.keras.optimizers.Adamax,
            "Adafactor": tf.keras.optimizers.Adafactor,
            "Nadam": tf.keras.optimizers.Nadam,
            "Ftrl": tf.keras.optimizers.Ftrl,
            "Lion": tf.keras.optimizers.Lion,
            "Lamb": tf.keras.optimizers.Lamb,
            # "LossScaleOptimizer": tf.keras.mixed_precision.LossScaleOptimizer,
        }

        # Returns the selected optimizer initialized with the sampled learning rate
        return mapping[optim](learning_rate=lr)

    def get_scaler(self, trial: optuna.Trial):
        """
        Samples and returns a configured scikit-learn scaler object.

        Args:
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            A scikit-learn scaler object.

        Raises:
            ValueError: If the selected scaler name is not recognized.
        """
        # If length of is 1, return the the only option (So, no trial)
        if len(self.scaler_choices) == 1:
            choice = self.scaler_choices[0]
        else:
            # Suggests a scaler choice from the list
            choice = trial.suggest_categorical("scaler", self.scaler_choices)

        # Maps each string option to its corresponding scaler object with parameters
        if choice == "StandardScaler":
            return StandardScaler()
        if choice == "MinMaxScaler_0_1":
            return MinMaxScaler(feature_range=(0, 1))
        if choice == "MinMaxScaler_-1_1":
            return MinMaxScaler(feature_range=(-1, 1))
        if choice == "RobustScaler":
            return RobustScaler()
        if choice == "QuantileTransformer":
            return QuantileTransformer(output_distribution="normal")
        if choice == "PowerTransformer":
            return PowerTransformer(method="yeo-johnson")

        # Raises error if unknown scaler is selected
        raise ValueError(choice)
