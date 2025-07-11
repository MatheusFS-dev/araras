"""
Hyperparameter utilities for Keras models.

Classes:
    - KParams: Dataclass with methods to sample activation functions,
      regularizers, optimizers, and scalers.

Example:
    >>> from araras.keras.kparams import KParams
    >>> hp = KParams()
    >>> hp.get_optimizer(optuna.trial.FixedTrial({}))
"""
from araras.commons import *

from dataclasses import dataclass, field
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
class KParams:
    """Container for hyperparameter search spaces.

    The class stores lists of possible options for activations, regularizers,
    optimizers, scalers and initializers. Options can be specified as strings or
    as the actual objects themselves, offering maximum flexibility.
    """

    activation_choices: List[
        Union[str, Callable[..., Any], tf.keras.layers.Layer, None]
    ] = field(
        default_factory=lambda: [
            "relu",
            "tanh",
            "sigmoid",
            "linear",
            None,
        ]
    )
    regularizer_choices: List[
        Union[str, tf.keras.regularizers.Regularizer, None]
    ] = field(default_factory=lambda: ["none", "l1", "l2", "l1l2"])
    optimizer_choices: List[
        Union[str, type[tf.keras.optimizers.Optimizer], tf.keras.optimizers.Optimizer]
    ] = field(
        default_factory=lambda: [
            "Adam",
            "RMSprop",
            "SGD",
            "AdamW",
        ]
    )
    scaler_choices: List[Union[str, Any]] = field(
        default_factory=lambda: [
            "StandardScaler",
            "MinMaxScaler_0_1",
            "MinMaxScaler_-1_1",
            "RobustScaler",
            "QuantileTransformer",
            "PowerTransformer",
        ]
    )
    initializer_choices: List[
        Union[str, tf.keras.initializers.Initializer]
    ] = field(
        default_factory=lambda: [
            "glorot_uniform",
            "glorot_normal",
            tf.keras.initializers.HeNormal(),
            tf.keras.initializers.HeUniform(),
        ]
    )

    dropout_range: Tuple[float, float] = (0.0, 0.5)

    l1_value: float = 1e-2
    l2_value: float = 1e-2
    orthogonal_factor: float = 0.01
    orthogonal_mode: str = "rows"

    min_lr: float = 1e-5
    max_lr: float = 1e-2

    lr_value: float = None # Can be set for fixed learning rate

    def get_activation(
        self, trial: optuna.Trial, name: str
    ) -> Optional[Union[str, Callable[..., Any], tf.keras.layers.Layer]]:
        """Sample or return an activation.

        The returned object can be a string, callable, ``tf.keras.layers.Layer``
        instance or ``None``.
        """

        if len(self.activation_choices) == 1:
            choice = self.activation_choices[0]
        else:
            idx = trial.suggest_int(name, 0, len(self.activation_choices) - 1)
            choice = self.activation_choices[idx]

        if isinstance(choice, str):
            return None if choice.lower() == "none" else choice

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

        if len(self.regularizer_choices) == 1:
            choice = self.regularizer_choices[0]
        else:
            idx = trial.suggest_int(name, 0, len(self.regularizer_choices) - 1)
            choice = self.regularizer_choices[idx]

        if isinstance(choice, str):
            if choice == "none":
                return None
            if choice == "l1":
                return tf.keras.regularizers.L1(l1=self.l1_value)
            if choice == "l2":
                return tf.keras.regularizers.L2(l2=self.l2_value)
            if choice == "l1l2":
                return tf.keras.regularizers.L1L2(
                    l1=self.l1_value, l2=self.l2_value
                )
            if choice == "orthogonal":
                return tf.keras.regularizers.OrthogonalRegularizer(
                    factor=self.orthogonal_factor, mode=self.orthogonal_mode
                )
            raise ValueError(f"Unknown regularizer {choice}")

        return choice

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
        if len(self.optimizer_choices) == 1:
            optim = self.optimizer_choices[0]
        else:
            idx = trial.suggest_int("optimizer", 0, len(self.optimizer_choices) - 1)
            optim = self.optimizer_choices[idx]

        lr = self.lr_value if self.lr_value is not None else trial.suggest_float(
            "lr", self.min_lr, self.max_lr, log=True
        )

        if isinstance(optim, str):
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
            }

            return mapping[optim](learning_rate=lr)

        if isinstance(optim, type) and issubclass(optim, tf.keras.optimizers.Optimizer):
            return optim(learning_rate=lr)

        if isinstance(optim, tf.keras.optimizers.Optimizer):
            optim.learning_rate = lr
            return optim

        raise ValueError(f"Unknown optimizer {optim}")

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
        if len(self.scaler_choices) == 1:
            choice = self.scaler_choices[0]
        else:
            idx = trial.suggest_int("scaler", 0, len(self.scaler_choices) - 1)
            choice = self.scaler_choices[idx]

        if isinstance(choice, str):
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

            raise ValueError(choice)

        return choice

    def get_initializer(
        self, trial: optuna.Trial, name: str
    ) -> tf.keras.initializers.Initializer:
        """Sample or return a kernel initializer."""

        if len(self.initializer_choices) == 1:
            choice = self.initializer_choices[0]
        else:
            idx = trial.suggest_int(name, 0, len(self.initializer_choices) - 1)
            choice = self.initializer_choices[idx]

        if isinstance(choice, str):
            mapping = {
                "glorot_uniform": tf.keras.initializers.GlorotUniform,
                "glorot_normal": tf.keras.initializers.GlorotNormal,
                "he_uniform": tf.keras.initializers.HeUniform,
                "he_normal": tf.keras.initializers.HeNormal,
                "orthogonal": tf.keras.initializers.Orthogonal,
            }
            if choice not in mapping:
                raise ValueError(choice)
            return mapping[choice]()

        if isinstance(choice, type) and issubclass(
            choice, tf.keras.initializers.Initializer
        ):
            return choice()

        if isinstance(choice, tf.keras.initializers.Initializer):
            return choice

        raise ValueError(choice)

    @classmethod
    def default(cls) -> "KParams":
        """Return an ``KParams`` instance with all default options."""

        return cls()
