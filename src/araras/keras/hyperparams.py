"""
Hyperparameter utilities for Keras models.

Classes:
    - KParams: Dataclass with methods to sample activation functions,
      regularizers, optimizers, and scalers.

Example using only default parameters:
    >>> from araras.keras.kparams import KParams
    >>> kparams = KParams.default()
    
Example using custom parameters:
    >>> from araras.keras.kparams import KParams
    >>> kparams = KParams(
    ...     activation_choices=[tf.keras.activations.relu, tf.keras.activations.tanh],
    ...     regularizer_choices=[tf.keras.regularizers.L2(1e-3)],
    ...     optimizer_choices=[tf.keras.optimizers.Adam],
    ...     scaler_choices=[StandardScaler, MinMaxScaler],
    ...     initializer_choices=[tf.keras.initializers.GlorotUniform],
    ... )
    
"""
from araras.commons import *

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import optuna
import tensorflow as tf
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)


def _sample_choice(trial: optuna.Trial, name: str, choices: Sequence[Any]) -> Any:
    """Sample a value from ``choices`` using Optuna."""
    if len(choices) == 1:
        return choices[0]
    return trial.suggest_categorical(name, list(choices))


class BaseSampler(ABC):
    """Abstract sampler for Keras hyperparameters."""

    def __init__(self, choices: Sequence[Any], name: str) -> None:
        self.choices = choices
        self.name = name

    def sample(self, trial: optuna.Trial) -> Any:
        try:
            choice = _sample_choice(trial, self.name, self.choices)
            return self._process(choice, trial)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Error sampling {self.name}") from exc

    @abstractmethod
    def _process(self, choice: Any, trial: optuna.Trial) -> Any:
        """Transform the chosen value into the final object."""
        raise NotImplementedError


class ActivationSampler(BaseSampler):
    def _process(self, choice: Any, trial: optuna.Trial) -> Any:  # noqa: D401
        return choice


class RegularizerSampler(BaseSampler):
    def _process(self, choice: Any, trial: optuna.Trial) -> Any:
        if choice is None:
            return None
        if isinstance(choice, tf.keras.regularizers.Regularizer):
            return choice
        if isinstance(choice, type) and issubclass(choice, tf.keras.regularizers.Regularizer):
            return choice()
        if callable(choice):
            obj = choice()
            if isinstance(obj, tf.keras.regularizers.Regularizer):
                return obj
        raise ValueError(f"Unsupported regularizer: {choice}")


class OptimizerSampler(BaseSampler):
    def __init__(self, choices: Sequence[Any], name: str, lr_range: tuple[float, float] = (1e-5, 1e-2)) -> None:
        super().__init__(choices, name)
        self.lr_range = lr_range

    def _process(self, choice: Any, trial: optuna.Trial) -> tf.keras.optimizers.Optimizer:
        lr = trial.suggest_float(f"{self.name}_lr", *self.lr_range, log=True)
        if isinstance(choice, tf.keras.optimizers.Optimizer):
            choice.learning_rate = lr
            return choice
        if isinstance(choice, type) and issubclass(choice, tf.keras.optimizers.Optimizer):
            return choice(learning_rate=lr)
        if callable(choice):
            obj = choice()
            if isinstance(obj, tf.keras.optimizers.Optimizer):
                obj.learning_rate = lr
                return obj
        raise ValueError(f"Unsupported optimizer: {choice}")


class ScalerSampler(BaseSampler):
    def _process(self, choice: Any, trial: optuna.Trial) -> Any:
        if isinstance(choice, (StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer)):
            return choice
        if isinstance(choice, type):
            return choice()
        if callable(choice):
            return choice()
        raise ValueError(f"Unsupported scaler: {choice}")


class InitializerSampler(BaseSampler):
    def _process(self, choice: Any, trial: optuna.Trial) -> tf.keras.initializers.Initializer:
        if isinstance(choice, tf.keras.initializers.Initializer):
            return choice
        if isinstance(choice, type) and issubclass(choice, tf.keras.initializers.Initializer):
            return choice()
        if callable(choice):
            obj = choice()
            if isinstance(obj, tf.keras.initializers.Initializer):
                return obj
        raise ValueError(f"Unsupported initializer: {choice}")


@dataclass
class KParams:
    """Container for hyperparameter search spaces."""

    activation_choices: List[Optional[Callable[..., Any]]] = field(
        default_factory=lambda: [
            tf.keras.activations.relu,
            tf.keras.activations.tanh,
            tf.keras.activations.sigmoid,
            tf.keras.activations.linear,
            None,
        ]
    )

    regularizer_choices: List[Optional[Union[type[tf.keras.regularizers.Regularizer], tf.keras.regularizers.Regularizer, Callable[[], tf.keras.regularizers.Regularizer]]]] = field(
        default_factory=lambda: [
            None,
            tf.keras.regularizers.L1(1e-2),
            tf.keras.regularizers.L2(1e-2),
            tf.keras.regularizers.L1L2(l1=1e-2, l2=1e-2),
        ]
    )

    optimizer_choices: List[Union[type[tf.keras.optimizers.Optimizer], tf.keras.optimizers.Optimizer, Callable[[], tf.keras.optimizers.Optimizer]]] = field(
        default_factory=lambda: [
            tf.keras.optimizers.Adam,
            tf.keras.optimizers.RMSprop,
            tf.keras.optimizers.SGD,
            tf.keras.optimizers.AdamW,
        ]
    )

    scaler_choices: List[Union[type, Callable[[], Any], Any]] = field(
        default_factory=lambda: [
            StandardScaler,
            lambda: MinMaxScaler(feature_range=(0, 1)),
            lambda: MinMaxScaler(feature_range=(-1, 1)),
            RobustScaler,
            QuantileTransformer,
            PowerTransformer,
        ]
    )

    initializer_choices: List[Union[type[tf.keras.initializers.Initializer], tf.keras.initializers.Initializer, Callable[[], tf.keras.initializers.Initializer]]] = field(
        default_factory=lambda: [
            tf.keras.initializers.GlorotUniform,
            tf.keras.initializers.GlorotNormal,
            tf.keras.initializers.HeNormal(),
            tf.keras.initializers.HeUniform(),
        ]
    )

    def get_activation(self, trial: optuna.Trial, name: str) -> Optional[Callable[..., Any]]:
        """Sample or return an activation."""

        sampler = ActivationSampler(self.activation_choices, name)
        return sampler.sample(trial)

    def get_regularizer(
        self, trial: optuna.Trial, name: str
    ) -> Optional[tf.keras.regularizers.Regularizer]:
        """Sample a regularizer."""

        sampler = RegularizerSampler(self.regularizer_choices, name)
        return sampler.sample(trial)

    def get_optimizer(self, trial: optuna.Trial) -> tf.keras.optimizers.Optimizer:
        """Sample an optimizer."""

        sampler = OptimizerSampler(self.optimizer_choices, "optimizer")
        return sampler.sample(trial)

    def get_scaler(self, trial: optuna.Trial):
        """Sample a scikit-learn scaler."""

        sampler = ScalerSampler(self.scaler_choices, "scaler")
        return sampler.sample(trial)

    def get_initializer(
        self, trial: optuna.Trial, name: str
    ) -> tf.keras.initializers.Initializer:
        """Sample a kernel initializer."""

        sampler = InitializerSampler(self.initializer_choices, name)
        return sampler.sample(trial)

    @classmethod
    def get_default_params(cls) -> "KParams":
        """Return an instance with all default options."""

        return cls()

    @classmethod
    def default(cls) -> "KParams":
        """Alias for :meth:`get_default_params`."""

        return cls.get_default_params()
