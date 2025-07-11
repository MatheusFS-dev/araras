"""
Hyperparameter utilities for Keras models.

Classes:
    - KParams: Dataclass with methods to sample activation functions,
      regularizers, optimizers, and scalers, and to set custom search spaces.

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
from typing import Optional, Sequence, Any, Callable, List, Union
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
    def __init__(
        self, choices: Sequence[Any], name: str, lr: Union[float, tuple[float, float]]
    ) -> None:
        super().__init__(choices, name)
        self.lr = lr

    def _process(self, choice: Any, trial: optuna.Trial) -> tf.keras.optimizers.Optimizer:
        if isinstance(self.lr, tuple):
            lr = trial.suggest_float(
                f"{self.name}_lr", self.lr[0], self.lr[1], log=True
            )
        else:
            lr = self.lr
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
        if isinstance(
            choice, (StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer)
        ):
            return choice
        if isinstance(choice, type):
            return choice()
        if callable(choice):
            return choice()
        raise ValueError(f"Unsupported scaler: {choice}")


class InitializerSampler(BaseSampler):
    def __init__(self, choices: Sequence[Any], name: str, seed: Optional[int] = None) -> None:
        super().__init__(choices, name)
        self.seed = seed

    def _call_with_seed(self, obj: Callable[..., Any]):
        if self.seed is None:
            return obj()
        try:
            return obj(seed=self.seed)
        except TypeError:
            return obj()

    def _process(self, choice: Any, trial: optuna.Trial) -> tf.keras.initializers.Initializer:
        if isinstance(choice, tf.keras.initializers.Initializer):
            if self.seed is not None:
                try:
                    choice.seed = self.seed
                except Exception:
                    pass
            return choice
        if isinstance(choice, type) and issubclass(choice, tf.keras.initializers.Initializer):
            return self._call_with_seed(choice)
        if callable(choice):
            obj = self._call_with_seed(choice)
            if isinstance(obj, tf.keras.initializers.Initializer):
                return obj
        raise ValueError(f"Unsupported initializer: {choice}")


@dataclass
class KParams:
    """Container for hyperparameter search spaces."""

    # ———————————————————————————————————————————————————————————————————————————— #
    #                                Default Values                                #
    # ———————————————————————————————————————————————————————————————————————————— #

    learning_rate: Union[float, tuple[float, float]] = 1e-2
    seed: Optional[int] = None

    activation_choices: List[Optional[Callable[..., Any]]] = field(
        default_factory=lambda: [
            tf.keras.activations.relu,
            tf.keras.activations.gelu,
            tf.keras.activations.silu,
            tf.keras.activations.elu,
            tf.keras.activations.sigmoid,
            tf.keras.activations.tanh,
            None,
        ]
    )

    regularizer_choices: List[
        Optional[
            Union[
                type[tf.keras.regularizers.Regularizer],
                tf.keras.regularizers.Regularizer,
                Callable[[], tf.keras.regularizers.Regularizer],
            ]
        ]
    ] = field(
        default_factory=lambda: [
            None,
            tf.keras.regularizers.L2(1e-2),
        ]
    )

    optimizer_choices: List[
        Union[
            type[tf.keras.optimizers.Optimizer],
            tf.keras.optimizers.Optimizer,
            Callable[[], tf.keras.optimizers.Optimizer],
        ]
    ] = field(
        default_factory=lambda: [
            tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            tf.keras.optimizers.Adam(learning_rate=0.001),
            tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
            tf.keras.optimizers.Lion(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
            tf.keras.optimizers.RMSprop(learning_rate=0.001),
        ]
    )

    scaler_choices: List[Union[type, Callable[[], Any], Any]] = field(
        default_factory=lambda: [
            StandardScaler,
            lambda: MinMaxScaler(feature_range=(0, 1)),
            lambda: MinMaxScaler(feature_range=(-1, 1)),
        ]
    )

    initializer_choices: List[
        Union[
            type[tf.keras.initializers.Initializer],
            tf.keras.initializers.Initializer,
            Callable[[], tf.keras.initializers.Initializer],
        ]
    ] = field(
        default_factory=lambda: [
            tf.keras.initializers.GlorotUniform,
        ]
    )

    def __post_init__(self) -> None:
        if isinstance(self.learning_rate, tuple):
            if (
                len(self.learning_rate) != 2
                or self.learning_rate[0] <= 0
                or self.learning_rate[1] <= 0
                or self.learning_rate[0] >= self.learning_rate[1]
            ):
                raise ValueError(
                    "learning_rate tuple must be (min, max) with 0 < min < max"
                )
        elif not isinstance(self.learning_rate, (int, float)):
            raise TypeError(
                "learning_rate must be a float or a tuple of two floats"
            )

    # ———————————————————————————————————————————————————————————————————————————— #

    def set_activation_choices(self, choices: Sequence[Optional[Callable[..., Any]]]) -> None:
        """Set the available activation choices."""

        self.activation_choices = list(choices)

    def set_regularizer_choices(
        self,
        choices: Sequence[
            Optional[
                Union[
                    type[tf.keras.regularizers.Regularizer],
                    tf.keras.regularizers.Regularizer,
                    Callable[[], tf.keras.regularizers.Regularizer],
                ]
            ]
        ],
    ) -> None:
        """Set the available regularizer choices."""

        self.regularizer_choices = list(choices)

    def set_optimizer_choices(
        self,
        choices: Sequence[
            Union[
                type[tf.keras.optimizers.Optimizer],
                tf.keras.optimizers.Optimizer,
                Callable[[], tf.keras.optimizers.Optimizer],
            ]
        ],
    ) -> None:
        """Set the available optimizer choices."""

        self.optimizer_choices = list(choices)

    def set_scaler_choices(self, choices: Sequence[Any]) -> None:
        """Set the available scaler choices."""

        self.scaler_choices = list(choices)

    def set_initializer_choices(
        self,
        choices: Sequence[
            Union[
                type[tf.keras.initializers.Initializer],
                tf.keras.initializers.Initializer,
                Callable[[], tf.keras.initializers.Initializer],
            ]
        ],
    ) -> None:
        """Set the available initializer choices."""

        self.initializer_choices = list(choices)

    def get_activation(self, trial: optuna.Trial, name: str) -> Optional[Callable[..., Any]]:
        """Sample or return an activation."""

        sampler = ActivationSampler(self.activation_choices, name)
        return sampler.sample(trial)

    def get_regularizer(self, trial: optuna.Trial, name: str) -> Optional[tf.keras.regularizers.Regularizer]:
        """Sample a regularizer."""

        sampler = RegularizerSampler(self.regularizer_choices, name)
        return sampler.sample(trial)

    def get_optimizer(self, trial: optuna.Trial) -> tf.keras.optimizers.Optimizer:
        """Sample an optimizer."""

        sampler = OptimizerSampler(self.optimizer_choices, "optimizer", self.learning_rate)
        return sampler.sample(trial)

    def get_scaler(self, trial: optuna.Trial):
        """Sample a scikit-learn scaler."""

        sampler = ScalerSampler(self.scaler_choices, "scaler")
        return sampler.sample(trial)

    def get_initializer(self, trial: optuna.Trial, name: str) -> tf.keras.initializers.Initializer:
        """Sample a kernel initializer."""

        sampler = InitializerSampler(self.initializer_choices, name, self.seed)
        return sampler.sample(trial)

    @classmethod
    def get_default_params(cls) -> "KParams":
        """Return an instance with all default options."""

        return cls()

    @classmethod
    def default(cls) -> "KParams":
        """Alias for :meth:`get_default_params`."""

        return cls.get_default_params()

    @classmethod
    def full_search_space(cls) -> "KParams":
        """
        Return an instance of the class with all available options from Keras for the search spaces.
        Excluded options:
            - OrthogonalRegularizer

        """

        return cls(
            activation_choices=[
                tf.keras.activations.celu,
                tf.keras.activations.elu,
                tf.keras.activations.exponential,
                tf.keras.activations.gelu,
                tf.keras.activations.glu,
                tf.keras.activations.hard_shrink,
                tf.keras.activations.hard_sigmoid,
                tf.keras.activations.hard_silu,
                tf.keras.activations.hard_tanh,
                tf.keras.activations.leaky_relu,
                tf.keras.activations.linear,
                tf.keras.activations.log_sigmoid,
                tf.keras.activations.log_softmax,
                tf.keras.activations.mish,
                tf.keras.activations.relu,
                tf.keras.activations.relu6,
                tf.keras.activations.selu,
                tf.keras.activations.sigmoid,
                tf.keras.activations.silu,
                tf.keras.activations.softmax,
                tf.keras.activations.soft_shrink,
                tf.keras.activations.softplus,
                tf.keras.activations.softsign,
                tf.keras.activations.sparse_plus,
                tf.keras.activations.sparsemax,
                tf.keras.activations.squareplus,
                tf.keras.activations.tanh,
                tf.keras.activations.tanh_shrink,
                tf.keras.activations.threshold,
                None,
            ],
            regularizer_choices=[
                tf.keras.regularizers.L1(1e-2),
                tf.keras.regularizers.L2(1e-2),
                tf.keras.regularizers.L1L2(l1=1e-2, l2=1e-2),
                # tf.keras.regularizers.OrthogonalRegularizer(factor=0.01, mode="rows"),
                None,
            ],
            optimizer_choices=[
                tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
                tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0),
                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                tf.keras.optimizers.AdamW(
                    learning_rate=0.001, weight_decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-7
                ),
                tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-7),
                tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-7),
                tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                tf.keras.optimizers.Ftrl(
                    learning_rate=0.001,
                    learning_rate_power=-0.5,
                    l1_regularization_strength=0.0,
                    l2_regularization_strength=0.0,
                ),
                tf.keras.optimizers.Adafactor(learning_rate=None, relative_step=True, weight_decay=0.0),
                tf.keras.optimizers.Lion(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                tf.keras.optimizers.Lamb(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-6),
                tf.keras.optimizers.Muon(learning_rate=0.001, weight_decay=0.1),
                tf.keras.optimizers.LossScaleOptimizer(
                    inner_optimizer=tf.keras.optimizers.Adam(), initial_scale=2**15
                ),
            ],
            scaler_choices=[
                StandardScaler,
                lambda: MinMaxScaler(feature_range=(0, 1)),
                lambda: MinMaxScaler(feature_range=(-1, 1)),
                RobustScaler,
                QuantileTransformer,
                PowerTransformer,
            ],
            initializer_choices=[
                tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                tf.keras.initializers.Zeros(),
                tf.keras.initializers.Ones(),
                tf.keras.initializers.GlorotNormal(seed=None),
                tf.keras.initializers.GlorotUniform(seed=None),
                tf.keras.initializers.HeNormal(seed=None),
                tf.keras.initializers.HeUniform(seed=None),
                tf.keras.initializers.Orthogonal(gain=1.0, seed=None),
                tf.keras.initializers.Constant(value=0.0),
                tf.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None
                ),
                tf.keras.initializers.LecunNormal(seed=None),
                tf.keras.initializers.LecunUniform(seed=None),
                tf.keras.initializers.Identity(gain=1.0),
            ],
        )
