from araras.core import *

from dataclasses import dataclass, field
from typing import Optional, Sequence, Any, Callable, Union, Mapping, Dict
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


def _ensure_mapping(choices: Union[Sequence[Any], Mapping[str, Any]]) -> Dict[str, Any]:
    """Convert choices to a ``dict`` keyed by unique strings."""
    if isinstance(choices, Mapping):
        return dict(choices)
    mapping: Dict[str, Any] = {}
    for idx, choice in enumerate(choices):
        if hasattr(choice, "__name__"):
            base = str(choice.__name__)
        elif hasattr(choice, "__class__"):
            base = str(choice.__class__.__name__)
        else:
            base = f"choice_{idx}"
        key = base
        counter = 1
        while key in mapping:
            key = f"{base}_{counter}"
            counter += 1
        mapping[key] = choice
    return mapping


class BaseSampler(ABC):
    """Abstract sampler for Keras hyperparameters."""

    def __init__(self, choices: Union[Sequence[Any], Mapping[str, Any]], name: str) -> None:
        self.mapping = _ensure_mapping(choices)
        self.choices = list(self.mapping.keys())
        self.name = name

    def sample(self, trial: optuna.Trial) -> Any:
        try:
            key = _sample_choice(trial, self.name, self.choices)
            choice = self.mapping[key]
            return self._process(choice, trial)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
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
    def __init__(self, choices: Sequence[Any], name: str, lr: Union[float, tuple[float, float]]) -> None:
        super().__init__(choices, name)
        self.lr = lr

    def _process(self, choice: Any, trial: optuna.Trial) -> tf.keras.optimizers.Optimizer:
        if isinstance(self.lr, tuple):
            lr = trial.suggest_float("learning_rate", self.lr[0], self.lr[1], log=True)
        else:
            lr = self.lr
        if isinstance(choice, tf.keras.optimizers.Optimizer):
            config = choice.get_config()
            config["learning_rate"] = lr
            return choice.__class__.from_config(config)
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

    activation_choices: Dict[str, Optional[Callable[..., Any]]] = field(
        default_factory=lambda: {
            "relu": tf.keras.activations.relu,
            "gelu": tf.keras.activations.gelu,
            "silu": tf.keras.activations.silu,
            "elu": tf.keras.activations.elu,
            "sigmoid": tf.keras.activations.sigmoid,
            "tanh": tf.keras.activations.tanh,
            "none": None,
        }
    )

    regularizer_choices: Dict[
        str,
        Optional[
            Union[
                type[tf.keras.regularizers.Regularizer],
                tf.keras.regularizers.Regularizer,
                Callable[[], tf.keras.regularizers.Regularizer],
            ]
        ],
    ] = field(
        default_factory=lambda: {
            "none": None,
            "l2": tf.keras.regularizers.L2(1e-2),
        }
    )

    optimizer_choices: Dict[
        str,
        Union[
            type[tf.keras.optimizers.Optimizer],
            tf.keras.optimizers.Optimizer,
            Callable[[], tf.keras.optimizers.Optimizer],
        ],
    ] = field(
        default_factory=lambda: {
            "sgd": tf.keras.optimizers.SGD(momentum=0.9),
            "adam": tf.keras.optimizers.Adam(),
            "adamw": tf.keras.optimizers.AdamW(weight_decay=1e-4),
            "lion": tf.keras.optimizers.Lion(beta_1=0.9, beta_2=0.99),
            "rmsprop": tf.keras.optimizers.RMSprop(),
        }
    )

    scaler_choices: Dict[str, Union[type, Callable[[], Any], Any]] = field(
        default_factory=lambda: {
            "standard": StandardScaler,
            "minmax_0_1": lambda: MinMaxScaler(feature_range=(0, 1)),
            "minmax_-1_1": lambda: MinMaxScaler(feature_range=(-1, 1)),
        }
    )

    initializer_choices: Dict[
        str,
        Union[
            type[tf.keras.initializers.Initializer],
            tf.keras.initializers.Initializer,
            Callable[[], tf.keras.initializers.Initializer],
        ],
    ] = field(
        default_factory=lambda: {
            "glorot_uniform": tf.keras.initializers.GlorotUniform,
        }
    )

    def __post_init__(self) -> None:
        if isinstance(self.learning_rate, tuple):
            if (
                len(self.learning_rate) != 2
                or self.learning_rate[0] <= 0
                or self.learning_rate[1] <= 0
                or self.learning_rate[0] >= self.learning_rate[1]
            ):
                raise ValueError("learning_rate tuple must be (min, max) with 0 < min < max")
        elif not isinstance(self.learning_rate, (int, float)):
            raise TypeError("learning_rate must be a float or a tuple of two floats")

    # ———————————————————————————————————————————————————————————————————————————— #

    def set_activation_choices(self, choices: Union[Sequence[Any], Mapping[str, Any]]) -> None:
        """Set the available activation choices."""

        self.activation_choices = _ensure_mapping(choices)

    def set_regularizer_choices(
        self,
        choices: Union[
            Sequence[
                Optional[
                    Union[
                        type[tf.keras.regularizers.Regularizer],
                        tf.keras.regularizers.Regularizer,
                        Callable[[], tf.keras.regularizers.Regularizer],
                    ]
                ]
            ],
            Mapping[str, Any],
        ],
    ) -> None:
        """Set the available regularizer choices."""

        self.regularizer_choices = _ensure_mapping(choices)

    def set_optimizer_choices(
        self,
        choices: Union[
            Sequence[
                Union[
                    type[tf.keras.optimizers.Optimizer],
                    tf.keras.optimizers.Optimizer,
                    Callable[[], tf.keras.optimizers.Optimizer],
                ]
            ],
            Mapping[str, Any],
        ],
    ) -> None:
        """Set the available optimizer choices."""

        self.optimizer_choices = _ensure_mapping(choices)

    def set_scaler_choices(self, choices: Union[Sequence[Any], Mapping[str, Any]]) -> None:
        """Set the available scaler choices."""

        self.scaler_choices = _ensure_mapping(choices)

    def set_initializer_choices(
        self,
        choices: Union[
            Sequence[
                Union[
                    type[tf.keras.initializers.Initializer],
                    tf.keras.initializers.Initializer,
                    Callable[[], tf.keras.initializers.Initializer],
                ]
            ],
            Mapping[str, Any],
        ],
    ) -> None:
        """Set the available initializer choices."""

        self.initializer_choices = _ensure_mapping(choices)

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
        Return an instance of the class with all available options from Keras and their default values for the search space.
        Excluded options:
            - OrthogonalRegularizer

        """

        return cls(
            activation_choices={
                "celu": tf.keras.activations.celu,
                "elu": tf.keras.activations.elu,
                "exponential": tf.keras.activations.exponential,
                "gelu": tf.keras.activations.gelu,
                "glu": tf.keras.activations.glu,
                "hard_shrink": tf.keras.activations.hard_shrink,
                "hard_sigmoid": tf.keras.activations.hard_sigmoid,
                "hard_silu": tf.keras.activations.hard_silu,
                "hard_tanh": tf.keras.activations.hard_tanh,
                "leaky_relu": tf.keras.activations.leaky_relu,
                "linear": tf.keras.activations.linear,
                "log_sigmoid": tf.keras.activations.log_sigmoid,
                "log_softmax": tf.keras.activations.log_softmax,
                "mish": tf.keras.activations.mish,
                "relu": tf.keras.activations.relu,
                "relu6": tf.keras.activations.relu6,
                "selu": tf.keras.activations.selu,
                "sigmoid": tf.keras.activations.sigmoid,
                "silu": tf.keras.activations.silu,
                "softmax": tf.keras.activations.softmax,
                "soft_shrink": tf.keras.activations.soft_shrink,
                "softplus": tf.keras.activations.softplus,
                "softsign": tf.keras.activations.softsign,
                "sparse_plus": tf.keras.activations.sparse_plus,
                "sparsemax": tf.keras.activations.sparsemax,
                "squareplus": tf.keras.activations.squareplus,
                "tanh": tf.keras.activations.tanh,
                "tanh_shrink": tf.keras.activations.tanh_shrink,
                "threshold": tf.keras.activations.threshold,
                "none": None,
            },
            regularizer_choices={
                "l1": tf.keras.regularizers.L1(1e-2),
                "l2": tf.keras.regularizers.L2(1e-2),
                "l1_l2": tf.keras.regularizers.L1L2(l1=1e-2, l2=1e-2),
                # "orthogonal": tf.keras.regularizers.OrthogonalRegularizer(factor=0.01, mode="rows"),
                "none": None,
            },
            optimizer_choices={
                "sgd": tf.keras.optimizers.SGD(momentum=0.0, nesterov=False),
                "rmsprop": tf.keras.optimizers.RMSprop(rho=0.9, momentum=0.0),
                "adam": tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                "adamw": tf.keras.optimizers.AdamW(weight_decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                "adadelta": tf.keras.optimizers.Adadelta(rho=0.95, epsilon=1e-7),
                "adagrad": tf.keras.optimizers.Adagrad(initial_accumulator_value=0.1, epsilon=1e-7),
                "adamax": tf.keras.optimizers.Adamax(beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                "nadam": tf.keras.optimizers.Nadam(beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                "ftrl": tf.keras.optimizers.Ftrl(
                    learning_rate_power=-0.5,
                    l1_regularization_strength=0.0,
                    l2_regularization_strength=0.0,
                ),
                "adafactor": tf.keras.optimizers.Adafactor(
                    learning_rate=None, relative_step=True, weight_decay=0.0
                ),
                "lion": tf.keras.optimizers.Lion(beta_1=0.9, beta_2=0.99),
                "lamb": tf.keras.optimizers.Lamb(beta_1=0.9, beta_2=0.999, epsilon=1e-6),
                "muon": tf.keras.optimizers.Muon(weight_decay=0.1),
                "loss_scale": tf.keras.optimizers.LossScaleOptimizer(
                    inner_optimizer=tf.keras.optimizers.Adam(), initial_scale=2**15
                ),
            },
            scaler_choices={
                "standard": StandardScaler,
                "minmax_0_1": lambda: MinMaxScaler(feature_range=(0, 1)),
                "minmax_-1_1": lambda: MinMaxScaler(feature_range=(-1, 1)),
                "robust": RobustScaler,
                "quantile": QuantileTransformer,
                "power": PowerTransformer,
            },
            initializer_choices={
                "random_normal": tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                "random_uniform": tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
                "truncated_normal": tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05),
                "zeros": tf.keras.initializers.Zeros(),
                "ones": tf.keras.initializers.Ones(),
                "glorot_normal": tf.keras.initializers.GlorotNormal(),
                "glorot_uniform": tf.keras.initializers.GlorotUniform(),
                "he_normal": tf.keras.initializers.HeNormal(),
                "he_uniform": tf.keras.initializers.HeUniform(),
                "orthogonal": tf.keras.initializers.Orthogonal(gain=1.0),
                "constant": tf.keras.initializers.Constant(value=0.0),
                "variance_scaling": tf.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
                "lecun_normal": tf.keras.initializers.LecunNormal(),
                "lecun_uniform": tf.keras.initializers.LecunUniform(),
                "identity": tf.keras.initializers.Identity(gain=1.0),
            },
        )
