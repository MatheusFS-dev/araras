"""Transformer language-model building block."""

from araras.core import *

import optuna
import tensorflow as tf
from tensorflow.keras import layers, initializers

from araras.ml.model.hyperparams import KParams


def _suggest_dropout_rate(
    trial: optuna.Trial,
    dropout_range: Union[float, tuple[float, float]],
    step: float,
    param_name: str,
) -> float:
    """Sample or validate a dropout rate using an Optuna trial.

    Args:
        trial: Active Optuna trial for hyperparameter suggestions.
        dropout_range: Either a fixed dropout rate or an inclusive ``(min, max)``
            interval to sample from.
        step: Step size used when sampling within the provided range.
        param_name: Name assigned to the sampled parameter inside the trial.

    Returns:
        float: Final dropout rate residing in ``[0.0, 1.0]``.

    Raises:
        ValueError: If the provided ``dropout_range`` is malformed or outside the
            valid probability interval.
    """

    if isinstance(dropout_range, float):
        if not 0.0 <= dropout_range <= 1.0:
            raise ValueError("Dropout rate must lie within [0.0, 1.0].")
        return dropout_range

    min_drop, max_drop = dropout_range
    if min_drop > max_drop:
        raise ValueError("Dropout range lower bound must not exceed upper bound.")

    if not (0.0 <= min_drop <= 1.0 and 0.0 <= max_drop <= 1.0):
        raise ValueError("Dropout range bounds must lie within [0.0, 1.0].")

    return trial.suggest_float(
        param_name,
        min_drop,
        max_drop,
        step=step,
    )


def tokenize(
    x: layers.Layer,
    sequence_length: int,
    token_dim: int,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    name_prefix: str = "tokenizer",
) -> layers.Layer:
    """Project inputs into token embeddings and add positional encodings.

    This helper applies a dense projection to ``token_dim`` features for each input
    token, then injects a learned positional embedding with the same dimensionality.
    The output preserves the batch and sequence axes while replacing the feature
    dimension with the token embedding size.

    Notes:
        The function expects the input tensor ``x`` to be rank-3 with shape
        ``(batch_size, sequence_length, feature_dim)``. The provided
        ``sequence_length`` must match the actual number of time steps present in
        ``x``.
        When pairing with :func:`build_lm`, choose ``token_dim`` so that it is
        divisible by the desired number of attention heads. This ensures
        ``token_dim`` can be factored into ``num_heads * key_dim`` inside the
        Transformer block.

    Warnings:
        Providing an incorrect ``sequence_length`` leads to mismatched positional
        embeddings and will raise shape errors at runtime. Ensure the sequence axis
        aligns with ``sequence_length`` before calling this function.

    Args:
        x: Input tensor or Keras layer containing token-level features.
        sequence_length: Number of tokens in each sequence. Must be positive.
        token_dim: Dimensionality of the token embeddings to generate.
        kernel_initializer: Initializer applied to the Dense projection weights and
            the positional embedding table.
        bias_initializer: Initializer used for the Dense projection bias term.
        name_prefix: Prefix used to name the underlying Keras layers.

    Returns:
        layers.Layer: Tensor with shape ``(batch_size, sequence_length, token_dim)``
        that includes positional information.

    Raises:
        ValueError: If ``sequence_length`` or ``token_dim`` are not positive.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer.")

    if token_dim <= 0:
        raise ValueError("token_dim must be a positive integer.")

    x = layers.Dense(
        token_dim,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=f"{name_prefix}_embed_points",
    )(x)

    pos_indices = tf.range(start=0, limit=sequence_length, dtype=tf.int32)
    pos_emb_layer = layers.Embedding(
        input_dim=sequence_length,
        output_dim=token_dim,
        embeddings_initializer=kernel_initializer,
        name=f"{name_prefix}_positional_embedding",
    )
    pos_emb = pos_emb_layer(pos_indices)
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x = layers.Add(name=f"{name_prefix}_add_pos")([x, pos_emb])

    return x


def build_lm(
    trial: optuna.Trial,
    kparams: KParams,
    x: layers.Layer,
    num_heads_range: Union[int, tuple[int, int]] = 4,
    key_dim_range: Optional[Union[int, tuple[int, int]]] = None,
    ffn_multiplier_range: Union[int, tuple[int, int]] = (2, 4),
    attn_dropout_rate_range: Union[float, tuple[float, float]] = (0.0, 0.3),
    ffn_dropout_rate_range: Union[float, tuple[float, float]] = (0.0, 0.3),
    num_heads_step: int = 1,
    key_dim_step: int = 1,
    ffn_multiplier_step: int = 1,
    attn_dropout_rate_step: float = 0.05,
    ffn_dropout_rate_step: float = 0.05,
    trial_kernel_reg: bool = False,
    activation: Optional[Union[str, Callable[..., Any]]] = None,
    name_prefix: str = "lm",
) -> layers.Layer:
    """Build a single Transformer encoder block for language modeling.

    The block applies layer-normalized multi-head self-attention followed by a
    position-wise feed-forward network with residual connections. Hyperparameters
    such as the number of heads, key dimensions, feed-forward width multipliers,
    dropout rates, kernel regularizer, and activation can be tuned via an Optuna
    ``trial``.

    Note:
        The input ``x`` must already be tokenized and include positional
        encodings. When using :func:`tokenize`, the ``token_dim`` argument defines
        the embedding width for ``x``. To preserve residual additions, ensure that
        ``token_dim`` equals ``num_heads * key_dim``. The default ``key_dim``
        behavior enforces this by deriving ``token_dim // num_heads``. When
        providing custom ranges for ``num_heads`` or ``key_dim``, only
        combinations satisfying that equality are viable.

    Warning:
        The trailing dimension of ``x`` must be statically known and divisible by
        the sampled number of heads. Mismatched dimensions prevent residual
        connections and will raise descriptive errors.

    Args:
        trial: Optuna trial used to sample hyperparameters.
        kparams: Hyperparameter helper that provides regularizers and activations.
        x: Input tensor or Keras layer with token embeddings.
        num_heads_range: Fixed head count or inclusive range to sample from.
        key_dim_range: Fixed or ranged per-head key dimension. ``None`` derives a
            balanced value ``token_dim // num_heads`` that preserves residual
            compatibility.
        ffn_multiplier_range: Fixed or ranged multiplier for the feed-forward
            hidden dimension relative to ``token_dim``.
        attn_dropout_rate_range: Fixed dropout rate or inclusive range for the
            attention module.
        ffn_dropout_rate_range: Fixed dropout rate or inclusive range for the
            feed-forward module.
        num_heads_step: Step size when enumerating candidate head counts.
        key_dim_step: Step size when enumerating candidate key dimensions.
        ffn_multiplier_step: Step size when enumerating feed-forward multipliers.
        attn_dropout_rate_step: Step size for attention dropout rate sampling.
        ffn_dropout_rate_step: Step size for feed-forward dropout rate sampling.
        trial_kernel_reg: When ``True``, sample a kernel regularizer via
            :class:`KParams` for the feed-forward layers.
        activation: Optional activation spec applied inside the feed-forward
            network. When ``None`` and ``kparams`` is provided, an activation is
            sampled via :meth:`KParams.get_activation`. Strings equal to "none"
            (case-insensitive) disable the activation.
        name_prefix: Prefix used for layer naming.

    Returns:
        layers.Layer: Output tensor after applying the Transformer block.

    Raises:
        ValueError: If the trailing dimension of ``x`` is undefined.
        ValueError: If no attention head counts divide the token dimension.
        ValueError: If ``key_dim`` candidates are incompatible with the token
            dimension.
        ValueError: If range arguments are malformed (e.g., min > max or step <= 0).
    """

    token_dim = x.shape[-1]
    if token_dim is None:
        raise ValueError("Input tensor must have a statically defined feature dimension.")

    if num_heads_step <= 0:
        raise ValueError("num_heads_step must be a positive integer.")

    if ffn_multiplier_step <= 0:
        raise ValueError("ffn_multiplier_step must be a positive integer.")

    if attn_dropout_rate_step <= 0:
        raise ValueError("attn_dropout_rate_step must be positive.")

    if ffn_dropout_rate_step <= 0:
        raise ValueError("ffn_dropout_rate_step must be positive.")

    if key_dim_step <= 0:
        raise ValueError("key_dim_step must be a positive integer.")

    if isinstance(num_heads_range, int):
        candidate_heads = [num_heads_range]
    else:
        min_heads, max_heads = num_heads_range
        if min_heads > max_heads:
            raise ValueError("num_heads_range lower bound must not exceed upper bound.")
        candidate_heads = list(range(min_heads, max_heads + 1, num_heads_step))

    compatible_heads = [head for head in candidate_heads if token_dim % head == 0]
    if not compatible_heads:
        raise ValueError("No compatible attention heads found for the token dimension.")

    if len(compatible_heads) == 1:
        num_heads = compatible_heads[0]
    else:
        num_heads = trial.suggest_categorical(
            f"{name_prefix}_num_heads",
            compatible_heads,
        )

    if isinstance(ffn_multiplier_range, int):
        candidate_multipliers = [ffn_multiplier_range]
    else:
        min_mul, max_mul = ffn_multiplier_range
        if min_mul > max_mul:
            raise ValueError("ffn_multiplier_range lower bound must not exceed upper bound.")
        candidate_multipliers = list(range(min_mul, max_mul + 1, ffn_multiplier_step))

    if not candidate_multipliers:
        raise ValueError("No feed-forward multipliers available with the provided range and step.")

    if len(candidate_multipliers) == 1:
        ffn_multiplier = candidate_multipliers[0]
    else:
        ffn_multiplier = trial.suggest_categorical(
            f"{name_prefix}_ffn_multiplier",
            candidate_multipliers,
        )

    if key_dim_range is None:
        key_dim_candidates = [token_dim // num_heads]
    elif isinstance(key_dim_range, int):
        key_dim_candidates = [key_dim_range]
    else:
        min_key, max_key = key_dim_range
        if min_key > max_key:
            raise ValueError("key_dim_range lower bound must not exceed upper bound.")
        key_dim_candidates = list(range(min_key, max_key + 1, key_dim_step))

    key_dim_candidates = [candidate for candidate in key_dim_candidates if candidate > 0]
    key_dim_candidates = [candidate for candidate in key_dim_candidates if num_heads * candidate == token_dim]

    if not key_dim_candidates:
        raise ValueError(
            "No compatible key dimensions found. Ensure token_dim equals num_heads * key_dim.",
        )

    if len(key_dim_candidates) == 1:
        key_dim = key_dim_candidates[0]
    else:
        key_dim = trial.suggest_categorical(
            f"{name_prefix}_key_dim",
            key_dim_candidates,
        )

    attn_dropout_rate = _suggest_dropout_rate(
        trial,
        attn_dropout_rate_range,
        attn_dropout_rate_step,
        f"{name_prefix}_attn_dropout",
    )

    ffn_dropout_rate_1 = _suggest_dropout_rate(
        trial,
        ffn_dropout_rate_range,
        ffn_dropout_rate_step,
        f"{name_prefix}_ffn_dropout_1",
    )
    
    ffn_dropout_rate_2 = _suggest_dropout_rate(
        trial,
        ffn_dropout_rate_range,
        ffn_dropout_rate_step,
        f"{name_prefix}_ffn_dropout_2",
    )

    kernel_initializer = kparams.get_initializer(trial, f"{name_prefix}_kernel_init")
    kernel_regularizer = (
        kparams.get_regularizer(trial, f"{name_prefix}_kernel_reg")
        if trial_kernel_reg
        else None
    )

    explicit_none = isinstance(activation, str) and activation.lower() == "none"
    if explicit_none:
        activation = None

    if activation is None and not explicit_none:
        activation = kparams.get_activation(trial, f"{name_prefix}_ffn_act")

    attn_in = layers.LayerNormalization(name=f"{name_prefix}_ln_attn")(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=attn_dropout_rate,
        name=f"{name_prefix}_mha",
    )(attn_in, attn_in)
    x = layers.Add(name=f"{name_prefix}_attn_residual")([x, attn_out])

    ffn_in = layers.LayerNormalization(name=f"{name_prefix}_ln_ffn")(x)
    ffn = layers.Dense(
        token_dim * ffn_multiplier,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name_prefix}_ffn_dense1",
    )(ffn_in)
    ffn = layers.Dropout(
        ffn_dropout_rate_1,
        name=f"{name_prefix}_drop_ffn1",
    )(ffn)
    ffn = layers.Dense(
        token_dim,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name_prefix}_ffn_dense2",
    )(ffn)
    ffn = layers.Dropout(
        ffn_dropout_rate_2,
        name=f"{name_prefix}_drop_ffn2",
    )(ffn)
    x = layers.Add(name=f"{name_prefix}_ffn_residual")([x, ffn])

    return x

