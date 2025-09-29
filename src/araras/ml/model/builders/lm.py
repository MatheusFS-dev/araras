"""Transformer language-model building block."""

from araras.core import *

from collections.abc import Sequence

import optuna
import tensorflow as tf
from tensorflow.keras import layers

from araras.ml.model.hyperparams import KParams


def build_lm(
    trial: optuna.Trial,
    kparams: KParams,
    x: layers.Layer,
    sequence_length: int,
    token_dim_options: Sequence[int] = (64, 128, 256),
    head_options: Sequence[int] = (2, 4, 8),
    transformer_layers_range: tuple[int, int] = (1, 2),
    ffn_multiplier_options: Sequence[int] = (2, 4),
    dropout_range: tuple[float, float] = (0.0, 0.3),
    dropout_step: float = 0.05,
    trial_kernel_reg: bool = False,
    trial_activation: bool = True,
    name_prefix: str = "lm",
) -> layers.Layer:
    """Build a configurable Transformer encoder stack for language modeling.

    This builder projects the incoming features into a token embedding space, adds a
    learned positional encoding, and applies a stack of Transformer encoder blocks.
    Hyperparameters such as the embedding width, number of attention heads, dropout
    rates, feed-forward expansion factor, and activation function are optimized via
    Optuna by sampling from the provided search spaces.

    Notes:
        The function expects the input tensor ``x`` to have shape ``(batch_size,
        sequence_length, feature_dim)``. The ``sequence_length`` argument must match
        the second dimension so that the positional embedding table is created with
        the appropriate size.

    Warnings:
        A mismatch between ``sequence_length`` and the actual time dimension of
        ``x`` causes shape inconsistencies when broadcasting the positional
        embeddings. Ensure these values align to avoid runtime errors.

    Args:
        trial: Optuna trial used to sample hyperparameters.
        kparams: Hyperparameter helper that provides regularizers and activations.
        x: Input tensor or Keras layer representing the token features.
        sequence_length: Number of tokens per sequence; must be positive.
        token_dim_options: Candidate embedding widths for the token projection.
        head_options: Candidate numbers of attention heads.
        transformer_layers_range: Inclusive range for the number of Transformer
            encoder blocks to stack.
        ffn_multiplier_options: Candidate expansion factors applied to the
            feed-forward network hidden width.
        dropout_range: Inclusive range for dropout probabilities.
        dropout_step: Step size used when sampling dropout probabilities.
        trial_kernel_reg: When ``True``, sample a kernel regularizer via
            :class:`KParams` for the feed-forward layers.
        trial_activation: When ``True``, sample the feed-forward activation via
            :class:`KParams`. If ``False``, the activation defaults to ReLU.
        name_prefix: Prefix used for layer naming.

    Returns:
        layers.Layer: Output tensor after applying the Transformer stack.

    Raises:
        ValueError: If ``sequence_length`` is not positive.
        ValueError: If ``token_dim_options`` or ``head_options`` is empty.
        ValueError: If the sampled embedding width is incompatible with all head
            options (i.e., it is not divisible by any candidate head count).
        ValueError: If ``dropout_range`` does not fall within the ``[0.0, 1.0]``
            interval or ``dropout_range[0]`` exceeds ``dropout_range[1]``.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer.")

    if not token_dim_options:
        raise ValueError("token_dim_options must contain at least one value.")

    if not head_options:
        raise ValueError("head_options must contain at least one value.")

    if not (0.0 <= dropout_range[0] <= dropout_range[1] <= 1.0):
        raise ValueError("dropout_range must define values within [0.0, 1.0].")

    token_dim = trial.suggest_categorical(
        f"{name_prefix}_token_dim",
        list(token_dim_options),
    )

    compatible_heads = [head for head in head_options if token_dim % head == 0]
    if not compatible_heads:
        raise ValueError(
            "No compatible attention heads found for the sampled token dimension."
        )

    num_heads = trial.suggest_categorical(
        f"{name_prefix}_num_heads",
        compatible_heads,
    )

    dropout_rate = trial.suggest_float(
        f"{name_prefix}_dropout",
        dropout_range[0],
        dropout_range[1],
        step=dropout_step,
    )

    kernel_initializer = kparams.get_initializer(trial, f"{name_prefix}_kernel_init")
    kernel_regularizer = (
        kparams.get_regularizer(trial, f"{name_prefix}_kernel_reg")
        if trial_kernel_reg
        else None
    )

    activation = (
        kparams.get_activation(trial, f"{name_prefix}_ffn_act")
        if trial_activation
        else tf.keras.activations.relu
    )

    # ——————————————————————————————— Tokenization ——————————————————————————————— #
    x = layers.Dense(
        token_dim,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer if trial_kernel_reg else None,
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

    # ———————————————————————————————— Transformer ——————————————————————————————— #
    num_layers = trial.suggest_int(
        f"{name_prefix}_num_layers",
        transformer_layers_range[0],
        transformer_layers_range[1],
    )

    ffn_multiplier = trial.suggest_categorical(
        f"{name_prefix}_ffn_multiplier",
        list(ffn_multiplier_options),
    )

    for block_idx in range(num_layers):
        attn_in = layers.LayerNormalization(name=f"{name_prefix}_ln_attn_{block_idx}")(x)
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=token_dim // num_heads,
            dropout=dropout_rate,
            name=f"{name_prefix}_mha_{block_idx}",
        )(attn_in, attn_in)
        attn_out = layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_drop_attn_{block_idx}",
        )(attn_out)
        x = layers.Add(name=f"{name_prefix}_attn_residual_{block_idx}")([x, attn_out])

        ffn_in = layers.LayerNormalization(name=f"{name_prefix}_ln_ffn_{block_idx}")(x)
        ffn = layers.Dense(
            token_dim * ffn_multiplier,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f"{name_prefix}_ffn_dense1_{block_idx}",
        )(ffn_in)
        ffn = layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_drop_ffn1_{block_idx}",
        )(ffn)
        ffn = layers.Dense(
            token_dim,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f"{name_prefix}_ffn_dense2_{block_idx}",
        )(ffn)
        ffn = layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_drop_ffn2_{block_idx}",
        )(ffn)
        x = layers.Add(name=f"{name_prefix}_ffn_residual_{block_idx}")([x, ffn])

    return x

