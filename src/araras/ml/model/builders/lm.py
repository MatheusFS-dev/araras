from araras.core import *

from collections.abc import Sequence
from typing import Optional

import optuna
import tensorflow as tf
from tensorflow.keras import Model, initializers, layers

from araras.ml.model.hyperparams import KParams


def build_lm(
    trial: optuna.Trial,
    kparams: KParams,
    sequence_length: int,
    output_units: int,
    feature_dim: int = 1,
    embedding_input_dim: Optional[int] = None,
    embedding_mask_zero: bool = False,
    embedding_dim_options: Sequence[int] = (64, 128, 256),
    heads_options: Sequence[int] = (2, 4, 8),
    key_dim_options: Sequence[int] = (16, 32, 64),
    transformer_layers_range: tuple[int, int] = (1, 4),
    ffn_multiplier_range: tuple[int, int] = (2, 4),
    head_units_options: Sequence[int] = (128, 256, 512),
    dropout_range: tuple[float, float] = (0.0, 0.5),
    dropout_step: float = 0.1,
    activation_choices: Optional[Sequence[str]] = None,
    output_activation: Optional[str] = None,
    show_summary: bool = True,
    initializer: Optional[initializers.Initializer] = None,
    name_prefix: str = "lm",
) -> Model:
    """Build a generic transformer-style language model.

    This architecture accepts either dense feature sequences or integer token identifiers
    and applies configurable Transformer encoder blocks for contextualization. The
    resulting token representations are aggregated into a compact sequence embedding and
    projected onto the requested number of output units, enabling reuse across
    classification, regression, or multi-label prediction tasks.

    Notes:
        When ``embedding_input_dim`` is provided the input is assumed to consist of
        integer token identifiers and an embedding layer is inserted automatically.
        Otherwise the model expects floating point features with shape ``(sequence_length,
        feature_dim)``.

    Warnings:
        Large ``sequence_length`` values increase the memory footprint of the positional
        embeddings and attention mechanism quadratically.

    Args:
        trial: Hyperparameter sampling interface, typically provided by Optuna.
        kparams: Helper that supplies optimizers, regularizers, and activations.
        sequence_length: Number of time steps or tokens consumed by the model.
        output_units: Size of the final prediction layer.
        feature_dim: Number of features per token when embeddings are not used.
        embedding_input_dim: Vocabulary size for the optional embedding layer. If
            ``None``, the inputs are treated as continuous features.
        embedding_mask_zero: Whether the embedding layer should mask zero tokens.
        embedding_dim_options: Candidate output dimensions for the embedding layer.
        heads_options: Candidate numbers of attention heads.
        key_dim_options: Candidate key dimensions per attention head.
        transformer_layers_range: Inclusive range for the number of Transformer blocks.
        ffn_multiplier_range: Inclusive range for the feed-forward expansion multiplier.
        head_units_options: Candidate units for the dense head preceding the outputs.
        dropout_range: Inclusive range for dropout probabilities sampled per trial.
        dropout_step: Step size used when sampling dropout probabilities.
        activation_choices: Optional sequence of activation names for the feed-forward
            blocks. Defaults to ("relu", "gelu", "silu").
        output_activation: Activation function applied to the final Dense layer.
        show_summary: Whether to display the model summary after construction.
        initializer: Optional initializer applied to dense and embedding projections. If
            ``None`` a Glorot uniform initializer seeded with ``kparams.seed`` is used.
        name_prefix: Prefix applied to generated layer names for easier inspection.

    Returns:
        Model: A compiled Keras model ready for training.

    Raises:
        ValueError: If ``sequence_length`` or ``output_units`` is not positive.
        ValueError: If ``embedding_input_dim`` is provided but not positive.
        ValueError: If ``feature_dim`` is not positive when embeddings are disabled.
        ValueError: If ``dropout_range`` falls outside the ``[0.0, 1.0]`` interval.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer.")

    if output_units <= 0:
        raise ValueError("output_units must be a positive integer.")

    if embedding_input_dim is not None and embedding_input_dim <= 0:
        raise ValueError("embedding_input_dim must be positive when provided.")

    if embedding_input_dim is None and feature_dim <= 0:
        raise ValueError("feature_dim must be positive when embeddings are disabled.")

    if not (0.0 <= dropout_range[0] <= dropout_range[1] <= 1.0):
        raise ValueError("dropout_range must define values within [0.0, 1.0].")

    activation_choices = activation_choices or ("relu", "gelu", "silu")

    if initializer is None:
        seed = getattr(kparams, "seed", None)
        initializer = initializers.GlorotUniform(seed=seed)

    if embedding_input_dim is not None:
        inputs = layers.Input(
            shape=(sequence_length,),
            dtype="int32",
            name=f"{name_prefix}_token_ids",
        )
        embedding_dim = trial.suggest_categorical(
            f"{name_prefix}_embedding_dim",
            list(embedding_dim_options),
        )
        x = layers.Embedding(
            input_dim=embedding_input_dim,
            output_dim=embedding_dim,
            embeddings_initializer=initializer,
            mask_zero=embedding_mask_zero,
            name=f"{name_prefix}_token_embedding",
        )(inputs)
    else:
        inputs = layers.Input(
            shape=(sequence_length, feature_dim),
            dtype="float32",
            name=f"{name_prefix}_features",
        )
        x = inputs

    heads = trial.suggest_categorical(f"{name_prefix}_attn_heads", list(heads_options))
    key_dim = trial.suggest_categorical(f"{name_prefix}_attn_key_dim", list(key_dim_options))
    d_model = heads * key_dim

    dropout_rate = trial.suggest_float(
        f"{name_prefix}_dropout",
        dropout_range[0],
        dropout_range[1],
        step=dropout_step,
    )

    if embedding_input_dim is None:
        x = layers.Dense(
            d_model,
            kernel_initializer=initializer,
            name=f"{name_prefix}_feature_projection",
        )(x)
    else:
        x = layers.Dense(
            d_model,
            kernel_initializer=initializer,
            name=f"{name_prefix}_embedding_projection",
        )(x)

    pos_indices = tf.range(start=0, limit=sequence_length, dtype=tf.int32)
    pos_emb_layer = layers.Embedding(
        input_dim=sequence_length,
        output_dim=d_model,
        name=f"{name_prefix}_positional_embedding",
    )
    pos_table = pos_emb_layer(pos_indices)
    pos_table = tf.expand_dims(pos_table, axis=0)
    x = layers.Add(name=f"{name_prefix}_add_pos")([x, pos_table])

    layer_count = trial.suggest_int(
        f"{name_prefix}_transformer_layers",
        transformer_layers_range[0],
        transformer_layers_range[1],
    )
    ffn_multiplier = trial.suggest_int(
        f"{name_prefix}_ffn_mult",
        ffn_multiplier_range[0],
        ffn_multiplier_range[1],
    )
    act_name = trial.suggest_categorical(
        f"{name_prefix}_ffn_act",
        list(activation_choices),
    )
    activation = tf.keras.activations.get(act_name)

    for idx_block in range(layer_count):
        attn_norm = layers.LayerNormalization(name=f"{name_prefix}_ln_attn_{idx_block}")(x)
        attn_out = layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            name=f"{name_prefix}_mha_{idx_block}",
        )(attn_norm, attn_norm)
        attn_out = layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_drop_attn_{idx_block}",
        )(attn_out)
        x = layers.Add(name=f"{name_prefix}_res_attn_{idx_block}")([x, attn_out])

        ffn_norm = layers.LayerNormalization(name=f"{name_prefix}_ln_ffn_{idx_block}")(x)
        ffn_hidden = layers.Dense(
            ffn_multiplier * d_model,
            activation=activation,
            kernel_initializer=initializer,
            name=f"{name_prefix}_ffn1_{idx_block}",
        )(ffn_norm)
        ffn_hidden = layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_drop_ffn1_{idx_block}",
        )(ffn_hidden)
        ffn_out = layers.Dense(
            d_model,
            kernel_initializer=initializer,
            name=f"{name_prefix}_ffn2_{idx_block}",
        )(ffn_hidden)
        ffn_out = layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_drop_ffn2_{idx_block}",
        )(ffn_out)
        x = layers.Add(name=f"{name_prefix}_res_ffn_{idx_block}")([x, ffn_out])

    x = layers.LayerNormalization(name=f"{name_prefix}_ln_out")(x)
    x = layers.GlobalAveragePooling1D(name=f"{name_prefix}_pool")(x)

    head_units = trial.suggest_categorical(
        f"{name_prefix}_head_units",
        list(head_units_options),
    )
    x = layers.Dense(
        head_units,
        activation=tf.keras.activations.get("relu"),
        kernel_initializer=initializer,
        name=f"{name_prefix}_head_dense",
    )(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_head_drop")(x)

    outputs = layers.Dense(
        output_units,
        activation=output_activation,
        kernel_initializer=initializer,
        name=f"{name_prefix}_head_logits",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"{name_prefix}_model")

    optimizer = kparams.get_optimizer(trial)
    if show_summary:
        model.summary()
    model.compile(
        optimizer=optimizer,
        loss=kparams.get_loss(trial) if hasattr(kparams, "get_loss") else "mse",
        metrics=kparams.get_metrics(trial) if hasattr(kparams, "get_metrics") else ["mse", "mae"],
        jit_compile=False,
    )
    return model
