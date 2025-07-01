"""
This module provides a function to create skip connections in a Keras model.

Function:
    - `trial_skip_connections`: Creates skip connections based on a trial object and a list of layers.

Usage example:
    ```python
    from araras.keras.skip_connections import trial_skip_connections
    import tensorflow as tf
    # Assuming `trial` is an Optuna trial object and `layers_list` is a list of Keras layers
    final_tensor = trial_skip_connections(trial, layers_list)
    ```
"""

from typing import *
import itertools
import tensorflow as tf
from tensorflow.keras import layers


def trial_skip_connections(
    trial: Any,
    layers_list: Sequence[tf.Tensor],
    axis_to_concat: int = -1,
    print_combinations: bool = False,
    strategy: str = "final",
    merge_mode: str = "concat",
) -> tf.Tensor:
    """Constructs conditional skip connections between layers based on Optuna trial choices.

    This function introduces optional skip connections in a neural network architecture,
    governed by a hyperparameter search using Optuna's `trial.suggest_categorical` method.

    It allows experimentation with skip connection topology by conditionally merging outputs
    from earlier layers into later ones. The merging is done via concatenation or addition.

    **Important**:
    All tensors that are merged must have identical shapes in all dimensions **except** for
    the `axis_to_concat` dimension when using `'concat'`. For `'add'`, tensors must be of
    exactly the same shape.

    Args:
        trial (optuna.trial.Trial): Optuna trial object used to sample categorical decisions
            on whether to include each potential skip connection. It is expected to have the
            method `suggest_categorical(name: str, choices: List[Any]) -> Any`.
        layers_list (Sequence[tf.Tensor]): List of layer output tensors from a Keras model.
            These are the candidate sources and targets for skip connections. The order in
            the list reflects the network's topological sequence.
        axis_to_concat (int, optional): Axis along which tensors will be concatenated if
            `merge_mode` is `'concat'`. Default is -1 (last axis). All tensors to be
            concatenated must match on all other dimensions.
        print_combinations (bool, optional): If True, prints every possible combination
            of skip connections as dictionaries mapping skip flags to booleans. Primarily
            for debugging and audit purposes. Defaults to False.
        strategy (str, optional): Strategy for selecting candidate skip connections.
            - `'final'`: Allows skips only to the final layer.
            - `'any'`: Allows skips from any earlier layer `i` to any later layer `j`.
            Defaults to `'final'`.
        merge_mode (str, optional): Defines how selected tensors are merged:
            - `'concat'`: Tensors are concatenated along `axis_to_concat`.
            - `'add'`: Tensors are added element-wise (must be same shape).
            Defaults to `'concat'`.

    Returns:
        tf.Tensor: The output tensor resulting from applying the selected skip connections
        and merging strategy to the input layer sequence.

    Raises:
        ValueError: If `strategy` is not one of `'final'` or `'any'`.
        ValueError: If `merge_mode` is not one of `'concat'` or `'add'`.
    """

    # Determine number of layers to consider
    N = len(layers_list)

    # If fewer than two layers, no skips can be applied; return the only available layer
    if N < 2:
        return layers_list[-1]

    # Index of the final layer
    last_idx = N - 1

    # Define skip connection candidates based on selected strategy
    if strategy == "any":
        # All possible forward-directed layer pairs (i < j)
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    elif strategy == "final":
        # Only connections from earlier layers to the final layer
        pairs = [(i, last_idx) for i in range(last_idx)]
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'final' or 'any'.")

    # Calculate total number of skip combinations
    num_skips = len(pairs)
    total_combinations = 2**num_skips

    # Optionally print every combination of skip configuration
    if print_combinations:
        print(f"Total skip possibilities: {total_combinations}")
        for combo in itertools.product([False, True], repeat=num_skips):
            settings = {f"skip_{i}_{j}": val for (i, j), val in zip(pairs, combo)}
            print(settings)

    # Validate merge mode
    if merge_mode not in ("concat", "add"):
        raise ValueError(f"Unknown merge_mode '{merge_mode}'. Use 'concat' or 'add'.")

    if strategy == "final":
        # Apply skip connections to the final layer only
        selected = []
        for i in range(last_idx):
            # Sample whether to include skip from layer i to the final layer
            include = trial.suggest_categorical(f"skip_{i}_{last_idx}", [False, True])
            if include:
                selected.append(layers_list[i])  # Add selected source tensor

        # If no skips are selected, return the unmodified final layer
        if not selected:
            return layers_list[-1]

        # Always include the final layer itself in the merge
        selected.append(layers_list[-1])

        # Merge selected tensors
        if merge_mode == "concat":
            return layers.Concatenate(axis=axis_to_concat, name="skip_concat_final")(selected)
        else:  # merge_mode == "add"
            return layers.Add(name="skip_add_final")(selected)

    # strategy == "any": allow skips to all layers, not just the final one
    updated = list(layers_list)  # Copy of the original list to be modified
    for j in range(1, N):
        sources = []
        for i in range(0, j):
            # Sample whether to include skip from layer i to layer j
            include = trial.suggest_categorical(f"skip_{i}_{j}", [False, True])
            if include:
                sources.append(updated[i])

        if not sources:
            continue  # Skip merging if no skip sources selected

        # Always include the layer j itself
        sources.append(updated[j])

        # Merge sources into a new tensor and replace layer j with merged result
        if merge_mode == "concat":
            updated[j] = layers.Concatenate(axis=axis_to_concat, name=f"skip_concat_{j}")(sources)
        else:  # merge_mode == "add"
            updated[j] = layers.Add(name=f"skip_add_{j}")(sources)

    # Return the last layer in the modified list
    return updated[-1]
