# API Documentation

## ml.model.builders.cnn

### build_cnn1d
```python
build_cnn1d(
    trial,
    kparams,
    x,
    filters_range,
    kernel_size_range,
    filters_step,
    kernel_size_step,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    strides,
    dilation_rate,
    groups,
    use_bias,
    padding,
    data_format,
    kernel_initializer,
    bias_initializer,
    name_prefix,
)
```
Builds a 1D convolutional layer with optional hyperparameter tuning and regularization.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| filters_range | `Union[int, tuple[int, int]]` |  |
| kernel_size_range | `Union[int, tuple[int, int]]` |  |
| filters_step | `int` |  |
| kernel_size_step | `int` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| strides | `int` |  |
| dilation_rate | `int` |  |
| groups | `int` |  |
| use_bias | `bool` |  |
| padding | `str` |  |
| data_format | `str` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_dense_as_conv1d
```python
build_dense_as_conv1d(
    trial,
    kparams,
    x,
    filters_range,
    filters_step,
    padding,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Simulate a Dense layer using a Conv1D with kernel_size=1.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| filters_range | `int` |  |
| filters_step | `int` |  |
| padding | `str` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_cnn2d
```python
build_cnn2d(
    trial,
    kparams,
    x,
    filters_range,
    kernel_size_range,
    filters_step,
    kernel_size_step,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    strides,
    dilation_rate,
    groups,
    use_bias,
    padding,
    data_format,
    kernel_initializer,
    bias_initializer,
    name_prefix,
)
```
Builds a 2D convolutional layer with optional hyperparameter tuning and regularization.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| filters_range | `Union[int, tuple[int, int]]` |  |
| kernel_size_range | `Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]` |  |
| filters_step | `int` |  |
| kernel_size_step | `int` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| strides | `tuple[int, int]` |  |
| dilation_rate | `tuple[int, int]` |  |
| groups | `int` |  |
| use_bias | `bool` |  |
| padding | `str` |  |
| data_format | `str` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_dense_as_conv2d
```python
build_dense_as_conv2d(
    trial,
    kparams,
    x,
    filters_range,
    filters_step,
    padding,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Simulate a Dense layer using a Conv2D with kernel_size=(1, 1).

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| filters_range | `int` |  |
| filters_step | `int` |  |
| padding | `str` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_cnn3d
```python
build_cnn3d(
    trial,
    kparams,
    x,
    filters_range,
    kernel_size_range,
    filters_step,
    kernel_size_step,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    strides,
    dilation_rate,
    groups,
    use_bias,
    padding,
    data_format,
    kernel_initializer,
    bias_initializer,
    name_prefix,
)
```
Builds a 3D convolutional layer with optional hyperparameter tuning and regularization.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| filters_range | `Union[int, Tuple[int, int]]` |  |
| kernel_size_range | `Union[
        Tuple[int, int, int],
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ]` |  |
| filters_step | `int` |  |
| kernel_size_step | `int` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| strides | `Tuple[int, int, int]` |  |
| dilation_rate | `Tuple[int, int, int]` |  |
| groups | `int` |  |
| use_bias | `bool` |  |
| padding | `str` |  |
| data_format | `str` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_dense_as_conv3d
```python
build_dense_as_conv3d(
    trial,
    kparams,
    x,
    filters_range,
    filters_step,
    padding,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Simulate a Dense layer using a Conv3D with kernel_size=(1, 1, 1).

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| filters_range | `int` |  |
| filters_step | `int` |  |
| padding | `str` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.builders.tcnn

### build_tcnn1d
```python
build_tcnn1d(
    trial,
    kparams,
    x,
    filters_range,
    filters_step,
    kernel_size_range,
    kernel_size_step,
    data_format,
    padding,
    strides,
    dilation_rate,
    use_bias,
    kernel_initializer,
    bias_initializer,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Builds a single 1D transposed convolution block with optional batch norm and activation.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| filters_range | `Union[int, tuple[int, int]]` |  |
| filters_step | `int` |  |
| kernel_size_range | `Union[int, tuple[int, int]]` |  |
| kernel_size_step | `int` |  |
| data_format | `str` |  |
| padding | `str` |  |
| strides | `int` |  |
| dilation_rate | `int` |  |
| use_bias | `bool` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_tcnn2d
```python
build_tcnn2d(
    trial,
    kparams,
    x,
    filters_range,
    filters_step,
    kernel_size_range,
    kernel_size_step,
    data_format,
    padding,
    strides,
    dilation_rate,
    use_bias,
    kernel_initializer,
    bias_initializer,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Builds a single 2D transposed convolution block with optional batch norm and activation.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| filters_range | `Union[int, tuple[int, int]]` |  |
| filters_step | `int` |  |
| kernel_size_range | `Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]` |  |
| kernel_size_step | `int` |  |
| data_format | `str` |  |
| padding | `str` |  |
| strides | `tuple[int, int]` |  |
| dilation_rate | `tuple[int, int]` |  |
| use_bias | `bool` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_tcnn3d
```python
build_tcnn3d(
    trial,
    kparams,
    x,
    filters_range,
    filters_step,
    kernel_size_range,
    kernel_size_step,
    data_format,
    padding,
    strides,
    dilation_rate,
    use_bias,
    kernel_initializer,
    bias_initializer,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Builds a single 3D transposed convolution block with optional batch norm and activation.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| filters_range | `Union[int, Tuple[int, int]]` |  |
| filters_step | `int` |  |
| kernel_size_range | `Union[
        Tuple[int, int, int],
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ]` |  |
| kernel_size_step | `int` |  |
| data_format | `str` |  |
| padding | `str` |  |
| strides | `Tuple[int, int, int]` |  |
| dilation_rate | `Tuple[int, int, int]` |  |
| use_bias | `bool` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.builders.dnn

### build_dnn
```python
build_dnn(
    trial,
    kparams,
    x,
    units_range,
    dropout_rate_range,
    units_step,
    dropout_rate_step,
    kernel_initializer,
    bias_initializer,
    use_bias,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Builds a single dense neural network (DNN) block with optional regularization, batch normalization, and dropout.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| units_range | `Union[int, tuple[int, int]]` |  |
| dropout_rate_range | `Union[float, tuple[float, float]]` |  |
| units_step | `int` |  |
| dropout_rate_step | `float` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| use_bias | `bool` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.builders.gnn

### build_grid_adjacency
```python
build_grid_adjacency(
    rows,
    cols,
)
```
Build a grid adjacency matrix with GCN normalization.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| rows | `int` |  |
| cols | `int` |  |

**Returns**
`tf.sparse.SparseTensor` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_knn_adjacency
```python
build_knn_adjacency(
    rows,
    cols,
    k,
)
```
Construct a k-nearest neighbour adjacency matrix on a 2-D grid.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| rows | `int` |  |
| cols | `int` |  |
| k | `int` |  |

**Returns**
`tf.sparse.SparseTensor` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_gcn
```python
build_gcn(
    trial,
    kparams,
    x,
    a_graph,
    units_range,
    dropout_rate_range,
    units_step,
    dropout_rate_step,
    kernel_initializer,
    bias_initializer,
    use_bias,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Build a single Graph Convolutional Network (GCN) layer.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| a_graph | `tf.sparse.SparseTensor` |  |
| units_range | `Union[int, Tuple[int, int]]` |  |
| dropout_rate_range | `Union[float, Tuple[float, float]]` |  |
| units_step | `int` |  |
| dropout_rate_step | `float` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| use_bias | `bool` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_gat
```python
build_gat(
    trial,
    kparams,
    x,
    a_graph,
    units_range,
    dropout_rate_range,
    heads_range,
    units_step,
    dropout_rate_step,
    heads_step,
    concat_heads,
    kernel_initializer,
    bias_initializer,
    use_bias,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Build a single Graph Attention (GAT) layer.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| a_graph | `tf.sparse.SparseTensor` |  |
| units_range | `Union[int, Tuple[int, int]]` |  |
| dropout_rate_range | `Union[float, Tuple[float, float]]` |  |
| heads_range | `Union[int, Tuple[int, int]]` |  |
| units_step | `int` |  |
| dropout_rate_step | `float` |  |
| heads_step | `int` |  |
| concat_heads | `bool` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| use_bias | `bool` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### build_cheb
```python
build_cheb(
    trial,
    kparams,
    x,
    a_graph,
    units_range,
    dropout_rate_range,
    K_range,
    units_step,
    dropout_rate_step,
    K_step,
    kernel_initializer,
    bias_initializer,
    use_bias,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Build a single Chebyshev graph convolution layer.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| a_graph | `tf.sparse.SparseTensor` |  |
| units_range | `Union[int, Tuple[int, int]]` |  |
| dropout_rate_range | `Union[float, Tuple[float, float]]` |  |
| K_range | `Union[int, Tuple[int, int]]` |  |
| units_step | `int` |  |
| dropout_rate_step | `float` |  |
| K_step | `int` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| use_bias | `bool` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.builders.lstm

### build_lstm
```python
build_lstm(
    trial,
    kparams,
    x,
    return_sequences,
    units_range,
    units_step,
    dropout_rate_range,
    dropout_rate_step,
    kernel_initializer,
    bias_initializer,
    use_bias,
    use_batch_norm,
    trial_kernel_reg,
    trial_bias_reg,
    trial_activity_reg,
    name_prefix,
)
```
Builds a single LSTM block with optional regularization, batch normalization, and dynamic dropout.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| return_sequences | `bool` |  |
| units_range | `Union[int, tuple[int, int]]` |  |
| units_step | `int` |  |
| dropout_rate_range | `Union[float, tuple[float, float]]` |  |
| dropout_rate_step | `float` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| use_bias | `bool` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.builders.se

### build_squeeze_excite_1d
```python
build_squeeze_excite_1d(
    x,
    trial,
    kparams,
    ratio_choices,
    name_prefix,
)
```
Apply a Squeeze-and-Excitation (SE) block 1D with Optuna-tuned hyperparameters.
Based on the paper: https://arxiv.org/pdf/1709.01507

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| x | `tf.keras.layers.Layer` |  |
| trial | `optuna.Trial` |  |
| kparams | `KParams` |  |
| ratio_choices | `List[int]` |  |
| name_prefix | `str` |  |

**Returns**
`tf.keras.layers.Layer` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.builders.skip

### trial_skip_connections
```python
trial_skip_connections(
    trial,
    layers_list,
    axis_to_concat,
    print_combinations,
    strategy,
    merge_mode,
)
```
Constructs conditional skip connections between layers based on Optuna trial choices.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.trial.Trial` |  |
| layers_list | `Sequence[tf.Tensor]` |  |
| axis_to_concat | `int` |  |
| print_combinations | `bool` |  |
| strategy | `str` |  |
| merge_mode | `str` |  |

**Returns**
`tf.Tensor` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.callbacks

### get_callbacks_model
```python
get_callbacks_model(
    backup_dir,
    tensorboard_logs,
)
```
Constructs and returns a list of Keras callbacks for model training.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| backup_dir | `str` |  |
| tensorboard_logs | `str` |  |

**Returns**
`List[tf.keras.callbacks.Callback]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.hyperparams

### KParams
```python
class KParams
```
Container for hyperparameter search spaces.

Container for hyperparameter search spaces.

## ml.model.stats

### get_flops
```python
get_flops(
    model,
    batch_size,
)
```
Calculates the total number of floating-point operations (FLOPs) needed
to perform a single forward pass of the given Keras model.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` |  |
| batch_size | `int` |  |

**Returns**
`int` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### get_macs
```python
get_macs(
    model,
    batch_size,
)
```
Estimates the number of Multiply-Accumulate operations (MACs) required
for a single forward pass of the model. Assumes 1 MAC = 2 FLOPs.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` |  |
| batch_size | `int` |  |

**Returns**
`int` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### get_memory_and_time
```python
get_memory_and_time(
    model,
    batch_size,
    device,
    warmup_runs,
    test_runs,
    verbose,
)
```
Measures the peak memory usage and average inference time of a Keras model
on GPU or CPU.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` |  |
| batch_size | `int` |  |
| device | `int` |  |
| warmup_runs | `int` |  |
| test_runs | `int` |  |
| verbose | `bool` |  |

**Returns**
`Tuple[int, float]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### get_model_usage_stats
```python
get_model_usage_stats(
    saved_model,
    n_trials,
    device,
    rapl_path,
    verbose,
)
```
Estimate average power draw and energy usage.
Careful with the RAPL path; it may vary by system.
The RAPL interface is typically found at:
    $ ls /sys/class/powercap
    intel-rapl
    $ ls /sys/class/powercap/intel-rapl
    intel-rapl:0       intel-rapl:0:0    intel-rapl:1    …
    $ ls /sys/class/powercap/intel-rapl/intel-rapl:0
    energy_uj  max_energy_range_uj  name
Also, you MUST run this on a linux system with Intel CPUs!!!!!
And run the python script with SUDO to access RAPL files.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| saved_model | `str | tf.keras.Model` |  |
| n_trials | `int` |  |
| device | `int` |  |
| rapl_path | `str` |  |
| verbose | `bool` |  |

**Returns**
`Tuple[float, float, float]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### write_model_stats_to_file
```python
write_model_stats_to_file(
    model,
    file_path,
    bits_per_param,
    batch_size,
    device,
    n_trials,
    extra_attrs,
    verbose,
)
```
Write model statistics to a file.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` |  |
| file_path | `str` |  |
| bits_per_param | `int` |  |
| batch_size | `int` |  |
| device | `int` |  |
| n_trials | `int` |  |
| extra_attrs | `Optional[List[str]]` |  |
| verbose | `bool` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.tools

### convert_to_saved_model
```python
convert_to_saved_model(
    input_keras_path,
    output_zip_path,
)
```
Convert a Keras `.keras` model archive into a zipped TensorFlow SavedModel.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| input_keras_path | `str` |  |
| output_zip_path | `str` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### punish_model_flops
```python
punish_model_flops(
    target,
    model,
    penalty_factor,
    direction,
)
```
Penalize an objective according to the model's FLOPs.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| target | `Union[float, Sequence[float]]` |  |
| model | `tf.keras.Model` |  |
| penalty_factor | `float` |  |
| direction | `Literal["minimize", "maximize"]` |  |

**Returns**
`Union[float, Sequence[float]]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### punish_model_params
```python
punish_model_params(
    target,
    model,
    penalty_factor,
    direction,
)
```
Penalize an objective according to the model's parameter count.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| target | `Union[float, Sequence[float]]` |  |
| model | `tf.keras.Model` |  |
| penalty_factor | `float` |  |
| direction | `Literal["minimize", "maximize"]` |  |

**Returns**
`Union[float, Sequence[float]]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### punish_model
```python
punish_model(
    target,
    model,
    type,
    flops_penalty_factor,
    params_penalty_factor,
    direction,
)
```
Apply both FLOPs and parameter penalties to an objective.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| target | `Union[float, Sequence[float]]` |  |
| model | `tf.keras.Model` |  |
| type | `Literal["flops", "params", None]` |  |
| flops_penalty_factor | `float` |  |
| params_penalty_factor | `float` |  |
| direction | `Literal["minimize", "maximize"]` |  |

**Returns**
`Union[float, Sequence[float]]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.model.utils

### capture_model_summary
```python
capture_model_summary(
    model,
)
```
Capture model summary as a string.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `Any` |  |

**Returns**
`Any` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.optuna.analyzer

### PlotConfig
```python
class PlotConfig
```
Global configuration for matplotlib plots used in this module.

Global configuration for matplotlib plots used in this module.

### set_plot_config_param
```python
set_plot_config_param(
    param_name,
    value,
)
```
Set a single parameter in :data:`PLOT_CFG`.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| param_name | `str` |  |
| value | `Any` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### set_plot_config_params
```python
set_plot_config_params(
    ,
)
```
Set multiple parameters in :data:`PLOT_CFG`.

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### analyze_study
```python
analyze_study(
    study,
    table_dir,
    top_frac,
    param_name_mapping,
    create_standalone,
    save_data,
    create_plotly,
    plots,
)
```
Comprehensive analysis of Optuna hyperparameter optimization study results.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| table_dir | `str` |  |
| top_frac | `float` |  |
| param_name_mapping | `Dict[str, str]` |  |
| create_standalone | `bool` |  |
| save_data | `bool` |  |
| create_plotly | `bool` |  |
| plots | `Optional[List[str]]` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.optuna.callbacks

### ImprovementStagnation
```python
class ImprovementStagnation
```
Stop a study when the terminator improvement variance plateaus.

Stop a study when the terminator improvement variance plateaus.

After each completed trial the callback computes the potential future
improvement using an ``optuna.terminator`` improvement evaluator. The
variance of the most recent ``window_size`` improvement values is measured
and if it falls below ``variance_threshold`` after ``min_n_trials`` trials
the study is terminated via :meth:`optuna.Study.stop`.

Parameters
----------
min_n_trials:
    Minimum number of completed trials before starting variance checks.
window_size:
    Number of recent improvement values used to compute the variance.
variance_threshold:
    Threshold below which the variance of improvements indicates stagnation.
improvement_evaluator:
    Custom improvement evaluator. Defaults to :class:`RegretBoundEvaluator`.

### StopIfKeepBeingPruned
```python
class StopIfKeepBeingPruned
```
A callback for Optuna studies that stops the optimization process
when a specified number of consecutive trials are pruned.

A callback for Optuna studies that stops the optimization process
when a specified number of consecutive trials are pruned.

Args:
    threshold (int): The number of consecutive pruned trials required to stop the study.

### NanLossPrunerOptuna
```python
class NanLossPrunerOptuna
```
A custom Keras callback that prunes an Optuna trial if NaN is encountered in training loss.

A custom Keras callback that prunes an Optuna trial if NaN is encountered in training loss.

This is useful for skipping unpromising model configurations early, especially
those that are unstable or diverging during training.

Args:
    trial (optuna.Trial): The Optuna trial associated with this model run.

Example:
    model.fit(..., callbacks=[NanLossPrunerOptuna(trial)])

### get_callbacks_study
```python
get_callbacks_study(
    trial,
    tensorboard_logs,
    monitor,
)
```
Constructs and returns a list of Keras callbacks tailored for Optuna trials.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` |  |
| tensorboard_logs | `str` |  |
| monitor | `str` |  |

**Returns**
`List[tf.keras.callbacks.Callback]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.optuna.model_tools

### estimate_training_memory
```python
estimate_training_memory(
    model,
    batch_size,
)
```
Estimate total VRAM needed for training a Keras model in bytes.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `keras.Model` |  |
| batch_size | `int` |  |

**Returns**
`int` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### plot_model_param_distribution
```python
plot_model_param_distribution(
    build_model_fn,
    bits_per_param,
    batch_size,
    n_trials,
)
```
Sample random models and plot parameter and size histograms.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| build_model_fn | `Callable[[optuna.Trial], tf.keras.Model]` |  |
| bits_per_param | `int` |  |
| batch_size | `int` |  |
| n_trials | `int` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### set_user_attr_model_stats
```python
set_user_attr_model_stats(
    trial,
    model,
    bits_per_param,
    batch_size,
    n_trials,
    device,
    verbose,
)
```
Extract and return model statistics from the given Optuna trial.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` |  |
| model | `tf.keras.Model` |  |
| bits_per_param | `int` |  |
| batch_size | `int` |  |
| n_trials | `int` |  |
| device | `int` |  |
| verbose | `bool` |  |

**Returns**
`Dict[str, float]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## ml.optuna.utils

### cleanup_non_top_trials
```python
cleanup_non_top_trials(
    all_trial_ids,
    top_trial_ids,
    cleanup_paths,
)
```
Remove files or directories for trials not in the top-K set.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| all_trial_ids | `Set[int]` |  |
| top_trial_ids | `Set[int]` |  |
| cleanup_paths | `List[Tuple[str, str]]` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### get_remaining_trials
```python
get_remaining_trials(
    study,
    num_trials,
)
```
Returns a list of completed trials from the given Optuna study.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| num_trials | `int` |  |

**Returns**
`list[optuna.trial.FrozenTrial]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### get_top_trials
```python
get_top_trials(
    study,
    top_k,
    rank_key,
    order,
)
```
Get the top-K trials from an Optuna study based on ranking criteria.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| top_k | `int` |  |
| rank_key | `str` |  |
| order | `str` |  |

**Returns**
`List[optuna.Trial]` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### rename_top_k_files
```python
rename_top_k_files(
    top_trials,
    file_configs,
)
```
Rename top-K trial files with ranking prefix.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| top_trials | `List[optuna.Trial]` |  |
| file_configs | `List[Tuple[str, str]]` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### save_trial_params_to_file
```python
save_trial_params_to_file(
    filepath,
    params,
)
```
Save Optuna trial parameters and associated metadata to a text file.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| filepath | `str` |  |
| params | `dict[str, float]` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### save_top_k_trials
```python
save_top_k_trials(
    top_trials,
    args_dir,
    study,
    extra_attrs,
)
```
Save top-K trials to text files.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| top_trials | `List[optuna.Trial]` |  |
| args_dir | `str` |  |
| study | `optuna.Study` |  |
| extra_attrs | `Optional[List[str]]` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### init_study_dirs
```python
init_study_dirs(
    run_dir,
    study_name,
    subdirs,
)
```
Create and return study directory structure for experiments.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| run_dir | `Any` |  |
| study_name | `Any` |  |
| subdirs | `Any` |  |

**Returns**
`Any` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## notifications.email

### send_email
```python
send_email(
    subject,
    body,
    recipients_file,
    credentials_file,
    text_type,
    smtp_server,
    smtp_port,
)
```
Sends an email notification with the specified subject and body content to multiple recipients.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| subject | `str` |  |
| body | `str` |  |
| recipients_file | `str` |  |
| credentials_file | `str` |  |
| text_type | `str` |  |
| smtp_server | `str` |  |
| smtp_port | `int` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## runtime.monitoring

### run_auto_restart
```python
run_auto_restart(
    file_path,
    success_flag_file,
    title,
    max_restarts,
    restart_delay,
    recipients_file,
    credentials_file,
    restart_after_delay,
    retry_attempts,
    supress_tf_warnings,
    resource_usage_log_file,
)
```
Main function with notebook conversion, file cleanup, and consolidated email notification support.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| file_path | `str` |  |
| success_flag_file | `str` |  |
| title | `Optional[str]` |  |
| max_restarts | `int` |  |
| restart_delay | `float` |  |
| recipients_file | `Optional[str]` |  |
| credentials_file | `Optional[str]` |  |
| restart_after_delay | `Optional[float]` |  |
| retry_attempts | `int` |  |
| supress_tf_warnings | `bool` |  |
| resource_usage_log_file | `Optional[str]` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## utils.io

### create_run_directory
```python
create_run_directory(
    prefix,
    base_dir,
)
```
Creates a new run directory with an incremented numeric suffix and returns its full path.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| prefix | `str` |  |
| base_dir | `str` |  |

**Returns**
`str` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## utils.misc

### clear
```python
clear(
    ,
)
```
Clear all prints from terminal or notebook cell.

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`Any` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### format_number
```python
format_number(
    number,
    precision,
)
```
Format a number using scientific suffixes.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| number | `Any` |  |
| precision | `Any` |  |

**Returns**
`Any` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### format_bytes
```python
format_bytes(
    bytes_value,
    precision,
)
```
Format bytes using binary suffixes (B, KB, MB, GB, etc.).

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| bytes_value | `Any` |  |
| precision | `Any` |  |

**Returns**
`Any` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### format_scientific
```python
format_scientific(
    number,
    max_precision,
)
```
Format to scientific notation with automatic precision based on number magnitude.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| number | `Any` |  |
| max_precision | `Any` |  |

**Returns**
`Any` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### format_number_commas
```python
format_number_commas(
    number,
    precision,
)
```
Format a number with commas as thousands separators.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| number | `Any` |  |
| precision | `Any` |  |

**Returns**
`Any` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## utils.system

### get_user_gpu_choice
```python
get_user_gpu_choice(
    ,
)
```
Prompts the user to select a GPU index and validates the input.

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`Any` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### get_gpu_info
```python
get_gpu_info(
    ,
)
```
Prints detailed TensorFlow and GPU configuration information in nvidia-smi style format.

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### gpu_summary
```python
gpu_summary(
    ,
)
```
Prints a compact GPU summary similar to nvidia-smi output.

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

### log_resources
```python
log_resources(
    log_dir,
    interval,
)
```
Logs selected system and ML resources (CPU, RAM, GPU, CUDA, TensorFlow) at regular time intervals.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| log_dir | `str` |  |
| interval | `int` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.

## visualization.configs

### config_plt
```python
config_plt(
    style,
)
```
Configure matplotlib rcParams for IEEE‑style figures

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| style | `str` |  |

**Returns**
`None` – return value.

**Raises**
- None

> [!NOTE]
> Auto-generated documentation.
