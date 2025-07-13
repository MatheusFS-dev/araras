# API Documentation

## commons

### make_logger
```python
make_logger(name: str, fmt: str, datefmt: str | None, level: int)  [source]
```
Create a logger with its own StreamHandler and ColorFormatter.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| name | `str` |  |
| fmt | `str` |  |
| datefmt | `str | None` |  |
| level | `int` |  |

**Returns**

`logging.Logger` – 


**Raises**

- None

**Examples**

```python
result = make_logger(...)
```

## email.utils

### get_credentials
```python
get_credentials(file_path: str)  [source]
```
Reads the sender's email and password from a JSON file.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| file_path | `str` | Path to the credentials JSON file. |

**Returns**

`tuple[str, str]` – 


**Raises**

- `ValueError` – If the credentials cannot be read or parsed.

**Examples**

```python
result = get_credentials(...)
```

### get_recipient_emails
```python
get_recipient_emails(file_path: str)  [source]
```
Reads a list of recipient email addresses from a JSON file.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| file_path | `str` | Path to the recipient JSON file. |

**Returns**

`list[str]` – 


**Raises**

- `ValueError` – If the file or its contents cannot be read.

**Examples**

```python
result = get_recipient_emails(...)
```

### send_email
```python
send_email(subject: str, body: str, recipients_file: str, credentials_file: str, text_type: str, smtp_server: str, smtp_port: int)  [source]
```
Sends an email notification with the specified subject and body content to multiple recipients.
> [!CAUTION]
> Requires valid SMTP credentials and network access.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| subject | `str` | The subject of the email. |
| body | `str` | The main content of the email. |
| recipients_file | `str` | Path to the recipients JSON file. |
| credentials_file | `str` | Path to the credentials JSON file. |
| text_type | `str` | The type of text content (e.g., "plain" or "html"). |
| smtp_server | `str` | The SMTP server address (default is Gmail's SMTP server). |
| smtp_port | `int` | The port number for the SMTP server (default is 587 for TLS). |

**Returns**

`None` – None


**Raises**

- None

**Examples**

```python
result = send_email(...)
```

## keras.analysis.estimator

### get_model_trainable_params
```python
get_model_trainable_params(model: keras.Model)  [source]
```
Get number of trainable parameters in the model.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| model | `keras.Model` |  |

**Returns**

`int` – 


**Raises**

- None

**Examples**

```python
result = get_model_trainable_params(...)
```

### get_precision_bytes
```python
get_precision_bytes(model: keras.Model)  [source]
```
Determine bytes per parameter based on model's actual dtype.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| model | `keras.Model` |  |

**Returns**

`int` – 


**Raises**

- None

**Examples**

```python
result = get_precision_bytes(...)
```

### get_optimizer_state_factor
```python
get_optimizer_state_factor(model: keras.Model)  [source]
```
Determine optimizer state factor from compiled model.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| model | `keras.Model` |  |

**Returns**

`int` – 


**Raises**

- None

**Examples**

```python
result = get_optimizer_state_factor(...)
```

### calculate_activation_memory
```python
calculate_activation_memory(model: keras.Model, bytes_per_param: int)  [source]
```
Calculate activation memory needed during forward/backward pass.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| model | `keras.Model` |  |
| bytes_per_param | `int` |  |

**Returns**

`int` – 


**Raises**

- None

**Examples**

```python
result = calculate_activation_memory(...)
```

### get_framework_overhead
```python
get_framework_overhead()  [source]
```
Calculate framework overhead based on available GPU memory.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`int` – 


**Raises**

- None

**Examples**

```python
result = get_framework_overhead(...)
```

### estimate_training_memory
```python
estimate_training_memory(model: keras.Model, batch_size: int)  [source]
```
Estimate total VRAM needed for training a Keras model in bytes.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| model | `keras.Model` | Keras model object |
| batch_size | `int` | Training batch size |

**Returns**

`int` – Total memory needed in bytes


**Raises**

- None

**Examples**

```python
result = estimate_training_memory(...)
```

### model_param_distribution
```python
model_param_distribution(build_model_fn: Callable[[optuna.Trial], tf.keras.Model], bits_per_param: int, batch_size: int, n_trials: int)  [source]
```
Sample random models and plot parameter and size histograms.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| build_model_fn | `Callable[[optuna.Trial]` | Function that builds a Keras model given an Optuna |
| tf.keras.Model] | `Any` |  |
| bits_per_param | `int` | Number of bits used to store each parameter. |
| batch_size | `int` |  |
| n_trials | `int` | Number of random trials to run. |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = model_param_distribution(...)
```

## keras.analysis.profiler

### get_flops
```python
get_flops(model: tf.keras.Model, batch_size: int)  [source]
```
Calculates the total number of floating-point operations (FLOPs) needed

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` | The Keras model to analyze. |
| batch_size | `int` | The batch size to simulate for input. Defaults to 1. |

**Returns**

`int` – int: The total number of floating-point operations (FLOPs) for one forward pass.


**Raises**

- None

**Examples**

```python
result = get_flops(...)
```

### get_macs
```python
get_macs(model: tf.keras.Model, batch_size: int)  [source]
```
Estimates the number of Multiply-Accumulate operations (MACs) required

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` | The Keras model to analyze. |
| batch_size | `int` | The batch size to simulate for input. Defaults to 1. |

**Returns**

`int` – int: The estimated number of MACs for one forward pass.


**Raises**

- None

**Examples**

```python
result = get_macs(...)
```

### get_memory_and_time
```python
get_memory_and_time(model: tf.keras.Model, batch_size: int, device: int, warmup_runs: int, test_runs: int, verbose: bool)  [source]
```
Measures the peak memory usage and average inference time of a Keras model

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` | The Keras model to analyze. |
| batch_size | `int` | The batch size to simulate for input. Defaults to 1. |
| device | `int` | GPU index to run the model on. Use ``-1`` to run on CPU. |
| warmup_runs | `int` | Number of warm-up runs before timing. Defaults to 10. |
| test_runs | `int` | Number of runs to measure average inference time. Defaults to 50. |
| verbose | `bool` | If True, displays a progress bar during test runs. |

**Returns**

`Tuple[int, float]` – Tuple[int, float]: - peak memory usage in bytes (0 if CPU measurement fails after several attempts) - average inference time in seconds


**Raises**

- None

**Examples**

```python
result = get_memory_and_time(...)
```

## keras.builders.cnn

### build_cnn1d
```python
build_cnn1d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, tuple[int, int]], kernel_size_range: Union[int, tuple[int, int]], filters_step: int, kernel_size_step: int, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, strides: int, dilation_rate: int, groups: int, use_bias: bool, padding: str, data_format: str, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, name_prefix: str)  [source]
```
Builds a 1D convolutional layer with optional hyperparameter tuning and regularization.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | An Optuna trial object used for hyperparameter optimization. |
| kparams | `KParams` | A utility object to provide regularizers and activations. |
| x | `layers.Layer` | The input Keras layer. |
| filters_range | `Union[int` | Number of filters or a range for tuning. |
| tuple[int | `Any` |  |
| int]] | `Any` |  |
| kernel_size_range | `Union[int` | Kernel size or a range for tuning. |
| tuple[int | `Any` |  |
| int]] | `Any` |  |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_step | `int` | Step size for kernel size tuning. |
| use_batch_norm | `bool` | Whether to include batch normalization. |
| trial_kernel_reg | `bool` | Whether to tune and apply kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune and apply bias regularization. |
| trial_activity_reg | `bool` | Whether to tune and apply activity regularization. |
| strides | `int` | Stride size for the convolution. |
| dilation_rate | `int` | Dilation rate for convolution. |
| groups | `int` | Number of filter groups. |
| use_bias | `bool` | Whether to use a bias term in the convolution. If using batch norm, this can be set to False. |
| padding | `str` | Padding method ('valid' or 'same'). |
| data_format | `str` | Data format, either 'channels_last' or 'channels_first'. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**

`layers.Layer` – layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.


**Raises**

- None

**Examples**

```python
result = build_cnn1d(...)
```

### build_dense_as_conv1d
```python
build_dense_as_conv1d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: int, filters_step: int, padding: str, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Simulate a Dense layer using a Conv1D with kernel_size=1.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | An Optuna trial object used for hyperparameter optimization. |
| kparams | `KParams` | A utility object to provide regularizers and activations. |
| x | `layers.Layer` | The input Keras layer, expected to be of shape (batch_size, length, features_in). |
| filters_range | `int` | The number of output filters for the Conv1D layer. |
| filters_step | `int` | Step size for tuning the number of filters. |
| padding | `str` | Padding method ('valid' or 'same'). |
| trial_kernel_reg | `bool` | Whether to tune and apply kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune and apply bias regularization. |
| trial_activity_reg | `bool` | Whether to tune and apply activity regularization. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**

`layers.Layer` – layers.Layer: A Keras layer with output shape (batch_size, 1, units), equivalent to Dense(units). References: https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks


**Raises**

- None

**Examples**

```python
result = build_dense_as_conv1d(...)
```

### build_cnn2d
```python
build_cnn2d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, tuple[int, int]], kernel_size_range: Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]], filters_step: int, kernel_size_step: int, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, strides: tuple[int, int], dilation_rate: tuple[int, int], groups: int, use_bias: bool, padding: str, data_format: str, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, name_prefix: str)  [source]
```
Builds a 2D convolutional layer with optional hyperparameter tuning and regularization.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | An Optuna trial object used for hyperparameter optimization. |
| kparams | `KParams` | A utility object to provide regularizers and activations. |
| x | `layers.Layer` | The input Keras layer. |
| filters_range | `Union[int` | Number of filters or a range for tuning. |
| tuple[int | `Any` |  |
| int]] | `Any` |  |
| kernel_size_range | `Union[tuple[int` |  |
| int] | `Any` |  |
| tuple[tuple[int | `Any` |  |
| int] | `Any` |  |
| tuple[int | `Any` |  |
| int]]] | `Any` |  |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_step | `int` | Step size for kernel dimension tuning. |
| use_batch_norm | `bool` | Whether to include batch normalization. |
| trial_kernel_reg | `bool` | Whether to tune and apply kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune and apply bias regularization. |
| trial_activity_reg | `bool` | Whether to tune and apply activity regularization. |
| strides | `tuple[int` | Stride size for height and width. |
| int] | `Any` |  |
| dilation_rate | `tuple[int` | Dilation rate for height and width. |
| int] | `Any` |  |
| groups | `int` | Number of filter groups. |
| use_bias | `bool` | Whether to use a bias term in the convolution. If using batch norm, this can be set to False. |
| padding | `str` | Padding method ('valid' or 'same'). |
| data_format | `str` | Data format, either 'channels_last' or 'channels_first'. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**

`layers.Layer` – layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.


**Raises**

- None

**Examples**

```python
result = build_cnn2d(...)
```

### build_dense_as_conv2d
```python
build_dense_as_conv2d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: int, filters_step: int, padding: str, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Simulate a Dense layer using a Conv2D with kernel_size=(1, 1).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | An Optuna trial object used for hyperparameter optimization. |
| kparams | `KParams` | A utility object to provide regularizers and activations. |
| x | `layers.Layer` | The input Keras layer, expected to be of shape (batch_size, height, width, features_in). |
| filters_range | `int` | The number of output filters for the Conv2D layer. |
| filters_step | `int` | Step size for tuning the number of filters. |
| padding | `str` | Padding method ('valid' or 'same'). |
| trial_kernel_reg | `bool` | Whether to tune and apply kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune and apply bias regularization. |
| trial_activity_reg | `bool` | Whether to tune and apply activity regularization. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**

`layers.Layer` – layers.Layer: A Keras layer with output shape (batch_size, height, width, units), equivalent to Dense(units). References: https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks


**Raises**

- None

**Examples**

```python
result = build_dense_as_conv2d(...)
```

### build_cnn3d
```python
build_cnn3d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, Tuple[int, int]], kernel_size_range: Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], filters_step: int, kernel_size_step: int, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, strides: Tuple[int, int, int], dilation_rate: Tuple[int, int, int], groups: int, use_bias: bool, padding: str, data_format: str, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, name_prefix: str)  [source]
```
Builds a 3D convolutional layer with optional hyperparameter tuning and regularization.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | An Optuna trial object used for hyperparameter optimization. |
| kparams | `KParams` | A utility object to provide regularizers and activations. |
| x | `layers.Layer` | The input Keras layer. |
| filters_range | `Union[int` | Number of filters or a range for tuning. |
| Tuple[int | `Any` |  |
| int]] | `Any` |  |
| kernel_size_range | `Union[Tuple[int` |  |
| int | `Any` |  |
| int] | `Any` |  |
| Tuple[Tuple[int | `Any` |  |
| int] | `Any` |  |
| Tuple[int | `Any` |  |
| int] | `Any` |  |
| Tuple[int | `Any` |  |
| int]]] | `Any` |  |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_step | `int` | Step size for kernel dimension tuning. |
| use_batch_norm | `bool` | Whether to include batch normalization. |
| trial_kernel_reg | `bool` | Whether to tune and apply kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune and apply bias regularization. |
| trial_activity_reg | `bool` | Whether to tune and apply activity regularization. |
| strides | `Tuple[int` | Stride size for depth, height, and width. |
| int | `Any` |  |
| int] | `Any` |  |
| dilation_rate | `Tuple[int` | Dilation rate for depth, height, and width. |
| int | `Any` |  |
| int] | `Any` |  |
| groups | `int` | Number of filter groups. |
| use_bias | `bool` | Whether to use a bias term in the convolution. If using batch norm, this can be set to False. |
| padding | `str` | Padding method ('valid' or 'same'). |
| data_format | `str` | Data format, either 'channels_last' or 'channels_first'. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**

`layers.Layer` – layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.


**Raises**

- None

**Examples**

```python
result = build_cnn3d(...)
```

### build_dense_as_conv3d
```python
build_dense_as_conv3d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: int, filters_step: int, padding: str, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Simulate a Dense layer using a Conv3D with kernel_size=(1, 1, 1).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | An Optuna trial object used for hyperparameter optimization. |
| kparams | `KParams` | A utility object to provide regularizers and activations. |
| x | `layers.Layer` | The input Keras layer, expected to be of shape (batch_size, depth, height, width, features_in). |
| filters_range | `int` | The number of output filters for the Conv3D layer. |
| filters_step | `int` | Step size for tuning the number of filters. |
| padding | `str` | Padding method ('valid' or 'same'). |
| trial_kernel_reg | `bool` | Whether to tune and apply kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune and apply bias regularization. |
| trial_activity_reg | `bool` | Whether to tune and apply activity regularization. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**

`layers.Layer` – layers.Layer: A Keras layer with output shape (batch_size, depth, height, width, units), equivalent to Dense(units). References: https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks


**Raises**

- None

**Examples**

```python
result = build_dense_as_conv3d(...)
```

## keras.builders.dnn

### build_dnn
```python
build_dnn(trial: Any, kparams: KParams, x: layers.Layer, units_range: Union[int, tuple[int, int]], dropout_rate_range: Union[float, tuple[float, float]], units_step: int, dropout_rate_step: float, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Builds a single dense neural network (DNN) block with optional regularization, batch normalization, and dropout.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Hyperparameter tuning trial object, e.g., from Optuna. |
| kparams | `KParams` | Custom hyperparameter handler that provides regularizers and activations. |
| x | `layers.Layer` | Input tensor or layer to build on. |
| units_range | `Union[int` | Either a fixed unit count or a range for tuning. |
| tuple[int | `Any` |  |
| int]] | `Any` |  |
| dropout_rate_range | `Union[float` | Either a fixed dropout rate or a range. |
| tuple[float | `Any` |  |
| float]] | `Any` |  |
| units_step | `int` | Step size for unit range tuning. |
| dropout_rate_step | `float` | Step size for dropout rate tuning. |
| kernel_initializer | `initializers.Initializer` | Initializer for Dense layer weights. |
| bias_initializer | `initializers.Initializer` | Initializer for Dense layer biases. |
| use_bias | `bool` | Whether to include a bias term in the Dense layer. |
| use_batch_norm | `bool` | Whether to include a batch normalization layer. |
| trial_kernel_reg | `bool` | Whether to tune and apply a kernel regularizer. |
| trial_bias_reg | `bool` | Whether to tune and apply a bias regularizer. |
| trial_activity_reg | `bool` | Whether to tune and apply an activity regularizer. |
| name_prefix | `str` | Prefix to use for naming the layers. |

**Returns**

`layers.Layer` – 


**Raises**

- `None` – 

**Examples**

```python
result = build_dnn(...)
```

## keras.builders.gnn

### print_warning_jit
```python
print_warning_jit()  [source]
```
Print a warning about JIT compilation.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`Any` – 


**Raises**

- None

**Examples**

```python
result = print_warning_jit(...)
```

### build_grid_adjacency
```python
build_grid_adjacency(rows: int, cols: int)  [source]
```
Build a grid adjacency matrix with GCN normalization.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| rows | `int` | Number of grid rows. |
| cols | `int` | Number of grid columns. |

**Returns**

`tf.sparse.SparseTensor` – tf.sparse.SparseTensor: Normalized sparse adjacency matrix.


**Raises**

- None

**Examples**

```python
result = build_grid_adjacency(...)
```

### build_knn_adjacency
```python
build_knn_adjacency(rows: int, cols: int, k: int)  [source]
```
Construct a k-nearest neighbour adjacency matrix on a 2-D grid.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| rows | `int` | Number of grid rows. |
| cols | `int` | Number of grid columns. |
| k | `int` | Number of neighbours for each node. |

**Returns**

`tf.sparse.SparseTensor` – tf.sparse.SparseTensor: Normalized sparse adjacency matrix.


**Raises**

- None

**Examples**

```python
result = build_knn_adjacency(...)
```

### _select_range_value
```python
_select_range_value(trial: Any, name: str, value_range: Union[int, Tuple[int, int]], step: int)  [source]
```
Helper to pick an integer from a fixed value or an Optuna range.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| name | `str` |  |
| value_range | `Union[int` |  |
| Tuple[int | `Any` |  |
| int]] | `Any` |  |
| step | `int` |  |

**Returns**

`int` – 


**Raises**

- None

**Examples**

```python
result = _select_range_value(...)
```

### _select_float_range_value
```python
_select_float_range_value(trial: Any, name: str, value_range: Union[float, Tuple[float, float]], step: float)  [source]
```
Helper to pick a float from a fixed value or an Optuna range.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| name | `str` |  |
| value_range | `Union[float` |  |
| Tuple[float | `Any` |  |
| float]] | `Any` |  |
| step | `float` |  |

**Returns**

`float` – 


**Raises**

- None

**Examples**

```python
result = _select_float_range_value(...)
```

### build_gcn
```python
build_gcn(trial: Any, kparams: KParams, x: layers.Layer, a_graph: tf.sparse.SparseTensor, units_range: Union[int, Tuple[int, int]], dropout_rate_range: Union[float, Tuple[float, float]], units_step: int, dropout_rate_step: float, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Build a single Graph Convolutional Network (GCN) layer.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| a_graph | `tf.sparse.SparseTensor` |  |
| units_range | `Union[int` |  |
| Tuple[int | `Any` |  |
| int]] | `Any` |  |
| dropout_rate_range | `Union[float` |  |
| Tuple[float | `Any` |  |
| float]] | `Any` |  |
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

`layers.Layer` – 


**Raises**

- None

**Examples**

```python
result = build_gcn(...)
```

### build_gat
```python
build_gat(trial: Any, kparams: KParams, x: layers.Layer, a_graph: tf.sparse.SparseTensor, units_range: Union[int, Tuple[int, int]], dropout_rate_range: Union[float, Tuple[float, float]], heads_range: Union[int, Tuple[int, int]], units_step: int, dropout_rate_step: float, heads_step: int, concat_heads: bool, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Build a single Graph Attention (GAT) layer.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| a_graph | `tf.sparse.SparseTensor` |  |
| units_range | `Union[int` |  |
| Tuple[int | `Any` |  |
| int]] | `Any` |  |
| dropout_rate_range | `Union[float` |  |
| Tuple[float | `Any` |  |
| float]] | `Any` |  |
| heads_range | `Union[int` |  |
| Tuple[int | `Any` |  |
| int]] | `Any` |  |
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

`layers.Layer` – 


**Raises**

- None

**Examples**

```python
result = build_gat(...)
```

### build_cheb
```python
build_cheb(trial: Any, kparams: KParams, x: layers.Layer, a_graph: tf.sparse.SparseTensor, units_range: Union[int, Tuple[int, int]], dropout_rate_range: Union[float, Tuple[float, float]], K_range: Union[int, Tuple[int, int]], units_step: int, dropout_rate_step: float, K_step: int, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Build a single Chebyshev graph convolution layer.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` |  |
| a_graph | `tf.sparse.SparseTensor` |  |
| units_range | `Union[int` |  |
| Tuple[int | `Any` |  |
| int]] | `Any` |  |
| dropout_rate_range | `Union[float` |  |
| Tuple[float | `Any` |  |
| float]] | `Any` |  |
| K_range | `Union[int` |  |
| Tuple[int | `Any` |  |
| int]] | `Any` |  |
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

`layers.Layer` – 


**Raises**

- None

**Examples**

```python
result = build_cheb(...)
```

## keras.builders.lstm

### build_lstm
```python
build_lstm(trial: Any, kparams: KParams, x: layers.Layer, return_sequences: bool, units_range: Union[int, tuple[int, int]], units_step: int, dropout_rate_range: Union[float, tuple[float, float]], dropout_rate_step: float, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Builds a single LSTM block with optional regularization, batch normalization, and dynamic dropout.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Object used for suggesting hyperparameters, typically from a tuner like Optuna. |
| kparams | `KParams` | Hyperparameter manager used to retrieve regularizers and activations. |
| x | `layers.Layer` | Input tensor or layer. |
| return_sequences | `bool` | Whether to return the full sequence of outputs or just the last output. |
| units_range | `Union[int` | Fixed or tunable number of LSTM units. |
| tuple[int | `Any` |  |
| int]] | `Any` |  |
| units_step | `int` | Step size for tuning LSTM units if a range is given. |
| dropout_rate_range | `Union[float` | Fixed or tunable dropout rate. |
| tuple[float | `Any` |  |
| float]] | `Any` |  |
| dropout_rate_step | `float` | Step size for tuning dropout rate. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for biases. |
| use_bias | `bool` | Whether to include a bias term in the LSTM layer. |
| use_batch_norm | `bool` | Whether to apply batch normalization after LSTM. |
| trial_kernel_reg | `bool` | Whether to apply/tune a kernel regularizer. |
| trial_bias_reg | `bool` | Whether to apply/tune a bias regularizer. |
| trial_activity_reg | `bool` | Whether to apply/tune an activity regularizer. |
| name_prefix | `str` | Prefix to use for naming the layers. |

**Returns**

`layers.Layer` – 


**Raises**

- `None` – 

**Examples**

```python
result = build_lstm(...)
```

## keras.builders.se

### build_squeeze_excite_1d
```python
build_squeeze_excite_1d(x: tf.keras.layers.Layer, trial: optuna.Trial, kparams: KParams, ratio_choices: List[int], name_prefix: str)  [source]
```
Apply a Squeeze-and-Excitation (SE) block 1D with Optuna-tuned hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| x | `tf.keras.layers.Layer` | Input 3D tensor (batch, length, channels). |
| trial | `optuna.Trial` | Optuna Trial object for suggesting hyperparameters. |
| kparams | `KParams` | KParams object containing hyperparameter choices. |
| ratio_choices | `List[int]` | List of integers representing reduction ratios for SE block. |
| name_prefix | `str` | Prefix for naming layers and trial parameters. |

**Returns**

`tf.keras.layers.Layer` – 


**Raises**

- `ValueError` – If `x.shape[-1]` is None (undefined channel dimension).

**Examples**

```python
result = build_squeeze_excite_1d(...)
```

## keras.builders.skip

### trial_skip_connections
```python
trial_skip_connections(trial: optuna.trial.Trial, layers_list: Sequence[tf.Tensor], axis_to_concat: int, print_combinations: bool, strategy: str, merge_mode: str)  [source]
```
Constructs conditional skip connections between layers based on Optuna trial choices.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.trial.Trial` | Optuna trial object used to sample categorical decisions |
| layers_list | `Sequence[tf.Tensor]` | List of layer output tensors from a Keras model. |
| axis_to_concat | `int` | Axis along which tensors will be concatenated if |
| print_combinations | `bool` | If True, prints every possible combination |
| strategy | `str` | Strategy for selecting candidate skip connections. |
| merge_mode | `str` | Defines how selected tensors are merged: |

**Returns**

`tf.Tensor` – 


**Raises**

- `ValueError` – If `strategy` is not one of `'final'` or `'any'`.
- `ValueError` – If `merge_mode` is not one of `'concat'` or `'add'`.

**Examples**

```python
result = trial_skip_connections(...)
```

## keras.builders.tcnn

### build_tcnn1d
```python
build_tcnn1d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, tuple[int, int]], filters_step: int, kernel_size_range: Union[int, tuple[int, int]], kernel_size_step: int, data_format: str, padding: str, strides: int, dilation_rate: int, use_bias: bool, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Builds a single 1D transposed convolution block with optional batch norm and activation.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Hyperparameter tuning object, such as from Optuna. |
| kparams | `KParams` | Hyperparameter manager providing activation and regularizer configurations. |
| x | `layers.Layer` | Input layer/tensor to process. |
| filters_range | `Union[int` | Fixed or tunable number of filters. |
| tuple[int | `Any` |  |
| int]] | `Any` |  |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_range | `Union[int` | Fixed or tunable kernel size. |
| tuple[int | `Any` |  |
| int]] | `Any` |  |
| kernel_size_step | `int` | Step size for kernel size tuning. |
| data_format | `str` | Format of the input data (e.g., "channels_last"). |
| padding | `str` | Type of padding to use in the convolution (e.g., "same" or "valid"). |
| strides | `int` | Stride length of the convolution. |
| dilation_rate | `int` | Dilation rate for dilated convolution. |
| use_bias | `bool` | Whether to include a bias term in the Conv1DTranspose layer. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias values. |
| use_batch_norm | `bool` | Whether to apply batch normalization. |
| trial_kernel_reg | `bool` | Whether to enable and tune kernel regularization. |
| trial_bias_reg | `bool` | Whether to enable and tune bias regularization. |
| trial_activity_reg | `bool` | Whether to enable and tune activity regularization. |
| name_prefix | `str` | Prefix used for naming all internal layers. |

**Returns**

`layers.Layer` – 


**Raises**

- `None` – 

**Examples**

```python
result = build_tcnn1d(...)
```

### build_tcnn2d
```python
build_tcnn2d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, tuple[int, int]], filters_step: int, kernel_size_range: Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]], kernel_size_step: int, data_format: str, padding: str, strides: tuple[int, int], dilation_rate: tuple[int, int], use_bias: bool, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Builds a single 2D transposed convolution block with optional batch norm and activation.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Hyperparameter tuning object, such as from Optuna. |
| kparams | `KParams` | Hyperparameter manager providing activation and regularizer configurations. |
| x | `layers.Layer` | Input layer/tensor to process. |
| filters_range | `Union[int` | Fixed or tunable number of filters. |
| tuple[int | `Any` |  |
| int]] | `Any` |  |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_range | `Union[tuple[int` |  |
| int] | `Any` |  |
| tuple[tuple[int | `Any` |  |
| int] | `Any` |  |
| tuple[int | `Any` |  |
| int]]] | `Any` |  |
| kernel_size_step | `int` | Step size for kernel size tuning. |
| data_format | `str` | Format of the input data (e.g., "channels_last"). |
| padding | `str` | Type of padding to use in the convolution (e.g., "same" or "valid"). |
| strides | `tuple[int` | Stride length for height and width. |
| int] | `Any` |  |
| dilation_rate | `tuple[int` | Dilation rate for height and width. |
| int] | `Any` |  |
| use_bias | `bool` | Whether to include a bias term in the Conv2DTranspose layer. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias values. |
| use_batch_norm | `bool` | Whether to apply batch normalization. |
| trial_kernel_reg | `bool` | Whether to enable and tune kernel regularization. |
| trial_bias_reg | `bool` | Whether to enable and tune bias regularization. |
| trial_activity_reg | `bool` | Whether to enable and tune activity regularization. |
| name_prefix | `str` | Prefix used for naming all internal layers. |

**Returns**

`layers.Layer` – layers.Layer: Final output tensor after applying the Conv2DTranspose, optional batch norm, and activation.


**Raises**

- None

**Examples**

```python
result = build_tcnn2d(...)
```

### build_tcnn3d
```python
build_tcnn3d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, Tuple[int, int]], filters_step: int, kernel_size_range: Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], kernel_size_step: int, data_format: str, padding: str, strides: Tuple[int, int, int], dilation_rate: Tuple[int, int, int], use_bias: bool, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str)  [source]
```
Builds a single 3D transposed convolution block with optional batch norm and activation.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Hyperparameter tuning object, such as from Optuna. |
| kparams | `KParams` | Hyperparameter manager providing activation and regularizer configurations. |
| x | `layers.Layer` | Input layer/tensor to process. |
| filters_range | `Union[int` | Fixed or tunable number of filters. |
| Tuple[int | `Any` |  |
| int]] | `Any` |  |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_range | `Union[Tuple[int` |  |
| int | `Any` |  |
| int] | `Any` |  |
| Tuple[Tuple[int | `Any` |  |
| int] | `Any` |  |
| Tuple[int | `Any` |  |
| int] | `Any` |  |
| Tuple[int | `Any` |  |
| int]]] | `Any` |  |
| kernel_size_step | `int` | Step size for kernel size tuning. |
| data_format | `str` | Format of the input data (e.g., "channels_last"). |
| padding | `str` | Type of padding to use in the convolution (e.g., "same" or "valid"). |
| strides | `Tuple[int` | Stride length for depth, height, and width. |
| int | `Any` |  |
| int] | `Any` |  |
| dilation_rate | `Tuple[int` | Dilation rate for depth, height, and width. |
| int | `Any` |  |
| int] | `Any` |  |
| use_bias | `bool` | Whether to include a bias term in the Conv3DTranspose layer. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias values. |
| use_batch_norm | `bool` | Whether to apply batch normalization. |
| trial_kernel_reg | `bool` | Whether to enable and tune kernel regularization. |
| trial_bias_reg | `bool` | Whether to enable and tune bias regularization. |
| trial_activity_reg | `bool` | Whether to enable and tune activity regularization. |
| name_prefix | `str` | Prefix used for naming all internal layers. |

**Returns**

`layers.Layer` – layers.Layer: Final output tensor after applying the Conv3DTranspose, optional batch norm, and activation.


**Raises**

- None

**Examples**

```python
result = build_tcnn3d(...)
```

## keras.callbacks

### get_callbacks_study
```python
get_callbacks_study(trial: optuna.Trial, tensorboard_logs: str, monitor: str)  [source]
```
Constructs and returns a list of Keras callbacks tailored for Optuna trials.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | The current Optuna trial object. |
| tensorboard_logs | `str` | Directory where TensorBoard logs will be stored. |
| monitor | `str` | The metric to monitor for early stopping and learning rate reduction. |

**Returns**

`List[tf.keras.callbacks.Callback]` – List[tf.keras.callbacks.Callback]: A list of callbacks to pass into `model.fit()`.


**Raises**

- None

**Examples**

```python
result = get_callbacks_study(...)
```

### get_callbacks_model
```python
get_callbacks_model(backup_dir: str, tensorboard_logs: str)  [source]
```
Constructs and returns a list of Keras callbacks for model training.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| backup_dir | `str` | Directory where the backup files will be stored. |
| tensorboard_logs | `str` | Directory where TensorBoard logs will be stored. |

**Returns**

`List[tf.keras.callbacks.Callback]` – List[tf.keras.callbacks.Callback]: A list of callbacks to pass into `model.fit()`.


**Raises**

- None

**Examples**

```python
result = get_callbacks_model(...)
```

## keras.hyperparams

### _sample_choice
```python
_sample_choice(trial: optuna.Trial, name: str, choices: Sequence[Any])  [source]
```
Sample a value from ``choices`` using Optuna.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` |  |
| name | `str` |  |
| choices | `Sequence[Any]` |  |

**Returns**

`Any` – 


**Raises**

- None

**Examples**

```python
result = _sample_choice(...)
```

### _ensure_mapping
```python
_ensure_mapping(choices: Union[Sequence[Any], Mapping[str, Any]])  [source]
```
Convert choices to a ``dict`` keyed by unique strings.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| choices | `Union[Sequence[Any]` |  |
| Mapping[str | `Any` |  |
| Any]] | `Any` |  |

**Returns**

`Dict[str, Any]` – 


**Raises**

- None

**Examples**

```python
result = _ensure_mapping(...)
```

## keras.utils

### convert_to_saved_model
```python
convert_to_saved_model(input_keras_path: str, output_zip_path: str)  [source]
```
Convert a Keras `.keras` model archive into a zipped TensorFlow SavedModel.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| input_keras_path | `str` | Path to the source `.keras` model file. |
| output_zip_path | `str` | Desired path for the output zip (e.g. 'saved_model.zip'). |

**Returns**

`None` – 


**Raises**

- `Any exception raised by TensorFlow I/O (e.g. file not found, load/save errors).` – 

**Examples**

```python
result = convert_to_saved_model(...)
```

### capture_model_summary
```python
capture_model_summary(model: Any)  [source]
```
Capture model summary as a string.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| model | `Any` | Keras model |

**Returns**

`Any` – str: Model summary as string


**Raises**

- None

**Examples**

```python
result = capture_model_summary(...)
```

### punish_model_flops
```python
punish_model_flops(target: Union[float, Sequence[float]], model: tf.keras.Model, penalty_factor: float, direction: Literal['minimize', 'maximize'])  [source]
```
Penalize an objective according to the model's FLOPs.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| target | `Union[float` | Base objective value (scalar or list of scalars). |
| Sequence[float]] | `Any` |  |
| model | `tf.keras.Model` | Model whose FLOPs will be used for the penalty. |
| penalty_factor | `float` | Multiplicative factor applied to the FLOPs count. |
| direction | `Literal['minimize'` | Whether the objective should be minimised or maximised. |
| 'maximize'] | `Any` |  |

**Returns**

`Union[float, Sequence[float]]` – The penalised objective value or list of values.


**Raises**

- None

**Examples**

```python
result = punish_model_flops(...)
```

### punish_model_params
```python
punish_model_params(target: Union[float, Sequence[float]], model: tf.keras.Model, penalty_factor: float, direction: Literal['minimize', 'maximize'])  [source]
```
Penalize an objective according to the model's parameter count.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| target | `Union[float` | Base objective value (scalar or list of scalars). |
| Sequence[float]] | `Any` |  |
| model | `tf.keras.Model` | Model whose parameters will be used for the penalty. |
| penalty_factor | `float` | Multiplicative factor applied to the parameter count. |
| direction | `Literal['minimize'` | Whether the objective should be minimised or maximised. |
| 'maximize'] | `Any` |  |

**Returns**

`Union[float, Sequence[float]]` – The penalised objective value or list of values.


**Raises**

- None

**Examples**

```python
result = punish_model_params(...)
```

### punish_model
```python
punish_model(target: Union[float, Sequence[float]], model: tf.keras.Model, type: Literal['flops', 'params', None], flops_penalty_factor: float, params_penalty_factor: float, direction: Literal['minimize', 'maximize'])  [source]
```
Apply both FLOPs and parameter penalties to an objective.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| target | `Union[float` | Base objective value (scalar or list of scalars). |
| Sequence[float]] | `Any` |  |
| model | `tf.keras.Model` | Model whose complexity will be penalised. |
| type | `Literal['flops'` | Type of penalty to apply, either "flops" or "params". |
| 'params' | `Any` |  |
| None] | `Any` |  |
| flops_penalty_factor | `float` | Factor for FLOPs penalty. |
| params_penalty_factor | `float` | Factor for parameters penalty. |
| direction | `Literal['minimize'` | Whether the objective should be minimised or maximised. |
| 'maximize'] | `Any` |  |

**Returns**

`Union[float, Sequence[float]]` – The penalised objective value or list of values.


**Raises**

- None

**Examples**

```python
result = punish_model(...)
```

## kernel.monitoring

### print_monitoring_config_summary
```python
print_monitoring_config_summary(file_path: str, file_type: str, success_flag_file: str, max_restarts: int, email_enabled: bool, title: str, restart_after_delay: Optional[float])  [source]
```
Print a summary of monitoring configuration only once.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| file_path | `str` |  |
| file_type | `str` |  |
| success_flag_file | `str` |  |
| max_restarts | `int` |  |
| email_enabled | `bool` |  |
| title | `str` |  |
| restart_after_delay | `Optional[float]` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_monitoring_config_summary(...)
```

### print_process_status
```python
print_process_status(message: str, pid: Optional[int], runtime: Optional[float])  [source]
```
Print process status messages with consistent formatting.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| message | `str` |  |
| pid | `Optional[int]` |  |
| runtime | `Optional[float]` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_process_status(...)
```

### print_restart_info
```python
print_restart_info(restart_count: int, max_restarts: int, delay: float)  [source]
```
Print restart information with formatting.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| restart_count | `int` |  |
| max_restarts | `int` |  |
| delay | `float` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_restart_info(...)
```

### print_completion_summary
```python
print_completion_summary(restart_count: int, total_runtime: Optional[float])  [source]
```
Print final completion summary.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| restart_count | `int` |  |
| total_runtime | `Optional[float]` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_completion_summary(...)
```

### print_error_message
```python
print_error_message(error_type: str, message: str)  [source]
```
Print error messages with consistent formatting.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| error_type | `str` |  |
| message | `str` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_error_message(...)
```

### print_warning_message
```python
print_warning_message(message: str)  [source]
```
Print warning messages with consistent formatting.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| message | `str` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_warning_message(...)
```

### print_success_message
```python
print_success_message(message: str)  [source]
```
Print success messages with consistent formatting.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| message | `str` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_success_message(...)
```

### print_cleanup_info
```python
print_cleanup_info(terminated: int, killed: int)  [source]
```
Print child process cleanup information.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| terminated | `int` |  |
| killed | `int` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_cleanup_info(...)
```

### _cleanup_stale_monitor_files
```python
_cleanup_stale_monitor_files()  [source]
```


**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`Any` – 


**Raises**

- None

**Examples**

```python
result = _cleanup_stale_monitor_files(...)
```

### get_process_resource_usage
```python
get_process_resource_usage(pid: int)  [source]
```
Return memory percentage, memory in GB, and CPU percentage for a process.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| pid | `int` | Process ID of the process to query. |

**Returns**

`Tuple[float, float, float]` – 


**Raises**

- `psutil.NoSuchProcess` – If the PID does not exist.

**Examples**

```python
result = get_process_resource_usage(...)
```

### print_process_resource_usage
```python
print_process_resource_usage(pid: int)  [source]
```
Display CPU and memory usage for a process in a single updating line.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| pid | `int` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_process_resource_usage(...)
```

### start_monitor
```python
start_monitor(pid: int, title: str, supress_tf_warnings: bool)  [source]
```
Start simplified crash monitor without email capabilities.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| pid | `int` | Process ID to monitor |
| title | `str` | Process title for alerts |
| supress_tf_warnings | `bool` | Suppress TensorFlow warnings (default: False) |

**Returns**

`Dict[str, Any]` – 


**Raises**

- `ValueError` – If PID doesn't exist
- `OSError` – If monitor startup fails

**Examples**

```python
result = start_monitor(...)
```

### stop_monitor
```python
stop_monitor(monitor_info: Dict[str, Any])  [source]
```
Stop monitor and cleanup files with optimized batch operations.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| monitor_info | `Dict[str` | Monitor control info from start_monitor() |
| Any] | `Any` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = stop_monitor(...)
```

### check_crash_signal
```python
check_crash_signal(monitor_info: Dict[str, Any])  [source]
```
Check if process crashed with minimal I/O operations.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| monitor_info | `Dict[str` | Monitor control info |
| Any] | `Any` |  |

**Returns**

`Dict[str, Any]` – Dictionary with crash info or empty dict if no crash


**Raises**

- None

**Examples**

```python
result = check_crash_signal(...)
```

### run_auto_restart
```python
run_auto_restart(file_path: str, success_flag_file: str, title: Optional[str], max_restarts: int, restart_delay: float, recipients_file: Optional[str], credentials_file: Optional[str], restart_after_delay: Optional[float], retry_attempts: int, supress_tf_warnings: bool, resource_usage_log_file: Optional[str])  [source]
```
Main function with notebook conversion, file cleanup, and consolidated email notification support.
> [!WARNING]
> Automatically restarts the target process when it crashes.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| file_path | `str` | Path to .py or .ipynb file to execute |
| success_flag_file | `str` | Path to success flag file |
| title | `Optional[str]` | Custom title for monitoring and email alerts |
| max_restarts | `int` | Maximum restart attempts |
| restart_delay | `float` | Delay between restarts in seconds |
| recipients_file | `Optional[str]` | Path to recipients JSON file (defaults to ./json/recipients.json) |
| credentials_file | `Optional[str]` | Path to credentials JSON file (defaults to ./json/credentials.json) |
| restart_after_delay | `Optional[float]` | restart the run after a delay in seconds |
| retry_attempts | `int` | Number of retry attempts before sending failure email |
| supress_tf_warnings | `bool` | Suppress TensorFlow warnings (default: False) |
| resource_usage_log_file | `Optional[str]` | Path to write process resource usage logs. If None, logging is disabled. |

**Returns**

`None` – 


**Raises**

- `FileNotFoundError` – If file doesn't exist
- `ValueError` – If file type is unsupported
- `ImportError` – If notebook dependencies missing for .ipynb files

**Examples**

```python
result = run_auto_restart(...)
```

## optuna.analysis.analyzer

### set_plot_config_param
```python
set_plot_config_param(param_name: str, value: Any)  [source]
```
Set a single parameter in :data:`PLOT_CFG`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| param_name | `str` |  |
| value | `Any` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = set_plot_config_param(...)
```

### set_plot_config_params
```python
set_plot_config_params()  [source]
```
Set multiple parameters in :data:`PLOT_CFG`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = set_plot_config_params(...)
```

### format_title
```python
format_title(template: str, display_name: str)  [source]
```
Format a title template with the given display name.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| template | `str` |  |
| display_name | `str` |  |

**Returns**

`str` – 


**Raises**

- None

**Examples**

```python
result = format_title(...)
```

### calculate_grid
```python
calculate_grid(n_plots: int, subplot_width: int, subplot_height: int, base_max_cols: int)  [source]
```
Calculate grid dimensions ensuring the resulting figure stays within

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| n_plots | `int` |  |
| subplot_width | `int` |  |
| subplot_height | `int` |  |
| base_max_cols | `int` |  |

**Returns**

`Tuple[int, int]` – 


**Raises**

- None

**Examples**

```python
result = calculate_grid(...)
```

### draw_warning_box
```python
draw_warning_box(ax: plt.Axes, message: str)  [source]
```
Display a warning message inside a plot area.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| ax | `plt.Axes` |  |
| message | `str` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = draw_warning_box(...)
```

### create_directories
```python
create_directories(table_dir: str, create_standalone: bool, save_data: bool, create_plotly: bool)  [source]
```
Create organized subdirectories for storing analysis outputs.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| table_dir | `str` |  |
| create_standalone | `bool` |  |
| save_data | `bool` |  |
| create_plotly | `bool` |  |

**Returns**

`Dict[str, str]` – 


**Raises**

- None

**Examples**

```python
result = create_directories(...)
```

### save_data_for_latex
```python
save_data_for_latex(data_dict: Dict[str, Any], filename: str, data_dir: str)  [source]
```
Save graph data to CSV files for LaTeX plotting.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| data_dict | `Dict[str` |  |
| Any] | `Any` |  |
| filename | `str` |  |
| data_dir | `str` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = save_data_for_latex(...)
```

### save_plotly_html
```python
save_plotly_html(fig: Any, filepath: str)  [source]
```
Save a Plotly figure to an HTML file.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| fig | `Any` |  |
| filepath | `str` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = save_plotly_html(...)
```

### save_plot
```python
save_plot(fig: plt.Figure, dirs: Dict[str, str], base_name: str, subdir_key: str, create_plotly: bool, plotly_fig: Any)  [source]
```
Save Matplotlib figure and optionally a Plotly HTML version.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| fig | `plt.Figure` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| base_name | `str` |  |
| subdir_key | `str` |  |
| create_plotly | `bool` |  |
| plotly_fig | `Any` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = save_plot(...)
```

### get_param_display_name
```python
get_param_display_name(param_name: str, param_name_mapping: Dict[str, str])  [source]
```
Get display name for parameter, using mapping if provided.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| param_name | `str` |  |
| param_name_mapping | `Dict[str` |  |
| str] | `Any` |  |

**Returns**

`str` – 


**Raises**

- None

**Examples**

```python
result = get_param_display_name(...)
```

### prepare_dataframe
```python
prepare_dataframe(study: optuna.Study)  [source]
```
Extract and clean completed trial data from Optuna study.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |

**Returns**

`pd.DataFrame` – 


**Raises**

- None

**Examples**

```python
result = prepare_dataframe(...)
```

### classify_columns
```python
classify_columns(df: pd.DataFrame)  [source]
```
Split DataFrame columns into numeric and categorical parameter types.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| df | `pd.DataFrame` |  |

**Returns**

`Tuple[List[str], List[str]]` – 


**Raises**

- None

**Examples**

```python
result = classify_columns(...)
```

### get_trial_subsets
```python
get_trial_subsets(df: pd.DataFrame, top_frac: float)  [source]
```
Extract best and worst performing trial subsets based on loss values.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| df | `pd.DataFrame` |  |
| top_frac | `float` |  |

**Returns**

`Tuple[pd.DataFrame, pd.DataFrame]` – 


**Raises**

- None

**Examples**

```python
result = get_trial_subsets(...)
```

### format_numeric_value
```python
format_numeric_value(x: float)  [source]
```
Format numeric values with appropriate precision for readability.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| x | `float` |  |

**Returns**

`Union[int, float, str]` – 


**Raises**

- None

**Examples**

```python
result = format_numeric_value(...)
```

### save_summary_tables
```python
save_summary_tables(df: pd.DataFrame, best: pd.DataFrame, worst: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], dirs: Dict[str, str])  [source]
```
Generate and save statistical summary tables for different trial subsets.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| df | `pd.DataFrame` |  |
| best | `pd.DataFrame` |  |
| worst | `pd.DataFrame` |  |
| numeric_cols | `List[str]` |  |
| categorical_cols | `List[str]` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = save_summary_tables(...)
```

### _safe_plot
```python
_safe_plot(plot_name: str, func: Callable)  [source]
```
Execute a plotting function, catching and reporting any errors.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| plot_name | `str` |  |
| func | `Callable` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = _safe_plot(...)
```

### print_study_columns
```python
print_study_columns(study: optuna.Study, exclude: Optional[List[str]], param_name_mapping: Optional[Dict[str, str]])  [source]
```
Print the names of the DataFrame columns from the study as a bullet list.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| exclude | `Optional[List[str]]` |  |
| param_name_mapping | `Optional[Dict[str` |  |
| str]] | `Any` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = print_study_columns(...)
```

### analyze_study
```python
analyze_study(study: optuna.Study, table_dir: str, top_frac: float, param_name_mapping: Dict[str, str], create_standalone: bool, save_data: bool, create_plotly: bool, plots: Optional[List[str]])  [source]
```
Comprehensive analysis of Optuna hyperparameter optimization study results.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` | Optuna study object containing trials to analyze. |
| table_dir | `str` | Directory to save analysis results and figures. |
| top_frac | `float` | Fraction of best/worst trials to analyze (default: 0.2). |
| param_name_mapping | `Dict[str` | Optional mapping of parameter names to display names. |
| str] | `Any` |  |
| create_standalone | `bool` | If True, generates standalone images for each plot type. |
| save_data | `bool` | If True, saves data for LaTeX plotting into CSV files. |
| create_plotly | `bool` | If True, also saves interactive Plotly HTML versions of the figures. |
| plots | `Optional[List[str]]` | List of plot types to generate. Available options: |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = analyze_study(...)
```

## optuna.analysis.create_frequency_table

### create_frequency_table
```python
create_frequency_table(data: pd.DataFrame, cols: List[str])  [source]
```
Generate frequency tables for categorical hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| data | `pd.DataFrame` |  |
| cols | `List[str]` |  |

**Returns**

`pd.DataFrame` – 


**Raises**

- None

**Examples**

```python
result = create_frequency_table(...)
```

## optuna.analysis.describe_numeric

### describe_numeric
```python
describe_numeric(data: pd.DataFrame, cols: List[str])  [source]
```
Generate descriptive statistics for numeric hyperparameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| data | `pd.DataFrame` |  |
| cols | `List[str]` |  |

**Returns**

`pd.DataFrame` – 


**Raises**

- None

**Examples**

```python
result = describe_numeric(...)
```

## optuna.analysis.plot_contour

### plot_contour
```python
plot_contour(study: optuna.Study, params: List[str], dirs: Dict[str, str], create_standalone: bool, create_plotly: bool)  [source]
```
Generate contour plots for parameter pairs.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| params | `List[str]` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| create_standalone | `bool` |  |
| create_plotly | `bool` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_contour(...)
```

## optuna.analysis.plot_edf

### plot_edf
```python
plot_edf(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool)  [source]
```
Plot the empirical distribution function of objective values.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| create_plotly | `bool` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_edf(...)
```

## optuna.analysis.plot_hyperparameter_distributions

### plot_hyperparameter_distributions
```python
plot_hyperparameter_distributions(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], dirs: Dict[str, str], param_name_mapping: Dict[str, str], create_standalone: bool, create_plotly: bool)  [source]
```
Generate and save distribution plots for numeric and categorical hyperparameters in separate figures.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| df | `pd.DataFrame` | DataFrame containing hyperparameter data |
| numeric_cols | `List[str]` | List of numeric column names |
| categorical_cols | `List[str]` | List of categorical column names |
| dirs | `Dict[str` | Dictionary of directory paths for saving plots |
| str] | `Any` |  |
| param_name_mapping | `Dict[str` | Optional mapping for parameter display names |
| str] | `Any` |  |
| create_standalone | `bool` | Whether to create standalone images for each parameter |
| create_plotly | `bool` | Whether to save interactive HTML versions |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_hyperparameter_distributions(...)
```

## optuna.analysis.plot_intermediate_values

### plot_intermediate_values
```python
plot_intermediate_values(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool)  [source]
```
Plot intermediate values reported during trials.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| create_plotly | `bool` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_intermediate_values(...)
```

## optuna.analysis.plot_optimal_ranges_analysis

### plot_optimal_ranges_analysis
```python
plot_optimal_ranges_analysis(df: pd.DataFrame, best: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str], param_name_mapping: Dict[str, str], create_standalone: bool, create_plotly: bool)  [source]
```
Create a single comprehensive visualization showing optimal parameter ranges based on best-performing trials.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| df | `pd.DataFrame` | Complete dataset with all trials |
| best | `pd.DataFrame` | Subset of best-performing trials |
| numeric_cols | `List[str]` | List of numeric parameter column names |
| dirs | `Dict[str` | Directory paths for saving outputs |
| str] | `Any` |  |
| param_name_mapping | `Dict[str` | Optional mapping for parameter display names |
| str] | `Any` |  |
| create_standalone | `bool` | Whether to create standalone images for each parameter |
| create_plotly | `bool` | Whether to save interactive HTML versions |

**Returns**

`None` – None: Saves the optimal ranges visualization to fig_ranges directory


**Raises**

- None

**Examples**

```python
result = plot_optimal_ranges_analysis(...)
```

## optuna.analysis.plot_optimization_history

### plot_optimization_history
```python
plot_optimization_history(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool)  [source]
```
Plot optimization history of the study.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| create_plotly | `bool` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_optimization_history(...)
```

## optuna.analysis.plot_parallel_coordinate

### plot_parallel_coordinate
```python
plot_parallel_coordinate(study: optuna.Study, params: List[str], dirs: Dict[str, str], create_plotly: bool)  [source]
```
Create a parallel coordinate plot for trials.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| params | `List[str]` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| create_plotly | `bool` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_parallel_coordinate(...)
```

## optuna.analysis.plot_param_importances

### plot_param_importances
```python
plot_param_importances(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool)  [source]
```
Generate and save parameter importance analysis.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` | Optuna study object containing optimization history |
| dirs | `Dict[str` | Directory paths for saving outputs |
| str] | `Any` |  |
| create_plotly | `bool` | Whether to save an interactive HTML version |

**Returns**

`None` – None: Saves importance table as CSV and bar chart as pdf


**Raises**

- None

**Examples**

```python
result = plot_param_importances(...)
```

## optuna.analysis.plot_parameter_boxplots

### plot_parameter_boxplots
```python
plot_parameter_boxplots(df: pd.DataFrame, best: pd.DataFrame, worst: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str], param_name_mapping: Dict[str, str], create_standalone: bool, create_plotly: bool)  [source]
```
Create separate comprehensive boxplot comparisons for numeric parameters across trial subsets.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| df | `pd.DataFrame` | Complete dataset with all trials |
| best | `pd.DataFrame` | Subset of best-performing trials |
| worst | `pd.DataFrame` | Subset of worst-performing trials |
| numeric_cols | `List[str]` | List of numeric parameter column names |
| dirs | `Dict[str` | Directory paths for saving outputs |
| str] | `Any` |  |
| param_name_mapping | `Dict[str` | Optional mapping for parameter display names |
| str] | `Any` |  |
| create_standalone | `bool` | Whether to create standalone images for each parameter |
| create_plotly | `bool` | Whether to save interactive HTML versions |

**Returns**

`None` – None: Saves separate boxplot files for numeric parameters


**Raises**

- None

**Examples**

```python
result = plot_parameter_boxplots(...)
```

## optuna.analysis.plot_rank

### make_rank_plotly
```python
make_rank_plotly(df: pd.DataFrame, pairs: List[Tuple[str, str]])  [source]
```


**Parameters**

| Name | Type | Description |
|------|------|-------------|
| df | `pd.DataFrame` |  |
| pairs | `List[Tuple[str` |  |
| str]] | `Any` |  |

**Returns**

`Any` – 


**Raises**

- None

**Examples**

```python
result = make_rank_plotly(...)
```

### plot_rank
```python
plot_rank(study: optuna.Study, params: List[str], dirs: Dict[str, str], create_standalone: bool, create_plotly: bool)  [source]
```
Plot parameter relations colored by rank.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| params | `List[str]` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| create_standalone | `bool` |  |
| create_plotly | `bool` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_rank(...)
```

## optuna.analysis.plot_slice

### plot_slice
```python
plot_slice(study: optuna.Study, params: List[str], dirs: Dict[str, str], create_standalone: bool, create_plotly: bool)  [source]
```
Create slice plots for each parameter.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| params | `List[str]` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| create_standalone | `bool` |  |
| create_plotly | `bool` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_slice(...)
```

## optuna.analysis.plot_spearman_correlation

### plot_spearman_correlation
```python
plot_spearman_correlation(df: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str], create_plotly: bool)  [source]
```
Generate and save Spearman correlation heatmap for numeric parameters and loss.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| df | `pd.DataFrame` | Dataset containing numeric parameters and loss values |
| numeric_cols | `List[str]` | List of numeric parameter column names |
| dirs | `Dict[str` | Directory paths for saving outputs |
| str] | `Any` |  |
| create_plotly | `bool` | Whether to save an interactive HTML version |

**Returns**

`None` – None: Saves correlation heatmap as pdf file


**Raises**

- None

**Examples**

```python
result = plot_spearman_correlation(...)
```

## optuna.analysis.plot_terminator_improvement

### _get_improvement_info
```python
_get_improvement_info(study: optuna.Study, get_error: bool, improvement_evaluator: Optional[BaseImprovementEvaluator], error_evaluator: Optional[BaseErrorEvaluator])  [source]
```


**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| get_error | `bool` |  |
| improvement_evaluator | `Optional[BaseImprovementEvaluator]` |  |
| error_evaluator | `Optional[BaseErrorEvaluator]` |  |

**Returns**

`_ImprovementInfo` – 


**Raises**

- None

**Examples**

```python
result = _get_improvement_info(...)
```

### _get_y_range
```python
_get_y_range(info: _ImprovementInfo, min_n_trials: int)  [source]
```


**Parameters**

| Name | Type | Description |
|------|------|-------------|
| info | `_ImprovementInfo` |  |
| min_n_trials | `int` |  |

**Returns**

`tuple[float, float]` – 


**Raises**

- None

**Examples**

```python
result = _get_y_range(...)
```

### plot_terminator_improvement
```python
plot_terminator_improvement(study: optuna.Study, dirs: Dict[str, str], plot_error: bool, print_variance: bool, improvement_evaluator: Optional[BaseImprovementEvaluator], error_evaluator: Optional[BaseErrorEvaluator], min_n_trials: int, create_plotly: bool)  [source]
```
Plot the potentials for future objective improvement using Matplotlib.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| plot_error | `bool` |  |
| print_variance | `bool` |  |
| improvement_evaluator | `Optional[BaseImprovementEvaluator]` |  |
| error_evaluator | `Optional[BaseErrorEvaluator]` |  |
| min_n_trials | `int` |  |
| create_plotly | `bool` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_terminator_improvement(...)
```

## optuna.analysis.plot_timeline

### plot_timeline
```python
plot_timeline(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool)  [source]
```
Visualize trial durations on a timeline with detailed information.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` |  |
| dirs | `Dict[str` |  |
| str] | `Any` |  |
| create_plotly | `bool` |  |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = plot_timeline(...)
```

## optuna.analysis.plot_trend_analysis

### plot_trend_analysis
```python
plot_trend_analysis(df: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str], param_name_mapping: Dict[str, str], create_standalone: bool, create_plotly: bool)  [source]
```
Create a single comprehensive plot with trend analysis for parameter-loss relationships.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| df | `pd.DataFrame` | Dataset containing parameters and loss values |
| numeric_cols | `List[str]` | List of numeric parameter column names |
| dirs | `Dict[str` | Directory paths for saving outputs |
| str] | `Any` |  |
| param_name_mapping | `Dict[str` | Optional mapping for parameter display names |
| str] | `Any` |  |
| create_standalone | `bool` | Whether to create standalone images for each parameter |
| create_plotly | `bool` | Whether to save interactive HTML versions |

**Returns**

`None` – None: Saves single comprehensive trend plot as pdf file and trend statistics as CSV


**Raises**

- None

**Examples**

```python
result = plot_trend_analysis(...)
```

## optuna.keras.stats

### get_model_stats
```python
get_model_stats(trial: optuna.Trial, model: tf.keras.Model, bits_per_param: int, batch_size: int, n_trials: int, device: int, verbose: bool)  [source]
```
Extract and return model statistics from the given Optuna trial.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | The Optuna trial object |
| model | `tf.keras.Model` | The Keras model to analyze. |
| bits_per_param | `int` |  |
| batch_size | `int` | The batch size to simulate for input. |
| n_trials | `int` | Number of trials for power and energy measurement. |
| device | `int` | GPU index to run the model on. Use ``-1`` for CPU. |
| verbose | `bool` | If True, print detailed information. |

**Returns**

`Dict[str, float]` – Dict[str, float]: A dictionary containing model statistics


**Raises**

- None

**Examples**

```python
result = get_model_stats(...)
```

## optuna.utils

### supress_optuna_warnings
```python
supress_optuna_warnings()  [source]
```
Suppress only Optuna experimental warnings.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = supress_optuna_warnings(...)
```

### get_remaining_trials
```python
get_remaining_trials(study: optuna.Study, num_trials: int)  [source]
```
Returns a list of completed trials from the given Optuna study.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` | The Optuna study to retrieve trials from. |
| num_trials | `int` | The total number of trials to consider. |

**Returns**

`list[optuna.trial.FrozenTrial]` – list[optuna.trial.FrozenTrial]: A list of completed trials.


**Raises**

- None

**Examples**

```python
result = get_remaining_trials(...)
```

### cleanup_non_top_trials
```python
cleanup_non_top_trials(all_trial_ids: Set[int], top_trial_ids: Set[int], cleanup_paths: List[Tuple[str, str]])  [source]
```
Remove files or directories for trials not in the top-K set.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| all_trial_ids | `Set[int]` | Set of all trial IDs in the study. |
| top_trial_ids | `Set[int]` | Set of top-K trial IDs to preserve. |
| cleanup_paths | `List[Tuple[str` | List of (base_directory, filename_template) |
| str]] | `Any` |  |

**Returns**

`None` – 


**Raises**

- `OSError` – If file removal operations fail.

**Examples**

```python
result = cleanup_non_top_trials(...)
```

### rename_top_k_files
```python
rename_top_k_files(top_trials: List[optuna.Trial], file_configs: List[Tuple[str, str]])  [source]
```
Rename top-K trial files with ranking prefix.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| top_trials | `List[optuna.Trial]` | List of top trials in ranked order. |
| file_configs | `List[Tuple[str` | List of (base_directory, file_extension) |
| str]] | `Any` |  |

**Returns**

`None` – 


**Raises**

- `OSError` – If file rename operations fail.

**Examples**

```python
result = rename_top_k_files(...)
```

### save_trial_params_to_file
```python
save_trial_params_to_file(filepath: str, params: dict[str, float])  [source]
```
Save Optuna trial parameters and associated metadata to a text file.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| filepath | `str` | Path where the parameter file should be saved. |
| params | `dict[str` | Dictionary of trial hyperparameters. |
| float] | `Any` |  |

**Returns**

`None` – None


**Raises**

- None

**Examples**

```python
result = save_trial_params_to_file(...)
```

### get_top_trials
```python
get_top_trials(study: optuna.Study, top_k: int, rank_key: str, order: str)  [source]
```
Get the top-K trials from an Optuna study based on ranking criteria.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` | The completed Optuna study. |
| top_k | `int` | Number of top trials to retrieve. |
| rank_key | `str` | Key to rank trials by ("value" for objective value, |
| order | `str` | "descending" for highest values first or "ascending" for |

**Returns**

`List[optuna.Trial]` – List[optuna.Trial]: List of top-K trials sorted by the ranking criteria.


**Raises**

- None

**Examples**

```python
result = get_top_trials(...)
```

### save_top_k_trials
```python
save_top_k_trials(top_trials: List[optuna.Trial], args_dir: str, study: optuna.Study, extra_attrs: Optional[List[str]])  [source]
```
Save top-K trials to text files.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| top_trials | `List[optuna.Trial]` | List of trials to save. |
| args_dir | `str` | Directory to save trial parameter files. |
| study | `optuna.Study` | The Optuna study (needed for sampler info). |
| extra_attrs | `Optional[List[str]]` | List of additional user attributes to save. |

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = save_top_k_trials(...)
```

### init_study_dirs
```python
init_study_dirs(run_dir: Any, study_name: Any, subdirs: Any)  [source]
```
Create and return study directory structure for experiments.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| run_dir | `Any` | Base directory for the run |
| study_name | `Any` | Name of the study directory (default: "optuna_study") |
| subdirs | `Any` | List of subdirectory names to create |

**Returns**

`Any` – tuple: (study_dir, *subdirectory_paths) in the order specified by subdirs


**Raises**

- None

**Examples**

```python
result = init_study_dirs(...)
```

## plot.configs

### config_plt
```python
config_plt(style: str)  [source]
```
Configure matplotlib rcParams for IEEE‑style figures

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| style | `str` | The figure style to use. Options are 'single-column' or |

**Returns**

`None` – None


**Raises**

- None

**Examples**

```python
result = config_plt(...)
```

## tensorflow.model

### get_model_usage_stats
```python
get_model_usage_stats(saved_model: str | tf.keras.Model, n_trials: int, device: int, rapl_path: str, verbose: bool)  [source]
```
Estimate average power draw and energy usage.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| saved_model | `str | tf.keras.Model` | Path to the TensorFlow SavedModel directory, |
| n_trials | `int` | Number of inference trials to perform. Defaults to 100000. |
| device | `int` | GPU index for power measurement, or ``-1`` to use the CPU. |
| rapl_path | `str` | Path to the RAPL energy counter file for CPU measurements. |
| verbose | `bool` | If True, displays a progress bar during the trials. |

**Returns**

`Tuple[float, float, float]` – Tuple[float, float, float]: - per_run_time (float): Average run time in seconds. Measures a mix of tracing, initialization, asynchronous queuing, Python overhead, and power-reading delays, so its “average” can be dominated by non-inference costs. - avg_power (float): Average power draw in watts. If a negative value is measured repeatedly, the function returns 0 after two retries. - avg_energy (float): Average energy consumed per inference in joules. This will also be ``0`` if ``avg_power`` could not be measured correctly.


**Raises**

- `RuntimeError` – If GPU NVML initialization fails when ``device`` refers to a GPU index.
- `ValueError` – If ``device`` is neither ``-1`` nor a valid GPU index.

**Examples**

```python
result = get_model_usage_stats(...)
```

## utils.dir

### create_run_directory
```python
create_run_directory(prefix: str, base_dir: str)  [source]
```
Creates a new run directory with an incremented numeric suffix and returns its full path.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| prefix | `str` | Prefix to be used in the name of each run directory (e.g., "run"). |
| base_dir | `str` | Directory under which all runs are stored. Defaults to "runs". |

**Returns**

`str` – str: Absolute path to the newly created run directory. Example: run_path = create_run_directory(prefix="run") print(run_path)  # outputs: runs/run1, runs/run2, etc.


**Raises**

- None

**Examples**

```python
result = create_run_directory(...)
```

## utils.gpu

### get_user_gpu_choice
```python
get_user_gpu_choice()  [source]
```
Prompts the user to select a GPU index and validates the input.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`Any` – str: Valid GPU index as string


**Raises**

- None

**Examples**

```python
result = get_user_gpu_choice(...)
```

### _get_nvidia_smi_data
```python
_get_nvidia_smi_data()  [source]
```
Retrieves GPU information using nvidia-smi command.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`Any` – list: List of GPU information dictionaries or empty list if failed


**Raises**

- None

**Examples**

```python
result = _get_nvidia_smi_data(...)
```

### _print_tensorflow_info
```python
_print_tensorflow_info()  [source]
```
Print TensorFlow configuration information.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`Any` – 


**Raises**

- None

**Examples**

```python
result = _print_tensorflow_info(...)
```

### _print_gpu_table
```python
_print_gpu_table(gpu_data: Any)  [source]
```
Print GPU information in nvidia-smi style table format.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| gpu_data | `Any` |  |

**Returns**

`Any` – 


**Raises**

- None

**Examples**

```python
result = _print_gpu_table(...)
```

### _print_memory_summary
```python
_print_memory_summary(gpu_data: Any)  [source]
```
Print memory summary similar to nvidia-smi bottom section.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| gpu_data | `Any` |  |

**Returns**

`Any` – 


**Raises**

- None

**Examples**

```python
result = _print_memory_summary(...)
```

### get_gpu_info
```python
get_gpu_info()  [source]
```
Prints detailed TensorFlow and GPU configuration information in nvidia-smi style format.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`None` – None Example: get_gpu_info()


**Raises**

- None

**Examples**

```python
result = get_gpu_info(...)
```

### gpu_summary
```python
gpu_summary()  [source]
```
Prints a compact GPU summary similar to nvidia-smi output.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`None` – 


**Raises**

- None

**Examples**

```python
result = gpu_summary(...)
```

## utils.logs

### log_resources
```python
log_resources(log_dir: str, interval: int)  [source]
```
Logs selected system and ML resources (CPU, RAM, GPU, CUDA, TensorFlow) at regular time intervals.
> [!TIP]
> Logging continues until the program exits.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| log_dir | `str` | Directory where log files will be stored. |
| interval | `int` | Time interval between consecutive logs in seconds. Defaults to 5. |

**Returns**

`None` – None Example: log_resources("logs", interval=10, cpu=True, ram=True, gpu=True)


**Raises**

- None

**Examples**

```python
result = log_resources(...)
```

## utils.misc

### clear
```python
clear()  [source]
```
Clear all prints from terminal or notebook cell.

**Parameters**

| Name | Type | Description |
|------|------|-------------|

**Returns**

`Any` – 


**Raises**

- None

**Examples**

```python
result = clear(...)
```

### format_number
```python
format_number(number: Any, precision: Any)  [source]
```
Format a number using scientific suffixes.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| number | `Any` | The number to format |
| precision | `Any` | Number of decimal places to show (default: 2) |

**Returns**

`Any` – str: Formatted number with appropriate suffix


**Raises**

- None

**Examples**

```python
result = format_number(...)
```

### format_bytes
```python
format_bytes(bytes_value: Any, precision: Any)  [source]
```
Format bytes using binary suffixes (B, KB, MB, GB, etc.).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| bytes_value | `Any` | The number of bytes |
| precision | `Any` | Number of decimal places to show (default: 2) |

**Returns**

`Any` – str: Formatted bytes with appropriate suffix


**Raises**

- None

**Examples**

```python
result = format_bytes(...)
```

### format_scientific
```python
format_scientific(number: Any, max_precision: Any)  [source]
```
Format to scientific notation with automatic precision based on number magnitude.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| number | `Any` | The number to format |
| max_precision | `Any` | Maximum number of decimal places (default: 2) |

**Returns**

`Any` – str: Number formatted in scientific notation


**Raises**

- None

**Examples**

```python
result = format_scientific(...)
```

### format_number_commas
```python
format_number_commas(number: Any, precision: Any)  [source]
```
Format a number with commas as thousands separators.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| number | `Any` | The number to format |
| precision | `Any` | Number of decimal places to show (default: 2) |

**Returns**

`Any` – str: Number formatted with commas


**Raises**

- None

**Examples**

```python
result = format_number_commas(...)
```