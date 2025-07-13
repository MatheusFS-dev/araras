# API Documentation

## araras.__init__

Top-level API for the araras package.

## araras.commons

Common setup for the Araras library.

Functions:
    - make_logger: Create a logger with its own StreamHandler and ColorFormatter.

Example:
    >>> from araras.commons import make_logger
    >>> make_logger(...)

### make_logger

```python
def make_logger(name: str, fmt: str, datefmt: str | None, level: int) -> logging.Logger
```

Create a logger with its own StreamHandler and ColorFormatter.
Propagation is turned off so it won’t inherit root handlers.

### class ColorFormatter

#### format

```python
def format(record: logging.LogRecord) -> str
```

## araras.email.__init__

## araras.email.utils

This module provides functions to send emails using SMTP with Gmail.

Functions:
    - get_credentials: Reads the sender's email and password from a JSON file.
    - get_recipient_emails: Reads a list of recipient email addresses from a JSON file.
    - send_email: Sends an email notification with the specified subject and body content to multiple recipients.

### get_credentials

```python
def get_credentials(file_path: str) -> tuple[str, str]
```

Reads the sender's email and password from a JSON file.
The json file format should be:
{
    "email": "your_email@gmail.com",
    "password": "your_password"
}

Args:
    file_path (str): Path to the credentials JSON file.

Returns:
    tuple[str, str]: A tuple containing the sender email and password.

Raises:
    ValueError: If the credentials cannot be read or parsed.

### get_recipient_emails

```python
def get_recipient_emails(file_path: str) -> list[str]
```

Reads a list of recipient email addresses from a JSON file.
The json file format should be:
{
    "emails": ["recipient1@example.com", "recipient2@example.com"]
}

Args:
    file_path (str): Path to the recipient JSON file.

Returns:
    list[str]: A list of recipient email addresses.

Raises:
    ValueError: If the file or its contents cannot be read.

### send_email

```python
def send_email(subject: str, body: str, recipients_file: str, credentials_file: str, text_type: str, smtp_server: str, smtp_port: int) -> None
```

Sends an email notification with the specified subject and body content to multiple recipients.

Example:
    send_email("Hi", "This is a test", "recipients.json", "credentials.json", text_type="html")

Args:
    subject (str): The subject of the email.
    body (str): The main content of the email.
    recipients_file (str): Path to the recipients JSON file.
    credentials_file (str): Path to the credentials JSON file.
    text_type (str): The type of text content (e.g., "plain" or "html").
    smtp_server (str): The SMTP server address (default is Gmail's SMTP server).
    smtp_port (int): The port number for the SMTP server (default is 587 for TLS).

Returns:
    None

## araras.keras.__init__

## araras.keras.analysis.estimator

This module provides utilities for analyzing and visualizing the distribution of model parameters

Functions:
    - get_model_trainable_params: Get the number of trainable parameters in a Keras model.
    - get_precision_bytes: Determine bytes per parameter based on model's actual dtype.
    - get_optimizer_state_factor: Determine optimizer state factor from compiled model.
    - calculate_activation_memory: Calculate activation memory needed during forward/backward pass.
    - get_framework_overhead: Calculate framework overhead based on available GPU memory.
    - estimate_training_memory: Estimate total VRAM needed for training a Keras model in bytes
    - model_param_distribution: Sample random models and plot parameter, size and training memory necessity histograms.

Example:
    >>> from araras.keras.analysis.estimator import model_param_distribution
    >>> model_param_distribution(...)

### calculate_activation_memory

```python
def calculate_activation_memory(model: keras.Model, bytes_per_param: int) -> int
```

Calculate activation memory needed during forward/backward pass.

### estimate_training_memory

```python
def estimate_training_memory(model: keras.Model, batch_size: int) -> int
```

Estimate total VRAM needed for training a Keras model in bytes.

Args:
    model: Keras model object
    batch_size: Training batch size

Returns:
    Total memory needed in bytes

### get_framework_overhead

```python
def get_framework_overhead() -> int
```

Calculate framework overhead based on available GPU memory.

### get_model_trainable_params

```python
def get_model_trainable_params(model: keras.Model) -> int
```

Get number of trainable parameters in the model.

### get_optimizer_state_factor

```python
def get_optimizer_state_factor(model: keras.Model) -> int
```

Determine optimizer state factor from compiled model.

### get_precision_bytes

```python
def get_precision_bytes(model: keras.Model) -> int
```

Determine bytes per parameter based on model's actual dtype.

### model_param_distribution

```python
def model_param_distribution(build_model_fn: Callable[[optuna.Trial], tf.keras.Model], bits_per_param: int, batch_size: int, n_trials: int) -> None
```

Sample random models and plot parameter and size histograms.

Args:
    build_model_fn: Function that builds a Keras model given an Optuna
        ``Trial``.
    bits_per_param: Number of bits used to store each parameter.
    n_trials: Number of random trials to run.

## araras.keras.analysis.profiler

This module provides utilities to calculate the number of floating-point operations (FLOPs)

Functions:
    - get_flops: Calculates the total number of floating-point operations (FLOPs) needed
    - get_macs: Estimates the number of Multiply-Accumulate operations (MACs) required
    - get_memory_and_time: Measures the peak memory usage and average inference time of a Keras model

Example:
    >>> from araras.keras.utils.profiler import get_flops
    >>> get_flops(...)

### get_flops

```python
def get_flops(model: tf.keras.Model, batch_size: int) -> int
```

Calculates the total number of floating-point operations (FLOPs) needed
to perform a single forward pass of the given Keras model.

Flow:
    model -> input_shape -> TensorSpec -> tf.function -> concrete function
    -> graph -> profile(graph) -> total_float_ops -> return

Args:
    model (tf.keras.Model): The Keras model to analyze.
    batch_size (int, optional): The batch size to simulate for input. Defaults to 1.

Returns:
    int: The total number of floating-point operations (FLOPs) for one forward pass.

### get_macs

```python
def get_macs(model: tf.keras.Model, batch_size: int) -> int
```

Estimates the number of Multiply-Accumulate operations (MACs) required
for a single forward pass of the model. Assumes 1 MAC = 2 FLOPs.

Flow:
    model -> input_shape -> TensorSpec -> tf.function -> concrete function
    -> graph -> profile(graph) -> total_float_ops // 2 -> return

Args:
    model (tf.keras.Model): The Keras model to analyze.
    batch_size (int, optional): The batch size to simulate for input. Defaults to 1.

Returns:
    int: The estimated number of MACs for one forward pass.

### get_memory_and_time

```python
def get_memory_and_time(model: tf.keras.Model, batch_size: int, device: int, warmup_runs: int, test_runs: int, verbose: bool) -> Tuple[int, float]
```

Measures the peak memory usage and average inference time of a Keras model
on GPU or CPU.

Observations:
    Warmup runs exclude one-time initialization costs from your measurements. On GPU
    the very first inference will trigger things like driver wake-up, context setup,
    PTX→BIN compilation and power-state switching, and cache fills. By running a few
    warmup inferences you force all of that work to happen before timing, so your
    measured latencies reflect true steady-state performance rather than setup overhead.

    Under @tf.function the first call also traces and builds the execution graph,
    applies optimizations and allocates buffers. Those activities inflate both time
    and memory on the “cold” run. Warmup runs let TensorFlow complete tracing and
    graph compilation once, so your timed loop measures only the optimized graph
    execution path.

The CPU memory probe occasionally reports zero usage. When this happens, the
measurement is retried up to two additional times. If all attempts still
report zero memory, the function returns ``0`` for the peak usage and emits a
warning in red.

Args:
    model (tf.keras.Model): The Keras model to analyze.
    batch_size (int): The batch size to simulate for input. Defaults to 1.
        Measure with batch_size=1 to get base per-sample latency.
    device (int): GPU index to run the model on. Use ``-1`` to run on CPU.
    warmup_runs (int): Number of warm-up runs before timing. Defaults to 10.
    test_runs (int): Number of runs to measure average inference time. Defaults to 50.
    verbose (bool): If True, displays a progress bar during test runs.

Returns:
    Tuple[int, float]:
        - peak memory usage in bytes (0 if CPU measurement fails after
          several attempts)
        - average inference time in seconds

## araras.keras.builders.__init__

## araras.keras.builders.cnn

Builders for Convolutional Neural Networks (CNNs) in Keras.

Functions:
    - build_cnn1d: Builds a 1D convolutional layer with optional hyperparameter tuning and regularization.
    - build_dense_as_conv1d: Simulate a Dense layer using a Conv1D with kernel_size=1.
    - build_cnn2d: Builds a 2D convolutional layer with optional hyperparameter tuning and regularization.
    - build_dense_as_conv2d: Simulate a Dense layer using a Conv2D with kernel_size=(1, 1).
    - build_cnn3d: Builds a 3D convolutional layer with optional hyperparameter tuning and regularization.
    - build_dense_as_conv3d: Simulate a Dense layer using a Conv3D with kernel_size=(1, 1, 1).

Example:
    >>> from araras.keras.builders.cnn import build_cnn1d
    >>> x = build_cnn1d(...)

### build_cnn1d

```python
def build_cnn1d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, tuple[int, int]], kernel_size_range: Union[int, tuple[int, int]], filters_step: int, kernel_size_step: int, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, strides: int, dilation_rate: int, groups: int, use_bias: bool, padding: str, data_format: str, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, name_prefix: str) -> layers.Layer
```

Builds a 1D convolutional layer with optional hyperparameter tuning and regularization.

This function creates a Conv1D layer whose hyperparameters (filters, kernel size, regularizers, etc.)
can be either statically defined or dynamically tuned through an Optuna trial. It optionally applies
batch normalization and a user-defined activation function.

Args:
    trial (Any): An Optuna trial object used for hyperparameter optimization.
    kparams (KParams): A utility object to provide regularizers and activations.
    x (layers.Layer): The input Keras layer.
    filters_range (Union[int, tuple[int, int]]): Number of filters or a range for tuning.
    kernel_size_range (Union[int, tuple[int, int]]): Kernel size or a range for tuning.
    filters_step (int): Step size for filter tuning.
    kernel_size_step (int): Step size for kernel size tuning.
    use_batch_norm (bool): Whether to include batch normalization.
    trial_kernel_reg (bool): Whether to tune and apply kernel regularization.
    trial_bias_reg (bool): Whether to tune and apply bias regularization.
    trial_activity_reg (bool): Whether to tune and apply activity regularization.
    strides (int): Stride size for the convolution.
    dilation_rate (int): Dilation rate for convolution.
    groups (int): Number of filter groups.
    use_bias (bool): Whether to use a bias term in the convolution. If using batch norm, this can be set to False.
    padding (str): Padding method ('valid' or 'same').
    data_format (str): Data format, either 'channels_last' or 'channels_first'.
    kernel_initializer (initializers.Initializer): Initializer for kernel weights.
    bias_initializer (initializers.Initializer): Initializer for bias.
    name_prefix (str): Prefix for layer names.

Returns:
    layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.

### build_cnn2d

```python
def build_cnn2d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, tuple[int, int]], kernel_size_range: Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]], filters_step: int, kernel_size_step: int, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, strides: tuple[int, int], dilation_rate: tuple[int, int], groups: int, use_bias: bool, padding: str, data_format: str, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, name_prefix: str) -> layers.Layer
```

Builds a 2D convolutional layer with optional hyperparameter tuning and regularization.

This function creates a Conv2D layer whose hyperparameters (filters, kernel size, regularizers, etc.)
can be either statically defined or dynamically tuned through an Optuna trial. It optionally applies
batch normalization and a user-defined activation function.

Args:
    trial (Any): An Optuna trial object used for hyperparameter optimization.
    kparams (KParams): A utility object to provide regularizers and activations.
    x (layers.Layer): The input Keras layer.
    filters_range (Union[int, tuple[int, int]]): Number of filters or a range for tuning.
    kernel_size_range (Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]):
        Fixed (height, width) or ranges ((h_min, h_max), (w_min, w_max)) for tuning.
    filters_step (int): Step size for filter tuning.
    kernel_size_step (int): Step size for kernel dimension tuning.
    use_batch_norm (bool): Whether to include batch normalization.
    trial_kernel_reg (bool): Whether to tune and apply kernel regularization.
    trial_bias_reg (bool): Whether to tune and apply bias regularization.
    trial_activity_reg (bool): Whether to tune and apply activity regularization.
    strides (tuple[int, int]): Stride size for height and width.
    dilation_rate (tuple[int, int]): Dilation rate for height and width.
    groups (int): Number of filter groups.
    use_bias (bool): Whether to use a bias term in the convolution. If using batch norm, this can be set to False.
    padding (str): Padding method ('valid' or 'same').
    data_format (str): Data format, either 'channels_last' or 'channels_first'.
    kernel_initializer (initializers.Initializer): Initializer for kernel weights.
    bias_initializer (initializers.Initializer): Initializer for bias.
    name_prefix (str): Prefix for layer names.

Returns:
    layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.

### build_cnn3d

```python
def build_cnn3d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, Tuple[int, int]], kernel_size_range: Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], filters_step: int, kernel_size_step: int, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, strides: Tuple[int, int, int], dilation_rate: Tuple[int, int, int], groups: int, use_bias: bool, padding: str, data_format: str, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, name_prefix: str) -> layers.Layer
```

Builds a 3D convolutional layer with optional hyperparameter tuning and regularization.

This function creates a Conv3D layer whose hyperparameters (filters, kernel size, regularizers, etc.)
can be either statically defined or dynamically tuned through an Optuna trial. It optionally applies
batch normalization and a user-defined activation function.

Args:
    trial (Any): An Optuna trial object used for hyperparameter optimization.
    kparams (KParams): A utility object to provide regularizers and activations.
    x (layers.Layer): The input Keras layer.
    filters_range (Union[int, Tuple[int, int]]): Number of filters or a range for tuning.
    kernel_size_range (Union[
        Tuple[int, int, int],
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ]): Fixed (depth, height, width) or ranges ((d_min, d_max), (h_min, h_max), (w_min, w_max)) for tuning.
    filters_step (int): Step size for filter tuning.
    kernel_size_step (int): Step size for kernel dimension tuning.
    use_batch_norm (bool): Whether to include batch normalization.
    trial_kernel_reg (bool): Whether to tune and apply kernel regularization.
    trial_bias_reg (bool): Whether to tune and apply bias regularization.
    trial_activity_reg (bool): Whether to tune and apply activity regularization.
    strides (Tuple[int, int, int]): Stride size for depth, height, and width.
    dilation_rate (Tuple[int, int, int]): Dilation rate for depth, height, and width.
    groups (int): Number of filter groups.
    use_bias (bool): Whether to use a bias term in the convolution. If using batch norm, this can be set to False.
    padding (str): Padding method ('valid' or 'same').
    data_format (str): Data format, either 'channels_last' or 'channels_first'.
    kernel_initializer (initializers.Initializer): Initializer for kernel weights.
    bias_initializer (initializers.Initializer): Initializer for bias.
    name_prefix (str): Prefix for layer names.

Returns:
    layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.

### build_dense_as_conv1d

```python
def build_dense_as_conv1d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: int, filters_step: int, padding: str, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Simulate a Dense layer using a Conv1D with kernel_size=1.

This function builds a 1D convolutional layer that, when applied to a 3D input
of shape (batch_size, length, features_in), produces an output of shape
(batch_size, length, units).

Note:
    If your goal is to emulate a classic Dense(units) on a flat vector of shape
    (batch_size, features_in), you must first reshape that vector to (batch_size, 1, features_in)
    and then apply this function. After Conv1D, you should call Flatten() to collapse
    back to (batch_size, units). Without reshaping, Conv1D will raise a shape mismatch
    on 2D inputs.

Args:
    trial (Any): An Optuna trial object used for hyperparameter optimization.
    kparams (KParams): A utility object to provide regularizers and activations.
    x (layers.Layer): The input Keras layer, expected to be of shape (batch_size, length, features_in).
    filters_range (int): The number of output filters for the Conv1D layer.
    filters_step (int): Step size for tuning the number of filters.
    padding (str): Padding method ('valid' or 'same').
    trial_kernel_reg (bool): Whether to tune and apply kernel regularization.
    trial_bias_reg (bool): Whether to tune and apply bias regularization.
    trial_activity_reg (bool): Whether to tune and apply activity regularization.
    name_prefix (str): Prefix for layer names.

Returns:
    layers.Layer: A Keras layer with output shape (batch_size, 1, units), equivalent to Dense(units).

References:
    https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer

    https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers

    https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks

### build_dense_as_conv2d

```python
def build_dense_as_conv2d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: int, filters_step: int, padding: str, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Simulate a Dense layer using a Conv2D with kernel_size=(1, 1).

This function builds a 2D convolutional layer that, when applied to a 4D input
of shape (batch_size, height, width, features_in), produces an output of shape
(batch_size, height, width, units).

Note:
    If your goal is to emulate a classic Dense(units) on a flat vector of shape
    (batch_size, features_in), you must first reshape that vector to (batch_size, 1, 1, features_in)
    and then apply this function. After Conv2D, you should call Flatten() to collapse
    back to (batch_size, units). Without reshaping, Conv2D will raise a shape mismatch
    on 3D inputs.

Args:
    trial (Any): An Optuna trial object used for hyperparameter optimization.
    kparams (KParams): A utility object to provide regularizers and activations.
    x (layers.Layer): The input Keras layer, expected to be of shape (batch_size, height, width, features_in).
    filters_range (int): The number of output filters for the Conv2D layer.
    filters_step (int): Step size for tuning the number of filters.
    padding (str): Padding method ('valid' or 'same').
    trial_kernel_reg (bool): Whether to tune and apply kernel regularization.
    trial_bias_reg (bool): Whether to tune and apply bias regularization.
    trial_activity_reg (bool): Whether to tune and apply activity regularization.
    name_prefix (str): Prefix for layer names.

Returns:
    layers.Layer: A Keras layer with output shape (batch_size, height, width, units), equivalent to Dense(units).

References:
    https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer

    https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers

    https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks

### build_dense_as_conv3d

```python
def build_dense_as_conv3d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: int, filters_step: int, padding: str, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Simulate a Dense layer using a Conv3D with kernel_size=(1, 1, 1).

This function builds a 3D convolutional layer that, when applied to a 5D input
of shape (batch_size, depth, height, width, features_in), produces an output of shape
(batch_size, depth, height, width, units).

Note:
    If your goal is to emulate a classic Dense(units) on a flat vector of shape
    (batch_size, features_in), you must first reshape that vector to (batch_size, 1, 1, 1, features_in)
    and then apply this function. After Conv3D, you should call Flatten() to collapse
    back to (batch_size, units). Without reshaping, Conv3D will raise a shape mismatch
    on 4D inputs.

Args:
    trial (Any): An Optuna trial object used for hyperparameter optimization.
    kparams (KParams): A utility object to provide regularizers and activations.
    x (layers.Layer): The input Keras layer, expected to be of shape (batch_size, depth, height, width, features_in).
    filters_range (int): The number of output filters for the Conv3D layer.
    filters_step (int): Step size for tuning the number of filters.
    padding (str): Padding method ('valid' or 'same').
    trial_kernel_reg (bool): Whether to tune and apply kernel regularization.
    trial_bias_reg (bool): Whether to tune and apply bias regularization.
    trial_activity_reg (bool): Whether to tune and apply activity regularization.
    name_prefix (str): Prefix for layer names.

Returns:
    layers.Layer: A Keras layer with output shape (batch_size, depth, height, width, units), equivalent to Dense(units).

References:
    https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer

    https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers

    https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks

## araras.keras.builders.dnn

Builders for Deep Neural Networks (DNNs) in Keras.

Functions:
    - build_dnn: Builds a single dense neural network (DNN) block with optional regularization, batch normalization, and dropout.

Example:
    >>> from araras.keras.builders.dnn import build_dnn
    >>> x = build_dnn(...)

### build_dnn

```python
def build_dnn(trial: Any, kparams: KParams, x: layers.Layer, units_range: Union[int, tuple[int, int]], dropout_rate_range: Union[float, tuple[float, float]], units_step: int, dropout_rate_step: float, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Builds a single dense neural network (DNN) block with optional regularization, batch normalization, and dropout.

This function constructs a configurable DNN layer consisting of a Dense layer followed by optional
batch normalization, a user-specified activation function, and dropout. It supports hyperparameter
tuning via the `trial` object.

Args:
    trial (Any): Hyperparameter tuning trial object, e.g., from Optuna.
    kparams (KParams): Custom hyperparameter handler that provides regularizers and activations.
    x (layers.Layer): Input tensor or layer to build on.
    units_range (Union[int, tuple[int, int]]): Either a fixed unit count or a range for tuning.
    units_step (int): Step size for unit range tuning.
    dropout_rate_range (Union[float, tuple[float, float]]): Either a fixed dropout rate or a range.
    dropout_rate_step (float): Step size for dropout rate tuning.
    kernel_initializer (initializers.Initializer): Initializer for Dense layer weights.
    bias_initializer (initializers.Initializer): Initializer for Dense layer biases.
    use_bias (bool): Whether to include a bias term in the Dense layer.
    use_batch_norm (bool): Whether to include a batch normalization layer.
    trial_kernel_reg (bool): Whether to tune and apply a kernel regularizer.
    trial_bias_reg (bool): Whether to tune and apply a bias regularizer.
    trial_activity_reg (bool): Whether to tune and apply an activity regularizer.
    name_prefix (str): Prefix to use for naming the layers.

Returns:
    layers.Layer: Output tensor after applying the DNN block.

Raises:
    None

## araras.keras.builders.gnn

Builders for Graph Neural Networks (GNNs) in Keras.

Functions:
    - build_grid_adjacency: Builds a grid adjacency matrix with GCN normalization.
    - build_knn_adjacency: Constructs a k-nearest neighbour adjacency matrix on a 2-D grid.
    - build_gcn: Builds a single Graph Convolutional Network (GCN) layer.
    - build_gat: Builds a single Graph Attention (GAT) layer.
    - build_cheb: Builds a single Chebyshev graph convolution layer.

Example:
    >>> from araras.keras.builders.gnn import build_grid_adjacency, build_gcn
    >>> A = build_grid_adjacency(...)
    >>> x_graph, a_graph = GraphMasking(..., A)
    >>> x = build_gcn(..., x_graph, a_graph)

### build_cheb

```python
def build_cheb(trial: Any, kparams: KParams, x: layers.Layer, a_graph: tf.sparse.SparseTensor, units_range: Union[int, Tuple[int, int]], dropout_rate_range: Union[float, Tuple[float, float]], K_range: Union[int, Tuple[int, int]], units_step: int, dropout_rate_step: float, K_step: int, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Build a single Chebyshev graph convolution layer.

### build_gat

```python
def build_gat(trial: Any, kparams: KParams, x: layers.Layer, a_graph: tf.sparse.SparseTensor, units_range: Union[int, Tuple[int, int]], dropout_rate_range: Union[float, Tuple[float, float]], heads_range: Union[int, Tuple[int, int]], units_step: int, dropout_rate_step: float, heads_step: int, concat_heads: bool, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Build a single Graph Attention (GAT) layer.

### build_gcn

```python
def build_gcn(trial: Any, kparams: KParams, x: layers.Layer, a_graph: tf.sparse.SparseTensor, units_range: Union[int, Tuple[int, int]], dropout_rate_range: Union[float, Tuple[float, float]], units_step: int, dropout_rate_step: float, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Build a single Graph Convolutional Network (GCN) layer.

### build_grid_adjacency

```python
def build_grid_adjacency(rows: int, cols: int) -> tf.sparse.SparseTensor
```

Build a grid adjacency matrix with GCN normalization.

Each node is connected to its four direct neighbours (up, down, left and
right).  The resulting adjacency matrix is returned as a TensorFlow sparse
tensor ready to be fed to Spektral layers.

Args:
    rows: Number of grid rows.
    cols: Number of grid columns.

Returns:
    tf.sparse.SparseTensor: Normalized sparse adjacency matrix.

### build_knn_adjacency

```python
def build_knn_adjacency(rows: int, cols: int, k: int) -> tf.sparse.SparseTensor
```

Construct a k-nearest neighbour adjacency matrix on a 2-D grid.

Nodes correspond to cells of a `rows` × `cols` grid.  Each node is
connected to its `k` spatially nearest neighbours.  The adjacency matrix is
symmetrised, normalised with the GCN filter and returned as a TensorFlow
sparse tensor.

Args:
    rows: Number of grid rows.
    cols: Number of grid columns.
    k: Number of neighbours for each node.

Returns:
    tf.sparse.SparseTensor: Normalized sparse adjacency matrix.

### print_warning_jit

```python
def print_warning_jit() -> None
```

Print a warning about JIT compilation.

## araras.keras.builders.lstm

Builders for Long Short-Term Memory (LSTM) networks in Keras.

Functions:
    - build_lstm: Builds a single LSTM block with optional regularization, batch normalization, and dynamic dropout.

Example:
    >>> from araras.keras.builders.lstm import build_lstm
    >>> x = build_lstm(...)

### build_lstm

```python
def build_lstm(trial: Any, kparams: KParams, x: layers.Layer, return_sequences: bool, units_range: Union[int, tuple[int, int]], units_step: int, dropout_rate_range: Union[float, tuple[float, float]], dropout_rate_step: float, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_bias: bool, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Builds a single LSTM block with optional regularization, batch normalization, and dynamic dropout.

This function creates a tunable LSTM layer with optional regularization and batch normalization,
followed by a customizable activation layer. It supports hyperparameter optimization through a tuning trial.

Args:
    trial (Any): Object used for suggesting hyperparameters, typically from a tuner like Optuna.
    kparams (KParams): Hyperparameter manager used to retrieve regularizers and activations.
    x (layers.Layer): Input tensor or layer.
    return_sequences (bool): Whether to return the full sequence of outputs or just the last output.
    units_range (Union[int, tuple[int, int]]): Fixed or tunable number of LSTM units.
    units_step (int): Step size for tuning LSTM units if a range is given.
    dropout_rate_range (Union[float, tuple[float, float]]): Fixed or tunable dropout rate.
    dropout_rate_step (float): Step size for tuning dropout rate.
    kernel_initializer (initializers.Initializer): Initializer for kernel weights.
    bias_initializer (initializers.Initializer): Initializer for biases.
    use_bias (bool): Whether to include a bias term in the LSTM layer.
    use_batch_norm (bool): Whether to apply batch normalization after LSTM.
    trial_kernel_reg (bool): Whether to apply/tune a kernel regularizer.
    trial_bias_reg (bool): Whether to apply/tune a bias regularizer.
    trial_activity_reg (bool): Whether to apply/tune an activity regularizer.
    name_prefix (str): Prefix to use for naming the layers.

Returns:
    layers.Layer: Output tensor after applying the LSTM block.

Raises:
    None

## araras.keras.builders.se

This module provides a function to build a Squeeze-and-Excitation (SE) block with hyperparameter tuning using Optuna.

Based on the paper: https://arxiv.org/pdf/1709.01507

Functions:
    - build_squeeze_excite_1d: Apply a Squeeze-and-Excitation (SE) block 1D with Optuna-tuned hyperparameters.

Example:
    >>> from araras.keras.builders.se import build_squeeze_excite_1d
    >>> x = build_squeeze_excite_1d(...)

### build_squeeze_excite_1d

```python
def build_squeeze_excite_1d(x: tf.keras.layers.Layer, trial: optuna.Trial, kparams: KParams, ratio_choices: List[int], name_prefix: str) -> tf.keras.layers.Layer
```

Apply a Squeeze-and-Excitation (SE) block 1D with Optuna-tuned hyperparameters.

Args:
    x: Input 3D tensor (batch, length, channels).
    trial: Optuna Trial object for suggesting hyperparameters.
    kparams: KParams object containing hyperparameter choices.
    ratio_choices: List of integers representing reduction ratios for SE block.
    name_prefix: Prefix for naming layers and trial parameters.

Returns:
    A tensor the same shape as `x`, re-scaled by the SE attention weights.

Raises:
    ValueError: If `x.shape[-1]` is None (undefined channel dimension).

## araras.keras.builders.skip

This module provides a function to create skip connections in a Keras model using Optuna trials.

Function:
    - trial_skip_connections: Creates skip connections based on a trial object and a list of layers.

Example:
    >>> from araras.keras.skip import trial_skip_connections
    >>> # Assuming `trial` is an Optuna trial object and `layers_list` is a list of Keras layers
    >>> final_tensor = trial_skip_connections(trial, layers_list)

### trial_skip_connections

```python
def trial_skip_connections(trial: optuna.trial.Trial, layers_list: Sequence[tf.Tensor], axis_to_concat: int, print_combinations: bool, strategy: str, merge_mode: str) -> tf.Tensor
```

Constructs conditional skip connections between layers based on Optuna trial choices.

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

## araras.keras.builders.tcnn

Builders for Transposed Convolutional Networks (TCNNs) in Keras.

Functions:
    - build_tcnn1d: Builds a single 1D transposed convolution block with optional batch norm and activation.
    - build_tcnn2d: Builds a single 2D transposed convolution block with optional batch norm and activation.
    - build_tcnn3d: Builds a single 3D transposed convolution block with optional batch norm and activation.

Example:
    >>> from araras.keras.builders.tcnn import build_tcnn1d
    >>> x = build_tcnn1d(...)

### build_tcnn1d

```python
def build_tcnn1d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, tuple[int, int]], filters_step: int, kernel_size_range: Union[int, tuple[int, int]], kernel_size_step: int, data_format: str, padding: str, strides: int, dilation_rate: int, use_bias: bool, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Builds a single 1D transposed convolution block with optional batch norm and activation.

This function constructs a tunable Conv1DTranspose layer, optionally applies batch normalization,
and concludes with a customizable activation function. Hyperparameter tuning is integrated via a `trial` object.

Args:
    trial (Any): Hyperparameter tuning object, such as from Optuna.
    kparams (KParams): Hyperparameter manager providing activation and regularizer configurations.
    x (layers.Layer): Input layer/tensor to process.
    filters_range (Union[int, tuple[int, int]]): Fixed or tunable number of filters.
    filters_step (int): Step size for filter tuning.
    kernel_size_range (Union[int, tuple[int, int]]): Fixed or tunable kernel size.
    kernel_size_step (int): Step size for kernel size tuning.
    data_format (str): Format of the input data (e.g., "channels_last").
    padding (str): Type of padding to use in the convolution (e.g., "same" or "valid").
    strides (int): Stride length of the convolution.
    dilation_rate (int): Dilation rate for dilated convolution.
    use_bias (bool): Whether to include a bias term in the Conv1DTranspose layer.
    kernel_initializer (initializers.Initializer): Initializer for kernel weights.
    bias_initializer (initializers.Initializer): Initializer for bias values.
    use_batch_norm (bool): Whether to apply batch normalization.
    trial_kernel_reg (bool): Whether to enable and tune kernel regularization.
    trial_bias_reg (bool): Whether to enable and tune bias regularization.
    trial_activity_reg (bool): Whether to enable and tune activity regularization.
    name_prefix (str): Prefix used for naming all internal layers.

Returns:
    layers.Layer: Final output tensor after applying the Conv1DTranspose, optional batch norm, and activation.

Raises:
    None

### build_tcnn2d

```python
def build_tcnn2d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, tuple[int, int]], filters_step: int, kernel_size_range: Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]], kernel_size_step: int, data_format: str, padding: str, strides: tuple[int, int], dilation_rate: tuple[int, int], use_bias: bool, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Builds a single 2D transposed convolution block with optional batch norm and activation.

This function constructs a tunable Conv2DTranspose layer, optionally applies batch normalization,
and concludes with a customizable activation function. Hyperparameter tuning is integrated via a `trial` object.

Args:
    trial (Any): Hyperparameter tuning object, such as from Optuna.
    kparams (KParams): Hyperparameter manager providing activation and regularizer configurations.
    x (layers.Layer): Input layer/tensor to process.
    filters_range (Union[int, tuple[int, int]]): Fixed or tunable number of filters.
    filters_step (int): Step size for filter tuning.
    kernel_size_range (Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]):
        Fixed (height, width) or ranges ((h_min, h_max), (w_min, w_max)) for tuning.
    kernel_size_step (int): Step size for kernel size tuning.
    data_format (str): Format of the input data (e.g., "channels_last").
    padding (str): Type of padding to use in the convolution (e.g., "same" or "valid").
    strides (tuple[int, int]): Stride length for height and width.
    dilation_rate (tuple[int, int]): Dilation rate for height and width.
    use_bias (bool): Whether to include a bias term in the Conv2DTranspose layer.
    kernel_initializer (initializers.Initializer): Initializer for kernel weights.
    bias_initializer (initializers.Initializer): Initializer for bias values.
    use_batch_norm (bool): Whether to apply batch normalization.
    trial_kernel_reg (bool): Whether to enable and tune kernel regularization.
    trial_bias_reg (bool): Whether to enable and tune bias regularization.
    trial_activity_reg (bool): Whether to enable and tune activity regularization.
    name_prefix (str): Prefix used for naming all internal layers.

Returns:
    layers.Layer: Final output tensor after applying the Conv2DTranspose, optional batch norm, and activation.

### build_tcnn3d

```python
def build_tcnn3d(trial: Any, kparams: KParams, x: layers.Layer, filters_range: Union[int, Tuple[int, int]], filters_step: int, kernel_size_range: Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], kernel_size_step: int, data_format: str, padding: str, strides: Tuple[int, int, int], dilation_rate: Tuple[int, int, int], use_bias: bool, kernel_initializer: initializers.Initializer, bias_initializer: initializers.Initializer, use_batch_norm: bool, trial_kernel_reg: bool, trial_bias_reg: bool, trial_activity_reg: bool, name_prefix: str) -> layers.Layer
```

Builds a single 3D transposed convolution block with optional batch norm and activation.

This function constructs a tunable Conv3DTranspose layer, optionally applies batch normalization,
and concludes with a customizable activation function. Hyperparameter tuning is integrated via a `trial` object.

Args:
    trial (Any): Hyperparameter tuning object, such as from Optuna.
    kparams (KParams): Hyperparameter manager providing activation and regularizer configurations.
    x (layers.Layer): Input layer/tensor to process.
    filters_range (Union[int, Tuple[int, int]]): Fixed or tunable number of filters.
    filters_step (int): Step size for filter tuning.
    kernel_size_range (Union[
        Tuple[int, int, int],
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ]): Fixed (depth, height, width) or ranges ((d_min, d_max), (h_min, h_max), (w_min, w_max)) for tuning.
    kernel_size_step (int): Step size for kernel size tuning.
    data_format (str): Format of the input data (e.g., "channels_last").
    padding (str): Type of padding to use in the convolution (e.g., "same" or "valid").
    strides (Tuple[int, int, int]): Stride length for depth, height, and width.
    dilation_rate (Tuple[int, int, int]): Dilation rate for depth, height, and width.
    use_bias (bool): Whether to include a bias term in the Conv3DTranspose layer.
    kernel_initializer (initializers.Initializer): Initializer for kernel weights.
    bias_initializer (initializers.Initializer): Initializer for bias values.
    use_batch_norm (bool): Whether to apply batch normalization.
    trial_kernel_reg (bool): Whether to enable and tune kernel regularization.
    trial_bias_reg (bool): Whether to enable and tune bias regularization.
    trial_activity_reg (bool): Whether to enable and tune activity regularization.
    name_prefix (str): Prefix used for naming all internal layers.

Returns:
    layers.Layer: Final output tensor after applying the Conv3DTranspose, optional batch norm, and activation.

## araras.keras.callbacks

Keras callback for pruning Optuna trials when the training loss becomes NaN.
It does the same as `keras.callbacks.TerminateOnNaN()`, but also reports the NaN loss to Optuna.

Classes:
    - NanLossPrunerOptuna: Stops a trial once `loss` is NaN.

Functions:
    - get_callbacks_study: Returns a list of Keras callbacks for Optuna trials.
    - get_callbacks_model: Returns a list of Keras callbacks for model training.

Example:
    >>> from araras.keras.callbacks.nan_loss_pruner import NanLossPrunerOptuna
    >>> NanLossPrunerOptuna(trial)

### get_callbacks_model

```python
def get_callbacks_model(backup_dir: str, tensorboard_logs: str) -> List[tf.keras.callbacks.Callback]
```

Constructs and returns a list of Keras callbacks for model training.

Args:
    backup_dir (str): Directory where the backup files will be stored.
    tensorboard_logs (str): Directory where TensorBoard logs will be stored.

Returns:
    List[tf.keras.callbacks.Callback]: A list of callbacks to pass into `model.fit()`.

### get_callbacks_study

```python
def get_callbacks_study(trial: optuna.Trial, tensorboard_logs: str, monitor: str) -> List[tf.keras.callbacks.Callback]
```

Constructs and returns a list of Keras callbacks tailored for Optuna trials.

Args:
    trial (optuna.Trial): The current Optuna trial object.
    tensorboard_logs (str): Directory where TensorBoard logs will be stored.
    monitor (str): The metric to monitor for early stopping and learning rate reduction.

Returns:
    List[tf.keras.callbacks.Callback]: A list of callbacks to pass into `model.fit()`.

### class NanLossPrunerOptuna

A custom Keras callback that prunes an Optuna trial if NaN is encountered in training loss.

This is useful for skipping unpromising model configurations early, especially
those that are unstable or diverging during training.

Args:
    trial (optuna.Trial): The Optuna trial associated with this model run.

Example:
    model.fit(..., callbacks=[NanLossPrunerOptuna(trial)])

#### on_epoch_end

```python
def on_epoch_end(epoch: int, logs: dict) -> None
```

Called automatically at the end of each training epoch.

If training loss is NaN, the trial is reported and pruned.

Args:
    epoch (int): Index of the current epoch.
    logs (dict, optional): Metric results from the epoch (e.g., {"loss": ..., "val_loss": ...}).

## araras.keras.hyperparams

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
    ...     activation_choices={"relu": tf.keras.activations.relu, "tanh": tf.keras.activations.tanh},
    ...     regularizer_choices={"l2": tf.keras.regularizers.L2(1e-3)},
    ...     optimizer_choices={"adam": tf.keras.optimizers.Adam},
    ...     scaler_choices={"standard": StandardScaler, "minmax": MinMaxScaler},
    ...     initializer_choices={"glorot": tf.keras.initializers.GlorotUniform},
    ... )

### class ActivationSampler

### class BaseSampler

Abstract sampler for Keras hyperparameters.

#### sample

```python
def sample(trial: optuna.Trial) -> Any
```

### class InitializerSampler

### class KParams

Container for hyperparameter search spaces.

#### set_activation_choices

```python
def set_activation_choices(choices: Union[Sequence[Any], Mapping[str, Any]]) -> None
```

Set the available activation choices.

#### set_regularizer_choices

```python
def set_regularizer_choices(choices: Union[Sequence[Optional[Union[type[tf.keras.regularizers.Regularizer], tf.keras.regularizers.Regularizer, Callable[[], tf.keras.regularizers.Regularizer]]]], Mapping[str, Any]]) -> None
```

Set the available regularizer choices.

#### set_optimizer_choices

```python
def set_optimizer_choices(choices: Union[Sequence[Union[type[tf.keras.optimizers.Optimizer], tf.keras.optimizers.Optimizer, Callable[[], tf.keras.optimizers.Optimizer]]], Mapping[str, Any]]) -> None
```

Set the available optimizer choices.

#### set_scaler_choices

```python
def set_scaler_choices(choices: Union[Sequence[Any], Mapping[str, Any]]) -> None
```

Set the available scaler choices.

#### set_initializer_choices

```python
def set_initializer_choices(choices: Union[Sequence[Union[type[tf.keras.initializers.Initializer], tf.keras.initializers.Initializer, Callable[[], tf.keras.initializers.Initializer]]], Mapping[str, Any]]) -> None
```

Set the available initializer choices.

#### get_activation

```python
def get_activation(trial: optuna.Trial, name: str) -> Optional[Callable[..., Any]]
```

Sample or return an activation.

#### get_regularizer

```python
def get_regularizer(trial: optuna.Trial, name: str) -> Optional[tf.keras.regularizers.Regularizer]
```

Sample a regularizer.

#### get_optimizer

```python
def get_optimizer(trial: optuna.Trial) -> tf.keras.optimizers.Optimizer
```

Sample an optimizer.

#### get_scaler

```python
def get_scaler(trial: optuna.Trial) -> None
```

Sample a scikit-learn scaler.

#### get_initializer

```python
def get_initializer(trial: optuna.Trial, name: str) -> tf.keras.initializers.Initializer
```

Sample a kernel initializer.

#### get_default_params

```python
def get_default_params(cls: Any) -> 'KParams'
```

Return an instance with all default options.

#### default

```python
def default(cls: Any) -> 'KParams'
```

Alias for :meth:`get_default_params`.

#### full_search_space

```python
def full_search_space(cls: Any) -> 'KParams'
```

Return an instance of the class with all available options from Keras and their default values for the search space.
Excluded options:
    - OrthogonalRegularizer

### class OptimizerSampler

### class RegularizerSampler

### class ScalerSampler

## araras.keras.utils

A Collection on tools for using with Keras.

Funtions:
    - convert_to_saved_model: Convert a Keras `.keras` model archive into a zipped TensorFlow SavedModel.
    - capture_model_summary: Capture model summary as a string.
    - punish_model_flops: Penalize an objective according to the model's FLOPs.
    - punish_model_params: Penalize an objective according to the model's parameter count.
    - punish_model: A convenience function to apply both FLOPs and parameter penalties to an objective.

### capture_model_summary

```python
def capture_model_summary(model: Any) -> None
```

Capture model summary as a string.

Args:
    model: Keras model

Returns:
    str: Model summary as string

### convert_to_saved_model

```python
def convert_to_saved_model(input_keras_path: str, output_zip_path: str) -> None
```

Convert a Keras `.keras` model archive into a zipped TensorFlow SavedModel.

This will load the model, export it in SavedModel directory format,
then compress that directory into a .zip file.

Args:
    input_keras_path (str): Path to the source `.keras` model file.
    output_zip_path (str): Desired path for the output zip (e.g. 'saved_model.zip').

Returns:
    None

Raises:
    Any exception raised by TensorFlow I/O (e.g. file not found, load/save errors).

### punish_model

```python
def punish_model(target: Union[float, Sequence[float]], model: tf.keras.Model, type: Literal['flops', 'params', None], flops_penalty_factor: float, params_penalty_factor: float, direction: Literal['minimize', 'maximize']) -> Union[float, Sequence[float]]
```

Apply both FLOPs and parameter penalties to an objective.

Args:
    target: Base objective value (scalar or list of scalars).
    model: Model whose complexity will be penalised.
    type: Type of penalty to apply, either "flops" or "params".
    flops_penalty_factor: Factor for FLOPs penalty.
    params_penalty_factor: Factor for parameters penalty.
    direction: Whether the objective should be minimised or maximised.

Returns:
    The penalised objective value or list of values.

### punish_model_flops

```python
def punish_model_flops(target: Union[float, Sequence[float]], model: tf.keras.Model, penalty_factor: float, direction: Literal['minimize', 'maximize']) -> Union[float, Sequence[float]]
```

Penalize an objective according to the model's FLOPs.

Args:
    target: Base objective value (scalar or list of scalars).
    model: Model whose FLOPs will be used for the penalty.
    penalty_factor: Multiplicative factor applied to the FLOPs count.
    direction: Whether the objective should be minimised or maximised.

Returns:
    The penalised objective value or list of values.

### punish_model_params

```python
def punish_model_params(target: Union[float, Sequence[float]], model: tf.keras.Model, penalty_factor: float, direction: Literal['minimize', 'maximize']) -> Union[float, Sequence[float]]
```

Penalize an objective according to the model's parameter count.

Args:
    target: Base objective value (scalar or list of scalars).
    model: Model whose parameters will be used for the penalty.
    penalty_factor: Multiplicative factor applied to the parameter count.
    direction: Whether the objective should be minimised or maximised.

Returns:
    The penalised objective value or list of values.

## araras.kernel.__init__

## araras.kernel.consolidated_email_manager

Manage consolidated email notifications for restart events.

Classes:
    - ConsolidatedEmailManager: Sends aggregated status emails with retry logic.

Example:
    >>> from araras.kernel.consolidated_email_manager import ConsolidatedEmailManager
    >>> manager = ConsolidatedEmailManager()
    >>> manager.send_consolidated_status_email("task_complete", {})

### class ConsolidatedEmailManager

Handles consolidated email notifications with retry logic.

#### send_consolidated_status_email

```python
def send_consolidated_status_email(status_type: str, process_data: Dict[str, Any]) -> None
```

#### should_attempt_restart

```python
def should_attempt_restart(title: str, restart_count: int, max_restarts: int) -> bool
```

#### report_successful_restart

```python
def report_successful_restart(title: str, old_pid: Optional[int], new_pid: int, restart_count: int, runtime: float) -> None
```

#### report_task_completion

```python
def report_task_completion(title: str, restart_count: int, total_runtime: float) -> None
```

#### report_final_failure

```python
def report_final_failure(title: str, restart_count: int, error: str) -> None
```

## araras.kernel.file_type_handler

Utility class for detecting file types and building execution commands.

Classes:
    - FileTypeHandler: Provides methods to identify file types and construct
      commands for running them.

Example:
    >>> from araras.kernel.file_type_handler import FileTypeHandler
    >>> FileTypeHandler.build_execution_command(Path("train.py"), "success.txt")

### class FileTypeHandler

File type detection and command generation with caching.

#### get_file_type

```python
def get_file_type(cls: Any, file_path: Path) -> str
```

Return the file type for a path.

#### build_execution_command

```python
def build_execution_command(cls: Any, file_path: Path, success_flag_file: str) -> Tuple[List[str], str]
```

Build execution command based on file type.

#### validate_file

```python
def validate_file(cls: Any, file_path: str) -> Path
```

Validate file existence and type.

## araras.kernel.flag_based_restart_manager

Restart manager based on flag files.

Classes:
    - FlagBasedRestartManager: Handles auto-restart logic and monitoring for a
      target script.

Example:
    >>> from araras.kernel.flag_based_restart_manager import FlagBasedRestartManager
    >>> manager = FlagBasedRestartManager()
    >>> manager.run_file_with_restart("train.py")

### class FlagBasedRestartManager

Enhanced restart manager with consolidated email notifications and retry logic.

#### run_file_with_restart

```python
def run_file_with_restart(file_path: str, success_flag_file: str, title: Optional[str], restart_after_delay: Optional[float], supress_tf_warnings: bool) -> None
```

Run file with flag-based restart logic and consolidated email notifications.

Args:
    file_path: Path to Python or Jupyter notebook file
    success_flag_file: Path where target process writes completion flag
    title: Custom title for monitoring
    restart_after_delay: Optional delay after which the run will be restarted
    supress_tf_warnings: Suppress TensorFlow warnings (default: False)

Raises:
    FileNotFoundError: If file doesn't exist
    ValueError: If file type is unsupported

#### force_stop

```python
def force_stop() -> None
```

Request the currently running loop to stop and cleanup.

## araras.kernel.monitoring

This module provides a restarting monitoring system for processes with email alert capabilities.

Functions:
    - print_monitoring_config_summary: Print a summary of monitoring configuration only once.
    - print_process_status: Print process status messages with consistent formatting.
    - print_restart_info: Print restart information with formatting.
    - print_completion_summary: Print final completion summary.
    - print_error_message: Print error messages with consistent formatting.
    - print_warning_message: Print warning messages with consistent formatting.
    - print_success_message: Print success messages with consistent formatting.
    - print_cleanup_info: Print child process cleanup information.
    - _cleanup_stale_monitor_files: No description.
    - get_process_resource_usage: Return memory percentage, memory in GB, and CPU percentage for a process.
    - print_process_resource_usage: Display CPU and memory usage for a process in a single updating line.
    - start_monitor: Start simplified crash monitor without email capabilities.
    - stop_monitor: Stop monitor and cleanup files with optimized batch operations.
    - check_crash_signal: Check if process crashed with minimal I/O operations.
    - run_auto_restart: Main function with notebook conversion, file cleanup, and consolidated email notification support.

Example:
    >>> from araras.kernel.monitoring import run_auto_restart
    >>> run_auto_restart("train.py", title="Training Process")

### check_crash_signal

```python
def check_crash_signal(monitor_info: Dict[str, Any]) -> Dict[str, Any]
```

Check if process crashed with minimal I/O operations.

Args:
    monitor_info: Monitor control info

Returns:
    Dictionary with crash info or empty dict if no crash

### get_process_resource_usage

```python
def get_process_resource_usage(pid: int) -> Tuple[float, float, float]
```

Return memory percentage, memory in GB, and CPU percentage for a process.

Args:
    pid: Process ID of the process to query.

Returns:
    Tuple containing memory percentage, memory usage in GB and CPU percentage.

Raises:
    psutil.NoSuchProcess: If the PID does not exist.

### print_cleanup_info

```python
def print_cleanup_info(terminated: int, killed: int) -> None
```

Print child process cleanup information.

### print_completion_summary

```python
def print_completion_summary(restart_count: int, total_runtime: Optional[float]) -> None
```

Print final completion summary.

### print_error_message

```python
def print_error_message(error_type: str, message: str) -> None
```

Print error messages with consistent formatting.

### print_monitoring_config_summary

```python
def print_monitoring_config_summary(file_path: str, file_type: str, success_flag_file: str, max_restarts: int, email_enabled: bool, title: str, restart_after_delay: Optional[float]) -> None
```

Print a summary of monitoring configuration only once.

### print_process_resource_usage

```python
def print_process_resource_usage(pid: int) -> None
```

Display CPU and memory usage for a process in a single updating line.

### print_process_status

```python
def print_process_status(message: str, pid: Optional[int], runtime: Optional[float]) -> None
```

Print process status messages with consistent formatting.

### print_restart_info

```python
def print_restart_info(restart_count: int, max_restarts: int, delay: float) -> None
```

Print restart information with formatting.

### print_success_message

```python
def print_success_message(message: str) -> None
```

Print success messages with consistent formatting.

### print_warning_message

```python
def print_warning_message(message: str) -> None
```

Print warning messages with consistent formatting.

### run_auto_restart

```python
def run_auto_restart(file_path: str, success_flag_file: str, title: Optional[str], max_restarts: int, restart_delay: float, recipients_file: Optional[str], credentials_file: Optional[str], restart_after_delay: Optional[float], retry_attempts: int, supress_tf_warnings: bool, resource_usage_log_file: Optional[str]) -> None
```

Main function with notebook conversion, file cleanup, and consolidated email notification support.

Args:
    file_path: Path to .py or .ipynb file to execute
    success_flag_file: Path to success flag file
    title: Custom title for monitoring and email alerts
    max_restarts: Maximum restart attempts
    restart_delay: Delay between restarts in seconds
    recipients_file: Path to recipients JSON file (defaults to ./json/recipients.json)
    credentials_file: Path to credentials JSON file (defaults to ./json/credentials.json)
    restart_after_delay: restart the run after a delay in seconds
    retry_attempts: Number of retry attempts before sending failure email
    supress_tf_warnings: Suppress TensorFlow warnings (default: False)
    resource_usage_log_file: Path to write process resource usage logs. If None, logging is disabled.

Raises:
    FileNotFoundError: If file doesn't exist
    ValueError: If file type is unsupported
    ImportError: If notebook dependencies missing for .ipynb files

### start_monitor

```python
def start_monitor(pid: int, title: str, supress_tf_warnings: bool) -> Dict[str, Any]
```

Start simplified crash monitor without email capabilities.

Args:
    pid: Process ID to monitor
    title: Process title for alerts
    supress_tf_warnings: Suppress TensorFlow warnings (default: False)

Returns:
    Monitor control info dictionary

Raises:
    ValueError: If PID doesn't exist
    OSError: If monitor startup fails

### stop_monitor

```python
def stop_monitor(monitor_info: Dict[str, Any]) -> None
```

Stop monitor and cleanup files with optimized batch operations.

Args:
    monitor_info: Monitor control info from start_monitor()

## araras.optuna.__init__

## araras.optuna.analysis.__init__

## araras.optuna.analysis.analyzer

Utility functions for analyzing Optuna study results.

Functions:
    - set_plot_config_param: Set a single parameter in the global PlotConfig.
    - analyze_study: Comprehensive analysis of Optuna hyperparameter optimization study results.

Example:
    >>> from araras.optuna.analysis.analyze import analyze_study
    >>> analyze_study("path/to/study")

### analyze_study

```python
def analyze_study(study: optuna.Study, table_dir: str, top_frac: float, param_name_mapping: Dict[str, str], create_standalone: bool, save_data: bool, create_plotly: bool, plots: Optional[List[str]]) -> None
```

Comprehensive analysis of Optuna hyperparameter optimization study results.

Args:
    study: Optuna study object containing trials to analyze.
    table_dir: Directory to save analysis results and figures.
    top_frac: Fraction of best/worst trials to analyze (default: 0.2).
    param_name_mapping: Optional mapping of parameter names to display names.
        Example: {'params_learning_rate': 'Learning Rate'}
    create_standalone: If True, generates standalone images for each plot type.
    save_data: If True, saves data for LaTeX plotting into CSV files.
    create_plotly: If True, also saves interactive Plotly HTML versions of the figures.
    plots: List of plot types to generate. Available options:
        'distributions', 'importances', 'correlations', 'boxplots',
        'trends', 'ranges', 'contours', 'edf', 'intermediate',
        'parallel_coordinate', 'slice', 'rank', 'history', 'timeline',
        'terminator'.
        Deactivated by default:
            - 'parallel_coordinate' (Too much of a mess to be useful)
            - 'rank' (Can cause crashes and not very useful)
        If None, generates all plots.

### calculate_grid

```python
def calculate_grid(n_plots: int, subplot_width: int, subplot_height: int, base_max_cols: int) -> Tuple[int, int]
```

Calculate grid dimensions ensuring the resulting figure stays within
Matplotlib's maximum image size.

Parameters
----------
n_plots : int
    Number of subplots to create.
subplot_width : int
    Width of each subplot in inches.
subplot_height : int
    Height of each subplot in inches.
base_max_cols : int
    Desired number of columns before auto-adjustment.

Returns
-------
Tuple[int, int]
    (n_rows, n_cols) suitable for ``plt.subplots``.

### classify_columns

```python
def classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]
```

Split DataFrame columns into numeric and categorical parameter types.

### create_directories

```python
def create_directories(table_dir: str, create_standalone: bool, save_data: bool, create_plotly: bool) -> Dict[str, str]
```

Create organized subdirectories for storing analysis outputs.

### draw_warning_box

```python
def draw_warning_box(ax: plt.Axes, message: str) -> None
```

Display a warning message inside a plot area.

### format_numeric_value

```python
def format_numeric_value(x: float) -> Union[int, float, str]
```

Format numeric values with appropriate precision for readability.

### format_title

```python
def format_title(template: str, display_name: str) -> str
```

Format a title template with the given display name.

### get_param_display_name

```python
def get_param_display_name(param_name: str, param_name_mapping: Dict[str, str]) -> str
```

Get display name for parameter, using mapping if provided.

### get_trial_subsets

```python
def get_trial_subsets(df: pd.DataFrame, top_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]
```

Extract best and worst performing trial subsets based on loss values.

### prepare_dataframe

```python
def prepare_dataframe(study: optuna.Study) -> pd.DataFrame
```

Extract and clean completed trial data from Optuna study.

### print_study_columns

```python
def print_study_columns(study: optuna.Study, exclude: Optional[List[str]], param_name_mapping: Optional[Dict[str, str]]) -> None
```

Print the names of the DataFrame columns from the study as a bullet list.

### save_data_for_latex

```python
def save_data_for_latex(data_dict: Dict[str, Any], filename: str, data_dir: str) -> None
```

Save graph data to CSV files for LaTeX plotting.

### save_plot

```python
def save_plot(fig: plt.Figure, dirs: Dict[str, str], base_name: str, subdir_key: str, create_plotly: bool, plotly_fig: Any) -> None
```

Save Matplotlib figure and optionally a Plotly HTML version.

### save_plotly_html

```python
def save_plotly_html(fig: Any, filepath: str) -> None
```

Save a Plotly figure to an HTML file.

### save_summary_tables

```python
def save_summary_tables(df: pd.DataFrame, best: pd.DataFrame, worst: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], dirs: Dict[str, str]) -> None
```

Generate and save statistical summary tables for different trial subsets.

### set_plot_config_param

```python
def set_plot_config_param(param_name: str, value: Any) -> None
```

Set a single parameter in :data:`PLOT_CFG`.

### set_plot_config_params

```python
def set_plot_config_params(**kwargs: Any) -> None
```

Set multiple parameters in :data:`PLOT_CFG`.

### class PlotConfig

Global configuration for matplotlib plots used in this module.

## araras.optuna.analysis.create_frequency_table

Module create_frequency_table of analysis

Functions:
    - create_frequency_table: Generate frequency tables for categorical hyperparameters.

Example:
    >>> from araras.optuna.analysis.create_frequency_table import create_frequency_table
    >>> create_frequency_table(...)

### create_frequency_table

```python
def create_frequency_table(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame
```

Generate frequency tables for categorical hyperparameters.

## araras.optuna.analysis.describe_numeric

Module describe_numeric of analysis

Functions:
    - describe_numeric: Generate descriptive statistics for numeric hyperparameters.

Example:
    >>> from araras.optuna.analysis.describe_numeric import describe_numeric
    >>> describe_numeric(...)

### describe_numeric

```python
def describe_numeric(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame
```

Generate descriptive statistics for numeric hyperparameters.

## araras.optuna.analysis.plot_contour

Module plot_contour of analysis

Functions:
    - plot_contour: Generate contour plots for parameter pairs.

Example:
    >>> from araras.optuna.analysis.plot_contour import plot_contour
    >>> plot_contour(...)

### plot_contour

```python
def plot_contour(study: optuna.Study, params: List[str], dirs: Dict[str, str], create_standalone: bool, create_plotly: bool) -> None
```

Generate contour plots for parameter pairs.

This creates a single multipanel figure covering all provided parameters
and optionally standalone figures for each pair of parameters.

Parameters
----------
create_plotly : bool
    Whether to save interactive HTML versions of the plots.

## araras.optuna.analysis.plot_edf

Module plot_edf of analysis

Functions:
    - plot_edf: Plot the empirical distribution function of objective values.

Example:
    >>> from araras.optuna.analysis.plot_edf import plot_edf
    >>> plot_edf(...)

### plot_edf

```python
def plot_edf(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool) -> None
```

Plot the empirical distribution function of objective values.

Parameters
----------
create_plotly : bool
    Whether to save an interactive HTML version of the plot.

## araras.optuna.analysis.plot_hyperparameter_distributions

Module plot_hyperparameter_distributions of analysis

Functions:
    - plot_hyperparameter_distributions: Generate and save distribution plots for numeric and categorical hyperparameters in separate figures.

Example:
    >>> from araras.optuna.analysis.plot_hyperparameter_distributions import plot_hyperparameter_distributions
    >>> plot_hyperparameter_distributions(...)

### plot_hyperparameter_distributions

```python
def plot_hyperparameter_distributions(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], dirs: Dict[str, str], param_name_mapping: Dict[str, str], create_standalone: bool, create_plotly: bool) -> None
```

Generate and save distribution plots for numeric and categorical hyperparameters in separate figures.

For numeric parameters, a KDE curve is estimated prior to plotting. If the
KDE computation fails (e.g., due to singular covariance or insufficient
unique values), the parameter plot is replaced with a placeholder message so
that the remaining plots can still be generated.

Args:
    df (pd.DataFrame): DataFrame containing hyperparameter data
    numeric_cols (List[str]): List of numeric column names
    categorical_cols (List[str]): List of categorical column names
    dirs (Dict[str, str]): Dictionary of directory paths for saving plots
    param_name_mapping (Dict[str, str]): Optional mapping for parameter display names
create_standalone (bool): Whether to create standalone images for each parameter
create_plotly (bool): Whether to save interactive HTML versions

## araras.optuna.analysis.plot_intermediate_values

Module plot_intermediate_values of analysis

Functions:
    - plot_intermediate_values: Plot intermediate values reported during trials.

Example:
    >>> from araras.optuna.analysis.plot_intermediate_values import plot_intermediate_values
    >>> plot_intermediate_values(...)

### plot_intermediate_values

```python
def plot_intermediate_values(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool) -> None
```

Plot intermediate values reported during trials.

Parameters
----------
create_plotly : bool
    Whether to save an interactive HTML version of the plot.

## araras.optuna.analysis.plot_optimal_ranges_analysis

Module plot_optimal_ranges_analysis of analysis

Functions:
    - plot_optimal_ranges_analysis: Create a single comprehensive visualization showing optimal parameter ranges based on best-performing trials.

Example:
    >>> from araras.optuna.analysis.plot_optimal_ranges_analysis import plot_optimal_ranges_analysis
    >>> plot_optimal_ranges_analysis(...)

### plot_optimal_ranges_analysis

```python
def plot_optimal_ranges_analysis(df: pd.DataFrame, best: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str], param_name_mapping: Dict[str, str], create_standalone: bool, create_plotly: bool) -> None
```

Create a single comprehensive visualization showing optimal parameter ranges based on best-performing trials.

This function generates a single plot with subplots for each parameter showing the distribution of parameters
in all trials versus best trials, with indicators for conservative and aggressive
optimal ranges, plus the median of best trials.

Args:
    df (pd.DataFrame): Complete dataset with all trials
    best (pd.DataFrame): Subset of best-performing trials
    numeric_cols (List[str]): List of numeric parameter column names
    dirs (Dict[str, str]): Directory paths for saving outputs
    param_name_mapping (Dict[str, str]): Optional mapping for parameter display names
    create_standalone (bool): Whether to create standalone images for each parameter
    create_plotly (bool): Whether to save interactive HTML versions

Returns:
    None: Saves the optimal ranges visualization to fig_ranges directory

## araras.optuna.analysis.plot_optimization_history

Module plot_optimization_history of analysis

Functions:
    - plot_optimization_history: Plot optimization history of the study.

Example:
    >>> from araras.optuna.analysis.plot_optimization_history import plot_optimization_history
    >>> plot_optimization_history(...)

### plot_optimization_history

```python
def plot_optimization_history(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool) -> None
```

Plot optimization history of the study.

Parameters
----------
create_plotly : bool
    Whether to save an interactive HTML version of the plot.

## araras.optuna.analysis.plot_parallel_coordinate

Module plot_parallel_coordinate of analysis

Functions:
    - plot_parallel_coordinate: Create a parallel coordinate plot for trials.

Example:
    >>> from araras.optuna.analysis.plot_parallel_coordinate import plot_parallel_coordinate
    >>> plot_parallel_coordinate(...)

### plot_parallel_coordinate

```python
def plot_parallel_coordinate(study: optuna.Study, params: List[str], dirs: Dict[str, str], create_plotly: bool) -> None
```

Create a parallel coordinate plot for trials.

Parameters
----------
create_plotly : bool
    Whether to save an interactive HTML version of the plot.

## araras.optuna.analysis.plot_param_importances

Module plot_param_importances of analysis

Functions:
    - plot_param_importances: Generate and save parameter importance analysis.

Example:
    >>> from araras.optuna.analysis.plot_param_importances import plot_param_importances
    >>> plot_param_importances(...)

### plot_param_importances

```python
def plot_param_importances(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool) -> None
```

Generate and save parameter importance analysis.

This function computes parameter importances using Optuna's built-in
importance calculation and creates both a CSV table and bar chart
visualization to identify which parameters most influence the objective.

Args:
    study (optuna.Study): Optuna study object containing optimization history
    dirs (Dict[str, str]): Directory paths for saving outputs
    create_plotly (bool): Whether to save an interactive HTML version

Returns:
    None: Saves importance table as CSV and bar chart as pdf

## araras.optuna.analysis.plot_parameter_boxplots

Module plot_parameter_boxplots of analysis

Functions:
    - plot_parameter_boxplots: Create separate comprehensive boxplot comparisons for numeric parameters across trial subsets.

Example:
    >>> from araras.optuna.analysis.plot_parameter_boxplots import plot_parameter_boxplots
    >>> plot_parameter_boxplots(...)

### plot_parameter_boxplots

```python
def plot_parameter_boxplots(df: pd.DataFrame, best: pd.DataFrame, worst: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str], param_name_mapping: Dict[str, str], create_standalone: bool, create_plotly: bool) -> None
```

Create separate comprehensive boxplot comparisons for numeric parameters across trial subsets.

Args:
    df (pd.DataFrame): Complete dataset with all trials
    best (pd.DataFrame): Subset of best-performing trials
    worst (pd.DataFrame): Subset of worst-performing trials
    numeric_cols (List[str]): List of numeric parameter column names
    dirs (Dict[str, str]): Directory paths for saving outputs
    param_name_mapping (Dict[str, str]): Optional mapping for parameter display names
    create_standalone (bool): Whether to create standalone images for each parameter
    create_plotly (bool): Whether to save interactive HTML versions

Returns:
    None: Saves separate boxplot files for numeric parameters

## araras.optuna.analysis.plot_rank

Module plot_rank of analysis

Functions:
    - plot_rank: Plot parameter relations colored by rank.

Example:
    >>> from araras.optuna.analysis.plot_rank import plot_rank
    >>> plot_rank(...)

### make_rank_plotly

```python
def make_rank_plotly(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> None
```

### plot_rank

```python
def plot_rank(study: optuna.Study, params: List[str], dirs: Dict[str, str], create_standalone: bool, create_plotly: bool) -> None
```

Plot parameter relations colored by rank.

Parameters
----------
create_plotly : bool
    Whether to save interactive HTML versions of the plots.

## araras.optuna.analysis.plot_slice

Module plot_slice of analysis

Functions:
    - plot_slice: Create slice plots for each parameter.

Example:
    >>> from araras.optuna.analysis.plot_slice import plot_slice
    >>> plot_slice(...)

### plot_slice

```python
def plot_slice(study: optuna.Study, params: List[str], dirs: Dict[str, str], create_standalone: bool, create_plotly: bool) -> None
```

Create slice plots for each parameter.

Parameters
----------
create_plotly : bool
    Whether to save interactive HTML versions of the plots.

## araras.optuna.analysis.plot_spearman_correlation

Module plot_spearman_correlation of analysis

Functions:
    - plot_spearman_correlation: Generate and save Spearman correlation heatmap for numeric parameters and loss.

Example:
    >>> from araras.optuna.analysis.plot_spearman_correlation import plot_spearman_correlation
    >>> plot_spearman_correlation(...)

### plot_spearman_correlation

```python
def plot_spearman_correlation(df: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str], create_plotly: bool) -> None
```

Generate and save Spearman correlation heatmap for numeric parameters and loss.

This function computes rank-based correlations between all numeric parameters
and the loss function, creating a heatmap visualization to identify
relationships between parameters and their impact on optimization performance.

Args:
    df (pd.DataFrame): Dataset containing numeric parameters and loss values
    numeric_cols (List[str]): List of numeric parameter column names
    dirs (Dict[str, str]): Directory paths for saving outputs
    create_plotly (bool): Whether to save an interactive HTML version

Returns:
    None: Saves correlation heatmap as pdf file

## araras.optuna.analysis.plot_terminator_improvement

Module plot_terminator_improvement of analysis

Functions:
    - _get_improvement_info: No description.
    - _get_y_range: No description.
    - plot_terminator_improvement: Plot the potentials for future objective improvement using Matplotlib.

Example:
    >>> from araras.optuna.analysis.plot_terminator_improvement import _get_improvement_info
    >>> _get_improvement_info(...)

### plot_terminator_improvement

```python
def plot_terminator_improvement(study: optuna.Study, dirs: Dict[str, str], plot_error: bool, print_variance: bool, improvement_evaluator: Optional[BaseImprovementEvaluator], error_evaluator: Optional[BaseErrorEvaluator], min_n_trials: int, create_plotly: bool) -> None
```

Plot the potentials for future objective improvement using Matplotlib.

Parameters
----------
create_plotly : bool
    Whether to save an interactive HTML version of the plot.

### class _ImprovementInfo

## araras.optuna.analysis.plot_timeline

Module plot_timeline of analysis

Functions:
    - plot_timeline: Visualize trial durations on a timeline with detailed information.

Example:
    >>> from araras.optuna.analysis.plot_timeline import plot_timeline
    >>> plot_timeline(...)

### plot_timeline

```python
def plot_timeline(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool) -> None
```

Visualize trial durations on a timeline with detailed information.

Parameters
----------
create_plotly : bool
    Whether to save an interactive HTML version of the plot.

## araras.optuna.analysis.plot_trend_analysis

Module plot_trend_analysis of analysis

Functions:
    - plot_trend_analysis: Create a single comprehensive plot with trend analysis for parameter-loss relationships.

Example:
    >>> from araras.optuna.analysis.plot_trend_analysis import plot_trend_analysis
    >>> plot_trend_analysis(...)

### plot_trend_analysis

```python
def plot_trend_analysis(df: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str], param_name_mapping: Dict[str, str], create_standalone: bool, create_plotly: bool) -> None
```

Create a single comprehensive plot with trend analysis for parameter-loss relationships.

This function generates a single plot with subplots showing the relationship between
each numeric parameter and the loss function, fitting linear trends
to identify parameter directions that improve performance.

Args:
    df (pd.DataFrame): Dataset containing parameters and loss values
    numeric_cols (List[str]): List of numeric parameter column names
    dirs (Dict[str, str]): Directory paths for saving outputs
    param_name_mapping (Dict[str, str]): Optional mapping for parameter display names
    create_standalone (bool): Whether to create standalone images for each parameter
    create_plotly (bool): Whether to save interactive HTML versions

Returns:
    None: Saves single comprehensive trend plot as pdf file and trend statistics as CSV

## araras.optuna.callbacks

Callback to stop an Optuna study when improvement stagnates.

Classes:
    - ImprovementStagnationCallback: Monitors improvement variance and stops the
      study once it falls below a threshold.
    - StopIfKeepBeingPruned: Stops the study if a certain number of consecutive trials are pruned.

Example:
    >>> from araras.optuna.callbacks.improvement_stagnation import ImprovementStagnationCallback
    >>> ImprovementStagnationCallback()  # used as a callback in study.optimize

### class ImprovementStagnation

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

#### variance_threshold

```python
def variance_threshold() -> float
```

Variance threshold triggering study stop.

#### variance_threshold

```python
def variance_threshold(value: float) -> None
```

### class StopIfKeepBeingPruned

A callback for Optuna studies that stops the optimization process
when a specified number of consecutive trials are pruned.

Args:
    threshold (int): The number of consecutive pruned trials required to stop the study.

## araras.optuna.keras.__init__

## araras.optuna.keras.stats

Module model_stats of keras

Functions:
    - get_model_stats: Extract and return model statistics from the given Optuna trial.

Example:
    >>> from araras.optuna.keras.model_stats import get_model_stats
    >>> get_model_stats(...)

### get_model_stats

```python
def get_model_stats(trial: optuna.Trial, model: tf.keras.Model, bits_per_param: int, batch_size: int, n_trials: int, device: int, verbose: bool) -> Dict[str, float]
```

Extract and return model statistics from the given Optuna trial.

Args:
    trial (optuna.Trial): The Optuna trial object
    model (tf.keras.Model): The Keras model to analyze.
    policy (tf.keras.DTypePolicy): The precision policy used for the model.
    batch_size (int): The batch size to simulate for input.
    n_trials (int): Number of trials for power and energy measurement.
    device (int): GPU index to run the model on. Use ``-1`` for CPU.
    verbose (bool): If True, print detailed information.

Returns:
    Dict[str, float]: A dictionary containing model statistics

## araras.optuna.utils

This module contains utility functions for Optuna integration.

Functions:
    - supress_optuna_warnings: Suppress only Optuna experimental warnings.
    - get_remaining_trials: Returns a list of completed trials from the given Optuna study.
    - cleanup_non_top_trials: Remove files for trials not in the top-K set.
    - rename_top_k_files: Rename top-K trial files with ranking prefix.
    - save_trial_params_to_file: Save Optuna trial parameters and associated metadata to a text file.
    - get_top_trials: Get the top-K trials from an Optuna study based on ranking criteria.
    - save_top_k_trials: Save top-K trials to text files.
    - init_study_dirs: Create and return study directory structure for experiments.

Example:
    >>> from araras.optuna.utils import supress_optuna_warnings
    >>> supress_optuna_warnings(...)

### cleanup_non_top_trials

```python
def cleanup_non_top_trials(all_trial_ids: Set[int], top_trial_ids: Set[int], cleanup_paths: List[Tuple[str, str]]) -> None
```

Remove files or directories for trials not in the top-K set.

Args:
    all_trial_ids (Set[int]): Set of all trial IDs in the study.
    top_trial_ids (Set[int]): Set of top-K trial IDs to preserve.
    cleanup_paths (List[Tuple[str, str]]): List of (base_directory, filename_template)
        tuples. The filename_template should contain '{trial_id}' placeholder.

Raises:
    OSError: If file removal operations fail.

### get_remaining_trials

```python
def get_remaining_trials(study: optuna.Study, num_trials: int) -> list[optuna.trial.FrozenTrial]
```

Returns a list of completed trials from the given Optuna study.

Args:
    study (optuna.Study): The Optuna study to retrieve trials from.
    num_trials (int): The total number of trials to consider.

Returns:
    list[optuna.trial.FrozenTrial]: A list of completed trials.

### get_top_trials

```python
def get_top_trials(study: optuna.Study, top_k: int, rank_key: str, order: str) -> List[optuna.Trial]
```

Get the top-K trials from an Optuna study based on ranking criteria.

Args:
    study (optuna.Study): The completed Optuna study.
    top_k (int): Number of top trials to retrieve.
    rank_key (str): Key to rank trials by ("value" for objective value,
                   or any user attribute key).
    order (str): "descending" for highest values first or "ascending" for
        lowest values first.

Returns:
    List[optuna.Trial]: List of top-K trials sorted by the ranking criteria.

### init_study_dirs

```python
def init_study_dirs(run_dir: Any, study_name: Any, subdirs: Any) -> None
```

Create and return study directory structure for experiments.

Args:
    run_dir (str): Base directory for the run
    study_name (str): Name of the study directory (default: "optuna_study")
    subdirs (list): List of subdirectory names to create
                   (default: ["args", "figures", "backup", "history", "models", "logs", "tensorboard"])

Returns:
    tuple: (study_dir, *subdirectory_paths) in the order specified by subdirs

### rename_top_k_files

```python
def rename_top_k_files(top_trials: List[optuna.Trial], file_configs: List[Tuple[str, str]]) -> None
```

Rename top-K trial files with ranking prefix.

Args:
    top_trials (List[optuna.Trial]): List of top trials in ranked order.
    file_configs (List[Tuple[str, str]]): List of (base_directory, file_extension)
        tuples. Files are expected to follow pattern 'trial_{trial_id}{extension}'.

Raises:
    OSError: If file rename operations fail.

### save_top_k_trials

```python
def save_top_k_trials(top_trials: List[optuna.Trial], args_dir: str, study: optuna.Study, extra_attrs: Optional[List[str]]) -> None
```

Save top-K trials to text files.

Args:
    top_trials (List[optuna.Trial]): List of trials to save.
    args_dir (str): Directory to save trial parameter files.
    study (optuna.Study): The Optuna study (needed for sampler info).
    extra_attrs (Optional[List[str]]): List of additional user attributes to save.
                                      If None, defaults to common accuracy metrics.

### save_trial_params_to_file

```python
def save_trial_params_to_file(filepath: str, params: dict[str, float], **kwargs: str) -> None
```

Save Optuna trial parameters and associated metadata to a text file.

Args:
    filepath (str): Path where the parameter file should be saved.
    params (dict[str, float]): Dictionary of trial hyperparameters.
    **kwargs (str): Additional information such as trial ID, rank, or loss.

Returns:
    None

### supress_optuna_warnings

```python
def supress_optuna_warnings() -> None
```

Suppress only Optuna experimental warnings.

## araras.plot.__init__

## araras.plot.configs

This module contains functions to configure matplotlib rcParams for IEEE-style

Functions:
    - config_plt: Configure matplotlib rcParams for IEEE‑style figures

Example:
    >>> from araras.plot.configs import config_plt
    >>> config_plt(...)

### config_plt

```python
def config_plt(style: str) -> None
```

Configure matplotlib rcParams for IEEE‑style figures

Args:
    style (str): The figure style to use. Options are 'single-column' or
        'double-column'. Default is 'single-column'.

Returns:
    None

## araras.tensorflow.__init__

## araras.tensorflow.model

Module for estimating average power and energy consumption

Functions:
    - get_model_usage_stats: Estimate average power draw and energy usage.

Example:
    >>> from araras.tensorflow.utils.model import get_model_usage_stats
    >>> get_model_usage_stats(...)

### get_model_usage_stats

```python
def get_model_usage_stats(saved_model: str | tf.keras.Model, n_trials: int, device: int, rapl_path: str, verbose: bool) -> Tuple[float, float, float]
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

Args:
    saved_model (str | tf.keras.Model): Path to the TensorFlow SavedModel directory,
        a .keras model file, or a Keras Model instance.
    n_trials (int): Number of inference trials to perform. Defaults to 100000.
    device (int): GPU index for power measurement, or ``-1`` to use the CPU.
    rapl_path (str): Path to the RAPL energy counter file for CPU measurements.
    verbose (bool): If True, displays a progress bar during the trials.

Raises:
    RuntimeError: If GPU NVML initialization fails when ``device`` refers to a GPU index.
    ValueError: If ``device`` is neither ``-1`` nor a valid GPU index.

Returns:
    Tuple[float, float, float]:
        - per_run_time (float):
            Average run time in seconds. Measures a mix of tracing, initialization,
            asynchronous queuing, Python overhead, and power-reading delays,
            so its “average” can be dominated by non-inference costs.
        - avg_power (float): Average power draw in watts. If a negative value is
          measured repeatedly, the function returns 0 after two retries.
        - avg_energy (float): Average energy consumed per inference in joules. This
          will also be ``0`` if ``avg_power`` could not be measured correctly.

## araras.utils.__init__

## araras.utils.cleanup

Utility for cleaning up child processes efficiently.

Classes:
    - ChildProcessCleanup: Terminates or kills child processes with grace.

Example:
    >>> from araras.utils.cleanup import ChildProcessCleanup
    >>> ChildProcessCleanup().cleanup_children()

### class ChildProcessCleanup

Efficient child process cleanup with graceful termination and force kill fallback.

#### cleanup_children

```python
def cleanup_children(exclude_pids: Optional[List[int]]) -> Tuple[int, int]
```

Clean up all child processes with optimized batch operations.

Args:
    exclude_pids: Additional PIDs to exclude from cleanup

Returns:
    Tuple of (terminated_count, killed_count)

Raises:
    psutil.NoSuchProcess: If current process doesn't exist

#### add_protected_pid

```python
def add_protected_pid(pid: int) -> None
```

Add a PID to the protected (exclude) list.

Args:
    pid: Process ID to protect from cleanup

#### remove_protected_pid

```python
def remove_protected_pid(pid: int) -> None
```

Remove a PID from the protected list.

Args:
    pid: Process ID to remove from protection

#### get_child_count

```python
def get_child_count() -> int
```

Get current number of child processes.

Returns:
    Number of child processes (including nested children)

## araras.utils.dir

This module provides utility functions for managing directories, such as creating

Functions:
    - create_run_directory: Creates a new run directory with an incremented numeric suffix and returns its full path.

Example:
    >>> from araras.utils.dir import create_run_directory
    >>> create_run_directory(...)

### create_run_directory

```python
def create_run_directory(prefix: str, base_dir: str) -> str
```

Creates a new run directory with an incremented numeric suffix and returns its full path.

The directory name is generated using the given prefix followed by the next available number.
For example, if directories "run1", "run2", and "run3" exist, calling with prefix="run" will create "run4".

Logic:
    -> Ensure base_dir exists
    -> List existing directories with matching prefix and numeric suffix
    -> Parse suffix numbers and find the next available integer
    -> Construct full path using prefix + next number
    -> Create the new run directory and return its path

Args:
    prefix (str): Prefix to be used in the name of each run directory (e.g., "run").
    base_dir (str, optional): Directory under which all runs are stored. Defaults to "runs".

Returns:
    str: Absolute path to the newly created run directory.

Example:
    run_path = create_run_directory(prefix="run")
    print(run_path)  # outputs: runs/run1, runs/run2, etc.

## araras.utils.gpu

This module provides utility functions for inspecting and reporting GPU-related

Functions:
    - get_user_gpu_choice: Prompts the user to select a GPU index and validates the input.
    - _get_nvidia_smi_data: Retrieves GPU information using nvidia-smi command.
    - _print_tensorflow_info: Print TensorFlow configuration information.
    - _print_gpu_table: Print GPU information in nvidia-smi style table format.
    - _print_memory_summary: Print memory summary similar to nvidia-smi bottom section.
    - get_gpu_info: Prints detailed TensorFlow and GPU configuration information in nvidia-smi style format.
    - gpu_summary: Prints a compact GPU summary similar to nvidia-smi output.

Example:
    >>> from araras.utils.gpu import get_user_gpu_choice
    >>> get_user_gpu_choice(...)

### get_gpu_info

```python
def get_gpu_info() -> None
```

Prints detailed TensorFlow and GPU configuration information in nvidia-smi style format.

This function reports:
  - TensorFlow version and CUDA configuration
  - GPU devices in tabular format similar to nvidia-smi
  - Memory usage summary
  - Temperature and utilization data (when available)

Args:
    None

Returns:
    None

Example:
    get_gpu_info()

### get_user_gpu_choice

```python
def get_user_gpu_choice() -> None
```

Prompts the user to select a GPU index and validates the input.

Returns:
    str: Valid GPU index as string

### gpu_summary

```python
def gpu_summary() -> None
```

Prints a compact GPU summary similar to nvidia-smi output.

## araras.utils.logs

This module provides utilities for logging system.

Functions:
    - log_resources: Logs selected system and ML resources (CPU, RAM, GPU, CUDA, TensorFlow) at regular time intervals.

Example:
    >>> from araras.utils.logs import log_resources
    >>> log_resources(...)

### log_resources

```python
def log_resources(log_dir: str, interval: int, **kwargs: Any) -> None
```

Logs selected system and ML resources (CPU, RAM, GPU, CUDA, TensorFlow) at regular time intervals.

Args:
    log_dir (str): Directory where log files will be stored.
    interval (int): Time interval between consecutive logs in seconds. Defaults to 5.
    kwargs: Boolean flags to specify which resources to log.
            Supported flags: "cpu", "ram", "gpu", "cuda", "tensorflow".

Returns:
    None

Example:
    log_resources("logs", interval=10, cpu=True, ram=True, gpu=True)

## araras.utils.misc

Miscellaneous utility functions for the Araras project.

Functions:
    - clear: Clear all prints from terminal or notebook cell.
    - format_number: Format a number using scientific suffixes.
    - format_bytes: Format bytes using binary suffixes (B, KB, MB, GB, etc.).
    - format_scientific: Format to scientific notation with automatic precision based on number magnitude.
    - format_number_commas: Format a number with commas as thousands separators.

Example:
    >>> from araras.utils.misc import clear
    >>> clear(...)

### clear

```python
def clear() -> None
```

Clear all prints from terminal or notebook cell.

This function works in multiple environments:
- Jupyter notebooks/JupyterLab
- Terminal/command prompt (Windows, macOS, Linux)
- Python scripts run from command line

### format_bytes

```python
def format_bytes(bytes_value: Any, precision: Any) -> None
```

Format bytes using binary suffixes (B, KB, MB, GB, etc.).

Args:
    bytes_value (int, float): The number of bytes
    precision (int): Number of decimal places to show (default: 2)

Returns:
    str: Formatted bytes with appropriate suffix

### format_number

```python
def format_number(number: Any, precision: Any) -> None
```

Format a number using scientific suffixes.

Args:
    number (int, float): The number to format
    precision (int): Number of decimal places to show (default: 2)

Returns:
    str: Formatted number with appropriate suffix

### format_number_commas

```python
def format_number_commas(number: Any, precision: Any) -> None
```

Format a number with commas as thousands separators.

Args:
    number (int, float): The number to format
    precision (int): Number of decimal places to show (default: 2)

Returns:
    str: Number formatted with commas

### format_scientific

```python
def format_scientific(number: Any, max_precision: Any) -> None
```

Format to scientific notation with automatic precision based on number magnitude.

Args:
    number (int, float): The number to format
    max_precision (int): Maximum number of decimal places (default: 2)

Returns:
    str: Number formatted in scientific notation

### class NotebookConverter

Notebook to Python conversion.

#### convert_notebook_to_python

```python
def convert_notebook_to_python(notebook_path: Path) -> Path
```

Convert Jupyter notebook to Python file with same name.

Args:
    notebook_path: Path to .ipynb file

Returns:
    Path to generated .py file

Raises:
    ImportError: If notebook dependencies missing
    ValueError: If notebook conversion fails

## araras.utils.terminal

Cross-platform terminal launcher utilities.

Classes:
    - SimpleTerminalLauncher: Launches commands in new terminals and captures PIDs.

Example:
    >>> from araras.utils.terminal import SimpleTerminalLauncher
    >>> SimpleTerminalLauncher().launch(["echo", "hello"], ".")

### class SimpleTerminalLauncher

Minimal terminal launcher for cross-platform execution.

#### set_supress_tf_warnings

```python
def set_supress_tf_warnings(value: bool) -> None
```

Set the supress_tf_warnings attribute.

Args:
    value: Boolean indicating whether to suppress TensorFlow warnings.

#### launch

```python
def launch(command: List[str], working_dir: str) -> subprocess.Popen
```

Launch command in new terminal with PID capture.

Args:
    command: Command array to execute
    working_dir: Working directory

Returns:
    Terminal process object with pid_file attribute

Raises:
    OSError: If unsupported OS or launch fails

