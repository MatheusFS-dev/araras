# API Documentation

This document provides an overview of the API functions available in the ARARAS package.

## Table of Contents

- [API Documentation](#api-documentation)
  - [Table of Contents](#table-of-contents)
  - [core](#core)
    - [supress\_optuna\_warnings](#supress_optuna_warnings)
  - [ml.model.builders.cnn](#mlmodelbuilderscnn)
    - [build\_cnn1d](#build_cnn1d)
    - [build\_dense\_as\_conv1d](#build_dense_as_conv1d)
    - [generate\_conv1d\_pool\_table](#generate_conv1d_pool_table)
    - [build\_cnn2d](#build_cnn2d)
    - [build\_dense\_as\_conv2d](#build_dense_as_conv2d)
    - [build\_cnn3d](#build_cnn3d)
    - [build\_dense\_as\_conv3d](#build_dense_as_conv3d)
  - [ml.model.builders.dnn](#mlmodelbuildersdnn)
    - [build\_dnn](#build_dnn)
  - [ml.model.builders.gnn](#mlmodelbuildersgnn)
    - [build\_grid\_adjacency](#build_grid_adjacency)
    - [build\_knn\_adjacency](#build_knn_adjacency)
    - [check\_gpu\_limit](#check_gpu_limit)
    - [build\_gcn](#build_gcn)
    - [build\_gat](#build_gat)
    - [build\_cheb](#build_cheb)
  - [ml.model.builders.lstm](#mlmodelbuilderslstm)
    - [build\_lstm](#build_lstm)
  - [ml.model.builders.se](#mlmodelbuildersse)
    - [build\_squeeze\_excite\_1d](#build_squeeze_excite_1d)
  - [ml.model.builders.skip](#mlmodelbuildersskip)
    - [trial\_skip\_connections](#trial_skip_connections)
  - [ml.model.callbacks](#mlmodelcallbacks)
    - [get\_callbacks\_model](#get_callbacks_model)
  - [ml.model.stats](#mlmodelstats)
    - [get\_flops](#get_flops)
    - [get\_macs](#get_macs)
    - [get\_memory\_and\_time](#get_memory_and_time)
    - [get\_model\_usage\_stats](#get_model_usage_stats)
    - [write\_model\_stats\_to\_file](#write_model_stats_to_file)
  - [ml.model.tools](#mlmodeltools)
    - [convert\_to\_saved\_model](#convert_to_saved_model)
    - [punish\_model\_flops](#punish_model_flops)
    - [punish\_model\_params](#punish_model_params)
    - [punish\_model](#punish_model)
  - [ml.model.utils](#mlmodelutils)
    - [capture\_model\_summary](#capture_model_summary)
  - [ml.optuna.analyzer](#mloptunaanalyzer)
    - [analyze\_study](#analyze_study)
    - [PlotConfig](#plotconfig)
    - [set\_plot\_config\_param](#set_plot_config_param)
    - [set\_plot\_config\_params](#set_plot_config_params)
  - [ml.optuna.callbacks](#mloptunacallbacks)
    - [get\_callbacks\_study](#get_callbacks_study)
    - [ImprovementStagnation](#improvementstagnation)
    - [StopIfKeepBeingPruned](#stopifkeepbeingpruned)
    - [StopWhenNoValueImprovement](#stopwhennovalueimprovement)
    - [NanLossPrunerOptuna](#nanlosspruneroptuna)
  - [ml.optuna.model\_tools](#mloptunamodel_tools)
    - [estimate\_training\_memory](#estimate_training_memory)
    - [plot\_model\_param\_distribution](#plot_model_param_distribution)
    - [set\_user\_attr\_model\_stats](#set_user_attr_model_stats)
  - [ml.optuna.utils](#mloptunautils)
    - [cleanup\_non\_top\_trials](#cleanup_non_top_trials)
    - [get\_remaining\_trials](#get_remaining_trials)
    - [get\_top\_trials](#get_top_trials)
    - [rename\_top\_k\_files](#rename_top_k_files)
    - [save\_trial\_params\_to\_file](#save_trial_params_to_file)
    - [save\_top\_k\_trials](#save_top_k_trials)
    - [init\_study\_dirs](#init_study_dirs)
    - [log\_trial\_error](#log_trial_error)
  - [notifications.email](#notificationsemail)
    - [send\_email](#send_email)
  - [runtime.monitoring](#runtimemonitoring)
    - [run\_auto\_restart](#run_auto_restart)
    - [monitor CLI](#monitor-cli)
  - [utils](#utils)
  - [utils.io](#utilsio)
    - [create\_run\_directory](#create_run_directory)
    - [get\_caller\_stem](#get_caller_stem)
  - [utils.misc](#utilsmisc)
    - [clear](#clear)
    - [format\_number](#format_number)
    - [format\_bytes](#format_bytes)
    - [format\_scientific](#format_scientific)
    - [format\_number\_commas](#format_number_commas)
  - [utils.system](#utilssystem)
    - [get\_user\_gpu\_choice](#get_user_gpu_choice)
    - [get\_gpu\_info](#get_gpu_info)
    - [gpu\_summary](#gpu_summary)
    - [log\_resources](#log_resources)
  - [visualization.configs](#visualizationconfigs)
    - [config\_plt](#config_plt)


## core

### supress_optuna_warnings

```python
supress_optuna_warnings(

)
```
Suppress Optuna experimental warnings. This helper inspects Optuna for
``ExperimentalWarning`` classes and filters them out.

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`None`

**Raises**
- None



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
Builds a 1D convolutional layer with optional hyperparameter tuning and regularization. This function creates a Conv1D layer whose hyperparameters (filters, kernel size, regularizers, etc.) can be either statically defined or dynamically tuned through an Optuna trial. It optionally applies batch normalization and a user-defined activation function.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | An Optuna trial object used for hyperparameter optimization. |
| kparams | `KParams` | A utility object to provide regularizers and activations. |
| x | `layers.Layer` | The input Keras layer. |
| filters_range | `Union[int, tuple[int, int]]` | Number of filters or a range for tuning. |
| kernel_size_range | `Union[int, tuple[int, int]]` | Kernel size or a range for tuning. |
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
` layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.`

**Raises**
- None

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
    data_format,
    kernel_initializer,
    bias_initializer,
    name_prefix,
)
```
Simulate a Dense layer using a Conv1D with kernel_size=1. This function builds a 1D convolutional layer that, when applied to a 3D input of shape (batch_size, length, features_in), produces an output of shape (batch_size, length, units).
> [NOTE]
>  If your goal is to emulate a classic Dense(units) on a flat vector of shape (batch_size, features_in), you must first reshape that vector to (batch_size, 1, features_in) and then apply this function. After Conv1D, you should call Flatten() to collapse back to (batch_size, units). Without reshaping, Conv1D will raise a shape mismatch on 2D inputs.


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
| data_format | `str` | Data format, either 'channels_last' or 'channels_first'. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
` layers.Layer: A Keras layer with output shape (batch_size, 1, units), equivalent to Dense(units). References: https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks`

**Raises**
- None

### generate_conv1d_pool_table

```python
generate_conv1d_pool_table(
    L0,
    n_layers,
    kernel_sizes,
    pool_sizes,
    filters,
    conv_stride=1,
    conv_dilation=1,
    pool_stride=None,
    csv_path=None,
    verbose=True,
    plot=False,
    plot_dir=None,
)
```
Generate pooling length combinations for stacked ``Conv1D`` blocks.

Each block applies ``Conv1D`` followed by ``MaxPooling1D`` using ``same`` padding.
The Cartesian product of ``kernel_sizes``, ``pool_sizes`` and ``filters`` is
enumerated ``n_layers`` times and the resulting temporal length after each
pooling operation is computed. Optionally, the table is streamed to ``csv_path``
and histograms of final lengths are saved.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| L0 | `int` | Initial temporal length before the first block. |
| n_layers | `int` | Number of ``Conv1D`` + ``MaxPooling1D`` blocks. |
| kernel_sizes | `Sequence[int]` | Allowed kernel sizes for ``Conv1D``. |
| pool_sizes | `Sequence[int]` | Allowed pool sizes for ``MaxPooling1D``. |
| filters | `Sequence[int]` | Allowed filter counts for ``Conv1D``. |
| conv_stride | `int` | Stride for all ``Conv1D`` layers. |
| conv_dilation | `int` | Dilation rate for all ``Conv1D`` layers. |
| pool_stride | `Optional[int]` | Stride for ``MaxPooling1D`` layers. Uses ``pool_size`` when ``None``. |
| csv_path | `Optional[str]` | Stream the table to this CSV file if provided. |
| verbose | `bool` | Display a progress bar while generating combinations. |
| plot | `bool` | Whether to generate histograms for each layer's final length. |
| plot_dir | `Optional[str]` | Directory to save histogram images. |

**Returns**
` pandas.DataFrame: Table of parameter combinations and resulting lengths.`

**Raises**
- None

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
Builds a 2D convolutional layer with optional hyperparameter tuning and regularization. This function creates a Conv2D layer whose hyperparameters (filters, kernel size, regularizers, etc.) can be either statically defined or dynamically tuned through an Optuna trial. It optionally applies batch normalization and a user-defined activation function.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | An Optuna trial object used for hyperparameter optimization. |
| kparams | `KParams` | A utility object to provide regularizers and activations. |
| x | `layers.Layer` | The input Keras layer. |
| filters_range | `Union[int, tuple[int, int]]` | Number of filters or a range for tuning. |
| kernel_size_range | `Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]` |  Fixed (height, width) or ranges ((h_min, h_max), (w_min, w_max)) for tuning. |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_step | `int` | Step size for kernel dimension tuning. |
| use_batch_norm | `bool` | Whether to include batch normalization. |
| trial_kernel_reg | `bool` | Whether to tune and apply kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune and apply bias regularization. |
| trial_activity_reg | `bool` | Whether to tune and apply activity regularization. |
| strides | `tuple[int, int]` | Stride size for height and width. |
| dilation_rate | `tuple[int, int]` | Dilation rate for height and width. |
| groups | `int` | Number of filter groups. |
| use_bias | `bool` | Whether to use a bias term in the convolution. If using batch norm, this can be set to False. |
| padding | `str` | Padding method ('valid' or 'same'). |
| data_format | `str` | Data format, either 'channels_last' or 'channels_first'. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
` layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.`

**Raises**
- None

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
    data_format,
    kernel_initializer,
    bias_initializer,
    name_prefix,
)
```
Simulate a Dense layer using a Conv2D with kernel_size=(1, 1). This function builds a 2D convolutional layer that, when applied to a 4D input of shape (batch_size, height, width, features_in), produces an output of shape (batch_size, height, width, units).
> [NOTE]
>  If your goal is to emulate a classic Dense(units) on a flat vector of shape (batch_size, features_in), you must first reshape that vector to (batch_size, 1, 1, features_in) and then apply this function. After Conv2D, you should call Flatten() to collapse back to (batch_size, units). Without reshaping, Conv2D will raise a shape mismatch on 3D inputs.


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
| data_format | `str` | Data format, either 'channels_last' or 'channels_first'. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
` layers.Layer: A Keras layer with output shape (batch_size, height, width, units), equivalent to Dense(units). References: https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks`

**Raises**
- None

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
Builds a 3D convolutional layer with optional hyperparameter tuning and regularization. This function creates a Conv3D layer whose hyperparameters (filters, kernel size, regularizers, etc.) can be either statically defined or dynamically tuned through an Optuna trial. It optionally applies batch normalization and a user-defined activation function.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | An Optuna trial object used for hyperparameter optimization. |
| kparams | `KParams` | A utility object to provide regularizers and activations. |
| x | `layers.Layer` | The input Keras layer. |
| filters_range | `Union[int, Tuple[int, int]]` | Number of filters or a range for tuning. kernel_size_range (Union[ Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] ]): Fixed (depth, height, width) or ranges ((d_min, d_max), (h_min, h_max), (w_min, w_max)) for tuning. |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_step | `int` | Step size for kernel dimension tuning. |
| use_batch_norm | `bool` | Whether to include batch normalization. |
| trial_kernel_reg | `bool` | Whether to tune and apply kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune and apply bias regularization. |
| trial_activity_reg | `bool` | Whether to tune and apply activity regularization. |
| strides | `Tuple[int, int, int]` | Stride size for depth, height, and width. |
| dilation_rate | `Tuple[int, int, int]` | Dilation rate for depth, height, and width. |
| groups | `int` | Number of filter groups. |
| use_bias | `bool` | Whether to use a bias term in the convolution. If using batch norm, this can be set to False. |
| padding | `str` | Padding method ('valid' or 'same'). |
| data_format | `str` | Data format, either 'channels_last' or 'channels_first'. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
` layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.`

**Raises**
- None

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
    data_format,
    kernel_initializer,
    bias_initializer,
    name_prefix,
)
```
Simulate a Dense layer using a Conv3D with kernel_size=(1, 1, 1). This function builds a 3D convolutional layer that, when applied to a 5D input of shape (batch_size, depth, height, width, features_in), produces an output of shape (batch_size, depth, height, width, units).
> [NOTE]
>  If your goal is to emulate a classic Dense(units) on a flat vector of shape (batch_size, features_in), you must first reshape that vector to (batch_size, 1, 1, 1, features_in) and then apply this function. After Conv3D, you should call Flatten() to collapse back to (batch_size, units). Without reshaping, Conv3D will raise a shape mismatch on 4D inputs.


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
| data_format | `str` | Data format, either 'channels_last' or 'channels_first'. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
` layers.Layer: A Keras layer with output shape (batch_size, depth, height, width, units), equivalent to Dense(units). References: https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks`

**Raises**
- None

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
Builds a single dense neural network (DNN) block with optional regularization, batch normalization, and dropout. This function constructs a configurable DNN layer consisting of a Dense layer followed by optional batch normalization, a user-specified activation function, and dropout. It supports hyperparameter tuning via the `trial` object.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Hyperparameter tuning trial object, e.g., from Optuna. |
| kparams | `KParams` | Custom hyperparameter handler that provides regularizers and activations. |
| x | `layers.Layer` | Input tensor or layer to build on. |
| units_range | `Union[int, tuple[int, int]]` | Either a fixed unit count or a range for tuning. |
| units_step | `int` | Step size for unit range tuning. |
| dropout_rate_range | `Union[float, tuple[float, float]]` | Either a fixed dropout rate or a range. |
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
` layers.Layer: Output tensor after applying the DNN block.`

**Raises**
- None

## ml.model.builders.gnn

### build_grid_adjacency

```python
build_grid_adjacency(
    rows,
    cols,
)
```
Build a grid adjacency matrix with GCN normalization. Each node is connected to its four direct neighbours (up, down, left and right).  The resulting adjacency matrix is returned as a TensorFlow sparse tensor ready to be fed to Spektral layers.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| rows | `int` | Number of grid rows. |
| cols | `int` | Number of grid columns. |

**Returns**
` tf.sparse.SparseTensor: Normalized sparse adjacency matrix.`

**Raises**
- None

### build_knn_adjacency

```python
build_knn_adjacency(
    rows,
    cols,
    k,
)
```
Construct a k-nearest neighbour adjacency matrix on a 2-D grid. Nodes correspond to cells of a `rows` × `cols` grid.  Each node is connected to its `k` spatially nearest neighbours.  The adjacency matrix is symmetrised, normalised with the GCN filter and returned as a TensorFlow sparse tensor.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| rows | `int` | Number of grid rows. |
| cols | `int` | Number of grid columns. |
| k | `int` | Number of neighbours for each node. |

**Returns**
` tf.sparse.SparseTensor: Normalized sparse adjacency matrix.`

**Raises**
- None

### check_gpu_limit

```python
check_gpu_limit(
    knn_list,
    K_list,
    units_list,
    n=20 * 200,
    save_path=None,
)
```
Evaluate TensorFlow's GPU sparse--dense multiplication limit.

For each combination of ``knn_k`` (neighbour count), Chebyshev order ``K`` and
output ``units`` this utility estimates whether ``output_channels * nnz(support)``
exceeds ``2^31 - 1``. Results are shown in a :class:`pandas.DataFrame` and may be
written to ``save_path``.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| knn_list | `list[int]` | List of ``k`` values for the kNN graph. |
| K_list | `list[int]` | Chebyshev orders to test. |
| units_list | `list[int]` | Candidate output channel counts. |
| n | `int` | Number of nodes in the graph. |
| save_path | `str, optional` | Path to save the resulting table as CSV. |

**Returns**
` pandas.DataFrame: Table summarising safe and failing unit counts.`

**Raises**
- OSError: If writing the CSV file fails.

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
| x | `Any` |  |
| a_graph | `Any` |  |
| units_range | `Any` |  |
| dropout_rate_range | `Any` |  |
| units_step | `int` |  |
| dropout_rate_step | `float` |  |
| kernel_initializer | `Any` |  |
| bias_initializer | `Any` |  |
| use_bias | `bool` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`Any`

**Raises**
- None

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
| x | `Any` |  |
| a_graph | `Any` |  |
| units_range | `Any` |  |
| dropout_rate_range | `Any` |  |
| heads_range | `Any` |  |
| units_step | `int` |  |
| dropout_rate_step | `float` |  |
| heads_step | `int` |  |
| concat_heads | `bool` |  |
| kernel_initializer | `Any` |  |
| bias_initializer | `Any` |  |
| use_bias | `bool` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`Any`

**Raises**
- None

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
| x | `Any` |  |
| a_graph | `Any` |  |
| units_range | `Any` |  |
| dropout_rate_range | `Any` |  |
| K_range | `Any` |  |
| units_step | `int` |  |
| dropout_rate_step | `float` |  |
| K_step | `int` |  |
| kernel_initializer | `Any` |  |
| bias_initializer | `Any` |  |
| use_bias | `bool` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| name_prefix | `str` |  |

**Returns**
`Any`

**Raises**
- None

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
Builds a single LSTM block with optional regularization, batch normalization, and dynamic dropout. This function creates a tunable LSTM layer with optional regularization and batch normalization, followed by a customizable activation layer. It supports hyperparameter optimization through a tuning trial.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Object used for suggesting hyperparameters, typically from a tuner like Optuna. |
| kparams | `KParams` | Hyperparameter manager used to retrieve regularizers and activations. |
| x | `layers.Layer` | Input tensor or layer. |
| return_sequences | `bool` | Whether to return the full sequence of outputs or just the last output. |
| units_range | `Union[int, tuple[int, int]]` | Fixed or tunable number of LSTM units. |
| units_step | `int` | Step size for tuning LSTM units if a range is given. |
| dropout_rate_range | `Union[float, tuple[float, float]]` | Fixed or tunable dropout rate. |
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
` layers.Layer: Output tensor after applying the LSTM block.`

**Raises**
- None

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
Apply a Squeeze-and-Excitation (SE) block 1D with Optuna-tuned hyperparameters. Based on the paper: https://arxiv.org/pdf/1709.01507

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| x | `Any` | Input 3D tensor (batch, length, channels). |
| trial | `Any` | Optuna Trial object for suggesting hyperparameters. |
| kparams | `KParams` | KParams object containing hyperparameter choices. |
| ratio_choices | `Any` | List of integers representing reduction ratios for SE block. |
| name_prefix | `str` | Prefix for naming layers and trial parameters. |

**Returns**
` A tensor the same shape as `x`, re-scaled by the SE attention weights.`

**Raises**
- ValueError: If `x.shape[

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
Constructs conditional skip connections between layers based on Optuna trial choices. This function introduces optional skip connections in a neural network architecture, governed by a hyperparameter search using Optuna's `trial.suggest_categorical` method. It allows experimentation with skip connection topology by conditionally merging outputs from earlier layers into later ones. The merging is done via concatenation or addition. **Important**: All tensors that are merged must have identical shapes in all dimensions **except** for the `axis_to_concat` dimension when using `'concat'`. For `'add'`, tensors must be of exactly the same shape.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.trial.Trial` | Optuna trial object used to sample categorical decisions on whether to include each potential skip connection. It is expected to have the method `suggest_categorical(name: str, choices: List[Any]) -> Any`. |
| layers_list | `Sequence[tf.Tensor]` | List of layer output tensors from a Keras model. These are the candidate sources and targets for skip connections. The order in the list reflects the network's topological sequence. |
| axis_to_concat | `int, optional` | Axis along which tensors will be concatenated if `merge_mode` is `'concat'`. Default is -1 (last axis). All tensors to be concatenated must match on all other dimensions. |
| print_combinations | `bool, optional` | If True, prints every possible combination of skip connections as dictionaries mapping skip flags to booleans. Primarily for debugging and audit purposes. Defaults to False. |
| strategy | `str, optional` | Strategy for selecting candidate skip connections. - `'final'`: Allows skips only to the final layer. - `'any'`: Allows skips from any earlier layer `i` to any later layer `j`. Defaults to `'final'`. |
| merge_mode | `str, optional` | Defines how selected tensors are merged: - `'concat'`: Tensors are concatenated along `axis_to_concat`. - `'add'`: Tensors are added element-wise (must be same shape). Defaults to `'concat'`. |

**Returns**
` tf.Tensor: The output tensor resulting from applying the selected skip connections and merging strategy to the input layer sequence.`

**Raises**
- ValueError: If `strategy` is not one of `'final'` or `'any'`.
- ValueError: If `merge_mode` is not one of `'concat'` or `'add'`.

## ml.model.callbacks

### get_callbacks_model

```python
get_callbacks_model(
    backup_dir,
    tensorboard_logs,
    early_stopping_patience,
    reduce_lr_patience,
)
```
Constructs and returns a list of Keras callbacks for model training.

> [!CAUTION]
> The `write_graph` option in the TensorBoard callback is disabled because
> enabling it drastically increases memory usage.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| backup_dir | `str` | Directory where the backup files will be stored. |
| tensorboard_logs | `str` | Directory where TensorBoard logs will be stored. |
| early_stopping_patience | `int, optional` | Epochs to wait before stopping training. Set to `None` to disable `EarlyStopping`. |
| reduce_lr_patience | `int, optional` | Epochs to wait before reducing the learning rate. Set to `None` to disable `ReduceLROnPlateau`. |

**Returns**
` List[tf.keras.callbacks.Callback]: A list of callbacks to pass into `model.fit()`.`

**Raises**
- None

## ml.model.stats

### get_flops

```python
get_flops(
    model,
    batch_size,
)
```
Calculates the total number of floating-point operations (FLOPs) needed to perform a single forward pass of the given Keras model. Flow: model -> input_shape -> TensorSpec -> tf.function -> concrete function -> graph -> profile(graph) -> total_float_ops -> return

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` | The Keras model to analyze. |
| batch_size | `int, optional` | The batch size to simulate for input. Defaults to 1. |

**Returns**
` int: The total number of floating-point operations (FLOPs) for one forward pass.`

**Raises**
- None

### get_macs

```python
get_macs(
    model,
    batch_size,
)
```
Estimates the number of Multiply-Accumulate operations (MACs) required for a single forward pass of the model. Assumes 1 MAC = 2 FLOPs. Flow: model -> input_shape -> TensorSpec -> tf.function -> concrete function -> graph -> profile(graph) -> total_float_ops // 2 -> return

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` | The Keras model to analyze. |
| batch_size | `int, optional` | The batch size to simulate for input. Defaults to 1. |

**Returns**
` int: The estimated number of MACs for one forward pass.`

**Raises**
- None

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
Measures the peak memory usage and average inference time of a Keras model on GPU or CPU. Observations: Warmup runs exclude one-time initialization costs from your measurements. On GPU the very first inference will trigger things like driver wake-up, context setup, PTX→BIN compilation and power-state switching, and cache fills. By running a few warmup inferences you force all of that work to happen before timing, so your measured latencies reflect true steady-state performance rather than setup overhead. Under @tf.function the first call also traces and builds the execution graph, applies optimizations and allocates buffers. Those activities inflate both time and memory on the “cold” run. Warmup runs let TensorFlow complete tracing and graph compilation once, so your timed loop measures only the optimized graph execution path. The CPU memory probe occasionally reports zero usage. When this happens, the measurement is retried up to two additional times. If all attempts still report zero memory, the function returns ``0`` for the peak usage and emits a warning in red.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` | The Keras model to analyze. |
| batch_size | `int` | The batch size to simulate for input. Defaults to 1. Measure with batch_size=1 to get base per-sample latency. |
| device | `int` | GPU index to run the model on. Use ``-1`` to run on CPU. |
| warmup_runs | `int` | Number of warm-up runs before timing. Defaults to 10. |
| test_runs | `int` | Number of runs to measure average inference time. Defaults to 50. |
| verbose | `bool` | If True, displays a progress bar during test runs. |

**Returns**
` Tuple[int, float]: - peak memory usage in bytes (0 if CPU measurement fails after several attempts) - average inference time in seconds`

**Raises**
- None

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
Estimate average power draw and energy usage. Careful with the RAPL path; it may vary by system. The RAPL interface is typically found at: $ ls /sys/class/powercap intel-rapl $ ls /sys/class/powercap/intel-rapl intel-rapl:0       intel-rapl:0:0    intel-rapl:1    … $ ls /sys/class/powercap/intel-rapl/intel-rapl:0 energy_uj  max_energy_range_uj  name Also, you MUST run this on a linux system with Intel CPUs!!!!! And run the python script with SUDO to access RAPL files.
> [!IMPORTANT]
> This function must be executed on a Linux system with Intel CPUs and requires sudo privileges to read RAPL counters.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| saved_model | `str | tf.keras.Model` | Path to the TensorFlow SavedModel directory, a .keras model file, or a Keras Model instance. |
| n_trials | `int` | Number of inference trials to perform. Defaults to 100000. |
| device | `int` | GPU index for power measurement, or ``-1`` to use the CPU. |
| rapl_path | `str` | Path to the RAPL energy counter file for CPU measurements. |
| verbose | `bool` | If True, displays a progress bar during the trials. |

**Returns**
` Tuple[float, float, float]: - per_run_time (float): Average run time in seconds. Measures a mix of tracing, initialization, asynchronous queuing, Python overhead, and power-reading delays, so its “average” can be dominated by non-inference costs. - avg_power (float): Average power draw in watts. If a negative value is measured repeatedly, the function returns 0 after two retries. - avg_energy (float): Average energy consumed per inference in joules. This will also be ``0`` if ``avg_power`` could not be measured correctly.`

**Raises**
- RuntimeError: If GPU NVML initialization fails when ``device`` refers to a GPU index.
- ValueError: If ``device`` is neither ``-1`` nor a valid GPU index.

### write_model_stats_to_file

```python
write_model_stats_to_file(
    model,
    file_path,
    bytes_per_param,
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
| model | `tf.keras.Model` | The Keras model to analyze. |
| file_path | `str` | The path to the output file. |
| bytes_per_param | `int` | Number of bytes per parameter for model size calculation. |
| batch_size | `int` | The batch size to simulate for input. |
| device | `int` | GPU index to run the model on. Use ``-1`` for CPU. |
| n_trials | `int` | Number of trials for power and energy measurement. |
| extra_attrs | `Optional[Dict[str, Any]]` | Mapping of attribute names to values written after the main statistics. |
| verbose | `bool` | If True, print detailed information. |

> [!NOTE]
> Extra attributes can be used to record custom metrics such as accuracy or F1 score alongside the default statistics.

**Returns**
`Any`

**Raises**
- None

## ml.model.tools

### convert_to_saved_model

```python
convert_to_saved_model(
    input_keras_path,
    output_zip_path,
)
```
Convert a Keras `.keras` model archive into a zipped TensorFlow SavedModel. This will load the model, export it in SavedModel directory format, then compress that directory into a .zip file.
> [!TIP]
> The resulting zip file can be unzipped and loaded with `tf.saved_model.load` just like a standard SavedModel directory.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| input_keras_path | `str` | Path to the source `.keras` model file. |
| output_zip_path | `str` | Desired path for the output zip (e.g. 'saved_model.zip'). |

**Returns**
` None`

**Raises**
- Any exception raised by TensorFlow I/O (e.g. file not found, load/save errors).

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
| target | `Any` | Base objective value (scalar or list of scalars). |
| model | `Any` | Model whose FLOPs will be used for the penalty. |
| penalty_factor | `float` | Multiplicative factor applied to the FLOPs count. |
| direction | `Any` | Whether the objective should be minimised or maximised. |

**Returns**
` The penalised objective value or list of values.`

**Raises**
- None

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
| target | `Any` | Base objective value (scalar or list of scalars). |
| model | `Any` | Model whose parameters will be used for the penalty. |
| penalty_factor | `float` | Multiplicative factor applied to the parameter count. |
| direction | `Any` | Whether the objective should be minimised or maximised. |

**Returns**
` The penalised objective value or list of values.`

**Raises**
- None

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
| target | `Any` | Base objective value (scalar or list of scalars). |
| model | `Any` | Model whose complexity will be penalised. |
| type | `Any` | Type of penalty to apply, either "flops" or "params". |
| flops_penalty_factor | `float` | Factor for FLOPs penalty. |
| params_penalty_factor | `float` | Factor for parameters penalty. |
| direction | `Any` | Whether the objective should be minimised or maximised. |

**Returns**
` The penalised objective value or list of values.`

**Raises**
- None

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
| model | `Any` | Keras model |

**Returns**
` str: Model summary as string`

**Raises**
- None

## ml.optuna.analyzer

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
| study | `Any` | Optuna study object containing trials to analyze. |
| table_dir | `str` | Directory to save analysis results and figures. |
| top_frac | `float` | Fraction of best/worst trials to analyze (default: 0.2). |
| param_name_mapping | `Any` | Optional mapping of parameter names to display names. |
| Example | `None` | {'params_learning_rate': 'Learning Rate'} |
| create_standalone | `bool` | If True, generates standalone images for each plot type. |
| save_data | `bool` | If True, saves data for LaTeX plotting into CSV files. |
| create_plotly | `bool` | If True, also saves interactive Plotly HTML versions of the figures. |
| plots | `Any` | List of plot types to generate. Valid options: 'distributions', 'importances', 'correlations', 'boxplots', 'trends', 'ranges', 'contours', 'edf', 'intermediate', 'parallel_coordinate', 'slice', 'rank', 'history', 'timeline', 'terminator'. If ``None`` the default set is used. |

**Returns**
`Any`

**Raises**
- None

### PlotConfig

```python
PlotConfig
```
Global configuration for matplotlib plots used in this module.

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`None`

**Raises**
- None

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
`Any`

**Raises**
- None

### set_plot_config_params

```python
set_plot_config_params(

)
```
Set multiple parameters in :data:`PLOT_CFG`.

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`Any`

**Raises**
- None

## ml.optuna.callbacks

### get_callbacks_study

```python
get_callbacks_study(
    trial,
    tensorboard_logs,
    monitor,
    early_stopping_patience,
    reduce_lr_patience,
    pruning_interval,
)
```
Constructs and returns a list of Keras callbacks tailored for Optuna trials.

> [!CAUTION]
> The `write_graph` option in the TensorBoard callback is disabled because
> enabling it drastically increases memory usage.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | The current Optuna trial object. |
| tensorboard_logs | `str` | Directory where TensorBoard logs will be stored. |
| monitor | `str` | The metric to monitor for early stopping and learning rate reduction. |
| early_stopping_patience | `int, optional` | Epochs to wait before stopping training. Set to `None` to disable `EarlyStopping`. |
| reduce_lr_patience | `int, optional` | Epochs to wait before reducing the learning rate. Set to `None` to disable `ReduceLROnPlateau`. |
| pruning_interval | `int, optional` | Frequency in epochs to check metric improvement for pruning. Set to `None` to disable pruning. |

**Returns**
` List[tf.keras.callbacks.Callback]: A list of callbacks to pass into `model.fit()`.`

**Raises**
- None

### ImprovementStagnation

```python
ImprovementStagnation
```
Stop a study when the terminator improvement variance plateaus. After each completed trial the callback computes the potential future improvement using an ``optuna.terminator`` improvement evaluator. The variance of the most recent ``window_size`` improvement values is measured and if it falls below ``variance_threshold`` after ``min_n_trials`` trials the study is terminated via :meth:`optuna.Study.stop`. Parameters ---------- min_n_trials: Minimum number of completed trials before starting variance checks. window_size: Number of recent improvement values used to compute the variance. variance_threshold: Threshold below which the variance of improvements indicates stagnation. improvement_evaluator: Custom improvement evaluator. Defaults to :class:`RegretBoundEvaluator`.

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`None`

**Raises**
- None

### StopIfKeepBeingPruned

```python
StopIfKeepBeingPruned
```
A callback for Optuna studies that stops the optimization process when a specified number of consecutive trials are pruned.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| threshold | `int` | The number of consecutive pruned trials required to stop the study. |

**Returns**
`None`

**Raises**
- None

### StopWhenNoValueImprovement

```python
StopWhenNoValueImprovement
```
Stop a study if the best objective value fails to improve for ``patience`` consecutive trials.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| patience | `int` | Number of completed trials allowed without improvement. |
| min_delta | `float` | Minimum change in value to qualify as improvement. |
| verbose | `bool` | Log a warning when stopping the study. |

**Returns**
`None`

**Raises**
- ValueError

### NanLossPrunerOptuna

```python
NanLossPrunerOptuna
```
A custom Keras callback that prunes an Optuna trial if NaN is encountered in training loss. This is useful for skipping unpromising model configurations early, especially those that are unstable or diverging during training.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | The Optuna trial associated with this model run. |
| Example | `None` |  model.fit(..., callbacks=[NanLossPrunerOptuna(trial)]) |

**Returns**
`None`

**Raises**
- None

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
| model | `Any` | Keras model object |
| batch_size | `int` | Training batch size |

**Returns**
` Total memory needed in bytes`

**Raises**
- None

### plot_model_param_distribution

```python
plot_model_param_distribution(
    build_model_fn,
    bytes_per_param,
    batch_size,
    n_trials,
    fig_save_path=None,
    figsize=(18, 6),
    csv_path=None,
    logs_dir=None,
    corr_csv_path=None,
)
```
Sample random models and plot parameter and size histograms.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| build_model_fn | `Callable` | Callable that receives an Optuna ``Trial`` and returns a compiled ``tf.keras.Model``. |
| bytes_per_param | `int` | Number of bytes used to store each parameter. |
| batch_size | `int` | Batch size used when estimating the training memory. |
| n_trials | `int` | Total number of random trials to sample. |
| fig_save_path | `str, optional` | Path to save the figure. If ``None`` the figure is shown only. |
| figsize | `Tuple[int, int]` | Figure size for the histograms. |
| csv_path | `str, optional` | Path to save trial results in CSV format. |
| logs_dir | `str, optional` | Directory to store error logs for failed trials. |
| corr_csv_path | `str, optional` | Path to save hyperparameter correlations. |

**Returns**
`None`

**Raises**
- None

**Notes**
- Clearing the Keras backend session between trials mitigates ``ResourceExhaustedError``.
- Trials that raise ``ResourceExhaustedError`` are skipped and the count is printed.
- When ``csv_path`` is provided the sampled statistics are saved to CSV. If
  ``logs_dir`` is set, parameters of failed trials along with their traceback are
  written to individual log files.
- When ``corr_csv_path`` is set, a Spearman correlation analysis between numeric
  hyperparameters and the number of parameters is written to CSV.

### set_user_attr_model_stats

```python
set_user_attr_model_stats(
    trial,
    model,
    bytes_per_param,
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
| trial | `optuna.Trial` | The Optuna trial object |
| model | `tf.keras.Model` | The Keras model to analyze. |
| policy | `tf.keras.DTypePolicy` | The precision policy used for the model. |
| batch_size | `int` | The batch size to simulate for input. |
| n_trials | `int` | Number of trials for power and energy measurement. |
| device | `int` | GPU index to run the model on. Use ``-1`` for CPU. |
| verbose | `bool` | If True, print detailed information. |

**Returns**
` Dict[str, float]: A dictionary containing model statistics`

**Raises**
- None

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
| all_trial_ids | `Set[int]` | Set of all trial IDs in the study. |
| top_trial_ids | `Set[int]` | Set of top-K trial IDs to preserve. |
| cleanup_paths | `List[Tuple[str, str]]` | List of (base_directory, filename_template) tuples. The filename_template should contain '{trial_id}' placeholder. |

**Returns**
`Any`

**Raises**
- OSError: If file removal operations fail.

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
| study | `optuna.Study` | The Optuna study to retrieve trials from. |
| num_trials | `int` | The total number of trials to consider. |

**Returns**
` list[optuna.trial.FrozenTrial]: A list of completed trials.`

**Raises**
- None

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
| study | `optuna.Study` | The completed Optuna study. |
| top_k | `int` | Number of top trials to retrieve. |
| rank_key | `str` | Key to rank trials by ("value" for objective value, or any user attribute key). |
| order | `str` | "descending" for highest values first or "ascending" for lowest values first. |

**Returns**
` List[optuna.Trial]: List of top-K trials sorted by the ranking criteria.`

**Raises**
- None

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
| top_trials | `List[optuna.Trial]` | List of top trials in ranked order. |
| file_configs | `List[Tuple[str, str]]` | List of (base_directory, file_extension) tuples. Files are expected to follow pattern 'trial_{trial_id}{extension}'. |

**Returns**
`Any`

**Raises**
- OSError: If file rename operations fail.

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
| filepath | `str` | Path where the parameter file should be saved. |
| params | `dict[str, float]` | Dictionary of trial hyperparameters. **kwargs (str): Additional information such as trial ID, rank, or loss. |

**Returns**
` None`

**Raises**
- None

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
| top_trials | `List[optuna.Trial]` | List of trials to save. |
| args_dir | `str` | Directory to save trial parameter files. |
| study | `optuna.Study` | The Optuna study (needed for sampler info). |
| extra_attrs | `Optional[List[str]]` | List of additional user attributes to save. If None, defaults to common accuracy metrics. |

**Returns**
`Any`

**Raises**
- None

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
| run_dir | `str` | Base directory for the run |
| study_name | `str` | Name of the study directory (default: "optuna_study") |
| subdirs | `list` | List of subdirectory names to create (default: ["args", "figures", "backup", "history", "models", "logs", "tensorboard"]) |

**Returns**
` tuple: (study_dir, *subdirectory_paths) in the order specified by subdirs`

**Raises**
- None

### log_trial_error

```python
log_trial_error(
    trial,
    exc,
    logs_dir,
    prune_on=None,
    propagate=None,
)
```
Log a JSON file for a failed trial and decide whether to prune.

Exceptions listed in ``prune_on`` cause the trial to be pruned after logging.
Those in ``propagate`` are immediately re-raised. All others are saved to
``logs_dir`` for later inspection.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | The current Optuna trial. |
| exc | `Exception` | The exception that occurred. |
| logs_dir | `str` | Directory where the log file is written. |
| prune_on | `Iterable[Exception], optional` | Exception types that trigger pruning. |
| propagate | `Iterable[Exception], optional` | Exception types that are simply re-raised. |

**Returns**
` None`

**Raises**
- optuna.TrialPruned: If pruning is triggered.
- Exception: Re-raised when listed in ``propagate``.

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
Sends an email notification with the specified subject and body content to multiple recipients. Example: send_email("Hi", "This is a test", "recipients.json", "credentials.json", text_type="html")

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
` None`

**Raises**
- None

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
    restart_email_warning,
)
```
Main function with notebook conversion, file cleanup, and consolidated email notification support.
The function validates that the provided ``file_path`` exists before any
monitoring resources are created and raises ``FileNotFoundError`` if not.
> [!CAUTION]
> The target script is executed repeatedly until a success flag is detected or the maximum number of restarts is reached.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| file_path | `str` | Path to .py or .ipynb file to execute |
| success_flag_file | `str` | Path to success flag file |
| title | `Any` | Custom title for monitoring and email alerts |
| max_restarts | `int` | Maximum restart attempts |
| restart_delay | `float` | Delay between restarts in seconds |
| recipients_file | `Any` | Path to recipients JSON file (defaults to ./json/recipients.json) |
| credentials_file | `Any` | Path to credentials JSON file (defaults to ./json/credentials.json) |
| restart_after_delay | `Any` | restart the run after a delay in seconds |
| retry_attempts | `int` | Number of retry attempts before sending failure email |
| supress_tf_warnings | `bool` | Suppress TensorFlow `ptxas` register spill warnings (default: False) |
| resource_usage_log_file | `Any` | Path to write process resource usage logs. If None, logging is disabled. |
| restart_email_warning | `bool` | Enable or disable email warnings for restart events. |

**Returns**
`Any`

**Raises**
- FileNotFoundError: If file doesn't exist
- ValueError: If file type is unsupported
- ImportError: If notebook dependencies missing for .ipynb files

### monitor CLI

The package exposes a console script named `monitor` through the entry
point `monitor = "araras.runtime.monitoring:main"`. After installing the
package you can execute:

```bash
monitor path/to/job.py [another_job.ipynb ...] [options]
```

> [!TIP]
> You can use `CUDA_VISIBLE_DEVICES=0,1 monitor ...` to limit the GPUs

| Flag | Type | Description |
|------|------|-------------|
| `-t, --title` | `str` | Custom title for monitoring and email alerts. |
| `-m, --max-restarts` | `int` | Maximum number of restart attempts. Defaults to `10`. |
| `-d, --restart-delay` | `float` | Delay between restarts in seconds. Defaults to `3.0`. |
| `-r, --recipients-file` | `str` | Path to a JSON file listing email recipients. |
| `-c, --credentials-file` | `str` | Path to a JSON file with email credentials. |
| `-f, --force-restart` | `float` | Restart the job after this many seconds even if it has not crashed. |
| `-a, --retry-attempts` | `int` | Number of retry attempts before a failure email is sent. |
| `-w, --supress-tf-warnings` | `bool` | Filter TensorFlow `ptxas` register spill warnings. |
| `--no-restart-email` | `bool` | Disable email warnings for process restarts. |
| `-u, --resource-usage-log-file` | `str` | File to log process resource usage statistics. |
| `-s, --success-flag-file` | `str` | Path where the monitored script writes a completion flag. Defaults to `/tmp/success.flag`. |

> [!NOTE]
> Avoid changing the success flag file path unless necessary. That could break the monitoring logic and cause a spam of emails.

> [!CAUTION]
> Run the `monitor` command from the **same directory** as the file being monitored so that relative paths resolve correctly.

## utils

Convenience imports for the `araras.utils` package. This initializer exposes
commonly used utilities while avoiding heavy dependencies until their attributes
are accessed. Functions from `io` and `misc` are lightweight, but `system`
relies on TensorFlow. To prevent TensorFlow from loading when the
`araras.utils` package is imported, the attributes from `system` are loaded
on demand via `__getattr__`.

## utils.io

### create_run_directory

```python
create_run_directory(
    prefix,
    base_dir,
)
```
Creates a new run directory with an incremented numeric suffix and returns its full path. The directory name is generated using the given prefix followed by the next available number. For example, if directories "run1", "run2", and "run3" exist, calling with prefix="run" will create "run4". Logic: -> Ensure base_dir exists -> List existing directories with matching prefix and numeric suffix -> Parse suffix numbers and find the next available integer -> Construct full path using prefix + next number -> Create the new run directory and return its path

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| prefix | `str` | Prefix to be used in the name of each run directory (e.g., "run"). |
| base_dir | `str, optional` | Directory under which all runs are stored. Defaults to "runs". |

**Returns**
` str: Absolute path to the newly created run directory. Example: run_path = create_run_directory(prefix="run") print(run_path)  # outputs: runs/run1, runs/run2, etc.`

**Raises**
- None

### get_caller_stem

```python
get_caller_stem(
    remove='temp_monitor_',
)
```
Return the stem of the calling script or notebook.

The helper inspects different execution contexts (VS Code notebooks, Python
scripts or pure notebooks via ``ipynbname``) to infer the file stem. The optional
``remove`` substring is stripped from the detected stem when present.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| remove | `Optional[str]` | Substring to remove from the detected stem. |

**Returns**
` str: The cleaned stem name of the caller.`

**Raises**
- RuntimeError: If the stem name cannot be determined.

## utils.misc

### clear

```python
clear(

)
```
Clear all prints from terminal or notebook cell. This function works in multiple environments: - Jupyter notebooks/JupyterLab - Terminal/command prompt (Windows, macOS, Linux) - Python scripts run from command line

**Parameters**
| Name | Type | Description |
|------|------|-------------|

**Returns**
`None`

**Raises**
- None

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
| number | `int, float` | The number to format |
| precision | `int` | Number of decimal places to show (default: 2) |

**Returns**
` str: Formatted number with appropriate suffix`

**Raises**
- None

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
| bytes_value | `int, float` | The number of bytes |
| precision | `int` | Number of decimal places to show (default: 2) |

**Returns**
` str: Formatted bytes with appropriate suffix`

**Raises**
- None

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
| number | `int, float` | The number to format |
| max_precision | `int` | Maximum number of decimal places (default: 2) |

**Returns**
` str: Number formatted in scientific notation`

**Raises**
- None

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
| number | `int, float` | The number to format |
| precision | `int` | Number of decimal places to show (default: 2) |

**Returns**
` str: Number formatted with commas`

**Raises**
- None

## utils.system

### get_user_gpu_choice

```python
get_user_gpu_choice(

)
```
Prompts the user to select a GPU index and validates the input.
- This function does not accept any arguments.

**Returns**
` str: Valid GPU index as string`

**Raises**
- None

> [!TIP]
> If only one GPU is detected, the function automatically selects index `0` and skips prompting the user.

### get_gpu_info

```python
get_gpu_info(

)
```
Prints detailed TensorFlow and GPU configuration information in nvidia-smi style format. This function reports: - TensorFlow version and CUDA configuration - GPU devices in tabular format similar to nvidia-smi - Memory usage summary - Temperature and utilization data (when available)

**Parameters**
This function does not accept any arguments.

**Returns**
` None Example: get_gpu_info()`

**Raises**
- None
> [!IMPORTANT]
> Requires an NVIDIA driver and the `nvidia-smi` utility installed to query hardware information.

### gpu_summary

```python
gpu_summary(

)
```
Prints a compact GPU summary similar to nvidia-smi output.

**Parameters**
This function does not accept any arguments.

**Returns**
` None`

**Raises**
- None
> [!NOTE]
> The summary lists only GPUs visible to TensorFlow and may omit devices hidden by environment variables.

### log_resources

```python
log_resources(
    log_dir,
    interval,
    pid=None,
)
```
Logs selected system and ML resources (CPU, RAM, GPU with temperatures, CUDA, TensorFlow) at regular time intervals. If ``pid`` is provided, CPU usage for that process is also logged.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| log_dir | `str` | Directory where log files will be stored. |
| interval | `int` | Time interval between consecutive logs in seconds. Defaults to 5. |
| pid | `int \| None` | Process ID to monitor CPU usage for. Defaults to the current process. |
| kwargs | `bool` flags | Set a flag to ``True`` to log a given resource. Supported flags: "cpu", "ram", "gpu", "cuda", "tensorflow". |

**Returns**
` None Example: log_resources("logs", interval=10, pid=os.getpid(), cpu=True, ram=True, gpu=True)`

**Raises**
- None
> [!CAUTION]
> Log files are appended indefinitely; on long-running experiments they can grow very large. Ensure there is enough disk space.

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
| style | `str` | Figure style. Use `'single-column'` for narrow figures or `'double-column'` for wide figures. Defaults to `'single-column'`. |

**Returns**
` None`

**Raises**
- None

> [!TIP]
> Use `'double-column'` when creating figures that span both columns in a conference paper for consistent font sizes.
