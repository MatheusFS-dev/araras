# API Documentation

This document provides an overview of the API functions available in the ARARAS package.

## Table of Contents

- [API Documentation](#api-documentation)
  - [Table of Contents](#table-of-contents)
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
    - [trial\_skip\_connection](#trial_skip_connection)
  - [ml.model.builders.tcnn](#mlmodelbuilderst-cnn)
    - [build\_tcnn1d](#build_tcnn1d)
    - [build\_tcnn2d](#build_tcnn2d)
    - [build\_tcnn3d](#build_tcnn3d)
  - [ml.model.builders.lm](#mlmodelbuilderslm)
  - [ml.model.callbacks](#mlmodelcallbacks)
    - [get\_callbacks\_model](#get_callbacks_model)
  - [ml.model.stats](#mlmodelstats)
    - [get\_flops](#get_flops)
    - [get\_macs](#get_macs)

    - [write\_model\_stats\_to\_file](#write_model_stats_to_file)
  - [ml.model.tools](#mlmodeltools)
    - [convert\_to\_saved\_model](#convert_to_saved_model)
    - [punish\_model\_flops](#punish_model_flops)
    - [punish\_model\_params](#punish_model_params)
    - [punish\_model](#punish_model)
  - [ml.model.utils](#mlmodelutils)
    - [capture\_model\_summary](#capture_model_summary)
    - [run\_dummy\_inference](#run_dummy_inference)
  - [ml.model (classes and utilities)](#mlmodel-classes-and-utilities)
    - [KParams](#kparams)
    - [print\_tensor\_mem](#print_tensor_mem)
    - [validate\_steps\_per\_execution](#validate_steps_per_execution)
  - [ml.torch](#mltorch)
    - [seed\_everything](#seed_everything)
    - [clear\_torch\_session](#clear_torch_session)
    - [save\_model\_as\_torchscript](#save_model_as_torchscript)
    - [save\_model\_as\_exported\_program](#save_model_as_exported_program)
  - [ml.torch.callbacks](#mltorchcallbacks)
    - [EarlyStopping](#earlystopping)
    - [TorchPruningCallback](#torchpruningcallback)
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
    - [get\_credentials](#get_credentials)
    - [get\_recipient\_emails](#get_recipient_emails)
    - [send\_email](#send_email)
  - [runtime.monitoring](#runtimemonitoring)
    - [run\_auto\_restart](#run_auto_restart)
    - [FlagBasedRestartManager](#flagbasedrestartmanager)
    - [start\_monitor](#start_monitor)
    - [stop\_monitor](#stop_monitor)
    - [check\_crash\_signal](#check_crash_signal)
    - [monitor CLI](#monitor-cli)
  - [utils](#utils)
  - [utils.io](#utilsio)
    - [create\_run\_directory](#create_run_directory)
    - [get\_caller\_stem](#get_caller_stem)
    - [select\_path](#select_path)
  - [utils.misc](#utilsmisc)
    - [clear](#clear)
    - [format\_number](#format_number)
    - [format\_bytes](#format_bytes)
    - [format\_scientific](#format_scientific)
    - [format\_number\_commas](#format_number_commas)
    - [supress\_optuna\_warnings](#supress_optuna_warnings)
    - [NotebookConverter](#notebookconverter)
  - [utils.system](#utilssystem)
    - [setup_gpu_env](#setup_gpu_env)
    - [get\_gpu\_info](#get_gpu_info)
    - [gpu\_summary](#gpu_summary)
    - [log\_resources](#log_resources)
  - [visualization.configs](#visualizationconfigs)
    - [config\_plt](#config_plt)

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
    activation,
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
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. When provided, ``kparams`` may be omitted. |
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
- `ValueError` – If ``batch_size`` is empty or when ``benchmark_training`` is enabled with an invalid ``device`` specification.
- `RuntimeError` – If ``benchmark_training`` targets a GPU device that is not available on the current system.

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
    activation,
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
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. When provided, ``kparams`` may be omitted. |
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
    verbosity=1,
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
| verbosity | `int` | Verbosity level. ``0`` disables output; ``1`` enables a progress bar. |
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
    activation,
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
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. When provided, ``kparams`` may be omitted. |
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
    activation,
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
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. When provided, ``kparams`` may be omitted. |
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
    activation,
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
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. When provided, ``kparams`` may be omitted. |
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
    activation,
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
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. When provided, ``kparams`` may be omitted. |
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
    activation,
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
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. When provided, ``kparams`` may be omitted. |
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
    retry_on_cpu,
    activation,
    name_prefix,
)
```
Build a single Graph Convolutional Network (GCN) layer.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` |  |
| kparams | `KParams` |  |
| x | `layers.Layer` | Input feature tensor. |
| a_graph | `tf.sparse.SparseTensor` | Normalized sparse adjacency matrix. |
| units_range | `Union[int, tuple[int, int]]` | Output units or tuning range. |
| dropout_rate_range | `Union[float, tuple[float, float]]` | Dropout rate or tuning range. |
| units_step | `int` |  |
| dropout_rate_step | `float` |  |
| kernel_initializer | `initializers.Initializer` |  |
| bias_initializer | `initializers.Initializer` |  |
| use_bias | `bool` |  |
| use_batch_norm | `bool` |  |
| trial_kernel_reg | `bool` |  |
| trial_bias_reg | `bool` |  |
| trial_activity_reg | `bool` |  |
| retry_on_cpu | `bool` | Retry on CPU if GPU sparse-dense op fails. |
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. |
| name_prefix | `str` |  |

**Returns**
`layers.Layer`

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
    retry_on_cpu,
    activation,
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
| units_range | `Union[int, tuple[int, int]]` |  |
| dropout_rate_range | `Union[float, tuple[float, float]]` |  |
| heads_range | `Any` |  |
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
| retry_on_cpu | `bool` | Retry on CPU if GPU sparse-dense op fails. |
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. |
| name_prefix | `str` |  |

**Returns**
`layers.Layer`

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
    retry_on_cpu,
    activation,
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
| units_range | `Union[int, tuple[int, int]]` |  |
| dropout_rate_range | `Union[float, tuple[float, float]]` |  |
| K_range | `Union[int, tuple[int, int]]` |  |
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
| retry_on_cpu | `bool` | Retry on CPU if GPU sparse-dense op fails. |
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. |
| name_prefix | `str` |  |

**Returns**
`layers.Layer`

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
    activation,
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
| dropout_rate_range | `Union[float, tuple[float, float]]` | Fixed or tunable dropout rate. |
| units_step | `int` | Step size for tuning LSTM units if a range is given. |
| dropout_rate_step | `float` | Step size for tuning dropout rate. |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for biases. |
| use_bias | `bool` | Whether to include a bias term in the LSTM layer. |
| use_batch_norm | `bool` | Whether to apply batch normalization after LSTM. |
| trial_kernel_reg | `bool` | Whether to apply/tune a kernel regularizer. |
| trial_bias_reg | `bool` | Whether to apply/tune a bias regularizer. |
| trial_activity_reg | `bool` | Whether to apply/tune an activity regularizer. |
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. |
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

### trial_skip_connection

```python
trial_skip_connection(
    trial,
    source,
    target,
    axis_to_concat,
    use_batch_norm,
    merge_mode,
)
```
Conditionally merge a projected skip tensor into a target tensor. This is
primarily intended for GNN layers where the feature dimension may change across
the network while the number of nodes stays constant.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.trial.Trial` | Optuna trial controlling inclusion of the skip. |
| source | `tf.Tensor` | Tensor supplying the skip connection. |
| target | `tf.Tensor` | Destination tensor for the skip connection. |
| axis_to_concat | `int, optional` | Axis used when concatenating tensors. Default is -1. |
| use_batch_norm | `bool, optional` | Apply batch normalization in the projection branch. Defaults to False. |
| merge_mode | `str, optional` | Merge strategy: `'add'` or `'concat'`. Defaults to `'add'`. |

**Returns**
` tf.Tensor: The resulting tensor after applying the skip connection if
selected; otherwise `target`.`

**Raises**
- ValueError: If `merge_mode` is not `'concat'` or `'add'`.

## ml.model.builders.tcnn

Temporal Convolutional Neural Network (TCNN) builders for sequential data.

### build_tcnn1d

```python
build_tcnn1d(
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
    activation,
    strides,
    dilation_rate,
    padding,
    kernel_initializer,
    bias_initializer,
    name_prefix,
)
```
Builds a 1D temporal convolutional layer. Similar to standard Conv1D but optimized for temporal sequences.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Optuna trial object for hyperparameter optimization. |
| kparams | `KParams` | Hyperparameter handler for regularizers and activations. |
| x | `layers.Layer` | Input Keras layer. |
| filters_range | `Union[int, tuple[int, int]]` | Number of filters or range for tuning. |
| kernel_size_range | `Union[int, tuple[int, int]]` | Kernel size or range for tuning. |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_step | `int` | Step size for kernel size tuning. |
| use_batch_norm | `bool` | Whether to include batch normalization. |
| trial_kernel_reg | `bool` | Whether to tune kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune bias regularization. |
| trial_activity_reg | `bool` | Whether to tune activity regularization. |
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. |
| strides | `int` | Stride size for convolution. |
| dilation_rate | `int` | Dilation rate for convolution. |
| padding | `str` | Padding method ('valid' or 'same'). |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
` layers.Layer: Output Keras layer after applying temporal convolution and optional batch norm/activation.`

**Raises**
- None

### build_tcnn2d

```python
build_tcnn2d(
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
    activation,
    strides,
    dilation_rate,
    padding,
    kernel_initializer,
    bias_initializer,
    name_prefix,
)
```
Builds a 2D temporal convolutional layer. Designed for spatio-temporal data with temporal and spatial dimensions.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Optuna trial object for hyperparameter optimization. |
| kparams | `KParams` | Hyperparameter handler for regularizers and activations. |
| x | `layers.Layer` | Input Keras layer. |
| filters_range | `Union[int, tuple[int, int]]` | Number of filters or range for tuning. |
| kernel_size_range | `Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]` | Fixed or ranges for height/width tuning. |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_step | `int` | Step size for kernel dimension tuning. |
| use_batch_norm | `bool` | Whether to include batch normalization. |
| trial_kernel_reg | `bool` | Whether to tune kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune bias regularization. |
| trial_activity_reg | `bool` | Whether to tune activity regularization. |
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. |
| strides | `tuple[int, int]` | Stride size for height and width. |
| dilation_rate | `tuple[int, int]` | Dilation rate for height and width. |
| padding | `str` | Padding method ('valid' or 'same'). |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
` layers.Layer: Output Keras layer after applying 2D temporal convolution and optional batch norm/activation.`

**Raises**
- None

### build_tcnn3d

```python
build_tcnn3d(
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
    activation,
    strides,
    dilation_rate,
    padding,
    kernel_initializer,
    bias_initializer,
    name_prefix,
)
```
Builds a 3D temporal convolutional layer. Designed for volumetric spatio-temporal data.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Optuna trial object for hyperparameter optimization. |
| kparams | `KParams` | Hyperparameter handler for regularizers and activations. |
| x | `layers.Layer` | Input Keras layer. |
| filters_range | `Union[int, tuple[int, int]]` | Number of filters or range for tuning. |
| kernel_size_range | `Union[tuple[int, int, int], tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]` | Fixed or ranges for depth/height/width tuning. |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_step | `int` | Step size for kernel dimension tuning. |
| use_batch_norm | `bool` | Whether to include batch normalization. |
| trial_kernel_reg | `bool` | Whether to tune kernel regularization. |
| trial_bias_reg | `bool` | Whether to tune bias regularization. |
| trial_activity_reg | `bool` | Whether to tune activity regularization. |
| activation | `Optional[Union[str, Callable]]` | Activation function or ``None``/``"none"`` to skip. |
| strides | `tuple[int, int, int]` | Stride size for depth, height, and width. |
| dilation_rate | `tuple[int, int, int]` | Dilation rate for depth, height, and width. |
| padding | `str` | Padding method ('valid' or 'same'). |
| kernel_initializer | `initializers.Initializer` | Initializer for kernel weights. |
| bias_initializer | `initializers.Initializer` | Initializer for bias. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
` layers.Layer: Output Keras layer after applying 3D temporal convolution and optional batch norm/activation.`

**Raises**
- None

## ml.model.builders.lm

Language Model builders module. Provides functionality for building language model architectures.

**Note**: See submodule functions for detailed API documentation.

### get_callbacks_model

```python
get_callbacks_model(
    backup_dir,
    checkpoint_dir,
    tensorboard_logs,
    early_stopping_patience,
    reduce_lr_patience,
    mode,
    reduce_lr_factor,
    reduce_lr_min_lr,
    restore_best_weights,
)
```
Constructs and returns a list of Keras callbacks for model training.

> [!CAUTION]
> The `write_graph` option in the TensorBoard callback is disabled because
> enabling it drastically increases memory usage.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| backup_dir | `str, optional` | Directory where the backup files will be stored. |
| checkpoint_dir | `str, optional` | Directory for saving model checkpoints. Required when `restore_best_weights` is `True` and `early_stopping_patience` is `None`. |
| tensorboard_logs | `str, optional` | Directory where TensorBoard logs will be stored. |
| early_stopping_patience | `int, optional` | Epochs to wait before stopping training. Set to `None` to disable `EarlyStopping`. |
| reduce_lr_patience | `int, optional` | Epochs to wait before reducing the learning rate. Set to `None` to disable `ReduceLROnPlateau`. |
| mode | `str, optional` | One of `"auto"`, `"min"`, or `"max"` to control `EarlyStopping` and `ReduceLROnPlateau` metric direction. |
| reduce_lr_factor | `float, optional` | Factor by which the learning rate is reduced. |
| reduce_lr_min_lr | `float, optional` | Lower bound on the learning rate. |
| restore_best_weights | `bool, optional` | Whether to restore the best model weights after training. Defaults to `True`. |

**Returns**
` List[tf.keras.callbacks.Callback]: A list of callbacks to pass into `model.fit()`.`

**Raises**
- `ValueError`: If `restore_best_weights` is `True` while both `early_stopping_patience` and `checkpoint_dir` are `None`.

## ml.model.stats

### get_inference_latency

```python
get_inference_latency(
    model,
    batch_size,
    device,
    warmup_runs,
    runs,
    verbose,
)
```
Execute dummy inference passes on ``model`` and time them. The helper creates zero-filled tensors matching ``model.inputs`` for the requested ``batch_size`` and runs the model repeatedly on the selected device. Optional warm-up executions may be performed before timing begins to exclude one-off initialization overheads.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` | Model whose inference latency should be measured. |
| batch_size | `int` | Batch size for the dummy inputs. Defaults to ``1``. |
| device | `str` | Device specification. Accepts ``"cpu"`` or ``"gpu/<index>"``. ``"both"`` is not supported. Defaults to ``"cpu"``. |
| warmup_runs | `Optional[int]` | Number of warm-up executions performed before timing. ``None`` disables warm-ups. Defaults to ``None``. |
| runs | `int` | Number of timed executions. Must be positive. Defaults to ``1``. |
| verbose | `int` | Verbosity level. Values greater than zero render a progress bar. Defaults to ``1``. |

**Returns**
` Tuple[float, float]: Average and peak inference latency in seconds.`

**Raises**
- `ValueError`: If ``runs`` is less than ``1``, if ``batch_size`` is less than ``1``, or if ``device`` resolves to ``"both"``.
- `RuntimeError`: If the requested GPU device is unavailable.

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

### get_model_stats

```python
get_model_stats(
    model,
    batch_size,
    device,
    stats_to_measure,
    test_runs,
    verbose,
    bytes_per_param,
)
```
Collect structural and runtime statistics for a Keras model. The helper aggregates structural information (parameter count, FLOPs, MACs, and summary) alongside runtime metrics derived from dummy inference runs. Latency calculations rely on ``get_inference_latency``, while resource utilisation readings use a ``ResourceMonitor`` configured with short sampling windows.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` | Model to profile. |
| batch_size | `int` | Batch size for dummy inputs. Defaults to ``1``. |
| device | `str` | Target device string. Accepts ``"cpu"`` or ``"gpu/<index>"``. Defaults to ``"gpu/0"``. |
| stats_to_measure | `Iterable[str]` | Iterable of metric identifiers to compute. Defaults to measuring all: ``"parameters"``, ``"model_size"``, ``"flops"``, ``"macs"``, ``"summary"``, ``"inference_latency"``, ``"cpu_util_percent"``, ``"cpu_power_rapl_w"``, ``"ram_used_bytes"``, ``"ram_util_percent"``, ``"gpu_util_percent"``, ``"gpu_mem_used_bytes"``, ``"gpu_power_w"``. |
| test_runs | `int` | Number of repetitions for each resource metric. Defaults to ``10``. |
| verbose | `int` | Verbosity level. Values above ``0`` render progress bars. Defaults to ``1``. |
| bytes_per_param | `int` | Number of bytes assigned to each trainable parameter when estimating the model size. Defaults to ``4`` (the footprint of ``float32`` weights). |

**Returns**
` Dict[str, Any]: Mapping from metric names to their computed statistics. Latency metrics return a mapping with ``average_s`` and ``peak_s``. Resource metrics provide aggregation metadata, the collected measurements, and simple summary statistics. The ``model_size`` metric reports the estimated footprint in bytes. Metrics that cannot be computed return ``None`` or an error string.`

**Raises**
- `ValueError`: If ``device`` resolves to ``"both"``, if ``test_runs`` is less than ``1``, or if ``bytes_per_param`` is less than ``1``.
- `RuntimeError`: If the requested GPU index is unavailable on the current system.

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
    stats_to_measure,
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
| device | `str` | Device selection. Use ``"cpu"`` for CPU-only execution, ``"gpu/<index>"`` for a specific GPU, or ``"both/<index>"`` to profile CPU and GPU sequentially. |
| n_trials | `int` | Number of trials for power and energy measurement. |
| extra_attrs | `Optional[Dict[str, Any]]` | Mapping of attribute names to values written after the main statistics. |
| verbose | `bool` | If True, print detailed information. |
| stats_to_measure | `Iterable[str]` | Collection of statistic groups to compute. Accepted values are ``"parameters"``, ``"flops"``, ``"macs"``, ``"summary"``, ``"resource_usage"``, and ``"usage_stats"``. Any omitted group is skipped. |

> [!NOTE]
> Extra attributes can be used to record custom metrics such as accuracy or F1 score alongside the default statistics.

> [!TIP]
> Skipping expensive groups (for example resource usage or usage statistics) can
> significantly reduce profiling time when only lightweight metrics are needed.

**Returns**
`Any`

**Raises**
- TypeError: If ``stats_to_measure`` is ``None`` or not iterable.
- ValueError: If ``stats_to_measure`` contains unsupported statistic names.

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
    mode,
    reduce_lr_factor,
    reduce_lr_min_lr,
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
| mode | `str, optional` | One of `"auto"`, `"min"`, or `"max"` to control `EarlyStopping` and `ReduceLROnPlateau` metric direction. |
| reduce_lr_factor | `float, optional` | Factor by which the learning rate is reduced. |
| reduce_lr_min_lr | `float, optional` | Lower bound on the learning rate. |
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
    batch_size=32,
    verbose=0,
)
```
Estimate total VRAM needed for training a Keras model in bytes.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `keras.Model` | Compiled Keras model to analyse. |
| batch_size | `int` | Training batch size used for the estimate. |
| verbose | `int` | ``1`` to emit detailed logger messages, ``0`` otherwise. |

**Returns**
`int`: Total memory needed in bytes.

**Raises**
- `ValueError`: If ``verbose`` is not ``0`` or ``1``.

**Notes**
- Includes model weights, gradients, optimizer slots, activations, and framework overhead.

**Warnings**
- Falls back to heuristics when layer output shapes are unavailable, potentially under-estimating memory.

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
    plot_model_dir=None,
    show_plot=False,
    benchmark_training=False,
    device="gpu/0",
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
| fig_save_path | `str, optional` | Path to save the figure. If ``None`` the figure is only displayed when ``show_plot`` is ``True``. |
| figsize | `Tuple[int, int]` | Figure size for the histograms. |
| csv_path | `str, optional` | Path to save trial results in CSV format. |
| logs_dir | `str, optional` | Directory to store error logs for failed trials. |
| corr_csv_path | `str, optional` | Path to save hyperparameter correlations. |
| plot_model_dir | `str, optional` | Directory where individual model architecture plots are saved. |
| show_plot | `bool` | Whether to display the histogram figure after sampling. Defaults to ``False``. |
| benchmark_training | `bool` | If ``True`` runs a synthetic single-epoch training benchmark on the smallest and largest sampled models. |
| device | `str` | Device specification (``"cpu"`` or ``"gpu/<index>"``) used for the synthetic training benchmark. Defaults to ``"gpu/0"``. |

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
- Enabling ``benchmark_training`` executes a single synthetic training epoch on
  the smallest and largest sampled models using the first provided batch size.
  When ``csv_path`` is set the benchmark summary is stored alongside the main
  results using the ``_training_benchmark`` suffix; otherwise it is saved to
  ``logs_dir`` (or the current working directory when ``logs_dir`` is omitted).
- High DPI values substantially increase the allocated canvas size. When the
  helper is invoked from the ``monitor`` CLI on a headless server, prefer the
  defaults (``show_plot=False``) to avoid ``X Error of
  failed request: BadAlloc`` failures from the X11 server.

### set_user_attr_model_stats

```python
set_user_attr_model_stats(
    trial,
    model,
    bytes_per_param,
    batch_size,
    n_trials=10000,
    device="both/0",
    stats_to_measure=(
        "parameters",
        "flops",
        "macs",
        "summary",
        "resource_usage",
        "usage_stats",
    ),
    verbose=False,
)
```
Extract and return model statistics from the given Optuna trial.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | The Optuna trial object |
| model | `tf.keras.Model` | The Keras model to analyze. |
| bytes_per_param | `int` | Number of bytes allocated per parameter. |
| batch_size | `int` | The batch size to simulate for input. |
| n_trials | `int` | Number of trials for power and energy measurement. |
| device | `str` | Device selection. Use ``"cpu"`` for CPU-only profiling, ``"gpu/<index>"`` for a dedicated GPU, or ``"both/<index>"`` to profile CPU and GPU sequentially (default ``"both/0"``). |
| stats_to_measure | `Iterable[str]` | Collection of statistic groups to measure. Use ``"parameters"`` for parameter counts and serialized size, ``"flops"`` for floating-point operations, ``"macs"`` for multiply-accumulate estimates, ``"summary"`` for captured ``model.summary`` output, ``"resource_usage"`` for exclusive RAM/VRAM consumption with latency, and ``"usage_stats"`` for power, energy, and per-run timing metrics. Defaults to measuring every group. |
| verbose | `bool` | If True, print detailed information. |

**Returns**
`Dict[str, Any]`: A dictionary containing the collected statistics and formatted displays.

**Raises**
- `TypeError`: If `stats_to_measure` is `None` or not iterable.
- `ValueError`: If `stats_to_measure` includes unsupported statistic names.

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
    history_dir,
    convergence_epoch_column,
    convergence_epoch_direction,
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
| history_dir | `str | None` | Directory containing per-trial history CSV files; used to derive convergence epochs. |
| convergence_epoch_column | `str` | Column to use when computing convergence epochs; defaults to ``"train_loss"``. |
| convergence_epoch_direction | `str` | Direction applied when deriving the convergence epoch (``"minimize"`` or ``"maximize"``). Defaults to ``"minimize"``. |

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
| subdirs | `list` | List of subdirectory names to create (default: ["args", "fig", "backup", "history", "scaler", "model", "logs", "tensorboard"]) |

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
    min_consecutive_oom_failures=None,
)
```
Log a JSON file for a failed trial and decide whether to prune or crash.

Exceptions listed in ``prune_on`` cause the trial to be pruned after logging.
Those in ``propagate`` are immediately re-raised. All others are saved to
``logs_dir`` for later inspection. When ``min_consecutive_oom_failures`` is set,
the process aborts after that many consecutive
``tf.errors.ResourceExhaustedError`` instances.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | The current Optuna trial. |
| exc | `Exception` | The exception that occurred. |
| logs_dir | `str` | Directory where the log file is written. |
| prune_on | `Iterable[Exception], optional` | Exception types that trigger pruning. |
| propagate | `Iterable[Exception], optional` | Exception types that are simply re-raised. |
| min_consecutive_oom_failures | `int, optional` | Minimum number of consecutive OOM errors before forcing a crash. Defaults to `None`. | 
**Returns**
` None`

**Raises**
- optuna.TrialPruned: If pruning is triggered.
- Exception: Re-raised when listed in ``propagate``.

## notifications.email

### get_credentials

```python
get_credentials(
    file_path,
)
```
Load sender credentials from a JSON file containing ``"email"`` and
``"password"`` keys.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| file_path | `str` | Path to the credentials JSON file. |

**Returns**
`Tuple[str, str]`: Sender email address and password.

**Raises**
- ValueError: If the file cannot be read or is missing required keys.

**Examples**
```python
>>> get_credentials("credentials.json")
('your_email@gmail.com', 'your_password')
```

### get_recipient_emails

```python
get_recipient_emails(
    file_path,
)
```
Load recipient addresses from a JSON document with an ``"emails"`` list.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| file_path | `str` | Path to the recipient JSON file. |

**Returns**
`List[str]`: Email addresses that should receive notifications.

**Raises**
- ValueError: If the file cannot be read or lacks an ``"emails"`` list.

**Examples**
```python
>>> get_recipient_emails("recipients.json")
['recipient1@example.com', 'recipient2@example.com']
```

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
Send a notification message to every configured recipient. The helper reads
credentials and recipients from JSON files and logs any failure instead of
raising it.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| subject | `str` | Subject line for the message. |
| body | `str` | Message body, plain text or HTML. |
| recipients_file | `str` | JSON file consumed by :func:`get_recipient_emails`. |
| credentials_file | `str` | JSON file consumed by :func:`get_credentials`. |
| text_type | `str`, optional | MIME subtype for the message body. Defaults to ``"plain"``. |
| smtp_server | `str`, optional | SMTP server hostname. Defaults to ``"smtp.gmail.com"``. |
| smtp_port | `int`, optional | SMTP port. Defaults to ``587``. |

**Examples**
```python
>>> send_email(
...     "Experiment finished",
...     "Model converged successfully.",
...     "recipients.json",
...     "credentials.json",
...     text_type="html",
... )
```

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
    force_restart,
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
| force_restart | `Any` | Force a restart after a delay in seconds regardless of status |
| retry_attempts | `int` | Number of retry attempts before sending failure email |
| supress_tf_warnings | `bool` | Suppress TensorFlow `ptxas` register spill warnings (default: False) |
| resource_usage_log_file | `Any` | Path to write process resource usage logs. If None, logging is disabled. |
| restart_email_warning | `bool` | Enable or disable email warnings for restart events. |

**Returns**
` None`

**Raises**
- FileNotFoundError: If file doesn't exist
- ValueError: If file type is unsupported
- ImportError: If notebook dependencies missing for .ipynb files

### monitor CLI

The package exposes a console script named `monitor` through the entry
point `monitor = "araras.runtime._monitor_script:main"`. After installing the
package you can execute:

```bash
monitor path/to/job.py [another_job.ipynb ...] [options]
```

> [!TIP]
> You can use `CUDA_VISIBLE_DEVICES=0,1 monitor ...` to limit the GPUs

| Flag | Type | Description |
|------|------|-------------|
| `-t, --title` | `str` | Custom title for monitoring and email alerts. |
| `-m, --max-restarts` | `int` | Maximum number of restart attempts. Defaults to `1000`. |
| `-d, --restart-delay` | `float` | Delay between restarts in seconds. Defaults to `3.0`. |
| `-r, --recipients-file` | `str` | Path to a JSON file listing email recipients. |
| `-c, --credentials-file` | `str` | Path to a JSON file with email credentials. |
| `-f, --force-restart` | `float` | Restart the job after this many seconds even if it has not crashed. |
| `-a, --retry-attempts` | `int` | Number of retry attempts before a failure email is sent. |
| `-w, --supress-tf-warnings` | `bool` | Filter TensorFlow `ptxas` register spill warnings. |
| `--no-restart-email` | `bool` | Disable email warnings for process restarts. |
| `-u, --resource-usage-log-file` | `str` | File to log process resource usage statistics. |
| `-s, --success-flag-file` | `str` | Path where the monitored script writes a completion flag. Defaults to a unique file inside the system temporary directory. |

> [!NOTE]
> The monitor still honors custom success flag locations. If you omit ``-s`` the tool now generates a unique flag file under ``/tmp`` (or the platform-specific temporary directory) to prevent interference between concurrent monitors.

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
Create a run directory whose numeric suffix is automatically incremented. The
directory is created inside ``base_dir`` if it does not already exist.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| prefix | `str` | Prefix used in the folder name, for example ``"run"``. |
| base_dir | `str`, optional | Parent directory that holds all runs. Defaults to ``"runs"``. |

**Returns**
`str`: Path to the newly created directory relative to ``base_dir``.

**Examples**
```python
>>> create_run_directory("experiment")
'runs/experiment1'
```

### get_caller_stem

```python
get_caller_stem(
    remove='temp_monitor_',
)
```
Return the stem of the script or notebook that invoked this helper. Supports
VS Code notebooks, Python scripts and classic notebooks; optionally removes a
substring from the detected name.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| remove | `Optional[str]` | Substring to remove from the detected stem. |

**Returns**
`str`: Cleaned stem of the calling script or notebook.

**Raises**
- RuntimeError: If a stem cannot be determined from runtime metadata.

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

### supress_optuna_warnings

```python
supress_optuna_warnings(

)
```
Suppress ``ExperimentalWarning`` messages emitted by Optuna's experimental
features without importing the library at module import time.

> [!NOTE]
> The Optuna import happens lazily inside the function so that optional
> dependencies are only loaded when needed.

## utils.system

### setup_gpu_env

```python
setup_gpu_env(
    *,
    visible_device_indices=None,
    memory_limit_mb=None,
    memory_growth=True,
    op_determinism=None,
    xla_jit=None,
    intra_op_threads=None,
    inter_op_threads=None,
    env_variables=None,
    show_cuda_summary=False,
    verbosity=1,
    clear_screen=True,
) -> dict
```
Configure TensorFlow GPU runtime and related CUDA/XLA environment flags in a single call. Prints concise status lines that include the actual GPU model names where applicable.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| visible_device_indices | `str \| int \| Iterable[int] \| None` | Restrict TensorFlow to a subset of GPUs via `tf.config.set_visible_devices`. Indices are validated against TensorFlow-detected devices. |
| memory_limit_mb | `int \| float \| dict[int, float] \| None` | Virtual device memory caps (MiB). Scalar applies to all visible GPUs; dict applies per index. Out-of-range dict keys are warned and ignored. |
| memory_growth | `bool \| None` | Enable/disable on-demand VRAM allocation (`tf.config.experimental.set_memory_growth`). Skipped on devices where a memory limit is set. |
| op_determinism | `bool \| None` | If `True`, call `tf.config.experimental.enable_op_determinism()` to enforce determinism. |
| xla_jit | `bool \| None` | Call `tf.config.optimizer.set_jit(<value>)` to enable/disable XLA JIT. |
| intra_op_threads | `int \| None` | Cap intra-op CPU threads. |
| inter_op_threads | `int \| None` | Cap inter-op scheduling threads. |
| env_variables | `dict[str, str|int|float] \| None` | Extra environment variables to set verbatim (e.g., `{"CUDA_VISIBLE_DEVICES": "0"}`). |
| show_cuda_summary | `bool` | If `True`, print a short CUDA/GPU summary after configuration. |
| verbosity | `int` | Verbosity for status messages (0+). Warnings and errors still print even when `verbosity=0`. |
| clear_screen | `bool` | Clear the console before printing status lines. |

**Returns**
` dict[str, Any]`: A summary with keys: `environment`, `tensorflow`, `warnings`, `errors`.

**Raises**
- `TypeError`: On invalid argument types.
- `ValueError`: On invalid numeric values or out-of-range device indices.

**Behavior highlights**
- Success messages include GPU model names:
  - Setting mask: `Successfully set CUDA_VISIBLE_DEVICES=0 (NVIDIA GeForce RTX 4070)`
  - Restricting TF devices: `Successfully restricted TensorFlow to GPU indices: [0] (NVIDIA GeForce RTX 4070)`
- Memory configuration messages use friendly model names instead of raw device paths.
- `memory_limit_mb` dict keys outside the currently visible GPUs are warned and ignored.
- `verbosity=0` still prints warnings and errors.

**Examples**
```python
>>> setup_gpu_env(
...     visible_device_indices=[0],
...     memory_limit_mb={0: 1024},
...     memory_growth=False,
...     env_variables={
...         "CUDA_VISIBLE_DEVICES": "0",
...         "TF_GPU_ALLOCATOR": "cuda_malloc_async",
...         "TF_DETERMINISTIC_OPS": "1",
...         "TF_CUDNN_DETERMINISM": "1",
...     },
...     verbosity=1,
... )
```

### get_gpu_info

```python
get_gpu_info(

)
```
Print detailed TensorFlow and GPU configuration information similar to
``nvidia-smi``.

> [!IMPORTANT]
> The most complete output requires an NVIDIA driver and the ``nvidia-smi``
> utility. Missing tooling degrades the report gracefully.

**Examples**
```python
>>> get_gpu_info()
```

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
Continuously log system metrics (CPU, RAM, GPU, CUDA, TensorFlow) to CSV files.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| log_dir | `str` | Directory where log files are saved. |
| interval | `int` | Seconds between samples. Defaults to ``5``. |
| pid | `int \| None` | Process ID whose CPU usage is recorded. Defaults to the current process. |
| kwargs | `bool` flags | Flags such as ``cpu=True`` or ``gpu=True`` to enable specific logs. |

**Examples**
```python
>>> log_resources("logs", interval=10, pid=os.getpid(), cpu=True, ram=True, gpu=True)
```

> [!CAUTION]
> Log files grow without bound while the monitoring threads are running.

## visualization.configs

### config_plt

```python
config_plt(
    style,
)
```
Configure Matplotlib defaults for IEEE-style figures.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| style | `str` | Either ``"single-column"`` or ``"double-column"`` to apply the corresponding figure preset. |

**Raises**
- ValueError: If ``style`` is not supported.

> [!TIP]
> Use ``"double-column"`` when creating figures that span both columns in a conference paper.

## ml.model (classes and utilities)

### KParams

```python
KParams
```
A utility class for managing hyperparameters, providing regularizers and activations for use in model builders.

**Parameters**
- This is a class for storing and providing hyperparameter configurations.

**Returns**
`None`

**Raises**
- None

### print_tensor_mem

```python
print_tensor_mem(
    tensor,
)
```
Print the memory footprint of a tensor in human-readable format.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| tensor | `tf.Tensor` | The tensor to analyze. |

**Returns**
`None`

**Raises**
- None

### validate_steps_per_execution

```python
validate_steps_per_execution(
    steps_per_execution,
)
```
Validate and return the steps_per_execution parameter for model compilation.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| steps_per_execution | `int` | Number of steps to run per execution call. |

**Returns**
`int`: Validated steps_per_execution value.

**Raises**
- None

## ml.model.utils

### run_dummy_inference

```python
run_dummy_inference(
    model,
    batch_size,
    device,
    warmup_runs,
    runs,
    verbose,
)
```
Execute dummy inference passes on ``model`` and time them. Creates zero-filled tensors matching ``model.inputs`` for the requested ``batch_size`` and runs the model repeatedly on the selected device.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `tf.keras.Model` | Model whose inference latency should be measured. |
| batch_size | `int` | Batch size for the dummy inputs. Defaults to ``1``. |
| device | `str` | Device specification. Accepts ``"cpu"`` or ``"gpu/<index>"``. Defaults to ``"cpu"``. |
| warmup_runs | `Optional[int]` | Number of warm-up executions performed before timing. ``None`` disables warm-ups. Defaults to ``None``. |
| runs | `int` | Number of timed executions. Must be positive. Defaults to ``1``. |
| verbose | `int` | Verbosity level. Values greater than zero render a progress bar. Defaults to ``1``. |

**Returns**
` Tuple[float, float]: Average and peak inference latency in seconds.`

**Raises**
- `ValueError`: If ``runs`` is less than ``1`` or if ``batch_size`` is less than ``1``.
- `RuntimeError`: If the requested GPU device is unavailable.

## ml.torch

PyTorch-specific helpers for training and Optuna integration.

### seed_everything

```python
seed_everything(
    seed,
)
```
Set random seeds for reproducibility across PyTorch, NumPy, and Python.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| seed | `int` | Random seed value. |

**Returns**
`None`

**Raises**
- None

### clear_torch_session

```python
clear_torch_session(

)
```
Clear PyTorch session and free GPU memory.

**Parameters**
- This function takes no parameters.

**Returns**
`None`

**Raises**
- None

### save_model_as_torchscript

```python
save_model_as_torchscript(
    model,
    save_path,
)
```
Save a PyTorch model in TorchScript format.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `torch.nn.Module` | The PyTorch model to save. |
| save_path | `str` | Path where the TorchScript model will be saved. |

**Returns**
`None`

**Raises**
- None

### save_model_as_exported_program

```python
save_model_as_exported_program(
    model,
    example_inputs,
    save_path,
)
```
Save a PyTorch model as an ExportedProgram (torch.export format).

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| model | `torch.nn.Module` | The PyTorch model to export. |
| example_inputs | `Any` | Example inputs for tracing the model. |
| save_path | `str` | Path where the exported program will be saved. |

**Returns**
`None`

**Raises**
- None

## ml.torch.callbacks

### EarlyStopping

```python
EarlyStopping
```
Early stopping callback for PyTorch training loops.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| patience | `int` | Number of epochs to wait for improvement before stopping. |
| min_delta | `float` | Minimum change in the monitored value to qualify as an improvement. |

**Returns**
`None`

**Raises**
- None

### TorchPruningCallback

```python
TorchPruningCallback
```
Optuna pruning callback for PyTorch training within Optuna trials.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | The Optuna trial object. |

**Returns**
`None`

**Raises**
- None

## runtime.monitoring (additional functions)

### FlagBasedRestartManager

```python
FlagBasedRestartManager
```
Manages process restarts based on flag files. Monitors a success flag file and automatically restarts a process if it hasn't been created within a timeout.

**Parameters**
- This is a class for managing process restarts.

**Returns**
`None`

**Raises**
- None

### start_monitor

```python
start_monitor(
    file_path,
    success_flag_file,
)
```
Start monitoring a file for crashes and restarts.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| file_path | `str` | Path to the script to monitor. |
| success_flag_file | `str` | Path to the success flag file. |

**Returns**
`None`

**Raises**
- None

### stop_monitor

```python
stop_monitor(

)
```
Stop the current monitoring process.

**Parameters**
- This function takes no parameters.

**Returns**
`None`

**Raises**
- None

### check_crash_signal

```python
check_crash_signal(

)
```
Check if a crash signal has been received.

**Parameters**
- This function takes no parameters.

**Returns**
`bool`: True if a crash signal was detected, False otherwise.

**Raises**
- None

## utils.io

### select_path

```python
select_path(
    select_dir,
    extensions,
    description,
    initial_dir,
)
```
Open a native file/folder picker dialog and return the selected path.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| select_dir | `bool` | Controls which dialog mode is used. If ``True``, opens a directory chooser. If ``False``, opens a file chooser. Defaults to ``True``. |
| extensions | `Optional[Iterable[str]]` | Iterable of extensions to allow when ``select_dir`` is ``False``. Extensions may include or omit the leading dot (for example ``"csv"`` and ``".csv"`` are both accepted). If ``None`` or empty, all file types are shown. Defaults to ``None``. |
| description | `str` | Dialog title shown to the user. Defaults to ``"Select a path"``. |
| initial_dir | `Optional[str]` | Initial directory to open the dialog in. If ``None``, uses the current working directory. Defaults to ``None``. |

**Returns**
` str: The path selected by the user.`

**Raises**
- RuntimeError: If the dialog is cancelled or not supported on the current platform.

## utils.misc

### NotebookConverter

```python
NotebookConverter
```
Utility class for converting Jupyter notebooks to Python scripts and vice versa.

**Parameters**
- This is a class for notebook conversion operations.

**Returns**
`None`

**Raises**
- None

Notes:
- Supports conversion between .ipynb and .py formats.
- Handles notebook metadata and cell transformations.